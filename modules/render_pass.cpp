#include "render_pass.hpp"
#include "bindless_system.hpp"
#include "structs.hpp"


void RenderPass::resize(uint32_t& index, uint32_t width, uint32_t height, vk::Format format, const char* debugName, vk::ImageUsageFlags extraUsage) {
    if (index != 0xFFFFFFFF) {
        bindless.descriptorSet->freeTexture(index);
    }
    vk::raii::Image image = nullptr;
    vk::raii::DeviceMemory memory = nullptr;
    bindless.resourceManager->createImage(width, height, 1, vk::SampleCountFlagBits::e1, format, vk::ImageTiling::eOptimal,
                                 vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled | extraUsage,
                                 vk::MemoryPropertyFlagBits::eDeviceLocal, image, memory, 1);
    auto view = bindless.resourceManager->createImageView(image, format, vk::ImageAspectFlagBits::eColor);
    bindless.resourceManager->transitionImageLayout(nullptr, image, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal);
    index = bindless.descriptorSet->allocateTexture(std::move(image), std::move(memory), std::move(view), debugName, false, width, height);
}

void RenderPass::resize3DStorageImage(uint32_t& textureIndex, uint32_t& storageIndex, uint32_t width, uint32_t height, uint32_t depth,
                                      vk::Format format, const char* debugName) {
    // Recycle old slots (device idle — init/resize only).
    if (storageIndex != 0xFFFFFFFF) bindless.descriptorSet->freeStorageImage(storageIndex);
    if (textureIndex != 0xFFFFFFFF) bindless.descriptorSet->freeTexture(textureIndex);

    vk::raii::Image image = nullptr;
    vk::raii::DeviceMemory memory = nullptr;
    bindless.resourceManager->create3DImage(width, height, depth, 1, vk::SampleCountFlagBits::e1, format, vk::ImageTiling::eOptimal,
                                 vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
                                 vk::MemoryPropertyFlagBits::eDeviceLocal, image, memory, 1);
    auto view = bindless.resourceManager->create3DImageView(image, format, vk::ImageAspectFlagBits::eColor);
    // Resting layout is sampled; compute passes transition to eGeneral before writing, then back.
    bindless.resourceManager->transitionImageLayout(nullptr, image, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal);

    vk::ImageView rawView = *view;
    textureIndex = bindless.descriptorSet->allocateTexture(std::move(image), std::move(memory), std::move(view), debugName, false, width, height);
    storageIndex = bindless.descriptorSet->allocateStorageImage(rawView);
}

void setFullscreenViewport(vk::raii::CommandBuffer& cmd, vk::Extent2D extent) {
    cmd.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.0f, 1.0f));
    cmd.setScissor(0, vk::Rect2D({0, 0}, extent));
}

void bindPipeline(vk::raii::CommandBuffer& cmd, PipelineBase& pipeline) {
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.pipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline.layout, 0, {**pipeline.descriptorSet}, {});
}

void blurAttachment(BindlessSystem& bindless, vk::raii::CommandBuffer& cmd, uint32_t blurPipelineIndex,
                    uint32_t sourceTextureIndex, uint32_t tempTextureIndex, uint32_t width, uint32_t height,
                    float blurRadius, uint32_t samplerIndex) {
    auto& blurPipeline = *bindless.pipelineManager->getPostProcessPipelines()[blurPipelineIndex];
    auto& sourceTexture = bindless.descriptorSet->getTextureResource(sourceTextureIndex);
    auto& tempTexture = bindless.descriptorSet->getTextureResource(tempTextureIndex);
    vk::Extent2D extent{width, height};

    // Horizontal blur (source -> temp)
    bindless.resourceManager->transitionImageLayout(&cmd, *tempTexture.image, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eColorAttachmentOptimal);
    drawFullscreenPass(cmd, blurPipeline, *tempTexture.imageView, extent,
        BlurPushConstants{.inputTextureIndex = sourceTextureIndex, .samplerIndex = samplerIndex, .isHorizontal = 1, .blurRadius = blurRadius, .resolution = glm::uvec2(width, height)});
    bindless.resourceManager->transitionImageLayout(&cmd, *tempTexture.image, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

    // Vertical blur (temp -> source)
    bindless.resourceManager->transitionImageLayout(&cmd, *sourceTexture.image, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eColorAttachmentOptimal);
    drawFullscreenPass(cmd, blurPipeline, *sourceTexture.imageView, extent,
        BlurPushConstants{.inputTextureIndex = tempTextureIndex, .samplerIndex = samplerIndex, .isHorizontal = 0, .blurRadius = blurRadius, .resolution = glm::uvec2(width, height)});
    bindless.resourceManager->transitionImageLayout(&cmd, *sourceTexture.image, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
}
