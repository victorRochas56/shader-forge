#pragma once
#include <array>
#include <cstdint>
#include <vulkan/vulkan_raii.hpp>
#include "pipelines.hpp"

class GpuContext;
class BindlessSystem;
class Scene;
struct RenderFeatures;
struct RenderBuffers;

// Bindless indices (and other handles) that are created once by the app and
// shared across passes. Passes read from this; they do not own the resources.
struct RenderPassResources {
    uint32_t blurPipelineIndex = 0xFFFFFFFF;

    uint32_t depthSamplerIndex = 0xFFFFFFFF;
    uint32_t defaultSamplerIndex = 0xFFFFFFFF;

    // Global geometry buffers, for passes that draw meshes (thumbnails).
    uint32_t vertexBufferIndex = 0xFFFFFFFF;
    uint32_t indexBufferIndex = 0xFFFFFFFF;

    // gbuffer
    uint32_t colorResolveTextureIndex = 0xFFFFFFFF;
    uint32_t roughnessMetalTextureIndex = 0xFFFFFFFF;
    uint32_t normalTextureIndex = 0xFFFFFFFF;
    uint32_t motionVectorTextureIndex = 0xFFFFFFFF;
    RenderBuffers& buffers;
    // hiZ
    uint32_t hiZTextureIndex = 0xFFFFFFFF;
    uint32_t hiZMipLevels = 0;

    // blurred color mip chain (for cone tracing)
    uint32_t tempBlurTextureIndex = 0xFFFFFFFF;
    uint32_t fullscreenMipLevels = 1;
    std::vector<vk::raii::ImageView>* colorResolveMipViews = nullptr;
    std::vector<vk::raii::ImageView>* tempBlurMipViews = nullptr;
};

enum PassId {
    SSAO,
    SSR,
    VOLUMETRICS,
    THUMBNAIL,
};

class RenderPass {
public:
    virtual ~RenderPass() = default;
    void resize(uint32_t& index, uint32_t width, uint32_t height, vk::Format format, const char* debugName, vk::ImageUsageFlags extraUsage = {});
    virtual void init(uint32_t width, uint32_t height) = 0;
    virtual void record(vk::raii::CommandBuffer& cmd, uint32_t imageIndex) = 0;

protected:
    RenderPass(GpuContext& gpu, BindlessSystem& bindless, Scene& scene, RenderFeatures& features, RenderPassResources& shared)
        : gpu(gpu), bindless(bindless), scene(scene), features(features), shared(shared) {}

    GpuContext&          gpu;
    BindlessSystem&      bindless;
    Scene&               scene;
    RenderFeatures&      features;
    RenderPassResources& shared;
};

void setFullscreenViewport(vk::raii::CommandBuffer& cmd, vk::Extent2D extent);
void bindPipeline(vk::raii::CommandBuffer& cmd, PipelineBase& pipeline);

template <typename T>
void drawFullscreenPass(vk::raii::CommandBuffer& cmd, PipelineBase& pipeline, vk::ImageView targetView, vk::Extent2D extent,
                        const T& pushConstants, vk::AttachmentLoadOp loadOp = vk::AttachmentLoadOp::eClear,
                        std::array<float, 4> clearColor = {0.0f, 0.0f, 0.0f, 0.0f}) {
    vk::ClearValue clear{.color = vk::ClearColorValue(clearColor)};
    vk::RenderingAttachmentInfo colorAttachment{.imageView = targetView,
                                                .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                .loadOp = loadOp,
                                                .storeOp = vk::AttachmentStoreOp::eStore,
                                                .clearValue = clear};
    vk::RenderingInfo renderInfo{.renderArea = {{0, 0}, extent}, .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &colorAttachment};

    cmd.beginRendering(renderInfo);
    setFullscreenViewport(cmd, extent);
    bindPipeline(cmd, pipeline);
    cmd.pushConstants<T>(pipeline.layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pushConstants);
    cmd.draw(3, 1, 0, 0);
    cmd.endRendering();
}

void blurAttachment(BindlessSystem& bindless, vk::raii::CommandBuffer& cmd, uint32_t blurPipelineIndex,
                    uint32_t sourceTextureIndex, uint32_t tempTextureIndex, uint32_t width, uint32_t height,
                    float blurRadius = 1.0f, uint32_t samplerIndex = 0);
