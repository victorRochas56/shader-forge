#pragma once
#include "render_pass.hpp"
#include "bindless_system.hpp"
#include "scene.hpp"
#include "structs.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>

// Renders each mesh once into its own small bindless color target for GUI previews.
// A mesh is (re)rendered only while its thumbnailDirty flag is set; a few are processed
// per frame to spread the cost. The targets are sampled directly by ImGui in showMeshList.
class ThumbnailPass : public RenderPass {

    static constexpr uint32_t SIZE = 128;             // rendered size; ImGui downscales to 64
    static constexpr vk::Format COLOR_FORMAT = vk::Format::eR8G8B8A8Unorm;
    static constexpr uint32_t MAX_PER_FRAME = 8;      // dirty meshes processed each frame

    uint32_t pipelineIndex = 0xFFFFFFFF;

    // Shared depth buffer, reused (cleared) for every thumbnail since they render serially.
    vk::raii::Image        depthImage  = nullptr;
    vk::raii::DeviceMemory depthMemory = nullptr;
    vk::raii::ImageView    depthView   = nullptr;

public:
    ThumbnailPass(GpuContext& gpu, BindlessSystem& bindless, Scene& scene, RenderFeatures& features, RenderPassResources& shared)
        : RenderPass(gpu, bindless, scene, features, shared) {}

    void init(uint32_t /*width*/, uint32_t /*height*/) override {
        if (pipelineIndex == 0xFFFFFFFF) {
            pipelineIndex = bindless.pipelineManager->createPipeline<ThumbnailPushConstants>(
                PipelineCategory::THUMBNAIL, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone,
                vk::True, vk::True, "shaders/thumbnail.spv", bindless.descriptorSet->getDescriptorSetLayout(),
                bindless.descriptorSet->getDescriptorSet(), COLOR_FORMAT);
        }
        if (*depthImage == VK_NULL_HANDLE) {
            resource::createImage(*bindless.resourceCtx, SIZE, SIZE, 1, vk::SampleCountFlagBits::e1, vk::Format::eD32Sfloat,
                vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment,
                vk::MemoryPropertyFlagBits::eDeviceLocal, depthImage, depthMemory, 1);
            depthView = resource::createImageView(*bindless.resourceCtx, depthImage, vk::Format::eD32Sfloat, vk::ImageAspectFlagBits::eDepth);
            resource::transitionImageLayout(*bindless.resourceCtx, nullptr, *depthImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal);
        }
    }

    void record(vk::raii::CommandBuffer& cmd, uint32_t /*imageIndex*/) override {
        tracing::startTrace("mesh thumb pass");
        uint32_t budget = MAX_PER_FRAME;
        for (Mesh& mesh : scene.assetManager.meshes) {
            if (mesh.freed || !mesh.thumbnailDirty) continue;
            if (mesh.indexCount == 0) { mesh.thumbnailDirty = false; continue; }

            if (mesh.thumbnailTextureIndex == 0xFFFFFFFF)
                resize(mesh.thumbnailTextureIndex, SIZE, SIZE, COLOR_FORMAT, std::string("internal/thumbnail" + mesh.name).c_str());

            renderMesh(cmd, mesh);
            mesh.thumbnailDirty = false;
            if (--budget == 0) break;
        }
        tracing::endTrace("mesh thumb pass");
    }

private:
    void renderMesh(vk::raii::CommandBuffer& cmd, const Mesh& mesh) {
        TextureResource& target = bindless.descriptorSet->getTextureResource(mesh.thumbnailTextureIndex);
        resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *target.image, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eColorAttachmentOptimal);

        vk::ClearValue clearColor{.color = vk::ClearColorValue(std::array<float, 4>{0.10f, 0.10f, 0.12f, 1.0f})};
        vk::ClearValue clearDepth{.depthStencil = vk::ClearDepthStencilValue{1.0f, 0}};

        vk::RenderingAttachmentInfo colorAttachment{.imageView = *target.imageView,
                                                    .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                    .loadOp = vk::AttachmentLoadOp::eClear,
                                                    .storeOp = vk::AttachmentStoreOp::eStore,
                                                    .clearValue = clearColor};
        vk::RenderingAttachmentInfo depthAttachment{.imageView = *depthView,
                                                    .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                                    .loadOp = vk::AttachmentLoadOp::eClear,
                                                    .storeOp = vk::AttachmentStoreOp::eDontCare,
                                                    .clearValue = clearDepth};
        vk::RenderingInfo renderInfo{.renderArea = {{0, 0}, {SIZE, SIZE}}, .layerCount = 1,
                                     .colorAttachmentCount = 1, .pColorAttachments = &colorAttachment, .pDepthAttachment = &depthAttachment};

        cmd.beginRendering(renderInfo);
        setFullscreenViewport(cmd, {SIZE, SIZE});
        bindPipeline(cmd, *bindless.pipelineManager->getBeforeGeoPipelines()[pipelineIndex]);
        cmd.bindIndexBuffer(bindless.descriptorSet->getVariableBuffer(shared.indexBufferIndex), 0, vk::IndexType::eUint32);

        ThumbnailPushConstants pc{.mvp = framingMatrix(mesh),
                                  .vertexBufferAddress = bindless.descriptorSet->getVariableBuffers()[shared.vertexBufferIndex]->address,
                                  .vertexStride = mesh.vertexStride,
                                  .vertexOffset = static_cast<uint32_t>(mesh.vertexOffset)};
        cmd.pushConstants<ThumbnailPushConstants>(bindless.pipelineManager->getBeforeGeoPipelines()[pipelineIndex]->layout,
                                                  vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pc);
        cmd.drawIndexed(mesh.indexCount, 1, static_cast<uint32_t>(mesh.indexOffset / sizeof(uint32_t)), 0, 0);
        cmd.endRendering();

        resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *target.image, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
    }

    // Frame the mesh's AABB with a fixed 3/4 view. Matches the engine's Vulkan clip convention.
    glm::mat4 framingMatrix(const Mesh& mesh) const {
        glm::vec3 center = (mesh.boundingBoxMin + mesh.boundingBoxMax) * 0.5f;
        float radius = glm::length(mesh.boundingBoxMax - center);
        if (radius < 1e-4f) radius = 1.0f;

        const float fov = glm::radians(40.0f);
        float dist = radius / std::sin(fov * 0.5f) * 1.15f;
        glm::vec3 eye = center + glm::normalize(glm::vec3(0.6f, 0.45f, 1.0f)) * dist;

        glm::mat4 view = glm::lookAt(eye, center, glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 proj = glm::perspective(fov, 1.0f, std::max(0.01f, dist - radius * 2.0f), dist + radius * 2.0f);
        proj[1][1] *= -1.0f;
        return proj * view;
    }
};
