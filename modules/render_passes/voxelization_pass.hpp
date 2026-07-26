#pragma once
#include "render_pass.hpp"
#include "bindless_system.hpp"
#include "profiling.hpp"
#include "scene_elements.hpp"
#include "scene.hpp"

#ifndef VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_NO_CONSTRUCTORS 1 // for structs constructors
#endif


// Single-pass slicemap voxelization. The scene is rasterized once into a 2D R32_UINT target where bit k
// of texel (x,y) marks occupancy of slice k along the sweep axis — so the image holds a
// VOXEL_RESOLUTION^2 x 32 binary grid. Fragments merge with a fixed-function OR
// (PipelineCategory::VOXELIZATION), so overlapping triangles accumulate without atomics.
class VoxelizationPass : public RenderPass {
    static constexpr uint32_t VOXEL_RESOLUTION = 512;                 // lateral (x,y); the third axis is the uint's 32 bits
    static constexpr vk::Format VOXEL_FORMAT = vk::Format::eR64Uint; // integer target: the frag shader emits raw bits
    static constexpr float VOXEL_WORLD_EXTENT = 20.0f;               // side of the world-space cube the grid covers, centred on the origin

    uint32_t voxelSceneTexIndex     = 0xFFFFFFFF; // sampled slot — cone tracing reads this
    uint32_t voxelSceneStorageIndex = 0xFFFFFFFF; // storage slot — compute post-processing of the slicemap
    uint32_t pipelineIndex          = 0xFFFFFFFF;

  public:
    VoxelizationPass(GpuContext& gpu, BindlessSystem& bindless, Scene& scene, RenderFeatures& features, RenderPassResources& shared)
        : RenderPass(gpu, bindless, scene, features, shared) {}

    void init(uint32_t width, uint32_t height) override {
        (void)width;
        (void)height;

        // Screen-independent, so allocated once.
        if(voxelSceneTexIndex == 0xFFFFFFFF) {
            resize2DStorageImage(voxelSceneTexIndex, voxelSceneStorageIndex, VOXEL_RESOLUTION, VOXEL_RESOLUTION, VOXEL_FORMAT,
                                 "voxels", vk::ImageUsageFlagBits::eColorAttachment);
        }

        if(pipelineIndex == 0xFFFFFFFF) {
            pipelineIndex = bindless.pipelineManager->createPipeline<VoxelizationPushConstants>(PipelineCategory::VOXELIZATION,vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone, vk::False, vk::False, "shaders/voxelization.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet(), VOXEL_FORMAT);
        }
    }

    void record(vk::raii::CommandBuffer& cmd, uint32_t imageIndex) override {
        (void)imageIndex;
        tracing::startTrace("voxelization");

        // Orthographic sweep down -Y. It has to be ortho, not perspective: voxels must be uniform in
        // world space or the slice index the fragment shader derives from z_ndc isn't linear in world Y.
        // Up is +Z since the view direction is parallel to the world up axis. Near sits on the eye and
        // far on the far cube face, so z_ndc runs [0,1] across the volume: slice = min(uint(z_ndc * 32), 31).
        constexpr float half = VOXEL_WORLD_EXTENT * 0.5f;
        glm::mat4 view = glm::lookAt(glm::vec3(0.0f/*scene.activeCamera.position.x*/,scene.activeCamera.position.y + half, /*scene.activeCamera.position.z*/0.0f), scene.activeCamera.position, glm::vec3(0.0f, 0.0f, 1.0f));
        glm::mat4 proj = glm::orthoRH_ZO(-half, half, -half, half, 0.0f, VOXEL_WORLD_EXTENT);
        proj[1][1] *= -1.0f; // Flip Y axis for vulkan

        VoxelizationPushConstants pc{};
        pc.vpm = proj * view;

        auto& voxelTexture = bindless.descriptorSet->getTextureResource(voxelSceneTexIndex);
        vk::Extent2D voxelExtent{VOXEL_RESOLUTION, VOXEL_RESOLUTION};
        resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *voxelTexture.image, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eColorAttachmentOptimal);

        vk::RenderingAttachmentInfo voxelColorAttachment = {
            .imageView = *voxelTexture.imageView,
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .resolveMode = vk::ResolveModeFlagBits::eNone,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue =  vk::ClearValue { .color = vk::ClearColorValue { .uint32 = std::array<uint32_t,4>{0,0,0,0} } }
        };

        vk::RenderingInfo renderingInfo = {.renderArea = {.offset = {0, 0}, .extent = voxelExtent},
                                    .layerCount = 1,
                                    .colorAttachmentCount = 1,
                                    .pColorAttachments = &voxelColorAttachment};
        cmd.beginRendering(renderingInfo);

        auto& pipeline = *bindless.pipelineManager->getBeforeGeoPipelines()[pipelineIndex];
        bindPipeline(cmd, pipeline);
        setFullscreenViewport(cmd, voxelExtent); // viewport/scissor are dynamic state
        cmd.bindIndexBuffer(bindless.descriptorSet->getVariableBuffer(shared.indexBufferIndex), 0, vk::IndexType::eUint32);
        pc.vertexBufferAddress = bindless.descriptorSet->getVariableBuffers()[shared.vertexBufferIndex]->address;

        // One draw per visible node. Culling is against the voxel volume itself, so anything outside
        // the grid is skipped rather than clipped.
        std::array<Plane,6> frustumPlanes = extractFrustumPlanes(pc.vpm);
        for(Node& node : scene.sceneGraph.getNodes()) {
            if(node.meshIndex == MAX_MESHES || !node.alive) continue;
            // Only cull on a bbox the scene graph has actually filled in — an unset one is all zeroes.
            if(node.isBoundingBoxValid() && !isAABBInFrustum(node.boundingBoxMin, node.boundingBoxMax, frustumPlanes)) continue;

            const Mesh& mesh = scene.assetManager.meshes[node.meshIndex];
            if(mesh.freed || mesh.indexCount == 0) continue;

            pc.model        = node.getTransform();
            pc.vertexStride = mesh.vertexStride;
            pc.vertexOffset = static_cast<uint32_t>(mesh.vertexOffset);
            pc.meshIndex    = node.meshIndex;
            cmd.pushConstants<VoxelizationPushConstants>(pipeline.layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pc);
            cmd.drawIndexed(mesh.indexCount, 1, static_cast<uint32_t>(mesh.indexOffset / sizeof(uint32_t)), 0, 0);
        }

        cmd.endRendering();
        resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *voxelTexture.image, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

        tracing::endTrace("voxelization");
    }
};