#pragma once
#include "render_pass.hpp"
#include "bindless_system.hpp"
#include "profiling.hpp"
#include "scene_elements.hpp"
#include "scene.hpp"

#ifndef VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_NO_CONSTRUCTORS 1 // for structs constructors
#endif


// Voxelizes the scene into a filterable RGBA16F volume for cone tracing, in three stages:
//
//   [raster scatter] -> [resolve to mip 0] -> [downsample mip chain]
//
// The scene is rasterized once with no attachments; the fragment shader scatters albedo and radiance
// into two uint[VOXEL_RESOLUTION^3] BDA buffers with InterlockedMax (needs fragmentStoresAndAtomics).
// Max on a luma-keyed packed RGB word is order-independent, so overlapping triangles converge to the
// brightest contributor without depending on draw order. A compute pass then unpacks those buffers
// into mip 0 of the volume, and a second pass folds each level into the next with an opacity-weighted
// 2x2x2 filter — a plain box filter would average empty voxels in and drag coverage toward black.
//
// The raster stage does per-triangle dominant-axis selection in a geometry shader: each triangle is
// projected along the grid axis its face normal is most aligned with, so nothing is edge-on to its
// own sweep. The FS derives the voxel coord from the interpolated grid position, not SV_Position.
//
// Known limitations, in rough order of how much they'll show up:
//  - No conservative rasterization. Triangles covering a texel but missing its center write nothing,
//    which pinholes thin walls. VK_EXT_conservative_rasterization or shader-side edge expansion.
//  - The grid follows the camera, snapped to whole-voxel steps so sub-voxel motion doesn't shift the
//    world→voxel mapping. Geometry beyond the VOXEL_WORLD_EXTENT cube around the camera still never
//    voxelizes, and the volume is fully rebuilt every frame even when nothing moved.
//  - Isotropic mips, so cone tracing off this will leak light through thin geometry. The alternative
//    is Crassin's six directional chains, which also changes the cone tracer.
class VoxelizationPass : public RenderPass {
    static constexpr uint32_t VOXEL_RESOLUTION  = 128;    // cubic grid side
    static constexpr float    VOXEL_WORLD_EXTENT = 25.0f; // side of the world-space cube the grid covers
    static constexpr vk::Format VOXEL_VOLUME_FORMAT = vk::Format::eR16G16B16A16Sfloat;

    // Scatter targets. Two uint per voxel, cleared every frame — 8 MB each at 128^3, and the same
    // again in clear bandwidth per frame. 256^3 is 8x both.
    uint32_t albedoBufferIndex   = 0xFFFFFFFF;
    uint32_t radianceBufferIndex = 0xFFFFFFFF;

    // The filterable volume. voxelVolumeTextureIndex is the sampled full-chain slot cone tracing
    // reads; volumeMipStorageIndices[i] is a single-mip storage slot the resolve/downsample write
    // through (an RWTexture3D binding must resolve to exactly one mip).
    uint32_t voxelVolumeTextureIndex = 0xFFFFFFFF;
    std::vector<uint32_t> volumeMipStorageIndices;
    std::vector<vk::raii::ImageView> volumeMipViews; // owns the views the storage slots point at
    uint32_t volumeMipLevels = 0;

    uint32_t rasterPipelineIndex     = 0xFFFFFFFF;
    uint32_t resolvePipelineIndex    = 0xFFFFFFFF;
    uint32_t downsamplePipelineIndex = 0xFFFFFFFF;
    uint32_t debugPipelineIndex      = 0xFFFFFFFF; // features.voxelDebug ray-march overlay
    // Cube debug view: extract compacts occupied voxels into cubeInstanceBuffer and bumps the
    // instanceCount in cubeIndirectBuffer; cubeDraw renders them with one indirect draw.
    uint32_t cubeExtractPipelineIndex = 0xFFFFFFFF;
    uint32_t cubeDrawPipelineIndex    = 0xFFFFFFFF;
    uint32_t cubeInstanceBufferIndex  = 0xFFFFFFFF;
    uint32_t cubeIndirectBufferIndex  = 0xFFFFFFFF;

    static constexpr uint32_t mipLevelsFor(uint32_t resolution) {
        uint32_t levels = 1;
        while (resolution > 1) { resolution >>= 1; ++levels; }
        return levels;
    }

public:
    glm::mat4 voxelCamVPM;
    glm::mat4 voxelCamInvVPM;

    VoxelizationPass(GpuContext& gpu, BindlessSystem& bindless, Scene& scene, RenderFeatures& features, RenderPassResources& shared)
        : RenderPass(gpu, bindless, scene, features, shared) {}

    uint32_t getVolumeTextureIndex() const { return voxelVolumeTextureIndex; }
    uint32_t getVolumeMipLevels() const { return volumeMipLevels; }

    void init(uint32_t width, uint32_t height) override {
        (void)width;
        (void)height;

        // Screen-independent, so allocated once.
        if (albedoBufferIndex == 0xFFFFFFFF) {
            constexpr vk::DeviceSize voxelCount = static_cast<vk::DeviceSize>(VOXEL_RESOLUTION) * VOXEL_RESOLUTION * VOXEL_RESOLUTION;
            // eTransferDst for the per-frame fillBuffer clear. Device-local is not optional: the
            // fragment shader does two atomics per fragment on these, and on the default host-visible
            // allocation every one of them crosses PCIe.
            auto usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eTransferDst;
            constexpr uint32_t bufferBytes = static_cast<uint32_t>(voxelCount * sizeof(uint32_t));
            static_assert(voxelCount * sizeof(uint32_t) <= 0xFFFFFFFFull, "voxel buffer exceeds createVariableBuffer's uint32 size");
            albedoBufferIndex   = bindless.descriptorSet->createVariableBuffer(bufferBytes, usage, false, "VoxelAlbedo", true);
            radianceBufferIndex = bindless.descriptorSet->createVariableBuffer(bufferBytes, usage, false, "VoxelRadiance", true);

            // Cube debug view. Instance capacity is the worst case (every mip-0 voxel occupied), so
            // the extract's InterlockedAdd can never run past the end.
            cubeInstanceBufferIndex = bindless.descriptorSet->createVariableBuffer(static_cast<uint32_t>(voxelCount * 2 * sizeof(uint32_t)),
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress, false, "VoxelCubeInstances", true);
            cubeIndirectBufferIndex = bindless.descriptorSet->createVariableBuffer(4 * sizeof(uint32_t),
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress |
                vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eTransferDst, false, "VoxelCubeIndirect", true);
        }

        if (voxelVolumeTextureIndex == 0xFFFFFFFF) createVolume();

        auto& setLayout = bindless.descriptorSet->getDescriptorSetLayout();
        auto& set = bindless.descriptorSet->getDescriptorSet();
        if (rasterPipelineIndex == 0xFFFFFFFF)
            rasterPipelineIndex = bindless.pipelineManager->createPipeline<VoxelizationPushConstants>(
                PipelineCategory::VOXELIZATION, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone,
                vk::False, vk::False, "shaders/voxelization.spv", setLayout, set, vk::Format::eUndefined, "geomMain");
        if (resolvePipelineIndex == 0xFFFFFFFF)
            resolvePipelineIndex = bindless.pipelineManager->createComputePipeline<VoxelResolvePushConstants>("shaders/voxel_resolve.spv", setLayout, set, "resolveMain");
        if (downsamplePipelineIndex == 0xFFFFFFFF)
            downsamplePipelineIndex = bindless.pipelineManager->createComputePipeline<VoxelResolvePushConstants>("shaders/voxel_resolve.spv", setLayout, set, "downsampleMain");
        // Overlay onto the HDR composite, so it sits alongside the other post passes.
        if (debugPipelineIndex == 0xFFFFFFFF)
            debugPipelineIndex = bindless.pipelineManager->createPipeline<VoxelDebugPushConstants>(
                PipelineCategory::POSTPROCESS_ALPHA_BLEND, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone,
                vk::False, vk::False, "shaders/voxel_debug.spv", setLayout, set, gpu.getSwapchain().getHDRColorFormat());
        if (cubeExtractPipelineIndex == 0xFFFFFFFF)
            cubeExtractPipelineIndex = bindless.pipelineManager->createComputePipeline<VoxelCubePushConstants>("shaders/voxel_cubes.spv", setLayout, set, "extractMain");
        // Depth test AND write (reverse-Z GE): cube mode replaces the scene, so the cubes only need to
        // occlude each other — recordCubes clears both attachments before the draw.
        if (cubeDrawPipelineIndex == 0xFFFFFFFF)
            cubeDrawPipelineIndex = bindless.pipelineManager->createPipeline<VoxelCubePushConstants>(
                PipelineCategory::POSTPROCESS, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone,
                vk::True, vk::True, "shaders/voxel_cubes.spv", setLayout, set, gpu.getSwapchain().getHDRColorFormat());
    }

    void record(vk::raii::CommandBuffer& cmd, uint32_t imageIndex) override {
        (void)imageIndex;
        // Skipping is safe: the volume keeps its last contents and rests in ShaderReadOnly (all
        // transitions are paired inside this function), so the debug views still work.
        if (!features.voxelDebug.voxelizeScene) return;
        tracing::startTrace("voxelization");

        clearScatterBuffers(cmd);
        recordRaster(cmd);
        recordResolve(cmd);

        tracing::endTrace("voxelization");
    }

    // Called separately from record() — this draws into the HDR composite, so it has to run after the
    // geometry and lighting passes have filled it, not inside the voxel build at the top of the frame.
    void recordDebugOverlay(vk::raii::CommandBuffer& cmd) {
        if (!features.voxelDebug.enabled || voxelVolumeTextureIndex == 0xFFFFFFFF) return;
        if (features.voxelDebug.drawCubes) { recordCubes(cmd); return; }
        tracing::startTrace("voxel debug");

        // Camera NDC -> grid clip in one matrix, so the fragment shader can march straight through the
        // grid's cube without ever going back to world space (the grid is an OBB in world space).
        glm::mat4 camNdcToGrid = voxelCamVPM * glm::inverse(scene.activeCamera.viewProjection);

        glm::vec4 camGridClip = voxelCamVPM * glm::vec4(scene.activeCamera.position, 1.0f);
        glm::vec3 camGridNdc = glm::vec3(camGridClip) / camGridClip.w;
        // Same clip->UVW mapping as gridClipToUVW() in the shader: xy is NDC, z is already [0,1].
        glm::vec3 cameraPosGrid = glm::vec3(camGridNdc.x * 0.5f + 0.5f, camGridNdc.y * 0.5f + 0.5f, camGridNdc.z);

        // The depth resolve rests in eDepthStencilAttachmentOptimal; sampling it requires the shader
        // read layout, same as the SSAO and particle passes do around their own depth reads.
        auto& depthResolveTex = bindless.descriptorSet->getTextureResource(gpu.getSwapchain().getDepthResolveIndex());
        resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *depthResolveTex.image,
                                        vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

        drawFullscreenPass(cmd, *bindless.pipelineManager->getPostProcessPipelines()[debugPipelineIndex],
            *bindless.descriptorSet->getTextureResource(shared.compositeColorTextureIndex).imageView,
            gpu.getSwapchain().getSwapChainExtent(),
            VoxelDebugPushConstants{
                .camNdcToGrid      = camNdcToGrid,
                .cameraPosGrid     = cameraPosGrid,
                .mipLevel          = std::min(features.voxelDebug.mipLevel, volumeMipLevels - 1),
                .volumeTexIndex    = voxelVolumeTextureIndex,
                .samplerIndex      = shared.volumeSamplerIndex, // clamped; defaultSampler's eRepeat wraps the grid
                .depthTexIndex     = gpu.getSwapchain().getDepthResolveIndex(),
                .depthSamplerIndex = shared.depthSamplerIndex,
                .resolution        = VOXEL_RESOLUTION,
                .mode              = static_cast<uint32_t>(features.voxelDebug.mode),
                .maxSteps          = features.voxelDebug.maxSteps,
                .alphaScale        = features.voxelDebug.alphaScale,
            }, vk::AttachmentLoadOp::eLoad);

        resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *depthResolveTex.image,
                                        vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eDepthStencilAttachmentOptimal);

        tracing::endTrace("voxel debug");
    }

private:
    // Cube view: extract occupied voxels (compute) -> one indirect draw of shaded cubes into the HDR
    // composite, depth-tested against the scene. The volume rests in ShaderReadOnly, which is what the
    // extract's Load needs, and the depth resolve rests in DepthStencilAttachmentOptimal, which is what
    // the draw binds — so no layout transitions at all.
    void recordCubes(vk::raii::CommandBuffer& cmd) {
        tracing::startTrace("voxel debug cubes");

        uint32_t mip = std::min(features.voxelDebug.mipLevel, volumeMipLevels - 1);
        VoxelCubePushConstants pc{
            .gridToClip            = scene.activeCamera.viewProjection * voxelCamInvVPM,
            .instanceBufferAddress = bindless.descriptorSet->getVariableBuffers()[cubeInstanceBufferIndex]->address,
            .indirectBufferAddress = bindless.descriptorSet->getVariableBuffers()[cubeIndirectBufferIndex]->address,
            .volumeTexIndex        = voxelVolumeTextureIndex,
            .mipLevel              = mip,
            .mipRes                = std::max(1u, VOXEL_RESOLUTION >> mip),
            .threshold             = features.voxelDebug.cubeThreshold,
        };

        // Last frame's indirect draw must finish reading these buffers before this frame's reset
        // overwrites them (WAR — execution ordering only, no data to flush).
        vk::MemoryBarrier warBarrier{};
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eDrawIndirect | vk::PipelineStageFlagBits::eVertexShader,
                            vk::PipelineStageFlagBits::eTransfer, {}, warBarrier, {}, {});

        // Reset the indirect args (36 verts/cube, instanceCount 0), then let the extract's atomics see it.
        const std::array<uint32_t, 4> drawArgs{36, 0, 0, 0};
        cmd.updateBuffer<uint32_t>(bindless.descriptorSet->getVariableBuffer(cubeIndirectBufferIndex), 0, drawArgs);
        vk::MemoryBarrier resetBarrier{.srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                                       .dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite};
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, resetBarrier, {}, {});

        auto& extract = static_cast<ComputePipeline<VoxelCubePushConstants>&>(*bindless.pipelineManager->getComputePipelines()[cubeExtractPipelineIndex]);
        extract.pushConstantData = pc;
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, extract.pipeline);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, extract.layout, 0, {**extract.descriptorSet}, {});
        extract.pushConstants(cmd);
        constexpr uint32_t WG = 4; // [numthreads(4,4,4)]
        uint32_t groups = (pc.mipRes + WG - 1) / WG;
        cmd.dispatch(groups, groups, groups);

        // Instance data to the vertex shader, instanceCount to the indirect fetch.
        vk::MemoryBarrier extractBarrier{.srcAccessMask = vk::AccessFlagBits::eShaderWrite,
                                         .dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead | vk::AccessFlagBits::eShaderRead};
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eDrawIndirect | vk::PipelineStageFlagBits::eVertexShader, {}, extractBarrier, {}, {});

        // Cube mode replaces the scene: clear the composite to a dark backdrop and depth to far
        // (reverse-Z: 0), then depth-write so cubes occlude each other. The rendered geometry is
        // hidden, not skipped — depth/normal targets stay valid for the other passes. Anything after
        // this (billboards, SDF, gizmos) now depth-tests against the voxel world, which is the point.
        vk::RenderingAttachmentInfo colorAttachment{.imageView = *bindless.descriptorSet->getTextureResource(shared.compositeColorTextureIndex).imageView,
                                                    .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                    .loadOp = vk::AttachmentLoadOp::eClear,
                                                    .storeOp = vk::AttachmentStoreOp::eStore,
                                                    .clearValue = {.color = vk::ClearColorValue(std::array<float, 4>{0.02f, 0.025f, 0.035f, 1.0f})}};
        vk::RenderingAttachmentInfo depthAttachment{.imageView = *gpu.getSwapchain().getDepthResolveImageView(),
                                                    .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                                    .loadOp = vk::AttachmentLoadOp::eClear,
                                                    .storeOp = vk::AttachmentStoreOp::eStore,
                                                    .clearValue = {.depthStencil = {0.0f, 0}}};
        vk::Extent2D extent = gpu.getSwapchain().getSwapChainExtent();
        vk::RenderingInfo renderInfo{.renderArea = {{0, 0}, extent}, .layerCount = 1,
                                     .colorAttachmentCount = 1, .pColorAttachments = &colorAttachment,
                                     .pDepthAttachment = &depthAttachment};

        cmd.beginRendering(renderInfo);
        auto& pipeline = *bindless.pipelineManager->getPostProcessPipelines()[cubeDrawPipelineIndex];
        bindPipeline(cmd, pipeline);
        setFullscreenViewport(cmd, extent);
        cmd.pushConstants<VoxelCubePushConstants>(pipeline.layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pc);
        cmd.drawIndirect(bindless.descriptorSet->getVariableBuffer(cubeIndirectBufferIndex), 0, 1, sizeof(vk::DrawIndirectCommand));
        cmd.endRendering();

        tracing::endTrace("voxel debug cubes");
    }

    // Mipped storage+sampled volume. Built by hand rather than through resize3DStorageImage because
    // that helper is single-mip and each level here needs its own storage slot.
    void createVolume() {
        volumeMipLevels = mipLevelsFor(VOXEL_RESOLUTION);

        vk::raii::Image image = nullptr;
        vk::raii::DeviceMemory memory = nullptr;
        resource::create3DImage(*bindless.resourceCtx, VOXEL_RESOLUTION, VOXEL_RESOLUTION, VOXEL_RESOLUTION, volumeMipLevels,
                                vk::SampleCountFlagBits::e1, VOXEL_VOLUME_FORMAT, vk::ImageTiling::eOptimal,
                                vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
                                vk::MemoryPropertyFlagBits::eDeviceLocal, image, memory, 1);

        // Per-mip storage views first — they capture the VkImage handle, which survives the move into
        // the texture slot below.
        volumeMipViews.reserve(volumeMipLevels);
        for (uint32_t mip = 0; mip < volumeMipLevels; mip++)
            volumeMipViews.push_back(resource::create3DImageView(*bindless.resourceCtx, image, VOXEL_VOLUME_FORMAT, vk::ImageAspectFlagBits::eColor, mip, 1));

        auto sampledView = resource::create3DImageView(*bindless.resourceCtx, image, VOXEL_VOLUME_FORMAT, vk::ImageAspectFlagBits::eColor, 0, volumeMipLevels);
        // Resting layout is sampled; record() transitions the whole chain to eGeneral to write it.
        resource::transitionImageLayout(*bindless.resourceCtx, nullptr, image, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal, 0, volumeMipLevels);

        voxelVolumeTextureIndex = bindless.descriptorSet->allocateTexture(std::move(image), std::move(memory), std::move(sampledView),
                                                                         "internal/voxel_volume", false, VOXEL_RESOLUTION, VOXEL_RESOLUTION);
        volumeMipStorageIndices.reserve(volumeMipLevels);
        for (auto& view : volumeMipViews)
            volumeMipStorageIndices.push_back(bindless.descriptorSet->allocateStorageImage(*view));
    }

    // Zero both scatter buffers, then make the fill visible to the fragment shader's atomics.
    void clearScatterBuffers(vk::raii::CommandBuffer& cmd) {
        // The previous frame's raster atomics and resolve reads must finish before this frame's clear
        // overwrites the buffers (WAR/WAW — execution ordering only).
        vk::MemoryBarrier warBarrier{};
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eFragmentShader | vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eTransfer, {}, warBarrier, {}, {});

        cmd.fillBuffer(bindless.descriptorSet->getVariableBuffer(albedoBufferIndex), 0, vk::WholeSize, 0u);
        cmd.fillBuffer(bindless.descriptorSet->getVariableBuffer(radianceBufferIndex), 0, vk::WholeSize, 0u);
        vk::MemoryBarrier fillBarrier{.srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                                      .dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite};
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, fillBarrier, {}, {});
    }

    // Attachment-less rasterization. renderArea is what drives coverage — there is nothing bound to
    // write to, so the fragment shader's atomics are the only output.
    void recordRaster(vk::raii::CommandBuffer& cmd) {
        constexpr float half = VOXEL_WORLD_EXTENT * 0.5f;
        // Snap the grid centre to whole-voxel steps: sub-voxel camera motion no longer shifts the
        // world→voxel mapping, which kills crawl and keeps voxel contents comparable frame-to-frame
        // (prerequisite for skipping or caching the rebuild later).
        constexpr float voxelSize = VOXEL_WORLD_EXTENT / VOXEL_RESOLUTION;
        glm::vec3 gridCenter = glm::floor(scene.activeCamera.position / voxelSize) * voxelSize;
        // Top-down ortho defining the grid OBB; the GS re-projects each triangle along its dominant
        // axis, so this is a coordinate frame, not the sweep direction. Up must not be parallel to
        // the -Y view direction — (0,1,0) makes lookAt's cross product zero and NaNs the matrix.
        glm::mat4 view = glm::lookAt(gridCenter + glm::vec3(0.0f, half, 0.0f),
                                     gridCenter,
                                     glm::vec3(0.0f, 0.0f, -1.0f));
        glm::mat4 proj = glm::orthoRH_ZO(-half, half, -half, half, 0.0f, VOXEL_WORLD_EXTENT);
        proj[1][1] *= -1.0f; // Flip Y axis for vulkan

        VoxelizationPushConstants pc{};
        pc.vpm = proj * view;
        voxelCamVPM = pc.vpm;
        voxelCamInvVPM = glm::inverse(pc.vpm);
        pc.vertexBufferAddress   = bindless.descriptorSet->getVariableBuffers()[shared.vertexBufferIndex]->address;
        pc.voxelAlbedoAddress    = bindless.descriptorSet->getVariableBuffers()[albedoBufferIndex]->address;
        pc.voxelRadianceAddress  = bindless.descriptorSet->getVariableBuffers()[radianceBufferIndex]->address;
        pc.samplerIndex          = shared.defaultSamplerIndex;
        pc.voxelResolution       = VOXEL_RESOLUTION;

        vk::Extent2D gridExtent{VOXEL_RESOLUTION, VOXEL_RESOLUTION};
        vk::RenderingInfo renderingInfo = {.renderArea = {.offset = {0, 0}, .extent = gridExtent},
                                           .layerCount = 1,
                                           .colorAttachmentCount = 0};
        cmd.beginRendering(renderingInfo);

        auto& pipeline = *bindless.pipelineManager->getBeforeGeoPipelines()[rasterPipelineIndex];
        bindPipeline(cmd, pipeline);
        setFullscreenViewport(cmd, gridExtent); // viewport/scissor are dynamic state
        cmd.bindIndexBuffer(bindless.descriptorSet->getVariableBuffer(shared.indexBufferIndex), 0, vk::IndexType::eUint32);

        // One draw per visible node. Culling is against the voxel volume itself, so anything outside
        // the grid is skipped rather than clipped.
        std::array<Plane, 6> frustumPlanes = extractFrustumPlanes(pc.vpm);
        for (Node& node : scene.sceneGraph.getNodes()) {
            if (node.meshIndex == MAX_MESHES || !node.alive) continue;
            // Only cull on a bbox the scene graph has actually filled in — an unset one is all zeroes.
            if (node.isBoundingBoxValid() && !isAABBInFrustum(node.boundingBoxMin, node.boundingBoxMax, frustumPlanes)) continue;

            const Mesh& mesh = scene.assetManager.meshes[node.meshIndex];
            if (mesh.freed || mesh.indexCount == 0) continue;

            uint32_t matIdx = node.getMaterialIndex();
            const Material& material = (matIdx < scene.materials.size()) ? scene.materials[matIdx]
                                                                         : scene.materials[scene.getFallBackMaterial()];
            pc.model              = node.getTransform();
            pc.vertexStride       = mesh.vertexStride;
            pc.vertexOffset       = static_cast<uint32_t>(mesh.vertexOffset);
            pc.albedoTextureIndex = material.albedoTextureIndex;
            cmd.pushConstants<VoxelizationPushConstants>(pipeline.layout,
                vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eGeometry | vk::ShaderStageFlagBits::eFragment, 0, pc);
            cmd.drawIndexed(mesh.indexCount, 1, static_cast<uint32_t>(mesh.indexOffset / sizeof(uint32_t)), 0, 0);
        }

        cmd.endRendering();
    }

    // Unpack the scatter buffers into mip 0, then fold the chain. Each level depends on the previous
    // one, so every dispatch is separated by a compute->compute barrier.
    void recordResolve(vk::raii::CommandBuffer& cmd) {
        tracing::startTrace("voxel resolve");
        auto& volumeTex = bindless.descriptorSet->getTextureResource(voxelVolumeTextureIndex);

        // The raster pass wrote through storage buffers from the fragment stage — that has to land
        // before the resolve reads it.
        vk::MemoryBarrier scatterBarrier{.srcAccessMask = vk::AccessFlagBits::eShaderWrite, .dstAccessMask = vk::AccessFlagBits::eShaderRead};
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eFragmentShader, vk::PipelineStageFlagBits::eComputeShader, {}, scatterBarrier, {}, {});

        resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *volumeTex.image,
                                        vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eGeneral, 0, volumeMipLevels);

        auto dispatchCompute = [&](uint32_t pipeIdx, const VoxelResolvePushConstants& pc, uint32_t groups) {
            auto& pipe = static_cast<ComputePipeline<VoxelResolvePushConstants>&>(*bindless.pipelineManager->getComputePipelines()[pipeIdx]);
            pipe.pushConstantData = pc;
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipe.pipeline);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipe.layout, 0, {**pipe.descriptorSet}, {});
            pipe.pushConstants(cmd);
            cmd.dispatch(groups, groups, groups);
        };
        auto computeBarrier = [&]() {
            vk::MemoryBarrier b{.srcAccessMask = vk::AccessFlagBits::eShaderWrite,
                                .dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite};
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, b, {}, {});
        };

        constexpr uint32_t WG = 4; // [numthreads(4,4,4)]
        VoxelResolvePushConstants pc{
            .voxelAlbedoAddress   = bindless.descriptorSet->getVariableBuffers()[albedoBufferIndex]->address,
            .voxelRadianceAddress = bindless.descriptorSet->getVariableBuffers()[radianceBufferIndex]->address,
            .dstStorageIndex      = volumeMipStorageIndices[0],
            .srcStorageIndex      = volumeMipStorageIndices[0],
            .dstResolution        = VOXEL_RESOLUTION,
            .voxelResolution      = VOXEL_RESOLUTION,
        };
        dispatchCompute(resolvePipelineIndex, pc, (VOXEL_RESOLUTION + WG - 1) / WG);

        for (uint32_t mip = 1; mip < volumeMipLevels; mip++) {
            computeBarrier();
            pc.srcStorageIndex = volumeMipStorageIndices[mip - 1];
            pc.dstStorageIndex = volumeMipStorageIndices[mip];
            pc.dstResolution   = std::max(1u, VOXEL_RESOLUTION >> mip);
            dispatchCompute(downsamplePipelineIndex, pc, (pc.dstResolution + WG - 1) / WG);
        }

        // Back to the sampled resting layout; this also makes the writes visible to cone tracing.
        resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *volumeTex.image,
                                        vk::ImageLayout::eGeneral, vk::ImageLayout::eShaderReadOnlyOptimal, 0, volumeMipLevels);
        tracing::endTrace("voxel resolve");
    }
};
