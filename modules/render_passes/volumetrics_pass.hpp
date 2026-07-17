#pragma once
#include "render_pass.hpp"
#include "bindless_system.hpp"
#include "scene.hpp"
#include "structs.hpp"
#include "render_buffers.hpp"
#include <algorithm>
#include <type_traits>

// Froxel volumetrics (FROXEL_VOLUMETRICS_PLAN.md). A view-aligned 3D "froxel" grid is the single
// medium representation: analytic Volumes (A) and volumetric particles (B) inject density into
// mediaVol, it's lit once per froxel (C), integrated front-to-back (D), and the composite (E) does a
// single depth-based sample. Replaces the old per-pixel analytic ray-march.
//
//   [A0 clear] -> [A inject volumes] -> [B bin+gather particles] -> [C light] -> [D integrate] -> [E composite]
// B is a clustered inject: a bin pass buckets each volumetric particle into the screen tiles it covers,
// then a gather pass (1 thread/froxel) sums its tile's particles into mediaVol — so per-thread cost is
// bounded and a screen-filling particle no longer stalls the dispatch.
class VolumetricsPass : public RenderPass {
    uint32_t applyPipelineIndex = 0xFFFFFFFF; // Pass E composite (POSTPROCESS_VOLUMETRIC)

    // Three RGBA16F 3D volumes; each registered as a sampled slot (reads/composite) + storage slot
    // (compute writes). See resize3DStorageImage / FROXEL_VOLUMETRICS_PLAN.md prereq 1.
    uint32_t mediaTexIndex = 0xFFFFFFFF,      mediaStorageIndex = 0xFFFFFFFF;      // A+B: (scattering.rgb, extinction)
    uint32_t mediaPhaseTexIndex = 0xFFFFFFFF, mediaPhaseStorageIndex = 0xFFFFFFFF; // A:   R32F density-weighted phase
    uint32_t scatterTexIndex = 0xFFFFFFFF,    scatterStorageIndex = 0xFFFFFFFF;    // C:   (inScatter.rgb, extinction)
    uint32_t integratedTexIndex = 0xFFFFFFFF, integratedStorageIndex = 0xFFFFFFFF; // D:   (accumInScatter.rgb, transmittance)

    uint32_t clearPipelineIndex = 0xFFFFFFFF;      // A0
    uint32_t injectVolPipelineIndex = 0xFFFFFFFF;  // A
    uint32_t binPipelineIndex = 0xFFFFFFFF;        // B bin   (particle -> per-tile lists)
    uint32_t gatherPipelineIndex = 0xFFFFFFFF;     // B gather (tiles -> mediaVol, 1 thread/froxel)
    uint32_t lightPipelineIndex = 0xFFFFFFFF;      // C
    uint32_t integratePipelineIndex = 0xFFFFFFFF;  // D

    // Particle-injection tile bins (clustered gather). Screen-independent, sized from froxelDims.
    uint32_t tileCountsBufferIndex = 0xFFFFFFFF;   // uint per tile (atomic counter)
    uint32_t tileListBufferIndex = 0xFFFFFFFF;     // uint[tiles * MAX_PARTICLES_PER_TILE]
    uint32_t tilesX = 0, tilesY = 0;

  public:
    VolumetricsPass(GpuContext& gpu, BindlessSystem& bindless, Scene& scene, RenderFeatures& features, RenderPassResources& shared) : RenderPass(gpu, bindless, scene, features, shared) {
        shared.buffers.volumeBufferIndex = bindless.descriptorSet->createFixedBuffer<GPUVolume>(MAX_FRAMES_IN_FLIGHT * MAX_FIXED_BUFFER, true, "Volume");
        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            bindless.descriptorSet->setBufferFrameOffset(shared.buffers.volumeBufferIndex, i, MAX_FIXED_BUFFER * i);
        }
    }

    void init(uint32_t width, uint32_t height) override {
        (void)width;
        (void)height;

        // Froxel grid is screen-independent (sized by features.volumetrics.froxelDims), so allocate
        // once. TODO: reallocate at device idle if froxelDims changes at runtime.
        if (mediaStorageIndex == 0xFFFFFFFF) {
            glm::uvec3 fd = features.volumetrics.froxelDims;
            constexpr vk::Format fmt = vk::Format::eR16G16B16A16Sfloat;
            resize3DStorageImage(mediaTexIndex, mediaStorageIndex, fd.x, fd.y, fd.z, fmt, "internal/froxel_media");
            resize3DStorageImage(scatterTexIndex, scatterStorageIndex, fd.x, fd.y, fd.z, fmt, "internal/froxel_scatter");
            resize3DStorageImage(integratedTexIndex, integratedStorageIndex, fd.x, fd.y, fd.z, fmt, "internal/froxel_integrated");
            resize3DStorageImage(mediaPhaseTexIndex, mediaPhaseStorageIndex, fd.x, fd.y, fd.z, vk::Format::eR32Sfloat, "internal/froxel_phase");

            // Particle tile bins (clustered gather). Tiles cover FROXEL_TILE_SIZE^2 froxels in XY.
            tilesX = (fd.x + FROXEL_TILE_SIZE - 1) / FROXEL_TILE_SIZE;
            tilesY = (fd.y + FROXEL_TILE_SIZE - 1) / FROXEL_TILE_SIZE;
            uint32_t numTiles = tilesX * tilesY;
            auto storageUsage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress;
            // MAX_FRAMES_IN_FLIGHT copies: the bins are rebuilt each frame, so a frame in flight must
            // not clear/overwrite the slice a previous frame's gather is still reading.
            tileCountsBufferIndex = bindless.descriptorSet->createVariableBuffer(MAX_FRAMES_IN_FLIGHT * numTiles * sizeof(uint32_t), storageUsage | vk::BufferUsageFlagBits::eTransferDst, false, "FroxelTileCounts");
            tileListBufferIndex   = bindless.descriptorSet->createVariableBuffer(MAX_FRAMES_IN_FLIGHT * numTiles * MAX_PARTICLES_PER_TILE * sizeof(uint32_t), storageUsage, false, "FroxelTileList");
        }

        auto& setLayout = bindless.descriptorSet->getDescriptorSetLayout();
        auto& set = bindless.descriptorSet->getDescriptorSet();
        if (clearPipelineIndex == 0xFFFFFFFF)
            clearPipelineIndex = bindless.pipelineManager->createComputePipeline<FroxelInjectPushConstants>("shaders/froxel_inject.spv", setLayout, set, "clearMain");
        if (injectVolPipelineIndex == 0xFFFFFFFF)
            injectVolPipelineIndex = bindless.pipelineManager->createComputePipeline<FroxelInjectPushConstants>("shaders/froxel_inject.spv", setLayout, set, "injectVolumesMain");
        if (binPipelineIndex == 0xFFFFFFFF)
            binPipelineIndex = bindless.pipelineManager->createComputePipeline<FroxelBinPushConstants>("shaders/froxel_particle_bin.spv", setLayout, set, "binMain");
        if (gatherPipelineIndex == 0xFFFFFFFF)
            gatherPipelineIndex = bindless.pipelineManager->createComputePipeline<FroxelGatherPushConstants>("shaders/froxel_particle_gather.spv", setLayout, set, "gatherMain");
        if (lightPipelineIndex == 0xFFFFFFFF)
            lightPipelineIndex = bindless.pipelineManager->createComputePipeline<FroxelLightPushConstants>("shaders/froxel_light.spv", setLayout, set, "lightScatterMain");
        if (integratePipelineIndex == 0xFFFFFFFF)
            integratePipelineIndex = bindless.pipelineManager->createComputePipeline<FroxelIntegratePushConstants>("shaders/froxel_integrate.spv", setLayout, set, "integrateMain");

        // Pass E — fullscreen composite onto the HDR composite target with the froxel blend.
        if (applyPipelineIndex == 0xFFFFFFFF)
            applyPipelineIndex = bindless.pipelineManager->createPipeline<VolumetricApplyPushConstants>(
                PipelineCategory::POSTPROCESS_VOLUMETRIC, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone,
                vk::False, vk::False, "shaders/volumetrics_apply.spv", setLayout, set, gpu.getSwapchain().getHDRColorFormat());
    }

    void record(vk::raii::CommandBuffer& cmd, uint32_t imageIndex) override {
        (void)imageIndex;
        tracing::startTrace("volumetric pass");
        if (!features.volumetrics.enabled) {
            tracing::endTrace("volumetric pass");
            return;
        }

        // Stream live analytic volumes into this frame's slice (feeds Pass A): rebuild the contiguous
        // list every frame, pulling each volume's world center fresh from its node. Runs during
        // command recording (post-fence), so the slice written is never in flight.
        std::vector<GPUVolume> volumeWriteBuf;
        volumeWriteBuf.reserve(scene.getVolumes().size());
        for (const auto& [nodeIdx, vol] : scene.getVolumes()) {
            if (!scene.sceneGraph.isNodeValid(nodeIdx)) continue;
            volumeWriteBuf.push_back(vol.toGPU(scene.sceneGraph.getNode(nodeIdx).getWorldPosition()));
        }
        uint32_t volumeCount = static_cast<uint32_t>(volumeWriteBuf.size());
        bindless.descriptorSet->writeFixedBuffer<GPUVolume>(shared.buffers.volumeBufferIndex, volumeWriteBuf.data(),
                                                            volumeCount, gpu.currentFrame * MAX_FIXED_BUFFER, gpu.currentFrame);

        recordFroxel(cmd, volumeCount);
        recordComposite(cmd);

        tracing::endTrace("volumetric pass");
    }

  private:
    // Passes A0..D: build the lit, integrated froxel grid. Each dispatch is separated from the next
    // by a compute->compute barrier for the read-after-write on the volumes (kept in General layout
    // across the chain). integratedVol ends in ShaderReadOnly for the composite to sample.
    void recordFroxel(vk::raii::CommandBuffer& cmd, uint32_t volumeCount) {
        tracing::startTrace("froxel build");
        glm::uvec3 fd = features.volumetrics.froxelDims;
        const uint32_t gx = (fd.x + 3) / 4, gy = (fd.y + 3) / 4, gz = (fd.z + 3) / 4; // [numthreads(4,4,4)]

        auto& mediaTex        = bindless.descriptorSet->getTextureResource(mediaTexIndex);
        auto& mediaPhaseTex   = bindless.descriptorSet->getTextureResource(mediaPhaseTexIndex);
        auto& scatterTex      = bindless.descriptorSet->getTextureResource(scatterTexIndex);
        auto& integratedTex   = bindless.descriptorSet->getTextureResource(integratedTexIndex);

        // Sampled (resting) -> General for the compute writes.
        resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *mediaTex.image,        vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eGeneral);
        resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *mediaPhaseTex.image,   vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eGeneral);
        resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *scatterTex.image,      vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eGeneral);
        resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *integratedTex.image,   vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eGeneral);
        // Note: the shadow atlas rests in eShaderReadOnlyOptimal (see createShadowAtlas / the shadow
        // pass transitions), matching its descriptor, and the DepthAttachment->ShaderReadOnly barrier
        // makes the cascade writes visible to this compute pass. Pass C can sample it directly.

        glm::mat4 invVP = glm::inverse(scene.activeCamera.viewProjection);
        glm::vec3 camPos = scene.activeCamera.position;
        glm::vec3 camFwd = scene.activeCamera.getLookDir();
        // Billboard basis for sampling particle textures in pass B (assumes no camera roll).
        glm::vec3 camRight = glm::normalize(glm::cross(camFwd, glm::vec3(0.0f, 1.0f, 0.0f)));
        glm::vec3 camUp    = glm::cross(camRight, camFwd);
        float nearZ = scene.activeCamera.nearPlane;
        float gridFar = features.volumetrics.gridFar;

        auto computeBarrier = [&]() {
            vk::MemoryBarrier b{.srcAccessMask = vk::AccessFlagBits::eShaderWrite,
                                .dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite};
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, b, {}, {});
        };
        auto dispatchCompute = [&](uint32_t pipeIdx, const auto& pc, uint32_t x, uint32_t y, uint32_t z) {
            using PC = std::decay_t<decltype(pc)>;
            auto& pipe = static_cast<ComputePipeline<PC>&>(*bindless.pipelineManager->getComputePipelines()[pipeIdx]);
            pipe.pushConstantData = pc;
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipe.pipeline);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipe.layout, 0, {**pipe.descriptorSet}, {});
            pipe.pushConstants(cmd);
            cmd.dispatch(x, y, z);
        };

        // ---- A0 clear + A inject analytic volumes (both write mediaVol) ----
        FroxelInjectPushConstants ipc = {
            .volsAddress   = bindless.descriptorSet->getFixedBuffers()[shared.buffers.volumeBufferIndex]->address
                             + static_cast<vk::DeviceSize>(gpu.currentFrame) * MAX_FIXED_BUFFER * sizeof(GPUVolume),
            .volumeCount   = volumeCount,
            .mediaVolIndex = mediaStorageIndex,
            .dims = fd, .nearZ = nearZ, .cameraPos = camPos, .gridFar = gridFar, .cameraForward = camFwd,
            .mediaPhaseIndex = mediaPhaseStorageIndex,
            .invViewProjection = invVP,
            .frame = gpu.totalFrames,
        };
        dispatchCompute(clearPipelineIndex, ipc, gx, gy, gz);
        computeBarrier();
        dispatchCompute(injectVolPipelineIndex, ipc, gx, gy, gz);
        computeBarrier();

        // ---- B inject volumetric particles: bin (particle -> tiles) then gather (tiles -> mediaVol,
        // 1 thread/froxel). Reads the shared particle pool; runs before the particle sim so positions
        // are one frame late — fine for fog. Per-froxel gather bounds the cost so a screen-filling
        // particle can't stall the dispatch.
        uint32_t activePool = 0;
        for (auto& [id, emitter] : scene.particleEmitters)
            activePool = std::max(activePool, emitter.particleOffset + emitter.particleCapacity);
        if (activePool > 0 && !scene.particleEmitters.empty()) {
            // p.size is a world-space extent (the billboard builds its quad in world space, half-
            // extent 0.5*size), so the froxel sphere radius is 0.5*size directly -- no fov factor.
            float billboardScale = 1.0f;
            uint32_t numTiles = tilesX * tilesY;
            vk::DeviceSize countsFrameOffset = static_cast<vk::DeviceSize>(gpu.currentFrame) * numTiles * sizeof(uint32_t);
            vk::DeviceSize listFrameOffset   = static_cast<vk::DeviceSize>(gpu.currentFrame) * numTiles * MAX_PARTICLES_PER_TILE * sizeof(uint32_t);
            vk::DeviceAddress particlesBDA = bindless.descriptorSet->getVariableBuffers()[shared.buffers.particlePoolBufferIndex]->address;
            vk::DeviceAddress emittersBDA  = bindless.descriptorSet->getFixedBuffers()[shared.buffers.emitterBufferIndex]->address
                                             + static_cast<vk::DeviceSize>(gpu.currentFrame) * MAX_EMITTERS * sizeof(GPUParticleEmitter);
            vk::DeviceAddress tileCountsBDA = bindless.descriptorSet->getVariableBuffers()[tileCountsBufferIndex]->address + countsFrameOffset;
            vk::DeviceAddress tileListBDA   = bindless.descriptorSet->getVariableBuffers()[tileListBufferIndex]->address + listFrameOffset;

            // Zero this frame's per-tile counters, then make the fill visible to the bin's atomics.
            cmd.fillBuffer(bindless.descriptorSet->getVariableBuffer(tileCountsBufferIndex), countsFrameOffset, numTiles * sizeof(uint32_t), 0u);
            vk::MemoryBarrier fillBarrier{.srcAccessMask = vk::AccessFlagBits::eTransferWrite, .dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite};
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, fillBarrier, {}, {});

            FroxelBinPushConstants bpc = {
                .particlesAddress = particlesBDA, .emittersAddress = emittersBDA,
                .tileCountsAddress = tileCountsBDA, .tileListAddress = tileListBDA,
                .particleCount = activePool, .tilesX = tilesX, .tilesY = tilesY, .maxPerTile = MAX_PARTICLES_PER_TILE,
                .dims = fd, .nearZ = nearZ, .cameraPos = camPos, .gridFar = gridFar, .cameraForward = camFwd,
                .billboardScale = billboardScale, .viewProj = scene.activeCamera.viewProjection, .invViewProjection = invVP,
            };
            constexpr uint32_t WG = 64; // [numthreads(64,1,1)]
            dispatchCompute(binPipelineIndex, bpc, (activePool + WG - 1) / WG, 1, 1);
            computeBarrier();  // tile lists visible to gather

            FroxelGatherPushConstants gpc = {
                .particlesAddress = particlesBDA, .emittersAddress = emittersBDA,
                .tileCountsAddress = tileCountsBDA, .tileListAddress = tileListBDA,
                .mediaVolIndex = mediaStorageIndex, .tilesX = tilesX, .tilesY = tilesY, .maxPerTile = MAX_PARTICLES_PER_TILE,
                .samplerIndex = shared.defaultSamplerIndex, .frame = gpu.totalFrames, ._padB = 0, ._padC = 0,
                .dims = fd, .nearZ = nearZ, .cameraPos = camPos, .gridFar = gridFar, .cameraForward = camFwd,
                .billboardScale = billboardScale, .cameraRight = camRight, ._padD = 0.0f, .cameraUp = camUp, ._padE = 0.0f,
                .invViewProjection = invVP,
            };
            dispatchCompute(gatherPipelineIndex, gpc, gx, gy, gz);
            computeBarrier();  // gather's mediaVol writes visible to C
        }

        // ---- C light scatter (mediaVol -> scatterVol) ----
        FroxelLightPushConstants lpc = {
            .lightsAddress    = bindless.descriptorSet->getFixedBuffers()[shared.buffers.lightBufferIndex]->address
                                + static_cast<vk::DeviceSize>(gpu.currentFrame) * MAX_FIXED_BUFFER * sizeof(GPULight),
            .lightCount       = scene.getLightLoopBound(),
            .shadowAtlasIndex = scene.shadowAtlas.textureIndex,
            .dims = fd, .mediaVolIndex = mediaStorageIndex, .scatterVolIndex = scatterStorageIndex,
            .nearZ = nearZ, .gridFar = gridFar, .mediaPhaseIndex = mediaPhaseStorageIndex,
            .cameraPos = camPos, .debugMode = static_cast<uint32_t>(features.volumetrics.debugView),
            .cameraForward = camFwd, .frame = gpu.totalFrames,
            .invViewProjection = invVP,
        };
        dispatchCompute(lightPipelineIndex, lpc, gx, gy, gz);
        computeBarrier();

        // ---- D integrate front-to-back (scatterVol -> integratedVol) ----
        FroxelIntegratePushConstants dpc = {
            .dims = fd, .scatterVolIndex = scatterStorageIndex, .integratedVolIndex = integratedStorageIndex,
            .nearZ = nearZ, .gridFar = gridFar,
        };
        dispatchCompute(integratePipelineIndex, dpc, (fd.x + 7) / 8, (fd.y + 7) / 8, 1); // [numthreads(8,8,1)]

        // Return to sampled resting layout. The General->ShaderReadOnly transition also serves as the
        // compute-write -> fragment-read barrier that the composite (E) depends on for integratedVol.
        resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *integratedTex.image,   vk::ImageLayout::eGeneral, vk::ImageLayout::eShaderReadOnlyOptimal);
        resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *scatterTex.image,      vk::ImageLayout::eGeneral, vk::ImageLayout::eShaderReadOnlyOptimal);
        resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *mediaPhaseTex.image,   vk::ImageLayout::eGeneral, vk::ImageLayout::eShaderReadOnlyOptimal);
        resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *mediaTex.image,        vk::ImageLayout::eGeneral, vk::ImageLayout::eShaderReadOnlyOptimal);

        tracing::endTrace("froxel build");
    }

    // Pass E — sample integratedVol by depth->slice and blend onto the HDR composite target.
    void recordComposite(vk::raii::CommandBuffer& cmd) {
        drawFullscreenPass(cmd, *bindless.pipelineManager->getPostProcessPipelines()[applyPipelineIndex],
            *bindless.descriptorSet->getTextureResource(shared.compositeColorTextureIndex).imageView,
            gpu.getSwapchain().getSwapChainExtent(),
            VolumetricApplyPushConstants {
                .integratedTexIndex = integratedTexIndex,
                .samplerIndex       = shared.defaultSamplerIndex,
                .depthTexIndex      = gpu.getSwapchain().getDepthResolveIndex(),
                .depthSamplerIndex  = shared.depthSamplerIndex,
                .dims               = features.volumetrics.froxelDims,
                .nearZ              = scene.activeCamera.nearPlane,
                .cameraPos          = scene.activeCamera.position,
                .gridFar            = features.volumetrics.gridFar,
                .cameraForward      = scene.activeCamera.getLookDir(),
                .debugMode          = static_cast<uint32_t>(features.volumetrics.debugView),
                .invViewProjection  = glm::inverse(scene.activeCamera.viewProjection),
                .mediaTexIndex      = mediaTexIndex,
                .scatterTexIndex    = scatterTexIndex,
                .mediaPhaseTexIndex = mediaPhaseTexIndex,
                .frame              = gpu.totalFrames,
            }, vk::AttachmentLoadOp::eLoad);
    }
};
