#pragma once
#include "bindless_system.hpp"
#include "render_buffers.hpp"
#include "render_pass.hpp"
#include "scene.hpp"
#include "structs.hpp"

// GPU particle system. All emitters share ONE particle pool buffer; each emitter owns a
// contiguous sub-range (reserved in Scene::addEmitter) that it runs as a ring buffer. This pass owns the three buffers.
//
// Buffers:
//   particleBufferIndex       per-frame GPUParticleEmitter descriptors
//   particlePoolBufferIndex   the shared Particle pool, range-allocated per emitter
//   emitterRuntimeBufferIndex persistent EmitterRuntime state (ring head / accumulator), GPU-owned, single copy (must carry across frames)
class ParticlePass : public RenderPass {
    uint32_t spawnPipelineIndex = 0xFFFFFFFF; // compute: spawn + integrate
    uint32_t simPipelineIndex = 0xFFFFFFFF;   // compute: spawn + integrate
    uint32_t drawPipelineIndex = 0xFFFFFFFF;  // graphics: instanced billboards

  public:
    ParticlePass(GpuContext& gpu, BindlessSystem& bindless, Scene& scene, RenderFeatures& features, RenderPassResources& shared)
        : RenderPass(gpu, bindless, scene, features, shared) {
        // Per-frame emitter descriptors (dynamic-offset), sized like the light buffer.
        shared.buffers.emitterBufferIndex =
            bindless.descriptorSet->createFixedBuffer<GPUParticleEmitter>(MAX_FRAMES_IN_FLIGHT * MAX_EMITTERS, true, "ParticleEmitters");
        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
            bindless.descriptorSet->setBufferFrameOffset(shared.buffers.emitterBufferIndex, i, MAX_EMITTERS * i);

        // Shared pool: one big device-address buffer
        // eTransferDst lets a compute clear zero freshly reserved ranges.
        // TODO : Host-visible for now (matches the rest of the infra); a device-local pool is the eventual bandwidth win.
        shared.buffers.particlePoolBufferIndex = bindless.descriptorSet->createVariableBuffer(
            PARTICLE_POOL_SIZE,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eTransferDst, false,
            "ParticlePool");
        shared.buffers.emitterRuntimeBufferIndex = bindless.descriptorSet->createFixedBuffer<EmitterRuntime>(MAX_EMITTERS, false, "EmitterRuntime");
    }

    void init(uint32_t width, uint32_t height) override {
        // TODO(compute pipeline): create the sim compute pipeline (spawn from emissionRate +
        // spawnAccumulator, integrate velocity/drag/age, retire age >= lifeSpan) writing the pool
        // by device address, and the instanced billboard draw pipeline (alpha-blended, depth-tested
        // against the resolved depth, reads Particle + GPUParticleEmitter by device address).
        spawnPipelineIndex = bindless.pipelineManager->createComputePipeline<ParticleComputePushConstants>(
            "shaders/particle_compute.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet(),
            "spawnMain");
        simPipelineIndex = bindless.pipelineManager->createComputePipeline<ParticleComputePushConstants>(
            "shaders/particle_compute.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet(),
            "integrateMain");
        drawPipelineIndex = bindless.pipelineManager->createPipeline<ParticleDrawPushConstants>(
            PipelineCategory::POSTPROCESS_ALPHA_BLEND, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone, vk::True, vk::False,
            "shaders/particle_draw.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet(),
            gpu.getSwapchain().getHDRColorFormat());
        (void)width;
        (void)height;
    }

    void record(vk::raii::CommandBuffer& cmd, uint32_t imageIndex) override {
        tracing::startTrace("particle pass");

        // Refresh each live emitter's descriptor for this frame slice. Emitters use persistent
        // slots (keyed by id), so we update in place — mirroring the per-frame light loop — rather
        // than rebuilding a contiguous list like volumes/billboards. World spawn frame is pulled
        // fresh from the owning node so a moving/rotating emitter tracks it. Runs during command
        // recording (post-fence), so the slice written is never in flight.
        for (auto& [id, emitter] : scene.particleEmitters) {
            if (!scene.sceneGraph.isNodeValid(emitter.nodeIndex))
                continue;
            Node& node = scene.sceneGraph.getNode(emitter.nodeIndex);
            glm::quat worldRot = node.getWorldRotation();
            glm::vec3 spawnPos = node.getWorldPosition() + worldRot * emitter.positionOffset;
            glm::quat spawnRot = worldRot * emitter.rotationOffset;
            bindless.descriptorSet->updateFixedBufferWithOffset<GPUParticleEmitter>(shared.buffers.emitterBufferIndex, id,
                                                                                    emitter.toGPU(spawnPos, spawnRot), gpu.currentFrame);
        }
        // Two-stage sim: spawn (1 thread/emitter, owns ring bookkeeping) then integrate
        // (1 thread/particle over the whole pool), with a barrier between them for the RAW hazard
        // on the pool + runtime buffers. TODO: draw the pool into the HDR composite target after.
        if (spawnPipelineIndex != 0xFFFFFFFF && simPipelineIndex != 0xFFFFFFFF && !scene.particleEmitters.empty()) {
            auto& spawnPipeline = static_cast<ComputePipeline<ParticleComputePushConstants>&>(*bindless.pipelineManager->getComputePipelines()[spawnPipelineIndex]);
            auto& simPipeline   = static_cast<ComputePipeline<ParticleComputePushConstants>&>(*bindless.pipelineManager->getComputePipelines()[simPipelineIndex]);

            // Emitters allocate contiguous sub-ranges from the front of the pool, so only the region
            // up to the high-water mark (max offset+capacity) is ever live. Integrating the full
            // 128MB pool every frame would launch ~2.79M threads regardless of load — bound it here.
            uint32_t activePoolParticles = 0;
            for (auto& [id, emitter] : scene.particleEmitters)
                activePoolParticles = std::max(activePoolParticles, emitter.particleOffset + emitter.particleCapacity);

            // Pool is a variable buffer (getVariableBuffers). The emitter descriptors are per-frame,
            // so offset the address to this frame's slice — record() just wrote it above.
            vk::DeviceAddress emitterBase = bindless.descriptorSet->getFixedBuffers()[shared.buffers.emitterBufferIndex]->address;
            ParticleComputePushConstants pushConstants = {
                .runtimeBDA   = bindless.descriptorSet->getFixedBuffers()[shared.buffers.emitterRuntimeBufferIndex]->address,
                .emittersBDA  = emitterBase + static_cast<vk::DeviceSize>(gpu.currentFrame) * MAX_EMITTERS * sizeof(GPUParticleEmitter),
                .particlesBDA = bindless.descriptorSet->getVariableBuffers()[shared.buffers.particlePoolBufferIndex]->address,
                .emitterCount = static_cast<uint32_t>(scene.particleEmitters.size()),
                .particleCount = activePoolParticles,
                .dt           = gpu.deltaTime
            };
            spawnPipeline.pushConstantData = pushConstants;
            simPipeline.pushConstantData   = pushConstants;

            constexpr uint32_t WG = 64; // matches [NumThreads(64,1,1)]
            uint32_t poolParticles = activePoolParticles;

            // Spawn: one thread per emitter.
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, spawnPipeline.pipeline);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, spawnPipeline.layout, 0, {**spawnPipeline.descriptorSet}, {});
            spawnPipeline.pushConstants(cmd);
            cmd.dispatch((pushConstants.emitterCount + WG - 1) / WG, 1, 1);

            // Barrier: spawn's writes to the pool + runtime must be visible to integrate.
            vk::MemoryBarrier simBarrier{.srcAccessMask = vk::AccessFlagBits::eShaderWrite,
                                         .dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite};
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, simBarrier, {}, {});

            // Integrate: one thread per particle over the whole pool.
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, simPipeline.pipeline);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, simPipeline.layout, 0, {**simPipeline.descriptorSet}, {});
            simPipeline.pushConstants(cmd);
            cmd.dispatch((poolParticles + WG - 1) / WG, 1, 1);

            // Barrier: integrate's pool writes must be visible to the draw pass' vertex read.
            vk::MemoryBarrier drawBarrier{.srcAccessMask = vk::AccessFlagBits::eShaderWrite,
                                          .dstAccessMask = vk::AccessFlagBits::eShaderRead};
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eVertexShader, {}, drawBarrier, {}, {});
        }

        // ---- Draw: instanced billboards into the HDR composite, one draw per emitter ----
        // NOTE: particles are not depth-sorted, so alpha-blend order between overlapping particles
        // is approximate (fine for additive/soft particles). Each emitter draws its whole ring; the
        // vertex shader degenerates dead slots (age < 0).
        if (drawPipelineIndex != 0xFFFFFFFF && !scene.particleEmitters.empty()) {
            auto extent = gpu.getSwapchain().getSwapChainExtent();

            // Depth test is done in-shader by sampling the resolved depth, so make it readable.
            auto& depthResolveTex = bindless.descriptorSet->getTextureResource(gpu.getSwapchain().getDepthResolveIndex());
            resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *depthResolveTex.image,
                vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

            vk::RenderingAttachmentInfo colorAttachment = {
                .imageView   = *bindless.descriptorSet->getTextureResource(shared.compositeColorTextureIndex).imageView,
                .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .loadOp      = vk::AttachmentLoadOp::eLoad,
                .storeOp     = vk::AttachmentStoreOp::eStore};
            vk::RenderingInfo renderInfo = {.renderArea = {.offset = {0, 0}, .extent = extent},
                                            .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &colorAttachment};

            cmd.beginRendering(renderInfo);
            setFullscreenViewport(cmd, extent);

            auto& pipeline = bindless.pipelineManager->getPostProcessPipelines()[drawPipelineIndex];
            bindPipeline(cmd, *pipeline);

            vk::DeviceAddress poolAddress = bindless.descriptorSet->getVariableBuffers()[shared.buffers.particlePoolBufferIndex]->address;
            for (auto& [id, emitter] : scene.particleEmitters) {
                if (emitter.particleCapacity == 0) continue;
                ParticleDrawPushConstants pc = {
                    .viewProjection    = scene.activeCamera.viewProjection,
                    .particlesBDA      = poolAddress,
                    .emittersBDA       = bindless.descriptorSet->getFixedBuffers()[shared.buffers.emitterBufferIndex]->address + static_cast<vk::DeviceSize>(gpu.currentFrame) * MAX_EMITTERS * sizeof(GPUParticleEmitter),
                    .lightsBDA         = bindless.descriptorSet->getFixedBuffers()[shared.buffers.lightBufferIndex]->address + static_cast<vk::DeviceSize>(gpu.currentFrame) * MAX_FIXED_BUFFER * sizeof(GPULight),
                    .lightCount        = scene.getLightLoopBound(),
                    .samplerIndex      = shared.defaultSamplerIndex,
                    .depthTextureIndex = gpu.getSwapchain().getDepthResolveIndex(),
                    .depthSamplerIndex = shared.depthSamplerIndex,
                    .resolution        = glm::uvec2(extent.width, extent.height),
                    .cameraPos         = scene.activeCamera.position,
                    .farPlane          = scene.activeCamera.farPlane,
                    .cameraForward     = scene.activeCamera.getLookDir(),
                    .nearPlane         = scene.activeCamera.nearPlane,
                    .shadowAtlasIndex  = scene.shadowAtlas.textureIndex,
                    .emitterIndex      = id,
                    .sphereRoundness   = emitter.sphereRoundness,
                    .opacity           = emitter.opacity,
                };
                cmd.pushConstants<ParticleDrawPushConstants>(*pipeline->layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pc);
                cmd.draw(6, emitter.particleCapacity, 0, 0); // 6 verts/quad, one instance per ring slot
            }

            cmd.endRendering();

            resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *depthResolveTex.image,
                vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eDepthStencilAttachmentOptimal);
        }

        (void)imageIndex;

        tracing::endTrace("particle pass");
    }
};
