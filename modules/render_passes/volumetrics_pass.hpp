#pragma once
#include "render_pass.hpp"
#include "bindless_system.hpp"
#include "scene.hpp"
#include "structs.hpp"
#include "render_buffers.hpp"
#include <random>
#include <algorithm>


class VolumetricsPass : public RenderPass {

    uint32_t textureIndex = 0xFFFFFFFF;
    uint32_t blurTextureIndex = 0xFFFFFFFF;
    uint32_t pipelineIndex = 0xFFFFFFFF;
    uint32_t applyPipelineIndex = 0xFFFFFFFF;
public:

    VolumetricsPass(GpuContext& gpu, BindlessSystem& bindless, Scene& scene, RenderFeatures& features, RenderPassResources& shared) : RenderPass(gpu, bindless, scene, features, shared) {
        shared.buffers.volumeBufferIndex = bindless.descriptorSet->createFixedBuffer<Volume>(MAX_FIXED_BUFFER, false, "Volume");
    }

    void init(uint32_t width, uint32_t height) {
        uint32_t volW = std::max(1u, static_cast<uint32_t>(width * features.volumetrics.resolutionScale));
        uint32_t volH = std::max(1u, static_cast<uint32_t>(height * features.volumetrics.resolutionScale)); 
        resize(textureIndex,volW,volH,gpu.getSwapchain().getHDRColorFormat(),"internal/volumetrics");
        resize(blurTextureIndex,volW,volH,gpu.getSwapchain().getHDRColorFormat(),"internal/volumetrics_blur");

        if (pipelineIndex == 0xFFFFFFFF) {
            pipelineIndex =
            bindless.pipelineManager->createPipeline<VolumetricPushConstants>(PipelineCategory::POSTPROCESS, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone, vk::False,
                                                               vk::False, "shaders/volumetrics.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet(),
                                                               gpu.getSwapchain().getHDRColorFormat());
        }
        // apply pipeline — composites onto the HDR composite target
        if (applyPipelineIndex == 0xFFFFFFFF) {
            applyPipelineIndex = bindless.pipelineManager->createPipeline<VolumetricApplyPushConstants>(
                PipelineCategory::POSTPROCESS_ALPHA_BLEND, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone,
                vk::False, vk::False, "shaders/volumetrics_apply.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet(),
                gpu.getSwapchain().getHDRColorFormat());
        }
    }

    void record(vk::raii::CommandBuffer& cmd, uint32_t imageIndex) {
        tracing::startTrace("volumetric pass");
        if(!features.volumetrics.enabled)
            return;

        auto& volRes = bindless.descriptorSet->getTextureResource(textureIndex);
        vk::Extent2D extent = {volRes.width, volRes.height};

        drawFullscreenPass(cmd, *bindless.pipelineManager->getPostProcessPipelines()[pipelineIndex], *bindless.descriptorSet->getTextureResource(textureIndex).imageView,
                        extent,
                        VolumetricPushConstants {
                                .lightsAddress = bindless.descriptorSet->getFixedBuffers()[shared.buffers.lightBufferIndex]->address + static_cast<vk::DeviceSize>(gpu.currentFrame) * MAX_FIXED_BUFFER * sizeof(GPULight),
                                .volumeBufferAddress = bindless.descriptorSet->getFixedBuffers()[shared.buffers.volumeBufferIndex]->address /* + static_cast<vk::DeviceSize>(gpu.currentFrame) * MAX_FIXED_BUFFER * sizeof(Volume)*/,
                                .lightCount = scene.getLightLoopBound(),
                                .shadowAtlasIndex = scene.shadowAtlas.textureIndex,
                                .depthTextureIndex = gpu.getSwapchain().getDepthResolveIndex(),
                                .depthSamplerIndex = shared.depthSamplerIndex,
                                .cameraPos = scene.activeCamera.position,
                                .numSteps = static_cast<uint32_t>(features.volumetrics.numSteps),
                                .cameraDir = scene.activeCamera.getLookDir(),
                                .volumeCount = scene.getVolumeLoopBound(),
                                .screenSize = {extent.width, extent.height},
                                .maxDist = features.volumetrics.maxDist,
                                .invViewProjection = glm::inverse(scene.activeCamera.viewProjection)
                        }, vk::AttachmentLoadOp::eClear);

        blurAttachment(bindless,cmd,shared.blurPipelineIndex,textureIndex,blurTextureIndex,extent.width,extent.height,features.volumetrics.blurRadius,shared.defaultSamplerIndex);

        drawFullscreenPass(cmd, *bindless.pipelineManager->getPostProcessPipelines()[applyPipelineIndex], *bindless.descriptorSet->getTextureResource(shared.compositeColorTextureIndex).imageView,
                        gpu.getSwapchain().getSwapChainExtent(),
                        VolumetricApplyPushConstants {
                                .volumetricTextureIndex = textureIndex,
                                .samplerIndex = shared.defaultSamplerIndex
                        }, vk::AttachmentLoadOp::eLoad);

        tracing::endTrace("volumetric pass");
    }
};