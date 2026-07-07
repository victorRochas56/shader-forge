#pragma once
#include "render_pass.hpp"
#include "bindless_system.hpp"
#include "scene.hpp"
#include "structs.hpp"
#include <random>
#include <algorithm>
#include <cmath>


class SSRPass : public RenderPass {

    uint32_t currentTextureIndex = 0xFFFFFFFF;
    uint32_t historyTextureIndices[2] = {0xFFFFFFFF, 0xFFFFFFFF};
    uint32_t historyFlip = 0;

    uint32_t pipelineIndex = 0xFFFFFFFF;
    uint32_t applyPipelineIndex = 0xFFFFFFFF;
    uint32_t accumulatePipelineIndex = 0xFFFFFFFF;
    uint32_t passDataBufferIndex;

    bool historyInvalid = true;

public:
    SSRPass(GpuContext& gpu, BindlessSystem& bindless, Scene& scene, RenderFeatures& features, RenderPassResources& shared) : RenderPass(gpu, bindless, scene, features, shared) {
        passDataBufferIndex = bindless.descriptorSet->createFixedBuffer<SSRPassData>(MAX_FRAMES_IN_FLIGHT, true, "SSRPassData");
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            bindless.descriptorSet->setBufferFrameOffset(passDataBufferIndex,i,i);
        }
    }

    void init(uint32_t width, uint32_t height) {
        uint32_t ssrW = std::max(1u, static_cast<uint32_t>(width * features.ssr.resolutionScale));
        uint32_t ssrH = std::max(1u, static_cast<uint32_t>(height * features.ssr.resolutionScale));
        // SSR runs at a reduced resolution; the Hi-Z trace still indexes the full-res pyramid.
        resize(currentTextureIndex, ssrW, ssrH, gpu.getSwapchain().getHDRColorFormat(), "internal/ssr_current");
        resize(historyTextureIndices[0], ssrW, ssrH, gpu.getSwapchain().getHDRColorFormat(), "internal/ssr_history0");
        resize(historyTextureIndices[1], ssrW, ssrH, gpu.getSwapchain().getHDRColorFormat(), "internal/ssr_history1");

        historyInvalid = true;

        // SSR ray-trace + accumulate write into ssr_current / ssr_history (HDR; reflections sample HDR scene color).
        if (pipelineIndex == 0xFFFFFFFF) {
            pipelineIndex = bindless.pipelineManager->createPipeline<SSRPushConstants>(
                PipelineCategory::POSTPROCESS, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone,
                vk::False, vk::False, "shaders/ssr.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet(),
                gpu.getSwapchain().getHDRColorFormat());

            //initialize pass data too
            bindless.descriptorSet->allocateFixedBuffer(passDataBufferIndex,SSRPassData {});
        }

        if (accumulatePipelineIndex == 0xFFFFFFFF) {
            accumulatePipelineIndex = bindless.pipelineManager->createPipeline<SSRAccumulatePushConstants>(
                PipelineCategory::POSTPROCESS, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone,
                vk::False, vk::False, "shaders/ssr_accumulate.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet(),
                gpu.getSwapchain().getHDRColorFormat());
        }

        // Apply composites onto the HDR composite target.
        if (applyPipelineIndex == 0xFFFFFFFF) {
            applyPipelineIndex = bindless.pipelineManager->createPipeline<SSRApplyPushConstants>(
                PipelineCategory::POSTPROCESS_ALPHA_BLEND, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone,
                vk::False, vk::False, "shaders/ssr_apply.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet(),
                gpu.getSwapchain().getHDRColorFormat());
        }
        
    }

    void record(vk::raii::CommandBuffer& cmd, uint32_t imageIndex) {
        tracing::startTrace("ssr pass");
        if(!features.ssr.enabled)
            return;

        auto swapExtent = gpu.getSwapchain().getSwapChainExtent();
        auto& ssrCurrent = bindless.descriptorSet->getTextureResource(currentTextureIndex);
        vk::Extent2D ssrExtent{ssrCurrent.width, ssrCurrent.height};

        // Trace down to mip 0: at one texel per cell, minZ is the exact depth under the ray,
        // so sloped/grazing surfaces don't band. Stopping coarser biases minZ to the cell's
        // nearest texel and stripes the floor.
        uint32_t hiZStopLevel = 0;

        uint32_t readHistory = historyTextureIndices[historyFlip];
        uint32_t writeHistory = historyTextureIndices[1 - historyFlip];
        auto& ssrWriteHist = bindless.descriptorSet->getTextureResource(writeHistory);

        // --- Sub-pass 0: Generate blurred mip chain for cone tracing (GPU Pro 5 style)
        // Mip 0 is already populated by the geometry pass MSAA resolve.
        // For each subsequent mip: 2-pass separable Gaussian blur reading mip N-1, writing to mip N.
        auto& colorRes = bindless.descriptorSet->getTextureResource(shared.colorResolveTextureIndex);
        auto& tempTexture = bindless.descriptorSet->getTextureResource(shared.tempBlurTextureIndex);
        auto& blurPipeline = *bindless.pipelineManager->getPostProcessPipelines()[shared.blurPipelineIndex];

        uint32_t maxBlurMips = std::min(shared.fullscreenMipLevels, 6u);
        for (uint32_t mip = 1; mip < shared.fullscreenMipLevels; ++mip) {
            uint32_t mipW = std::max(1u, colorRes.width >> mip);
            uint32_t mipH = std::max(1u, colorRes.height >> mip);
            vk::Extent2D mipExtent{mipW, mipH};

            // Horizontal blur: read colorResolve mip N-1 -> write temp mip N
            bindless.resourceManager->transitionImageLayout(&cmd, *tempTexture.image,
                vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eColorAttachmentOptimal, mip, 1);

            drawFullscreenPass(cmd, blurPipeline, (*shared.tempBlurMipViews)[mip], mipExtent,
                BlurPushConstants{.inputTextureIndex = shared.colorResolveTextureIndex, .samplerIndex = shared.defaultSamplerIndex,
                                .isHorizontal = 1, .blurRadius = 1.0f, .resolution = glm::uvec2(mipW, mipH),
                                .mipLevel = mip - 1});

            bindless.resourceManager->transitionImageLayout(&cmd, *tempTexture.image,
                vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, mip, 1);

            // Vertical blur: read temp mip N -> write colorResolve mip N
            bindless.resourceManager->transitionImageLayout(&cmd, *colorRes.image,
                vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eColorAttachmentOptimal, mip, 1);

            drawFullscreenPass(cmd, blurPipeline, (*shared.colorResolveMipViews)[mip], mipExtent,
                BlurPushConstants{.inputTextureIndex = shared.tempBlurTextureIndex, .samplerIndex = shared.defaultSamplerIndex,
                                .isHorizontal = 0, .blurRadius = 1.0f, .resolution = glm::uvec2(mipW, mipH),
                                .mipLevel = mip});

            bindless.resourceManager->transitionImageLayout(&cmd, *colorRes.image,
                vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, mip, 1);
        }

        // --- Sub-pass 1: Ray trace -> ssrCurrentTextureIndex ---
        bindless.resourceManager->transitionImageLayouts(cmd, {
            {gpu.getSwapchain().getDepthImage(),  vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal},
            {*ssrCurrent.image,           vk::ImageLayout::eShaderReadOnlyOptimal,          vk::ImageLayout::eColorAttachmentOptimal},
        });

        bindless.descriptorSet->updateFixedBufferWithOffset(passDataBufferIndex, 0,
                                        SSRPassData{
                                            .invViewProj = glm::inverse(scene.activeCamera.viewProjection),
                                            .viewProj = scene.activeCamera.viewProjection,
                                            .cameraPos = scene.activeCamera.position,
                                            .depthIndex = shared.hiZTextureIndex,
                                            .depthSamplerIndex = shared.depthSamplerIndex,
                                            .colorIndex = shared.colorResolveTextureIndex,
                                            .colorSamplerIndex = shared.defaultSamplerIndex,
                                            .roughnessMetalIndex = shared.roughnessMetalTextureIndex,
                                            .roughnessMetalSamplerIndex = shared.defaultSamplerIndex,
                                            .normalIndex = shared.normalTextureIndex,
                                            .normalSamplerIndex = shared.defaultSamplerIndex,
                                            // Full-res base: drives the Hi-Z cell grid and color-mip selection.
                                            // The half-res output is handled by the ssrExtent viewport below.
                                            .resolution = glm::uvec2(swapExtent.width, swapExtent.height),
                                            .hiZIndex = shared.hiZTextureIndex,
                                            .hiZMipLevels = shared.hiZMipLevels,
                                            .thickness = features.ssr.thickness,
                                            .roughnessThreshold = features.ssr.roughnessThreshold,
                                            .maxSteps = features.ssr.maxSteps,
                                            .frameIndex = gpu.totalFrames,
                                            .hiZStopLevel = hiZStopLevel,
                                        }, gpu.currentFrame);

        drawFullscreenPass(cmd, *bindless.pipelineManager->getPostProcessPipelines()[pipelineIndex], *ssrCurrent.imageView, ssrExtent,
            SSRPushConstants{ .ssrPassDataAddress = bindless.descriptorSet->getFixedBuffers()[passDataBufferIndex]->address + static_cast<vk::DeviceSize>(gpu.currentFrame) * sizeof(SSRPassData)},
            vk::AttachmentLoadOp::eClear, {0.0f, 0.0f, 0.0f, 0.0f});

        // --- Sub-pass 2: Temporal accumulate -> writeHistory ---
        bindless.resourceManager->transitionImageLayouts(cmd, {
            {*ssrCurrent.image,    vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal},
            {*ssrWriteHist.image,  vk::ImageLayout::eShaderReadOnlyOptimal,  vk::ImageLayout::eColorAttachmentOptimal},
        });

        drawFullscreenPass(cmd, *bindless.pipelineManager->getPostProcessPipelines()[accumulatePipelineIndex], *ssrWriteHist.imageView, ssrExtent,
            SSRAccumulatePushConstants{
                .currentSSRIndex = currentTextureIndex,
                .historySSRIndex = readHistory,
                .motionVectorIndex = shared.motionVectorTextureIndex,
                .samplerIndex = shared.defaultSamplerIndex,
                .temporalBlend = features.ssr.temporalBlend,
                .historyValid = historyInvalid ? 0u : 1u,
            },
            vk::AttachmentLoadOp::eClear, {0.0f, 0.0f, 0.0f, 0.0f});

        bindless.resourceManager->transitionImageLayout(&cmd, *ssrWriteHist.image, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

        // --- Sub-pass 3: Apply accumulated SSR to the HDR composite ---
        drawFullscreenPass(cmd, *bindless.pipelineManager->getPostProcessPipelines()[applyPipelineIndex], *bindless.descriptorSet->getTextureResource(shared.compositeColorTextureIndex).imageView, swapExtent,
            SSRApplyPushConstants{
                .samplerIndex = shared.defaultSamplerIndex,
                .ssrTextureIndex = writeHistory,
            },
            vk::AttachmentLoadOp::eLoad);

        // transition depth back
        bindless.resourceManager->transitionImageLayout(&cmd, gpu.getSwapchain().getDepthImage(), vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eDepthStencilAttachmentOptimal);

        historyFlip = 1 - historyFlip;
        historyInvalid = false;
        tracing::endTrace("ssr pass");
    }
};
