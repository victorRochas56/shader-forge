#pragma once
#include "render_pass.hpp"
#include "bindless_system.hpp"
#include "scene.hpp"
#include "structs.hpp"
#include <random>


class SSAOPass : RenderPass {

    uint32_t textureIndex      = 0xFFFFFFFF;
    uint32_t blurTextureIndex  = 0xFFFFFFFF;
    uint32_t noiseTextureIndex = 0xFFFFFFFF;
    uint32_t noiseSamplerIndex = 0xFFFFFFFF;

    uint32_t pipelineIndex      = 0xFFFFFFFF;
    uint32_t applyPipelineIndex = 0xFFFFFFFF;

public:
    using RenderPass::RenderPass;


    void init(uint32_t width, uint32_t height) {
        resize(textureIndex,width, height, vk::Format::eR8Unorm, "internal/ssao");
        resize(blurTextureIndex,width, height, vk::Format::eR8Unorm, "internal/ssao_blur");

        // 4x4 noise texture (RGBA8, random tangent-space rotation vectors)
        // Only create once — noise doesn't depend on screen size
        if (noiseTextureIndex == 0xFFFFFFFF) {
            std::mt19937 rng(42); // fixed seed for reproducibility
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            std::array<uint8_t, 4 * 4 * 4> noiseData; // 4x4 RGBA8
            for (int i = 0; i < 16; i++) {
                float x = dist(rng);
                float y = dist(rng);
                // Rotate around Z in tangent space, so z=0
                float len = std::sqrt(x * x + y * y);
                if (len > 0.0f) { x /= len; y /= len; }
                noiseData[i * 4 + 0] = static_cast<uint8_t>((x * 0.5f + 0.5f) * 255.0f);
                noiseData[i * 4 + 1] = static_cast<uint8_t>((y * 0.5f + 0.5f) * 255.0f);
                noiseData[i * 4 + 2] = 0;
                noiseData[i * 4 + 3] = 255;
            }
            auto [noiseImage, noiseMemory, noiseImageView] =
                bindless.resourceManager->createTexture(noiseData.data(), 4, 4, vk::Format::eR8G8B8A8Unorm, vk::ImageType::e2D, vk::ImageViewType::e2D, vk::SampleCountFlagBits::e1, false);
            noiseTextureIndex = bindless.descriptorSet->allocateTexture(std::move(noiseImage), std::move(noiseMemory), std::move(noiseImageView), "internal/ssao_noise");

            // Noise sampler: repeat + nearest (tiled across screen)
            noiseSamplerIndex = bindless.descriptorSet->allocateSampler(vk::Filter::eNearest, vk::SamplerMipmapMode::eNearest, vk::SamplerAddressMode::eRepeat,
                                                                    VK_FALSE, 1.0f, VK_FALSE, vk::CompareOp::eNever, vk::BorderColor::eFloatOpaqueBlack);
        }

        // SSAO pipeline (only create once)
        if (pipelineIndex == 0xFFFFFFFF) {
            pipelineIndex = bindless.pipelineManager->createPipeline<SSAOPushConstants>(
                PipelineCategory::POSTPROCESS, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone,
                vk::False, vk::False, "shaders/ssao.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet());
        }

        // SSAO apply pipeline with multiplicative blending (only create once)
        if (applyPipelineIndex == 0xFFFFFFFF) {
            applyPipelineIndex = bindless.pipelineManager->createPipeline<SSAOApplyPushConstants>(
                PipelineCategory::POSTPROCESS_MULTIPLY, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone,
                vk::False, vk::False, "shaders/ssao_apply.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet());
        }
    
    }

    void record(vk::raii::CommandBuffer& cmd, uint32_t imageIndex) {
        auto swapExtent = gpu.getSwapchain().getSwapChainExtent();
        auto& ssaoTexture = bindless.descriptorSet->getTextureResource(textureIndex);
        vk::Extent2D ssaoExtent{ssaoTexture.width, ssaoTexture.height};

        // Transition depth to readable, SSAO target to color attachment
        bindless.resourceManager->transitionImageLayouts(cmd, {
            {gpu.getSwapchain().getDepthImage(),  vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal},
            {*ssaoTexture.image,          vk::ImageLayout::eShaderReadOnlyOptimal,          vk::ImageLayout::eColorAttachmentOptimal},
        });

        // Render SSAO at (potentially lower) SSAO resolution
        drawFullscreenPass(cmd, *bindless.pipelineManager->getPostProcessPipelines()[pipelineIndex], *ssaoTexture.imageView, ssaoExtent,
            SSAOPushConstants{.invProjection = glm::inverse(scene.activeCamera.projectionMatrix),
                            .depthIndex = gpu.getSwapchain().getDepthResolveIndex(),
                            .depthSamplerIndex = shared.depthSamplerIndex,
                            .noiseIndex = noiseTextureIndex,
                            .noiseSamplerIndex = noiseSamplerIndex,
                            .resolution = glm::uvec2(ssaoExtent.width, ssaoExtent.height),
                            .radius = features.ssao.radius,
                            .bias = features.ssao.bias,
                            .power = features.ssao.power,
                            .kernelSize = 32},
            vk::AttachmentLoadOp::eClear, {1.0f, 1.0f, 1.0f, 1.0f});

        // Blur at SSAO resolution
        bindless.resourceManager->transitionImageLayout(&cmd, *ssaoTexture.image, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
        blurAttachment(bindless, cmd, shared.blurPipelineIndex, textureIndex, blurTextureIndex, ssaoExtent.width, ssaoExtent.height, 2.0f, shared.depthSamplerIndex);

        // Apply to swapchain at full resolution (sampler handles upscale)
        drawFullscreenPass(cmd, *bindless.pipelineManager->getPostProcessPipelines()[applyPipelineIndex], *gpu.getSwapchain().getSwapChainImageViews()[imageIndex], swapExtent,
            SSAOApplyPushConstants{.ssaoTextureIndex = textureIndex, .samplerIndex = shared.depthSamplerIndex},
            vk::AttachmentLoadOp::eLoad);

        // Transition depth back
        bindless.resourceManager->transitionImageLayout(&cmd, gpu.getSwapchain().getDepthImage(), vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eDepthStencilAttachmentOptimal);

    }
};