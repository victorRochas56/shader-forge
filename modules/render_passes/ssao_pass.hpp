#pragma once
#include "render_pass.hpp"
#include "bindless_system.hpp"
#include "scene.hpp"
#include "structs.hpp"
#include <random>

class SSAOPass : public RenderPass {

    uint32_t textureIndex      = 0xFFFFFFFF;
    uint32_t blurTextureIndex  = 0xFFFFFFFF;
    uint32_t noiseTextureIndex = 0xFFFFFFFF;
    uint32_t noiseSamplerIndex = 0xFFFFFFFF;

    uint32_t pipelineIndex      = 0xFFFFFFFF;
    uint32_t applyPipelineIndex = 0xFFFFFFFF;
    // Private blur pipeline because shared.blurPipelineIndex targets swapchain format,
    // and SSAO blurs an R8_UNORM target.
    uint32_t blurPipelineIndex  = 0xFFFFFFFF;

public:
    SSAOPass(GpuContext& gpu, BindlessSystem& bindless, Scene& scene, RenderFeatures& features, RenderPassResources& shared)
        : RenderPass(gpu, bindless, scene, features, shared) {}


    void init(uint32_t width, uint32_t height) {
        uint32_t ssaoW = std::max(1u, static_cast<uint32_t>(width * features.ssao.resolutionScale));
        uint32_t ssaoH = std::max(1u, static_cast<uint32_t>(height * features.ssao.resolutionScale));
        resize(textureIndex,ssaoW, ssaoH, vk::Format::eR8Unorm, "internal/ssao");
        resize(blurTextureIndex,ssaoW, ssaoH, vk::Format::eR8Unorm, "internal/ssao_blur");

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

        // SSAO pipeline writes into the SSAO R8_UNORM texture.
        if (pipelineIndex == 0xFFFFFFFF) {
            pipelineIndex = bindless.pipelineManager->createPipeline<SSAOPushConstants>(
                PipelineCategory::POSTPROCESS, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone,
                vk::False, vk::False, "shaders/ssao.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet(),
                vk::Format::eR8Unorm);
        }

        // R8_UNORM blur pipeline private to SSAO.
        if (blurPipelineIndex == 0xFFFFFFFF) {
            blurPipelineIndex = bindless.pipelineManager->createPipeline<BlurPushConstants>(
                PipelineCategory::POSTPROCESS, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone,
                vk::False, vk::False, "shaders/blur.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet(),
                vk::Format::eR8Unorm);
        }

        // SSAO apply uses multiplicative blending onto the HDR composite target.
        if (applyPipelineIndex == 0xFFFFFFFF) {
            applyPipelineIndex = bindless.pipelineManager->createPipeline<SSAOApplyPushConstants>(
                PipelineCategory::POSTPROCESS_MULTIPLY, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone,
                vk::False, vk::False, "shaders/ssao_apply.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet(),
                gpu.getSwapchain().getHDRColorFormat());
        }
    
    }

    void record(vk::raii::CommandBuffer& cmd, uint32_t imageIndex) {
        if(!features.ssao.enabled)
            return;

        auto swapExtent = gpu.getSwapchain().getSwapChainExtent();
        auto& ssaoTexture = bindless.descriptorSet->getTextureResource(textureIndex);
        vk::Extent2D ssaoExtent{ssaoTexture.width, ssaoTexture.height};

        // SSAO samples the resolved depth (single-sample), not the MSAA depth, so transition that one.
        auto& depthResolveTex = bindless.descriptorSet->getTextureResource(gpu.getSwapchain().getDepthResolveIndex());
        bindless.resourceManager->transitionImageLayouts(cmd, {
            {*depthResolveTex.image,  vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal},
            {*ssaoTexture.image,      vk::ImageLayout::eShaderReadOnlyOptimal,         vk::ImageLayout::eColorAttachmentOptimal},
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

        // Blur at SSAO resolution using the R8_UNORM-targeted blur pipeline.
        bindless.resourceManager->transitionImageLayout(&cmd, *ssaoTexture.image, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
        blurAttachment(bindless, cmd, blurPipelineIndex, textureIndex, blurTextureIndex, ssaoExtent.width, ssaoExtent.height, 2.0f, shared.depthSamplerIndex);

        // Apply to the HDR composite at full resolution (sampler handles upscale)
        drawFullscreenPass(cmd, *bindless.pipelineManager->getPostProcessPipelines()[applyPipelineIndex], *bindless.descriptorSet->getTextureResource(shared.compositeColorTextureIndex).imageView, swapExtent,
            SSAOApplyPushConstants{.ssaoTextureIndex = textureIndex, .samplerIndex = shared.depthSamplerIndex},
            vk::AttachmentLoadOp::eLoad);

        // Transition resolved depth back
        bindless.resourceManager->transitionImageLayout(&cmd, *depthResolveTex.image, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eDepthStencilAttachmentOptimal);

    }
};