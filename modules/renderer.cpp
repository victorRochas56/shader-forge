#include "renderer.hpp"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>

#include <stb_image.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
 
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#include "gizmo.hpp"
#include "node_ops.hpp"
#include "pipelines.hpp"
#include "profiling.hpp"
#include "swapchain.hpp"

// TODO clustered lights? (forward +)
//      pass the N nearest lights to the lit shader
// TODO spot and area lights
// TODO node deletion
// TODO multithread command buffer recording
// volumetrics

glm::mat4 calculateLightSpaceMatrix(Light& light, Camera& camera);

void calculatePointLightFaceMatrices(Light& light, const glm::vec3& lightPos);

void calculateCascadedLightSpaceMatrices(Light& light, Camera& camera, Renderer* renderer);

Renderer::Renderer(GpuContext& gpu, BindlessSystem& bindless, Scene& scene) : scene(scene), gpu(gpu), bindless(bindless) {}
Renderer::~Renderer() = default;

/////=================================================INIT=================================================/////

void Renderer::initVulkan(uint32_t startWidth, uint32_t startHeight) {
    // GpuContext::initCore() must have been called by App already.
    bindless.initResources(gpu);
    gpu.initSwapchain(*bindless.resourceManager, *bindless.descriptorSet);
    bindless.initPipelineManager(gpu);

    // initializing default camera
    scene.activeCamera = Camera{.position = glm::vec3(1, 1, 1),
                          .target = glm::vec3(0, 0, 0),
                          .fov = 45.0,
                          .aspectRatio = static_cast<float>(startWidth) / static_cast<float>(startHeight),
                          .nearPlane = 0.1,
                          .farPlane = 500.0};
    scene.activeCamera.calculateViewProjectionMatrix();

    /////=====================================DESCRIPTOR SET BUFFERS=================================================/////
    vertexBufferIndex = bindless.descriptorSet->createVariableBuffer(256 * 1024 * 1024);                                       // 256 mb vertex buffer
    indexBufferIndex = bindless.descriptorSet->createVariableBuffer(128 * 1024 * 1024, vk::BufferUsageFlagBits::eIndexBuffer); // index buffer (128 MB)
    scene.assetManager.init(bindless.resourceManager.get(), bindless.descriptorSet.get(), vertexBufferIndex, indexBufferIndex);

    sdfPassDataBufferIndex = bindless.descriptorSet->createFixedBuffer<SDF>(MAX_FIXED_BUFFER);
    volumeBufferIndex = bindless.descriptorSet->createFixedBuffer<Volume>(MAX_FIXED_BUFFER, false);

    // these buffers store the data once per frame in flight since they are usually accessed every frame by the CPU
    modelMatrixBufferIndex = bindless.descriptorSet->createFixedBuffer<glm::mat4>(MAX_FRAMES_IN_FLIGHT * MAX_FIXED_BUFFER, true);
    shadowDrawDataBufferIndex = bindless.descriptorSet->createFixedBuffer<ShadowDrawData>(MAX_FRAMES_IN_FLIGHT * MAX_SHADOW_CASTERS * MAX_FIXED_BUFFER, true);
    lightBufferIndex = bindless.descriptorSet->createFixedBuffer<GPULight>(MAX_LIGHTS * MAX_FRAMES_IN_FLIGHT, true);
    litPassDataBufferIndex = bindless.descriptorSet->createFixedBuffer<LitPassData>(MAX_FRAMES_IN_FLIGHT * MAX_FIXED_BUFFER, true);
    ssrPassDataBufferIndex = bindless.descriptorSet->createFixedBuffer<SSRPassData>(MAX_FRAMES_IN_FLIGHT, true);

    // sets the frame offsets for each buffer
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        bindless.descriptorSet->setBufferFrameOffset(modelMatrixBufferIndex, i, MAX_FIXED_BUFFER * i);
        bindless.descriptorSet->setBufferFrameOffset(lightBufferIndex, i, MAX_LIGHTS * i);
        bindless.descriptorSet->setBufferFrameOffset(shadowDrawDataBufferIndex, i, MAX_SHADOW_CASTERS * MAX_FIXED_BUFFER * i);
        bindless.descriptorSet->setBufferFrameOffset(litPassDataBufferIndex,i, MAX_FIXED_BUFFER * i);
        bindless.descriptorSet->setBufferFrameOffset(ssrPassDataBufferIndex,i,i);
    }

    
    litDrawDataBufferIndex = bindless.descriptorSet->createFixedBuffer<LitDrawData>(MAX_FRAMES_IN_FLIGHT * MAX_FIXED_BUFFER, true);
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        bindless.descriptorSet->setBufferFrameOffset(litDrawDataBufferIndex, i, MAX_FIXED_BUFFER * i);
    }
    
    // indirect draw buffers (separate for shadow and lit passes)
    // Shadow indirect buffer needs one slot per shadow-casting light per frame so multiple lights
    // recorded into the same command buffer don't overwrite each other's draw commands.
    std::tie(indirectDrawBuffer, indirectDrawBufferMemory, indirectDrawBufferMapped)       = bindless.resourceManager->createIndirectDrawBuffer(MAX_SHADOW_CASTERS);
    std::tie(litIndirectDrawBuffer, litIndirectDrawBufferMemory, litIndirectDrawBufferMapped) = bindless.resourceManager->createIndirectDrawBuffer();
    
    //init gizmos
    Gizmos::init(MAX_GIZMO_LINES, &*bindless.descriptorSet, sdfPassDataBufferIndex);

    // after having created all our desire buffers we can initialize the descriptor set
    bindless.descriptorSet->createDescriptorSet();

    gpu.createSwapchainAndSync();
    uint32_t ssaoW = std::max(1u, static_cast<uint32_t>(startWidth * features.ssao.resolutionScale));
    uint32_t ssaoH = std::max(1u, static_cast<uint32_t>(startHeight * features.ssao.resolutionScale));
    uint32_t ssrW = std::max(1u, static_cast<uint32_t>(startWidth * features.ssr.resolutionScale));
    uint32_t ssrH = std::max(1u, static_cast<uint32_t>(startHeight * features.ssr.resolutionScale));
    uint32_t volW = std::max(1u, static_cast<uint32_t>(startWidth * features.volumetrics.resolutionScale));
    uint32_t volH = std::max(1u, static_cast<uint32_t>(startHeight * features.volumetrics.resolutionScale));
    createSSAOResources(ssaoW, ssaoH);
    createShadowAtlas(SHADOW_ATLAS_SIZE);
    createRoughnessMetalResources(startWidth, startHeight);
    createNormalResources(startWidth, startHeight);
    createMotionVectorResources(startWidth,startHeight);
    createColorResolveResources(startWidth, startHeight);
    createSSRResources(ssrW, ssrH);
    createHiZResources(startWidth, startHeight);
    createSDFResources(startWidth,startHeight);
    createVolumetricResources(volW,volH);

#if DEBUG == 1
    bindless.descriptorSet->debugDescriptorSet("after_createDescriptorSet");
#endif

    /////S=================================================DEFAULTS=================================================/////

    defaultSamplerIndex = bindless.descriptorSet->allocateSampler(vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear, vk::SamplerAddressMode::eRepeat, VK_TRUE, 16.0, VK_FALSE,
                                                         vk::CompareOp::eLessOrEqual, vk::BorderColor::eFloatOpaqueBlack);
    depthSamplerIndex = bindless.descriptorSet->allocateSampler(vk::Filter::eNearest, vk::SamplerMipmapMode::eNearest, vk::SamplerAddressMode::eClampToEdge, VK_FALSE, 16.0, VK_FALSE,
                                                       vk::CompareOp::eLessOrEqual, vk::BorderColor::eFloatOpaqueBlack);
    shadowSamplerIndex = bindless.descriptorSet->allocateSampler(vk::Filter::eNearest,
                                                        vk::SamplerMipmapMode::eNearest,
                                                        vk::SamplerAddressMode::eClampToBorder,
                                                        VK_FALSE,
                                                        1.0f,
                                                        VK_FALSE,
                                                        vk::CompareOp::eLessOrEqual,
                                                        vk::BorderColor::eFloatOpaqueWhite
    );
    // Dedicated hardware-PCF comparison sampler — lives at its own binding.
    bindless.descriptorSet->allocateShadowCompareSampler(vk::Filter::eLinear,
                                                        vk::SamplerAddressMode::eClampToBorder,
                                                        vk::CompareOp::eLess,
                                                        vk::BorderColor::eFloatOpaqueWhite);
    // Default albedo (white)
    std::array<uint8_t, 4> whiteColor = {255, 255, 255, 255};
    auto [albedoImage, albedoMemory, albedoImageView] = bindless.resourceManager->createTexture(whiteColor.data(), 1, 1, vk::Format::eR8G8B8A8Srgb);
    uint32_t defaultAlbedoIndex = bindless.descriptorSet->allocateTexture(std::move(albedoImage), std::move(albedoMemory), std::move(albedoImageView));

    // Default normal (flat normal = 0.5, 0.5, 1.0 in RGB)
    std::array<uint8_t, 4> normalColor = {128, 128, 255, 255};
    auto [normalImage, normalMemory, normalImageView] = bindless.resourceManager->createTexture(normalColor.data(), 1, 1, vk::Format::eR8G8B8A8Unorm);
    defaultNormalIndex = bindless.descriptorSet->allocateTexture(std::move(normalImage), std::move(normalMemory), std::move(normalImageView));

    // Default roughness = 0.5
    std::array<uint8_t, 4> roughnessColor = {128, 128, 128, 255};
    auto [roughnessImage, roughnessMemory, roughnessImageView] = bindless.resourceManager->createTexture(roughnessColor.data(), 1, 1, vk::Format::eR8G8B8A8Unorm);
    uint32_t defaultRoughnessIndex = bindless.descriptorSet->allocateTexture(std::move(roughnessImage), std::move(roughnessMemory), std::move(roughnessImageView));

    // Default metallic = 0.0
    std::array<uint8_t, 4> metallicColor = {0, 0, 0, 255};
    auto [metallicImage, metallicMemory, metallicImageView] = bindless.resourceManager->createTexture(metallicColor.data(), 1, 1, vk::Format::eR8G8B8A8Unorm);
    uint32_t defaultMetallicIndex = bindless.descriptorSet->allocateTexture(std::move(metallicImage), std::move(metallicMemory), std::move(metallicImageView));

    /////S=================================================PIPELINES=================================================/////
    skyboxPipelineIndex =
        bindless.pipelineManager->createPipeline<SkyBoxPushConstants>(PipelineCategory::GEOMETRY, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone, vk::False,
                                                             vk::False, "shaders/skybox.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet());

    shadowPipelineIndex = bindless.pipelineManager->createPipeline<ShadowPushConstants>(PipelineCategory::SHADOW, vk::PrimitiveTopology::eTriangleList,
                                                                               vk::CullModeFlagBits::eNone, vk::True, vk::True, "shaders/shadow_geometry.spv",
                                                                               bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet());
    blurPipelineIndex =
        bindless.pipelineManager->createPipeline<BlurPushConstants>(PipelineCategory::POSTPROCESS, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone, vk::False,
                                                           vk::False, "shaders/blur.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet());

    depthPipelineIndex =
        bindless.pipelineManager->createPipeline<LitPushConstants>(PipelineCategory::DEPTH_PREPASS, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eBack,vk::True,
                                                                        vk::True,"shaders/depth_prepass.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet());

    litPipelineIndex = bindless.pipelineManager->createPipeline<LitPushConstants>(PipelineCategory::GEOMETRY, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eBack, vk::True,
                                                                         vk::True, "shaders/lit.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet());
    gizmoPipelineIndex =
        bindless.pipelineManager->createPipeline<LinePushConstants>(PipelineCategory::POSTPROCESS, vk::PrimitiveTopology::eLineList, vk::CullModeFlagBits::eNone, vk::False, vk::False,
                                                           "shaders/line.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet());
    imageViewPipelineIndex =
        bindless.pipelineManager->createPipeline<ImageVisPushConstants>(PipelineCategory::POSTPROCESS, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone, vk::False,
                                                               vk::False, "shaders/image_view.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet());

    // default litshader / material
    scene.fallbackLitShader = Shader{.sourceFile = "shaders/lit.spv", .pipelineIndex = litPipelineIndex};
    MaterialFlags defaultTexMask = MaterialFlags::MAT_NONE; // see the material struct definition
    // texMask |= (1U << 0);
    // texMask |= (1U << 1);
    // texMask |= (1U << 3);
    Material defaultMaterial = Material{.shaderSource = scene.fallbackLitShader,
                                        .flags = defaultTexMask,
                                        .color = glm::vec4(0.5, 0.5, 0.5, 1),
                                        .albedoTextureIndex = defaultAlbedoIndex,
                                        .metallic = 0.0,
                                        .metallicTextureIndex = defaultMetallicIndex,
                                        .roughness = 0.5,
                                        .roughnessTextureIndex = defaultRoughnessIndex,
                                        .normalTextureIndex = defaultNormalIndex};
    scene.fallbackDefaultMaterialIndex = scene.addMaterial(defaultMaterial);

#if DEBUG == 1
    bindless.descriptorSet->debugDescriptorSet("after_pipeline_creation");
#endif

    // create the root node - end of initialization
    scene.sceneGraph.init(this);
    bindless.descriptorSet->allocateFixedBuffer(litPassDataBufferIndex, LitPassData{.samplerIndex = defaultSamplerIndex,
                                                                           .lightCount = 0,
                                                                           .shadowSamplerIndex = shadowSamplerIndex,
                                                                           .shadowAtlasIndex = scene.shadowAtlas.textureIndex,
                                                                           .cameraPosition = scene.activeCamera.position,
                                                                           .cameraForward = glm::vec3(1, 0, 0),
                                                                           .viewProjection = scene.activeCamera.viewProjection,
                                                                           .prevViewProjection = scene.activeCamera.viewProjection});
}

/////=================================================DRAW FRAME=================================================/////

void Renderer::drawFrame() {
    Tracer::startTrace("draw frame");
    Tracer::startTrace("wait for fences");
    gpu.getDevice().getDevice().waitForFences(*gpu.getInFlightFence(gpu.currentFrame), vk::True, UINT64_MAX);
    Tracer::endTrace("wait for fences");
    //TODO make all full screen passes have a resolution scale that can be set dirty when changed/ needs to recreate
    if (features.ssr.resolutionDirty) {
        features.ssr.resolutionDirty = false;
        gpu.getDevice().getDevice().waitIdle();
        int w = 0, h = 0;
        glfwGetFramebufferSize(gpu.getWindow(), &w, &h);
        if (w > 0 && h > 0) {
            uint32_t ssrW = std::max(1u, static_cast<uint32_t>(w * features.ssr.resolutionScale));
            uint32_t ssrH = std::max(1u, static_cast<uint32_t>(h * features.ssr.resolutionScale));
            createSSRResources(ssrW, ssrH);
        }
    }

    Tracer::startTrace("acquire next image");
    auto [result, imageIndex] = gpu.getSwapchain().getSwapChain().acquireNextImage(UINT64_MAX, *gpu.getPresentCompleteSemaphore(gpu.currentFrame), nullptr);
    Tracer::endTrace("acquire next image");

    if (result == vk::Result::eErrorOutOfDateKHR) {
        gpu.recreateSwapchain();
        handleSwapchainResize();
        return;
    }
    if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
        throw std::runtime_error("failed to acquire swap chain image!");
    }
    Tracer::startTrace("wait image in flight");
    if (gpu.getImagesInFlight()[imageIndex] != VK_NULL_HANDLE) {
        vk::Result waitResult = gpu.getDevice().getDevice().waitForFences(gpu.getImagesInFlight()[imageIndex], vk::True, UINT64_MAX);
        if (waitResult != vk::Result::eSuccess) {
            throw std::runtime_error("failed to wait for image fence!");
        }
    }
    Tracer::endTrace("wait image in flight");

    Tracer::startTrace("reset fences");
    gpu.getImagesInFlight()[imageIndex] = *gpu.getInFlightFence(gpu.currentFrame);
    gpu.getDevice().getDevice().resetFences(*gpu.getInFlightFence(gpu.currentFrame));
    Tracer::endTrace("reset fences");

    for (auto& [id, light] : scene.lights) {
        glm::vec3 lightDir = scene.sceneGraph.getNode(light.nodeIndex).forward();
        glm::vec3 lightPos = scene.sceneGraph.getNode(light.nodeIndex).getWorldPosition();

        bool matricesUpdated = false;
        if (light.castsShadows == 1) {
            if (light.type == LightType::Directional) {
                // CSM depends on camera — always recalculate.
                calculateCascadedLightSpaceMatrices(light, scene.activeCamera, this);
                matricesUpdated = true;
            } else if (light.type == LightType::Point && light.shadowDirty) {
                calculatePointLightFaceMatrices(light, lightPos);
                matricesUpdated = true;
            }
        }
        if (matricesUpdated) light.gpuDirtyFrames = MAX_FRAMES_IN_FLIGHT;

        // Fan out the GPULight write across every frame-in-flight slice so the
        // per-frame buffer stays coherent instead of one slice winning the race.
        if (light.gpuDirtyFrames > 0) {
            bindless.descriptorSet->updateFixedBufferWithOffset<GPULight>(lightBufferIndex, id, light.toGPU(lightPos, lightDir), gpu.currentFrame);
            light.gpuDirtyFrames--;
        }
    }
    Tracer::startTrace("record command buffer");
    gpu.getCommandBuffer(gpu.currentFrame).reset();
    recordCommandBuffer(imageIndex);
    Tracer::endTrace("record command buffer");

    Tracer::startTrace("submit & present");
    vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
    const vk::SubmitInfo submitInfo{.waitSemaphoreCount = 1,
                                    .pWaitSemaphores = &*gpu.getPresentCompleteSemaphore(gpu.currentFrame),
                                    .pWaitDstStageMask = &waitDestinationStageMask,
                                    .commandBufferCount = 1,
                                    .pCommandBuffers = &*gpu.getCommandBuffer(gpu.currentFrame),
                                    .signalSemaphoreCount = 1,
                                    .pSignalSemaphores = &*gpu.getRenderFinishedSemaphore(imageIndex)};

    gpu.getDevice().getGraphicsQueue().submit(submitInfo, gpu.getInFlightFence(gpu.currentFrame));

    const vk::PresentInfoKHR presentInfoKHR{.waitSemaphoreCount = 1,
                                            .pWaitSemaphores = &*gpu.getRenderFinishedSemaphore(imageIndex),
                                            .swapchainCount = 1,
                                            .pSwapchains = &*gpu.getSwapchain().getSwapChain(),
                                            .pImageIndices = &imageIndex};

    try {
        result = gpu.getDevice().getPresentQueue().presentKHR(presentInfoKHR);
    } catch (const vk::OutOfDateKHRError&) {
        result = vk::Result::eErrorOutOfDateKHR;
    }

    if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || framebufferResized) {
        framebufferResized = false;
        gpu.recreateSwapchain();
        handleSwapchainResize();
    } else if (result != vk::Result::eSuccess) {
        throw std::runtime_error("failed to present swap chain image!");
    }
    Tracer::endTrace("submit & present");

    gpu.currentFrame = (gpu.currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    gpu.totalFrames++;
    //bindless.pipelineManager->checkForShaderUpdates(); // TODO enable this
    Tracer::endTrace("draw frame");
}

/////=================================================GET/SET=================================================/////

uint32_t Renderer::getModelMatrixBufferIndex() { return modelMatrixBufferIndex; }
uint32_t Renderer::getLightBufferIndex() { return lightBufferIndex; }
uint32_t Renderer::getVolumeBufferIndex() { return volumeBufferIndex; }
uint32_t Renderer::getShadowDrawDataBufferIndex() { return shadowDrawDataBufferIndex; }

void Renderer::addMeshToShader(uint32_t nodeIndex, Shader shader, Material material) {
    uint32_t matIdx = 0;
    for (uint32_t i = 0; i < scene.materials.size(); i++) {
        if (scene.materials[i] == material) { matIdx = i; break; }
    }
    for (const auto& e : renderEntries) {
        if (e.nodeIndex == nodeIndex &&
            e.materialIndex == matIdx && e.shaderPipelineIndex == shader.pipelineIndex)
            return;
    }
    renderEntries.push_back({nodeIndex, matIdx, shader.pipelineIndex});
    renderListDirty = true;
}
void Renderer::removeMeshFromShader(uint32_t nodeIndex, Shader shader, Material material) {
    uint32_t matIdx = 0;
    for (uint32_t i = 0; i < scene.materials.size(); i++) {
        if (scene.materials[i] == material) { matIdx = i; break; }
    }
    for (size_t i = 0; i < renderEntries.size(); ++i) {
        if (renderEntries[i].nodeIndex == nodeIndex &&
            renderEntries[i].materialIndex == matIdx && renderEntries[i].shaderPipelineIndex == shader.pipelineIndex) {
            renderEntries[i] = renderEntries.back();
            renderEntries.pop_back();
            renderListDirty = true;
            return;
        }
    }
}
void Renderer::removeNodeFromRenderList(uint32_t nodeIndex) {
    for (size_t i = 0; i < renderEntries.size();) {
        if (renderEntries[i].nodeIndex == nodeIndex) {
            renderEntries[i] = renderEntries.back();
            renderEntries.pop_back();
            renderListDirty = true;
        } else {
            ++i;
        }
    }
}
void Renderer::clearRenderList() { renderEntries.clear(); shaderDrawRanges.clear(); renderListDirty = false; }
void Renderer::clearLights() { scene.clearLights(bindless, lightBufferIndex); }
void Renderer::clearVolumes() { scene.clearVolumes(bindless, volumeBufferIndex); }

void Renderer::toggleVsync() {
    gpu.vSync = !gpu.vSync;
    gpu.recreateSwapchain();
    handleSwapchainResize();
}


void Renderer::handleSwapchainResize() {
    int width = 0, height = 0;
    glfwGetFramebufferSize(gpu.getWindow(), &width, &height);
    if (width > 0 && height > 0) {
        uint32_t ssaoW = std::max(1u, static_cast<uint32_t>(width * features.ssao.resolutionScale));
        uint32_t ssaoH = std::max(1u, static_cast<uint32_t>(height * features.ssao.resolutionScale));
        uint32_t ssrW = std::max(1u, static_cast<uint32_t>(width * features.ssr.resolutionScale));
        uint32_t ssrH = std::max(1u, static_cast<uint32_t>(height * features.ssr.resolutionScale));
        uint32_t volW = std::max(1u, static_cast<uint32_t>(width * features.volumetrics.resolutionScale));
        uint32_t volH = std::max(1u, static_cast<uint32_t>(height * features.volumetrics.resolutionScale));
        createSSAOResources(ssaoW, ssaoH);
        createRoughnessMetalResources(width, height);
        createNormalResources(width, height);
        createMotionVectorResources(width,height);
        createColorResolveResources(width, height);
        createSSRResources(ssrW, ssrH);
        createHiZResources(width, height);
        createSDFResources(width, height);
        createVolumetricResources(volW, volH);
    }
    scene.activeCamera.aspectRatio = static_cast<float>(width) / static_cast<float>(height);
    scene.activeCamera.calculateViewProjectionMatrix();
}

void Renderer::blurAttachment(vk::raii::CommandBuffer& cmd, uint32_t sourceTextureIndex, uint32_t tempTextureIndex, uint32_t width, uint32_t height, float blurRadius,
                    uint32_t samplerIndex) {

    auto& blurPipeline = *bindless.pipelineManager->getPostProcessPipelines()[blurPipelineIndex];
    auto& sourceTexture = bindless.descriptorSet->getTextureResource(sourceTextureIndex);
    auto& tempTexture = bindless.descriptorSet->getTextureResource(tempTextureIndex);
    vk::Extent2D extent{width, height};

    // Horizontal blur (source -> temp)
    bindless.resourceManager->transitionImageLayout(&cmd, *tempTexture.image, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eColorAttachmentOptimal);
    drawFullscreenPass(cmd, blurPipeline, *tempTexture.imageView, extent,
        BlurPushConstants{.inputTextureIndex = sourceTextureIndex, .samplerIndex = samplerIndex, .isHorizontal = 1, .blurRadius = blurRadius, .resolution = glm::uvec2(width, height)});
    bindless.resourceManager->transitionImageLayout(&cmd, *tempTexture.image, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

    // Vertical blur (temp -> source)
    bindless.resourceManager->transitionImageLayout(&cmd, *sourceTexture.image, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eColorAttachmentOptimal);
    drawFullscreenPass(cmd, blurPipeline, *sourceTexture.imageView, extent,
        BlurPushConstants{.inputTextureIndex = tempTextureIndex, .samplerIndex = samplerIndex, .isHorizontal = 0, .blurRadius = blurRadius, .resolution = glm::uvec2(width, height)});
    bindless.resourceManager->transitionImageLayout(&cmd, *sourceTexture.image, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
}

/////=================================================CREATE RESOURCES=================================================/////

void Renderer::createShadowAtlas(uint32_t resolution) {
    scene.shadowAtlas.init();
    vk::Format format = vk::Format::eD32Sfloat;
    vk::raii::Image image = nullptr;
    vk::raii::DeviceMemory memory = nullptr;
    bindless.resourceManager->createImage(resolution,resolution, 1, vk::SampleCountFlagBits::e1, format,
                                vk::ImageTiling::eOptimal,
                                vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
                                vk::MemoryPropertyFlagBits::eDeviceLocal, image, memory, 1);

    vk::raii::ImageView view = bindless.resourceManager->createImageView(image,format,vk::ImageAspectFlagBits::eDepth,1);

    scene.shadowAtlas.textureIndex = bindless.descriptorSet->allocateTexture(std::move(image),std::move(memory),std::move(view),"internal/scene.shadowAtlas",false,resolution,resolution);
}

void Renderer::createSSAOResources(uint32_t width, uint32_t height) {
    createOrResizeRenderTarget(ssaoTextureIndex, width, height, vk::Format::eR8Unorm, "internal/ssao");
    createOrResizeRenderTarget(ssaoBlurTextureIndex, width, height, vk::Format::eR8Unorm, "internal/ssao_blur");

    // 4x4 noise texture (RGBA8, random tangent-space rotation vectors)
    // Only create once — noise doesn't depend on screen size
    if (ssaoNoiseTextureIndex == 0xFFFFFFFF) {
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
        ssaoNoiseTextureIndex = bindless.descriptorSet->allocateTexture(std::move(noiseImage), std::move(noiseMemory), std::move(noiseImageView), "internal/ssao_noise");

        // Noise sampler: repeat + nearest (tiled across screen)
        ssaoNoiseSamplerIndex = bindless.descriptorSet->allocateSampler(vk::Filter::eNearest, vk::SamplerMipmapMode::eNearest, vk::SamplerAddressMode::eRepeat,
                                                                VK_FALSE, 1.0f, VK_FALSE, vk::CompareOp::eNever, vk::BorderColor::eFloatOpaqueBlack);
    }

    // SSAO pipeline (only create once)
    if (ssaoPipelineIndex == 0xFFFFFFFF) {
        ssaoPipelineIndex = bindless.pipelineManager->createPipeline<SSAOPushConstants>(
            PipelineCategory::POSTPROCESS, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone,
            vk::False, vk::False, "shaders/ssao.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet());
    }

    // SSAO apply pipeline with multiplicative blending (only create once)
    if (ssaoApplyPipelineIndex == 0xFFFFFFFF) {
        ssaoApplyPipelineIndex = bindless.pipelineManager->createPipeline<SSAOApplyPushConstants>(
            PipelineCategory::POSTPROCESS_MULTIPLY, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone,
            vk::False, vk::False, "shaders/ssao_apply.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet());
    }
}

void Renderer::createRoughnessMetalResources(uint32_t width, uint32_t height) {
    createOrResizeMSAATarget(roughnessMetal, width, height, vk::Format::eR8G8B8A8Unorm);
    createOrResizeRenderTarget(roughnessMetalTextureIndex, width, height, vk::Format::eR8G8B8A8Unorm, "internal/roughness_metal");
}

void Renderer::createNormalResources(uint32_t width, uint32_t height) {
    createOrResizeMSAATarget(normalMSAA, width, height, vk::Format::eR8G8B8A8Unorm);
    // Create with mip levels for SSR normal pre-filtering
    normalMipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1;
    if (normalTextureIndex != 0xFFFFFFFF) {
        bindless.descriptorSet->freeTexture(normalTextureIndex);
    }
    vk::raii::Image image = nullptr;
    vk::raii::DeviceMemory memory = nullptr;
    bindless.resourceManager->createImage(width, height, normalMipLevels, vk::SampleCountFlagBits::e1, vk::Format::eR8G8B8A8Unorm, vk::ImageTiling::eOptimal,
                                 vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled |
                                 vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst,
                                 vk::MemoryPropertyFlagBits::eDeviceLocal, image, memory, 1);
    auto view = bindless.resourceManager->createImageView(image, vk::Format::eR8G8B8A8Unorm, vk::ImageAspectFlagBits::eColor, normalMipLevels);
    bindless.resourceManager->transitionImageLayout(nullptr, image, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal, 0, normalMipLevels);
    normalTextureIndex = bindless.descriptorSet->allocateTexture(std::move(image), std::move(memory), std::move(view), "internal/normals", false, width, height);
}

void Renderer::createColorResolveResources(uint32_t width, uint32_t height) {
    colorResolveMipViews.clear();

    fullscreenMipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1;

    if (colorResolveTextureIndex != 0xFFFFFFFF) {
        bindless.descriptorSet->freeTexture(colorResolveTextureIndex);
    }

    auto format = gpu.getSwapchain().getSwapChainImageFormat();

    vk::raii::Image image = nullptr;
    vk::raii::DeviceMemory memory = nullptr;
    bindless.resourceManager->createImage(width, height, fullscreenMipLevels, vk::SampleCountFlagBits::e1, format, vk::ImageTiling::eOptimal,
                                 vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc,
                                 vk::MemoryPropertyFlagBits::eDeviceLocal, image, memory);

    // Create per-mip image views for rendering to individual levels
    for (uint32_t mip = 0; mip < fullscreenMipLevels; ++mip) {
        vk::ImageViewCreateInfo viewInfo{.image = image,
                                         .viewType = vk::ImageViewType::e2D,
                                         .format = format,
                                         .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = mip, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1}};
        colorResolveMipViews.emplace_back(gpu.getDevice().getDevice(), viewInfo);
    }

    // Create a full-chain view for sampling
    auto fullView = bindless.resourceManager->createImageView(image, format, vk::ImageAspectFlagBits::eColor, fullscreenMipLevels);
    bindless.resourceManager->transitionImageLayout(nullptr, image, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal, 0, fullscreenMipLevels);
    colorResolveTextureIndex = bindless.descriptorSet->allocateTexture(std::move(image), std::move(memory), std::move(fullView), "internal/color_resolve", false, width, height);

    // Temp texture for separable blur passes (mipmapped, matching color resolve)
    tempBlurMipViews.clear();

    if (tempBlurTextureIndex != 0xFFFFFFFF) {
        bindless.descriptorSet->freeTexture(tempBlurTextureIndex);
    }

    vk::raii::Image tempImage = nullptr;
    vk::raii::DeviceMemory tempMemory = nullptr;
    bindless.resourceManager->createImage(width, height, fullscreenMipLevels, vk::SampleCountFlagBits::e1, format, vk::ImageTiling::eOptimal,
                                 vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
                                 vk::MemoryPropertyFlagBits::eDeviceLocal, tempImage, tempMemory);

    for (uint32_t mip = 0; mip < fullscreenMipLevels; ++mip) {
        vk::ImageViewCreateInfo viewInfo{.image = tempImage,
                                         .viewType = vk::ImageViewType::e2D,
                                         .format = format,
                                         .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = mip, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1}};
        tempBlurMipViews.emplace_back(gpu.getDevice().getDevice(), viewInfo);
    }

    auto tempFullView = bindless.resourceManager->createImageView(tempImage, format, vk::ImageAspectFlagBits::eColor, fullscreenMipLevels);
    bindless.resourceManager->transitionImageLayout(nullptr, tempImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal, 0, fullscreenMipLevels);
    tempBlurTextureIndex = bindless.descriptorSet->allocateTexture(std::move(tempImage), std::move(tempMemory), std::move(tempFullView), "internal/blur_temp", false, width, height);
}

void Renderer::createMotionVectorResources(uint32_t width, uint32_t height) {
    createOrResizeMSAATarget(motionVectors,width,height, vk::Format::eR16G16Sfloat);
    createOrResizeRenderTarget(motionVectorTextureIndex, width, height, vk::Format::eR16G16Sfloat,"internal/motion_vectors");
}

void Renderer::createSSRResources(uint32_t width, uint32_t height) {

    createOrResizeRenderTarget(ssrCurrentTextureIndex, width, height, gpu.getSwapchain().getSwapChainImageFormat(), "internal/ssr_current");
    createOrResizeRenderTarget(ssrHistoryTextureIndices[0], width, height, gpu.getSwapchain().getSwapChainImageFormat(), "internal/ssr_history0");
    createOrResizeRenderTarget(ssrHistoryTextureIndices[1], width, height, gpu.getSwapchain().getSwapChainImageFormat(), "internal/ssr_history1");

    ssrHistoryInvalid = true;

    // SSR pipeline (only created once)
    if (ssrPipelineIndex == 0xFFFFFFFF) {
        ssrPipelineIndex = bindless.pipelineManager->createPipeline<SSRPushConstants>(
            PipelineCategory::POSTPROCESS, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone,
            vk::False, vk::False, "shaders/ssr.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet());

        //initialize pass data too
        bindless.descriptorSet->allocateFixedBuffer(ssrPassDataBufferIndex,SSRPassData {});
    }

    // SSR accumulate pipeline (only created once)
    if (ssrAccumulatePipelineIndex == 0xFFFFFFFF) {
        ssrAccumulatePipelineIndex = bindless.pipelineManager->createPipeline<SSRAccumulatePushConstants>(
            PipelineCategory::POSTPROCESS, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone,
            vk::False, vk::False, "shaders/ssr_accumulate.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet());
    }

    // SSR apply pipeline (only created once)
    if (ssrApplyPipelineIndex == 0xFFFFFFFF) {
        ssrApplyPipelineIndex = bindless.pipelineManager->createPipeline<SSRApplyPushConstants>(
            PipelineCategory::POSTPROCESS, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone,
            vk::False, vk::False, "shaders/ssr_apply.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet());
    }
}

void Renderer::createHiZResources(uint32_t width, uint32_t height) {
    hiZMipViews.clear();

    // Calculate mip levels for the Hi-Z pyramid
    hiZMipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1;

    // Free previous Hi-Z texture if it exists
    if (hiZTextureIndex != 0xFFFFFFFF) {
        bindless.descriptorSet->freeTexture(hiZTextureIndex);
    }

    // Create mipmapped R32Sfloat image
    vk::raii::Image image = nullptr;
    vk::raii::DeviceMemory memory = nullptr;
    bindless.resourceManager->createImage(width, height, hiZMipLevels, vk::SampleCountFlagBits::e1, vk::Format::eR32Sfloat, vk::ImageTiling::eOptimal,
                                 vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
                                 vk::MemoryPropertyFlagBits::eDeviceLocal, image, memory);

    // Create per-mip image views for rendering to individual levels
    for (uint32_t mip = 0; mip < hiZMipLevels; ++mip) {
        vk::ImageViewCreateInfo viewInfo{.image = image,
                                         .viewType = vk::ImageViewType::e2D,
                                         .format = vk::Format::eR32Sfloat,
                                         .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = mip, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1}};
        hiZMipViews.emplace_back(gpu.getDevice().getDevice(), viewInfo);
    }

    // Create a full-chain view for sampling
    auto fullView = bindless.resourceManager->createImageView(image, vk::Format::eR32Sfloat, vk::ImageAspectFlagBits::eColor, hiZMipLevels);
    bindless.resourceManager->transitionImageLayout(nullptr, image, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal, 0, hiZMipLevels);
    hiZTextureIndex = bindless.descriptorSet->allocateTexture(std::move(image), std::move(memory), std::move(fullView), "internal/hiZ", false, width, height);

    // Hi-Z pipeline (only created once)
    if (hiZPipelineIndex == 0xFFFFFFFF) {
        hiZPipelineIndex = bindless.pipelineManager->createPipeline<HiZPushConstants>(
            PipelineCategory::BEFORE_GEOMETRY, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone,
            vk::False, vk::False, "shaders/hiz_reduce.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet());
    }
}

void Renderer::createSDFResources(uint32_t width, uint32_t height) {
    createOrResizeRenderTarget(sdfTextureIndex,width,height,gpu.getSwapchain().getSwapChainImageFormat(),"internal/sdf");

    if (sdfPipelineIndex == 0xFFFFFFFF) {
        sdfPipelineIndex = 
        bindless.pipelineManager->createPipeline<SDFPushConstants>(PipelineCategory::POSTPROCESS, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone, vk::False,
                                                           vk::False, "shaders/sdf.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet());
    }
    // SDF apply pipeline
    if (sdfApplyPipelineIndex == 0xFFFFFFFF) {
        sdfApplyPipelineIndex = bindless.pipelineManager->createPipeline<SDFApplyPushConstants>(
            PipelineCategory::POSTPROCESS_ALPHA_BLEND, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone,
            vk::False, vk::False, "shaders/sdf_apply.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet());
    }
}

void Renderer::createVolumetricResources(uint32_t width, uint32_t height) {
    createOrResizeRenderTarget(volTextureIndex,width,height,gpu.getSwapchain().getSwapChainImageFormat(),"internal/volumetrics");
    createOrResizeRenderTarget(volBlurTextureIndex,width,height,gpu.getSwapchain().getSwapChainImageFormat(),"internal/volumetrics_blur");

    if (volPipelineIndex == 0xFFFFFFFF) {
        volPipelineIndex = 
        bindless.pipelineManager->createPipeline<VolumetricPushConstants>(PipelineCategory::POSTPROCESS, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone, vk::False,
                                                           vk::False, "shaders/volumetrics.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet());
    }
    // apply pipeline
    if (volApplyPipelineIndex == 0xFFFFFFFF) {
        volApplyPipelineIndex = bindless.pipelineManager->createPipeline<VolumetricApplyPushConstants>(
            PipelineCategory::POSTPROCESS_ALPHA_BLEND, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone,
            vk::False, vk::False, "shaders/volumetrics_apply.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet());
    }
}

/////=================================================RENDERING=================================================/////

void Renderer::recordCommandBuffer(uint32_t imageIndex) {
    auto& cmd = gpu.getCommandBuffer(gpu.currentFrame);
    cmd.begin({});

    Tracer::startTrace("record shadow pass");
    
    bindless.resourceManager->transitionImageLayout(&cmd, *bindless.descriptorSet->getTextureResource(scene.shadowAtlas.textureIndex).image, vk::ImageLayout::eDepthReadOnlyOptimal, vk::ImageLayout::eDepthStencilAttachmentOptimal);
    uint32_t shadowSlot = 0;
    for (auto& [lightId, light] : scene.lights) {
        if (light.castsShadows != 1) continue;
        // Skip point lights whose shadow maps are already up to date
        if (light.type == LightType::Point && !light.shadowDirty) continue;
        if (shadowSlot >= MAX_SHADOW_CASTERS) {
            std::cerr << "Warning: more than MAX_SHADOW_CASTERS shadow-casting lights in a frame; dropping extras" << std::endl;
            break;
        }
        recordShadowPass(cmd, light, shadowSlot);
        if (light.type == LightType::Point) light.shadowDirty = false;
        shadowSlot++;
    }
    bindless.resourceManager->transitionImageLayout(&cmd, *bindless.descriptorSet->getTextureResource(scene.shadowAtlas.textureIndex).image, vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::ImageLayout::eDepthReadOnlyOptimal);

    Tracer::endTrace("record shadow pass");

    Tracer::startTrace("record geo pass");
    recordGeometryPass(cmd, imageIndex);
    Tracer::endTrace("record geo pass");

    Tracer::startTrace("record ssr pass");
    if (features.ssr.enabled && ssrPipelineIndex != 0xFFFFFFFF)
        recordSSRPass(cmd, imageIndex);
    Tracer::endTrace("record ssr pass");

    Tracer::startTrace("record ssao pass");
    if (features.ssao.enabled && ssaoPipelineIndex != 0xFFFFFFFF)
        recordSSAOPass(cmd, imageIndex);
    Tracer::endTrace("record ssao pass");

    Tracer::startTrace("record Volumetric pass");
    if(features.volumetrics.enabled && volPipelineIndex != 0xFFFFFFFF) {
        recordVolumetricsPass(cmd, imageIndex);
    }

    Tracer::startTrace("record SDF pass");
    if(sdfPipelineIndex != 0xFFFFFFFF)
        recordSDFPass(cmd, imageIndex);
    Tracer::endTrace("record SDF pass");

    Tracer::startTrace("record image vis pass");
    if (features.imageVis.imageIndex != 0xFFFFFFFF)
        recordImageVisPass(cmd, imageIndex);
    Tracer::endTrace("record image vis pass");

    recordOverlayPass(cmd, imageIndex);

    bindless.resourceManager->transitionImageLayout(&cmd, gpu.getSwapchain().getSwapChainImages()[imageIndex], vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR);
    cmd.end();
}

template <typename PerMeshFn>
void Renderer::buildGeometryDrawCommands(const std::array<Plane, 6>& frustumPlanes, bool doCulling, PerMeshFn&& perMeshFn,
                                         const std::function<bool(const Node&)>& nodeFilter) {
    if (renderListDirty) {
        std::sort(renderEntries.begin(), renderEntries.end(), [](const RenderEntry& a, const RenderEntry& b) {
            return a.shaderPipelineIndex < b.shaderPipelineIndex;
        });
        renderListDirty = false;
    }

    indirectCommands.clear();
    shaderDrawRanges.clear();

    std::unordered_set<uint32_t> freedMeshes;
    uint32_t currentPipelineIdx = UINT32_MAX;

    for (const auto& entry : renderEntries) {
        Node* node = &scene.sceneGraph.getNode(entry.nodeIndex);
        uint32_t meshIdx = node->getMeshIndex();

        auto& mesh = scene.assetManager.meshes[meshIdx];

        if (mesh.freed) {
            if (freedMeshes.insert(meshIdx).second) {
                bindless.descriptorSet->freeVariableBuffer(vertexBufferIndex, mesh.vertexAllocationIndex);
                bindless.descriptorSet->freeVariableBuffer(indexBufferIndex, mesh.indexAllocationIndex);
                scene.assetManager.freeMeshes.push(meshIdx);
            }
            continue;
        }

        if (nodeFilter && !nodeFilter(*node)) continue;

        if (node->isBoundingBoxValid() && doCulling) {
            glm::vec3 worldMin, worldMax;
            transformAABBToWorldSpace(mesh.boundingBoxMin, mesh.boundingBoxMax, node->getTransform(), worldMin, worldMax);
            if (!isAABBInFrustum(worldMin, worldMax, frustumPlanes)) {
                culledCount++;
                continue;
            }

            if (features.showBBoxes)
                Gizmos::drawBox(worldMin, worldMax, glm::vec4(1.0f, 1.0f, 0.0f, 1.0f));
        }

        if (entry.shaderPipelineIndex != currentPipelineIdx) {
            currentPipelineIdx = entry.shaderPipelineIndex;
            shaderDrawRanges.push_back({currentPipelineIdx, static_cast<uint32_t>(indirectCommands.size()), 0});
        }

        indirectCommands.push_back({.indexCount    = mesh.indexCount,
                                    .instanceCount = 1,
                                    .firstIndex    = static_cast<uint32_t>(mesh.indexOffset / sizeof(uint32_t)),
                                    .vertexOffset  = 0,
                                    .firstInstance = 0});
        shaderDrawRanges.back().commandCount++;

        const Material& material = scene.materials[entry.materialIndex];
        perMeshFn(mesh, *node, material);
    }
}

void Renderer::recordHiZPass(vk::raii::CommandBuffer& cmd) {
    if (hiZTextureIndex == 0xFFFFFFFF || hiZPipelineIndex == 0xFFFFFFFF) return;

    auto& hiZRes = bindless.descriptorSet->getTextureResource(hiZTextureIndex);
    auto& pipeline = *bindless.pipelineManager->getBeforeGeoPipelines()[hiZPipelineIndex];
    uint32_t w = hiZRes.width;
    uint32_t h = hiZRes.height;

    // Transition depth resolve to shader read for sampling
    bindless.resourceManager->transitionImageLayout(&cmd, *bindless.descriptorSet->getTextureResource(gpu.getSwapchain().getDepthResolveIndex()).image,
        vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

    for (uint32_t mip = 0; mip < hiZMipLevels; ++mip) {
        uint32_t mipW = std::max(1u, w >> mip);
        uint32_t mipH = std::max(1u, h >> mip);

        // Transition this mip to color attachment
        bindless.resourceManager->transitionImageLayout(&cmd, *hiZRes.image,
            vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eColorAttachmentOptimal, mip, 1);

        HiZPushConstants hizPC;
        if (mip == 0) {
            // Mip 0: copy from depth resolve
            hizPC = {.inputTextureIndex = gpu.getSwapchain().getDepthResolveIndex(),
                     .samplerIndex = depthSamplerIndex,
                     .inputMipLevel = 0,
                     .reduceMode = 0,
                     .inputResolution = glm::uvec2(mipW, mipH)};
        } else {
            // Mip N: min-reduce from mip N-1 of the Hi-Z texture itself
            hizPC = {.inputTextureIndex = hiZTextureIndex,
                     .samplerIndex = depthSamplerIndex,
                     .inputMipLevel = mip - 1,
                     .reduceMode = 1,
                     .inputResolution = glm::uvec2(std::max(1u, w >> (mip - 1)), std::max(1u, h >> (mip - 1)))};
        }

        vk::Extent2D mipExtent{mipW, mipH};
        drawFullscreenPass(cmd, pipeline, *hiZMipViews[mip], mipExtent, hizPC);

        // Transition this mip back to shader read
        bindless.resourceManager->transitionImageLayout(&cmd, *hiZRes.image,
            vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, mip, 1);
    }
    // hiZ empty space calculation pass?

    // Transition depth resolve back to depth attachment
    bindless.resourceManager->transitionImageLayout(&cmd, *bindless.descriptorSet->getTextureResource(gpu.getSwapchain().getDepthResolveIndex()).image,
        vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eDepthStencilAttachmentOptimal);
}

void Renderer::recordGeometryPass(vk::raii::CommandBuffer& cmd, uint32_t imageIndex) {
    bindless.resourceManager->transitionImageLayouts(cmd, {
        {gpu.getSwapchain().getSwapChainImages()[imageIndex],                            vk::ImageLayout::eUndefined,              vk::ImageLayout::eColorAttachmentOptimal},
        {gpu.getSwapchain().getColorImage(),                                             vk::ImageLayout::eUndefined,              vk::ImageLayout::eColorAttachmentOptimal},
        {gpu.getSwapchain().getDepthImage(),                                             vk::ImageLayout::eUndefined,              vk::ImageLayout::eDepthStencilAttachmentOptimal},
        {roughnessMetal.image,                                                   vk::ImageLayout::eUndefined,              vk::ImageLayout::eColorAttachmentOptimal},
        {*bindless.descriptorSet->getTextureResource(roughnessMetalTextureIndex).image,   vk::ImageLayout::eShaderReadOnlyOptimal,  vk::ImageLayout::eColorAttachmentOptimal},
        {normalMSAA.image,                                                       vk::ImageLayout::eUndefined,              vk::ImageLayout::eColorAttachmentOptimal},
        {*bindless.descriptorSet->getTextureResource(normalTextureIndex).image,           vk::ImageLayout::eShaderReadOnlyOptimal,  vk::ImageLayout::eColorAttachmentOptimal},
        {motionVectors.image,                                                    vk::ImageLayout::eUndefined,              vk::ImageLayout::eColorAttachmentOptimal},
        {*bindless.descriptorSet->getTextureResource(motionVectorTextureIndex).image,     vk::ImageLayout::eShaderReadOnlyOptimal,  vk::ImageLayout::eColorAttachmentOptimal},
    });

    vk::ClearValue clearColor{.color = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f})};
    vk::ClearValue clearDepth{.depthStencil = vk::ClearDepthStencilValue{1.0f, 0}};

    // Frustum cull and build draw commands + lit draw data
    Camera fakeCam = scene.activeCamera;
    fakeCam.fov = cullFovScale * scene.activeCamera.fov;
    fakeCam.calculateViewProjectionMatrix();
    std::array<Plane, 6> frustumPlanes = extractFrustumPlanes(fakeCam.viewProjection);
    culledCount = 0;
    litDrawDataList.clear();
    buildGeometryDrawCommands(frustumPlanes, true, [&](const auto& mesh, auto& node, const auto& material) {
        litDrawDataList.push_back({.vertexAllocationIndex = mesh.vertexAllocationIndex,
                                   .vertexOffset          = static_cast<uint32_t>(mesh.vertexOffset),
                                   .vertexStride          = mesh.vertexStride,
                                   .modelMatrixIndex      = node.getModelMatrixIndex(),
                                   .albedoTextureIndex    = material.albedoTextureIndex,
                                   .roughnessTextureIndex = material.roughnessTextureIndex,
                                   .metallicTextureIndex  = material.metallicTextureIndex,
                                   .normalTextureIndex    = material.normalTextureIndex,
                                   .environmentMapIndex   = material.environmentMapIndex,
                                   .materialFlags         = static_cast<uint32_t>(material.flags),
                                   .metallic              = material.metallic,
                                   .roughness             = material.roughness,
                                   .alphaCutoff           = material.alphaCutoff});
    });

    vk::DeviceSize frameByteOffset = gpu.currentFrame * MAX_INDIRECT_COMMANDS * sizeof(DrawIndexedIndirectCommand);
    vk::Buffer indexBufferHandle = bindless.descriptorSet->getVariableBuffer(indexBufferIndex);

    glm::vec3 cameraForward = glm::normalize(scene.activeCamera.target - scene.activeCamera.position);

    LitPassData litPassData {
        .samplerIndex = defaultSamplerIndex,
        .lightCount = scene.getLightLoopBound(),
        .shadowSamplerIndex = shadowSamplerIndex,
        .shadowAtlasIndex = scene.shadowAtlas.textureIndex,
        .cameraPosition = scene.activeCamera.position,
        .cameraForward = cameraForward,
        .viewProjection = scene.activeCamera.viewProjection,
        .prevViewProjection = scene.activeCamera.prevViewProjection
    };
    bindless.descriptorSet->updateFixedBufferWithOffset<LitPassData>(litPassDataBufferIndex,0,litPassData,gpu.currentFrame);

    if (!indirectCommands.empty()) {
        // Upload indirect commands and per-draw data (shared between depth prepass and lit pass)
        memcpy(static_cast<char*>(litIndirectDrawBufferMapped) + frameByteOffset, indirectCommands.data(), indirectCommands.size() * sizeof(DrawIndexedIndirectCommand));

        auto* litDataPtr = bindless.descriptorSet->getFixedBufferMappedData<LitDrawData>(litDrawDataBufferIndex);
        if (litDataPtr) {
            uint32_t frameOffset = gpu.currentFrame * MAX_FIXED_BUFFER;
            memcpy(&litDataPtr[frameOffset], litDrawDataList.data(), litDrawDataList.size() * sizeof(LitDrawData));
        }

        LitPushConstants pushConstants = {.vertexBufferAddress  = bindless.descriptorSet->getVariableBuffers()[vertexBufferIndex]->address,
                                          .modelMatricesAddress = bindless.descriptorSet->getFixedBuffers()[modelMatrixBufferIndex]->address + static_cast<vk::DeviceSize>(gpu.currentFrame) * MAX_FIXED_BUFFER * sizeof(glm::mat4),
                                          .lightsAddress        = bindless.descriptorSet->getFixedBuffers()[lightBufferIndex]->address + static_cast<vk::DeviceSize>(gpu.currentFrame) * MAX_FIXED_BUFFER * sizeof(GPULight),
                                          .litDrawDataAddress   = bindless.descriptorSet->getFixedBuffers()[litDrawDataBufferIndex]->address + static_cast<vk::DeviceSize>(gpu.currentFrame) * MAX_FIXED_BUFFER * sizeof(LitDrawData),
                                          .litPassDataAddress   = bindless.descriptorSet->getFixedBuffers()[litPassDataBufferIndex]->address + static_cast<vk::DeviceSize>(gpu.currentFrame) * MAX_FIXED_BUFFER * sizeof(LitPassData)
                                        };

        // --- Depth prepass (depth-only, no color attachment) ---
        vk::RenderingAttachmentInfo depthPrepassAttachment = {.imageView = gpu.getSwapchain().getDepthImageView(),
                                                              .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                                              .resolveMode = vk::ResolveModeFlagBits::eMin,
                                                              .resolveImageView = gpu.getSwapchain().getDepthResolveImageView(),
                                                              .resolveImageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                                              .loadOp = vk::AttachmentLoadOp::eClear,
                                                              .storeOp = vk::AttachmentStoreOp::eStore,
                                                              .clearValue = clearDepth};

        auto& motionVectorResolve = bindless.descriptorSet->getTextureResource(motionVectorTextureIndex);
        vk::ClearValue clearMotionVectors{.color = vk::ClearColorValue(std::array<float, 4>{0.0f,0.0f,0.0f,1.0f})};
        vk::RenderingAttachmentInfo motionVectorAttachment = {  .imageView = motionVectors.view,
                                                                .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                                .resolveMode = vk::ResolveModeFlagBits::eAverage,
                                                                .resolveImageView = *motionVectorResolve.imageView,
                                                                .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                                .loadOp = vk::AttachmentLoadOp::eClear,
                                                                .storeOp = vk::AttachmentStoreOp::eStore,
                                                                .clearValue = clearMotionVectors};


        vk::RenderingInfo depthRenderingInfo = {.renderArea = {.offset = {0, 0}, .extent = gpu.getSwapchain().getSwapChainExtent()},
                                                .layerCount = 1,
                                                .colorAttachmentCount = 1,
                                                .pColorAttachments = &motionVectorAttachment,
                                                .pDepthAttachment = &depthPrepassAttachment};

        cmd.beginRendering(depthRenderingInfo);
        setFullscreenViewport(cmd, gpu.getSwapchain().getSwapChainExtent());

        auto& depthPipeline = bindless.pipelineManager->getBeforeGeoPipelines()[depthPipelineIndex];
        bindPipeline(cmd, *depthPipeline);
        cmd.bindIndexBuffer(indexBufferHandle, 0, vk::IndexType::eUint32);
        cmd.pushConstants<LitPushConstants>(depthPipeline->layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pushConstants);
        cmd.drawIndexedIndirect(*litIndirectDrawBuffer, frameByteOffset, static_cast<uint32_t>(indirectCommands.size()), sizeof(DrawIndexedIndirectCommand));

        cmd.endRendering();

        recordHiZPass(cmd);
    }

    // --- Lit geometry pass (2 color attachments: color + roughness/metallic) ---
    auto& colorResolve = bindless.descriptorSet->getTextureResource(colorResolveTextureIndex);
    bindless.resourceManager->transitionImageLayout(&cmd, *colorResolve.image,
        vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eColorAttachmentOptimal);

    vk::RenderingAttachmentInfo colorAttachment = {.imageView = gpu.getSwapchain().getColorImageView(),
                                                   .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                   .resolveMode = vk::ResolveModeFlagBits::eAverage,
                                                   .resolveImageView = *colorResolve.imageView,
                                                   .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                   .loadOp = vk::AttachmentLoadOp::eClear,
                                                   .storeOp = vk::AttachmentStoreOp::eStore,
                                                   .clearValue = clearColor};

    auto& roughnessMetalResolve = bindless.descriptorSet->getTextureResource(roughnessMetalTextureIndex);
    vk::ClearValue clearRoughMetal{.color = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f})};
    vk::RenderingAttachmentInfo roughnessMetalAttachment = {.imageView = *roughnessMetal.view,
                                                             .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                             .resolveMode = vk::ResolveModeFlagBits::eAverage,
                                                             .resolveImageView = *roughnessMetalResolve.imageView,
                                                             .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                             .loadOp = vk::AttachmentLoadOp::eClear,
                                                             .storeOp = vk::AttachmentStoreOp::eStore,
                                                             .clearValue = clearRoughMetal};

    auto& normalResolve = bindless.descriptorSet->getTextureResource(normalTextureIndex);
    vk::ClearValue clearNormal{.color = vk::ClearColorValue(std::array<float, 4>{0.5f, 0.5f, 1.0f, 1.0f})};
    vk::RenderingAttachmentInfo normalAttachment = {.imageView = *normalMSAA.view,
                                                     .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                     .resolveMode = vk::ResolveModeFlagBits::eAverage,
                                                     .resolveImageView = *normalResolve.imageView,
                                                     .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                     .loadOp = vk::AttachmentLoadOp::eClear,
                                                     .storeOp = vk::AttachmentStoreOp::eStore,
                                                     .clearValue = clearNormal};

    vk::RenderingAttachmentInfo depthAttachmentInfo = {.imageView = gpu.getSwapchain().getDepthImageView(),
                                                       .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                                       .resolveMode = vk::ResolveModeFlagBits::eMin,
                                                       .resolveImageView = gpu.getSwapchain().getDepthResolveImageView(),
                                                       .resolveImageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                                       .loadOp = vk::AttachmentLoadOp::eLoad,
                                                       .storeOp = vk::AttachmentStoreOp::eDontCare};

    std::array<vk::RenderingAttachmentInfo, 4> colorAttachments = {colorAttachment, roughnessMetalAttachment, normalAttachment};
    vk::RenderingInfo renderingInfo = {.renderArea = {.offset = {0, 0}, .extent = gpu.getSwapchain().getSwapChainExtent()},
                                       .layerCount = 1,
                                       .colorAttachmentCount = 4,
                                       .pColorAttachments = colorAttachments.data(),
                                       .pDepthAttachment = &depthAttachmentInfo};

    cmd.beginRendering(renderingInfo);
    setFullscreenViewport(cmd, gpu.getSwapchain().getSwapChainExtent());

    // skybox
    auto& skyboxPipeline = bindless.pipelineManager->getGeoPipelines()[skyboxPipelineIndex];
    bindPipeline(cmd, *skyboxPipeline);
    SkyBoxPushConstants skyboxConstants = {.skyboxIndex = scene.skyboxIndex, .blur = 0.5, .invViewProjMatrix = glm::inverse(scene.activeCamera.viewProjection)};
    cmd.pushConstants<SkyBoxPushConstants>(*skyboxPipeline->layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, skyboxConstants);
    cmd.draw(3, 1, 0, 0);

    // lit geometry — reuses the same indirect buffer from the prepass
    if (!indirectCommands.empty()) {
        cmd.bindIndexBuffer(indexBufferHandle, 0, vk::IndexType::eUint32);
        auto& geoPipelines = bindless.pipelineManager->getGeoPipelines();

        LitPushConstants pushConstants = {.vertexBufferAddress  = bindless.descriptorSet->getVariableBuffers()[vertexBufferIndex]->address,
                                          .modelMatricesAddress = bindless.descriptorSet->getFixedBuffers()[modelMatrixBufferIndex]->address + static_cast<vk::DeviceSize>(gpu.currentFrame) * MAX_FIXED_BUFFER * sizeof(glm::mat4),
                                          .lightsAddress        = bindless.descriptorSet->getFixedBuffers()[lightBufferIndex]->address + static_cast<vk::DeviceSize>(gpu.currentFrame) * MAX_FIXED_BUFFER * sizeof(GPULight),
                                          .litDrawDataAddress   = bindless.descriptorSet->getFixedBuffers()[litDrawDataBufferIndex]->address + static_cast<vk::DeviceSize>(gpu.currentFrame) * MAX_FIXED_BUFFER * sizeof(LitDrawData),
                                          .litPassDataAddress   = bindless.descriptorSet->getFixedBuffers()[litPassDataBufferIndex]->address + static_cast<vk::DeviceSize>(gpu.currentFrame) * MAX_FIXED_BUFFER * sizeof(LitPassData)
                                        };

        for (const auto& range : shaderDrawRanges) {
            auto currentPipeline = &(geoPipelines[range.pipelineIndex]);
            bindPipeline(cmd, **currentPipeline);
            cmd.pushConstants<LitPushConstants>((*currentPipeline)->layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pushConstants);
            vk::DeviceSize rangeOffset = frameByteOffset + range.firstCommand * sizeof(DrawIndexedIndirectCommand);
            cmd.drawIndexedIndirect(*litIndirectDrawBuffer, rangeOffset, range.commandCount, sizeof(DrawIndexedIndirectCommand));
        }
    }

    cmd.endRendering();

    // Copy color resolve to swapchain image
    bindless.resourceManager->transitionImageLayouts(cmd, {
        {*colorResolve.image,                         vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eTransferSrcOptimal},
        {gpu.getSwapchain().getSwapChainImages()[imageIndex],  vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eTransferDstOptimal},
    });

    vk::ImageCopy copyRegion{
        .srcSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .layerCount = 1},
        .dstSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .layerCount = 1},
        .extent = {gpu.getSwapchain().getSwapChainExtent().width, gpu.getSwapchain().getSwapChainExtent().height, 1}
    };
    cmd.copyImage(*colorResolve.image, vk::ImageLayout::eTransferSrcOptimal,
                  gpu.getSwapchain().getSwapChainImages()[imageIndex], vk::ImageLayout::eTransferDstOptimal,
                  copyRegion);

    // Transition back: color resolve to shader readable, swapchain to color attachment, roughness-metal to shader readable for SSR
    bindless.resourceManager->transitionImageLayouts(cmd, {
        {*colorResolve.image,                                                    vk::ImageLayout::eTransferSrcOptimal,    vk::ImageLayout::eShaderReadOnlyOptimal},
        {gpu.getSwapchain().getSwapChainImages()[imageIndex],                             vk::ImageLayout::eTransferDstOptimal,    vk::ImageLayout::eColorAttachmentOptimal},
        {*bindless.descriptorSet->getTextureResource(roughnessMetalTextureIndex).image,   vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal},
    });

    // Generate normal mips inline for SSR pre-filtering
    auto& normalRes = bindless.descriptorSet->getTextureResource(normalTextureIndex);
    bindless.resourceManager->generateMipmaps(*normalRes.image, vk::Format::eR8G8B8A8Unorm,
        static_cast<int32_t>(normalRes.width), static_cast<int32_t>(normalRes.height),
        normalMipLevels, 1, &cmd, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

    // Transition motion vecs to shader read only
    bindless.resourceManager->transitionImageLayout(&cmd, *bindless.descriptorSet->getTextureResource(motionVectorTextureIndex).image,
                                           vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
}

void Renderer::recordOverlayPass(vk::raii::CommandBuffer& cmd, uint32_t imageIndex) {
    auto extent = gpu.getSwapchain().getSwapChainExtent();
    vk::RenderingAttachmentInfo colorAttachment = {.imageView = gpu.getSwapchain().getSwapChainImageViews()[imageIndex],
                                                   .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                   .loadOp = vk::AttachmentLoadOp::eLoad,
                                                   .storeOp = vk::AttachmentStoreOp::eStore};
    vk::RenderingInfo renderInfo = {.renderArea = {.offset = {0, 0}, .extent = extent},
                                    .layerCount = 1,
                                    .colorAttachmentCount = 1,
                                    .pColorAttachments = &colorAttachment};

    cmd.beginRendering(renderInfo);
    setFullscreenViewport(cmd, extent);

    // gizmos
    if(features.showGizmos){
        for(auto& line : Gizmos::getNoDiscardLines()){
            Gizmos::drawLine(line.second);
        }
        auto& gizmoPipeline = bindless.pipelineManager->getPostProcessPipelines()[gizmoPipelineIndex];
        bindPipeline(cmd, *gizmoPipeline);
        LinePushConstants lineConstants = {.lineVertsAddress = Gizmos::getLineBufferAddress(),
                                        .depthTextureIndex = gpu.getSwapchain().getDepthResolveIndex(),
                                        .depthSamplerIndex = depthSamplerIndex,
                                        .viewProjection = scene.activeCamera.viewProjection};
        cmd.pushConstants<LinePushConstants>(*gizmoPipeline->layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, lineConstants);
        cmd.draw(Gizmos::getVertexCount(), 1, 0, 0);
    }
    // GUI
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), *cmd);
    cmd.endRendering();
}

void Renderer::recordSSAOPass(vk::raii::CommandBuffer& cmd, uint32_t imageIndex) {
    auto swapExtent = gpu.getSwapchain().getSwapChainExtent();
    auto& ssaoTexture = bindless.descriptorSet->getTextureResource(ssaoTextureIndex);
    vk::Extent2D ssaoExtent{ssaoTexture.width, ssaoTexture.height};

    // Transition depth to readable, SSAO target to color attachment
    bindless.resourceManager->transitionImageLayouts(cmd, {
        {gpu.getSwapchain().getDepthImage(),  vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal},
        {*ssaoTexture.image,          vk::ImageLayout::eShaderReadOnlyOptimal,          vk::ImageLayout::eColorAttachmentOptimal},
    });

    // Render SSAO at (potentially lower) SSAO resolution
    drawFullscreenPass(cmd, *bindless.pipelineManager->getPostProcessPipelines()[ssaoPipelineIndex], *ssaoTexture.imageView, ssaoExtent,
        SSAOPushConstants{.invProjection = glm::inverse(scene.activeCamera.projectionMatrix),
                          .depthIndex = gpu.getSwapchain().getDepthResolveIndex(),
                          .depthSamplerIndex = depthSamplerIndex,
                          .noiseIndex = ssaoNoiseTextureIndex,
                          .noiseSamplerIndex = ssaoNoiseSamplerIndex,
                          .resolution = glm::uvec2(ssaoExtent.width, ssaoExtent.height),
                          .radius = features.ssao.radius,
                          .bias = features.ssao.bias,
                          .power = features.ssao.power,
                          .kernelSize = 32},
        vk::AttachmentLoadOp::eClear, {1.0f, 1.0f, 1.0f, 1.0f});

    // Blur at SSAO resolution
    bindless.resourceManager->transitionImageLayout(&cmd, *ssaoTexture.image, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
    blurAttachment(cmd, ssaoTextureIndex, ssaoBlurTextureIndex, ssaoExtent.width, ssaoExtent.height, 2.0f, depthSamplerIndex);

    // Apply to swapchain at full resolution (sampler handles upscale)
    drawFullscreenPass(cmd, *bindless.pipelineManager->getPostProcessPipelines()[ssaoApplyPipelineIndex], *gpu.getSwapchain().getSwapChainImageViews()[imageIndex], swapExtent,
        SSAOApplyPushConstants{.ssaoTextureIndex = ssaoTextureIndex, .samplerIndex = depthSamplerIndex},
        vk::AttachmentLoadOp::eLoad);

    // Transition depth back
    bindless.resourceManager->transitionImageLayout(&cmd, gpu.getSwapchain().getDepthImage(), vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eDepthStencilAttachmentOptimal);
}

void Renderer::recordSSRPass(vk::raii::CommandBuffer& cmd, uint32_t imageIndex) {
    auto swapExtent = gpu.getSwapchain().getSwapChainExtent();
    auto& ssrCurrent = bindless.descriptorSet->getTextureResource(ssrCurrentTextureIndex);
    vk::Extent2D ssrExtent{ssrCurrent.width, ssrCurrent.height};

    uint32_t readHistory = ssrHistoryTextureIndices[ssrHistoryFlip];
    uint32_t writeHistory = ssrHistoryTextureIndices[1 - ssrHistoryFlip];
    auto& ssrWriteHist = bindless.descriptorSet->getTextureResource(writeHistory);

    // --- Sub-pass 0: Generate blurred mip chain for cone tracing (GPU Pro 5 style)
    // Mip 0 is already populated by the geometry pass MSAA resolve.
    // For each subsequent mip: 2-pass separable Gaussian blur reading mip N-1, writing to mip N.
    auto& colorRes = bindless.descriptorSet->getTextureResource(colorResolveTextureIndex);
    auto& tempTexture = bindless.descriptorSet->getTextureResource(tempBlurTextureIndex);
    auto& blurPipeline = *bindless.pipelineManager->getPostProcessPipelines()[blurPipelineIndex];

    uint32_t maxBlurMips = std::min(fullscreenMipLevels, 6u);
    for (uint32_t mip = 1; mip < fullscreenMipLevels; ++mip) {
        uint32_t mipW = std::max(1u, colorRes.width >> mip);
        uint32_t mipH = std::max(1u, colorRes.height >> mip);
        vk::Extent2D mipExtent{mipW, mipH};

        // Horizontal blur: read colorResolve mip N-1 -> write temp mip N
        bindless.resourceManager->transitionImageLayout(&cmd, *tempTexture.image,
            vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eColorAttachmentOptimal, mip, 1);

        drawFullscreenPass(cmd, blurPipeline, *tempBlurMipViews[mip], mipExtent,
            BlurPushConstants{.inputTextureIndex = colorResolveTextureIndex, .samplerIndex = defaultSamplerIndex,
                              .isHorizontal = 1, .blurRadius = 1.0f, .resolution = glm::uvec2(mipW, mipH),
                              .mipLevel = mip - 1});

        bindless.resourceManager->transitionImageLayout(&cmd, *tempTexture.image,
            vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, mip, 1);

        // Vertical blur: read temp mip N -> write colorResolve mip N
        bindless.resourceManager->transitionImageLayout(&cmd, *colorRes.image,
            vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eColorAttachmentOptimal, mip, 1);

        drawFullscreenPass(cmd, blurPipeline, *colorResolveMipViews[mip], mipExtent,
            BlurPushConstants{.inputTextureIndex = tempBlurTextureIndex, .samplerIndex = defaultSamplerIndex,
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

    bindless.descriptorSet->updateFixedBufferWithOffset(ssrPassDataBufferIndex, 0,
                                     SSRPassData{
                                        .invViewProj = glm::inverse(scene.activeCamera.viewProjection),
                                        .viewProj = scene.activeCamera.viewProjection,
                                        .cameraPos = scene.activeCamera.position,
                                        .depthIndex = hiZTextureIndex,
                                        .depthSamplerIndex = depthSamplerIndex,
                                        .colorIndex = colorResolveTextureIndex,
                                        .colorSamplerIndex = defaultSamplerIndex,
                                        .roughnessMetalIndex = roughnessMetalTextureIndex,
                                        .roughnessMetalSamplerIndex = defaultSamplerIndex,
                                        .normalIndex = normalTextureIndex,
                                        .normalSamplerIndex = defaultSamplerIndex,
                                        .resolution = glm::uvec2(ssrExtent.width, ssrExtent.height),
                                        .hiZIndex = hiZTextureIndex,
                                        .hiZMipLevels = hiZMipLevels,
                                        .thickness = features.ssr.thickness,
                                        .roughnessThreshold = features.ssr.roughnessThreshold,
                                        .maxSteps = features.ssr.maxSteps,
                                        .frameIndex = gpu.totalFrames,
                                     }, gpu.currentFrame);

    drawFullscreenPass(cmd, *bindless.pipelineManager->getPostProcessPipelines()[ssrPipelineIndex], *ssrCurrent.imageView, ssrExtent,
        SSRPushConstants{ .ssrPassDataAddress = bindless.descriptorSet->getFixedBuffers()[ssrPassDataBufferIndex]->address + static_cast<vk::DeviceSize>(gpu.currentFrame) * sizeof(SSRPassData)},
        vk::AttachmentLoadOp::eClear, {0.0f, 0.0f, 0.0f, 0.0f});

    // --- Sub-pass 2: Temporal accumulate -> writeHistory ---
    bindless.resourceManager->transitionImageLayouts(cmd, {
        {*ssrCurrent.image,    vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal},
        {*ssrWriteHist.image,  vk::ImageLayout::eShaderReadOnlyOptimal,  vk::ImageLayout::eColorAttachmentOptimal},
    });

    drawFullscreenPass(cmd, *bindless.pipelineManager->getPostProcessPipelines()[ssrAccumulatePipelineIndex], *ssrWriteHist.imageView, ssrExtent,
        SSRAccumulatePushConstants{
            .currentSSRIndex = ssrCurrentTextureIndex,
            .historySSRIndex = readHistory,
            .motionVectorIndex = motionVectorTextureIndex,
            .samplerIndex = defaultSamplerIndex,
            .temporalBlend = features.ssr.temporalBlend,
            .historyValid = ssrHistoryInvalid ? 0u : 1u,
        },
        vk::AttachmentLoadOp::eClear, {0.0f, 0.0f, 0.0f, 0.0f});

    bindless.resourceManager->transitionImageLayout(&cmd, *ssrWriteHist.image, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

    // --- Sub-pass 3: Apply accumulated SSR to swapchain ---
    drawFullscreenPass(cmd, *bindless.pipelineManager->getPostProcessPipelines()[ssrApplyPipelineIndex], *gpu.getSwapchain().getSwapChainImageViews()[imageIndex], swapExtent,
        SSRApplyPushConstants{
            .samplerIndex = defaultSamplerIndex,
            .sceneColorIndex = colorResolveTextureIndex,
            .sceneSamplerIndex = defaultSamplerIndex,
            .ssrTextureIndex = writeHistory,
        },
        vk::AttachmentLoadOp::eLoad);

    // transition depth back
    bindless.resourceManager->transitionImageLayout(&cmd, gpu.getSwapchain().getDepthImage(), vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eDepthStencilAttachmentOptimal);

    ssrHistoryFlip = 1 - ssrHistoryFlip;
    ssrHistoryInvalid = false;
}

void Renderer::recordImageVisPass(vk::raii::CommandBuffer& cmd, uint32_t imageIndex) {
    if (features.imageVis.imageIndex == 0xFFFFFFFF)
        return;

    auto extent = gpu.getSwapchain().getSwapChainExtent();
    auto& visTexture = bindless.descriptorSet->getTextureResource(features.imageVis.imageIndex);
    float imgAspect = (visTexture.width > 0 && visTexture.height > 0)
                          ? static_cast<float>(visTexture.width) / static_cast<float>(visTexture.height)
                          : static_cast<float>(extent.width) / static_cast<float>(extent.height);

    drawFullscreenPass(cmd, *bindless.pipelineManager->getPostProcessPipelines()[imageViewPipelineIndex], *gpu.getSwapchain().getSwapChainImageViews()[imageIndex],
                       extent,
                       ImageVisPushConstants{.imageIndex = features.imageVis.imageIndex,
                                             .samplerIndex = defaultSamplerIndex,
                                             .flags = features.imageVis.flags,
                                             .nearPlane = scene.activeCamera.nearPlane,
                                             .farPlane = scene.activeCamera.farPlane,
                                             .imageAspect = imgAspect,
                                             .screenAspect = static_cast<float>(extent.width) / static_cast<float>(extent.height),
                                             .mipLevel = features.imageVis.mipLevel},
                       vk::AttachmentLoadOp::eLoad);
}

void Renderer::recordShadowPass(vk::raii::CommandBuffer& cmd, Light& light, uint32_t shadowSlot) {
    auto& currentPipeline = bindless.pipelineManager->getBeforeGeoPipelines()[shadowPipelineIndex];
    vk::Buffer indexBufferHandle = bindless.descriptorSet->getVariableBuffer(indexBufferIndex);
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, currentPipeline->pipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, currentPipeline->layout, 0, {*currentPipeline->descriptorSet}, {});

    // Build indirect draw commands once — identical across all faces/cascades since culling is off
    std::array<Plane, 6> dummyPlanes{};
    drawDataList.clear();
    // Point lights only need to draw casters inside their range — LightInfluence
    // keeps that set current, so we filter by it here. Directional lights skip
    // the filter (they affect all geometry).
    std::function<bool(const Node&)> nodeFilter;
    if (light.type == LightType::Point) {
        nodeFilter = [&light](const Node& n) { return light.influencedNodes.count(n.nodeIndex) != 0; };
    }
    buildGeometryDrawCommands(dummyPlanes, false, [&](const auto& mesh, auto& node, const auto& material) {
        drawDataList.push_back({.vertexAllocationIndex = mesh.vertexAllocationIndex,
                                .vertexOffset = static_cast<uint32_t>(mesh.vertexOffset),
                                .vertexStride = mesh.vertexStride,
                                .modelMatrixIndex = node.getModelMatrixIndex()});
    }, nodeFilter);

    // Per-light slot within the frame so multiple shadow-casting lights don't stomp on each
    // other's indirect commands / draw data before the GPU reads them.
    uint32_t slotIdx = gpu.currentFrame * MAX_SHADOW_CASTERS + shadowSlot;
    vk::DeviceSize frameByteOffset = static_cast<vk::DeviceSize>(slotIdx) * MAX_INDIRECT_COMMANDS * sizeof(DrawIndexedIndirectCommand);

    // Upload indirect commands and draw data once
    if (!indirectCommands.empty()) {
        memcpy(static_cast<char*>(indirectDrawBufferMapped) + frameByteOffset, indirectCommands.data(), indirectCommands.size() * sizeof(DrawIndexedIndirectCommand));

        auto* shadowDataBuffer = bindless.descriptorSet->getFixedBufferMappedData<ShadowDrawData>(shadowDrawDataBufferIndex);
        if (shadowDataBuffer) {
            uint32_t drawDataOffset = slotIdx * MAX_FIXED_BUFFER;
            memcpy(&shadowDataBuffer[drawDataOffset], drawDataList.data(), drawDataList.size() * sizeof(ShadowDrawData));
        } else {
            std::cerr << "Error: shadow data buffer mapped memory is null!" << std::endl;
        }
    }

    // Determine face count
    uint32_t faceCount = 0;
    if (light.type == LightType::Directional) {
        faceCount = light.numCascades;
    } else if (light.type == LightType::Point) {
        faceCount = 6;
    }

    // Bind the atlas once; each face/cascade is rendered into its tile via viewport+scissor.
    auto& atlasTex = bindless.descriptorSet->getTextureResource(scene.shadowAtlas.textureIndex);
    vk::RenderingAttachmentInfo depthAttachment{.imageView   = *atlasTex.imageView,
                                                .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                                .loadOp      = vk::AttachmentLoadOp::eLoad,
                                                .storeOp     = vk::AttachmentStoreOp::eStore};
    vk::RenderingInfo renderInfo{.renderArea           = {{0, 0}, {SHADOW_ATLAS_SIZE, SHADOW_ATLAS_SIZE}},
                                 .layerCount           = 1,
                                 .colorAttachmentCount = 0,
                                 .pColorAttachments    = nullptr,
                                 .pDepthAttachment     = &depthAttachment};
    cmd.beginRendering(renderInfo);

    for (uint32_t i = 0; i < faceCount; i++) {
        glm::vec4 uvRange;
        glm::mat4 lightSpaceMatrix;
        if (light.type == LightType::Directional) {
            uvRange          = light.cascades[i].shadowAtlasUVRange;
            lightSpaceMatrix = light.cascades[i].lightSpaceMatrix;
        } else {
            uvRange          = light.cubeMapIndices[i].shadowAtlasUVRange;
            lightSpaceMatrix = light.cubeMapIndices[i].lightSpaceMatrix;
        }

        int32_t  tx = static_cast<int32_t>(uvRange.x * SHADOW_ATLAS_SIZE);
        int32_t  ty = static_cast<int32_t>(uvRange.y * SHADOW_ATLAS_SIZE);
        uint32_t tw = static_cast<uint32_t>((uvRange.z - uvRange.x) * SHADOW_ATLAS_SIZE);
        uint32_t th = static_cast<uint32_t>((uvRange.w - uvRange.y) * SHADOW_ATLAS_SIZE);
        if (tw == 0 || th == 0) continue;

        vk::Rect2D tileRect{{tx, ty}, {tw, th}};

        // Clear just this tile (the atlas was loaded, not cleared).
        vk::ClearAttachment clearInfo{.aspectMask = vk::ImageAspectFlagBits::eDepth,
                                      .colorAttachment = 0,
                                      .clearValue = vk::ClearValue{.depthStencil = vk::ClearDepthStencilValue{1.0f, 0}}};
        vk::ClearRect clearRect{.rect = tileRect, .baseArrayLayer = 0, .layerCount = 1};
        cmd.clearAttachments(clearInfo, clearRect);

        cmd.setViewport(0, vk::Viewport(static_cast<float>(tx), static_cast<float>(ty),
                                        static_cast<float>(tw), static_cast<float>(th), 0.0f, 1.0f));
        cmd.setScissor(0, tileRect);

        if (!indirectCommands.empty()) {
            ShadowPushConstants pushConstants = {
                .vertexBufferAddress   = bindless.descriptorSet->getVariableBuffers()[vertexBufferIndex]->address,
                .modelMatricesAddress  = bindless.descriptorSet->getFixedBuffers()[modelMatrixBufferIndex]->address + static_cast<vk::DeviceSize>(gpu.currentFrame) * MAX_FIXED_BUFFER * sizeof(glm::mat4),
                .shadowDrawDataAddress = bindless.descriptorSet->getFixedBuffers()[shadowDrawDataBufferIndex]->address + static_cast<vk::DeviceSize>(slotIdx) * MAX_FIXED_BUFFER * sizeof(ShadowDrawData),
                .lightSpaceMatrix      = lightSpaceMatrix};
            cmd.pushConstants<ShadowPushConstants>(*currentPipeline->layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pushConstants);

            cmd.bindIndexBuffer(indexBufferHandle, 0, vk::IndexType::eUint32);
            cmd.drawIndexedIndirect(*indirectDrawBuffer, frameByteOffset, static_cast<uint32_t>(indirectCommands.size()), sizeof(DrawIndexedIndirectCommand));
        }
    }

    cmd.endRendering();
}

void Renderer::recordVolumetricsPass(vk::raii::CommandBuffer& cmd, uint32_t imageIndex) {
    if (volTextureIndex == 0xFFFFFFFF)
        return;

    auto& volRes = bindless.descriptorSet->getTextureResource(volTextureIndex);
    vk::Extent2D extent = {volRes.width, volRes.height};

    drawFullscreenPass(cmd, *bindless.pipelineManager->getPostProcessPipelines()[volPipelineIndex], *bindless.descriptorSet->getTextureResource(volTextureIndex).imageView,
                       extent,
                       VolumetricPushConstants {
                            .lightsAddress = bindless.descriptorSet->getFixedBuffers()[lightBufferIndex]->address + static_cast<vk::DeviceSize>(gpu.currentFrame) * MAX_FIXED_BUFFER * sizeof(GPULight),
                            .volumeBufferAddress = bindless.descriptorSet->getFixedBuffers()[volumeBufferIndex]->address /* + static_cast<vk::DeviceSize>(gpu.currentFrame) * MAX_FIXED_BUFFER * sizeof(Volume)*/,
                            .lightCount = scene.getLightLoopBound(),
                            .shadowAtlasIndex = scene.shadowAtlas.textureIndex,
                            .depthTextureIndex = gpu.getSwapchain().getDepthResolveIndex(),
                            .depthSamplerIndex = depthSamplerIndex,
                            .cameraPos = scene.activeCamera.position,
                            .numSteps = static_cast<uint32_t>(features.volumetrics.numSteps),
                            .cameraDir = scene.activeCamera.getLookDir(),
                            .volumeCount = scene.getVolumeLoopBound(),
                            .screenSize = {extent.width, extent.height},
                            .maxDist = features.volumetrics.maxDist,
                            .invViewProjection = glm::inverse(scene.activeCamera.viewProjection)
                       }, vk::AttachmentLoadOp::eClear);

    //TODO make blur controllable
    blurAttachment(cmd,volTextureIndex,volBlurTextureIndex,extent.width,extent.height,2.0f,defaultSamplerIndex);

    drawFullscreenPass(cmd, *bindless.pipelineManager->getPostProcessPipelines()[volApplyPipelineIndex], *gpu.getSwapchain().getSwapChainImageViews()[imageIndex],
                       gpu.getSwapchain().getSwapChainExtent(),
                       VolumetricApplyPushConstants {
                            .volumetricTextureIndex = volTextureIndex,
                            .samplerIndex = defaultSamplerIndex
                       }, vk::AttachmentLoadOp::eLoad);
                    
}

void Renderer::recordSDFPass(vk::raii::CommandBuffer& cmd, uint32_t imageIndex) {
    if (sdfTextureIndex == 0xFFFFFFFF)  
        return;

    auto extent = gpu.getSwapchain().getSwapChainExtent();

    auto& sdfAllocations = bindless.descriptorSet->getFixedBufferAllocations(sdfPassDataBufferIndex);
    uint32_t sdfCount = 0;
    for (const auto& alloc : sdfAllocations) {
        if (alloc.inUse) sdfCount++;
    }

    if (sdfCount == 0) return;

    drawFullscreenPass(cmd,*bindless.pipelineManager->getPostProcessPipelines()[sdfPipelineIndex], *bindless.descriptorSet->getTextureResource(sdfTextureIndex).imageView,
                       extent,
                       SDFPushConstants{.sdfDataAddress = bindless.descriptorSet->getFixedBuffers()[sdfPassDataBufferIndex]->address,
                                        .sdfCount = sdfCount,
                                        .depthTextureIndex = gpu.getSwapchain().getDepthResolveIndex(),
                                        .depthSamplerIndex = depthSamplerIndex,
                                        .cameraPos = scene.activeCamera.position,
                                        .invViewProjection = glm::inverse(scene.activeCamera.viewProjection)
                                        }, vk::AttachmentLoadOp::eClear);

    drawFullscreenPass(cmd,*bindless.pipelineManager->getPostProcessPipelines()[sdfApplyPipelineIndex], *gpu.getSwapchain().getSwapChainImageViews()[imageIndex],
                       extent,
                       SDFApplyPushConstants{.sdfTextureIndex = sdfTextureIndex,
                                             .samplerIndex = defaultSamplerIndex},
                                            vk::AttachmentLoadOp::eLoad);
}

void Renderer::createOrResizeRenderTarget(uint32_t& index, uint32_t width, uint32_t height, vk::Format format, const char* debugName,
                                          vk::ImageUsageFlags extraUsage) {
    if (index != 0xFFFFFFFF) {
        bindless.descriptorSet->freeTexture(index);
    }
    vk::raii::Image image = nullptr;
    vk::raii::DeviceMemory memory = nullptr;
    bindless.resourceManager->createImage(width, height, 1, vk::SampleCountFlagBits::e1, format, vk::ImageTiling::eOptimal,
                                 vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled | extraUsage,
                                 vk::MemoryPropertyFlagBits::eDeviceLocal, image, memory, 1);
    auto view = bindless.resourceManager->createImageView(image, format, vk::ImageAspectFlagBits::eColor);
    bindless.resourceManager->transitionImageLayout(nullptr, image, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal);
    index = bindless.descriptorSet->allocateTexture(std::move(image), std::move(memory), std::move(view), debugName, false, width, height);
}

void Renderer::createOrResizeMSAATarget(Image& target, uint32_t width, uint32_t height, vk::Format format) {
    target.view = nullptr; // destroy view before image
    bindless.resourceManager->createImage(width, height, 1, gpu.getMsaaSamples(),format, vk::ImageTiling::eOptimal,
                                 vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment,
                                 vk::MemoryPropertyFlagBits::eDeviceLocal, target.image, target.memory);
    target.view = bindless.resourceManager->createImageView(target.image, format, vk::ImageAspectFlagBits::eColor, 1);
    bindless.resourceManager->transitionImageLayout(nullptr, target.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal);
}

void Renderer::setFullscreenViewport(vk::raii::CommandBuffer& cmd, vk::Extent2D extent) {
    cmd.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.0f, 1.0f));
    cmd.setScissor(0, vk::Rect2D({0, 0}, extent));
}

void Renderer::bindPipeline(vk::raii::CommandBuffer& cmd, PipelineBase& pipeline) {
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.pipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline.layout, 0, {**pipeline.descriptorSet}, {});
}

template <typename T>
void Renderer::drawFullscreenPass(vk::raii::CommandBuffer& cmd, PipelineBase& pipeline, vk::ImageView targetView, vk::Extent2D extent,
                                  const T& pushConstants, vk::AttachmentLoadOp loadOp, std::array<float, 4> clearColor) {

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


glm::mat4 calculateLightSpaceMatrix(Light& light, Camera& camera) {

    glm::mat4 lightProjection = glm::ortho(-light.range, light.range, -light.range, light.range, camera.nearPlane, camera.farPlane);
    glm::vec3 lightPos = camera.position - light.direction * 0.5f * camera.farPlane;
    glm::mat4 lightView = glm::lookAt(lightPos, lightPos + light.direction, glm::vec3(0.0f, 1.0f, 0.0f));
    return lightProjection * lightView;
}

void calculatePointLightFaceMatrices(Light& light, const glm::vec3& lightPos) {
    float nearPlane = 0.1f;
    float farPlane = light.range;
    glm::mat4 projection = glm::perspective(glm::radians(90.0f), 1.0f, nearPlane, farPlane);

    // 6 cubemap faces: +X, -X, +Y, -Y, +Z, -Z
    const glm::vec3 directions[6] = {
        { 1.0f,  0.0f,  0.0f},  // +X
        {-1.0f,  0.0f,  0.0f},  // -X
        { 0.0f,  1.0f,  0.0f},  // +Y
        { 0.0f, -1.0f,  0.0f},  // -Y
        { 0.0f,  0.0f,  1.0f},  // +Z
        { 0.0f,  0.0f, -1.0f},  // -Z
    };
    const glm::vec3 ups[6] = {
        { 0.0f, -1.0f,  0.0f},  // +X
        { 0.0f, -1.0f,  0.0f},  // -X
        { 0.0f,  0.0f,  1.0f},  // +Y
        { 0.0f,  0.0f, -1.0f},  // -Y
        { 0.0f, -1.0f,  0.0f},  // +Z
        { 0.0f, -1.0f,  0.0f},  // -Z
    };

    for (int i = 0; i < 6; i++) {
        glm::mat4 view = glm::lookAt(lightPos, lightPos + directions[i], ups[i]);
        light.cubeMapIndices[i].lightSpaceMatrix = projection * view;
    }
}

void calculateCascadedLightSpaceMatrices(Light& light, Camera& camera, Renderer* renderer) {
    glm::vec3 lightDir = glm::normalize(light.direction);

    // Stable up vector that avoids degeneracy when light is near-vertical
    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
    if (glm::abs(glm::dot(lightDir, up)) > 0.99f) {
        up = glm::vec3(1.0f, 0.0f, 0.0f);
    }
    // Unproject the full camera frustum to world space (Vulkan NDC: z in [0,1])
    // z-outermost so indices 0-3 = near plane, 4-7 = far plane
    glm::mat4 invCamVP = glm::inverse(camera.viewProjection);
    glm::vec3 fullCorners[8];
    int idx = 0;
    for (int z = 0; z < 2; ++z)
        for (int y = 0; y < 2; ++y)
            for (int x = 0; x < 2; ++x) {
                glm::vec4 c = invCamVP * glm::vec4(2.f * x - 1.f, 2.f * y - 1.f, static_cast<float>(z), 1.f);
                fullCorners[idx++] = glm::vec3(c / c.w);
            }

    float lastSplitDist = 0.0f;

    for (uint32_t i = 0; i < light.numCascades; i++) {

        // Cascade splits are in world-space units
        float splitDist;
        if (light.cascades[i].splitDistance > 0.0f) {
            splitDist = (light.cascades[i].splitDistance - camera.nearPlane) / (camera.farPlane - camera.nearPlane);
            splitDist = glm::clamp(splitDist, lastSplitDist + 0.001f, 1.0f);
        } else {
            splitDist = static_cast<float>(i + 1) / static_cast<float>(light.numCascades);
            light.cascades[i].splitDistance = camera.nearPlane + splitDist * (camera.farPlane - camera.nearPlane);
        }

        // Slice the full frustum into this cascade's sub-frustum
        glm::vec3 corners[8];
        for (int j = 0; j < 4; j++) {
            glm::vec3 ray = fullCorners[j + 4] - fullCorners[j];
            corners[j]     = fullCorners[j] + ray * lastSplitDist;
            corners[j + 4] = fullCorners[j] + ray * splitDist;
        }

        // Sub-frustum center
        glm::vec3 center(0.0f);
        for (const auto& c : corners) center += c;
        center /= 8.0f;

        // Build light view matrix looking at the frustum center
        float zPullBack = 500.0f;
        glm::mat4 lightView = glm::lookAt(
            center - lightDir * zPullBack,
            center,
            up
        );

        // Compute tight AABB in light space from the frustum corners
        glm::vec3 lsMin(std::numeric_limits<float>::max());
        glm::vec3 lsMax(std::numeric_limits<float>::lowest());
        for (const auto& c : corners) {
            glm::vec3 ls = glm::vec3(lightView * glm::vec4(c, 1.0f));
            lsMin = glm::min(lsMin, ls);
            lsMax = glm::max(lsMax, ls);
        }

        float extentX = lsMax.x - lsMin.x;
        float extentY = lsMax.y - lsMin.y;
        float maxExtent = glm::max(extentX, extentY);

        // Expand AABB for cascade overlap
        float overlapMargin = maxExtent * 0.1f;
        lsMin.x -= overlapMargin;
        lsMin.y -= overlapMargin;
        lsMax.x += overlapMargin;
        lsMax.y += overlapMargin;

        // Near=0.1 captures shadow casters between the light eye and the frustum.
        // Far extends just past the farthest frustum corner in light space.
        float orthoNear = 0.1f;
        float orthoFar  = -lsMin.z + 10.0f;

        glm::mat4 lightProj = glm::ortho(
            lsMin.x, lsMax.x,
            lsMin.y, lsMax.y,
            orthoNear, orthoFar
        );

        // Snap the shadow matrix translation to texel boundaries so the
        // shadow map stays locked to a fixed world-space grid as the camera moves.
        glm::mat4 shadowMatrix = lightProj * lightView;
        float halfRes = static_cast<float>(light.shadowResolution) * 0.5f;
        glm::vec4 shadowOrigin = shadowMatrix * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        shadowOrigin *= halfRes;
        glm::vec4 rounded = glm::round(shadowOrigin);
        glm::vec4 roundOffset = rounded - shadowOrigin;
        roundOffset /= halfRes;
        lightProj[3][0] += roundOffset.x;
        lightProj[3][1] += roundOffset.y;

        float finalExtent = glm::max(lsMax.x - lsMin.x, lsMax.y - lsMin.y);
        light.cascades[i].lightSpaceMatrix = lightProj * lightView;
        light.cascades[i].texelSize = 1.0f / static_cast<float>(light.shadowResolution);
        light.cascades[i].worldTexelSize = finalExtent / static_cast<float>(light.shadowResolution);

        lastSplitDist = splitDist;
    }
}