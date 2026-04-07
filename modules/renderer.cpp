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
 
#include"imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#include "gizmo.hpp"
#include "node_ops.hpp"
#include "pipelines.hpp"
#include "profiling.hpp"
#include "swapchain.hpp"

// TODO gpu side material data
// TODO clustered lights? (forward +)
//      pass the N nearest lights to the lit shader
// TODO point, spot and area lights
// TODO node deletion
// TODO other stuff idk
static const std::vector validationLayers = {"VK_LAYER_KHRONOS_validation"};

Renderer::Renderer() = default;
Renderer::~Renderer() = default;

/////=================================================INIT=================================================/////

void Renderer::initVulkan(uint32_t startWidth, uint32_t startHeight) {
    createInstance();
#if DEBUG == 1
    setupDebugMessenger();
#endif
    createSurface();
    device = std::make_unique<Device>(instance, requiredDeviceExtension, surface);
    msaaSamples = getMaxUsableSampleCount(*device);
    createCommandPool();
    createCommandBuffers();
    resourceManager = std::make_unique<ResourceManager>(*device, commandPool);
    descriptorSet = std::make_unique<DescriptorSet>(*device, *resourceManager, &commandPool);
    swapchain = std::make_unique<Swapchain>(*device, *resourceManager, *descriptorSet, surface, msaaSamples);
    pipelineManager = std::make_unique<PipelineManager>(*device, *swapchain, msaaSamples);

    // initializing default camera
    activeCamera = Camera{.position = glm::vec3(1, 1, 1),
                          .target = glm::vec3(0, 0, 0),
                          .fov = 45.0,
                          .aspectRatio = static_cast<float>(startWidth) / static_cast<float>(startHeight),
                          .nearPlane = 0.1,
                          .farPlane = 500.0};
    activeCamera.calculateViewProjectionMatrix();

    /////=====================================DESCRIPTOR SET BUFFERS=================================================/////
    vertexBufferIndex = descriptorSet->createVariableBuffer(256 * 1024 * 1024);                                       // 256 mb vertex buffer
    indexBufferIndex = descriptorSet->createVariableBuffer(128 * 1024 * 1024, vk::BufferUsageFlagBits::eIndexBuffer); // index buffer (128 MB)
    assetManager.init(resourceManager.get(), descriptorSet.get(), vertexBufferIndex, indexBufferIndex);

    // these buffers store the data once per frame in flight since they are usually accessed every frame by the CPU
    modelMatrixBufferIndex = descriptorSet->createFixedBuffer<glm::mat4>(MAX_FRAMES_IN_FLIGHT * MAX_FIXED_BUFFER, true);
    lightBufferIndex = descriptorSet->createFixedBuffer<GPULight>(MAX_FRAMES_IN_FLIGHT * MAX_FIXED_BUFFER, true);
    shadowDrawDataBufferIndex = descriptorSet->createFixedBuffer<ShadowDrawData>(MAX_FRAMES_IN_FLIGHT * MAX_FIXED_BUFFER, true);
    litPassDataBufferIndex = descriptorSet->createFixedBuffer<LitPassData>(MAX_FRAMES_IN_FLIGHT * MAX_FIXED_BUFFER, true);
    ssrPassDataBufferIndex = descriptorSet->createFixedBuffer<SSRPassData>(MAX_FRAMES_IN_FLIGHT, true);

    // sets the frame offsets for each buffer
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        descriptorSet->setBufferFrameOffset(modelMatrixBufferIndex, i, MAX_FIXED_BUFFER * i);
        descriptorSet->setBufferFrameOffset(lightBufferIndex, i, MAX_FIXED_BUFFER * i);
        descriptorSet->setBufferFrameOffset(shadowDrawDataBufferIndex, i, MAX_FIXED_BUFFER * i);
        descriptorSet->setBufferFrameOffset(litPassDataBufferIndex,i, MAX_FIXED_BUFFER * i);
        descriptorSet->setBufferFrameOffset(ssrPassDataBufferIndex,i,i);
    }

    Gizmos::init(MAX_GIZMO_LINES, &*descriptorSet);

    litDrawDataBufferIndex = descriptorSet->createFixedBuffer<LitDrawData>(MAX_FRAMES_IN_FLIGHT * MAX_FIXED_BUFFER, true);
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        descriptorSet->setBufferFrameOffset(litDrawDataBufferIndex, i, MAX_FIXED_BUFFER * i);
    }

    // indirect draw buffers (separate for shadow and lit passes)
    std::tie(indirectDrawBuffer, indirectDrawBufferMemory, indirectDrawBufferMapped)       = resourceManager->createIndirectDrawBuffer();
    std::tie(litIndirectDrawBuffer, litIndirectDrawBufferMemory, litIndirectDrawBufferMapped) = resourceManager->createIndirectDrawBuffer();

    // after having created all our desire buffers we can initialize the descriptor set
    descriptorSet->createDescriptorSet();

    swapchain->create(*window, vSync);
    createShadowDepthBuffer(DEFAULT_CSM_SHADOW_RESOLUTION);
    uint32_t ssaoW = std::max(1u, static_cast<uint32_t>(startWidth * ssaoResolutionScale));
    uint32_t ssaoH = std::max(1u, static_cast<uint32_t>(startHeight * ssaoResolutionScale));
    uint32_t ssrW = std::max(1u, static_cast<uint32_t>(startWidth * ssrResolutionScale));
    uint32_t ssrH = std::max(1u, static_cast<uint32_t>(startHeight * ssrResolutionScale));
    createSSAOResources(ssaoW, ssaoH);
    createRoughnessMetalResources(startWidth, startHeight);
    createNormalResources(startWidth, startHeight);
    createMotionVectorResources(startWidth,startHeight);
    createColorResolveResources(startWidth, startHeight);
    createSSRResources(ssrW, ssrH);
    createHiZResources(startWidth, startHeight);
    createSyncObjects();

#if DEBUG == 1
    descriptorSet->debugDescriptorSet("after_createDescriptorSet");
#endif

    /////S=================================================DEFAULTS=================================================/////

    defaultSamplerIndex = descriptorSet->allocateSampler(vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear, vk::SamplerAddressMode::eRepeat, VK_TRUE, 16.0, VK_FALSE,
                                                         vk::CompareOp::eLessOrEqual, vk::BorderColor::eFloatOpaqueBlack);
    depthSamplerIndex = descriptorSet->allocateSampler(vk::Filter::eNearest, vk::SamplerMipmapMode::eNearest, vk::SamplerAddressMode::eClampToEdge, VK_FALSE, 16.0, VK_FALSE,
                                                       vk::CompareOp::eLessOrEqual, vk::BorderColor::eFloatOpaqueBlack);
    shadowSamplerIndex = descriptorSet->allocateSampler(vk::Filter::eNearest,                   // Nearest filtering for PCF (shader does the filtering)
                                                        vk::SamplerMipmapMode::eNearest,        // No mipmaps
                                                        vk::SamplerAddressMode::eClampToBorder, // Clamp to avoid wrapping
                                                        VK_FALSE,                               // No anisotropy needed
                                                        1.0f,
                                                        VK_FALSE, // No comparison sampler for manual PCF
                                                        vk::CompareOp::eLessOrEqual,
                                                        vk::BorderColor::eFloatOpaqueWhite // 1.0 = far depth = not in shadow
    );
    // Default albedo (white)
    std::array<uint8_t, 4> whiteColor = {255, 255, 255, 255};
    auto [albedoImage, albedoMemory, albedoImageView] = resourceManager->createTexture(whiteColor.data(), 1, 1, vk::Format::eR8G8B8A8Srgb);
    uint32_t defaultAlbedoIndex = descriptorSet->allocateTexture(std::move(albedoImage), std::move(albedoMemory), std::move(albedoImageView));

    // Default normal (flat normal = 0.5, 0.5, 1.0 in RGB)
    std::array<uint8_t, 4> normalColor = {128, 128, 255, 255};
    auto [normalImage, normalMemory, normalImageView] = resourceManager->createTexture(normalColor.data(), 1, 1, vk::Format::eR8G8B8A8Unorm);
    defaultNormalIndex = descriptorSet->allocateTexture(std::move(normalImage), std::move(normalMemory), std::move(normalImageView));

    // Default roughness = 0.5
    std::array<uint8_t, 4> roughnessColor = {128, 128, 128, 255};
    auto [roughnessImage, roughnessMemory, roughnessImageView] = resourceManager->createTexture(roughnessColor.data(), 1, 1, vk::Format::eR8G8B8A8Unorm);
    uint32_t defaultRoughnessIndex = descriptorSet->allocateTexture(std::move(roughnessImage), std::move(roughnessMemory), std::move(roughnessImageView));

    // Default metallic = 0.0
    std::array<uint8_t, 4> metallicColor = {0, 0, 0, 255};
    auto [metallicImage, metallicMemory, metallicImageView] = resourceManager->createTexture(metallicColor.data(), 1, 1, vk::Format::eR8G8B8A8Unorm);
    uint32_t defaultMetallicIndex = descriptorSet->allocateTexture(std::move(metallicImage), std::move(metallicMemory), std::move(metallicImageView));

    /////S=================================================PIPELINES=================================================/////
    skyboxPipelineIndex =
        pipelineManager->createPipeline<SkyBoxPushConstants>(PipelineCategory::GEOMETRY, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone, vk::False,
                                                             vk::False, "shaders/skybox.spv", descriptorSet->getDescriptorSetLayout(), descriptorSet->getDescriptorSet());

    shadowPipelineIndex = pipelineManager->createPipeline<ShadowPushConstants>(PipelineCategory::SHADOW, vk::PrimitiveTopology::eTriangleList,
                                                                               vk::CullModeFlagBits::eNone, vk::True, vk::True, "shaders/shadow_geometry.spv",
                                                                               descriptorSet->getDescriptorSetLayout(), descriptorSet->getDescriptorSet());
    blurPipelineIndex =
        pipelineManager->createPipeline<BlurPushConstants>(PipelineCategory::POSTPROCESS, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone, vk::False,
                                                           vk::False, "shaders/blur.spv", descriptorSet->getDescriptorSetLayout(), descriptorSet->getDescriptorSet());

    depthPipelineIndex =
        pipelineManager->createPipeline<LitPushConstants>(PipelineCategory::DEPTH_PREPASS, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eBack,vk::True,
                                                                        vk::True,"shaders/depth_prepass.spv", descriptorSet->getDescriptorSetLayout(), descriptorSet->getDescriptorSet());

    litPipelineIndex = pipelineManager->createPipeline<LitPushConstants>(PipelineCategory::GEOMETRY, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eBack, vk::True,
                                                                         vk::True, "shaders/lit.spv", descriptorSet->getDescriptorSetLayout(), descriptorSet->getDescriptorSet());
    gizmoPipelineIndex =
        pipelineManager->createPipeline<LinePushConstants>(PipelineCategory::POSTPROCESS, vk::PrimitiveTopology::eLineList, vk::CullModeFlagBits::eNone, vk::False, vk::False,
                                                           "shaders/line.spv", descriptorSet->getDescriptorSetLayout(), descriptorSet->getDescriptorSet());
    imageViewPipelineIndex =
        pipelineManager->createPipeline<ImageVisPushConstants>(PipelineCategory::POSTPROCESS, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone, vk::False,
                                                               vk::False, "shaders/image_view.spv", descriptorSet->getDescriptorSetLayout(), descriptorSet->getDescriptorSet());

    // default litshader / material
    fallbackLitShader = Shader{.sourceFile = "shaders/lit.spv", .pipelineIndex = litPipelineIndex};
    MaterialFlags defaultTexMask = MaterialFlags::MAT_NONE; // see the material struct definition
    // texMask |= (1U << 0);
    // texMask |= (1U << 1);
    // texMask |= (1U << 3);
    Material defaultMaterial = Material{.shaderSource = fallbackLitShader,
                                        .flags = defaultTexMask,
                                        .color = glm::vec4(0.5, 0.5, 0.5, 1),
                                        .albedoTextureIndex = defaultAlbedoIndex,
                                        .metallic = 0.0,
                                        .metallicTextureIndex = defaultMetallicIndex,
                                        .roughness = 0.5,
                                        .roughnessTextureIndex = defaultRoughnessIndex,
                                        .normalTextureIndex = defaultNormalIndex};
    fallbackDefaultMaterialIndex = addMaterial(defaultMaterial);

#if DEBUG == 1
    descriptorSet->debugDescriptorSet("after_pipeline_creation");
#endif

    // create the root node - end of initialization
    sceneGraph.init(this);
    descriptorSet->allocateFixedBuffer(litPassDataBufferIndex, LitPassData{.samplerIndex = defaultSamplerIndex,
                                                                           .lightCount = 0,
                                                                           .shadowSamplerIndex = shadowSamplerIndex,
                                                                           .cameraPosition = activeCamera.position,
                                                                           .cameraForward = glm::vec3(1, 0, 0),
                                                                           .viewProjection = activeCamera.viewProjection,
                                                                           .prevViewProjection = activeCamera.viewProjection});
}

/////=================================================DRAW FRAME=================================================/////

void Renderer::drawFrame() {
    Tracer::startTrace("draw frame");
    Tracer::startTrace("wait for fences");
    device->getDevice().waitForFences(*inFlightFences[currentFrame], vk::True, UINT64_MAX);
    Tracer::endTrace("wait for fences");
    //TODO make all full screen passes have a resolution scale that can be set dirty when changed/ needs to recreate
    if (ssrResolutionDirty) {
        ssrResolutionDirty = false;
        device->getDevice().waitIdle();
        int w = 0, h = 0;
        glfwGetFramebufferSize(window, &w, &h);
        if (w > 0 && h > 0) {
            uint32_t ssrW = std::max(1u, static_cast<uint32_t>(w * ssrResolutionScale));
            uint32_t ssrH = std::max(1u, static_cast<uint32_t>(h * ssrResolutionScale));
            createSSRResources(ssrW, ssrH);
        }
    }

    Tracer::startTrace("acquire next image");
    auto [result, imageIndex] = swapchain->getSwapChain().acquireNextImage(UINT64_MAX, *presentCompleteSemaphores[currentFrame], nullptr);
    Tracer::endTrace("acquire next image");

    if (result == vk::Result::eErrorOutOfDateKHR) {
        swapchain->recreate(window, vSync);
        handleSwapchainResize();
        return;
    }
    if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
        throw std::runtime_error("failed to acquire swap chain image!");
    }
    Tracer::startTrace("wait image in flight");
    if (imagesInFlight[imageIndex] != VK_NULL_HANDLE) {
        vk::Result waitResult = device->getDevice().waitForFences(imagesInFlight[imageIndex], vk::True, UINT64_MAX);
        if (waitResult != vk::Result::eSuccess) {
            throw std::runtime_error("failed to wait for image fence!");
        }
    }
    Tracer::endTrace("wait image in flight");

    Tracer::startTrace("reset fences");
    imagesInFlight[imageIndex] = *inFlightFences[currentFrame];
    device->getDevice().resetFences(*inFlightFences[currentFrame]);
    Tracer::endTrace("reset fences");

    for (auto& [id, light] : lights) {
        if (light.castsShadows == 1) {
            if (light.type == LightType::Directional) {
                // CSM depends on camera — always recalculate
                NodeOps::calculateCascadedLightSpaceMatrices(light, activeCamera, this);
                descriptorSet->updateFixedBufferWithOffset<GPULight>(lightBufferIndex, id, light.toGPU(), currentFrame);
            } else if (light.type == LightType::Point && light.shadowDirty) {
                glm::mat4 modelMatrix = sceneGraph.getNodes()[light.nodeIndex].worldTransform;
                glm::vec3 lightPos = glm::vec3(modelMatrix[3]);
                NodeOps::calculatePointLightFaceMatrices(light, lightPos);
                descriptorSet->updateFixedBufferWithOffset<GPULight>(lightBufferIndex, id, light.toGPU(), currentFrame);
            }
        }
    }
    Tracer::startTrace("record command buffer");
    commandBuffers[currentFrame].reset();
    recordCommandBuffer(imageIndex);
    Tracer::endTrace("record command buffer");

    Tracer::startTrace("submit & present");
    vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
    const vk::SubmitInfo submitInfo{.waitSemaphoreCount = 1,
                                    .pWaitSemaphores = &*presentCompleteSemaphores[currentFrame],
                                    .pWaitDstStageMask = &waitDestinationStageMask,
                                    .commandBufferCount = 1,
                                    .pCommandBuffers = &*commandBuffers[currentFrame],
                                    .signalSemaphoreCount = 1,
                                    .pSignalSemaphores = &*renderFinishedSemaphores[imageIndex]};

    device->getGraphicsQueue().submit(submitInfo, inFlightFences[currentFrame]);

    const vk::PresentInfoKHR presentInfoKHR{.waitSemaphoreCount = 1,
                                            .pWaitSemaphores = &*renderFinishedSemaphores[imageIndex],
                                            .swapchainCount = 1,
                                            .pSwapchains = &*swapchain->getSwapChain(),
                                            .pImageIndices = &imageIndex};

    try {
        result = device->getPresentQueue().presentKHR(presentInfoKHR);
    } catch (const vk::OutOfDateKHRError&) {
        result = vk::Result::eErrorOutOfDateKHR;
    }

    if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || framebufferResized) {
        framebufferResized = false;
        swapchain->recreate(window, vSync);
        handleSwapchainResize();
    } else if (result != vk::Result::eSuccess) {
        throw std::runtime_error("failed to present swap chain image!");
    }
    Tracer::endTrace("submit & present");

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    totalFrames++;
    //pipelineManager->checkForShaderUpdates(); // TODO enable this
    Tracer::endTrace("draw frame");
}

/////=================================================GET/SET=================================================/////

GLFWwindow* Renderer::getWindow() { return window; }
void Renderer::setWindow(GLFWwindow* pWindow) { window = pWindow; }
const vk::Instance& Renderer::getInstance() const { return *instance; }
Device& Renderer::getDevice() { return *device; }
ResourceManager& Renderer::getResourceManager() { return *resourceManager; }
DescriptorSet& Renderer::getDescriptorSet() { return *descriptorSet; }

Swapchain& Renderer::getSwapchain() { return *swapchain; }
void Renderer::cleanupSwapchain() { swapchain->cleanupSwapChain(); }
const vk::SampleCountFlagBits& Renderer::getMsaaSamples() const { return msaaSamples; }

const uint32_t Renderer::getGraphicsIndex() const { return graphicsIndex; }

uint32_t Renderer::getModelMatrixBufferIndex() { return modelMatrixBufferIndex; }
uint32_t Renderer::getLightBufferIndex() { return lightBufferIndex; }
uint32_t Renderer::getShadowDrawDataBufferIndex() { return shadowDrawDataBufferIndex; }

std::vector<Material>& Renderer::getMaterials() { return materials; }
uint32_t Renderer::addMaterial(Material material) {
    material.materialID = static_cast<uint32_t>(std::hash<Material>{}(material));

    // check if it already exists
    for (uint32_t i = 0; i < materials.size(); i++) {
        if (materials[i] == material) {
            return i;
        }
    }
    materials.push_back(material);
    return materials.size() - 1;
}
void Renderer::addMeshToShader(Node* node, uint32_t submeshIndex, Shader shader, Material material) {
    uint32_t matIdx = 0;
    for (uint32_t i = 0; i < materials.size(); i++) {
        if (materials[i] == material) { matIdx = i; break; }
    }
    for (const auto& e : renderEntries) {
        if (e.node == node && e.submeshIndex == submeshIndex &&
            e.materialIndex == matIdx && e.shaderPipelineIndex == shader.pipelineIndex)
            return;
    }
    renderEntries.push_back({node, submeshIndex, matIdx, shader.pipelineIndex});
    renderListDirty = true;
}
void Renderer::removeMeshFromShader(Node* node, uint32_t subMeshIndex, Shader shader, Material material) {
    uint32_t matIdx = 0;
    for (uint32_t i = 0; i < materials.size(); i++) {
        if (materials[i] == material) { matIdx = i; break; }
    }
    for (auto it = renderEntries.begin(); it != renderEntries.end(); ++it) {
        if (it->node == node && it->submeshIndex == subMeshIndex &&
            it->materialIndex == matIdx && it->shaderPipelineIndex == shader.pipelineIndex) {
            *it = renderEntries.back();
            renderEntries.pop_back();
            renderListDirty = true;
            return;
        }
    }
}
Shader Renderer::getFallBackShader() { return fallbackLitShader; }
uint32_t Renderer::getFallBackMaterial() { return fallbackDefaultMaterialIndex; }
void Renderer::clearRenderList() { renderEntries.clear(); shaderDrawRanges.clear(); renderListDirty = false; }

const std::map<uint32_t, Light>& Renderer::getLights() { return lights; }
std::map<uint32_t, Light>& Renderer::getLightsMutable() { return lights; }
void Renderer::addLight(uint32_t index, Light light) { lights[index] = light; }
Light& Renderer::getLight(uint32_t index) { return lights[index]; }
void Renderer::clearLights() {
    descriptorSet->clearFixedBuffer(lightBufferIndex);
    lights.clear();
}

void Renderer::toggleVsync() {
    vSync = !vSync;
    swapchain->recreate(window, vSync);
    handleSwapchainResize();
}

void Renderer::toggleSSAO() { enableSSAO = !enableSSAO; }
void Renderer::toggleSSR() { enableSSR = !enableSSR; }

void Renderer::toggleBBOXes() { showBBOXes = !showBBOXes; }

void Renderer::setSkyBox(uint32_t skyboxIdx) { this->skyboxIndex = skyboxIdx; }

void Renderer::handleSwapchainResize() {
    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    if (width > 0 && height > 0) {
        uint32_t ssaoW = std::max(1u, static_cast<uint32_t>(width * ssaoResolutionScale));
        uint32_t ssaoH = std::max(1u, static_cast<uint32_t>(height * ssaoResolutionScale));
        uint32_t ssrW = std::max(1u, static_cast<uint32_t>(width * ssrResolutionScale));
        uint32_t ssrH = std::max(1u, static_cast<uint32_t>(height * ssrResolutionScale));
        createSSAOResources(ssaoW, ssaoH);
        createRoughnessMetalResources(width, height);
        createNormalResources(width, height);
        createMotionVectorResources(width,height);
        createColorResolveResources(width, height);
        createSSRResources(ssrW, ssrH);
        createHiZResources(width, height);
    }
    activeCamera.aspectRatio = static_cast<float>(width) / static_cast<float>(height);
    activeCamera.calculateViewProjectionMatrix();
}

void Renderer::blurAttachment(vk::raii::CommandBuffer& cmd, uint32_t sourceTextureIndex, uint32_t tempTextureIndex, uint32_t width, uint32_t height, float blurRadius,
                    uint32_t samplerIndex) {

    auto& blurPipeline = *pipelineManager->getPostProcessPipelines()[blurPipelineIndex];
    auto& sourceTexture = descriptorSet->getTextureResource(sourceTextureIndex);
    auto& tempTexture = descriptorSet->getTextureResource(tempTextureIndex);
    vk::Extent2D extent{width, height};

    // Horizontal blur (source -> temp)
    resourceManager->transitionImageLayout(&cmd, *tempTexture.image, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eColorAttachmentOptimal);
    drawFullscreenPass(cmd, blurPipeline, *tempTexture.imageView, extent,
        BlurPushConstants{.inputTextureIndex = sourceTextureIndex, .samplerIndex = samplerIndex, .isHorizontal = 1, .blurRadius = blurRadius, .resolution = glm::uvec2(width, height)});
    resourceManager->transitionImageLayout(&cmd, *tempTexture.image, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

    // Vertical blur (temp -> source)
    resourceManager->transitionImageLayout(&cmd, *sourceTexture.image, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eColorAttachmentOptimal);
    drawFullscreenPass(cmd, blurPipeline, *sourceTexture.imageView, extent,
        BlurPushConstants{.inputTextureIndex = tempTextureIndex, .samplerIndex = samplerIndex, .isHorizontal = 0, .blurRadius = blurRadius, .resolution = glm::uvec2(width, height)});
    resourceManager->transitionImageLayout(&cmd, *sourceTexture.image, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
}

/////=================================================CREATE RESOURCES=================================================/////

void Renderer::createInstance() {
    constexpr vk::ApplicationInfo appInfo{.pApplicationName = "Shader Forge",
                                          .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
                                          .pEngineName = "No Engine",
                                          .engineVersion = VK_MAKE_VERSION(1, 0, 0),
                                          .apiVersion = vk::ApiVersion13};
    // Get the required layers
    std::vector<char const*> requiredLayers;
#if DEBUG == 1
    requiredLayers.assign(validationLayers.begin(), validationLayers.end());
    // Check if the required layers are supported by the Vulkan implementation.
    auto layerProperties = context.enumerateInstanceLayerProperties();
    if (std::ranges::any_of(requiredLayers, [&layerProperties](auto const& requiredLayer) {
            return std::ranges::none_of(layerProperties, [requiredLayer](auto const& layerProperty) { return strcmp(layerProperty.layerName, requiredLayer) == 0; });
        })) {
        throw std::runtime_error("One or more required layers are not supported!");
    }
#endif
    // get the required extensions
    auto requiredExtensions = getRequiredExtensions();
    // Check if the required extensions are supported by the Vulkan implementation.
    auto extensionProperties = context.enumerateInstanceExtensionProperties();
    for (auto const& requiredExtension : requiredExtensions) {
        if (std::ranges::none_of(extensionProperties,
                                 [requiredExtension](auto const& extensionProperty) { return strcmp(extensionProperty.extensionName, requiredExtension) == 0; })) {
            throw std::runtime_error("Required extension not supported: " + std::string(requiredExtension));
        }
    }
    vk::InstanceCreateInfo createInfo{.pApplicationInfo = &appInfo,
                                      .enabledLayerCount = static_cast<uint32_t>(requiredLayers.size()),
                                      .ppEnabledLayerNames = requiredLayers.data(),
                                      .enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size()),
                                      .ppEnabledExtensionNames = requiredExtensions.data()};

    instance = vk::raii::Instance(context, createInfo);
}

std::vector<const char*> Renderer::getRequiredExtensions() {
    uint32_t glfwExtensionCount = 0;
    auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    return extensions;
}

void Renderer::setupDebugMessenger() {
    vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                                                        vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
    vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags(vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
                                                       vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
    vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT{.messageSeverity = severityFlags, .messageType = messageTypeFlags, .pfnUserCallback = &debugCallback};
    debugMessenger = instance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
}

VKAPI_ATTR vk::Bool32 VKAPI_CALL Renderer::debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity, vk::DebugUtilsMessageTypeFlagsEXT type,
                                                      const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData, void*) {
    std::cerr << "validation layer: type " << to_string(type) << " msg: " << pCallbackData->pMessage << std::endl;
    return vk::False;
}

void Renderer::createSurface() {
    VkSurfaceKHR _surface;
    if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }
    surface = vk::raii::SurfaceKHR(instance, _surface);
}

void Renderer::createCommandPool() {
    vk::CommandPoolCreateInfo poolInfo{.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer, .queueFamilyIndex = graphicsIndex};
    commandPool = vk::raii::CommandPool(device->getDevice(), poolInfo);
}

void Renderer::createCommandBuffers() {
    commandBuffers.clear();
    vk::CommandBufferAllocateInfo allocInfo{.commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = MAX_FRAMES_IN_FLIGHT};
    commandBuffers = vk::raii::CommandBuffers(device->getDevice(), allocInfo);
}

void Renderer::createShadowDepthBuffer(uint32_t resolution) {
    // only recreates if resolution changed
    if (currentShadowDepthResolution == resolution && shadowDepth.view != nullptr) {
        return;
    }
    // Free old resources if they exist
    shadowDepth.image = nullptr;
    shadowDepth.view = nullptr;
    shadowDepth.memory = nullptr;
    resourceManager->createImage(resolution, resolution, 1, vk::SampleCountFlagBits::e1, vk::Format::eD32Sfloat, vk::ImageTiling::eOptimal,
                                 vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal, shadowDepth.image, shadowDepth.memory, 1);
    shadowDepth.view = resourceManager->createImageView(shadowDepth.image, vk::Format::eD32Sfloat, vk::ImageAspectFlagBits::eDepth);
    resourceManager->transitionImageLayout(nullptr, shadowDepth.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal);
    currentShadowDepthResolution = resolution;
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
            resourceManager->createTexture(noiseData.data(), 4, 4, vk::Format::eR8G8B8A8Unorm, vk::ImageType::e2D, vk::ImageViewType::e2D, vk::SampleCountFlagBits::e1, false);
        ssaoNoiseTextureIndex = descriptorSet->allocateTexture(std::move(noiseImage), std::move(noiseMemory), std::move(noiseImageView), "internal/ssao_noise");

        // Noise sampler: repeat + nearest (tiled across screen)
        ssaoNoiseSamplerIndex = descriptorSet->allocateSampler(vk::Filter::eNearest, vk::SamplerMipmapMode::eNearest, vk::SamplerAddressMode::eRepeat,
                                                                VK_FALSE, 1.0f, VK_FALSE, vk::CompareOp::eNever, vk::BorderColor::eFloatOpaqueBlack);
    }

    // SSAO pipeline (only create once)
    if (ssaoPipelineIndex == 0xFFFFFFFF) {
        ssaoPipelineIndex = pipelineManager->createPipeline<SSAOPushConstants>(
            PipelineCategory::POSTPROCESS, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone,
            vk::False, vk::False, "shaders/ssao.spv", descriptorSet->getDescriptorSetLayout(), descriptorSet->getDescriptorSet());
    }

    // SSAO apply pipeline with multiplicative blending (only create once)
    if (ssaoApplyPipelineIndex == 0xFFFFFFFF) {
        ssaoApplyPipelineIndex = pipelineManager->createPipeline<SSAOApplyPushConstants>(
            PipelineCategory::POSTPROCESS_MULTIPLY, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone,
            vk::False, vk::False, "shaders/ssao_apply.spv", descriptorSet->getDescriptorSetLayout(), descriptorSet->getDescriptorSet());
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
        descriptorSet->freeTexture(normalTextureIndex);
    }
    vk::raii::Image image = nullptr;
    vk::raii::DeviceMemory memory = nullptr;
    resourceManager->createImage(width, height, normalMipLevels, vk::SampleCountFlagBits::e1, vk::Format::eR8G8B8A8Unorm, vk::ImageTiling::eOptimal,
                                 vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled |
                                 vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst,
                                 vk::MemoryPropertyFlagBits::eDeviceLocal, image, memory, 1);
    auto view = resourceManager->createImageView(image, vk::Format::eR8G8B8A8Unorm, vk::ImageAspectFlagBits::eColor, normalMipLevels);
    resourceManager->transitionImageLayout(nullptr, image, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal, 0, normalMipLevels);
    normalTextureIndex = descriptorSet->allocateTexture(std::move(image), std::move(memory), std::move(view), "internal/normals", false, width, height);
}

void Renderer::createColorResolveResources(uint32_t width, uint32_t height) {
    colorResolveMipViews.clear();

    fullscreenMipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1;

    if (colorResolveTextureIndex != 0xFFFFFFFF) {
        descriptorSet->freeTexture(colorResolveTextureIndex);
    }

    auto format = swapchain->getSwapChainImageFormat();

    vk::raii::Image image = nullptr;
    vk::raii::DeviceMemory memory = nullptr;
    resourceManager->createImage(width, height, fullscreenMipLevels, vk::SampleCountFlagBits::e1, format, vk::ImageTiling::eOptimal,
                                 vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc,
                                 vk::MemoryPropertyFlagBits::eDeviceLocal, image, memory);

    // Create per-mip image views for rendering to individual levels
    for (uint32_t mip = 0; mip < fullscreenMipLevels; ++mip) {
        vk::ImageViewCreateInfo viewInfo{.image = image,
                                         .viewType = vk::ImageViewType::e2D,
                                         .format = format,
                                         .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = mip, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1}};
        colorResolveMipViews.emplace_back(device->getDevice(), viewInfo);
    }

    // Create a full-chain view for sampling
    auto fullView = resourceManager->createImageView(image, format, vk::ImageAspectFlagBits::eColor, fullscreenMipLevels);
    resourceManager->transitionImageLayout(nullptr, image, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal, 0, fullscreenMipLevels);
    colorResolveTextureIndex = descriptorSet->allocateTexture(std::move(image), std::move(memory), std::move(fullView), "internal/color_resolve", false, width, height);

    // Temp texture for separable blur passes (mipmapped, matching color resolve)
    tempBlurMipViews.clear();

    if (tempBlurTextureIndex != 0xFFFFFFFF) {
        descriptorSet->freeTexture(tempBlurTextureIndex);
    }

    vk::raii::Image tempImage = nullptr;
    vk::raii::DeviceMemory tempMemory = nullptr;
    resourceManager->createImage(width, height, fullscreenMipLevels, vk::SampleCountFlagBits::e1, format, vk::ImageTiling::eOptimal,
                                 vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
                                 vk::MemoryPropertyFlagBits::eDeviceLocal, tempImage, tempMemory);

    for (uint32_t mip = 0; mip < fullscreenMipLevels; ++mip) {
        vk::ImageViewCreateInfo viewInfo{.image = tempImage,
                                         .viewType = vk::ImageViewType::e2D,
                                         .format = format,
                                         .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = mip, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1}};
        tempBlurMipViews.emplace_back(device->getDevice(), viewInfo);
    }

    auto tempFullView = resourceManager->createImageView(tempImage, format, vk::ImageAspectFlagBits::eColor, fullscreenMipLevels);
    resourceManager->transitionImageLayout(nullptr, tempImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal, 0, fullscreenMipLevels);
    tempBlurTextureIndex = descriptorSet->allocateTexture(std::move(tempImage), std::move(tempMemory), std::move(tempFullView), "internal/blur_temp", false, width, height);
}

void Renderer::createMotionVectorResources(uint32_t width, uint32_t height) {
    createOrResizeMSAATarget(motionVectors,width,height, vk::Format::eR16G16Sfloat);
    createOrResizeRenderTarget(motionVectorTextureIndex, width, height, vk::Format::eR16G16Sfloat,"internal/motion_vectors");
}

void Renderer::createSSRResources(uint32_t width, uint32_t height) {

    createOrResizeRenderTarget(ssrCurrentTextureIndex, width, height, swapchain->getSwapChainImageFormat(), "internal/ssr_current");
    createOrResizeRenderTarget(ssrHistoryTextureIndices[0], width, height, swapchain->getSwapChainImageFormat(), "internal/ssr_history0");
    createOrResizeRenderTarget(ssrHistoryTextureIndices[1], width, height, swapchain->getSwapChainImageFormat(), "internal/ssr_history1");

    ssrHistoryInvalid = true;

    // SSR pipeline (only created once)
    if (ssrPipelineIndex == 0xFFFFFFFF) {
        ssrPipelineIndex = pipelineManager->createPipeline<SSRPushConstants>(
            PipelineCategory::POSTPROCESS, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone,
            vk::False, vk::False, "shaders/ssr.spv", descriptorSet->getDescriptorSetLayout(), descriptorSet->getDescriptorSet());

        //initialize pass data too
        descriptorSet->allocateFixedBuffer(ssrPassDataBufferIndex,SSRPassData {});
    }

    // SSR accumulate pipeline (only created once)
    if (ssrAccumulatePipelineIndex == 0xFFFFFFFF) {
        ssrAccumulatePipelineIndex = pipelineManager->createPipeline<SSRAccumulatePushConstants>(
            PipelineCategory::POSTPROCESS, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone,
            vk::False, vk::False, "shaders/ssr_accumulate.spv", descriptorSet->getDescriptorSetLayout(), descriptorSet->getDescriptorSet());
    }

    // SSR apply pipeline (only created once)
    if (ssrApplyPipelineIndex == 0xFFFFFFFF) {
        ssrApplyPipelineIndex = pipelineManager->createPipeline<SSRApplyPushConstants>(
            PipelineCategory::POSTPROCESS, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone,
            vk::False, vk::False, "shaders/ssr_apply.spv", descriptorSet->getDescriptorSetLayout(), descriptorSet->getDescriptorSet());
    }
}

void Renderer::createHiZResources(uint32_t width, uint32_t height) {
    hiZMipViews.clear();

    // Calculate mip levels for the Hi-Z pyramid
    hiZMipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1;

    // Free previous Hi-Z texture if it exists
    if (hiZTextureIndex != 0xFFFFFFFF) {
        descriptorSet->freeTexture(hiZTextureIndex);
    }

    // Create mipmapped R32Sfloat image
    vk::raii::Image image = nullptr;
    vk::raii::DeviceMemory memory = nullptr;
    resourceManager->createImage(width, height, hiZMipLevels, vk::SampleCountFlagBits::e1, vk::Format::eR32Sfloat, vk::ImageTiling::eOptimal,
                                 vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
                                 vk::MemoryPropertyFlagBits::eDeviceLocal, image, memory);

    // Create per-mip image views for rendering to individual levels
    for (uint32_t mip = 0; mip < hiZMipLevels; ++mip) {
        vk::ImageViewCreateInfo viewInfo{.image = image,
                                         .viewType = vk::ImageViewType::e2D,
                                         .format = vk::Format::eR32Sfloat,
                                         .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = mip, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1}};
        hiZMipViews.emplace_back(device->getDevice(), viewInfo);
    }

    // Create a full-chain view for sampling
    auto fullView = resourceManager->createImageView(image, vk::Format::eR32Sfloat, vk::ImageAspectFlagBits::eColor, hiZMipLevels);
    resourceManager->transitionImageLayout(nullptr, image, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal, 0, hiZMipLevels);
    hiZTextureIndex = descriptorSet->allocateTexture(std::move(image), std::move(memory), std::move(fullView), "internal/hiZ", false, width, height);

    // Hi-Z pipeline (only created once)
    if (hiZPipelineIndex == 0xFFFFFFFF) {
        hiZPipelineIndex = pipelineManager->createPipeline<HiZPushConstants>(
            PipelineCategory::BEFORE_GEOMETRY, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone,
            vk::False, vk::False, "shaders/hiz_reduce.spv", descriptorSet->getDescriptorSetLayout(), descriptorSet->getDescriptorSet());
    }
}

void Renderer::createSyncObjects() {
    presentCompleteSemaphores.clear();
    renderFinishedSemaphores.clear();
    inFlightFences.clear();
    imagesInFlight.clear();

    // Separate semaphores for acquisition (per frame) and rendering (per image)
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        presentCompleteSemaphores.emplace_back(device->getDevice(), vk::SemaphoreCreateInfo());
        inFlightFences.emplace_back(device->getDevice(), vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
    }

    // Render finished semaphores per swapchain image
    for (size_t i = 0; i < swapchain->getSwapImageSize(); i++) {
        renderFinishedSemaphores.emplace_back(device->getDevice(), vk::SemaphoreCreateInfo());
    }

    imagesInFlight.resize(swapchain->getSwapImageSize(), VK_NULL_HANDLE);
}

/////=================================================RENDERING=================================================/////

void Renderer::recordCommandBuffer(uint32_t imageIndex) {
    auto& cmd = commandBuffers[currentFrame];
    cmd.begin({});

    Tracer::startTrace("record shadow pass");
    for (auto& [lightId, light] : lights) {
        if (light.castsShadows != 1) continue;
        // Skip point lights whose shadow maps are already up to date
        if (light.type == LightType::Point && !light.shadowDirty) continue;
        recordShadowPass(cmd, light);
        if (light.type == LightType::Point) light.shadowDirty = false;
    }
    Tracer::endTrace("record shadow pass");

    Tracer::startTrace("record geo pass");
    recordGeometryPass(cmd, imageIndex);
    Tracer::endTrace("record geo pass");

    Tracer::startTrace("record ssr pass");
    if (enableSSR && ssrPipelineIndex != 0xFFFFFFFF)
        recordSSRPass(cmd, imageIndex);
    Tracer::endTrace("record ssr pass");
    
    Tracer::startTrace("record ssao pass");
    if (enableSSAO && ssaoPipelineIndex != 0xFFFFFFFF)
        recordSSAOPass(cmd, imageIndex);
    Tracer::endTrace("record ssao pass");

    Tracer::startTrace("record image vis pass");
    if (imageVisIndex != 0xFFFFFFFF)
        recordImageVisPass(cmd, imageIndex);
    Tracer::endTrace("record image vis pass");

    recordOverlayPass(cmd, imageIndex);

    resourceManager->transitionImageLayout(&cmd, swapchain->getSwapChainImages()[imageIndex], vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR);
    cmd.end();
}

template <typename PerSubMeshFn>
void Renderer::buildGeometryDrawCommands(const std::array<Plane, 6>& frustumPlanes, bool doCulling, PerSubMeshFn&& perSubMeshFn) {
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
        Node* node = entry.node;
        uint32_t meshIdx = node->getMeshIndex();

        if (assetManager.meshes[meshIdx].freed) {
            if (freedMeshes.insert(meshIdx).second) {
                for (uint32_t subMesh : assetManager.meshes[meshIdx].subMeshes) {
                    descriptorSet->freeVariableBuffer(vertexBufferIndex, assetManager.subMeshes[subMesh].vertexAllocationIndex);
                    descriptorSet->freeVariableBuffer(indexBufferIndex, assetManager.subMeshes[subMesh].indexAllocationIndex);
                    assetManager.freeSubMeshes.push(subMesh);
                }
                assetManager.freeMeshes.push(meshIdx);
            }
            continue;
        }

        if (node->isBoundingBoxValid() && doCulling) {
            if (!isAABBInFrustum(node->getBoundingBoxMin(), node->getBoundingBoxMax(), frustumPlanes))
                continue;
        }

        const auto& subMesh = assetManager.subMeshes[entry.submeshIndex];

        glm::vec3 subWorldMin, subWorldMax;
        transformAABBToWorldSpace(subMesh.boundingBoxMin, subMesh.boundingBoxMax, node->getTransform(), subWorldMin, subWorldMax);
        if (!isAABBInFrustum(subWorldMin, subWorldMax, frustumPlanes) && doCulling) {
            culledCount++;
            continue;
        }

        if (showBBOXes && doCulling)
            Gizmos::drawBox(subWorldMin, subWorldMax, glm::vec4(1.0f, 1.0f, 0.0f, 1.0f));

        if (entry.shaderPipelineIndex != currentPipelineIdx) {
            currentPipelineIdx = entry.shaderPipelineIndex;
            shaderDrawRanges.push_back({currentPipelineIdx, static_cast<uint32_t>(indirectCommands.size()), 0});
        }

        indirectCommands.push_back({.indexCount    = subMesh.indexCount,
                                    .instanceCount = 1,
                                    .firstIndex    = static_cast<uint32_t>(subMesh.indexOffset / sizeof(uint32_t)),
                                    .vertexOffset  = 0,
                                    .firstInstance = 0});
        shaderDrawRanges.back().commandCount++;

        const Material& material = materials[entry.materialIndex];
        perSubMeshFn(subMesh, *node, material);
    }
}

void Renderer::recordHiZPass(vk::raii::CommandBuffer& cmd) {
    if (hiZTextureIndex == 0xFFFFFFFF || hiZPipelineIndex == 0xFFFFFFFF) return;

    auto& hiZRes = descriptorSet->getTextureResource(hiZTextureIndex);
    auto& pipeline = *pipelineManager->getBeforeGeoPipelines()[hiZPipelineIndex];
    uint32_t w = hiZRes.width;
    uint32_t h = hiZRes.height;

    // Transition depth resolve to shader read for sampling
    resourceManager->transitionImageLayout(&cmd, *descriptorSet->getTextureResource(swapchain->getDepthResolveIndex()).image,
        vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

    for (uint32_t mip = 0; mip < hiZMipLevels; ++mip) {
        uint32_t mipW = std::max(1u, w >> mip);
        uint32_t mipH = std::max(1u, h >> mip);

        // Transition this mip to color attachment
        resourceManager->transitionImageLayout(&cmd, *hiZRes.image,
            vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eColorAttachmentOptimal, mip, 1);

        HiZPushConstants hizPC;
        if (mip == 0) {
            // Mip 0: copy from depth resolve
            hizPC = {.inputTextureIndex = swapchain->getDepthResolveIndex(),
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
        resourceManager->transitionImageLayout(&cmd, *hiZRes.image,
            vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, mip, 1);
    }
    // hiZ empty space calculation pass?

    // Transition depth resolve back to depth attachment
    resourceManager->transitionImageLayout(&cmd, *descriptorSet->getTextureResource(swapchain->getDepthResolveIndex()).image,
        vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eDepthStencilAttachmentOptimal);
}

void Renderer::recordGeometryPass(vk::raii::CommandBuffer& cmd, uint32_t imageIndex) {
    resourceManager->transitionImageLayouts(cmd, {
        {swapchain->getSwapChainImages()[imageIndex],                            vk::ImageLayout::eUndefined,              vk::ImageLayout::eColorAttachmentOptimal},
        {swapchain->getColorImage(),                                             vk::ImageLayout::eUndefined,              vk::ImageLayout::eColorAttachmentOptimal},
        {swapchain->getDepthImage(),                                             vk::ImageLayout::eUndefined,              vk::ImageLayout::eDepthStencilAttachmentOptimal},
        {roughnessMetal.image,                                                   vk::ImageLayout::eUndefined,              vk::ImageLayout::eColorAttachmentOptimal},
        {*descriptorSet->getTextureResource(roughnessMetalTextureIndex).image,   vk::ImageLayout::eShaderReadOnlyOptimal,  vk::ImageLayout::eColorAttachmentOptimal},
        {normalMSAA.image,                                                       vk::ImageLayout::eUndefined,              vk::ImageLayout::eColorAttachmentOptimal},
        {*descriptorSet->getTextureResource(normalTextureIndex).image,           vk::ImageLayout::eShaderReadOnlyOptimal,  vk::ImageLayout::eColorAttachmentOptimal},
        {motionVectors.image,                                                    vk::ImageLayout::eUndefined,              vk::ImageLayout::eColorAttachmentOptimal},
        {*descriptorSet->getTextureResource(motionVectorTextureIndex).image,     vk::ImageLayout::eShaderReadOnlyOptimal,  vk::ImageLayout::eColorAttachmentOptimal},
    });

    vk::ClearValue clearColor{.color = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f})};
    vk::ClearValue clearDepth{.depthStencil = vk::ClearDepthStencilValue{1.0f, 0}};

    // Frustum cull and build draw commands + lit draw data
    Camera fakeCam = activeCamera;
    fakeCam.fov = cullFovScale * activeCamera.fov;
    fakeCam.calculateViewProjectionMatrix();
    std::array<Plane, 6> frustumPlanes = extractFrustumPlanes(fakeCam.viewProjection);
    culledCount = 0;
    litDrawDataList.clear();
    buildGeometryDrawCommands(frustumPlanes, true, [&](const auto& subMesh, auto& node, const auto& material) {
        litDrawDataList.push_back({.vertexAllocationIndex = subMesh.vertexAllocationIndex,
                                   .vertexOffset          = static_cast<uint32_t>(subMesh.vertexOffset),
                                   .vertexStride          = subMesh.vertexStride,
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

    vk::DeviceSize frameByteOffset = currentFrame * MAX_INDIRECT_COMMANDS * sizeof(DrawIndexedIndirectCommand);
    vk::Buffer indexBufferHandle = descriptorSet->getVariableBuffer(indexBufferIndex);

    glm::vec3 cameraForward = glm::normalize(activeCamera.target - activeCamera.position);

    LitPassData litPassData {
        .samplerIndex = defaultSamplerIndex,
        .lightCount = static_cast<uint32_t>(lights.size()),
        .shadowSamplerIndex = shadowSamplerIndex,
        .cameraPosition = activeCamera.position,
        .cameraForward = cameraForward,
        .viewProjection = activeCamera.viewProjection,
        .prevViewProjection = activeCamera.prevViewProjection
    };
    descriptorSet->updateFixedBufferWithOffset<LitPassData>(litPassDataBufferIndex,0,litPassData,currentFrame);

    if (!indirectCommands.empty()) {
        // Upload indirect commands and per-draw data (shared between depth prepass and lit pass)
        memcpy(static_cast<char*>(litIndirectDrawBufferMapped) + frameByteOffset, indirectCommands.data(), indirectCommands.size() * sizeof(DrawIndexedIndirectCommand));

        auto* litDataPtr = descriptorSet->getFixedBufferMappedData<LitDrawData>(litDrawDataBufferIndex);
        if (litDataPtr) {
            uint32_t frameOffset = currentFrame * MAX_FIXED_BUFFER;
            memcpy(&litDataPtr[frameOffset], litDrawDataList.data(), litDrawDataList.size() * sizeof(LitDrawData));
        }

        LitPushConstants pushConstants = {.vertexBufferAddress  = descriptorSet->getVariableBuffers()[vertexBufferIndex]->address,
                                          .modelMatricesAddress = descriptorSet->getFixedBuffers()[modelMatrixBufferIndex]->address + static_cast<vk::DeviceSize>(currentFrame) * MAX_FIXED_BUFFER * sizeof(glm::mat4),
                                          .lightsAddress        = descriptorSet->getFixedBuffers()[lightBufferIndex]->address + static_cast<vk::DeviceSize>(currentFrame) * MAX_FIXED_BUFFER * sizeof(GPULight),
                                          .litDrawDataAddress   = descriptorSet->getFixedBuffers()[litDrawDataBufferIndex]->address + static_cast<vk::DeviceSize>(currentFrame) * MAX_FIXED_BUFFER * sizeof(LitDrawData),
                                          .litPassDataAddress   = descriptorSet->getFixedBuffers()[litPassDataBufferIndex]->address + static_cast<vk::DeviceSize>(currentFrame) * MAX_FIXED_BUFFER * sizeof(LitPassData)
                                        };

        // --- Depth prepass (depth-only, no color attachment) ---
        vk::RenderingAttachmentInfo depthPrepassAttachment = {.imageView = swapchain->getDepthImageView(),
                                                              .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                                              .resolveMode = vk::ResolveModeFlagBits::eMin,
                                                              .resolveImageView = swapchain->getDepthResolveImageView(),
                                                              .resolveImageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                                              .loadOp = vk::AttachmentLoadOp::eClear,
                                                              .storeOp = vk::AttachmentStoreOp::eStore,
                                                              .clearValue = clearDepth};

        auto& motionVectorResolve = descriptorSet->getTextureResource(motionVectorTextureIndex);
        vk::ClearValue clearMotionVectors{.color = vk::ClearColorValue(std::array<float, 4>{0.0f,0.0f,0.0f,1.0f})};
        vk::RenderingAttachmentInfo motionVectorAttachment = {  .imageView = motionVectors.view,
                                                                .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                                .resolveMode = vk::ResolveModeFlagBits::eAverage,
                                                                .resolveImageView = *motionVectorResolve.imageView,
                                                                .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                                .loadOp = vk::AttachmentLoadOp::eClear,
                                                                .storeOp = vk::AttachmentStoreOp::eStore,
                                                                .clearValue = clearMotionVectors};


        vk::RenderingInfo depthRenderingInfo = {.renderArea = {.offset = {0, 0}, .extent = swapchain->getSwapChainExtent()},
                                                .layerCount = 1,
                                                .colorAttachmentCount = 1,
                                                .pColorAttachments = &motionVectorAttachment,
                                                .pDepthAttachment = &depthPrepassAttachment};

        cmd.beginRendering(depthRenderingInfo);
        setFullscreenViewport(cmd, swapchain->getSwapChainExtent());

        auto& depthPipeline = pipelineManager->getBeforeGeoPipelines()[depthPipelineIndex];
        bindPipeline(cmd, *depthPipeline);
        cmd.bindIndexBuffer(indexBufferHandle, 0, vk::IndexType::eUint32);
        cmd.pushConstants<LitPushConstants>(depthPipeline->layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pushConstants);
        cmd.drawIndexedIndirect(*litIndirectDrawBuffer, frameByteOffset, static_cast<uint32_t>(indirectCommands.size()), sizeof(DrawIndexedIndirectCommand));

        cmd.endRendering();

        recordHiZPass(cmd);
    }

    // --- Lit geometry pass (2 color attachments: color + roughness/metallic) ---
    auto& colorResolve = descriptorSet->getTextureResource(colorResolveTextureIndex);
    resourceManager->transitionImageLayout(&cmd, *colorResolve.image,
        vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eColorAttachmentOptimal);

    vk::RenderingAttachmentInfo colorAttachment = {.imageView = swapchain->getColorImageView(),
                                                   .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                   .resolveMode = vk::ResolveModeFlagBits::eAverage,
                                                   .resolveImageView = *colorResolve.imageView,
                                                   .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                   .loadOp = vk::AttachmentLoadOp::eClear,
                                                   .storeOp = vk::AttachmentStoreOp::eStore,
                                                   .clearValue = clearColor};

    auto& roughnessMetalResolve = descriptorSet->getTextureResource(roughnessMetalTextureIndex);
    vk::ClearValue clearRoughMetal{.color = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f})};
    vk::RenderingAttachmentInfo roughnessMetalAttachment = {.imageView = *roughnessMetal.view,
                                                             .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                             .resolveMode = vk::ResolveModeFlagBits::eAverage,
                                                             .resolveImageView = *roughnessMetalResolve.imageView,
                                                             .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                             .loadOp = vk::AttachmentLoadOp::eClear,
                                                             .storeOp = vk::AttachmentStoreOp::eStore,
                                                             .clearValue = clearRoughMetal};

    auto& normalResolve = descriptorSet->getTextureResource(normalTextureIndex);
    vk::ClearValue clearNormal{.color = vk::ClearColorValue(std::array<float, 4>{0.5f, 0.5f, 1.0f, 1.0f})};
    vk::RenderingAttachmentInfo normalAttachment = {.imageView = *normalMSAA.view,
                                                     .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                     .resolveMode = vk::ResolveModeFlagBits::eAverage,
                                                     .resolveImageView = *normalResolve.imageView,
                                                     .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                     .loadOp = vk::AttachmentLoadOp::eClear,
                                                     .storeOp = vk::AttachmentStoreOp::eStore,
                                                     .clearValue = clearNormal};

    vk::RenderingAttachmentInfo depthAttachmentInfo = {.imageView = swapchain->getDepthImageView(),
                                                       .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                                       .resolveMode = vk::ResolveModeFlagBits::eMin,
                                                       .resolveImageView = swapchain->getDepthResolveImageView(),
                                                       .resolveImageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                                       .loadOp = vk::AttachmentLoadOp::eLoad,
                                                       .storeOp = vk::AttachmentStoreOp::eDontCare};

    std::array<vk::RenderingAttachmentInfo, 4> colorAttachments = {colorAttachment, roughnessMetalAttachment, normalAttachment};
    vk::RenderingInfo renderingInfo = {.renderArea = {.offset = {0, 0}, .extent = swapchain->getSwapChainExtent()},
                                       .layerCount = 1,
                                       .colorAttachmentCount = 4,
                                       .pColorAttachments = colorAttachments.data(),
                                       .pDepthAttachment = &depthAttachmentInfo};

    cmd.beginRendering(renderingInfo);
    setFullscreenViewport(cmd, swapchain->getSwapChainExtent());

    // skybox
    auto& skyboxPipeline = pipelineManager->getGeoPipelines()[skyboxPipelineIndex];
    bindPipeline(cmd, *skyboxPipeline);
    SkyBoxPushConstants skyboxConstants = {.skyboxIndex = skyboxIndex, .blur = 0.5, .invViewProjMatrix = glm::inverse(activeCamera.viewProjection)};
    cmd.pushConstants<SkyBoxPushConstants>(*skyboxPipeline->layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, skyboxConstants);
    cmd.draw(3, 1, 0, 0);

    // lit geometry — reuses the same indirect buffer from the prepass
    if (!indirectCommands.empty()) {
        cmd.bindIndexBuffer(indexBufferHandle, 0, vk::IndexType::eUint32);
        auto& geoPipelines = pipelineManager->getGeoPipelines();

        LitPushConstants pushConstants = {.vertexBufferAddress  = descriptorSet->getVariableBuffers()[vertexBufferIndex]->address,
                                          .modelMatricesAddress = descriptorSet->getFixedBuffers()[modelMatrixBufferIndex]->address + static_cast<vk::DeviceSize>(currentFrame) * MAX_FIXED_BUFFER * sizeof(glm::mat4),
                                          .lightsAddress        = descriptorSet->getFixedBuffers()[lightBufferIndex]->address + static_cast<vk::DeviceSize>(currentFrame) * MAX_FIXED_BUFFER * sizeof(GPULight),
                                          .litDrawDataAddress   = descriptorSet->getFixedBuffers()[litDrawDataBufferIndex]->address + static_cast<vk::DeviceSize>(currentFrame) * MAX_FIXED_BUFFER * sizeof(LitDrawData),
                                          .litPassDataAddress   = descriptorSet->getFixedBuffers()[litPassDataBufferIndex]->address + static_cast<vk::DeviceSize>(currentFrame) * MAX_FIXED_BUFFER * sizeof(LitPassData)
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
    resourceManager->transitionImageLayouts(cmd, {
        {*colorResolve.image,                         vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eTransferSrcOptimal},
        {swapchain->getSwapChainImages()[imageIndex],  vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eTransferDstOptimal},
    });

    vk::ImageCopy copyRegion{
        .srcSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .layerCount = 1},
        .dstSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .layerCount = 1},
        .extent = {swapchain->getSwapChainExtent().width, swapchain->getSwapChainExtent().height, 1}
    };
    cmd.copyImage(*colorResolve.image, vk::ImageLayout::eTransferSrcOptimal,
                  swapchain->getSwapChainImages()[imageIndex], vk::ImageLayout::eTransferDstOptimal,
                  copyRegion);

    // Transition back: color resolve to shader readable, swapchain to color attachment, roughness-metal to shader readable for SSR
    resourceManager->transitionImageLayouts(cmd, {
        {*colorResolve.image,                                                    vk::ImageLayout::eTransferSrcOptimal,    vk::ImageLayout::eShaderReadOnlyOptimal},
        {swapchain->getSwapChainImages()[imageIndex],                             vk::ImageLayout::eTransferDstOptimal,    vk::ImageLayout::eColorAttachmentOptimal},
        {*descriptorSet->getTextureResource(roughnessMetalTextureIndex).image,   vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal},
    });

    // Generate normal mips inline for SSR pre-filtering
    auto& normalRes = descriptorSet->getTextureResource(normalTextureIndex);
    resourceManager->generateMipmaps(*normalRes.image, vk::Format::eR8G8B8A8Unorm,
        static_cast<int32_t>(normalRes.width), static_cast<int32_t>(normalRes.height),
        normalMipLevels, 1, &cmd, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

    // Transition motion vecs to shader read only
    resourceManager->transitionImageLayout(&cmd, *descriptorSet->getTextureResource(motionVectorTextureIndex).image,
                                           vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
}

void Renderer::recordOverlayPass(vk::raii::CommandBuffer& cmd, uint32_t imageIndex) {
    auto extent = swapchain->getSwapChainExtent();
    vk::RenderingAttachmentInfo colorAttachment = {.imageView = swapchain->getSwapChainImageViews()[imageIndex],
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
    for(auto& line : Gizmos::getNoDiscardLines()){
        Gizmos::drawLine(line.second);
    }
    auto& gizmoPipeline = pipelineManager->getPostProcessPipelines()[gizmoPipelineIndex];
    bindPipeline(cmd, *gizmoPipeline);
    LinePushConstants lineConstants = {.lineVertsAddress = Gizmos::getLineBufferAddress(),
                                       .depthTextureIndex = swapchain->getDepthResolveIndex(),
                                       .depthSamplerIndex = depthSamplerIndex,
                                       .viewProjection = activeCamera.viewProjection};
    cmd.pushConstants<LinePushConstants>(*gizmoPipeline->layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, lineConstants);
    cmd.draw(Gizmos::getVertexCount(), 1, 0, 0);

    // GUI
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), *cmd);
    cmd.endRendering();
}

void Renderer::recordSSAOPass(vk::raii::CommandBuffer& cmd, uint32_t imageIndex) {
    auto swapExtent = swapchain->getSwapChainExtent();
    auto& ssaoTexture = descriptorSet->getTextureResource(ssaoTextureIndex);
    vk::Extent2D ssaoExtent{ssaoTexture.width, ssaoTexture.height};

    // Transition depth to readable, SSAO target to color attachment
    resourceManager->transitionImageLayouts(cmd, {
        {swapchain->getDepthImage(),  vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal},
        {*ssaoTexture.image,          vk::ImageLayout::eShaderReadOnlyOptimal,          vk::ImageLayout::eColorAttachmentOptimal},
    });

    // Render SSAO at (potentially lower) SSAO resolution
    drawFullscreenPass(cmd, *pipelineManager->getPostProcessPipelines()[ssaoPipelineIndex], *ssaoTexture.imageView, ssaoExtent,
        SSAOPushConstants{.invProjection = glm::inverse(activeCamera.projectionMatrix),
                          .depthIndex = swapchain->getDepthResolveIndex(),
                          .depthSamplerIndex = depthSamplerIndex,
                          .noiseIndex = ssaoNoiseTextureIndex,
                          .noiseSamplerIndex = ssaoNoiseSamplerIndex,
                          .resolution = glm::uvec2(ssaoExtent.width, ssaoExtent.height),
                          .radius = ssaoRadius,
                          .bias = ssaoBias,
                          .power = ssaoPower,
                          .kernelSize = 32},
        vk::AttachmentLoadOp::eClear, {1.0f, 1.0f, 1.0f, 1.0f});

    // Blur at SSAO resolution
    resourceManager->transitionImageLayout(&cmd, *ssaoTexture.image, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
    blurAttachment(cmd, ssaoTextureIndex, ssaoBlurTextureIndex, ssaoExtent.width, ssaoExtent.height, 2.0f, depthSamplerIndex);

    // Apply to swapchain at full resolution (sampler handles upscale)
    drawFullscreenPass(cmd, *pipelineManager->getPostProcessPipelines()[ssaoApplyPipelineIndex], *swapchain->getSwapChainImageViews()[imageIndex], swapExtent,
        SSAOApplyPushConstants{.ssaoTextureIndex = ssaoTextureIndex, .samplerIndex = depthSamplerIndex},
        vk::AttachmentLoadOp::eLoad);

    // Transition depth back
    resourceManager->transitionImageLayout(&cmd, swapchain->getDepthImage(), vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eDepthStencilAttachmentOptimal);
}

void Renderer::recordSSRPass(vk::raii::CommandBuffer& cmd, uint32_t imageIndex) {
    auto swapExtent = swapchain->getSwapChainExtent();
    auto& ssrCurrent = descriptorSet->getTextureResource(ssrCurrentTextureIndex);
    vk::Extent2D ssrExtent{ssrCurrent.width, ssrCurrent.height};

    uint32_t readHistory = ssrHistoryTextureIndices[ssrHistoryFlip];
    uint32_t writeHistory = ssrHistoryTextureIndices[1 - ssrHistoryFlip];
    auto& ssrWriteHist = descriptorSet->getTextureResource(writeHistory);

    // --- Sub-pass 0: Generate blurred mip chain for cone tracing (GPU Pro 5 style)
    // Mip 0 is already populated by the geometry pass MSAA resolve.
    // For each subsequent mip: 2-pass separable Gaussian blur reading mip N-1, writing to mip N.
    auto& colorRes = descriptorSet->getTextureResource(colorResolveTextureIndex);
    auto& tempTexture = descriptorSet->getTextureResource(tempBlurTextureIndex);
    auto& blurPipeline = *pipelineManager->getPostProcessPipelines()[blurPipelineIndex];

    uint32_t maxBlurMips = std::min(fullscreenMipLevels, 6u);
    for (uint32_t mip = 1; mip < fullscreenMipLevels; ++mip) {
        uint32_t mipW = std::max(1u, colorRes.width >> mip);
        uint32_t mipH = std::max(1u, colorRes.height >> mip);
        vk::Extent2D mipExtent{mipW, mipH};

        // Horizontal blur: read colorResolve mip N-1 -> write temp mip N
        resourceManager->transitionImageLayout(&cmd, *tempTexture.image,
            vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eColorAttachmentOptimal, mip, 1);

        drawFullscreenPass(cmd, blurPipeline, *tempBlurMipViews[mip], mipExtent,
            BlurPushConstants{.inputTextureIndex = colorResolveTextureIndex, .samplerIndex = defaultSamplerIndex,
                              .isHorizontal = 1, .blurRadius = 1.0f, .resolution = glm::uvec2(mipW, mipH),
                              .mipLevel = mip - 1});

        resourceManager->transitionImageLayout(&cmd, *tempTexture.image,
            vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, mip, 1);

        // Vertical blur: read temp mip N -> write colorResolve mip N
        resourceManager->transitionImageLayout(&cmd, *colorRes.image,
            vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eColorAttachmentOptimal, mip, 1);

        drawFullscreenPass(cmd, blurPipeline, *colorResolveMipViews[mip], mipExtent,
            BlurPushConstants{.inputTextureIndex = tempBlurTextureIndex, .samplerIndex = defaultSamplerIndex,
                              .isHorizontal = 0, .blurRadius = 1.0f, .resolution = glm::uvec2(mipW, mipH),
                              .mipLevel = mip});

        resourceManager->transitionImageLayout(&cmd, *colorRes.image,
            vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, mip, 1);
    }

    // --- Sub-pass 1: Ray trace -> ssrCurrentTextureIndex ---
    resourceManager->transitionImageLayouts(cmd, {
        {swapchain->getDepthImage(),  vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal},
        {*ssrCurrent.image,           vk::ImageLayout::eShaderReadOnlyOptimal,          vk::ImageLayout::eColorAttachmentOptimal},
    });

    descriptorSet->updateFixedBufferWithOffset(ssrPassDataBufferIndex, 0,
                                     SSRPassData{
                                        .invViewProj = glm::inverse(activeCamera.viewProjection),
                                        .viewProj = activeCamera.viewProjection,
                                        .cameraPos = activeCamera.position,
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
                                        .thickness = ssrThickness,
                                        .roughnessThreshold = ssrRoughnessThreshold,
                                        .maxSteps = ssrMaxSteps,
                                        .frameIndex = totalFrames,
                                     }, currentFrame);

    drawFullscreenPass(cmd, *pipelineManager->getPostProcessPipelines()[ssrPipelineIndex], *ssrCurrent.imageView, ssrExtent,
        SSRPushConstants{ .ssrPassDataAddress = descriptorSet->getFixedBuffers()[ssrPassDataBufferIndex]->address + static_cast<vk::DeviceSize>(currentFrame) * sizeof(SSRPassData)},
        vk::AttachmentLoadOp::eClear, {0.0f, 0.0f, 0.0f, 0.0f});

    // --- Sub-pass 2: Temporal accumulate -> writeHistory ---
    resourceManager->transitionImageLayouts(cmd, {
        {*ssrCurrent.image,    vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal},
        {*ssrWriteHist.image,  vk::ImageLayout::eShaderReadOnlyOptimal,  vk::ImageLayout::eColorAttachmentOptimal},
    });

    drawFullscreenPass(cmd, *pipelineManager->getPostProcessPipelines()[ssrAccumulatePipelineIndex], *ssrWriteHist.imageView, ssrExtent,
        SSRAccumulatePushConstants{
            .currentSSRIndex = ssrCurrentTextureIndex,
            .historySSRIndex = readHistory,
            .motionVectorIndex = motionVectorTextureIndex,
            .samplerIndex = defaultSamplerIndex,
            .temporalBlend = ssrTemporalBlend,
            .historyValid = ssrHistoryInvalid ? 0u : 1u,
        },
        vk::AttachmentLoadOp::eClear, {0.0f, 0.0f, 0.0f, 0.0f});

    resourceManager->transitionImageLayout(&cmd, *ssrWriteHist.image, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

    // --- Sub-pass 3: Apply accumulated SSR to swapchain ---
    drawFullscreenPass(cmd, *pipelineManager->getPostProcessPipelines()[ssrApplyPipelineIndex], *swapchain->getSwapChainImageViews()[imageIndex], swapExtent,
        SSRApplyPushConstants{
            .samplerIndex = defaultSamplerIndex,
            .sceneColorIndex = colorResolveTextureIndex,
            .sceneSamplerIndex = defaultSamplerIndex,
            .ssrTextureIndex = writeHistory,
        },
        vk::AttachmentLoadOp::eLoad);

    // transition depth back
    resourceManager->transitionImageLayout(&cmd, swapchain->getDepthImage(), vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eDepthStencilAttachmentOptimal);

    ssrHistoryFlip = 1 - ssrHistoryFlip;
    ssrHistoryInvalid = false;
}

void Renderer::recordImageVisPass(vk::raii::CommandBuffer& cmd, uint32_t imageIndex) {
    if (imageVisIndex == 0xFFFFFFFF)
        return;

    auto extent = swapchain->getSwapChainExtent();
    auto& visTexture = descriptorSet->getTextureResource(imageVisIndex);
    float imgAspect = (visTexture.width > 0 && visTexture.height > 0)
                          ? static_cast<float>(visTexture.width) / static_cast<float>(visTexture.height)
                          : static_cast<float>(extent.width) / static_cast<float>(extent.height);

    drawFullscreenPass(cmd, *pipelineManager->getPostProcessPipelines()[imageViewPipelineIndex], *swapchain->getSwapChainImageViews()[imageIndex],
                       extent,
                       ImageVisPushConstants{.imageIndex = imageVisIndex,
                                             .samplerIndex = defaultSamplerIndex,
                                             .flags = imageVisFlags,
                                             .nearPlane = activeCamera.nearPlane,
                                             .farPlane = activeCamera.farPlane,
                                             .imageAspect = imgAspect,
                                             .screenAspect = static_cast<float>(extent.width) / static_cast<float>(extent.height),
                                             .mipLevel = imageVisMipLevel},
                       vk::AttachmentLoadOp::eLoad);
}

void Renderer::recordShadowPass(vk::raii::CommandBuffer& cmd, Light& light) {

    uint32_t shadowMapResolution = light.shadowResolution;

    auto& currentPipeline = pipelineManager->getBeforeGeoPipelines()[shadowPipelineIndex];
    vk::Buffer indexBufferHandle = descriptorSet->getVariableBuffer(indexBufferIndex);
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, currentPipeline->pipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, currentPipeline->layout, 0, {*currentPipeline->descriptorSet}, {});

    // Build indirect draw commands once — identical across all faces/cascades since culling is off
    std::array<Plane, 6> dummyPlanes{};
    drawDataList.clear();
    buildGeometryDrawCommands(dummyPlanes, false, [&](const auto& subMesh, auto& node, const auto& material) {
        drawDataList.push_back({.vertexAllocationIndex = subMesh.vertexAllocationIndex,
                                .vertexOffset = static_cast<uint32_t>(subMesh.vertexOffset),
                                .vertexStride = subMesh.vertexStride,
                                .modelMatrixIndex = node.getModelMatrixIndex()});
    });

    vk::DeviceSize frameByteOffset = currentFrame * MAX_INDIRECT_COMMANDS * sizeof(DrawIndexedIndirectCommand);

    // Upload indirect commands and draw data once
    if (!indirectCommands.empty()) {
        memcpy(static_cast<char*>(indirectDrawBufferMapped) + frameByteOffset, indirectCommands.data(), indirectCommands.size() * sizeof(DrawIndexedIndirectCommand));

        auto* shadowDataBuffer = descriptorSet->getFixedBufferMappedData<ShadowDrawData>(shadowDrawDataBufferIndex);
        if (shadowDataBuffer) {
            uint32_t frameOffset = currentFrame * MAX_FIXED_BUFFER;
            memcpy(&shadowDataBuffer[frameOffset], drawDataList.data(), drawDataList.size() * sizeof(ShadowDrawData));
        } else {
            std::cerr << "Error: shadow data buffer mapped memory is null!" << std::endl;
        }
    }

    // Determine face count and get shadow map + matrix per face
    uint32_t faceCount = 0;
    if (light.type == LightType::Directional) {
        faceCount = light.numCascades;
    } else if (light.type == LightType::Point) {
        faceCount = 6;
    }

    for (uint32_t i = 0; i < faceCount; i++) {
        TextureResource* shadowMap = nullptr;
        glm::mat4 lightSpaceMatrix;

        if (light.type == LightType::Directional) {
            shadowMap = &descriptorSet->getTextureResource(light.cascades[i].shadowMapIndex);
            lightSpaceMatrix = light.cascades[i].lightSpaceMatrix;
        } else if (light.type == LightType::Point) {
            shadowMap = &descriptorSet->getTextureResource(light.cubeMapIndices[i].shadowMapIndex);
            lightSpaceMatrix = light.cubeMapIndices[i].lightSpaceMatrix;
        }

        resourceManager->transitionImageLayout(&cmd, *shadowMap->image, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eDepthStencilAttachmentOptimal);

        vk::ClearValue clearDepth{.depthStencil = vk::ClearDepthStencilValue{1.0f, 0}};
        vk::RenderingAttachmentInfo depthAttachment{.imageView = *shadowMap->imageView,
                                                    .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                                    .loadOp = vk::AttachmentLoadOp::eClear,
                                                    .storeOp = vk::AttachmentStoreOp::eStore,
                                                    .clearValue = clearDepth};

        vk::RenderingInfo renderInfo{.renderArea = {{0, 0}, {shadowMapResolution, shadowMapResolution}},
                                     .layerCount = 1,
                                     .colorAttachmentCount = 0,
                                     .pColorAttachments = nullptr,
                                     .pDepthAttachment = &depthAttachment};

        cmd.beginRendering(renderInfo);
        cmd.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(shadowMapResolution), static_cast<float>(shadowMapResolution), 0.0f, 1.0f));
        cmd.setScissor(0, vk::Rect2D({0, 0}, {shadowMapResolution, shadowMapResolution}));

        if (!indirectCommands.empty()) {
            ShadowPushConstants pushConstants = {
                .vertexBufferAddress   = descriptorSet->getVariableBuffers()[vertexBufferIndex]->address,
                .modelMatricesAddress  = descriptorSet->getFixedBuffers()[modelMatrixBufferIndex]->address + static_cast<vk::DeviceSize>(currentFrame) * MAX_FIXED_BUFFER * sizeof(glm::mat4),
                .shadowDrawDataAddress = descriptorSet->getFixedBuffers()[shadowDrawDataBufferIndex]->address + static_cast<vk::DeviceSize>(currentFrame) * MAX_FIXED_BUFFER * sizeof(ShadowDrawData),
                .lightSpaceMatrix      = lightSpaceMatrix};
            cmd.pushConstants<ShadowPushConstants>(*currentPipeline->layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pushConstants);

            cmd.bindIndexBuffer(indexBufferHandle, 0, vk::IndexType::eUint32);
            cmd.drawIndexedIndirect(*indirectDrawBuffer, frameByteOffset, static_cast<uint32_t>(indirectCommands.size()), sizeof(DrawIndexedIndirectCommand));
        }
        cmd.endRendering();

        resourceManager->transitionImageLayout(&cmd, *shadowMap->image, vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
    }
}

void Renderer::createOrResizeRenderTarget(uint32_t& index, uint32_t width, uint32_t height, vk::Format format, const char* debugName,
                                          vk::ImageUsageFlags extraUsage) {
    if (index != 0xFFFFFFFF) {
        descriptorSet->freeTexture(index);
    }
    vk::raii::Image image = nullptr;
    vk::raii::DeviceMemory memory = nullptr;
    resourceManager->createImage(width, height, 1, vk::SampleCountFlagBits::e1, format, vk::ImageTiling::eOptimal,
                                 vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled | extraUsage,
                                 vk::MemoryPropertyFlagBits::eDeviceLocal, image, memory, 1);
    auto view = resourceManager->createImageView(image, format, vk::ImageAspectFlagBits::eColor);
    resourceManager->transitionImageLayout(nullptr, image, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal);
    index = descriptorSet->allocateTexture(std::move(image), std::move(memory), std::move(view), debugName, false, width, height);
}

void Renderer::createOrResizeMSAATarget(Image& target, uint32_t width, uint32_t height, vk::Format format) {
    target.view = nullptr; // destroy view before image
    resourceManager->createImage(width, height, 1, msaaSamples, format, vk::ImageTiling::eOptimal,
                                 vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment,
                                 vk::MemoryPropertyFlagBits::eDeviceLocal, target.image, target.memory);
    target.view = resourceManager->createImageView(target.image, format, vk::ImageAspectFlagBits::eColor, 1);
    resourceManager->transitionImageLayout(nullptr, target.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal);
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
