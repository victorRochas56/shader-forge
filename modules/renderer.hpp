#pragma once
#ifndef VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1 // for raii
#endif
#ifndef VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_NO_CONSTRUCTORS 1 // for structs constructors
#endif

#include <cstdlib>
#include <cstring>
#include <random>
#include <fstream>
#include <iostream>
#include <map>
#include <stb_image.h>
#include <stdexcept>
#include <unordered_set>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#define GLFW_INCLUDE_VULKAN // REQUIRED only for GLFW CreateWindowSurface.
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#define GLM_FORCE_RADIANS
#define GLM_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#include "constants.hpp"
#include "descriptor_sets.hpp"
#include "devices.hpp"
#include "gizmo.hpp"
#include "pipelines.hpp"
#include "scene_elements.hpp"
#include "swapchain.hpp"
#include "asset_manager.hpp"
#include "raycast.hpp"
#include "scene_graph.hpp"
#include "utils.hpp"

/*
main rendering engine holds the state of the scene, nodes, meshes, lights, materials etc...
handles vulkan initialization and the main render loop
*/

// TODO gpu side material data ("uber shader" approach)
const std::vector validationLayers = {"VK_LAYER_KHRONOS_validation"};

class Renderer {
  public:
    Camera                              activeCamera;
    Gizmos*                             gizmos = nullptr;
    AssetManager                        assetManager;
    SceneGraph                          sceneGraph;

  private:
    GLFWwindow*                         window = nullptr;
    bool                                framebufferResized = false;
    vk::raii::Instance                  instance = nullptr;
    vk::raii::Context                   context;
    vk::raii::DebugUtilsMessengerEXT    debugMessenger = nullptr;
    vk::raii::SurfaceKHR                surface = nullptr;
    vk::SampleCountFlagBits             msaaSamples;

    std::vector<const char*>            requiredDeviceExtension = {VK_KHR_SWAPCHAIN_EXTENSION_NAME,           VK_KHR_SPIRV_1_4_EXTENSION_NAME,
                                                                   VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,   VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME,
                                                                   VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME};

    vk::raii::CommandPool               commandPool = nullptr;
    std::vector<vk::raii::CommandBuffer>commandBuffers;
    uint32_t                            graphicsIndex = 0;

    //synchronization objects
    std::vector<vk::raii::Semaphore>    presentCompleteSemaphores;
    std::vector<vk::raii::Semaphore>    renderFinishedSemaphores;
    std::vector<vk::raii::Fence>        inFlightFences;
    std::vector<vk::Fence>              imagesInFlight;
    uint32_t                            currentFrame = 0;

    //my classes
    std::unique_ptr<Device>             device;
    std::unique_ptr<Swapchain>          swapchain;
    std::unique_ptr<PipelineManager>    pipelineManager;
    std::unique_ptr<ResourceManager>    resourceManager;
    std::unique_ptr<DescriptorSet>      descriptorSet;

    //pipelines
    uint32_t                            skyboxPipelineIndex;
    uint32_t                            shadowPipelineIndex;
    uint32_t                            litPipelineIndex;
    uint32_t                            gizmoPipelineIndex;
    uint32_t                            depthPipelineIndex;
    uint32_t                            blurPipelineIndex;

    //defaults
    uint32_t                            defaultSamplerIndex;
    uint32_t                            depthSamplerIndex;
    uint32_t                            shadowSamplerIndex;
    uint32_t                            defaultNormalIndex;
    uint32_t                            skyboxIndex;

    //rendering data
    std::vector<Material>               materials;                                                             
    std::map<Shader, std::map<Material, std::map<Node*, std::unordered_set<uint32_t>>>> shaders; // map between Shaders and Nodes + their submeshes to render
    Shader                              fallbackLitShader;
    uint32_t                            fallbackDefaultMaterialIndex;
    std::map<uint32_t, Light>           lights; 
    uint32_t                            vertexBufferIndex;
    uint32_t                            indexBufferIndex;
    uint32_t                            modelMatrixBufferIndex;
    uint32_t                            lightBufferIndex;
    uint32_t                            shadowDrawDataBufferIndex;

    // Persistent buffers for indirect drawing in shadow map
    vk::raii::Buffer                    indirectDrawBuffer = nullptr;
    vk::raii::DeviceMemory              indirectDrawBufferMemory = nullptr;

    // Reusable staging vectors for shadow pass indirect draw building (cleared per cascade)
    std::vector<DrawIndexedIndirectCommand> indirectCommands;
    std::vector<ShadowDrawData>             drawDataList;

    vk::raii::Image                     shadowDepthImage = nullptr;
    vk::raii::DeviceMemory              shadowDepthMemory = nullptr;
    vk::raii::ImageView                 shadowDepthView = nullptr;
    uint32_t                            currentShadowDepthResolution = 0;

    //SSAO
    uint32_t                            ssaoTextureIndex = 0xFFFFFFFF;
    uint32_t                            ssaoBlurTextureIndex = 0xFFFFFFFF;
    uint32_t                            ssaoNoiseTextureIndex = 0xFFFFFFFF;
    uint32_t                            ssaoNoiseSamplerIndex = 0xFFFFFFFF;
    uint32_t                            ssaoPipelineIndex = 0xFFFFFFFF;
    uint32_t                            ssaoApplyPipelineIndex = 0xFFFFFFFF;

  public:
    bool                                enableSSAO = true; 
    float                               ssaoRadius = 0.3f;
    float                               ssaoBias = 0.1f;
    float                               ssaoPower = 2.0f;

  private:
    // Temporary texture for gaussian blur
    uint32_t                            tempBlurTextureIndex = 0xFFFFFFFF;

    bool                                vSync = true;
    bool                                depthView = false;

  public:

    Renderer() = default;

    void initVulkan(uint32_t startWidth, uint32_t startHeight) {
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
                              .farPlane = 100.0};
        activeCamera.calculateViewProjectionMatrix();

        /////=====================================DESCRIPTOR SET BUFFERS=================================================/////
        vertexBufferIndex = descriptorSet->createVariableBuffer(256 * 1024 * 1024);                                       // 256 mb vertex buffer
        indexBufferIndex = descriptorSet->createVariableBuffer(128 * 1024 * 1024, vk::BufferUsageFlagBits::eIndexBuffer); // index buffer (128 MB)
        assetManager.init(resourceManager.get(), descriptorSet.get(), vertexBufferIndex, indexBufferIndex);

        // these buffers store the data once per frame in flight since they are usually accessed every frame by the CPU
        modelMatrixBufferIndex = descriptorSet->createFixedBuffer<glm::mat4>(MAX_FRAMES_IN_FLIGHT * MAX_FIXED_BUFFER, true);
        lightBufferIndex = descriptorSet->createFixedBuffer<Light>(MAX_FRAMES_IN_FLIGHT * MAX_FIXED_BUFFER, true);
        shadowDrawDataBufferIndex = descriptorSet->createFixedBuffer<ShadowDrawData>(MAX_FRAMES_IN_FLIGHT * MAX_FIXED_BUFFER, true);

        // sets the frame offsets for each buffer
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            descriptorSet->setBufferFrameOffset(modelMatrixBufferIndex, i, MAX_FIXED_BUFFER * i);
            descriptorSet->setBufferFrameOffset(lightBufferIndex, i, MAX_FIXED_BUFFER * i);
            descriptorSet->setBufferFrameOffset(shadowDrawDataBufferIndex, i, MAX_FIXED_BUFFER * i);
        }

        gizmos = new Gizmos(MAX_GIZMO_LINES, &*descriptorSet);

        // indirect draw buffer for shadows
        std::tie(indirectDrawBuffer, indirectDrawBufferMemory) = resourceManager->createIndirectDrawBuffer();

        // after having created all our desire buffers we can initialize the descriptor set
        descriptorSet->createDescriptorSet();

        swapchain->create(*window, vSync);
        createShadowDepthBuffer(DEFAULT_SHADOW_RESOLUTION);
        createSSAOResources(startWidth, startHeight);
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

        shadowPipelineIndex = pipelineManager->createPipeline<ShadowPushConstants>(PipelineCategory::BEFORE_GEOMETRY, vk::PrimitiveTopology::eTriangleList,
                                                                                   vk::CullModeFlagBits::eNone, vk::True, vk::True, "shaders/shadow_geometry.spv",
                                                                                   descriptorSet->getDescriptorSetLayout(), descriptorSet->getDescriptorSet());
        blurPipelineIndex =
            pipelineManager->createPipeline<BlurPushConstants>(PipelineCategory::BEFORE_GEOMETRY, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone, vk::False,
                                                               vk::False, "shaders/blur.spv", descriptorSet->getDescriptorSetLayout(), descriptorSet->getDescriptorSet());

        litPipelineIndex = pipelineManager->createPipeline<PushConstants>(PipelineCategory::GEOMETRY, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eBack, vk::True,
                                                                          vk::True, "shaders/lit.spv", descriptorSet->getDescriptorSetLayout(), descriptorSet->getDescriptorSet());
        gizmoPipelineIndex =
            pipelineManager->createPipeline<LinePushConstants>(PipelineCategory::POSTPROCESS, vk::PrimitiveTopology::eLineList, vk::CullModeFlagBits::eNone, vk::False, vk::False,
                                                               "shaders/line.spv", descriptorSet->getDescriptorSetLayout(), descriptorSet->getDescriptorSet());
        depthPipelineIndex =
            pipelineManager->createPipeline<ImageVisPushConstants>(PipelineCategory::POSTPROCESS, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone, vk::False,
                                                                   vk::False, "shaders/depth_view.spv", descriptorSet->getDescriptorSetLayout(), descriptorSet->getDescriptorSet());

        // default litshader / material
        fallbackLitShader = Shader{.sourceFile = "shaders/lit.spv", .pipelineIndex = litPipelineIndex};
        MaterialFlags defaultTexMask = MaterialFlags::NONE; // see the material struct definition
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
    }

    // main render loop
    void drawFrame() {
        device->getDevice().waitForFences(*inFlightFences[currentFrame], vk::True, UINT64_MAX);

        auto [result, imageIndex] = swapchain->getSwapChain().acquireNextImage(UINT64_MAX, *presentCompleteSemaphores[currentFrame], nullptr);

        if (result == vk::Result::eErrorOutOfDateKHR) {
            swapchain->recreate(window, vSync);
            int width = 0, height = 0;
            glfwGetFramebufferSize(window, &width, &height);
            if (width > 0 && height > 0) createSSAOResources(width, height);
            activeCamera.aspectRatio = static_cast<float>(width) / static_cast<float>(height);
            activeCamera.calculateViewProjectionMatrix();
            return;
        }
        if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }
        if (imagesInFlight[imageIndex] != VK_NULL_HANDLE) {
            vk::Result waitResult = device->getDevice().waitForFences(imagesInFlight[imageIndex], vk::True, UINT64_MAX);
            if (waitResult != vk::Result::eSuccess) {
                throw std::runtime_error("failed to wait for image fence!");
            }
        }

        imagesInFlight[imageIndex] = *inFlightFences[currentFrame];
        device->getDevice().resetFences(*inFlightFences[currentFrame]);

        for (auto& [id, light] : lights) {
            if (light.castsShadows == 1) {
                if (light.type == LightType::Directional) {
                    calculateCascadedLightSpaceMatrices(light, activeCamera, this);
                    descriptorSet->updateFixedBufferWithOffset<Light>(lightBufferIndex, id, light, currentFrame);
                }
            }
        }

        commandBuffers[currentFrame].reset();
        recordCommandBuffer(imageIndex);

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

            int width = 0, height = 0;
            glfwGetFramebufferSize(window, &width, &height);
            if (width > 0 && height > 0) createSSAOResources(width, height);
            activeCamera.aspectRatio = static_cast<float>(width) / static_cast<float>(height);
            activeCamera.calculateViewProjectionMatrix();

        } else if (result != vk::Result::eSuccess) {
            throw std::runtime_error("failed to present swap chain image!");
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
        pipelineManager->checkForShaderUpdates(); // TODO enable this
    }

    GLFWwindow* getWindow() { return window; }
    void setWindow(GLFWwindow* pWindow) { window = pWindow; }

    const vk::Instance& getInstance() const { return *instance; }

    Device& getDevice() { return *device; }
    ResourceManager& getResourceManager() { return *resourceManager; }
    DescriptorSet& getDescriptorSet() { return *descriptorSet; }

    Swapchain& getSwapchain() { return *swapchain; }
    void cleanupSwapchain() { swapchain->cleanupSwapChain(); }
    const vk::SampleCountFlagBits& getMsaaSamples() const { return msaaSamples; }
    
    const uint32_t getGraphicsIndex() const { return graphicsIndex; } // only used for IMGUI init

    uint32_t getModelMatrixBufferIndex() { return modelMatrixBufferIndex; }
    uint32_t getLightBufferIndex() { return lightBufferIndex; }
    uint32_t getShadowDrawDataBufferIndex() { return shadowDrawDataBufferIndex; }

    std::vector<Material>& getMaterials() { return materials; }
    uint32_t addMaterial(Material material) {
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

    // don't call this directly, should only be called from a node with a valid mesh index
    void addMeshToShader(Node* node, uint32_t submeshIndex, Shader shader, Material material) { shaders[shader][material][node].insert(submeshIndex); }
    void removeMeshFromShader(Node* node, uint32_t subMeshIndex, Shader shader, Material material) { shaders[shader][material][node].erase(subMeshIndex); }

    Shader getFallBackShader() { return fallbackLitShader; }
    uint32_t getFallBackMaterial() { return fallbackDefaultMaterialIndex; }
    void clearRenderList() { shaders.clear(); }

    const std::map<uint32_t, Light>& getLights() { return lights; }
    void addLight(uint32_t index, Light light) { lights[index] = light; }
    Light& getLight(uint32_t index) { return lights[index]; }
    void clearLights() {
        descriptorSet->clearFixedBuffer(lightBufferIndex);
        lights.clear();
    }


    //toggled by keyboard inputs
    void toggleVsync() {
        vSync = !vSync;
        swapchain->recreate(window, vSync);
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        if (width > 0 && height > 0) createSSAOResources(width, height);
    }
    
    void toggleSSAO() { enableSSAO = !enableSSAO; }
    
    void toggleDepthView() { depthView = !depthView; }
    
    void setSkyBox(uint32_t skyboxIndex) { this->skyboxIndex = skyboxIndex; }

    // Generic blur pass that can blur any attachment
    // Performs two-pass separable Gaussian blur (horizontal + vertical)
    // Requires a temporary texture of the same size and format as the source
    void blurAttachment(vk::raii::CommandBuffer& cmd, uint32_t sourceTextureIndex, uint32_t tempTextureIndex, uint32_t width, uint32_t height, float blurRadius = 1.0f,
                        uint32_t samplerIndex = 0) {

        auto& blurPipeline = pipelineManager->getBeforeGeoPipelines()[blurPipelineIndex];
        auto& sourceTexture = descriptorSet->getTextureResource(sourceTextureIndex);
        auto& tempTexture = descriptorSet->getTextureResource(tempTextureIndex);

        // === Horizontal blur (source -> temp) ===
        {
            resourceManager->transitionImageLayout(&cmd, *tempTexture.image, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eColorAttachmentOptimal);

            vk::ClearValue clearColor{.color = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f})};
            vk::RenderingAttachmentInfo colorAttachment{.imageView = *tempTexture.imageView,
                                                        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                        .loadOp = vk::AttachmentLoadOp::eClear,
                                                        .storeOp = vk::AttachmentStoreOp::eStore,
                                                        .clearValue = clearColor};

            vk::RenderingInfo renderInfo{.renderArea = {{0, 0}, {width, height}}, .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &colorAttachment};

            cmd.beginRendering(renderInfo);
            cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, blurPipeline->pipeline);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, blurPipeline->layout, 0, {*blurPipeline->descriptorSet}, {});
            cmd.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f));
            cmd.setScissor(0, vk::Rect2D({0, 0}, {width, height}));

            BlurPushConstants pushConstants{
                .inputTextureIndex = sourceTextureIndex, .samplerIndex = samplerIndex, .isHorizontal = 1, .blurRadius = blurRadius, .resolution = glm::uvec2(width, height)};

            cmd.pushConstants<BlurPushConstants>(*blurPipeline->layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pushConstants);
            cmd.draw(3, 1, 0, 0);
            cmd.endRendering();

            resourceManager->transitionImageLayout(&cmd, *tempTexture.image, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
        }

        // === Vertical blur (temp -> source) ===
        {
            resourceManager->transitionImageLayout(&cmd, *sourceTexture.image, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eColorAttachmentOptimal);

            vk::ClearValue clearColor{.color = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f})};
            vk::RenderingAttachmentInfo colorAttachment{.imageView = *sourceTexture.imageView,
                                                        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                        .loadOp = vk::AttachmentLoadOp::eClear,
                                                        .storeOp = vk::AttachmentStoreOp::eStore,
                                                        .clearValue = clearColor};

            vk::RenderingInfo renderInfo{.renderArea = {{0, 0}, {width, height}}, .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &colorAttachment};

            cmd.beginRendering(renderInfo);
            cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, blurPipeline->pipeline);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, blurPipeline->layout, 0, {*blurPipeline->descriptorSet}, {});
            cmd.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f));
            cmd.setScissor(0, vk::Rect2D({0, 0}, {width, height}));

            BlurPushConstants pushConstants{
                .inputTextureIndex = tempTextureIndex, .samplerIndex = samplerIndex, .isHorizontal = 0, .blurRadius = blurRadius, .resolution = glm::uvec2(width, height)};

            cmd.pushConstants<BlurPushConstants>(*blurPipeline->layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pushConstants);
            cmd.draw(3, 1, 0, 0);
            cmd.endRendering();

            resourceManager->transitionImageLayout(&cmd, *sourceTexture.image, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
        }
    }

    // debugging shadows TODO : make access cleaner
    uint32_t debugShowShadow = 3;
    uint32_t showShadowMapIndex = 0xFFFFFFFF;
    void showShadowMap() {
        debugShowShadow++;
        if (debugShowShadow == 3) {
            showShadowMapIndex = 0xFFFFFFFF;
            return;
        }
        if (debugShowShadow >= 4) {
            debugShowShadow = 0;
            return;
        }
    }


  private:
/////=================================================INITIALIZATION HELPER FUNCTIONS=================================================/////
    void createInstance() {
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

    std::vector<const char*> getRequiredExtensions() {
        uint32_t glfwExtensionCount = 0;
        auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

        return extensions;
    }

    void setupDebugMessenger() {
        vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                                                            vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
        vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags(vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
                                                           vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
        vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT{.messageSeverity = severityFlags, .messageType = messageTypeFlags, .pfnUserCallback = &debugCallback};
        debugMessenger = instance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
    }

    static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity, vk::DebugUtilsMessageTypeFlagsEXT type,
                                                          const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData, void*) {
        std::cerr << "validation layer: type " << to_string(type) << " msg: " << pCallbackData->pMessage << std::endl;
        return vk::False;
    }

    void createSurface() {
        VkSurfaceKHR _surface;
        if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
        surface = vk::raii::SurfaceKHR(instance, _surface);
    }

    void createCommandPool() {
        vk::CommandPoolCreateInfo poolInfo{.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer, .queueFamilyIndex = graphicsIndex};
        commandPool = vk::raii::CommandPool(device->getDevice(), poolInfo);
    }

    void createCommandBuffers() {
        commandBuffers.clear();
        vk::CommandBufferAllocateInfo allocInfo{.commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = MAX_FRAMES_IN_FLIGHT};
        commandBuffers = vk::raii::CommandBuffers(device->getDevice(), allocInfo);
    }

    void createShadowDepthBuffer(uint32_t resolution) {
        // only recreates if resolution changed
        if (currentShadowDepthResolution == resolution && shadowDepthView != nullptr) {
            return;
        }
        // Free old resources if they exist
        shadowDepthView = nullptr;
        shadowDepthImage = nullptr;
        shadowDepthMemory = nullptr;
        resourceManager->createImage(resolution, resolution, 1, vk::SampleCountFlagBits::e1, vk::Format::eD32Sfloat, vk::ImageTiling::eOptimal,
                                     vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal, shadowDepthImage, shadowDepthMemory, 1);
        shadowDepthView = resourceManager->createImageView(shadowDepthImage, vk::Format::eD32Sfloat, vk::ImageAspectFlagBits::eDepth);
        resourceManager->transitionImageLayout(nullptr, shadowDepthImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal);
        currentShadowDepthResolution = resolution;
    }

    void createSSAOResources(uint32_t width, uint32_t height) {
        // Free previous SSAO textures if they exist (resize case)
        if (ssaoTextureIndex != 0xFFFFFFFF) {
            descriptorSet->freeTexture(ssaoTextureIndex);
            ssaoTextureIndex = 0xFFFFFFFF;
        }
        if (ssaoBlurTextureIndex != 0xFFFFFFFF) {
            descriptorSet->freeTexture(ssaoBlurTextureIndex);
            ssaoBlurTextureIndex = 0xFFFFFFFF;
        }

        // SSAO render target (R8 single-channel)
        vk::raii::Image ssaoImage = nullptr;
        vk::raii::DeviceMemory ssaoMemory = nullptr;
        resourceManager->createImage(width, height, 1, vk::SampleCountFlagBits::e1, vk::Format::eR8Unorm, vk::ImageTiling::eOptimal,
                                     vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, ssaoImage,
                                     ssaoMemory, 1);
        auto ssaoView = resourceManager->createImageView(ssaoImage, vk::Format::eR8Unorm, vk::ImageAspectFlagBits::eColor);
        resourceManager->transitionImageLayout(nullptr, ssaoImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal);
        ssaoTextureIndex = descriptorSet->allocateTexture(std::move(ssaoImage), std::move(ssaoMemory), std::move(ssaoView), "internal/ssao");

        // SSAO blur temporary target (same format/size)
        vk::raii::Image blurImage = nullptr;
        vk::raii::DeviceMemory blurMemory = nullptr;
        resourceManager->createImage(width, height, 1, vk::SampleCountFlagBits::e1, vk::Format::eR8Unorm, vk::ImageTiling::eOptimal,
                                     vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, blurImage,
                                     blurMemory, 1);
        auto blurView = resourceManager->createImageView(blurImage, vk::Format::eR8Unorm, vk::ImageAspectFlagBits::eColor);
        resourceManager->transitionImageLayout(nullptr, blurImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal);
        ssaoBlurTextureIndex = descriptorSet->allocateTexture(std::move(blurImage), std::move(blurMemory), std::move(blurView), "internal/ssao_blur");

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

    void createSyncObjects() {
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

/////=================================================MAIN RENDERING LOGIC=================================================/////


    void recordCommandBuffer(uint32_t imageIndex) {
        auto& cmd = commandBuffers[currentFrame];
        cmd.begin({});

        for (auto& [lightId, light] : lights) {
            if (light.castsShadows == 1)
                recordShadowPass(cmd, light);
        }

        recordGeometryPass(cmd, imageIndex);

        if (enableSSAO && ssaoPipelineIndex != 0xFFFFFFFF)
            recordSSAOPass(cmd, imageIndex);

        recordOverlayPass(cmd, imageIndex);

        if (depthView || debugShowShadow < 4)
            recordDepthVisPass(cmd, imageIndex);

        resourceManager->transitionImageLayout(&cmd, swapchain->getSwapChainImages()[imageIndex], vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR);
        cmd.end();
    }

    void recordGeometryPass(vk::raii::CommandBuffer& cmd, uint32_t imageIndex) {
        resourceManager->transitionImageLayout(&cmd, swapchain->getSwapChainImages()[imageIndex], vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal);
        resourceManager->transitionImageLayout(&cmd, swapchain->getColorImage(), vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal);
        resourceManager->transitionImageLayout(&cmd, swapchain->getDepthImage(), vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal);

        vk::ClearValue clearColor{.color = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f})};
        vk::ClearValue clearDepth{.depthStencil = vk::ClearDepthStencilValue{1.0f, 0}};

        vk::RenderingAttachmentInfo colorAttachment = {.imageView = swapchain->getColorImageView(),
                                                       .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                       .resolveMode = vk::ResolveModeFlagBits::eAverage,
                                                       .resolveImageView = swapchain->getSwapChainImageViews()[imageIndex],
                                                       .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                       .loadOp = vk::AttachmentLoadOp::eClear,
                                                       .storeOp = vk::AttachmentStoreOp::eStore,
                                                       .clearValue = clearColor};

        vk::RenderingAttachmentInfo depthAttachmentInfo = {.imageView = swapchain->getDepthImageView(),
                                                           .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                                           .resolveMode = vk::ResolveModeFlagBits::eMin,
                                                           .resolveImageView = swapchain->getDepthResolveImageView(),
                                                           .resolveImageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                                           .loadOp = vk::AttachmentLoadOp::eClear,
                                                           .storeOp = vk::AttachmentStoreOp::eDontCare,
                                                           .clearValue = clearDepth};

        vk::RenderingInfo renderingInfo = {.renderArea = {.offset = {0, 0}, .extent = swapchain->getSwapChainExtent()},
                                           .layerCount = 1,
                                           .colorAttachmentCount = 1,
                                           .pColorAttachments = &colorAttachment,
                                           .pDepthAttachment = &depthAttachmentInfo};

        cmd.beginRendering(renderingInfo);
        setFullscreenViewport(cmd, swapchain->getSwapChainExtent());

        // skybox
        auto& skyboxPipeline = pipelineManager->getGeoPipelines()[skyboxPipelineIndex];
        bindPipeline(cmd, *skyboxPipeline);
        SkyBoxPushConstants skyboxConstants = {.skyboxIndex = skyboxIndex, .blur = 0.5, .invViewProjMatrix = glm::inverse(activeCamera.viewProjection)};
        cmd.pushConstants<SkyBoxPushConstants>(*skyboxPipeline->layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, skyboxConstants);
        cmd.draw(3, 1, 0, 0);

        // lit geometry
        vk::Buffer indexBufferHandle = descriptorSet->getVariableBuffer(indexBufferIndex);
        cmd.bindIndexBuffer(indexBufferHandle, 0, vk::IndexType::eUint32);
        auto& geoPipelines = pipelineManager->getGeoPipelines();
        for (const auto& [shader, materials] : shaders) {
            auto currentPipeline = &(geoPipelines[shader.pipelineIndex]);
            bindPipeline(cmd, **currentPipeline);

            for (const auto& [material, node_mesh] : materials) {
                for (const auto& [node, subMeshIndices] : node_mesh) {

                    if (assetManager.meshes[node->getMeshIndex()].freed == true) {
                        for (uint32_t subMesh : assetManager.meshes[node->getMeshIndex()].subMeshes) {
                            descriptorSet->freeVariableBuffer(vertexBufferIndex, assetManager.subMeshes[subMesh].vertexAllocationIndex);
                            descriptorSet->freeVariableBuffer(indexBufferIndex, assetManager.subMeshes[subMesh].indexAllocationIndex);
                            assetManager.freeSubMeshes.push(subMesh);
                        }
                        assetManager.freeMeshes.push(node->getMeshIndex());
                        continue;
                    }
                    for (auto mesh : subMeshIndices) {
                        const auto& subMesh = assetManager.subMeshes[mesh];
                        PushConstants pushConstants = {.vertexAllocationIndex = subMesh.vertexAllocationIndex,
                                                       .vertexOffset = static_cast<uint32_t>(subMesh.vertexOffset),
                                                       .vertexStride = subMesh.vertexStride,
                                                       .modelMatrixIndex = node->getModelMatrixIndex(),
                                                       .albedoTextureIndex = material.albedoTextureIndex,
                                                       .roughnessTextureIndex = material.roughnessTextureIndex,
                                                       .metallicTextureIndex = material.metallicTextureIndex,
                                                       .normalTextureIndex = material.normalTextureIndex,
                                                       .environmentMapIndex = material.environmentMapIndex,
                                                       .samplerIndex = defaultSamplerIndex,
                                                       .lightCount = static_cast<uint32_t>(lights.size()),
                                                       .shadowSamplerIndex = shadowSamplerIndex,
                                                       .cameraPosition = activeCamera.position,
                                                       .materialFlags = material.flags,
                                                       .elementOffsetModel = currentFrame * MAX_FIXED_BUFFER,
                                                       .elementOffsetLight = currentFrame * MAX_FIXED_BUFFER,
                                                       .metallic = material.metallic,
                                                       .roughness = material.roughness,
                                                       .viewProjection = activeCamera.viewProjection};

                        cmd.pushConstants<PushConstants>((*currentPipeline)->layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pushConstants);
                        cmd.drawIndexed(subMesh.indexCount, 1, static_cast<uint32_t>(subMesh.indexOffset / sizeof(uint32_t)), 0, 0);
                    }
                }
            }
        }

        cmd.endRendering();
    }

    void recordOverlayPass(vk::raii::CommandBuffer& cmd, uint32_t imageIndex) {
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
        auto& gizmoPipeline = pipelineManager->getPostProcessPipelines()[gizmoPipelineIndex];
        bindPipeline(cmd, *gizmoPipeline);
        LinePushConstants lineConstants = {.viewProjection = activeCamera.viewProjection};
        cmd.pushConstants<LinePushConstants>(*gizmoPipeline->layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, lineConstants);
        cmd.draw(gizmos->getVertexCount(), 1, 0, 0);

        // GUI
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), *cmd);
        cmd.endRendering();
    }

    void recordSSAOPass(vk::raii::CommandBuffer& cmd, uint32_t imageIndex) {
        auto extent = swapchain->getSwapChainExtent();
        auto& ssaoTexture = descriptorSet->getTextureResource(ssaoTextureIndex);

        // Transition depth to readable, SSAO target to color attachment
        resourceManager->transitionImageLayout(&cmd, swapchain->getDepthImage(),
            vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
        resourceManager->transitionImageLayout(&cmd, *ssaoTexture.image,
            vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eColorAttachmentOptimal);

        // Render SSAO
        vk::ClearValue ssaoClear{.color = vk::ClearColorValue(std::array<float, 4>{1.0f, 1.0f, 1.0f, 1.0f})};
        vk::RenderingAttachmentInfo ssaoAttachment{.imageView = *ssaoTexture.imageView,
                                                    .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                    .loadOp = vk::AttachmentLoadOp::eClear,
                                                    .storeOp = vk::AttachmentStoreOp::eStore,
                                                    .clearValue = ssaoClear};
        vk::RenderingInfo ssaoRenderInfo{.renderArea = {{0, 0}, extent}, .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &ssaoAttachment};

        auto& ssaoPipeline = pipelineManager->getPostProcessPipelines()[ssaoPipelineIndex];
        cmd.beginRendering(ssaoRenderInfo);
        setFullscreenViewport(cmd, extent);
        bindPipeline(cmd, *ssaoPipeline);
        SSAOPushConstants ssaoPC{.invProjection = glm::inverse(activeCamera.projectionMatrix),
                                 .depthIndex = swapchain->getDepthResolveIndex(),
                                 .depthSamplerIndex = depthSamplerIndex,
                                 .noiseIndex = ssaoNoiseTextureIndex,
                                 .noiseSamplerIndex = ssaoNoiseSamplerIndex,
                                 .resolution = glm::uvec2(extent.width, extent.height),
                                 .radius = ssaoRadius,
                                 .bias = ssaoBias,
                                 .power = ssaoPower,
                                 .kernelSize = 32};
        cmd.pushConstants<SSAOPushConstants>(*ssaoPipeline->layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, ssaoPC);
        cmd.draw(3, 1, 0, 0);
        cmd.endRendering();

        // Blur
        resourceManager->transitionImageLayout(&cmd, *ssaoTexture.image, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
        blurAttachment(cmd, ssaoTextureIndex, ssaoBlurTextureIndex, extent.width, extent.height, 2.0f, depthSamplerIndex);

        // Apply to swapchain via multiplicative blend
        vk::RenderingAttachmentInfo applyAttachment{.imageView = swapchain->getSwapChainImageViews()[imageIndex],
                                                     .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                     .loadOp = vk::AttachmentLoadOp::eLoad,
                                                     .storeOp = vk::AttachmentStoreOp::eStore};
        vk::RenderingInfo applyRenderInfo{.renderArea = {{0, 0}, extent}, .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &applyAttachment};

        auto& applyPipeline = pipelineManager->getPostProcessPipelines()[ssaoApplyPipelineIndex];
        cmd.beginRendering(applyRenderInfo);
        setFullscreenViewport(cmd, extent);
        bindPipeline(cmd, *applyPipeline);
        SSAOApplyPushConstants applyPC{.ssaoTextureIndex = ssaoTextureIndex, .samplerIndex = depthSamplerIndex};
        cmd.pushConstants<SSAOApplyPushConstants>(*applyPipeline->layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, applyPC);
        cmd.draw(3, 1, 0, 0);
        cmd.endRendering();

        // Transition depth back
        resourceManager->transitionImageLayout(&cmd, swapchain->getDepthImage(), vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eDepthStencilAttachmentOptimal);
    }

    void recordDepthVisPass(vk::raii::CommandBuffer& cmd, uint32_t imageIndex) {
        auto extent = swapchain->getSwapChainExtent();
        vk::RenderingAttachmentInfo swapchainAttachment{.imageView = swapchain->getSwapChainImageViews()[imageIndex],
                                                        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                        .loadOp = vk::AttachmentLoadOp::eLoad,
                                                        .storeOp = vk::AttachmentStoreOp::eStore};
        vk::RenderingInfo renderInfo{.renderArea = {{0, 0}, extent}, .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &swapchainAttachment};

        resourceManager->transitionImageLayout(&cmd, swapchain->getDepthImage(), vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

        auto& depthPipeline = pipelineManager->getPostProcessPipelines()[depthPipelineIndex];
        cmd.beginRendering(renderInfo);
        setFullscreenViewport(cmd, extent);
        bindPipeline(cmd, *depthPipeline);//TODO implement a register of all viewable images, that can be passed to here for simple debug viewing
        ImageVisPushConstants depthConstants = {.depthIndex = swapchain->getDepthResolveIndex(),
                                                .depthSamplerIndex = depthSamplerIndex,
                                                .showShadowMap = showShadowMapIndex,
                                                .shadowMapSamplerIndex = shadowSamplerIndex,
                                                .nearPlane = activeCamera.nearPlane,
                                                .farPlane = activeCamera.farPlane,
                                                .linearize = 1,
                                                .doDepthBuffering = depthView};
        cmd.pushConstants<ImageVisPushConstants>(*depthPipeline->layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, depthConstants);
        cmd.draw(3, 1, 0, 0);
        cmd.endRendering();

        resourceManager->transitionImageLayout(&cmd, swapchain->getDepthImage(), vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eDepthStencilAttachmentOptimal);
    }

    void recordShadowPass(vk::raii::CommandBuffer& cmd, Light& light) {

        if (debugShowShadow < 3 && sceneGraph.selectedNode != MAX_NODES) {
            if (lights[sceneGraph.getNodes()[sceneGraph.selectedNode]->getLightIndex()] == light) {
                showShadowMapIndex = light.cascades[debugShowShadow].shadowMapIndex;
            }
        }
        uint32_t cascadeCount = 1;
        TextureResource* shadowMap = nullptr;
        uint32_t shadowMapResolution = light.shadowResolution;

        // Ensure depth buffer matches the shadow map resolution
        createShadowDepthBuffer(shadowMapResolution);
        if (light.type == LightType::Directional) {
            cascadeCount = light.numCascades;
        } else {
            shadowMap = &descriptorSet->getTextureResource(light.shadowMapIndex);
        }

        auto& currentPipeline = pipelineManager->getBeforeGeoPipelines()[shadowPipelineIndex];
        vk::Buffer indexBufferHandle = descriptorSet->getVariableBuffer(indexBufferIndex);
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, currentPipeline->pipeline);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, currentPipeline->layout, 0, {*currentPipeline->descriptorSet}, {});

        //CSM shadows
        for (int i = 0; i < cascadeCount; i++) {
            if (light.type == LightType::Directional) {
                shadowMap = &descriptorSet->getTextureResource(light.cascades[i].shadowMapIndex);
            }
            resourceManager->transitionImageLayout(&cmd, *shadowMap->image, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eColorAttachmentOptimal);

            vk::ClearValue clearDepthValue{.color = vk::ClearColorValue(std::array<float, 4>{1.0f, 0.0f, 0.0f, 0.0f})};
            vk::RenderingAttachmentInfo colorAttachment{.imageView = *shadowMap->imageView,
                                                        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                        .loadOp = vk::AttachmentLoadOp::eClear,
                                                        .storeOp = vk::AttachmentStoreOp::eStore,
                                                        .clearValue = clearDepthValue};

            vk::ClearValue clearDepth{.depthStencil = vk::ClearDepthStencilValue{1.0f, 0}};
            vk::RenderingAttachmentInfo depthAttachment{.imageView = *shadowDepthView,
                                                        .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                                        .loadOp = vk::AttachmentLoadOp::eClear,
                                                        .storeOp = vk::AttachmentStoreOp::eDontCare,
                                                        .clearValue = clearDepth};

            vk::RenderingInfo renderInfo{.renderArea = {{0, 0}, {shadowMapResolution, shadowMapResolution}},
                                         .layerCount = 1,
                                         .colorAttachmentCount = 1,
                                         .pColorAttachments = &colorAttachment,
                                         .pDepthAttachment = &depthAttachment};

            cmd.beginRendering(renderInfo);
            cmd.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(shadowMapResolution), static_cast<float>(shadowMapResolution), 0.0f, 1.0f));
            cmd.setScissor(0, vk::Rect2D({0, 0}, {shadowMapResolution, shadowMapResolution}));

            // Extract frustum planes for this cascade
            glm::mat4 lightSpaceMatrix = light.lightSpaceMatrix;
            if (light.type == LightType::Directional) {
                lightSpaceMatrix = light.cascades[i].lightSpaceMatrix;
            }
            std::array<Plane, 6> frustumPlanes = extractFrustumPlanes(lightSpaceMatrix);

            // Build indirect draw commands and shadow draw data with CPU frustum culling
            indirectCommands.clear();
            drawDataList.clear();

            //basic culling for performance
            for (const auto& [shader, materials] : shaders) {
                for (const auto& [material, node_mesh] : materials) {
                    for (const auto& [node, subMeshIndices] : node_mesh) {
                        if (assetManager.meshes[node->getMeshIndex()].freed == true) {
                            continue;
                        }

                        // CPU frustum culling: skips node if outside frustum
                        if (node->isBoundingBoxValid()) {
                            if (!isAABBInFrustum(node->getBoundingBoxMin(), node->getBoundingBoxMax(), frustumPlanes)) {
                                continue;
                            }
                        }

                        // Per-submesh frustum culling
                        for (auto mesh : subMeshIndices) {
                            const auto& subMesh = assetManager.subMeshes[mesh];

                            // Add indirect draw command
                            indirectCommands.push_back({.indexCount = subMesh.indexCount,
                                                        .instanceCount = 1,
                                                        .firstIndex = static_cast<uint32_t>(subMesh.indexOffset / sizeof(uint32_t)),
                                                        .vertexOffset = 0,
                                                        .firstInstance = 0});

                            // Add per-draw data
                            drawDataList.push_back({.vertexAllocationIndex = subMesh.vertexAllocationIndex,
                                                    .vertexOffset = static_cast<uint32_t>(subMesh.vertexOffset),
                                                    .vertexStride = subMesh.vertexStride,
                                                    .modelMatrixIndex = node->getModelMatrixIndex()});
                        }
                    }
                }
            }

            // build the indirect draw command and submit it
            if (!indirectCommands.empty()) {
                void* data = indirectDrawBufferMemory.mapMemory(0, indirectCommands.size() * sizeof(DrawIndexedIndirectCommand));
                memcpy(data, indirectCommands.data(), indirectCommands.size() * sizeof(DrawIndexedIndirectCommand));
                indirectDrawBufferMemory.unmapMemory();

                // Get the fixed buffer's mapped memory pointer and write with frame offset
                auto* shadowDataBuffer = descriptorSet->getFixedBufferMappedData<ShadowDrawData>(shadowDrawDataBufferIndex);
                if (shadowDataBuffer) {
                    uint32_t frameOffset = currentFrame * MAX_FIXED_BUFFER;
                    memcpy(&shadowDataBuffer[frameOffset], drawDataList.data(), drawDataList.size() * sizeof(ShadowDrawData));
                } else {
                    std::cerr << "Error: shadow data buffer mapped memory is null!" << std::endl;
                }

                // Set push constants (same for all draws in this cascade)
                ShadowPushConstants pushConstants = {
                    .lightSpaceMatrix = lightSpaceMatrix, .elementOffsetModel = currentFrame * MAX_FIXED_BUFFER, .elementOffsetShadow = currentFrame * MAX_FIXED_BUFFER};
                cmd.pushConstants<ShadowPushConstants>(*currentPipeline->layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pushConstants);

                cmd.bindIndexBuffer(indexBufferHandle, 0, vk::IndexType::eUint32);
                cmd.drawIndexedIndirect(*indirectDrawBuffer, 0, static_cast<uint32_t>(indirectCommands.size()), sizeof(DrawIndexedIndirectCommand));
            }
            cmd.endRendering();

            // Transition shadow map back to shader read-only for fragment shader sampling
            resourceManager->transitionImageLayout(&cmd, *shadowMap->image, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
        }
    }

    void setFullscreenViewport(vk::raii::CommandBuffer& cmd, vk::Extent2D extent) {
        cmd.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.0f, 1.0f));
        cmd.setScissor(0, vk::Rect2D({0, 0}, extent));
    }

    void bindPipeline(vk::raii::CommandBuffer& cmd, PipelineBase& pipeline) {
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.pipeline);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline.layout, 0, {**pipeline.descriptorSet}, {});
    }
};
