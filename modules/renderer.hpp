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
#include "node_ops.hpp"

/*
main rendering engine holds the state of the scene, nodes, meshes, lights, materials etc...
handles vulkan initialization and the main render loop
*/

// TODO if an image has mips, allow them to be viewed in the image viewer
// TODO gpu side material data
// with a vis-buffer
const std::vector validationLayers = {"VK_LAYER_KHRONOS_validation"};

class Renderer {
#pragma region VARS
  public:
    Camera                              activeCamera;
    AssetManager                        assetManager;
    SceneGraph                          sceneGraph;
    uint32_t                            imageVisIndex = 0xFFFFFFFF;
    uint32_t                            imageVisSamplerIndex = 0xFFFFFFFF;
    ImageVisFlags                       imageVisFlags = ImageVisFlags::IMAGE_VIS_NONE;
    int                                 imageVisMipLevel = 0;
    float                               cullFovScale = 1.0f;
    uint32_t                            culledCount = 0;

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
    uint32_t                            imageViewPipelineIndex;
    uint32_t                            blurPipelineIndex;
    uint32_t                            depthPipelineIndex;

    //defaults
    uint32_t                            defaultSamplerIndex;
    uint32_t                            depthSamplerIndex;
    uint32_t                            shadowSamplerIndex;
    uint32_t                            defaultNormalIndex;
    uint32_t                            skyboxIndex = 0;

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
    uint32_t                            litDrawDataBufferIndex;
    uint32_t                            litPassDataBufferIndex;

    // Persistent buffers for indirect drawing
    vk::raii::Buffer                    indirectDrawBuffer = nullptr;      // shadow pass
    vk::raii::DeviceMemory              indirectDrawBufferMemory = nullptr;
    vk::raii::Buffer                    litIndirectDrawBuffer = nullptr;   // lit geometry pass
    vk::raii::DeviceMemory              litIndirectDrawBufferMemory = nullptr;

    // Reusable staging vectors (cleared per draw recording)
    std::vector<DrawIndexedIndirectCommand> indirectCommands;
    std::vector<ShadowDrawData>             drawDataList;
    std::vector<LitDrawData>                litDrawDataList;

    Image                               shadowDepth;
    uint32_t                            currentShadowDepthResolution = 0;

    //SSAO
    uint32_t                            ssaoTextureIndex = 0xFFFFFFFF;
    uint32_t                            ssaoBlurTextureIndex = 0xFFFFFFFF;
    uint32_t                            ssaoNoiseTextureIndex = 0xFFFFFFFF;
    uint32_t                            ssaoNoiseSamplerIndex = 0xFFFFFFFF;
    uint32_t                            ssaoPipelineIndex = 0xFFFFFFFF;
    uint32_t                            ssaoApplyPipelineIndex = 0xFFFFFFFF;

    // Roughness-metallic MRT from lit pass
    uint32_t                            roughnessMetalTextureIndex = 0xFFFFFFFF;
    Image                               roughnessMetal;

    // World-space normals MRT from lit pass
    uint32_t                            normalTextureIndex = 0xFFFFFFFF;
    uint32_t                            normalMipLevels = 1;
    Image                               normalMSAA;

    uint32_t                            motionVectorTextureIndex = 0xFFFFFFFF;
    Image                               motionVectors;

    //SSR
    std::vector<uint32_t>               ssrTextureIndices;
    uint32_t                            ssrIndex = 0;
    uint32_t                            ssrPipelineIndex = 0xFFFFFFFF;
    uint32_t                            ssrApplyPipelineIndex = 0xFFFFFFFF;
    uint32_t                            colorResolveTextureIndex = 0xFFFFFFFF;

    // Hi-Z (hierarchical depth pyramid)
    uint32_t                            hiZTextureIndex = 0xFFFFFFFF;
    uint32_t                            hiZMipLevels = 0;
    uint32_t                            hiZPipelineIndex = 0xFFFFFFFF;
    std::vector<vk::raii::ImageView>    hiZMipViews;

  public:
    bool                                enableSSAO = true;
    float                               ssaoRadius = 0.3f;
    float                               ssaoBias = 0.1f;
    float                               ssaoPower = 2.0f;
    float                               ssaoResolutionScale = 0.5f;

    bool                                enableSSR = true;
    float                               ssrResolutionScale = 1.0f;
    float                               ssrRoughnessThreshold = 0.6f;
    float                               ssrMaxDistance = 10.0f;
    int                                 ssrMaxSteps = 64;
    float                               ssrThickness = 0.5f;

  private:
    // Temporary texture for gaussian blur
    uint32_t                            tempBlurTextureIndex = 0xFFFFFFFF;

    bool                                vSync = true;
    bool                                showBBOXes = false;
#pragma endregion



  public:
  
#pragma region INIT
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
                              .farPlane = 500.0};
        activeCamera.calculateViewProjectionMatrix();

        /////=====================================DESCRIPTOR SET BUFFERS=================================================/////
        vertexBufferIndex = descriptorSet->createVariableBuffer(256 * 1024 * 1024);                                       // 256 mb vertex buffer
        indexBufferIndex = descriptorSet->createVariableBuffer(128 * 1024 * 1024, vk::BufferUsageFlagBits::eIndexBuffer); // index buffer (128 MB)
        assetManager.init(resourceManager.get(), descriptorSet.get(), vertexBufferIndex, indexBufferIndex);

        // these buffers store the data once per frame in flight since they are usually accessed every frame by the CPU
        modelMatrixBufferIndex = descriptorSet->createFixedBuffer<glm::mat4>(MAX_FRAMES_IN_FLIGHT * MAX_FIXED_BUFFER, true);
        lightBufferIndex = descriptorSet->createFixedBuffer<Light>(MAX_FRAMES_IN_FLIGHT * MAX_FIXED_BUFFER, true);
        shadowDrawDataBufferIndex = descriptorSet->createFixedBuffer<ShadowDrawData>(MAX_FRAMES_IN_FLIGHT * MAX_FIXED_BUFFER, true);
        litPassDataBufferIndex = descriptorSet->createFixedBuffer<LitPassData>(MAX_FRAMES_IN_FLIGHT * MAX_FIXED_BUFFER, true);

        // sets the frame offsets for each buffer
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            descriptorSet->setBufferFrameOffset(modelMatrixBufferIndex, i, MAX_FIXED_BUFFER * i);
            descriptorSet->setBufferFrameOffset(lightBufferIndex, i, MAX_FIXED_BUFFER * i);
            descriptorSet->setBufferFrameOffset(shadowDrawDataBufferIndex, i, MAX_FIXED_BUFFER * i);
            descriptorSet->setBufferFrameOffset(litPassDataBufferIndex,i, MAX_FIXED_BUFFER * i);
        }

        Gizmos::init(MAX_GIZMO_LINES, &*descriptorSet);

        litDrawDataBufferIndex = descriptorSet->createFixedBuffer<LitDrawData>(MAX_FRAMES_IN_FLIGHT * MAX_FIXED_BUFFER, true);
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            descriptorSet->setBufferFrameOffset(litDrawDataBufferIndex, i, MAX_FIXED_BUFFER * i);
        }

        // indirect draw buffers (separate for shadow and lit passes)
        std::tie(indirectDrawBuffer, indirectDrawBufferMemory)       = resourceManager->createIndirectDrawBuffer();
        std::tie(litIndirectDrawBuffer, litIndirectDrawBufferMemory) = resourceManager->createIndirectDrawBuffer();

        // after having created all our desire buffers we can initialize the descriptor set
        descriptorSet->createDescriptorSet();

        swapchain->create(*window, vSync);
        createShadowDepthBuffer(DEFAULT_SHADOW_RESOLUTION);
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
            pipelineManager->createPipeline<BlurPushConstants>(PipelineCategory::BEFORE_GEOMETRY, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone, vk::False,
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
    #pragma endregion






#pragma region DRAWFRAME
    // main render loop
    void drawFrame() {
        device->getDevice().waitForFences(*inFlightFences[currentFrame], vk::True, UINT64_MAX);

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

        auto [result, imageIndex] = swapchain->getSwapChain().acquireNextImage(UINT64_MAX, *presentCompleteSemaphores[currentFrame], nullptr);

        if (result == vk::Result::eErrorOutOfDateKHR) {
            swapchain->recreate(window, vSync);
            handleSwapchainResize();
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
                    NodeOps::calculateCascadedLightSpaceMatrices(light, activeCamera, this);
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
            handleSwapchainResize();
        } else if (result != vk::Result::eSuccess) {
            throw std::runtime_error("failed to present swap chain image!");
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
        //pipelineManager->checkForShaderUpdates(); // TODO enable this
    }
    #pragma endregion





#pragma region GET/SET
    
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
    std::map<uint32_t, Light>& getLightsMutable() { return lights; }
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
        handleSwapchainResize();
    }
    
    void toggleSSAO() { enableSSAO = !enableSSAO; }
    void toggleSSR() { enableSSR = !enableSSR; }

    bool ssrResolutionDirty = false;
    
    void toggleBBOXes() { showBBOXes = !showBBOXes; }
    
    void setSkyBox(uint32_t skyboxIndex) { this->skyboxIndex = skyboxIndex; }
    #pragma endregion

    
    // Recreates all screen-size render targets and updates the camera aspect ratio.
    // Called after swapchain recreation (resize, vsync toggle, etc.)
    void handleSwapchainResize() {
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



    // Generic blur pass that can blur any attachment
    // Performs two-pass separable Gaussian blur (horizontal + vertical)
    // Requires a temporary texture of the same size and format as the source
    void blurAttachment(vk::raii::CommandBuffer& cmd, uint32_t sourceTextureIndex, uint32_t tempTextureIndex, uint32_t width, uint32_t height, float blurRadius = 1.0f,
                        uint32_t samplerIndex = 0) {

        auto& blurPipeline = *pipelineManager->getBeforeGeoPipelines()[blurPipelineIndex];
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


  private:
#pragma region CREATE RESOURCES
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

    void createSSAOResources(uint32_t width, uint32_t height) {
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

    void createRoughnessMetalResources(uint32_t width, uint32_t height) {
        createOrResizeMSAATarget(roughnessMetal, width, height, vk::Format::eR8G8B8A8Unorm);
        createOrResizeRenderTarget(roughnessMetalTextureIndex, width, height, vk::Format::eR8G8B8A8Unorm, "internal/roughness_metal");
    }

    void createNormalResources(uint32_t width, uint32_t height) {
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

    // Color resolve is the MSAA resolve target for the geometry pass — must always be full resolution
    void createColorResolveResources(uint32_t width, uint32_t height) {
        createOrResizeRenderTarget(colorResolveTextureIndex, width, height, swapchain->getSwapChainImageFormat(), "internal/color_resolve", vk::ImageUsageFlagBits::eTransferSrc);
    }

    void createMotionVectorResources(uint32_t width, uint32_t height) {
        createOrResizeMSAATarget(motionVectors,width,height, vk::Format::eR16G16Sfloat);
        createOrResizeRenderTarget(motionVectorTextureIndex, width, height, vk::Format::eR16G16Sfloat,"internal/motion_vectors");
    }

    void createSSRResources(uint32_t width, uint32_t height, uint32_t numTemporalFrames = 1) {

        int i = 0;
        for(i = 0; i < numTemporalFrames; i++){
            if(ssrTextureIndices.size() <= i)
                ssrTextureIndices.push_back(0xFFFFFFFF);
            createOrResizeRenderTarget(ssrTextureIndices[i], width, height, swapchain->getSwapChainImageFormat(), (std::string("internal/ssr") + std::to_string(i)).c_str());
        }

        // SSR pipeline (only created once)
        if (ssrPipelineIndex == 0xFFFFFFFF) {
            ssrPipelineIndex = pipelineManager->createPipeline<SSRPushConstants>(
                PipelineCategory::POSTPROCESS, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone,
                vk::False, vk::False, "shaders/ssr.spv", descriptorSet->getDescriptorSetLayout(), descriptorSet->getDescriptorSet());
        }

        // SSR apply pipeline with multiplicative blending (only created once)
        if (ssrApplyPipelineIndex == 0xFFFFFFFF) {
            ssrApplyPipelineIndex = pipelineManager->createPipeline<SSRApplyPushConstants>(
                PipelineCategory::POSTPROCESS, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone,
                vk::False, vk::False, "shaders/ssr_apply.spv", descriptorSet->getDescriptorSetLayout(), descriptorSet->getDescriptorSet());
        }
    }

    void createHiZResources(uint32_t width, uint32_t height) {
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
    #pragma endregion






    /////=================================================MAIN RENDERING LOGIC=================================================/////

#pragma region RENDERING
    void recordCommandBuffer(uint32_t imageIndex) {
        auto& cmd = commandBuffers[currentFrame];
        cmd.begin({});

        for (auto& [lightId, light] : lights) {
            if (light.castsShadows == 1)
                recordShadowPass(cmd, light);
        }

        recordGeometryPass(cmd, imageIndex);

        if (enableSSR && ssrPipelineIndex != 0xFFFFFFFF)
            recordSSRPass(cmd, imageIndex);

        if (enableSSAO && ssaoPipelineIndex != 0xFFFFFFFF)
            recordSSAOPass(cmd, imageIndex);

        if (imageVisIndex != 0xFFFFFFFF)
            recordImageVisPass(cmd, imageIndex);

        recordOverlayPass(cmd, imageIndex);

        resourceManager->transitionImageLayout(&cmd, swapchain->getSwapChainImages()[imageIndex], vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR);
        cmd.end();
    }

    // Shared frustum culling — builds indirect draw commands and calls perSubMeshFn for each visible submesh
    template <typename PerSubMeshFn>
    void buildGeometryDrawCommands(const std::array<Plane, 6>& frustumPlanes, bool doCulling, PerSubMeshFn&& perSubMeshFn) {
        indirectCommands.clear();

        for (const auto& [shader, materials] : shaders) {
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

                    if (node->isBoundingBoxValid() && doCulling) {
                        if (!isAABBInFrustum(node->getBoundingBoxMin(), node->getBoundingBoxMax(), frustumPlanes))
                            continue;
                    }

                    for (auto mesh : subMeshIndices) {
                        const auto& subMesh = assetManager.subMeshes[mesh];

                        glm::vec3 subWorldMin, subWorldMax;
                        transformAABBToWorldSpace(subMesh.boundingBoxMin, subMesh.boundingBoxMax, node->getTransform(), subWorldMin, subWorldMax);
                        if (!isAABBInFrustum(subWorldMin, subWorldMax, frustumPlanes) && doCulling){
                            culledCount++;
                            continue;
                        }

                        if(showBBOXes && doCulling)
                            Gizmos::drawBox(subWorldMin,subWorldMax,glm::vec4(1.0f,1.0f,0.0f,1.0f));

                        indirectCommands.push_back({.indexCount    = subMesh.indexCount,
                                                    .instanceCount = 1,
                                                    .firstIndex    = static_cast<uint32_t>(subMesh.indexOffset / sizeof(uint32_t)),
                                                    .vertexOffset  = 0,
                                                    .firstInstance = 0});

                        perSubMeshFn(subMesh, *node, material);
                    }
                }
            }
        }
    }

    void recordHiZPass(vk::raii::CommandBuffer& cmd) {
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

        // Transition depth resolve back to depth attachment
        resourceManager->transitionImageLayout(&cmd, *descriptorSet->getTextureResource(swapchain->getDepthResolveIndex()).image,
            vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eDepthStencilAttachmentOptimal);
    }

    void recordGeometryPass(vk::raii::CommandBuffer& cmd, uint32_t imageIndex) {
        resourceManager->transitionImageLayout(&cmd, swapchain->getSwapChainImages()[imageIndex], vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal);
        resourceManager->transitionImageLayout(&cmd, swapchain->getColorImage(), vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal);
        resourceManager->transitionImageLayout(&cmd, swapchain->getDepthImage(), vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal);
        resourceManager->transitionImageLayout(&cmd, roughnessMetal.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal);
        resourceManager->transitionImageLayout(&cmd, *descriptorSet->getTextureResource(roughnessMetalTextureIndex).image,
                                               vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eColorAttachmentOptimal);
        resourceManager->transitionImageLayout(&cmd, normalMSAA.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal);
        resourceManager->transitionImageLayout(&cmd, *descriptorSet->getTextureResource(normalTextureIndex).image,
                                               vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eColorAttachmentOptimal);
        resourceManager->transitionImageLayout(&cmd, motionVectors.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal);
        resourceManager->transitionImageLayout(&cmd, *descriptorSet->getTextureResource(motionVectorTextureIndex).image,
                                               vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eColorAttachmentOptimal);

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
#pragma region DEPTH & HIZ
        if (!indirectCommands.empty()) {
            // Upload indirect commands and per-draw data (shared between depth prepass and lit pass)
            void* data = litIndirectDrawBufferMemory.mapMemory(frameByteOffset, indirectCommands.size() * sizeof(DrawIndexedIndirectCommand));
            memcpy(data, indirectCommands.data(), indirectCommands.size() * sizeof(DrawIndexedIndirectCommand));
            litIndirectDrawBufferMemory.unmapMemory();

            auto* litDataPtr = descriptorSet->getFixedBufferMappedData<LitDrawData>(litDrawDataBufferIndex);
            if (litDataPtr) {
                uint32_t frameOffset = currentFrame * MAX_FIXED_BUFFER;
                memcpy(&litDataPtr[frameOffset], litDrawDataList.data(), litDrawDataList.size() * sizeof(LitDrawData));
            }

            LitPushConstants pushConstants = {.vertexBufferAddress  = descriptorSet->getVariableBuffers()[vertexBufferIndex]->address,
                                              .modelMatricesAddress = descriptorSet->getFixedBuffers()[modelMatrixBufferIndex]->address + static_cast<vk::DeviceSize>(currentFrame) * MAX_FIXED_BUFFER * sizeof(glm::mat4),
                                              .lightsAddress        = descriptorSet->getFixedBuffers()[lightBufferIndex]->address + static_cast<vk::DeviceSize>(currentFrame) * MAX_FIXED_BUFFER * sizeof(Light),
                                              .litDrawDataAddress   = descriptorSet->getFixedBuffers()[litDrawDataBufferIndex]->address + static_cast<vk::DeviceSize>(currentFrame) * MAX_FIXED_BUFFER * sizeof(LitDrawData),
                                              .litPassData          = descriptorSet->getFixedBuffers()[litPassDataBufferIndex]->address + static_cast<vk::DeviceSize>(currentFrame) * MAX_FIXED_BUFFER * sizeof(LitPassData)
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
#pragma endregion
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
                                              .lightsAddress        = descriptorSet->getFixedBuffers()[lightBufferIndex]->address + static_cast<vk::DeviceSize>(currentFrame) * MAX_FIXED_BUFFER * sizeof(Light),
                                              .litDrawDataAddress   = descriptorSet->getFixedBuffers()[litDrawDataBufferIndex]->address + static_cast<vk::DeviceSize>(currentFrame) * MAX_FIXED_BUFFER * sizeof(LitDrawData),
                                              .litPassData          = descriptorSet->getFixedBuffers()[litPassDataBufferIndex]->address + static_cast<vk::DeviceSize>(currentFrame) * MAX_FIXED_BUFFER * sizeof(LitPassData)
                                            };

            for (const auto& [shader, materials] : shaders) {
                auto currentPipeline = &(geoPipelines[shader.pipelineIndex]);
                bindPipeline(cmd, **currentPipeline);
                cmd.pushConstants<LitPushConstants>((*currentPipeline)->layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pushConstants);
                cmd.drawIndexedIndirect(*litIndirectDrawBuffer, frameByteOffset, static_cast<uint32_t>(indirectCommands.size()), sizeof(DrawIndexedIndirectCommand));
            }
        }

        cmd.endRendering();

        // Blit color resolve to swapchain image
        resourceManager->transitionImageLayout(&cmd, *colorResolve.image,
            vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eTransferSrcOptimal);
        resourceManager->transitionImageLayout(&cmd, swapchain->getSwapChainImages()[imageIndex],
            vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eTransferDstOptimal);

        vk::ImageBlit blitRegion{
            .srcSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .layerCount = 1},
            .srcOffsets = std::array<vk::Offset3D, 2>{vk::Offset3D{0, 0, 0}, vk::Offset3D{static_cast<int32_t>(swapchain->getSwapChainExtent().width), static_cast<int32_t>(swapchain->getSwapChainExtent().height), 1}},
            .dstSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .layerCount = 1},
            .dstOffsets = std::array<vk::Offset3D, 2>{vk::Offset3D{0, 0, 0}, vk::Offset3D{static_cast<int32_t>(swapchain->getSwapChainExtent().width), static_cast<int32_t>(swapchain->getSwapChainExtent().height), 1}}
        };
        cmd.blitImage(*colorResolve.image, vk::ImageLayout::eTransferSrcOptimal,
                      swapchain->getSwapChainImages()[imageIndex], vk::ImageLayout::eTransferDstOptimal,
                      blitRegion, vk::Filter::eNearest);

        // Transition back: color resolve to shader readable, swapchain to color attachment
        resourceManager->transitionImageLayout(&cmd, *colorResolve.image,
            vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
        resourceManager->transitionImageLayout(&cmd, swapchain->getSwapChainImages()[imageIndex],
            vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eColorAttachmentOptimal);

        // Transition roughness-metal to shader readable for SSR
        resourceManager->transitionImageLayout(&cmd, *descriptorSet->getTextureResource(roughnessMetalTextureIndex).image,
                                               vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

        // Generate normal mips inline for SSR pre-filtering
        auto& normalRes = descriptorSet->getTextureResource(normalTextureIndex);
        resourceManager->generateMipmaps(*normalRes.image, vk::Format::eR8G8B8A8Unorm,
            static_cast<int32_t>(normalRes.width), static_cast<int32_t>(normalRes.height),
            normalMipLevels, 1, &cmd, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

        // Transition motion vecs to shader read only
        resourceManager->transitionImageLayout(&cmd, *descriptorSet->getTextureResource(motionVectorTextureIndex).image,
                                               vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
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
        for(auto& line : Gizmos::getNoDiscardLines()){
            Gizmos::drawLine(line.second);
        }
        auto& gizmoPipeline = pipelineManager->getPostProcessPipelines()[gizmoPipelineIndex];
        bindPipeline(cmd, *gizmoPipeline);
        LinePushConstants lineConstants = {.lineVertsAddress = Gizmos::getLineBufferAddress(), .viewProjection = activeCamera.viewProjection};
        cmd.pushConstants<LinePushConstants>(*gizmoPipeline->layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, lineConstants);
        cmd.draw(Gizmos::getVertexCount(), 1, 0, 0);

        // GUI
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), *cmd);
        cmd.endRendering();
    }

    void recordSSAOPass(vk::raii::CommandBuffer& cmd, uint32_t imageIndex) {
        auto swapExtent = swapchain->getSwapChainExtent();
        auto& ssaoTexture = descriptorSet->getTextureResource(ssaoTextureIndex);
        vk::Extent2D ssaoExtent{ssaoTexture.width, ssaoTexture.height};

        // Transition depth to readable, SSAO target to color attachment
        resourceManager->transitionImageLayout(&cmd, swapchain->getDepthImage(),
            vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
        resourceManager->transitionImageLayout(&cmd, *ssaoTexture.image,
            vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eColorAttachmentOptimal);

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

    void recordSSRPass(vk::raii::CommandBuffer& cmd, uint32_t imageIndex) {
        auto swapExtent = swapchain->getSwapChainExtent();
        auto& ssrTexture = descriptorSet->getTextureResource(ssrTextureIndices[ssrIndex]);
        vk::Extent2D ssrExtent{ssrTexture.width, ssrTexture.height};

        // transition depth to readable
        resourceManager->transitionImageLayout(&cmd, swapchain->getDepthImage(),
            vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
        // transition SSR target to color attachment
        resourceManager->transitionImageLayout(&cmd, *ssrTexture.image,
            vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eColorAttachmentOptimal);

        // Render SSR at SSR resolution
        drawFullscreenPass(cmd, *pipelineManager->getPostProcessPipelines()[ssrPipelineIndex], *ssrTexture.imageView, ssrExtent,
            SSRPushConstants{.invViewProj = glm::inverse(activeCamera.viewProjection),
                             .viewProj = activeCamera.viewProjection,
                             .cameraPos = activeCamera.position,
                             .depthIndex = swapchain->getDepthResolveIndex(),
                             .depthSamplerIndex = depthSamplerIndex,
                             .colorIndex = colorResolveTextureIndex,
                             .colorSamplerIndex = defaultSamplerIndex,
                             .roughnessMetalIndex = roughnessMetalTextureIndex,
                             .roughnessMetalSamplerIndex = defaultSamplerIndex,
                             .normalIndex = normalTextureIndex,
                             .normalSamplerIndex = defaultSamplerIndex,
                             .resolution = glm::uvec2(descriptorSet->getTextureResource(hiZTextureIndex).width,
                                              descriptorSet->getTextureResource(hiZTextureIndex).height),
                             .maxDistance = ssrMaxDistance,
                             .hiZIndex = hiZTextureIndex,
                             .hiZMipLevels = hiZMipLevels,
                             .thickness = ssrThickness,
                             .roughnessThreshold = ssrRoughnessThreshold,
                             .maxSteps = ssrMaxSteps},
            vk::AttachmentLoadOp::eClear, {1.0f, 1.0f, 1.0f, 1.0f});

        resourceManager->transitionImageLayout(&cmd, *ssrTexture.image, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

        // apply to swapchain at full resolution
        std::array<uint32_t, 8> ssrIndices;
        ssrIndices.fill(0xFFFFFFFF);
        std::copy(ssrTextureIndices.begin(), ssrTextureIndices.end(), ssrIndices.begin());
        drawFullscreenPass(cmd, *pipelineManager->getPostProcessPipelines()[ssrApplyPipelineIndex], *swapchain->getSwapChainImageViews()[imageIndex], swapExtent,
            SSRApplyPushConstants{.samplerIndex = defaultSamplerIndex, .sceneColorIndex = colorResolveTextureIndex, .sceneSamplerIndex = defaultSamplerIndex, .ssrTextureIndices = ssrIndices},
            vk::AttachmentLoadOp::eLoad);

        // transition depth back
        resourceManager->transitionImageLayout(&cmd, swapchain->getDepthImage(), vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eDepthStencilAttachmentOptimal);

        ssrIndex++;
        if(ssrIndex >= ssrTextureIndices.size())
            ssrIndex = 0;
    }

    void recordImageVisPass(vk::raii::CommandBuffer& cmd, uint32_t imageIndex) {
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

    void recordShadowPass(vk::raii::CommandBuffer& cmd, Light& light) {

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
            vk::RenderingAttachmentInfo depthAttachment{.imageView = *shadowDepth.view,
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
            //Gizmos::drawFrustum(lightSpaceMatrix,glm::vec4(0,1,0,1));

            // Build indirect draw commands and shadow draw data with CPU frustum culling
            drawDataList.clear();
            buildGeometryDrawCommands(frustumPlanes, false, [&](const auto& subMesh, auto& node, const auto& material) {
                drawDataList.push_back({.vertexAllocationIndex = subMesh.vertexAllocationIndex,
                                        .vertexOffset = static_cast<uint32_t>(subMesh.vertexOffset),
                                        .vertexStride = subMesh.vertexStride,
                                        .modelMatrixIndex = node.getModelMatrixIndex()});
            });

            // build the indirect draw command and submit it
            if (!indirectCommands.empty()) {
                vk::DeviceSize frameByteOffset = currentFrame * MAX_INDIRECT_COMMANDS * sizeof(DrawIndexedIndirectCommand);
                void* data = indirectDrawBufferMemory.mapMemory(frameByteOffset, indirectCommands.size() * sizeof(DrawIndexedIndirectCommand));
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
                    .vertexBufferAddress   = descriptorSet->getVariableBuffers()[vertexBufferIndex]->address,
                    .modelMatricesAddress  = descriptorSet->getFixedBuffers()[modelMatrixBufferIndex]->address + static_cast<vk::DeviceSize>(currentFrame) * MAX_FIXED_BUFFER * sizeof(glm::mat4),
                    .shadowDrawDataAddress = descriptorSet->getFixedBuffers()[shadowDrawDataBufferIndex]->address + static_cast<vk::DeviceSize>(currentFrame) * MAX_FIXED_BUFFER * sizeof(ShadowDrawData),
                    .lightSpaceMatrix      = lightSpaceMatrix};
                cmd.pushConstants<ShadowPushConstants>(*currentPipeline->layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pushConstants);

                cmd.bindIndexBuffer(indexBufferHandle, 0, vk::IndexType::eUint32);
                cmd.drawIndexedIndirect(*indirectDrawBuffer, frameByteOffset, static_cast<uint32_t>(indirectCommands.size()), sizeof(DrawIndexedIndirectCommand));
            }
            cmd.endRendering();

            // Transition shadow map back to shader read-only for fragment shader sampling
            resourceManager->transitionImageLayout(&cmd, *shadowMap->image, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
        }
    }

    // Creates (or frees and recreates) a single-sample render target registered in the descriptor set.
    // Handles free-if-exists, image creation, view, layout transition, and descriptor set allocation.
    void createOrResizeRenderTarget(uint32_t& index, uint32_t width, uint32_t height,
                                     vk::Format format, const char* debugName,
                                     vk::ImageUsageFlags extraUsage = {}) {
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

    // Creates (or recreates) an MSAA intermediate render target not registered in the descriptor set.
    void createOrResizeMSAATarget(Image& target, uint32_t width, uint32_t height, vk::Format format) {
        target.view = nullptr; // destroy view before image
        resourceManager->createImage(width, height, 1, msaaSamples, format, vk::ImageTiling::eOptimal,
                                     vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment,
                                     vk::MemoryPropertyFlagBits::eDeviceLocal, target.image, target.memory);
        target.view = resourceManager->createImageView(target.image, format, vk::ImageAspectFlagBits::eColor, 1);
        resourceManager->transitionImageLayout(nullptr, target.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal);
    }

    void setFullscreenViewport(vk::raii::CommandBuffer& cmd, vk::Extent2D extent) {
        cmd.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.0f, 1.0f));
        cmd.setScissor(0, vk::Rect2D({0, 0}, extent));
    }

    void bindPipeline(vk::raii::CommandBuffer& cmd, PipelineBase& pipeline) {
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.pipeline);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline.layout, 0, {**pipeline.descriptorSet}, {});
    }

    // Draws a fullscreen triangle to a single color attachment.
    // Handles begin/end rendering, viewport, pipeline bind, push constants, and draw.
    template <typename T>
    void drawFullscreenPass(vk::raii::CommandBuffer& cmd, PipelineBase& pipeline,
                            vk::ImageView targetView, vk::Extent2D extent, const T& pushConstants,
                            vk::AttachmentLoadOp loadOp = vk::AttachmentLoadOp::eClear,
                            std::array<float, 4> clearColor = {0.0f, 0.0f, 0.0f, 0.0f}) {

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

    #pragma endregion
};
