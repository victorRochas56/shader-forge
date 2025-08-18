#pragma once
#ifndef VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1 // for raii
#endif
#ifndef VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_NO_CONSTRUCTORS 1 // for structs constructors
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#define GLFW_INCLUDE_VULKAN // REQUIRED only for GLFW CreateWindowSurface.
#include <GLFW/glfw3.h>

#include "include/imgui.h"
#include "include/imgui_impl_glfw.h"
#include "include/imgui_impl_vulkan.h"

#define GLM_FORCE_RADIANS
#define GLM_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#endif

// my modules
#include <devices.hpp>
#include <load_resources.hpp>
#include <pipeline.hpp>
#include <resource_manager.hpp>
#include <scene_elements.hpp>
#include <structs.hpp>
#include <swapchain.hpp>
#include <utils.hpp>
///

constexpr int MAX_FRAMES_IN_FLIGHT = 2;
constexpr int MAX_NODES = 2048;

constexpr uint32_t SHOW_ALBEDO = 1;
constexpr uint32_t SHOW_ROUGHNESS = 2;
constexpr uint32_t SHOW_METALLIC = 3;
constexpr uint32_t SHOW_NORMAL = 4;

const std::vector validationLayers = {"VK_LAYER_KHRONOS_validation"};

class Renderer {

  public:
    Camera activeCamera;
    bool canRayCast = false;
    Renderer() = default;
    uint32_t selectedNode = MAX_NODES;

    void initVulkan(uint32_t start_width, uint32_t start_height) {
        createInstance();
        setupDebugMessenger();
        createSurface();
        devices = std::make_unique<Devices>(instance, requiredDeviceExtension, surface);
        msaaSamples = getMaxUsableSampleCount();
        createCommandPool();
        resourceManager = std::make_unique<ResourceManager>(&*devices, &commandPool);
        swapChain = std::make_unique<SwapChain>(window, &*devices, &surface, &instance, &*resourceManager, vSync);
        meshLoader = std::make_unique<MeshLoader>(*resourceManager, *devices);
        pipelineBuilder = std::make_unique<PipelineBuilder>(&*devices, &*resourceManager, &*swapChain);
        graphicsPipeline = pipelineBuilder->createGraphicsPipeline(msaaSamples);
        linePipeline = pipelineBuilder->createLinePipeline(msaaSamples);
        skyboxPipeline = pipelineBuilder->createCubemapPipeline(msaaSamples);
        std::tie(depthPipelineIndex, depthPipeline) = pipelineBuilder->createDepthPipeline(msaaSamples);

        swapChain->createColorResources(msaaSamples);
        swapChain->createDepthResources(msaaSamples);
        createCommandBuffers();
        createSyncObjects();
        resourceManager->initializeDefaults();

        (*meshUsage).fill(0);
        (*lightUsage).fill(0);
        (*nodeUsage).fill(0);

        activeCamera.position = glm::vec3(1.0f, 1.0f, 1.0f);
        activeCamera.lookAt(glm::vec3(0));
        activeCamera.fov = glm::radians(45.0f);
        activeCamera.aspectRatio = start_width / start_height;
        activeCamera.nearPlane = 0.05f;
        activeCamera.farPlane = 1000.0f;
        activeCamera.calculateViewProjectionMatrix();

        rootNode.name = "Scene Root";
    }

    void drawFrame() {
        while (vk::Result::eTimeout == devices->getLogicalDevice().waitForFences(*inFlightFences[currentFrame], vk::True, UINT64_MAX))
            // wait for gpu to finish rendering the frame we just submitted
            ;
        auto [result, imageIndex] = swapChain->getSwapChain().acquireNextImage(UINT64_MAX, *presentCompleteSemaphores[currentFrame], nullptr);

        if (result == vk::Result::eErrorOutOfDateKHR) {
            swapChain->recreateSwapChain(window, &*devices, msaaSamples, vSync);
            int width = 0, height = 0;
            glfwGetFramebufferSize(window, &width, &height);
            activeCamera.aspectRatio = static_cast<float>(width) / static_cast<float>(height);
            activeCamera.calculateViewProjectionMatrix();
            return;
        }
        if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }
        if (imagesInFlight[imageIndex] != VK_NULL_HANDLE) {
            vk::Result waitResult = devices->getLogicalDevice().waitForFences(imagesInFlight[imageIndex], vk::True, UINT64_MAX);
            if (waitResult != vk::Result::eSuccess) {
                throw std::runtime_error("failed to wait for image fence!");
            }
        }

        imagesInFlight[imageIndex] = *inFlightFences[currentFrame];
        devices->getLogicalDevice().resetFences(*inFlightFences[currentFrame]);

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

        devices->getGraphicsQueue().submit(submitInfo, inFlightFences[currentFrame]);

        const vk::PresentInfoKHR presentInfoKHR{.waitSemaphoreCount = 1,
                                                .pWaitSemaphores = &*renderFinishedSemaphores[imageIndex],
                                                .swapchainCount = 1,
                                                .pSwapchains = &*swapChain->getSwapChain(),
                                                .pImageIndices = &imageIndex};

        try {
            result = devices->getPresentQueue().presentKHR(presentInfoKHR);
        } catch (const vk::OutOfDateKHRError&) {
            result = vk::Result::eErrorOutOfDateKHR;
        }

        if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || framebufferResized) {
            framebufferResized = false;
            swapChain->recreateSwapChain(window, &*devices, msaaSamples, vSync);

            int width = 0, height = 0;
            glfwGetFramebufferSize(window, &width, &height);
            activeCamera.aspectRatio = static_cast<float>(width) / static_cast<float>(height);
            activeCamera.calculateViewProjectionMatrix();

        } else if (result != vk::Result::eSuccess) {
            throw std::runtime_error("failed to present swap chain image!");
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    const vk::CommandPool& getCommandPool() const { return *commandPool; }
    const vk::Instance& getInstance() const { return *instance; }
    const Devices* getDevices() { return &*devices; }
    ResourceManager* getResourceManager() const { return &*resourceManager; }
    SwapChain* getSwapChain() { return &*swapChain; }
    void cleanupSwapChain() { swapChain->cleanupSwapChain(); }

    GLFWwindow* getWindow() const { return window; }
    void setWindow(GLFWwindow* pWindow) { window = pWindow; }

    const vk::SampleCountFlagBits& getMsaaSamples() const { return msaaSamples; }

    const uint32_t getGraphicsIndex() const { return graphicsIndex; }

    ///////////////////////////////////////////////////////////////////////////////////////
    // NODES
    ///////////////////////////////////////////////////////////////////////////////////////

    Node& getRootNode() { return rootNode; }
    std::array<Node, MAX_NODES>& getNodes() { return *nodes; }

    uint32_t addNode(Node* parent, std::string name = "empty", glm::vec3 position = glm::vec3(0), glm::quat rotation = glm::quat(1, 0, 0, 0), glm::vec3 scale = glm::vec3(1),
                     bool worldSpace = false) {
        int i = 0;
        for (i = 0; i < MAX_NODES; i++) {
            if ((*nodeUsage)[i] == 0) {
                Node node;
                (*nodes)[i] = node;
                (*nodes)[i].name = name;
                (*nodes)[i].resourceManager = &*resourceManager;
                parent->addChild(&(*nodes)[i]);
                if (worldSpace) {
                    (*nodes)[i].updateWorldTransform(position, rotation, scale);
                }
                (*nodeUsage)[i] = 1;
                return i;
            }
        }
        if (i == MAX_NODES) {
            throw std::runtime_error("exceeded node limit!");
        }
        return MAX_NODES;
    };

    std::vector<uint32_t> rayCastNodes(glm::vec3 origin, glm::vec3 direction) {
        float margin = 0.05f;
        glm::vec3 dir = glm::normalize(direction);
        std::vector<uint32_t> foundNodes;

        for (int i = 0; i < MAX_NODES; i++) {
            if ((*nodeUsage)[i] == 0)
                continue;

            glm::vec3 nodeWorldLoc = glm::vec3((*nodes)[i].worldTransform[3]);
            glm::vec3 toNode = nodeWorldLoc - origin;
            // Project toNode onto the ray direction
            float projectionLength = glm::dot(toNode, dir);
            // Skip nodes behind the ray origin
            if (projectionLength < 0)
                continue;
            // Find closest point on ray to the node
            glm::vec3 closestPointOnRay = origin + dir * projectionLength;
            // Calculate perpendicular distance from node to ray
            float distanceToRay = glm::distance(nodeWorldLoc, closestPointOnRay);
            if (distanceToRay < margin) {
                foundNodes.push_back(i);
            }
        }
        return foundNodes;
    }

    void selectNode(uint32_t nodeIndex) {
        if ((*nodeUsage)[nodeIndex] != 0) {
            selectedNode = nodeIndex;
        }
    }

    void deSelectNode() { selectedNode = MAX_NODES; }

    void addMeshComponent(Node* node, uint32_t meshIndex) {
        node->meshIndex = meshIndex;
        node->meshes = &*meshes;
    };

    void addLightComponent(Node* node, uint32_t lightIndex) {
        node->lightIndex = lightIndex;
        node->lights = &*lights;
    };

    ///////////////////////////////////////////////////////////////////////////////////////
    // LOADING ASSETS
    ///////////////////////////////////////////////////////////////////////////////////////

    uint32_t addMeshFromFile(std::string meshPath, std::string albedoTexPath = "", std::string roughnessTexPath = "", std::string metallicTexPath = "",
                             std::string normalTexPath = "") {

        int i = 0;
        for (i = 0; i < MAX_VERTEX_ALLOCATIONS; i++) {
            if ((*meshUsage)[i] == 0) {
                (*meshes)[i] = meshLoader->loadFromFile(meshPath, albedoTexPath, roughnessTexPath, metallicTexPath, normalTexPath);
                (*meshUsage)[i] = 1;
                return i;
            }
        }
        if (i == MAX_VERTEX_ALLOCATIONS) {
            throw std::runtime_error("exceeded mesh limit!");
        }
        return MAX_VERTEX_ALLOCATIONS;
    }

    void addEnvironmentMap(uint32_t width, uint32_t height, std::string posX, std::string negX, std::string posY, std::string negY, std::string posZ, std::string negZ) {
        environmentMapIndex = meshLoader->loadCubeMapFromFiles(width, height, posX, negX, posY, negY, posZ, negZ);
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    // RENDERER FEATURES
    ///////////////////////////////////////////////////////////////////////////////////////

    uint32_t addPointLight(glm::vec3 color, float range, float intensity) {
        int i = 0;
        for (i = 0; i < MAX_LIGHTS; i++) {
            if ((*lightUsage)[i] == 0) {
                Light light = {.position = glm::vec4(0, 0, 0, 1.0), .color = glm::vec4(color, 1), .range = range, .intensity = intensity};
                light.allocationIndex = resourceManager->allocateLightBuffer(light);
                (*lights)[i] = light;
                (*lightUsage)[i] = 1;
                return i;
            }
        }
        if (i == MAX_LIGHTS) {
            throw std::runtime_error("exceeded light limit!");
        }
        return MAX_LIGHTS;
    }

    void toggleVsync() {
        vSync = !vSync;
        swapChain->recreateSwapChain(window, &*devices, msaaSamples, vSync);
    }

    void showMap(uint32_t map) { debugMaps = map; }

    uint32_t showFullPBR() {
        if (debugMaps != 0) {
            debugMaps = 0;
            return 0;
        }
        return 1;
    }

  private:
    GLFWwindow* window = nullptr;
    vk::raii::Context context;
    vk::raii::Instance instance = nullptr;
    vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
    vk::raii::SurfaceKHR surface = nullptr;

    std::unique_ptr<Devices> devices = nullptr;
    std::unique_ptr<SwapChain> swapChain = nullptr;
    std::unique_ptr<ResourceManager> resourceManager = nullptr;
    std::unique_ptr<MeshLoader> meshLoader = nullptr;
    std::unique_ptr<PipelineBuilder> pipelineBuilder = nullptr;

    vk::raii::Pipeline graphicsPipeline = nullptr;
    vk::raii::Pipeline linePipeline = nullptr;
    vk::raii::Pipeline skyboxPipeline = nullptr;
    vk::raii::Pipeline depthPipeline = nullptr;
    size_t depthPipelineIndex = 0;

    vk::raii::CommandPool commandPool = nullptr;
    std::vector<vk::raii::CommandBuffer> commandBuffers;
    uint32_t graphicsIndex = 0;

    Node rootNode;
    std::unique_ptr<std::array<Node, MAX_NODES>> nodes = std::make_unique<std::array<Node, MAX_NODES>>();
    std::unique_ptr<std::array<uint32_t, MAX_NODES>> nodeUsage = std::make_unique<std::array<uint32_t, MAX_NODES>>();
    std::unique_ptr<std::array<Light, MAX_LIGHTS>> lights = std::make_unique<std::array<Light, MAX_LIGHTS>>();
    std::unique_ptr<std::array<uint32_t, MAX_LIGHTS>> lightUsage = std::make_unique<std::array<uint32_t, MAX_LIGHTS>>();
    std::unique_ptr<std::array<Mesh, MAX_VERTEX_ALLOCATIONS>> meshes = std::make_unique<std::array<Mesh, MAX_VERTEX_ALLOCATIONS>>();
    std::unique_ptr<std::array<uint32_t, MAX_VERTEX_ALLOCATIONS>> meshUsage = std::make_unique<std::array<uint32_t, MAX_VERTEX_ALLOCATIONS>>();

    std::vector<vk::raii::Semaphore> presentCompleteSemaphores;
    std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
    std::vector<vk::raii::Fence> inFlightFences;
    std::vector<vk::Fence> imagesInFlight;

    uint32_t currentFrame = 0;

    bool framebufferResized = false;

    vk::SampleCountFlagBits msaaSamples = vk::SampleCountFlagBits::e1;
    bool vSync = true;

    uint32_t environmentMapIndex = MAX_BINDLESS_TEXTURES;
    uint32_t debugMaps = 0;

    std::vector<const char*> requiredDeviceExtension = {VK_KHR_SWAPCHAIN_EXTENSION_NAME,           VK_KHR_SPIRV_1_4_EXTENSION_NAME,
                                                        VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,   VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME,
                                                        VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME};

    void createInstance() {
        constexpr vk::ApplicationInfo appInfo{.pApplicationName = "Shader Forge",
                                              .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
                                              .pEngineName = "No Engine",
                                              .engineVersion = VK_MAKE_VERSION(1, 0, 0),
                                              .apiVersion = vk::ApiVersion14};

        // Get the required layers
        std::vector<char const*> requiredLayers;
        requiredLayers.assign(validationLayers.begin(), validationLayers.end());

        // Check if the required layers are supported by the Vulkan implementation.
        auto layerProperties = context.enumerateInstanceLayerProperties();
        if (std::ranges::any_of(requiredLayers, [&layerProperties](auto const& requiredLayer) {
                return std::ranges::none_of(layerProperties, [requiredLayer](auto const& layerProperty) { return strcmp(layerProperty.layerName, requiredLayer) == 0; });
            })) {
            throw std::runtime_error("One or more required layers are not supported!");
        }

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

    vk::SampleCountFlagBits getMaxUsableSampleCount() {
        vk::PhysicalDeviceProperties physicalDeviceProperties = devices->getPhysicalDevice().getProperties();

        vk::SampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
        if (counts & vk::SampleCountFlagBits::e64) {
            return vk::SampleCountFlagBits::e64;
        }
        if (counts & vk::SampleCountFlagBits::e32) {
            return vk::SampleCountFlagBits::e32;
        }
        if (counts & vk::SampleCountFlagBits::e16) {
            return vk::SampleCountFlagBits::e16;
        }
        if (counts & vk::SampleCountFlagBits::e8) {
            return vk::SampleCountFlagBits::e8;
        }
        if (counts & vk::SampleCountFlagBits::e4) {
            return vk::SampleCountFlagBits::e4;
        }
        if (counts & vk::SampleCountFlagBits::e2) {
            return vk::SampleCountFlagBits::e2;
        }

        return vk::SampleCountFlagBits::e1;
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
        commandPool = vk::raii::CommandPool(devices->getLogicalDevice(), poolInfo);
    }

    void createCommandBuffers() {
        commandBuffers.clear();
        vk::CommandBufferAllocateInfo allocInfo{.commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = MAX_FRAMES_IN_FLIGHT};

        commandBuffers = vk::raii::CommandBuffers(devices->getLogicalDevice(), allocInfo);
    }

    void recordCommandBuffer(uint32_t imageIndex) {
        commandBuffers[currentFrame].begin({});
        transition_image_layout(imageIndex, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal, {}, // srcAccessMask (no need to wait for previous operations)
                                vk::AccessFlagBits2::eColorAttachmentWrite,                                            // dstAccessMask
                                vk::PipelineStageFlagBits2::eTopOfPipe,                                                // srcStage
                                vk::PipelineStageFlagBits2::eColorAttachmentOutput                                     // dstStage
        );

        // Transition the multisampled color image to COLOR_ATTACHMENT_OPTIMAL
        transition_image_layout_custom(swapChain->getColorImage(), vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal, {},
                                       vk::AccessFlagBits2::eColorAttachmentWrite, vk::PipelineStageFlagBits2::eTopOfPipe, vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                                       vk::ImageAspectFlagBits::eColor);

        // Transition the depth image to DEPTH_ATTACHMENT_OPTIMAL
        transition_image_layout_custom(swapChain->getDepthImage(), vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthAttachmentOptimal, {},
                                       vk::AccessFlagBits2::eDepthStencilAttachmentWrite, vk::PipelineStageFlagBits2::eTopOfPipe, vk::PipelineStageFlagBits2::eEarlyFragmentTests,
                                       vk::ImageAspectFlagBits::eDepth);

        vk::ClearValue clearColor{.color = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f})};
        vk::ClearValue clearDepth{.depthStencil = vk::ClearDepthStencilValue{1.0f, 0}};

        // Color attachment (multisampled) with resolve attachment
        vk::RenderingAttachmentInfo colorAttachment = {.imageView = swapChain->getColorImageView(),
                                                       .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                       .resolveMode = vk::ResolveModeFlagBits::eAverage,
                                                       .resolveImageView = swapChain->getSwapChainImageViews()[imageIndex],
                                                       .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                       .loadOp = vk::AttachmentLoadOp::eClear,
                                                       .storeOp = vk::AttachmentStoreOp::eStore,
                                                       .clearValue = clearColor};

        // depth attachment
        vk::RenderingAttachmentInfo depthAttachmentInfo = {.imageView = swapChain->getDepthImageView(),
                                                           .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                                           .loadOp = vk::AttachmentLoadOp::eClear,
                                                           .storeOp = vk::AttachmentStoreOp::eDontCare,
                                                           .clearValue = clearDepth};

        vk::RenderingInfo renderingInfo = {.renderArea = {.offset = {0, 0}, .extent = swapChain->getSwapChainExtent()},
                                           .layerCount = 1,
                                           .colorAttachmentCount = 1,
                                           .pColorAttachments = &colorAttachment,
                                           .pDepthAttachment = &depthAttachmentInfo};

        commandBuffers[currentFrame].beginRendering(renderingInfo);

        commandBuffers[currentFrame].setViewport(
            0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChain->getSwapChainExtent().width), static_cast<float>(swapChain->getSwapChainExtent().height), 0.0f, 1.0f));
        commandBuffers[currentFrame].setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChain->getSwapChainExtent()));

        // skybox
        commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, *skyboxPipeline);
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineBuilder->getPipelineLayout(size_t(2)), 0, {resourceManager->getDescriptorSet()},
                                                        nullptr);
        SkyBoxConstants skyboxConstants = {
            .skyboxIndex = environmentMapIndex, .padding1 = 0, .padding2 = 0, .padding3 = 0, .invViewProjMatrix = glm::inverse(activeCamera.viewProjection)};
        commandBuffers[currentFrame].pushConstants<SkyBoxConstants>(pipelineBuilder->getPipelineLayout(size_t(2)),
                                                                    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, skyboxConstants);
        commandBuffers[currentFrame].draw(3, 1, 0, 0);

        // drawing meshes
        commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineBuilder->getPipelineLayout(size_t(0)), 0, {resourceManager->getDescriptorSet()},
                                                        nullptr);

        for (int i = 0; i < MAX_VERTEX_ALLOCATIONS; i++) {
            if ((*meshUsage)[i] == 0) {
                continue;
            }
            Mesh mesh = (*meshes)[i];
            PushConstants pushConstants = {.vertexBufferIndex = mesh.vertexAllocationIndex,
                                           .vertexOffset = static_cast<uint32_t>(mesh.vertexOffset), // Convert byte offset to element offset
                                           .vertexStride = mesh.vertexStride,
                                           .modelMatrixIndex = mesh.modelMatrixIndex,
                                           .albedoTextureIndex = mesh.albedoTextureIndex,
                                           .roughnessTextureIndex = mesh.roughnessTextureIndex,
                                           .metallicTextureIndex = mesh.metallicTextureIndex,
                                           .normalTextureIndex = mesh.normalTextureIndex,
                                           .samplerIndex = mesh.samplerIndex,
                                           .lightCount = resourceManager->getLightCount(),
                                           .environmentMapIndex = environmentMapIndex,
                                           .debugMaps = debugMaps,
                                           .viewProjection = activeCamera.viewProjection,
                                           .cameraPos = glm::vec4(activeCamera.position, 1.0)};

            commandBuffers[currentFrame].pushConstants<PushConstants>(pipelineBuilder->getPipelineLayout(size_t(0)),
                                                                      vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pushConstants);
            commandBuffers[currentFrame].draw(mesh.vertexCount, 1, 0, 0);
        }

        // drawing lines
        commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, *linePipeline);
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineBuilder->getPipelineLayout(size_t(1)), 0, {resourceManager->getDescriptorSet()},
                                                        nullptr);
        for (int i = 0; i < MAX_LINES; i++) {
            if ((*lineUsage)[i] != 0) {
                std::array<uint32_t, MAX_LINES> lineused = *lineUsage;
                LineAlloc line = (*lineAllocs)[i];
                LineAlloc lineConstants = {.viewProjection = activeCamera.viewProjection, .allocIndex = line.allocIndex, .offset = line.offset, .stride = line.stride};
                commandBuffers[currentFrame].pushConstants<LineAlloc>(pipelineBuilder->getPipelineLayout(size_t(1)),
                                                                      vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, lineConstants);
                commandBuffers[currentFrame].draw(2, 1, 0, 0);
            }
        }

        // depth buffer view
        resourceManager->transitionImageLayout(swapChain->getDepthImage(), vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, 1);
        commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, *depthPipeline);
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineBuilder->getPipelineLayout(depthPipelineIndex), 0,
                                                        {resourceManager->getDepthDescriptorSet()}, nullptr);
        DepthPushConstants depthConstants = {.nearPlane = activeCamera.nearPlane, .farPlane = activeCamera.farPlane, .linearize = 1};
        commandBuffers[currentFrame].pushConstants<DepthPushConstants>(pipelineBuilder->getPipelineLayout(depthPipelineIndex),
                                                                       vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, depthConstants);
        commandBuffers[currentFrame].draw(3, 1, 0, 0);
        resourceManager->transitionImageLayout(swapChain->getDepthImage(), vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eDepthStencilAttachmentOptimal, 1);
        // ui
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), *commandBuffers[currentFrame]);

        commandBuffers[currentFrame].endRendering();

        // After rendering, transition the swapchain image to PRESENT_SRC
        transition_image_layout(imageIndex, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR,
                                vk::AccessFlagBits2::eColorAttachmentWrite,         // srcAccessMask
                                {},                                                 // dstAccessMask
                                vk::PipelineStageFlagBits2::eColorAttachmentOutput, // srcStage
                                vk::PipelineStageFlagBits2::eBottomOfPipe           // dstStage
        );
        commandBuffers[currentFrame].end();
    }

    void transition_image_layout(uint32_t imageIndex, vk::ImageLayout oldLayout, vk::ImageLayout newLayout, vk::AccessFlags2 srcAccessMask, vk::AccessFlags2 dstAccessMask,
                                 vk::PipelineStageFlags2 srcStageMask, vk::PipelineStageFlags2 dstStageMask) {
        vk::ImageMemoryBarrier2 barrier = {
            .srcStageMask = srcStageMask,
            .srcAccessMask = srcAccessMask,
            .dstStageMask = dstStageMask,
            .dstAccessMask = dstAccessMask,
            .oldLayout = oldLayout,
            .newLayout = newLayout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = swapChain->getSwapChainImages()[imageIndex],
            .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1}};
        vk::DependencyInfo dependencyInfo = {.dependencyFlags = {}, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier};
        commandBuffers[currentFrame].pipelineBarrier2(dependencyInfo);
    }

    void transition_image_layout_custom(vk::raii::Image& image, vk::ImageLayout old_layout, vk::ImageLayout new_layout, vk::AccessFlags2 src_access_mask,
                                        vk::AccessFlags2 dst_access_mask, vk::PipelineStageFlags2 src_stage_mask, vk::PipelineStageFlags2 dst_stage_mask,
                                        vk::ImageAspectFlags aspect_mask) {
        vk::ImageMemoryBarrier2 barrier = {.srcStageMask = src_stage_mask,
                                           .srcAccessMask = src_access_mask,
                                           .dstStageMask = dst_stage_mask,
                                           .dstAccessMask = dst_access_mask,
                                           .oldLayout = old_layout,
                                           .newLayout = new_layout,
                                           .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                           .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                           .image = image,
                                           .subresourceRange = {.aspectMask = aspect_mask, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1}};
        vk::DependencyInfo dependency_info = {.dependencyFlags = {}, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier};
        commandBuffers[currentFrame].pipelineBarrier2(dependency_info);
    }

    void createSyncObjects() {
        presentCompleteSemaphores.clear();
        renderFinishedSemaphores.clear();
        inFlightFences.clear();
        imagesInFlight.clear();

        // Separate semaphores for acquisition (per frame) and rendering (per image)
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            presentCompleteSemaphores.emplace_back(devices->getLogicalDevice(), vk::SemaphoreCreateInfo());
            inFlightFences.emplace_back(devices->getLogicalDevice(), vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
        }

        // Render finished semaphores per swapchain image
        for (size_t i = 0; i < swapChain->getSwapImageSize(); i++) {
            renderFinishedSemaphores.emplace_back(devices->getLogicalDevice(), vk::SemaphoreCreateInfo());
        }

        imagesInFlight.resize(swapChain->getSwapImageSize(), VK_NULL_HANDLE);
    }
};