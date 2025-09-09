#pragma once
#ifndef VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1 // for raii
#endif
#ifndef VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_NO_CONSTRUCTORS 1 // for structs constructors
#endif

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
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

#include "descriptor_sets.hpp"
#include "devices.hpp"
#include "pipelines.hpp"
#include "scene_elements.hpp"
#include "swapchain.hpp"
#include "utils.hpp"

constexpr int MAX_FRAMES_IN_FLIGHT = 2;
const std::vector validationLayers = {"VK_LAYER_KHRONOS_validation"};

class Renderer {
  public:
    Camera activeCamera;

    void initVulkan(uint32_t startWidth, uint32_t startHeight) {
        createInstance();
        setupDebugMessenger();
        createSurface();
        device = std::make_unique<Device>(instance, getRequiredExtensions(), surface);
        msaaSamples = getMaxUsableSampleCount(*device);
        createCommandPool();
        createCommandBuffers();
        resourceManager = std::make_unique<ResourceManager>(*device,commandPool);
        swapchain = std::make_unique<Swapchain>(*device,*resourceManager, surface, msaaSamples);
        createSyncObjects();
        pipelineManager = std::make_unique<PipelineManager>(*device, *swapchain, msaaSamples);
        swapchain->create(*window, vSync);

        // initializing camera
        activeCamera = Camera{.position = glm::vec3(2, 2, 2),
                              .target = glm::vec3(0, 0, 0),
                              .fov = 90.0,
                              .aspectRatio = static_cast<float>(startWidth) / static_cast<float>(startHeight),
                              .nearPlane = 0.1,
                              .farPlane = 100.0};
        activeCamera.calculateViewProjectionMatrix();

        // descriptor set
        descriptorSet = std::make_unique<DescriptorSet>(*device, *resourceManager, &commandPool);

        descriptorSet->createVariableBuffer(256 * 1024 * 1024);                 // 256 mb vertex buffer
        modelMatrixBufferIndex = descriptorSet->createFixedBuffer<glm::mat4>(); // max 2048 model matrices by default
        descriptorSet->createDescriptorSet();
        defaultSamplerIndex = descriptorSet->allocateSampler(vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear, vk::SamplerAddressMode::eRepeat, VK_TRUE, 0.1, VK_FALSE,
                                                             vk::CompareOp::eLessOrEqual, vk::BorderColor::eFloatOpaqueBlack);

        // default normal texture
        std::array<float, 4> normalColor = {0.5, 0.0, 0.5, 1.0};
        auto [image, memory, imageView] = resourceManager->createTexture(normalColor.data(), 1, 1, vk::Format::eR8G8B8A8Srgb);
        defaultNormalIndex = descriptorSet->allocateTexture(std::move(image), std::move(memory), std::move(imageView));

        // pipeline(s)
        pipelineManager->createPipeline<PushConstants>(1, "/shaders/lit.spv", descriptorSet->getDescriptorSetLayout(), descriptorSet->getDescriptorSet());
        fallbackLitShader = Shader{.sourceFile = "/shaders/lit.spv", .pipelineIndex = static_cast<uint32_t>(pipelineManager->getGeoPipelines().size())};
        fallbackDefaultMaterial = Material{
            .shaderSource = fallbackLitShader,
            .textureMask = 0x00000000,
            // 1st bit : hasAlbedo
            // 2nd bit : hasRoughness
            // 3rd bit : hasMetallic
            // 4th bit : hasNormal
            .color = glm::vec4(0.5, 0.5, 0.5, 1),
            .metallic = 0.0,
            .roughness = 0.75,
            .normalTextureIndex = defaultNormalIndex // should be set to default normal if not present
        };

        nodes[0] =  Node(this, 0);
        rootNode = &*nodes[0];
        lastNode++;
    }

    void drawFrame() {
        while (vk::Result::eTimeout == device->getDevice().waitForFences(*inFlightFences[currentFrame], vk::True, UINT64_MAX))
            // wait for gpu to finish rendering the frame we just submitted
            ;
        auto [result, imageIndex] = swapchain->getSwapChain().acquireNextImage(UINT64_MAX, *presentCompleteSemaphores[currentFrame], nullptr);

        if (result == vk::Result::eErrorOutOfDateKHR) {
            swapchain->recreate(window, vSync);
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
            vk::Result waitResult = device->getDevice().waitForFences(imagesInFlight[imageIndex], vk::True, UINT64_MAX);
            if (waitResult != vk::Result::eSuccess) {
                throw std::runtime_error("failed to wait for image fence!");
            }
        }

        imagesInFlight[imageIndex] = *inFlightFences[currentFrame];
        device->getDevice().resetFences(*inFlightFences[currentFrame]);

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
            activeCamera.aspectRatio = static_cast<float>(width) / static_cast<float>(height);
            activeCamera.calculateViewProjectionMatrix();

        } else if (result != vk::Result::eSuccess) {
            throw std::runtime_error("failed to present swap chain image!");
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    GLFWwindow* getWindow() { return window; }
    void setWindow(GLFWwindow* pWindow) { window = pWindow; }

    Device& getDevice() { return *device; }

    const vk::Instance& getInstance() const { return *instance; }

    Swapchain& getSwapchain() { return *swapchain; }
    void cleanupSwapchain() { swapchain->cleanupSwapChain(); }

    const vk::SampleCountFlagBits& getMsaaSamples() const { return msaaSamples; }

    const uint32_t getGraphicsIndex() const { return graphicsIndex; }

    DescriptorSet& getDescriptorSet() { return *descriptorSet; }
    uint32_t getModelMatrixBufferIndex() { return modelMatrixBufferIndex; }
    std::vector<Material>& getMaterials() { return materials; }
    void addMeshToShader(Node* node, uint32_t submeshIndex, Shader shader, Material material) { shaders[shader][material][node].push_back(submeshIndex); }
    Shader getFallBackShader() { return fallbackLitShader; }
    std::vector<Mesh>& getMeshes() { return meshes; }

    uint32_t loadMeshFromFile(std::string filePath) {
        auto meshData = resourceManager->loadMeshFromFile(filePath);
        Mesh mainMesh {.sourceFile = filePath};
        for( auto mesh : meshData.subMeshes){

            uint32_t allocIndex = descriptorSet->allocateVariableBuffer<Vertex>(mesh);
            VariableBufferAllocation alloc = descriptorSet->getVariableBufferAllocation(allocIndex);
            SubMesh subMesh = {
                .vertexAllocationIndex = allocIndex,
                .vertexOffset = alloc.offset,
                .vertexCount = alloc.size,
                .vertexStride = alloc.stride
            };
            subMeshes.push_back(subMesh);
            mainMesh.subMeshes.push_back(subMeshes.size()-1);
        }
        meshes.push_back(mainMesh);
        return meshes.size()-1;
    }
    std::vector<Light>& getLights() { return lights; }

    Node* getRootNode() { return rootNode; }

    uint32_t addNode(uint32_t parentIndex = 0, glm::vec3 position = glm::vec3(0.0f), glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f), glm::vec3 scale = glm::vec3(1.0f),
                     bool keepWorldTransform = false) {
        nodes[lastNode].emplace(this, lastNode + 1, &*nodes[parentIndex], position, rotation, scale, keepWorldTransform);
        lastNode++;
        return lastNode;
    }

  private:
    GLFWwindow* window = nullptr;
    bool framebufferResized = true;
    vk::raii::Instance instance = nullptr;
    vk::raii::Context context;
    vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
    vk::raii::SurfaceKHR surface = nullptr;

    vk::raii::CommandPool commandPool = nullptr;
    std::vector<vk::raii::CommandBuffer> commandBuffers;
    uint32_t graphicsIndex = 0;

    std::vector<vk::raii::Semaphore> presentCompleteSemaphores;
    std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
    std::vector<vk::raii::Fence> inFlightFences;
    std::vector<vk::Fence> imagesInFlight;
    uint32_t currentFrame = 0;

    std::unique_ptr<Device> device;
    std::unique_ptr<Swapchain> swapchain;
    std::unique_ptr<PipelineManager> pipelineManager;
    std::unique_ptr<ResourceManager> resourceManager;
    std::unique_ptr<DescriptorSet> descriptorSet;

    std::vector<Material> materials;
    std::map<Shader, std::map<Material, std::map<Node*, std::vector<uint32_t>>>> shaders; // map between Shaders and Nodes + their submeshes to render
    Shader fallbackLitShader;
    Material fallbackDefaultMaterial;
    std::vector<Mesh> meshes;
    std::vector<SubMesh> subMeshes;
    std::vector<Light> lights;
    uint32_t modelMatrixBufferIndex;

    uint32_t defaultSamplerIndex;
    uint32_t defaultNormalIndex;

    Node* rootNode = nullptr;
    std::array<std::optional<Node>, 2048> nodes;
    uint32_t lastNode = 0;

    vk::SampleCountFlagBits msaaSamples;
    bool vSync = true;

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

    void recordCommandBuffer(uint32_t imageIndex) {

        commandBuffers[currentFrame].begin({});

        resourceManager->transitionImageLayout(&commandBuffers[currentFrame], swapchain->getSwapChainImages()[imageIndex], vk::ImageLayout::eUndefined,
                                               vk::ImageLayout::eColorAttachmentOptimal);
        // Transition the multisampled color image to COLOR_ATTACHMENT_OPTIMAL
        resourceManager->transitionImageLayout(&commandBuffers[currentFrame], swapchain->getColorImage(), vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal);

        vk::ClearValue clearColor{.color = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f})};
        vk::ClearValue clearDepth{.depthStencil = vk::ClearDepthStencilValue{1.0f, 0}};

        // Color attachment (multisampled) with resolve attachment
        vk::RenderingAttachmentInfo colorAttachment = {.imageView = swapchain->getColorImageView(),
                                                       .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                       .resolveMode = vk::ResolveModeFlagBits::eAverage,
                                                       .resolveImageView = swapchain->getSwapChainImageViews()[imageIndex],
                                                       .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                       .loadOp = vk::AttachmentLoadOp::eClear,
                                                       .storeOp = vk::AttachmentStoreOp::eStore,
                                                       .clearValue = clearColor};

        // depth attachment
        vk::RenderingAttachmentInfo depthAttachmentInfo = {.imageView = swapchain->getDepthImageView(),
                                                           .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                                           .resolveMode = vk::ResolveModeFlagBits::eMin,
                                                           .loadOp = vk::AttachmentLoadOp::eClear,
                                                           .storeOp = vk::AttachmentStoreOp::eDontCare,
                                                           .clearValue = clearDepth};

        vk::RenderingInfo renderingInfo = {.renderArea = {.offset = {0, 0}, .extent = swapchain->getSwapChainExtent()},
                                           .layerCount = 1,
                                           .colorAttachmentCount = 1,
                                           .pColorAttachments = &colorAttachment,
                                           .pDepthAttachment = &depthAttachmentInfo};

        commandBuffers[currentFrame].beginRendering(renderingInfo);
        commandBuffers[currentFrame].setViewport(
            0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapchain->getSwapChainExtent().width), static_cast<float>(swapchain->getSwapChainExtent().height), 0.0f, 1.0f));
        commandBuffers[currentFrame].setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapchain->getSwapChainExtent()));

        // draw geometry
        auto& geoPipelines = pipelineManager->getGeoPipelines();
        for (auto [shader, materials] : shaders) {
            auto currentPipeline = &(geoPipelines[shader.pipelineIndex]);
            commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, (*currentPipeline)->pipeline);
            commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, (*currentPipeline)->layout, 0, {(*currentPipeline)->descriptorSet}, {});
            for (auto [material, node_mesh] : materials) {
                for (auto [node, subMeshIndices] : node_mesh) {

                    if (meshes[node->getMeshIndex()].freed == true) { // skip if the mesh was freed
                        // TODO add function call to remove mesh from node
                        continue;
                    }
                    for (auto mesh : subMeshIndices) {
                        PushConstants pushConstants = {.vertexAllocationIndex = subMeshes[mesh].vertexAllocationIndex, // Index into vertex allocations
                                                       .vertexOffset = static_cast<uint32_t>(subMeshes[mesh].vertexOffset),                   // Byte offset in vertex buffer
                                                       .vertexStride = subMeshes[mesh].vertexStride,                   // Size of each vertex (e.g., sizeof(Vertex))
                                                       .modelMatrixIndex = node->getModelMatrixIndex(),                // Index into model matrices
                                                       .albedoTextureIndex = material.albedoTextureIndex,              // Index into textures
                                                       .roughnessTextureIndex = material.roughnessTextureIndex,
                                                       .metallicTextureIndex = material.metallicTextureIndex,
                                                       .normalTextureIndex = material.normalTextureIndex,
                                                       .samplerIndex = defaultSamplerIndex, // Index into samplers
                                                       .padding1 = 0,
                                                       .padding2 = 0,
                                                       .padding3 = 0,
                                                       .viewProjection = activeCamera.viewProjection};

                        commandBuffers[currentFrame].pushConstants<PushConstants>((*currentPipeline)->layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
                                                                                  0, pushConstants);
                        commandBuffers[currentFrame].draw(subMeshes[mesh].vertexCount, 1, 0, 0);
                    }
                }
            }
        }

        // draw GUI
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), *commandBuffers[currentFrame]);
        commandBuffers[currentFrame].endRendering();
        resourceManager->transitionImageLayout(&commandBuffers[currentFrame], swapchain->getSwapChainImages()[imageIndex], vk::ImageLayout::eColorAttachmentOptimal,
                                               vk::ImageLayout::ePresentSrcKHR);
        commandBuffers[currentFrame].end();
    }
};
