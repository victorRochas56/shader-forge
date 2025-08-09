#pragma once
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1 // for raii
#define VULKAN_HPP_NO_CONSTRUCTORS 1         // for structs constructors
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
#include <bindless_resources.hpp>
#include <debug.hpp>
#include <devices.hpp>
#include <load_resources.hpp>
#include <pipeline.hpp>
#include <structs.hpp>
#include <swapchain.hpp>
#include <utils.hpp>
///

constexpr int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector validationLayers = {"VK_LAYER_KHRONOS_validation"};

class Renderer {

  public:
    Renderer() = default;

    void initVulkan() {
        defaultCam.position = glm::vec3(1.0f, 1.0f, 1.0f);
        defaultCam.target = glm::vec3(0.0f);
        defaultCam.fov = glm::radians(45.0f);
        defaultCam.aspectRatio = 800.0f / 600.0f;
        defaultCam.nearPlane = 0.05f;
        defaultCam.farPlane = 1000.0f;
        defaultCam.calculateViewProjectionMatrix();
        createInstance();
        setupDebugMessenger();
        createSurface();
        devices = std::make_unique<Devices>(instance, requiredDeviceExtension, surface);
        msaaSamples = getMaxUsableSampleCount();
        swapChain = std::make_unique<SwapChain>(window, &*devices, &surface ,&instance);
        createCommandPool();
        resources = std::make_unique<BindlessResourceManager>(&*devices, &commandPool);
        meshLoader = std::make_unique<MeshLoader>(*resources);
        pipelineBuilder = std::make_unique<PipelineBuilder>(&*devices, &*resources, &*swapChain);
        graphicsPipeline = pipelineBuilder->createGraphicsPipeline(msaaSamples);
        linePipeline = pipelineBuilder->createLinePipeline(msaaSamples);
        swapChain->createColorResources(msaaSamples);
        swapChain->createDepthResources(msaaSamples);
        createCommandBuffers();
        createSyncObjects();
    }

    void drawFrame() {
        while (vk::Result::eTimeout == devices->getLogicalDevice().waitForFences(*inFlightFences[currentFrame], vk::True, UINT64_MAX))
            // wait for gpu to finish rendering the frame we just submitted
            ;
        auto [result, imageIndex] = swapChain->getSwapChain().acquireNextImage(UINT64_MAX, *presentCompleteSemaphores[semaphoreIndex], nullptr);

        if (result == vk::Result::eErrorOutOfDateKHR) {
            swapChain->recreateSwapChain(window,&*devices,msaaSamples);
            int width = 0, height = 0;
            glfwGetFramebufferSize(window, &width, &height);
            defaultCam.aspectRatio = static_cast<float>(width)/static_cast<float>(height);
            defaultCam.calculateViewProjectionMatrix();
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
                                        .pWaitSemaphores = &*presentCompleteSemaphores[semaphoreIndex],
                                        .pWaitDstStageMask = &waitDestinationStageMask,
                                        .commandBufferCount = 1,
                                        .pCommandBuffers = &*commandBuffers[currentFrame],
                                        .signalSemaphoreCount = 1,
                                        .pSignalSemaphores = &*renderFinishedSemaphores[semaphoreIndex]};

        devices->getGraphicsQueue().submit(submitInfo, inFlightFences[currentFrame]);

        const vk::PresentInfoKHR presentInfoKHR{.waitSemaphoreCount = 1,
                                                .pWaitSemaphores = &*renderFinishedSemaphores[semaphoreIndex],
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
            swapChain->recreateSwapChain(window,&*devices,msaaSamples);

            int width = 0, height = 0;
            glfwGetFramebufferSize(window, &width, &height);
            defaultCam.aspectRatio = static_cast<float>(width)/static_cast<float>(height);
            defaultCam.calculateViewProjectionMatrix();

        } else if (result != vk::Result::eSuccess) {
            throw std::runtime_error("failed to present swap chain image!");
        }

        semaphoreIndex = (semaphoreIndex + 1) % presentCompleteSemaphores.size();
        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void addMeshFromFile(std::string meshPath, std::string texPath = "", glm::vec3 location = glm::vec3(0.0f), glm::quat rotation = glm::quat(1.0, 0.0, 0.0, 0.0),
                         glm::vec3 scale = glm::vec3(0.5f)) {
        meshes.push_back(meshLoader->loadFromFile(meshPath, texPath, location, rotation, scale));
    }
    
    const Devices* getDevices() { return &*devices; }
    BindlessResourceManager* getResourceManager() { return &*resources;}
    void setWindow(GLFWwindow* pWindow) { window = pWindow; }
    void initializeResourceDefaults() { resources->initializeDefaults(); }
    void cleanupSwapChain() { swapChain->cleanupSwapChain(); }

  private:
    Camera defaultCam;

    GLFWwindow* window = nullptr;
    vk::raii::Context context;
    vk::raii::Instance instance = nullptr;
    vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;

    vk::raii::SurfaceKHR surface = nullptr;

    std::unique_ptr<Devices> devices = nullptr;
    std::unique_ptr<SwapChain> swapChain = nullptr;
    std::unique_ptr<BindlessResourceManager> resources = nullptr;
    std::unique_ptr<MeshLoader> meshLoader = nullptr;
    std::unique_ptr<PipelineBuilder> pipelineBuilder = nullptr;

    std::vector<Mesh> meshes;

    vk::raii::Pipeline graphicsPipeline = nullptr;
    vk::raii::Pipeline linePipeline = nullptr;
    vk::raii::CommandPool commandPool = nullptr;
    std::vector<vk::raii::CommandBuffer> commandBuffers;
    uint32_t graphicsIndex = 0;

    std::vector<vk::raii::Semaphore> presentCompleteSemaphores;
    std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
    std::vector<vk::raii::Fence> inFlightFences;
    std::vector<vk::Fence> imagesInFlight;

    uint32_t currentFrame = 0;
    uint32_t semaphoreIndex = 0;

    bool framebufferResized = false;

    uint32_t mipLevels = 0;
    vk::raii::Image textureImage = nullptr;
    vk::raii::DeviceMemory textureImageMemory = nullptr;
    vk::raii::ImageView textureImageView = nullptr;
    vk::raii::Sampler textureSampler = nullptr;

    vk::SampleCountFlagBits msaaSamples = vk::SampleCountFlagBits::e1;

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
        transition_image_layout_custom(swapChain->getColorImage(), vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal, {}, vk::AccessFlagBits2::eColorAttachmentWrite,
                                       vk::PipelineStageFlagBits2::eTopOfPipe, vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::ImageAspectFlagBits::eColor);

        // Transition the depth image to DEPTH_ATTACHMENT_OPTIMAL
        transition_image_layout_custom(swapChain->getDepthImage(), vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthAttachmentOptimal, {}, vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                                       vk::PipelineStageFlagBits2::eTopOfPipe, vk::PipelineStageFlagBits2::eEarlyFragmentTests, vk::ImageAspectFlagBits::eDepth);

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

        // drawing meshes
        commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineBuilder->getPipelineLayout(size_t(0)), 0, {resources->getDescriptorSet()},
                                                        nullptr);
        commandBuffers[currentFrame].setViewport(
            0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChain->getSwapChainExtent().width), static_cast<float>(swapChain->getSwapChainExtent().height), 0.0f, 1.0f));
        commandBuffers[currentFrame].setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChain->getSwapChainExtent()));

        for (const auto& mesh : meshes) {
            PushConstants pushConstants = {.vertexBufferIndex = mesh.vertexAllocationIndex,
                                           .vertexOffset = static_cast<uint32_t>(mesh.vertexOffset), // Convert byte offset to element offset
                                           .vertexStride = mesh.vertexStride,
                                           .modelMatrixIndex = mesh.modelMatrixIndex,
                                           .textureIndex = mesh.textureIndex,
                                           .samplerIndex = mesh.samplerIndex,
                                           .viewProjection = defaultCam.viewProjection};

            commandBuffers[currentFrame].pushConstants<PushConstants>(pipelineBuilder->getPipelineLayout(size_t(0)),
                                                                      vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pushConstants);
            commandBuffers[currentFrame].draw(mesh.vertexCount, 1, 0, 0);
        }

        // drawing lines
        commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, *linePipeline);
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineBuilder->getPipelineLayout(size_t(1)), 0, {resources->getDescriptorSet()},
                                                        nullptr);
        for (const auto& line : lines) {
            Line lineConstants = {.allocIndex = line.allocIndex, .offset = line.offset ,.stride = line.stride, .viewProjection = defaultCam.viewProjection};
            commandBuffers[currentFrame].pushConstants<Line>(pipelineBuilder->getPipelineLayout(size_t(1)), vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
                                                             0, lineConstants);
            commandBuffers[currentFrame].draw(2,1,0,0);
        }
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

        // Change: semaphores per swapchain image (for GPU-GPU sync)
        for (size_t i = 0; i < swapChain->getSwapImageSize(); i++) {
            presentCompleteSemaphores.emplace_back(devices->getLogicalDevice(), vk::SemaphoreCreateInfo());
            renderFinishedSemaphores.emplace_back(devices->getLogicalDevice(), vk::SemaphoreCreateInfo());
        }
        // Keep fences per frame (for CPU-GPU sync)
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            inFlightFences.emplace_back(devices->getLogicalDevice(), vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
        }
        // Track which fence is using each swapchain image
        imagesInFlight.resize(swapChain->getSwapImageSize(), VK_NULL_HANDLE);
    }
};