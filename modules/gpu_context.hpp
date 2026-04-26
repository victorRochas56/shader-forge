#pragma once
#ifndef VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#endif
#ifndef VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_NO_CONSTRUCTORS 1
#endif

#include <iostream>
#include <memory>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "constants.hpp"
#include "devices.hpp"
#include "swapchain.hpp"
#include "utils.hpp"

// Forward declarations — only needed by initSwapchain() signature.
class ResourceManager;
class DescriptorSet;

/*
GpuContext bundles the low-level Vulkan primitives that every subsystem
needs: instance, surface, device, command pool, command buffers, sync
objects, and the swapchain.

It's split into two init steps because Swapchain depends on ResourceManager
and DescriptorSet (for color/depth attachments and the bindless depth-resolve
slot), which live in BindlessSystem:

    gpu.initCore(window);                       // instance/device/pool/cmdbufs
    // ... build BindlessSystem using gpu ...
    gpu.initSwapchain(resourceMgr, descSet);    // swapchain + sync

Subsystems hold a GpuContext& (non-owning). App owns the GpuContext.
*/
class GpuContext {
  public:
    GpuContext() = default;
    ~GpuContext() = default;
    GpuContext(const GpuContext&) = delete;
    GpuContext& operator=(const GpuContext&) = delete;

    void initCore(GLFWwindow* pWindow) {
        window = pWindow;
        createInstance();
        setupDebugMessenger();
        createSurface();
        device = std::make_unique<Device>(instance, requiredDeviceExtension, surface);
        msaaSamples = getMaxUsableSampleCount(*device);
        createCommandPool();
        createCommandBuffers();
    }

    // Construct the Swapchain object. Does NOT call create() — that must be deferred
    // until after DescriptorSet::createDescriptorSet() has run, because Swapchain::create()
    // allocates a bindless slot for the depth-resolve texture.
    void initSwapchain(ResourceManager& resourceManager, DescriptorSet& descriptorSet) {
        swapchain = std::make_unique<Swapchain>(*device, resourceManager, descriptorSet, surface, msaaSamples);
    }

    // Call after DescriptorSet::createDescriptorSet() is done. Creates swapchain images
    // + per-frame sync objects.
    void createSwapchainAndSync() {
        swapchain->create(*window, vSync);
        createSyncObjects();
    }

    void recreateSwapchain() {
        swapchain->recreate(window, vSync);
        // renderFinishedSemaphores are per-image; rebuild them in case image count changed.
        renderFinishedSemaphores.clear();
        for (size_t i = 0; i < swapchain->getSwapImageSize(); i++) {
            renderFinishedSemaphores.emplace_back(device->getDevice(), vk::SemaphoreCreateInfo());
        }
        imagesInFlight.assign(swapchain->getSwapImageSize(), VK_NULL_HANDLE);
    }

    // --- accessors ------------------------------------------------------
    GLFWwindow*                     getWindow() const                  { return window; }
    const vk::raii::Instance&       getInstance() const                { return instance; }
    const vk::raii::SurfaceKHR&     getSurface() const                 { return surface; }
    Device&                         getDevice()                        { return *device; }
    const Device&                   getDevice() const                  { return *device; }
    Swapchain&                      getSwapchain()                     { return *swapchain; }
    vk::SampleCountFlagBits         getMsaaSamples() const             { return msaaSamples; }
    uint32_t                        getGraphicsIndex() const           { return graphicsIndex; }

    vk::raii::CommandPool&          getCommandPool()                   { return commandPool; }
    std::vector<vk::raii::CommandBuffer>& getCommandBuffers()          { return commandBuffers; }
    vk::raii::CommandBuffer&        getCommandBuffer(uint32_t frame)   { return commandBuffers[frame]; }

    vk::raii::Semaphore&            getPresentCompleteSemaphore(uint32_t frame)  { return presentCompleteSemaphores[frame]; }
    vk::raii::Semaphore&            getRenderFinishedSemaphore(uint32_t imgIdx)  { return renderFinishedSemaphores[imgIdx]; }
    vk::raii::Fence&                getInFlightFence(uint32_t frame)             { return inFlightFences[frame]; }
    std::vector<vk::Fence>&         getImagesInFlight()                          { return imagesInFlight; }

    // --- public per-frame state ----------------------------------------
    uint32_t currentFrame = 0;
    uint32_t totalFrames  = 0;
    bool     vSync        = true;

  private:
    GLFWwindow*                         window = nullptr;
    vk::raii::Context                   context;
    vk::raii::Instance                  instance = nullptr;
    vk::raii::DebugUtilsMessengerEXT    debugMessenger = nullptr;
    vk::raii::SurfaceKHR                surface = nullptr;

    std::unique_ptr<Device>             device;
    std::unique_ptr<Swapchain>          swapchain;

    vk::SampleCountFlagBits             msaaSamples = vk::SampleCountFlagBits::e1;
    uint32_t                            graphicsIndex = 0;

    vk::raii::CommandPool                commandPool = nullptr;
    std::vector<vk::raii::CommandBuffer> commandBuffers;

    std::vector<vk::raii::Semaphore>     presentCompleteSemaphores;
    std::vector<vk::raii::Semaphore>     renderFinishedSemaphores;
    std::vector<vk::raii::Fence>         inFlightFences;
    std::vector<vk::Fence>               imagesInFlight;

    std::vector<const char*> requiredDeviceExtension = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,           VK_KHR_SPIRV_1_4_EXTENSION_NAME,
        VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,   VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME,
        VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME};

    static inline const std::vector<const char*> validationLayers = {"VK_LAYER_KHRONOS_validation"};

    // --- init helpers --------------------------------------------------
    void createInstance() {
        constexpr vk::ApplicationInfo appInfo{.pApplicationName = "Shader Forge",
                                              .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
                                              .pEngineName = "No Engine",
                                              .engineVersion = VK_MAKE_VERSION(1, 0, 0),
                                              .apiVersion = vk::ApiVersion13};
        std::vector<char const*> requiredLayers;
#if DEBUG == 1
        requiredLayers.assign(validationLayers.begin(), validationLayers.end());
        auto layerProperties = context.enumerateInstanceLayerProperties();
        if (std::ranges::any_of(requiredLayers, [&layerProperties](auto const& requiredLayer) {
                return std::ranges::none_of(layerProperties,
                    [requiredLayer](auto const& layerProperty) { return strcmp(layerProperty.layerName, requiredLayer) == 0; });
            })) {
            throw std::runtime_error("One or more required layers are not supported!");
        }
#endif
        auto requiredExtensions = getRequiredExtensions();
        auto extensionProperties = context.enumerateInstanceExtensionProperties();
        for (auto const& requiredExtension : requiredExtensions) {
            if (std::ranges::none_of(extensionProperties,
                                     [requiredExtension](auto const& ep) { return strcmp(ep.extensionName, requiredExtension) == 0; })) {
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
        vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
                                                            vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                                                            vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
        vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags(vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                                                           vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
                                                           vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
        vk::DebugUtilsMessengerCreateInfoEXT info{.messageSeverity = severityFlags, .messageType = messageTypeFlags, .pfnUserCallback = &debugCallback};
        debugMessenger = instance.createDebugUtilsMessengerEXT(info);
    }

    static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT,
                                                          vk::DebugUtilsMessageTypeFlagsEXT type,
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
        vk::CommandBufferAllocateInfo allocInfo{.commandPool = commandPool,
                                                .level = vk::CommandBufferLevel::ePrimary,
                                                .commandBufferCount = MAX_FRAMES_IN_FLIGHT};
        commandBuffers = vk::raii::CommandBuffers(device->getDevice(), allocInfo);
    }

    void createSyncObjects() {
        presentCompleteSemaphores.clear();
        renderFinishedSemaphores.clear();
        inFlightFences.clear();
        imagesInFlight.clear();

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            presentCompleteSemaphores.emplace_back(device->getDevice(), vk::SemaphoreCreateInfo());
            inFlightFences.emplace_back(device->getDevice(), vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
        }
        for (size_t i = 0; i < swapchain->getSwapImageSize(); i++) {
            renderFinishedSemaphores.emplace_back(device->getDevice(), vk::SemaphoreCreateInfo());
        }
        imagesInFlight.resize(swapchain->getSwapImageSize(), VK_NULL_HANDLE);
    }
};
