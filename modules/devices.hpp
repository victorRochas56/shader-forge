#pragma once
#ifndef VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1 // for raii
#endif
#ifndef VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_NO_CONSTRUCTORS 1 // for structs constructors
#endif

#include <cassert>
#include <iostream>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

// part of the vulkan initialization, class holds the found physical and graphics devices to access them for rendering
class Device {
  public:
    Device(vk::raii::Instance& instance, std::vector<const char*> requiredDeviceExtension, vk::raii::SurfaceKHR& surface) {
        pickPhysicalDevice(instance, requiredDeviceExtension, surface);
        createLogicalDevice(instance, requiredDeviceExtension, surface);
    }

    const vk::raii::Device& getDevice() const { return logicalDevice; }
    const vk::raii::PhysicalDevice& getPhysicalDevice() const { return physicalDevice; }
    const vk::raii::Queue& getGraphicsQueue() const { return graphicsQueue; }
    const vk::raii::Queue& getPresentQueue() const { return presentQueue; }

  private:
    vk::raii::PhysicalDevice physicalDevice = nullptr;
    vk::raii::Device logicalDevice = nullptr;
    vk::raii::Queue graphicsQueue = nullptr;
    vk::raii::Queue presentQueue = nullptr;

    void pickPhysicalDevice(vk::raii::Instance& instance, std::vector<const char*> requiredDeviceExtension, vk::raii::SurfaceKHR& surface) {
        std::vector<vk::raii::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
        const auto devIter = std::ranges::find_if( // iterates through all devices and checks for a suitable one
            devices, [&](auto const& device) {
                // Check if the device supports the Vulkan 1.3
                bool supportsVulkan1_3 = device.getProperties().apiVersion >= VK_API_VERSION_1_3;

                // Check if any of the queue families support graphics operations
                auto queueFamilies = device.getQueueFamilyProperties();
                bool supportsGraphics = std::ranges::any_of(queueFamilies, [](auto const& qfp) { return !!(qfp.queueFlags & vk::QueueFlagBits::eGraphics); });

                // Check if all required device extensions are available
                auto availableDeviceExtensions = device.enumerateDeviceExtensionProperties();
                bool supportsAllRequiredExtensions = std::ranges::all_of(requiredDeviceExtension, [&availableDeviceExtensions](auto const& requiredDeviceExtension) {
                    return std::ranges::any_of(availableDeviceExtensions, [requiredDeviceExtension](auto const& availableDeviceExtension) {
                        return strcmp(availableDeviceExtension.extensionName, requiredDeviceExtension) == 0;
                    });
                });

                // check for the required features
                auto features = device.template getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
                bool supportsRequiredFeatures = features.template get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy &&
                                                features.template get<vk::PhysicalDeviceFeatures2>().features.sampleRateShading &&
                                                features.template get<vk::PhysicalDeviceFeatures2>().features.multiDrawIndirect &&
                                                features.template get<vk::PhysicalDeviceVulkan13Features>().synchronization2 &&
                                                features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
                                                features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState;

                // Check for surface support
                bool hasPresentSupport = false;
                for (uint32_t i = 0; i < queueFamilies.size(); i++) {
                    if (device.getSurfaceSupportKHR(i, *surface)) {
                        hasPresentSupport = true;
                        break;
                    }
                }
#if DEBUG == 1
                std::cout << "hasPresentSupport: " << hasPresentSupport << std::endl;
                std::cout << "supportsVulkan1_3: " << supportsVulkan1_3 << std::endl;
                std::cout << "supportsGraphics: " << supportsGraphics << std::endl;
                std::cout << "supportsAllRequiredExtensions: " << supportsAllRequiredExtensions << std::endl;
                std::cout << "supportsRequiredFeatures: " << supportsRequiredFeatures << std::endl;
#endif
                return supportsVulkan1_3 && supportsGraphics && supportsAllRequiredExtensions && supportsRequiredFeatures && hasPresentSupport;
            });

        if (devIter != devices.end()) {
            physicalDevice = std::move(*devIter);
            std::cout << "running on device: " << physicalDevice.getProperties().deviceName << std::endl;
            for(auto& presentMode : physicalDevice.getSurfacePresentModesKHR(surface)){
                std::cout << "available present modes: " << vk::to_string(presentMode) << std::endl;
            }
        } else {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }

    std::vector<const char*> getRequiredExtensions() {
        uint32_t glfwExtensionCount = 0;
        auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

        return extensions;
    }

    void createLogicalDevice(vk::raii::Instance& instance, std::vector<const char*> requiredDeviceExtension, vk::raii::SurfaceKHR& surface) {
        // find the index of the first queue family that supports graphics
        std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

        // get the first index into queueFamilyProperties which supports graphics
        auto graphicsQueueFamilyProperty =
            std::ranges::find_if(queueFamilyProperties, [](auto const& qfp) { return (qfp.queueFlags & vk::QueueFlagBits::eGraphics) != static_cast<vk::QueueFlags>(0); });

        auto graphicsIndex = static_cast<uint32_t>(std::distance(queueFamilyProperties.begin(), graphicsQueueFamilyProperty));

        // determine a queueFamilyIndex that supports present
        // first check if the graphicsIndex is good enough
        uint32_t presentIndex = physicalDevice.getSurfaceSupportKHR(graphicsIndex, *surface) ? graphicsIndex : std::numeric_limits<uint32_t>::max();
        if (presentIndex == std::numeric_limits<uint32_t>::max()) {
            for (size_t i = 0; i < queueFamilyProperties.size(); i++) {
                if ((queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics) && physicalDevice.getSurfaceSupportKHR(static_cast<uint32_t>(i), *surface)) {
                    graphicsIndex = static_cast<uint32_t>(i);
                    presentIndex = graphicsIndex;
                    break;
                }
            }
            if (presentIndex == std::numeric_limits<uint32_t>::max()) {
                for (size_t i = 0; i < queueFamilyProperties.size(); i++) {
                    if (physicalDevice.getSurfaceSupportKHR(static_cast<uint32_t>(i), *surface)) {
                        presentIndex = static_cast<uint32_t>(i);
                        break;
                    }
                }
            }
        }

        if ((graphicsIndex == queueFamilyProperties.size()) || (presentIndex == std::numeric_limits<uint32_t>::max())) {
            throw std::runtime_error("Could not find a queue for graphics or present -> terminating");
        }

        // query for Vulkan 1.3 features
        vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features, vk::PhysicalDeviceVulkan12Features, vk::PhysicalDeviceVulkan13Features,
                           vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>
            featureChain = {
                {.features =
                     {.sampleRateShading = true, .multiDrawIndirect = true, .drawIndirectFirstInstance = true, .fillModeNonSolid = true, .wideLines = true, .samplerAnisotropy = true, .shaderInt64 = true}}, // vk::PhysicalDeviceFeatures2
                {.shaderDrawParameters = true},
                {.shaderInt8 = true,
                 .descriptorIndexing = true,
                 .descriptorBindingSampledImageUpdateAfterBind = true,
                 .descriptorBindingStorageBufferUpdateAfterBind = true,
                 .descriptorBindingUpdateUnusedWhilePending = true,
                 .descriptorBindingPartiallyBound = true,
                 .descriptorBindingVariableDescriptorCount = true,
                 .runtimeDescriptorArray = true,
                 //.scalarBlockLayout = true,
                 .bufferDeviceAddress = true},
                {.synchronization2 = true, .dynamicRendering = true}, // vk::PhysicalDeviceVulkan13Features
                {.extendedDynamicState = true}};                      // vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT

        // create a Device
        float queuePriority = 0.0f;
        vk::DeviceQueueCreateInfo deviceQueueCreateInfo{.queueFamilyIndex = graphicsIndex, .queueCount = 1, .pQueuePriorities = &queuePriority};
        vk::DeviceCreateInfo deviceCreateInfo{.pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
                                              .queueCreateInfoCount = 1,
                                              .pQueueCreateInfos = &deviceQueueCreateInfo,
                                              .enabledExtensionCount = static_cast<uint32_t>(requiredDeviceExtension.size()),
                                              .ppEnabledExtensionNames = requiredDeviceExtension.data()};

        logicalDevice = vk::raii::Device(physicalDevice, deviceCreateInfo);
        assert(physicalDevice.getProperties().limits.maxPushConstantsSize >= 256 && "GPU must support at least 256 bytes of push constants for BDA");
        graphicsQueue = vk::raii::Queue(logicalDevice, graphicsIndex, 0);
        presentQueue = vk::raii::Queue(logicalDevice, presentIndex, 0);
    }
};