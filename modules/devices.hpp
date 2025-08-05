#pragma once
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#define VULKAN_HPP_NO_CONSTRUCTORS 1
#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

class Devices {

  public:
    // constructor picks physical device and initialized logical device
    Devices(vk::raii::Instance& instance, std::vector<const char*> requiredDeviceExtension, vk::raii::SurfaceKHR& surface) {
        pickPhysicalDevice(instance, requiredDeviceExtension, surface);
        createLogicalDevice(instance, requiredDeviceExtension, surface);
    }

    const vk::raii::PhysicalDevice& getPhysicalDevice() const { return physicalDevice; }
    const vk::raii::Device& getLogicalDevice() const { return logicalDevice; }
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
                // Check if the device supports the Vulkan 1.3 API version
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

                return supportsVulkan1_3 && supportsGraphics && supportsAllRequiredExtensions && supportsRequiredFeatures && hasPresentSupport;
            });

        if (devIter != devices.end()) {
            physicalDevice = std::move(*devIter);
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
            // the graphicsIndex doesn't support present -> look for another family index that supports both
            // graphics and present
            for (size_t i = 0; i < queueFamilyProperties.size(); i++) {
                if ((queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics) && physicalDevice.getSurfaceSupportKHR(static_cast<uint32_t>(i), *surface)) {
                    graphicsIndex = static_cast<uint32_t>(i);
                    presentIndex = graphicsIndex;
                    break;
                }
            }
            if (presentIndex == std::numeric_limits<uint32_t>::max()) {
                // there's nothing like a single family index that supports both graphics and present -> look for another
                // family index that supports present
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

        vk::PhysicalDeviceDescriptorIndexingFeatures descriptorIndexingFeature = {
            .sType = vk::StructureType::ePhysicalDeviceDescriptorIndexingFeatures, .descriptorBindingPartiallyBound = true, .descriptorBindingVariableDescriptorCount = true};
        // query for Vulkan 1.3 features
        vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
                           vk::PhysicalDeviceDescriptorIndexingFeatures>
            featureChain = {
                {.features = {.sampleRateShading = true, .samplerAnisotropy = true}}, // vk::PhysicalDeviceFeatures2
                {.synchronization2 = true, .dynamicRendering = true},                 // vk::PhysicalDeviceVulkan13Features
                {.extendedDynamicState = true},                                       // vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT
                {.sType = vk::StructureType::ePhysicalDeviceDescriptorIndexingFeatures, .descriptorBindingPartiallyBound = true, .descriptorBindingVariableDescriptorCount = true}};

        // create a Device
        float queuePriority = 0.0f;
        vk::DeviceQueueCreateInfo deviceQueueCreateInfo{.queueFamilyIndex = graphicsIndex, .queueCount = 1, .pQueuePriorities = &queuePriority};
        vk::DeviceCreateInfo deviceCreateInfo{.pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
                                              .queueCreateInfoCount = 1,
                                              .pQueueCreateInfos = &deviceQueueCreateInfo,
                                              .enabledExtensionCount = static_cast<uint32_t>(requiredDeviceExtension.size()),
                                              .ppEnabledExtensionNames = requiredDeviceExtension.data()};

        logicalDevice = vk::raii::Device(physicalDevice, deviceCreateInfo);
        graphicsQueue = vk::raii::Queue(logicalDevice, graphicsIndex, 0);
        presentQueue = vk::raii::Queue(logicalDevice, presentIndex, 0);
    }
};