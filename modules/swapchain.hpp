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
#include <cstring>
#include <devices.hpp>
#include <iostream>
#include <limits>
#include <resource_manager.hpp>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

class SwapChain {

  public:
    SwapChain(GLFWwindow* window, const Devices* devices, vk::raii::SurfaceKHR* surface, vk::raii::Instance* instance, ResourceManager* resourceManager, bool& useVsync)
        : devices(devices), surface(surface), resourceManager(resourceManager) {
        createSwapChain(*window, useVsync);
        createImageViews();
    }

    const vk::raii::SwapchainKHR& getSwapChain() { return swapChain; }
    const std::vector<vk::Image>& getSwapChainImages() { return swapChainImages; }
    const std::vector<vk::raii::ImageView>& getSwapChainImageViews() { return swapChainImageViews; }
    const vk::Format& getSwapChainImageFormat() { return swapChainImageFormat; }
    const vk::Extent2D& getSwapChainExtent() { return swapChainExtent; }
    const size_t getSwapImageSize() { return swapChainImages.size(); }
    vk::raii::Image& getDepthImage() { return depthImage; }
    vk::raii::Image& getColorImage() { return colorImage; }
    vk::raii::ImageView& getDepthImageView() { return depthImageView; }
    vk::raii::ImageView& getColorImageView() { return colorImageView; }

    void recreateSwapChain(GLFWwindow* window, const Devices* devices, vk::SampleCountFlagBits msaaSamples, bool& useVsync) {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        devices->getLogicalDevice().waitIdle();

        cleanupSwapChain();

        createSwapChain(*window, useVsync);
        createImageViews();
        createColorResources(msaaSamples);
        createDepthResources(msaaSamples);
    }

    void cleanupSwapChain() {

        swapChainImageViews.clear();
        swapChain = nullptr;
    }

    void createColorResources(vk::SampleCountFlagBits msaaSamples) {
        vk::Format colorFormat = swapChainImageFormat;

        resourceManager->createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, colorFormat, vk::ImageTiling::eOptimal,
                                     vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal, colorImage,
                                     colorImageMemory);
        colorImageView = resourceManager->createImageView(colorImage, colorFormat, vk::ImageAspectFlagBits::eColor, 1);
    }

    void createDepthResources(vk::SampleCountFlagBits msaaSamples) {
        vk::Format depthFormat = findDepthFormat(&*devices);
        resourceManager->createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, depthFormat, vk::ImageTiling::eOptimal,
                                     vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, depthImage,
                                     depthImageMemory);
        depthImageView = resourceManager->createImageView(depthImage, depthFormat, vk::ImageAspectFlagBits::eDepth, 1);

        resourceManager->transitionImageLayout(depthImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal, 1);
        resourceManager->updateImageDescriptorSet(resourceManager->getDepthDescriptorSet(), 0, depthImageView);
    }

  private:
    const Devices* devices;
    ResourceManager* resourceManager;
    vk::raii::SurfaceKHR* surface;

    vk::raii::SwapchainKHR swapChain = nullptr;
    std::vector<vk::Image> swapChainImages;
    vk::Format swapChainImageFormat = vk::Format::eUndefined;
    vk::Extent2D swapChainExtent;
    std::vector<vk::raii::ImageView> swapChainImageViews;

    vk::raii::Image depthImage = nullptr;
    vk::raii::DeviceMemory depthImageMemory = nullptr;
    vk::raii::ImageView depthImageView = nullptr;

    vk::raii::Image colorImage = nullptr;
    vk::raii::DeviceMemory colorImageMemory = nullptr;
    vk::raii::ImageView colorImageView = nullptr;

    void createSwapChain(GLFWwindow& window, bool& useVsync) {
        auto surfaceCapabilities = devices->getPhysicalDevice().getSurfaceCapabilitiesKHR(*surface);
        swapChainImageFormat = chooseSwapSurfaceFormat(devices->getPhysicalDevice().getSurfaceFormatsKHR(*surface)).format;
        swapChainExtent = chooseSwapExtent(surfaceCapabilities, &window);
        auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
        minImageCount = (surfaceCapabilities.maxImageCount > 0 && minImageCount > surfaceCapabilities.maxImageCount) ? surfaceCapabilities.maxImageCount : minImageCount;
        vk::SwapchainCreateInfoKHR swapChainCreateInfo{.surface = *surface,
                                                       .minImageCount = minImageCount,
                                                       .imageFormat = swapChainImageFormat,
                                                       .imageColorSpace = vk::ColorSpaceKHR::eSrgbNonlinear,
                                                       .imageExtent = swapChainExtent,
                                                       .imageArrayLayers = 1,
                                                       .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
                                                       .imageSharingMode = vk::SharingMode::eExclusive,
                                                       .preTransform = surfaceCapabilities.currentTransform,
                                                       .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
                                                       .presentMode = chooseSwapPresentMode(devices->getPhysicalDevice().getSurfacePresentModesKHR(*surface), useVsync),
                                                       .clipped = true};

        swapChain = vk::raii::SwapchainKHR(devices->getLogicalDevice(), swapChainCreateInfo);
        swapChainImages = swapChain.getImages();
    }

    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == vk::Format::eB8G8R8A8Srgb && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
                return availableFormat;
            }
        }
        return availableFormats[0];
    }

    vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes, bool& useVsync) {
        if (!useVsync) {
            for (const auto& availablePresentMode : availablePresentModes) {
                if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
                    return availablePresentMode;
                }
            }
        }
        return vk::PresentModeKHR::eFifo;
    }

    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities, GLFWwindow* window) {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        }
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        return {std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
                std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)};
    }

    void createImageViews() {
        vk::ImageViewCreateInfo imageViewCreateInfo{
            .viewType = vk::ImageViewType::e2D, .format = swapChainImageFormat, .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};
        for (auto image : swapChainImages) {
            imageViewCreateInfo.image = image;
            swapChainImageViews.emplace_back(devices->getLogicalDevice(), imageViewCreateInfo);
        }
    }
};
