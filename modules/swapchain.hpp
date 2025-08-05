#pragma once
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1

#define VULKAN_HPP_NO_CONSTRUCTORS 1
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <devices.hpp>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

class SwapChain {

  public:
    SwapChain(GLFWwindow* window, Devices* devices, vk::raii::SurfaceKHR* surface) {
        createSwapChain(*window, *devices, *surface);
        createImageViews(*devices);
    }

    const vk::Format& getSwapChainImageFormat() { return swapChainImageFormat; }
    const vk::Extent2D& getSwapChainExtent() { return swapChainExtent; }
    const size_t getSwapImageSize() { return swapChainImages.size(); }
    
    void cleanupSwapChain() {
        
        swapChainImageViews.clear();
        swapChain = nullptr;
    }

  private:
    vk::raii::SwapchainKHR swapChain = nullptr;
    std::vector<vk::Image> swapChainImages;
    vk::Format swapChainImageFormat = vk::Format::eUndefined;
    vk::Extent2D swapChainExtent;
    std::vector<vk::raii::ImageView> swapChainImageViews;

    void createSwapChain(GLFWwindow& window, Devices& devices, vk::raii::SurfaceKHR& surface) {
        auto surfaceCapabilities = devices.getPhysicalDevice().getSurfaceCapabilitiesKHR(*surface);
        swapChainImageFormat = chooseSwapSurfaceFormat(devices.getPhysicalDevice().getSurfaceFormatsKHR(*surface)).format;
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
                                                       .presentMode = chooseSwapPresentMode(devices.getPhysicalDevice().getSurfacePresentModesKHR(*surface)),
                                                       .clipped = true};

        swapChain = vk::raii::SwapchainKHR(devices.getLogicalDevice(), swapChainCreateInfo);
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

    vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) {
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
                return availablePresentMode;
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

    void createImageViews(Devices& devices) {
        vk::ImageViewCreateInfo imageViewCreateInfo{
            .viewType = vk::ImageViewType::e2D, .format = swapChainImageFormat, .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};
        for (auto image : swapChainImages) {
            imageViewCreateInfo.image = image;
            swapChainImageViews.emplace_back(devices.getLogicalDevice(), imageViewCreateInfo);
        }
    }
};
