#pragma once
#ifndef VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1 // for raii
#endif
#ifndef VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_NO_CONSTRUCTORS 1 // for structs constructors
#endif

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#define GLFW_INCLUDE_VULKAN // REQUIRED only for GLFW CreateWindowSurface.
#include <GLFW/glfw3.h>

#include "devices.hpp"
#include "resources.hpp"
#include "utils.hpp"

class Swapchain {

  public:
    Swapchain(Device& device, ResourceManager& resourceManager, DescriptorSet& descriptorSet, vk::raii::SurfaceKHR& surface, vk::SampleCountFlagBits msaaSamples)
        : msaaSamples(msaaSamples), resourceManager(resourceManager), descriptorSet(descriptorSet), device(device), surface(surface) {}

    void create(GLFWwindow& window, bool& useVsync) {
        auto surfaceCapabilities = device.getPhysicalDevice().getSurfaceCapabilitiesKHR(*surface);
        swapChainImageFormat = chooseSwapSurfaceFormat(device.getPhysicalDevice().getSurfaceFormatsKHR(*surface)).format;
        swapChainExtent = chooseSwapExtent(surfaceCapabilities, &window);
        auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
        minImageCount = (surfaceCapabilities.maxImageCount > 0 && minImageCount > surfaceCapabilities.maxImageCount) ? surfaceCapabilities.maxImageCount : minImageCount;
        vk::SwapchainCreateInfoKHR swapChainCreateInfo{.surface = *surface,
                                                       .minImageCount = minImageCount,
                                                       .imageFormat = swapChainImageFormat,
                                                       .imageColorSpace = vk::ColorSpaceKHR::eSrgbNonlinear,
                                                       .imageExtent = swapChainExtent,
                                                       .imageArrayLayers = 1,
                                                       .imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst,
                                                       .imageSharingMode = vk::SharingMode::eExclusive,
                                                       .preTransform = surfaceCapabilities.currentTransform,
                                                       .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
                                                       .presentMode = chooseSwapPresentMode(device.getPhysicalDevice().getSurfacePresentModesKHR(*surface), useVsync),
                                                       .clipped = true};

        swapChain = vk::raii::SwapchainKHR(device.getDevice(), swapChainCreateInfo);
        swapChainImages = swapChain.getImages();
        createImageViews();
        createColorResources();
        createDepthResources();
    }

    void recreate(GLFWwindow* window, bool& useVsync) {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }
        device.getDevice().waitIdle();
        cleanupSwapChain();
        create(*window, useVsync);
    }

    void cleanupSwapChain() {
        swapChainImageViews.clear();
        depthImageView.clear();

        colorImageView.clear();
        colorImage.clear();
        colorImageMemory.clear();
        depthImage.clear();
        depthImageMemory.clear();
        depthResolveImageView.clear();
        depthResolveImage.clear();
        depthResolveImageMemory.clear();

        swapChain = nullptr;
    }

    const vk::raii::SwapchainKHR& getSwapChain() { return swapChain; }
    const size_t getSwapImageSize() { return swapChainImages.size(); }
    const vk::Extent2D& getSwapChainExtent() { return swapChainExtent; }
    const vk::Format& getSwapChainImageFormat() { return swapChainImageFormat; }
    const vk::raii::Image& getColorImage() { return colorImage; }
    const vk::raii::Image& getDepthImage() { return depthImage; }
    const vk::raii::ImageView& getColorImageView() { return colorImageView; }
    const vk::raii::ImageView& getDepthImageView() { return depthImageView; }
    vk::raii::ImageView& getDepthResolveImageView() { return *descriptorSet.getTextureResource(depthResolveIndex).imageView; }
    uint32_t getDepthResolveIndex() { return depthResolveIndex; }
    const std::vector<vk::raii::ImageView>& getSwapChainImageViews() { return swapChainImageViews; }
    const std::vector<vk::Image>& getSwapChainImages() { return swapChainImages; }

  private:
    Device& device;
    ResourceManager& resourceManager;
    DescriptorSet& descriptorSet;
    vk::raii::SurfaceKHR& surface;
    vk::SampleCountFlagBits msaaSamples;

    vk::raii::SwapchainKHR swapChain = nullptr;
    std::vector<vk::Image> swapChainImages;
    vk::Format swapChainImageFormat = vk::Format::eUndefined;
    vk::Extent2D swapChainExtent;
    std::vector<vk::raii::ImageView> swapChainImageViews;

    vk::raii::Image colorImage = nullptr;
    vk::raii::DeviceMemory colorImageMemory = nullptr;
    vk::raii::ImageView colorImageView = nullptr;

    vk::raii::Image depthImage = nullptr;
    vk::raii::DeviceMemory depthImageMemory = nullptr;
    vk::raii::ImageView depthImageView = nullptr;

    vk::raii::Image depthResolveImage = nullptr;
    vk::raii::DeviceMemory depthResolveImageMemory = nullptr;
    vk::raii::ImageView depthResolveImageView = nullptr;
    uint32_t depthResolveIndex = 0xFFFFFFFF;

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
            swapChainImageViews.emplace_back(device.getDevice(), imageViewCreateInfo);
        }
    }

    void createColorResources() {
        vk::Format colorFormat = swapChainImageFormat;
        resourceManager.createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, colorFormat, vk::ImageTiling::eOptimal,
                                    vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal, colorImage,
                                    colorImageMemory);
        colorImageView = resourceManager.createImageView(colorImage, colorFormat, vk::ImageAspectFlagBits::eColor, 1);
        resourceManager.transitionImageLayout(nullptr, colorImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal);
    }

    void createDepthResources() {
        vk::Format depthFormat = findSupportedFormat({vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint}, vk::ImageTiling::eOptimal,
                                                     vk::FormatFeatureFlagBits::eDepthStencilAttachment, device);

        resourceManager.createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, depthFormat, vk::ImageTiling::eOptimal,
                                    vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, depthImage,
                                    depthImageMemory);
        resourceManager.createImage(swapChainExtent.width, swapChainExtent.height, 1, vk::SampleCountFlagBits::e1, depthFormat, vk::ImageTiling::eOptimal,
                                    vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, depthResolveImage,
                                    depthResolveImageMemory);

        depthImageView = resourceManager.createImageView(depthImage, depthFormat, vk::ImageAspectFlagBits::eDepth, 1);
        depthResolveImageView = resourceManager.createImageView(depthResolveImage, depthFormat, vk::ImageAspectFlagBits::eDepth, 1);
        resourceManager.transitionImageLayout(nullptr, depthImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal);
        resourceManager.transitionImageLayout(nullptr, depthResolveImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal);

        if (depthResolveIndex == 0xFFFFFFFF) {
            // First time: allocate new slot
            depthResolveIndex = descriptorSet.allocateTexture(std::move(depthResolveImage), std::move(depthResolveImageMemory), std::move(depthResolveImageView),"internal/depthResolve");
        } else {
            // Subsequent times: update existing slot
            descriptorSet.updateTexture(depthResolveIndex, std::move(depthResolveImage), std::move(depthResolveImageMemory), std::move(depthResolveImageView),"internal/depthResolve");
        }
    }
};