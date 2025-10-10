#pragma once
#ifndef VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1 // for raii
#endif
#ifndef VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_NO_CONSTRUCTORS 1 // for structs constructors
#endif

#include <vector>
#include <fstream>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include "devices.hpp"

#define GLM_FORCE_RADIANS
#define GLM_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>


static vk::Format findSupportedFormat(const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features, Device& device) {
    for (const auto format : candidates) {
        vk::FormatProperties props = device.getPhysicalDevice().getFormatProperties(format);

        if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features) {
            return format;
        }
        if (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features) {
            return format;
        }
    }
    throw std::runtime_error("failed to find supported format!");
}

static uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties, Device& device) {
    vk::PhysicalDeviceMemoryProperties memProperties = device.getPhysicalDevice().getMemoryProperties();

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("failed to find suitable memory type!");
}

static vk::Format findDepthFormat(Device& device) {
    return findSupportedFormat({vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint}, vk::ImageTiling::eOptimal,
                                vk::FormatFeatureFlagBits::eDepthStencilAttachment, device);
}

static vk::SampleCountFlagBits getMaxUsableSampleCount(Device& device) {
    vk::PhysicalDeviceProperties physicalDeviceProperties = device.getPhysicalDevice().getProperties();

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


static uint32_t getBytesPerPixel(vk::Format format) {
    switch (format) {
    case vk::Format::eR8G8B8A8Srgb:
    case vk::Format::eR8G8B8A8Unorm:
    case vk::Format::eB8G8R8A8Srgb:
    case vk::Format::eB8G8R8A8Unorm:
        return 4;
    case vk::Format::eR8G8B8Srgb:
    case vk::Format::eR8G8B8Unorm:
        return 3;
    case vk::Format::eR8G8Srgb:
    case vk::Format::eR8G8Unorm:
        return 2;
    case vk::Format::eR8Srgb:
    case vk::Format::eR8Unorm:
        return 1;
    default:
        return 4; // Default fallback
    }
}

static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }
    std::vector<char> buffer(file.tellg());
    file.seekg(0, std::ios::beg);
    file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    file.close();
    return buffer;
}

static bool hasFileChanged(const std::string& filePath) {
    return false;
}

static void decomposeTransform(const glm::mat4& matrix, glm::vec3& translation, glm::quat& rotation, glm::vec3& scale) {
    // Extract translation (4th column)
    translation = glm::vec3(matrix[3]);
    
    // Extract scale (length of first 3 columns)
    scale.x = glm::length(glm::vec3(matrix[0]));
    scale.y = glm::length(glm::vec3(matrix[1]));
    scale.z = glm::length(glm::vec3(matrix[2]));
    
    // Remove scaling from the matrix to extract rotation
    glm::mat3 rotMatrix = glm::mat3(matrix);
    rotMatrix[0] = rotMatrix[0]/ scale.x;
    rotMatrix[1] = rotMatrix[1]/ scale.y;
    rotMatrix[2] = rotMatrix[2]/ scale.z;
    
    // Convert rotation matrix to quaternion
    rotation = glm::quat_cast(rotMatrix);
}

static glm::mat4 makeTransform( glm::vec3 translation, glm::quat rotation, glm::vec3 scale) {
    glm::mat4 T = glm::translate(glm::mat4(1.0f), translation);
    glm::mat4 R = glm::mat4_cast(rotation);
    glm::mat4 S = glm::scale(glm::mat4(1.0f), scale);
    
    return T * R * S;
}

