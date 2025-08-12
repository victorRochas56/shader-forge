#pragma once
#ifndef VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1 // for raii
#endif
#ifndef VULKAN_HPP_NO_CONSTRUCTORS  
#define VULKAN_HPP_NO_CONSTRUCTORS 1 // for structs constructors
#endif
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties, const Devices* devices) {
    vk::PhysicalDeviceMemoryProperties memProperties = devices->getPhysicalDevice().getMemoryProperties();

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("failed to find suitable memory type!");
}

uint32_t getBytesPerPixel(vk::Format format) {
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

vk::Format findSupportedFormat(const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features, const Devices* devices) {
    for (const auto format : candidates) {
        vk::FormatProperties props = devices->getPhysicalDevice().getFormatProperties(format);

        if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features) {
            return format;
        }
        if (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features) {
            return format;
        }
    }
    throw std::runtime_error("failed to find supported format!");
}
vk::Format findDepthFormat(const Devices* devices) {
    return findSupportedFormat({vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint}, vk::ImageTiling::eOptimal,
                                vk::FormatFeatureFlagBits::eDepthStencilAttachment, &*devices);
}

//READING FILES///

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

void decomposeMatrix(const glm::mat4& matrix, glm::vec3& translation, glm::quat& rotation, glm::vec3& scale) {
    // Extract translation (4th column)
    translation = glm::vec3(matrix[3]);
    
    // Extract scale (length of first 3 columns)
    scale.x = glm::length(glm::vec3(matrix[0]));
    scale.y = glm::length(glm::vec3(matrix[1]));
    scale.z = glm::length(glm::vec3(matrix[2]));
    
    // Remove scaling from the matrix to extract rotation
    glm::mat3 rotMatrix = glm::mat3(matrix);
    rotMatrix[0] = glm::normalize(rotMatrix[0]);
    rotMatrix[1] = glm::normalize(rotMatrix[1]);
    rotMatrix[2] = glm::normalize(rotMatrix[2]);
    
    // Convert rotation matrix to quaternion
    rotation = glm::quat_cast(rotMatrix);
}