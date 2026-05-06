#pragma once
#ifndef VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1 // for raii
#endif
#ifndef VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_NO_CONSTRUCTORS 1 // for structs constructors
#endif

#include <array>
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

/*
the dreaded pile of random functions
these will eventually be put into appropriate .cpp or .hpp files (TODO)
but for now here they remain
*/

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

inline bool intersectPlane(const glm::vec3 &normal, const glm::vec3 &planeCenter, const glm::vec3 &rayOrigin, const glm::vec3 &rayDir, float& outParamDist)
{
    // Assuming vectors are all normalized
    float denom = glm::dot(normal, rayDir);
    if (std::abs(denom) > 1e-6f) {
        glm::vec3 planeRayDir = planeCenter - rayOrigin;
        outParamDist = glm::dot(planeRayDir, normal) / denom;
        return (outParamDist >= 0);
    }
    return false;
}

inline bool intersectDisk(const glm::vec3 &normal, const glm::vec3 &planeCenter, const float &radius, const glm::vec3 &rayOrigin, const glm::vec3 &rayDir)
{
    float t = 0;
    if (intersectPlane(normal, planeCenter, rayOrigin, rayDir, t)) {
        glm::vec3 p = rayOrigin + rayDir * t; // Calculate intersection point
        glm::vec3 v = p - planeCenter; // Vector from disk center to intersection point
        float d2 = glm::dot(v, v); // Squared distance from disk center to intersection point
        return d2 <= radius * radius; // Compare squared distances (more efficient)
    }
    return false;
}

inline bool intersectCircle(const glm::vec3 &normal, const glm::vec3 &planeCenter, const float &radius, const float& thickness, const glm::vec3 &rayOrigin, const glm::vec3 &rayDir, float& outParamDist)
{
    float t = 0;
    if (intersectPlane(normal, planeCenter, rayOrigin, rayDir, t)) {
        glm::vec3 p = rayOrigin + rayDir * t; // Calculate intersection point
        glm::vec3 v = p - planeCenter; // Vector from disk center to intersection point
        float d2 = glm::dot(v, v); // Squared distance from disk center to intersection point
        bool condA = d2 <= (radius + thickness) * (radius + thickness); // Compare squared distances (more efficient)
        bool condB = d2 >= (radius - thickness) * (radius - thickness); // Compare squared distances (more efficient)
        if (condA && condB) {
            outParamDist = t;
            return true;
        }
        outParamDist = 0;
        return false;
    }
    return false;
}

inline bool intersectCylinder(/*cylinder desc*/ const glm::vec3& base, const glm::vec3& axis, const float radius, const float height,/*ray desc*/ glm::vec3& rayOrigin, glm::vec3& rayDirection, float& outParamDist) {
    // axis assumed to be a unit vector; cylinder spans [base, base + axis*height]
    glm::vec3 oc = rayOrigin - base;

    float dPar  = glm::dot(rayDirection, axis);
    float ocPar = glm::dot(oc, axis);
    glm::vec3 dPerp  = rayDirection - axis * dPar;
    glm::vec3 ocPerp = oc - axis * ocPar;

    float a = glm::dot(dPerp, dPerp);
    float b = 2.0f * glm::dot(dPerp, ocPerp);
    float c = glm::dot(ocPerp, ocPerp) - radius * radius;

    float tBest = std::numeric_limits<float>::max();
    bool  hit   = false;

    // Side surface (skip when ray is parallel to the axis -> a ~= 0)
    if (a > 1e-8f) {
        float disc = b * b - 4.0f * a * c;
        if (disc >= 0.0f) {
            float sq  = std::sqrt(disc);
            float inv = 0.5f / a;
            float t0  = (-b - sq) * inv;
            float t1  = (-b + sq) * inv;
            for (float t : {t0, t1}) {
                if (t < 0.0f || t >= tBest) continue;
                float y = ocPar + t * dPar;
                if (y >= 0.0f && y <= height) { tBest = t; hit = true; }
            }
        }
    }

    // End caps
    auto testCap = [&](const glm::vec3& center) {
        float denom = glm::dot(axis, rayDirection);
        if (std::abs(denom) < 1e-6f) return;
        float t = glm::dot(center - rayOrigin, axis) / denom;
        if (t < 0.0f || t >= tBest) return;
        glm::vec3 p = rayOrigin + rayDirection * t;
        glm::vec3 v = p - center;
        if (glm::dot(v, v) <= radius * radius) { tBest = t; hit = true; }
    };
    testCap(base);
    testCap(base + axis * height);

    if (hit) outParamDist = tBest;
    return hit;
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

static glm::mat4 makeTransform( glm::vec3 translation, glm::quat rotation = glm::quat(1,0,0,0), glm::vec3 scale = glm::vec3(1.0f)) {
    glm::mat4 T = glm::translate(glm::mat4(1.0f), translation);
    glm::mat4 R = glm::mat4_cast(rotation);
    glm::mat4 S = glm::scale(glm::mat4(1.0f), scale);
    
    return T * R * S;
}

// Frustum culling structures and functions
struct Plane {
    glm::vec3 normal;
    float distance;
};

// Extracts the 6 frustum planes from a light space matrix
inline std::array<Plane, 6> extractFrustumPlanes(const glm::mat4& lightSpaceMatrix) {
    std::array<Plane, 6> planes;

    // Left plane
    planes[0].normal = glm::vec3(lightSpaceMatrix[0][3] + lightSpaceMatrix[0][0], lightSpaceMatrix[1][3] + lightSpaceMatrix[1][0], lightSpaceMatrix[2][3] + lightSpaceMatrix[2][0]);
    planes[0].distance = lightSpaceMatrix[3][3] + lightSpaceMatrix[3][0];

    // Right plane
    planes[1].normal = glm::vec3(lightSpaceMatrix[0][3] - lightSpaceMatrix[0][0], lightSpaceMatrix[1][3] - lightSpaceMatrix[1][0], lightSpaceMatrix[2][3] - lightSpaceMatrix[2][0]);
    planes[1].distance = lightSpaceMatrix[3][3] - lightSpaceMatrix[3][0];

    // Bottom plane
    planes[2].normal = glm::vec3(lightSpaceMatrix[0][3] + lightSpaceMatrix[0][1], lightSpaceMatrix[1][3] + lightSpaceMatrix[1][1], lightSpaceMatrix[2][3] + lightSpaceMatrix[2][1]);
    planes[2].distance = lightSpaceMatrix[3][3] + lightSpaceMatrix[3][1];

    // Top plane
    planes[3].normal = glm::vec3(lightSpaceMatrix[0][3] - lightSpaceMatrix[0][1], lightSpaceMatrix[1][3] - lightSpaceMatrix[1][1], lightSpaceMatrix[2][3] - lightSpaceMatrix[2][1]);
    planes[3].distance = lightSpaceMatrix[3][3] - lightSpaceMatrix[3][1];

    // Near plane (Vulkan/GLM_DEPTH_ZERO_TO_ONE: depth range [0,1], near at z_ndc=0, so just row2)
    planes[4].normal = glm::vec3(lightSpaceMatrix[0][2], lightSpaceMatrix[1][2], lightSpaceMatrix[2][2]);
    planes[4].distance = lightSpaceMatrix[3][2];

    // Far plane
    planes[5].normal = glm::vec3(lightSpaceMatrix[0][3] - lightSpaceMatrix[0][2], lightSpaceMatrix[1][3] - lightSpaceMatrix[1][2], lightSpaceMatrix[2][3] - lightSpaceMatrix[2][2]);
    planes[5].distance = lightSpaceMatrix[3][3] - lightSpaceMatrix[3][2];

    // Normalize all planes
    for (auto& plane : planes) {
        float length = glm::length(plane.normal);
        plane.normal /= length;
        plane.distance /= length;
    }

    return planes;
}

inline bool isAABBInFrustum(const glm::vec3& aabbMin, const glm::vec3& aabbMax, const std::array<Plane, 6>& planes, float epsilon = 0.01f) {
    // Test the AABB against each plane
    for (const auto& plane : planes) {
        glm::vec3 positiveVertex;
        positiveVertex.x = (plane.normal.x >= 0.0f) ? aabbMax.x : aabbMin.x;
        positiveVertex.y = (plane.normal.y >= 0.0f) ? aabbMax.y : aabbMin.y;
        positiveVertex.z = (plane.normal.z >= 0.0f) ? aabbMax.z : aabbMin.z;

        // Negative epsilon makes the frustum "bigger" (more conservative culling)
        if (glm::dot(plane.normal, positiveVertex) + plane.distance < -epsilon) {
            return false;
        }
    }

    return true;
}

// Helper function to transform a local-space AABB to world space
inline void transformAABBToWorldSpace(const glm::vec3& localMin, const glm::vec3& localMax,
                                      const glm::mat4& worldTransform,
                                      glm::vec3& worldMin, glm::vec3& worldMax) {
    // Transform all 8 corners and find new AABB in world space
    glm::vec3 corners[8] = {
        glm::vec3(localMin.x, localMin.y, localMin.z),
        glm::vec3(localMax.x, localMin.y, localMin.z),
        glm::vec3(localMin.x, localMax.y, localMin.z),
        glm::vec3(localMax.x, localMax.y, localMin.z),
        glm::vec3(localMin.x, localMin.y, localMax.z),
        glm::vec3(localMax.x, localMin.y, localMax.z),
        glm::vec3(localMin.x, localMax.y, localMax.z),
        glm::vec3(localMax.x, localMax.y, localMax.z)
    };

    worldMin = glm::vec3(std::numeric_limits<float>::max());
    worldMax = glm::vec3(std::numeric_limits<float>::lowest());

    for (const auto& corner : corners) {
        glm::vec3 worldCorner = glm::vec3(worldTransform * glm::vec4(corner, 1.0f));
        worldMin = glm::min(worldMin, worldCorner);
        worldMax = glm::max(worldMax, worldCorner);
    }
}