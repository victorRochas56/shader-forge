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
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utils.hpp>
#include <vector>

#define GLM_FORCE_RADIANS
#define GLM_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

struct LineAlloc {
    glm::mat4 viewProjection;
    uint32_t allocIndex;
    uint32_t offset;
    uint32_t stride;
    uint32_t padding;
};

struct Line {
    LineAlloc alloc;
    glm::vec3 startPoint;
    glm::vec3 endPoint;
    glm::vec4 color;
};

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texCoord;
    glm::vec3 tangent;
};

struct Point {
    glm::vec4 position;
    glm::vec4 color;
};

struct Mesh {
    uint32_t vertexAllocationIndex;
    vk::DeviceSize vertexOffset;
    uint32_t vertexCount;
    uint32_t vertexStride;
    uint32_t modelMatrixIndex;
    uint32_t albedoTextureIndex;
    uint32_t roughnessTextureIndex;
    uint32_t metallicTextureIndex;
    uint32_t normalTextureIndex;
    uint32_t samplerIndex;
};

struct PushConstants {
    uint32_t vertexBufferIndex;
    uint32_t vertexOffset;       // Byte offset in vertex buffer
    uint32_t vertexStride;       // Size of each vertex (e.g., sizeof(Vertex))
    uint32_t modelMatrixIndex;   // Index into model matrices
    uint32_t albedoTextureIndex; // Index into textures
    uint32_t roughnessTextureIndex;
    uint32_t metallicTextureIndex;
    uint32_t normalTextureIndex;
    uint32_t samplerIndex; // Index into samplers
    uint32_t lightCount;
    uint32_t environmentMapIndex;
    uint32_t debugMaps;
    glm::mat4 viewProjection;
    glm::vec4 cameraPos;
};

struct SkyBoxConstants {
    uint32_t skyboxIndex;
    uint32_t padding1;
    uint32_t padding2;
    uint32_t padding3;
    glm::mat4 invViewProjMatrix;
};

struct DepthPushConstants {
    float nearPlane;
    float farPlane;
    uint32_t linearize;
    uint32_t padding1;
};

struct Light {
    glm::vec4 position = glm::vec4(0, 0, 0, 1);
    glm::vec4 color = glm::vec4(0, 0, 0, 1);
    float range = 0;
    float intensity = 0;
    uint32_t type;
    uint32_t allocationIndex;

    void reset() {
        position = glm::vec4(0, 0, 0, 1);
        range = 0;
        intensity = 0;
        color = glm::vec4(0, 0, 0, 1);
    }
};
