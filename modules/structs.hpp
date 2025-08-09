#pragma once
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1 // for raii
#define VULKAN_HPP_NO_CONSTRUCTORS 1         // for structs constructors
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
#include <unordered_map>
#include <utils.hpp>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#define GLFW_INCLUDE_VULKAN // REQUIRED only for GLFW CreateWindowSurface.
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

struct Line{
    uint32_t allocIndex;
    uint32_t offset;
    uint32_t stride;
    uint32_t padding;
    glm::mat4 viewProjection;
};

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texCoord;
};

struct Point {
    glm::vec3 position;
    glm::vec3 color;
};

struct Mesh {
    uint32_t vertexAllocationIndex;
    vk::DeviceSize vertexOffset;
    uint32_t vertexCount;
    uint32_t vertexStride;

    uint32_t modelMatrixIndex;
    uint32_t textureIndex;
    uint32_t samplerIndex;
};

struct PushConstants {
    uint32_t vertexBufferIndex;
    uint32_t vertexOffset;     // Byte offset in vertex buffer
    uint32_t vertexStride;     // Size of each vertex (e.g., sizeof(Vertex))
    uint32_t modelMatrixIndex; // Index into model matrices
    uint32_t textureIndex;     // Index into textures
    uint32_t samplerIndex;     // Index into samplers
    uint32_t padding[2];       // 8 bytes - align to 16 bytes
    glm::mat4 viewProjection;
};

struct Camera {
    glm::vec3 position;
    glm::vec3 target;
    float fov;
    float aspectRatio;
    float nearPlane;
    float farPlane;
    glm::mat4 viewProjection;

    void calculateViewProjectionMatrix() {
        // === VIEW MATRIX ===
        glm::vec3 upVector = glm::vec3(0.0f, 1.0f, 0.0f); // World up direction

        // Create view matrix using lookAt
        glm::mat4 viewMatrix = glm::lookAt(position, target, upVector);
        
        // === PROJECTION MATRIX ===
            
        // Create perspective projection matrix
        glm::mat4 projectionMatrix = glm::perspective(fov, aspectRatio, nearPlane, farPlane);
        projectionMatrix[1][1] *= -1.0f; // Flip Y axis for vulkan 

        // === VIEW-PROJECTION MATRIX ===
        viewProjection = projectionMatrix * viewMatrix;
    }
};