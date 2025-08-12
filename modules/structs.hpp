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

struct Line {
    glm::mat4 viewProjection;
    uint32_t allocIndex;
    uint32_t offset;
    uint32_t stride;
    uint32_t padding;
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
    uint32_t padding1;
    uint32_t padding2;
    glm::mat4 viewProjection;
    glm::vec4 cameraPos;
};

struct Transform {
    glm::vec3 position = glm::vec3(0);
    glm::quat rotation = glm::quat(1, 0, 0, 0);
    glm::vec3 scale = glm::vec3(1);
    glm::mat4 transformMatrix = glm::mat4(1);

    void updateTransform(glm::vec3 position = glm::vec3(0), glm::quat rotation = glm::quat(1, 0, 0, 0), glm::vec3 scale = glm::vec3(1), glm::mat4 relativeMatrix = glm::mat4(1)) {
        position = glm::vec3(relativeMatrix * glm::vec4(position, 1));
        scale = glm::vec3(relativeMatrix * glm::vec4(scale, 1));

        glm::mat3 rotationMatrix = glm::mat3(relativeMatrix);
        glm::quat relativeRotation = glm::quat_cast(rotationMatrix);
        rotation *= relativeRotation;

        updateTransformMatrix();
    }

    void updateTransformMatrix() {

        glm::mat4 translationMatrix = glm::translate(glm::mat4(1.0f), position);
        glm::mat4 rotationMatrix = glm::mat4_cast(rotation);
        glm::mat4 scaleMatrix = glm::scale(glm::mat4(1.0f), scale);

        transformMatrix = translationMatrix * rotationMatrix * scaleMatrix;
    }
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

struct Camera {
    glm::vec3 position;
    glm::vec3 target;
    float fov;
    float aspectRatio;
    float nearPlane;
    float farPlane;
    glm::mat4 viewProjection;
    float yaw = -90.0f;
    float pitch = 0.0f;

    void rotateYaw(float deltaYaw = 0.0f) {
        yaw += deltaYaw;
        // we keep yaw in 0-360 range
        if (yaw > 360.0f)
            yaw -= 360.0f;
        if (yaw < -360.0f)
            yaw += 360.0f;
        updateTarget();
    }

    void rotatePitch(float deltaPitch = 0.0f) {
        pitch += deltaPitch;
        // Constrain pitch to prevent gimbal lock
        if (pitch > 89.0f)
            pitch = 89.0f;
        if (pitch < -89.0f)
            pitch = -89.0f;
        updateTarget();
    }

    void moveCamera(glm::vec3 deltaPosition) {
        glm::vec3 forward = glm::normalize(target - position);
        position += forward * deltaPosition.z;
        position += glm::cross(forward, glm::vec3(0.0, 1.0, 0.0)) * deltaPosition.x;
        updateTarget();
    }

    // update target based on current yaw and pitch
    void updateTarget() {
        glm::vec3 direction;
        direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        direction.y = sin(glm::radians(pitch));
        direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        direction = glm::normalize(direction);
        target = position + direction;
        calculateViewProjectionMatrix();
    }

    void lookAt(const glm::vec3& targetPos) {
        // Calculate yaw and pitch from the direction vector
        target = targetPos;
        glm::vec3 direction = glm::normalize(target - position);
        yaw = glm::degrees(atan2(direction.z, direction.x));
        pitch = glm::degrees(asin(direction.y));
        calculateViewProjectionMatrix();
    }

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

