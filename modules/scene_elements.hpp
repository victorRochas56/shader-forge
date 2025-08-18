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
#include <resource_manager.hpp>
#include <stdexcept>
#include <string>
#include <structs.hpp>
#include <unordered_map>
#include <utils.hpp>
#include <vector>

#define GLM_FORCE_RADIANS
#define GLM_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

constexpr uint32_t MAX_LINES = 2048;

std::unique_ptr<std::array<LineAlloc, MAX_LINES>> lineAllocs = std::make_unique<std::array<LineAlloc, MAX_LINES>>();
std::unique_ptr<std::array<Line, MAX_LINES>> lines = std::make_unique<std::array<Line, MAX_LINES>>();
std::unique_ptr<std::array<uint32_t, MAX_LINES>> lineUsage = std::make_unique<std::array<uint32_t, MAX_LINES>>(std::array<uint32_t, MAX_LINES>{});

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

struct Gizmo {
    glm::vec4 color;
    uint32_t lineIndices[32] = {};
    ResourceManager* resourceManager;

    void updateTransform(glm::mat4 worldTransform) {
        for (int i = 0; i < 32; i++) {
            uint32_t index = lineIndices[i];
            if ((*lineUsage)[index] != 0) {
                glm::vec3 scale;
                scale.x = glm::length(glm::vec3(worldTransform[0]));
                scale.y = glm::length(glm::vec3(worldTransform[1]));
                scale.z = glm::length(glm::vec3(worldTransform[2]));
                glm::vec3 startPoint = glm::vec3(worldTransform * glm::vec4((*lines)[index].startPoint / scale, 1.0));
                glm::vec3 endPoint = glm::vec3(worldTransform * glm::vec4((*lines)[index].endPoint / scale, 1.0));
                Point lineData[2] = {{.position = glm::vec4(startPoint, 1.0), .color = (*lines)[index].color},
                                     {.position = glm::vec4(endPoint, 1.0), .color = (*lines)[index].color}};
                resourceManager->updateVertexBuffer((*lines)[index].alloc.allocIndex, lineData, 2 * sizeof(Point));
            }
        }
    }
    uint32_t addLine(glm::vec3 startPoint, glm::vec3 endPoint, glm::vec4 color) {
        int i = 0;
        for (i = 0; i < MAX_LINES; i++) {
            if ((*lineUsage)[i] == 0) {
                std::vector<Point> lineData{{.position = glm::vec4(startPoint, 1.0), .color = color}, {.position = glm::vec4(endPoint, 1.0), .color = color}};
                auto vertexInfo = resourceManager->allocateVertexBuffer(lineData.data(), 2 * sizeof(Point), 2, sizeof(Point));
                LineAlloc alloc{.allocIndex = vertexInfo.allocationIndex, .offset = static_cast<uint32_t>(vertexInfo.offset), .stride = sizeof(Point)};
                Line line{.alloc = alloc, .startPoint = startPoint, .endPoint = endPoint, .color = color};
                (*lineAllocs)[i] = alloc;
                (*lines)[i] = line;
                (*lineUsage)[i] = 1;
                return i;
            }
        }
        if (i == MAX_LINES) {
            throw std::runtime_error("too many lines!");
        }
        return MAX_LINES;
    };
    void addAxes(glm::vec3 origin, float size) {
        lineIndices[0] = addLine(origin, origin + glm::vec3(0, size, 0), glm::vec4(0.0, 1.0, 0.0, 1.0));
        lineIndices[1] = addLine(origin, origin + glm::vec3(0, 0, size), glm::vec4(0.0, 0.0, 1.0, 1.0));
        lineIndices[2] = addLine(origin, origin + glm::vec3(size, 0, 0), glm::vec4(1.0, 0.0, 0.0, 1.0));
    }
};

struct Node {
    std::string name = "empty";
    glm::mat4 worldTransform = glm::mat4(1);
    Transform transform;

    ResourceManager* resourceManager;

    std::array<Mesh, MAX_VERTEX_ALLOCATIONS>* meshes = nullptr;
    uint32_t meshIndex;
    std::array<Light, MAX_LIGHTS>* lights = nullptr;
    uint32_t lightIndex;

    Gizmo gizmo;
    bool hasGizmo = false;

    Node* parent = nullptr;
    std::vector<Node*> children;

    void showInfo() {
        ImGui::Begin("selected node");
        ImGui::Text(name.c_str());
        ImGui::Text("position: ");
        ImGui::SliderFloat("Pos X", &transform.position.x, -2, 2);
        ImGui::SliderFloat("Pos Y", &transform.position.y, -2, 2);
        ImGui::SliderFloat("Pos Z", &transform.position.z, -2, 2);
        ImGui::Text("rotation: ");
        glm::vec3 eulerAngles = glm::eulerAngles(transform.rotation) ;
        eulerAngles *= 180.0 / 3.14159265; //to degrees
        ImGui::SliderFloat("Rot X", &eulerAngles.x, -180, 180);
        ImGui::SliderFloat("Rot Y", &eulerAngles.y, -180, 180);
        ImGui::SliderFloat("Rot Z", &eulerAngles.z, -180, 180);
        ImGui::Text("scale: ");
        ImGui::SliderFloat("Scale X", &transform.scale.x, -1, 1);
        ImGui::SliderFloat("Scale Y", &transform.scale.y, -1, 1);
        ImGui::SliderFloat("Scale Z", &transform.scale.z, -1, 1);

        if(lights!=nullptr){
            ImGui::Text("\n Light Parameters");
            float color[3];
            color[0] = lights[0][lightIndex].color.r; color[1] = lights[0][lightIndex].color.g; color[2] = lights[0][lightIndex].color.b; 
            ImGui::ColorPicker3("Color", color);
            lights[0][lightIndex].color = glm::vec4(color[0],color[1],color[2],1.0);
            ImGui::SliderFloat("Intensity", &lights[0][lightIndex].intensity, 0, 100);
            ImGui::SliderFloat("Range", &lights[0][lightIndex].range, 0, 100);
        }

        eulerAngles *= 3.14159265 / 180.0;
        transform.rotation = glm::quat(eulerAngles);
        calculateWorldTransform();
        ImGui::End();
    }

    void addChild(Node* node) {
        node->parent = this;
        children.emplace_back(node);
        node->calculateWorldTransform();
    }

    void updateTransform(glm::vec3 position = glm::vec3(0), glm::quat rotation = glm::quat(1, 0, 0, 0), glm::vec3 scale = glm::vec3(1)) {
        // Update the local transform with the given parameters
        transform.position = position;
        transform.rotation = rotation;
        transform.scale = scale;
        transform.updateTransformMatrix();

        // Recalculate world transform
        calculateWorldTransform();

        // Update all children recursively
        for (Node* child : children) {
            child->calculateWorldTransform();
        }
    }

    void updateWorldTransform(glm::vec3 worldPosition = glm::vec3(0), glm::quat worldRotation = glm::quat(1, 0, 0, 0), glm::vec3 worldScale = glm::vec3(1)) {
        if (parent != nullptr) {
            // Convert world transform to local transform relative to parent
            glm::mat4 parentWorldInverse = glm::inverse(parent->worldTransform);

            // Create the desired world transform matrix
            glm::mat4 desiredWorldTransform = glm::translate(glm::mat4(1), worldPosition) * glm::mat4_cast(worldRotation) * glm::scale(glm::mat4(1), worldScale);

            // Calculate local transform: local = parentInverse * world
            glm::mat4 localTransform = parentWorldInverse * desiredWorldTransform;

            // Decompose local transform back to components
            glm::vec3 localScale;
            glm::quat localRotation;
            glm::vec3 localTranslation;

            decomposeMatrix(localTransform, localTranslation, localRotation, localScale);

            // Update local transform
            transform.position = localTranslation;
            transform.rotation = localRotation;
            transform.scale = localScale;
        } else {
            // Root node: world space = local space
            transform.position = worldPosition;
            transform.rotation = worldRotation;
            transform.scale = worldScale;
        }
        // Update matrices
        calculateWorldTransform(); // Update world matrix
        // Update all children's world transforms
        for (Node* child : children) {
            child->calculateWorldTransform();
        }
    }

    void calculateWorldTransform() {
        transform.updateTransformMatrix();
        if (parent != nullptr) {
            // World transform = parent's world transform * this node's local transform
            worldTransform = parent->worldTransform * transform.transformMatrix;
        } else {
            // Root node: world transform equals local transform
            worldTransform = transform.transformMatrix;
        }
        if (meshes != nullptr) {
            resourceManager->updateModelMatrix(meshes[0][meshIndex].modelMatrixIndex, worldTransform);
        }
        if (lights != nullptr) {
            lights[0][lightIndex].position = glm::vec4(worldTransform[3]);;
            resourceManager->updateLight(lights[0][lightIndex].allocationIndex, lights[0][lightIndex]);
        }
        if (hasGizmo) {
            gizmo.updateTransform(worldTransform);
        }
    }

    // Update only local scale
    void updateScale(glm::vec3 scale) {
        transform.scale = scale;
        calculateWorldTransform();
        // Update all children recursively
        for (Node* child : children) {
            child->calculateWorldTransform();
        }
    }

    // Update only local rotation
    void updateRotation(glm::quat rotation) {
        transform.rotation = rotation;
        calculateWorldTransform();
        // Update all children recursively
        for (Node* child : children) {
            child->calculateWorldTransform();
        }
    }

    // Update only world scale
    void updateWorldScale(glm::vec3 worldScale) {
        if (parent != nullptr) {
            // Get current world position and rotation
            glm::vec3 currentWorldPos;
            glm::quat currentWorldRot;
            glm::vec3 currentWorldScale;
            decomposeMatrix(worldTransform, currentWorldPos, currentWorldRot, currentWorldScale);

            // Convert world transform to local transform relative to parent
            glm::mat4 parentWorldInverse = glm::inverse(parent->worldTransform);
            // Create the desired world transform matrix with new scale
            glm::mat4 desiredWorldTransform = glm::translate(glm::mat4(1), currentWorldPos) * glm::mat4_cast(currentWorldRot) * glm::scale(glm::mat4(1), worldScale);
            // Calculate local transform: local = parentInverse * world
            glm::mat4 localTransform = parentWorldInverse * desiredWorldTransform;

            // Decompose local transform back to components
            glm::vec3 localScale;
            glm::quat localRotation;
            glm::vec3 localTranslation;
            decomposeMatrix(localTransform, localTranslation, localRotation, localScale);

            // Update only local scale
            transform.scale = localScale;
        } else {
            // Root node: world space = local space
            transform.scale = worldScale;
        }

        // Update matrices
        calculateWorldTransform();
        // Update all children's world transforms
        for (Node* child : children) {
            child->calculateWorldTransform();
        }
    }

    // Update only world rotation
    void updateWorldRotation(glm::quat worldRotation) {
        if (parent != nullptr) {
            // Get current world position and scale
            glm::vec3 currentWorldPos;
            glm::quat currentWorldRot;
            glm::vec3 currentWorldScale;
            decomposeMatrix(worldTransform, currentWorldPos, currentWorldRot, currentWorldScale);

            // Convert world transform to local transform relative to parent
            glm::mat4 parentWorldInverse = glm::inverse(parent->worldTransform);
            // Create the desired world transform matrix with new rotation
            glm::mat4 desiredWorldTransform = glm::translate(glm::mat4(1), currentWorldPos) * glm::mat4_cast(worldRotation) * glm::scale(glm::mat4(1), currentWorldScale);
            // Calculate local transform: local = parentInverse * world
            glm::mat4 localTransform = parentWorldInverse * desiredWorldTransform;

            // Decompose local transform back to components
            glm::vec3 localScale;
            glm::quat localRotation;
            glm::vec3 localTranslation;
            decomposeMatrix(localTransform, localTranslation, localRotation, localScale);

            // Update only local rotation
            transform.rotation = localRotation;
        } else {
            // Root node: world space = local space
            transform.rotation = worldRotation;
        }

        // Update matrices
        calculateWorldTransform();
        // Update all children's world transforms
        for (Node* child : children) {
            child->calculateWorldTransform();
        }
    }

    // update only local position
    void updatePosition(glm::vec3 position) {
        transform.position = position;
        calculateWorldTransform();
        // Update all children recursively
        for (Node* child : children) {
            child->calculateWorldTransform();
        }
    }

    // update only world position
    void updateWorldPosition(glm::vec3 worldPosition) {
        if (parent != nullptr) {
            // Get current world rotation and scale from existing world transform
            glm::vec3 currentWorldPos;
            glm::quat currentWorldRot;
            glm::vec3 currentWorldScale;
            decomposeMatrix(worldTransform, currentWorldPos, currentWorldRot, currentWorldScale);

            // Create desired world transform with new position but same rotation/scale
            glm::mat4 desiredWorldTransform = glm::translate(glm::mat4(1), worldPosition) * glm::mat4_cast(currentWorldRot) * glm::scale(glm::mat4(1), currentWorldScale);

            // Convert to local space relative to parent
            glm::mat4 parentWorldInverse = glm::inverse(parent->worldTransform);
            glm::mat4 localTransform = parentWorldInverse * desiredWorldTransform;

            // Extract local position and update transform
            glm::vec3 localScale;
            glm::quat localRotation;
            glm::vec3 localTranslation;
            decomposeMatrix(localTransform, localTranslation, localRotation, localScale);

            transform.position = localTranslation;
        } else {
            // Root node: world space = local space
            transform.position = worldPosition;
        }

        // Update this node's matrices
        calculateWorldTransform();

        // Update all children's world transforms
        for (Node* child : children) {
            child->calculateWorldTransform();
        }
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
    glm::mat4 viewMatrix;
    glm::mat4 projectionMatrix;
    float yaw = -90.0f;
    float pitch = 0.0f;

    void rayFromScreenCoords(float x, float y, glm::vec3* origin, glm::vec3* direction) {
        // Two points along the ray in clip space (NDC with w=1)
        glm::vec4 nearClip = glm::vec4(x, y, -1.0f, 1.0f); // Near plane
        glm::vec4 farClip = glm::vec4(x, y, 1.0f, 1.0f);   // Far plane

        // Transform to view space
        glm::vec4 nearView = glm::inverse(projectionMatrix) * nearClip;
        glm::vec4 farView = glm::inverse(projectionMatrix) * farClip;

        // Perspective divide
        nearView /= nearView.w;
        farView /= farView.w;

        // Transform to world space
        glm::vec4 nearWorld = glm::inverse(viewMatrix) * nearView;
        glm::vec4 farWorld = glm::inverse(viewMatrix) * farView;

        // Ray from near to far
        *origin = glm::vec3(nearWorld);
        *direction = glm::normalize(glm::vec3(farWorld - nearWorld));
    }

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
        viewMatrix = glm::lookAt(position, target, upVector);
        // === PROJECTION MATRIX ===
        // Create perspective projection matrix
        projectionMatrix = glm::perspective(fov, aspectRatio, nearPlane, farPlane);
        projectionMatrix[1][1] *= -1.0f; // Flip Y axis for vulkan
        // === VIEW-PROJECTION MATRIX ===
        viewProjection = projectionMatrix * viewMatrix;
    }
};
