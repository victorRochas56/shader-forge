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

struct Node {
    std::string name = "empty";
    glm::mat4 worldTransform = glm::mat4(1);
    BindlessResourceManager* resources;
    Transform transform;
    Mesh* meshes = nullptr;
    uint32_t meshIndex;
    Light* lights = nullptr;
    uint32_t lightIndex;

    Node* parent = nullptr;
    std::vector<Node*> children;

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
        transform.updateTransformMatrix(); // Assuming Transform has this method

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
        transform.updateTransformMatrix(); // Update local matrix
        calculateWorldTransform();         // Update world matrix
        // Update all children's world transforms
        for (Node* child : children) {
            child->calculateWorldTransform();
        }
    }

    void calculateWorldTransform() {
        if (parent != nullptr) {
            // World transform = parent's world transform * this node's local transform
            worldTransform = parent->worldTransform * transform.transformMatrix;
        } else {
            // Root node: world transform equals local transform
            worldTransform = transform.transformMatrix;
        }
        if (meshes != nullptr) {
            resources->updateModelMatrix(meshes[meshIndex].modelMatrixIndex, worldTransform);
        }
        if (lights != nullptr) {
            lights[lightIndex].position = worldTransform * glm::vec4(transform.position, 1.0);
            resources->updateLight(lights[lightIndex].allocationIndex, lights[lightIndex]);
        }
    }

    // Update only local scale
    void updateScale(glm::vec3 scale = glm::vec3(1)) {
        transform.scale = scale;
        transform.updateTransformMatrix();
        calculateWorldTransform();
        // Update all children recursively
        for (Node* child : children) {
            child->calculateWorldTransform();
        }
    }

    // Update only local rotation
    void updateRotation(glm::quat rotation = glm::quat(1, 0, 0, 0)) {
        transform.rotation = rotation;
        transform.updateTransformMatrix();
        calculateWorldTransform();
        // Update all children recursively
        for (Node* child : children) {
            child->calculateWorldTransform();
        }
    }

    // Update only world scale
    void updateWorldScale(glm::vec3 worldScale = glm::vec3(1)) {
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
        transform.updateTransformMatrix();
        calculateWorldTransform();
        // Update all children's world transforms
        for (Node* child : children) {
            child->calculateWorldTransform();
        }
    }

    // Update only world rotation
    void updateWorldRotation(glm::quat worldRotation = glm::quat(1, 0, 0, 0)) {
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
        transform.updateTransformMatrix();
        calculateWorldTransform();
        // Update all children's world transforms
        for (Node* child : children) {
            child->calculateWorldTransform();
        }
    }

    // Convenience function to update only local position (if you don't have it already)
    void updatePosition(glm::vec3 position = glm::vec3(0)) {
        transform.position = position;
        transform.updateTransformMatrix();
        calculateWorldTransform();
        // Update all children recursively
        for (Node* child : children) {
            child->calculateWorldTransform();
        }
    }

    // Convenience function to update only world position (if you don't have it already)
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
        transform.updateTransformMatrix();
        calculateWorldTransform();

        // Update all children's world transforms
        for (Node* child : children) {
            child->calculateWorldTransform();
        }
    }
};