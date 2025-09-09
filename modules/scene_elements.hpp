#pragma once
#ifndef VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1 // for raii
#endif
#ifndef VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_NO_CONSTRUCTORS 1 // for structs constructors
#endif

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <string>

#include "descriptor_sets.hpp"
#include "structs.hpp"

#define GLM_FORCE_RADIANS
#define GLM_DEPTH_ZERO_TO_OzNE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>


class Renderer;

class Node {
  public:
    std::string name = "empty";
    
    Node(Renderer* pRenderer, uint32_t arrayIndex, Node* parent = nullptr, glm::vec3 position = glm::vec3(0.0f), glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
         glm::vec3 scale = glm::vec3(1.0f), bool keepWorldTransform = false);

    glm::vec3 getWorldPosition() { return glm::vec3(worldTransform[3]); }
    glm::vec3 getRelativePosition() { return relativePosition; }

    glm::vec3 getWorldScale() {
        glm::vec3 scale;
        scale.x = glm::length(glm::vec3(worldTransform[0]));
        scale.y = glm::length(glm::vec3(worldTransform[1]));
        scale.z = glm::length(glm::vec3(worldTransform[2]));
        return scale;
    }
    glm::vec3 getRelativeScale() { return relativeScale; }

    glm::quat getWorldRotation() {
        glm::mat3 rotationMatrix;
        glm::vec3 scale = getWorldScale();
        rotationMatrix[0] = glm::vec3(worldTransform[0]) / scale.x;
        rotationMatrix[1] = glm::vec3(worldTransform[1]) / scale.y;
        rotationMatrix[2] = glm::vec3(worldTransform[2]) / scale.z;
        // Convert to quaternion
        glm::quat rotation = glm::quat_cast(rotationMatrix);
        return rotation;
    }
    glm::quat getRelativeRotation() { return relativeRotation; }
    glm::vec3 getWorldRotationEuler() { return glm::eulerAngles(getWorldRotation()); }
    glm::vec3 getRelativeRotationEuler() { return relativeRotationEuler; }

    uint32_t getModelMatrixIndex() { return modelMatrixIndex; }

    void update();

    void addChild(Node* child, bool keepRelativeTransform = false) {
        children.push_back(child);
        child->parent = this;
        if (keepRelativeTransform) {
            // Calculate relative transform using matrix approach for accuracy
            glm::mat4 parentInverse = glm::inverse(worldTransform);
            glm::mat4 relativeTransform = parentInverse * child->worldTransform;

            // Extract position
            child->relativePosition = glm::vec3(relativeTransform[3]);

            // Extract scale
            child->relativeScale.x = glm::length(glm::vec3(relativeTransform[0]));
            child->relativeScale.y = glm::length(glm::vec3(relativeTransform[1]));
            child->relativeScale.z = glm::length(glm::vec3(relativeTransform[2]));

            // Extract rotation
            glm::mat3 rotMatrix;
            rotMatrix[0] = glm::vec3(relativeTransform[0]) / child->relativeScale.x;
            rotMatrix[1] = glm::vec3(relativeTransform[1]) / child->relativeScale.y;
            rotMatrix[2] = glm::vec3(relativeTransform[2]) / child->relativeScale.z;
            child->relativeRotation = glm::quat_cast(rotMatrix);

            // Update Euler angles
            child->relativeRotationEuler = glm::eulerAngles(child->relativeRotation);
        }
        child->update();
    }

    void removeChild(Node* child) {
        auto it = std::find(children.begin(), children.end(), child);
        if (it != children.end()) {
            children.erase(it);
            child->parent = nullptr;
        }
    }

    std::vector<Node*>& getChildren() {return children;}

    void addMesh(uint32_t meshIndex) { this->meshIndex = meshIndex; }

    uint32_t getMeshIndex() {return meshIndex;}
    void addMaterial(uint32_t index, uint32_t materialIndex);

    void addLight(uint32_t lightIndex) { this->lightIndex = lightIndex; }

  private:
    Renderer* renderer;
    uint32_t nodeIndex;

    glm::vec3 relativePosition;
    glm::vec3 relativeScale;
    glm::quat relativeRotation;
    glm::vec3 relativeRotationEuler;
    glm::mat4 worldTransform; // aka model matrix
    uint32_t modelMatrixIndex;
    glm::mat4 localTransform; // relative to parent
    std::vector<Node*> children;
    Node* parent = nullptr;

    uint32_t meshIndex;
    std::vector<uint32_t> materialIndices;
    uint32_t lightIndex;
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
