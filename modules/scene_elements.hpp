#pragma once
#ifndef VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1 // for raii
#endif
#ifndef VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_NO_CONSTRUCTORS 1 // for structs constructors
#endif

#include <string>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include "constants.hpp"
#include "descriptor_sets.hpp"
#include "structs.hpp"

#define GLM_FORCE_RADIANS
#define GLM_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

class Renderer; // forward declaration for free functions below

class Node {
  public:
    std::string name = "empty";

    Node(uint32_t arrayIndex, Node* parent = nullptr, glm::vec3 position = glm::vec3(0.0f), glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
         glm::vec3 scale = glm::vec3(1.0f), bool keepWorldTransform = false);

    void setParent(Node* newParent) { parent = newParent; }

    void addChild(Node* child, bool keepRelativeTransform = false) {
        children.push_back(child);
        child->setParent(this);
        if (keepRelativeTransform) {
            glm::mat4 parentInverse = glm::inverse(worldTransform);
            glm::mat4 relativeTransform = parentInverse * child->worldTransform;

            child->relativePosition = glm::vec3(relativeTransform[3]);

            child->relativeScale.x = glm::length(glm::vec3(relativeTransform[0]));
            child->relativeScale.y = glm::length(glm::vec3(relativeTransform[1]));
            child->relativeScale.z = glm::length(glm::vec3(relativeTransform[2]));

            glm::mat3 rotMatrix;
            rotMatrix[0] = glm::vec3(relativeTransform[0]) / child->relativeScale.x;
            rotMatrix[1] = glm::vec3(relativeTransform[1]) / child->relativeScale.y;
            rotMatrix[2] = glm::vec3(relativeTransform[2]) / child->relativeScale.z;
            child->relativeRotation = glm::quat_cast(rotMatrix);

            child->relativeRotationEuler = glm::eulerAngles(child->relativeRotation);
        }
    }

    void removeChild(Node* child) {
        auto it = std::find(children.begin(), children.end(), child);
        if (it != children.end()) {
            children.erase(it);
            child->parent = nullptr;
        }
    }
    glm::vec3 getWorldPosition() { return glm::vec3(worldTransform[3]); }
    glm::vec3 getRelativePosition() { return relativePosition; }
    glm::mat4 getTransform() { return worldTransform; }
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

    uint32_t getIndex() { return nodeIndex; }
    uint32_t getModelMatrixIndex() { return modelMatrixIndices[0]; } 

    std::vector<Node*>& getChildren() { return children; }

    uint32_t getMeshIndex() { return meshIndex; }
    uint32_t getLightIndex() { return lightIndex; }
    std::vector<uint32_t>& getMaterialIndices() {return materialIndices; }

    // Bounding box getters for frustum culling
    glm::vec3 getBoundingBoxMin() const { return boundingBoxMin; }
    glm::vec3 getBoundingBoxMax() const { return boundingBoxMax; }
    bool isBoundingBoxValid() const { return boundingBoxValid; }

    uint32_t nodeIndex;

    std::vector<Node*> children;
    Node* parent = nullptr;

    glm::vec3 relativePosition;
    glm::vec3 relativeScale;
    glm::quat relativeRotation;
    glm::vec3 relativeRotationEuler;
    glm::vec3 worldRotationEuler;
    glm::mat4 worldTransform; // aka model matrix
    std::array<uint32_t, MAX_FRAMES_IN_FLIGHT> modelMatrixIndices;
    glm::mat4 localTransform; // relative to parent

    uint32_t meshIndex = MAX_MESHES;
    std::vector<uint32_t> materialIndices;
    uint32_t lightIndex = MAX_LIGHTS;

    // Bounding box for frustum culling (in world space)
    glm::vec3 boundingBoxMin = glm::vec3(0.0f);
    glm::vec3 boundingBoxMax = glm::vec3(0.0f);
    bool boundingBoxValid = false;
};

struct Camera {
    glm::vec3 position;
    glm::vec3 target;
    float fov;
    float aspectRatio;
    float nearPlane;
    float farPlane;
    glm::mat4 viewProjection;
    glm::mat4 prevViewProjection;
    glm::mat4 viewMatrix;
    glm::mat4 projectionMatrix;
    float yaw = -90.0f;
    float pitch = 0.0f;

    void rayFromScreenCoords(float x, float y, glm::vec3* origin, glm::vec3* direction) {

        glm::vec4 nearClip = glm::vec4(x, y, -1.0f, 1.0f);
        glm::vec4 farClip = glm::vec4(x, y, 1.0f, 1.0f);

        // view space
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
        if (yaw > 360.0f)
            yaw -= 360.0f;
        if (yaw < -360.0f)
            yaw += 360.0f;
        updateTarget();
    }

    void rotatePitch(float deltaPitch = 0.0f) {
        pitch += deltaPitch;
        // prevents gimbal lock
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

    // based on current yaw and pitch
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
        // Calculates yaw and pitch
        target = targetPos;
        glm::vec3 direction = glm::normalize(target - position);
        yaw = glm::degrees(atan2(direction.z, direction.x));
        pitch = glm::degrees(asin(direction.y));
        calculateViewProjectionMatrix();
    }

    void calculateViewProjectionMatrix() {
        prevViewProjection = viewProjection;
        glm::vec3 upVector = glm::vec3(0.0f, 1.0f, 0.0f); // World up direction
        // Create view matrix using lookAt
        viewMatrix = glm::lookAt(position, target, upVector);
        projectionMatrix = glm::perspective(glm::radians(fov), aspectRatio, nearPlane, farPlane);
        projectionMatrix[1][1] *= -1.0f; // Flip Y axis for vulkan
        viewProjection = projectionMatrix * viewMatrix;
    }
    
    std::vector<glm::vec4> getFrustumCorners(Camera& camera) {
        glm::mat4 invViewProj = glm::inverse(camera.viewProjection);

        std::vector<glm::vec4> corners;
        corners.reserve(8);

        // NDC corners of the frustum
        for (int x = 0; x < 2; ++x) {
            for (int y = 0; y < 2; ++y) {
                for (int z = 0; z < 2; ++z) {
                    glm::vec4 corner = invViewProj * glm::vec4(2.0f * x - 1.0f, 2.0f * y - 1.0f, static_cast<float>(z), 1.0f);
                    corners.push_back(corner / corner.w);
                }
            }
        }

        return corners;
    }

};


