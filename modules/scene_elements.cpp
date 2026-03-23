#pragma once
#include "scene_elements.hpp"
#include "renderer.hpp"

#define GLM_DEPTH_ZERO_TO_ONE

/*
Node constructor - pure data initialization, no renderer dependency.
GPU allocation and transform sync are handled by SceneGraph/TransformSystem.
*/

Node::Node(uint32_t arrayIndex, Node* parent, glm::vec3 position, glm::quat rotation, glm::vec3 scale, bool keepWorldTransform) {

    nodeIndex = arrayIndex;

    relativePosition = position;
    relativeRotation = rotation;
    relativeScale = scale;
    relativeRotationEuler = glm::eulerAngles(rotation);
    worldRotationEuler = glm::eulerAngles(rotation);

    // Handle keepWorldTransform logic
    if (keepWorldTransform && parent != nullptr) {
        glm::mat4 desiredWorldTransform = makeTransform(position, rotation, scale);
        glm::mat4 parentInverse = glm::inverse(parent->worldTransform);
        glm::mat4 relativeTransformMatrix = parentInverse * desiredWorldTransform;
        relativePosition = glm::vec3(relativeTransformMatrix[3]);
        relativeScale.x = glm::length(glm::vec3(relativeTransformMatrix[0]));
        relativeScale.y = glm::length(glm::vec3(relativeTransformMatrix[1]));
        relativeScale.z = glm::length(glm::vec3(relativeTransformMatrix[2]));
        glm::mat3 rotMatrix;
        rotMatrix[0] = glm::vec3(relativeTransformMatrix[0]) / relativeScale.x;
        rotMatrix[1] = glm::vec3(relativeTransformMatrix[1]) / relativeScale.y;
        rotMatrix[2] = glm::vec3(relativeTransformMatrix[2]) / relativeScale.z;
        relativeRotation = glm::quat_cast(rotMatrix);
        relativeRotationEuler = glm::eulerAngles(relativeRotation);
        worldRotationEuler = glm::eulerAngles(parent->getWorldRotation() * relativeRotation);
    }

    if (parent != nullptr) {
        parent->addChild(this);
    }

    // Compute initial local/world transform (pure math, no GPU)
    localTransform = makeTransform(relativePosition, relativeRotation, relativeScale);
    if (parent != nullptr) {
        worldTransform = parent->worldTransform * localTransform;
    } else {
        worldTransform = localTransform;
    }

    std::cout << "Created node " << arrayIndex << ", position=("<<position.x<<","<<position.y<<","<<position.z<<") " << ", scale=(" << scale.x << "," << scale.y << "," << scale.z << ")" << std::endl;
}

