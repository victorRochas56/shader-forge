#pragma once
#include "scene_elements.hpp"

#define GLM_DEPTH_ZERO_TO_ONE

/*
Node constructor - pure data initialization, no renderer dependency.
GPU allocation, tree linking, and transform sync are handled by SceneGraph/TransformSystem.
*/

Node::Node(uint32_t arrayIndex, bool internal, uint32_t parentIdx, glm::vec3 position, glm::quat rotation, glm::vec3 scale, std::string nodeName) {

    unsigned long long nodeSize = sizeof(Node);
    name = nodeName+"_"+std::to_string(arrayIndex);
    nodeIndex = arrayIndex;
    parentIndex = parentIdx;

    relativePosition = position;
    relativeRotation = rotation;
    relativeScale = scale;
    relativeRotationEuler = glm::eulerAngles(rotation);
    worldRotationEuler = glm::eulerAngles(rotation);

    // Compute initial local transform (pure math, no GPU)
    // World transform is computed by TransformSystem after tree linking
    localTransform = makeTransform(relativePosition, relativeRotation, relativeScale);
    worldTransform = localTransform; // will be overwritten by TransformSystem::recomputeTransforms

    std::cout << "Created node " << arrayIndex << ", position=("<<position.x<<","<<position.y<<","<<position.z<<") " << ", scale=(" << scale.x << "," << scale.y << "," << scale.z << ")" << std::endl;
}
