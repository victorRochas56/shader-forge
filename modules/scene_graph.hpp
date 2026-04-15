#pragma once
#include <array>
#include <optional>
#include <stdexcept>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "constants.hpp"
#include "scene_elements.hpp"
#include "transform_system.hpp"

class Renderer;

class SceneGraph {
  public:
    uint32_t selectedNode = 0; // 0 = none selected

    SceneGraph() { nodes.reserve(MAX_NODES); }
    ~SceneGraph() = default;

    void init(Renderer* renderer);

    static constexpr uint32_t ROOT_INDEX = 1;

    Node& getRootNode() { return nodes[ROOT_INDEX]; }

    std::vector<Node>& getNodes() { return nodes; }

    Node& getNode(uint32_t idx) { return nodes[idx]; }
    const Node& getNode(uint32_t idx) const { return nodes[idx]; }

    uint32_t addNode(bool internal = false, uint32_t parentIndex = ROOT_INDEX, glm::vec3 position = glm::vec3(0.0f), glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
                     glm::vec3 scale = glm::vec3(1.0f));

    void removeNode(uint32_t index);

    void selectNode(uint32_t nodeIndex) {
        if (nodeIndex <= lastNode) {
            selectedNode = nodeIndex;
        }
    }

    void deSelectNode() { selectedNode = 0; }

    uint32_t getNodeCount() { return lastNode; } // excludes null node at 0
    uint32_t getLastNode() { return lastNode; }

    bool isNodeValid(uint32_t idx) { return idx > 0 && idx <= lastNode; }

    void resetLastNode() {
        lastNode = 0;
        selectedNode = 0;
    }

    // Link child into parent's sibling list (firstChild/nextSibling)
    void linkChild(uint32_t parentIdx, uint32_t childIdx) {
        nodes[childIdx].parentIndex = parentIdx;
        nodes[childIdx].nextSibling = nodes[parentIdx].firstChild;
        nodes[parentIdx].firstChild = childIdx;
    }

    // Unlink a node from its current parent's child list
    void unlinkChild(uint32_t childIdx) {
        uint32_t parentIdx = nodes[childIdx].parentIndex;
        if (parentIdx == 0) return;
        Node& parent = nodes[parentIdx];
        if (parent.firstChild == childIdx) {
            parent.firstChild = nodes[childIdx].nextSibling;
        } else {
            uint32_t sib = parent.firstChild;
            while (sib != 0 && nodes[sib].nextSibling != childIdx) {
                sib = nodes[sib].nextSibling;
            }
            if (sib != 0) {
                nodes[sib].nextSibling = nodes[childIdx].nextSibling;
            }
        }
        nodes[childIdx].nextSibling = 0;
        nodes[childIdx].parentIndex = 0;
    }

    // Reparent a node under a new parent. If keepWorldTransform is true,
    // the node's relative transform is adjusted to preserve its world position/rotation/scale.
    void reparent(uint32_t childIdx, uint32_t newParentIdx, bool keepWorldTransform = true) {
        Node& child = nodes[childIdx];
        if (child.parentIndex == newParentIdx) return;

        glm::mat4 savedWorldTransform = child.worldTransform;

        unlinkChild(childIdx);
        linkChild(newParentIdx, childIdx);

        if (keepWorldTransform) {
            Node& newParent = nodes[newParentIdx];
            glm::mat4 newLocal = glm::inverse(newParent.worldTransform) * savedWorldTransform;

            child.relativePosition = glm::vec3(newLocal[3]);

            child.relativeScale.x = glm::length(glm::vec3(newLocal[0]));
            child.relativeScale.y = glm::length(glm::vec3(newLocal[1]));
            child.relativeScale.z = glm::length(glm::vec3(newLocal[2]));

            glm::mat3 rotMat;
            rotMat[0] = glm::vec3(newLocal[0]) / child.relativeScale.x;
            rotMat[1] = glm::vec3(newLocal[1]) / child.relativeScale.y;
            rotMat[2] = glm::vec3(newLocal[2]) / child.relativeScale.z;
            child.relativeRotation = glm::quat_cast(rotMat);
            child.relativeRotationEuler = glm::eulerAngles(child.relativeRotation);
        }

        TransformSystem::recomputeTransforms(child, nodes);
    }

    // Iterate children of a node by index via callback: fn(Node& child)
    template<typename Fn>
    void forEachChild(uint32_t nodeIdx, Fn&& fn) {
        uint32_t child = nodes[nodeIdx].firstChild;
        while (child != 0) {
            fn(nodes[child]);
            child = nodes[child].nextSibling;
        }
    }

    template<typename Fn>
    void forEachChild(const Node& node, Fn&& fn) {
        uint32_t child = node.firstChild;
        while (child != 0) {
            fn(nodes[child]);
            child = nodes[child].nextSibling;
        }
    }

  private:
    Renderer* renderer = nullptr;
    std::vector<Node> nodes;
    uint32_t lastNode = 0;

    // Allocate GPU model matrix buffer for a node
    void allocateNodeGPU(Node& node);
};
