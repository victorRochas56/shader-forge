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

    uint32_t addNode(uint32_t parentIndex = ROOT_INDEX, glm::vec3 position = glm::vec3(0.0f), glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
                     glm::vec3 scale = glm::vec3(1.0f));

    void removeNode(uint32_t index) { throw std::runtime_error("remove node not implemented!"); }

    void selectNode(uint32_t nodeIndex) {
        if (nodeIndex <= lastNode) {
            selectedNode = nodeIndex;
        }
    }

    void deSelectNode() { selectedNode = 0; }

    uint32_t getNodeCount() { return lastNode; } // excludes null node at 0
    uint32_t getLastNode() { return lastNode; }

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
