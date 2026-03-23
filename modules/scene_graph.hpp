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
    uint32_t selectedNode = MAX_NODES;

    SceneGraph() : nodes(new std::array<std::optional<Node>, MAX_NODES>()) {}
    ~SceneGraph() { delete nodes; }

    void init(Renderer* renderer);

    Node* getRootNode() { return rootNode; }

    std::array<std::optional<Node>, MAX_NODES>& getNodes() { return *nodes; }

    uint32_t addNode(uint32_t parentIndex = 0, glm::vec3 position = glm::vec3(0.0f), glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
                     glm::vec3 scale = glm::vec3(1.0f), bool keepWorldTransform = false);

    void removeNode(uint32_t index) { throw std::runtime_error("remove node not implemented!"); }

    void selectNode(uint32_t nodeIndex) {
        if (nodeIndex <= lastNode) {
            selectedNode = nodeIndex;
        }
    }

    void deSelectNode() { selectedNode = MAX_NODES; }

    uint32_t getNodeCount() { return lastNode + 1; }
    uint32_t getLastNode() { return lastNode; }

    void resetLastNode() {
        lastNode = 0;
        selectedNode = MAX_NODES;
    }

  private:
    Renderer* renderer = nullptr;
    Node* rootNode = nullptr;
    std::array<std::optional<Node>, MAX_NODES>* nodes;
    uint32_t lastNode = 0;

    // Allocate GPU model matrix buffer for a node
    void allocateNodeGPU(Node& node);
};
