#include "scene_graph.hpp"
#include "renderer.hpp"

void SceneGraph::init(Renderer* renderer) {
    this->renderer = renderer;
    // Index 0 is the null/invalid node sentinel
    nodes.push_back({0});
    // Index 1 is the root node
    nodes.push_back({ROOT_INDEX, false, 0, glm::vec3(0.0), glm::quat(1.0, 0, 0, 0), glm::vec3(1, 1, 1)});
    nodes[ROOT_INDEX].name = "root";
    lastNode = ROOT_INDEX;
    allocateNodeGPU(nodes[ROOT_INDEX]);
    TransformSystem::updateAll(nodes[ROOT_INDEX], nodes, renderer->getDescriptorSet(), renderer->getModelMatrixBufferIndex(),
                               renderer->assetManager.meshes, renderer->getLightsMutable());
}

uint32_t SceneGraph::addNode(bool internal,uint32_t parentIndex, glm::vec3 position, glm::quat rotation, glm::vec3 scale) {
    uint32_t newIndex;

    if (!freeSlots.empty()) {
        newIndex = freeSlots.front();
        freeSlots.pop_front();
        nodes[newIndex] = Node(newIndex, internal, parentIndex, position, rotation, scale);
    } else {
        newIndex = static_cast<uint32_t>(nodes.size());
        nodes.push_back({newIndex, internal, parentIndex, position, rotation, scale});
        lastNode = newIndex;
    }

    linkChild(parentIndex, newIndex);

    // Compute world transform from parent
    Node& newNode = nodes[newIndex];
    Node& parent = nodes[parentIndex];
    newNode.worldTransform = parent.worldTransform * newNode.localTransform;

    allocateNodeGPU(newNode);
    TransformSystem::updateAll(newNode, nodes, renderer->getDescriptorSet(), renderer->getModelMatrixBufferIndex(),
                               renderer->assetManager.meshes, renderer->getLightsMutable());
    return newIndex;
}

void SceneGraph::killNode(uint32_t idx) {
    Node& node = nodes[idx];
    if (!node.alive) return;

    node.alive = false;
    unlinkChild(idx);
    deallocateNodeGPU(node);
    renderer->removeNodeFromRenderList(idx);

    // Clear selection if this node was selected
    if (selectedNode == idx) deSelectNode();

    // Reset tree links so nothing walks into dead nodes
    node.parentIndex = 0;
    node.firstChild = 0;
    node.nextSibling = 0;

    freeSlots.push_back(idx);
}

void SceneGraph::removeNode(uint32_t idx) {
    if (!isNodeValid(idx)) return;
    if (idx == ROOT_INDEX) return; // never delete root

    // Recursively kill children first (depth-first)
    uint32_t child = nodes[idx].firstChild;
    while (child != 0) {
        uint32_t next = nodes[child].nextSibling; // capture before killNode clears it
        removeNode(child);
        child = next;
    }

    killNode(idx);
}

void SceneGraph::allocateNodeGPU(Node& node) {
    uint32_t singleIndex = renderer->getDescriptorSet().allocateFixedBuffer(renderer->getModelMatrixBufferIndex(), glm::mat4(1.0f));
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        node.modelMatrixIndices[i] = singleIndex;
    }
}

void SceneGraph::deallocateNodeGPU(Node& node) {
    renderer->getDescriptorSet().freeFixedBuffer(renderer->getModelMatrixBufferIndex(), node.modelMatrixIndices[0]);
}
