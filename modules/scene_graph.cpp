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
    uint32_t newIndex = lastNode + 1;
    nodes.push_back({newIndex, internal, parentIndex, position, rotation, scale});
    linkChild(parentIndex, newIndex);

    // Compute world transform from parent
    Node& newNode = nodes[newIndex];
    Node& parent = nodes[parentIndex];
    newNode.worldTransform = parent.worldTransform * newNode.localTransform;

    allocateNodeGPU(newNode);
    TransformSystem::updateAll(newNode, nodes, renderer->getDescriptorSet(), renderer->getModelMatrixBufferIndex(),
                               renderer->assetManager.meshes, renderer->getLightsMutable());
    lastNode++;
    assert(lastNode == nodes.size() - 1);
    return lastNode;
}

void SceneGraph::removeNode(uint32_t idx) {
    throw std::runtime_error("remove node not implemented!");

    //delete the node
    // either 
    //      mark the slot free and next add put new node(s) in free slot(s) (problem is making sure nothing points to the empty slot)
    // cache coherency, when reparenting should children etc shift around ?? prob ugly
    // are EVENTS and NODES the only things that hold indexes to point to between frames?
}

void SceneGraph::allocateNodeGPU(Node& node) {
    uint32_t singleIndex = renderer->getDescriptorSet().allocateFixedBuffer(renderer->getModelMatrixBufferIndex(), glm::mat4(1.0f));
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        node.modelMatrixIndices[i] = singleIndex;
    }
}
