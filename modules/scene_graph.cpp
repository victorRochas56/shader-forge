#include "scene_graph.hpp"
#include "renderer.hpp"

void SceneGraph::init(Renderer* renderer) {
    this->renderer = renderer;
    (*nodes)[0].emplace(0, nullptr, glm::vec3(0.0), glm::quat(1.0, 0, 0, 0), glm::vec3(1, 1, 1));
    (*nodes)[0]->name = "root";
    rootNode = &*(*nodes)[0];
    allocateNodeGPU(*rootNode);
    TransformSystem::updateAll(*rootNode, renderer->getDescriptorSet(), renderer->getModelMatrixBufferIndex(),
                               renderer->assetManager.meshes, renderer->getLightsMutable());
}

uint32_t SceneGraph::addNode    (uint32_t parentIndex, glm::vec3 position, glm::quat rotation,
                              glm::vec3 scale, bool keepWorldTransform) {
    Node* parent = &*(*nodes)[parentIndex];
    (*nodes)[lastNode + 1].emplace(lastNode + 1, parent, position, rotation, scale, keepWorldTransform);
    Node& newNode = *(*nodes)[lastNode + 1];
    allocateNodeGPU(newNode);
    TransformSystem::updateAll(newNode, renderer->getDescriptorSet(), renderer->getModelMatrixBufferIndex(),
                               renderer->assetManager.meshes, renderer->getLightsMutable());
    lastNode++;
    return lastNode;
}

void SceneGraph::allocateNodeGPU(Node& node) {
    uint32_t singleIndex = renderer->getDescriptorSet().allocateFixedBuffer(renderer->getModelMatrixBufferIndex(), glm::mat4(1.0f));
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        node.modelMatrixIndices[i] = singleIndex;
    }
}
