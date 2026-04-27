#include "scene_graph.hpp"
#include "bindless_system.hpp"
#include "light_influence.hpp"
#include "node_ops.hpp"
#include "scene.hpp"

void SceneGraph::init(Scene& sceneRef, BindlessSystem& bindlessRef, const RenderBuffers& buffersRef) {
    this->scene = &sceneRef;
    this->bindless = &bindlessRef;
    this->buffers = &buffersRef;
    reset();
}

void SceneGraph::reset() {
    nodes.clear();
    lastNode = 0;
    selectedNode = 0;
    freeSlots = {};

    // Index 0 is the null/invalid node sentinel
    nodes.push_back({0});
    // Index 1 is the root node
    nodes.push_back({ROOT_INDEX, false, 0, glm::vec3(0.0), glm::quat(1.0, 0, 0, 0), glm::vec3(1, 1, 1)});
    nodes[ROOT_INDEX].name = "root";
    lastNode = ROOT_INDEX;
    allocateNodeGPU(nodes[ROOT_INDEX]);
    TransformSystem::updateAll(nodes[ROOT_INDEX], nodes, *bindless->descriptorSet, buffers->modelMatrixBufferIndex,
                               scene->assetManager.meshes, scene->getLightsMutable());
}

uint32_t SceneGraph::addNode(bool internal,uint32_t parentIndex, glm::vec3 position, glm::quat rotation, glm::vec3 scale) {
    uint32_t newIndex;

    if (!freeSlots.empty()) {
        newIndex = freeSlots.front();
        freeSlots.pop();
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
    TransformSystem::updateAll(newNode, nodes, *bindless->descriptorSet, buffers->modelMatrixBufferIndex,
                               scene->assetManager.meshes, scene->getLightsMutable());

    NodeOps::assignBillboard(newNode,{.textureIndex = buffers->nodeTextureIndex, .hidden = true, .screenSpaceSize = false, .size = 0.25f}, *scene);
    return newIndex;
}

void SceneGraph::killNode(uint32_t idx) {
    Node& node = nodes[idx];
    if (!node.alive) return;

    LightInfluence::onNodeRemoved(node, scene->getLightsMutable());

    node.alive = false;
    unlinkChild(idx);
    deallocateNodeGPU(node);

    if(node.lightIndex != MAX_LIGHTS) {
        scene->removeLight(*bindless, buffers->lightBufferIndex, node.lightIndex);
        node.lightIndex = MAX_LIGHTS;
    }

    if(node.volumeIndex != 0xFFFFFFFF) {
        scene->removeVolume(*bindless, buffers->volumeBufferIndex, node.volumeIndex);
        node.volumeIndex = 0xFFFFFFFF;
    }

    scene->removeBillboard(idx);

    scene->removeNodeFromRenderList(idx);

    // Clear selection if this node was selected
    if (selectedNode == idx) deSelectNode();

    // Reset tree links so nothing walks into dead nodes
    node.parentIndex = 0;
    node.firstChild = 0;
    node.nextSibling = 0;

    freeSlots.push(idx);
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

void SceneGraph::syncDirtyNodes() {
    for (uint32_t i = ROOT_INDEX; i <= lastNode; i++) {
        if (!nodes[i].alive || !nodes[i].transformDirty) continue;
        nodes[i].transformDirty = false;
        TransformSystem::updateAll(nodes[i], nodes, *bindless->descriptorSet, buffers->modelMatrixBufferIndex,
                                   scene->assetManager.meshes, scene->getLightsMutable());
        if(nodes[i].volumeIndex != 0xFFFFFFFF) {
            scene->volumes[nodes[i].volumeIndex].center = nodes[i].getWorldPosition();
            bindless->descriptorSet->updateFixedBuffer<Volume>(buffers->volumeBufferIndex, nodes[i].volumeIndex, scene->volumes[nodes[i].volumeIndex]);
        }
    }
}

void SceneGraph::allocateNodeGPU(Node& node) {
    uint32_t singleIndex = bindless->descriptorSet->allocateFixedBuffer(buffers->modelMatrixBufferIndex, glm::mat4(1.0f));
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        node.modelMatrixIndices[i] = singleIndex;
    }
}

void SceneGraph::deallocateNodeGPU(Node& node) {
    bindless->descriptorSet->freeFixedBuffer(buffers->modelMatrixBufferIndex, node.modelMatrixIndices[0]);
}
