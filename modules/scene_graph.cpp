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
    TransformSystem::updateAll(nodes[ROOT_INDEX], nodes, scene->assetManager.meshes, scene->getLightsMutable());
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
    TransformSystem::updateAll(newNode, nodes, scene->assetManager.meshes, scene->getLightsMutable());

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

    scene->removeVolume(idx);
    scene->removeBillboard(idx);

    if (node.particleIndex != 0xFFFFFFFF) {
        scene->removeEmitter(*bindless, *buffers, node.particleIndex);
        node.particleIndex = 0xFFFFFFFF;
    }

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
        TransformSystem::updateAll(nodes[i], nodes, scene->assetManager.meshes, scene->getLightsMutable());
        // Volumes need no per-move GPU work: VolumetricsPass streams them every frame and pulls
        // each volume's world center straight from its node, so a moved node is picked up for free.
    }
}

// Fan out model-matrix uploads: write the current frame-in-flight slice for every node whose world
// transform changed recently, one slice per frame, decrementing until all slices carry the new
// value. Must run after the current frame's fence wait (see Renderer::drawFrame) so the slice being
// written is guaranteed no longer in flight.
void SceneGraph::uploadDirtyTransforms(uint32_t currentFrame) {
    for (uint32_t i = ROOT_INDEX; i <= lastNode; i++) {
        Node& node = nodes[i];
        if (!node.alive || node.gpuDirtyFrames == 0) continue;
        TransformSystem::uploadModelMatrixSlice(node, *bindless->descriptorSet, buffers->modelMatrixBufferIndex, currentFrame);
        node.gpuDirtyFrames--;
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
