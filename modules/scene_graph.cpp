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

uint32_t SceneGraph::addNode(Node node) {
    uint32_t newIndex;
    if (!freeSlots.empty()) {
        newIndex = freeSlots.front();
        freeSlots.pop();
        nodes[newIndex] = node;
    } else {
        newIndex = static_cast<uint32_t>(nodes.size());
        nodes.push_back(node);
        lastNode = newIndex;
    }
    Node& newNode = nodes[newIndex];
    newNode.nodeIndex = newIndex;    
    newNode.firstChild = 0;          
    newNode.nextSibling = 0;
    newNode.name = makeUniqueNodeName(newNode.name);  
    linkChild(newNode.parentIndex, newIndex);         

    Node& parent = nodes[newNode.parentIndex];
    newNode.worldTransform = parent.worldTransform * newNode.localTransform;
    allocateNodeGPU(newNode);
    TransformSystem::updateAll(newNode, nodes, scene->assetManager.meshes, scene->getLightsMutable());
    NodeOps::assignBillboard(newNode, {.textureIndex = buffers->nodeTextureIndex, .hidden = true, .screenSpaceSize = false, .size = 0.25f}, *scene);
    return newIndex;
}

uint32_t SceneGraph::duplicateNodeToScene(const Node& src, glm::vec3 relativePosition) {
    // Resolve the source's handles to payloads up front. A handle can name a dead slot, so look up
    // rather than .at() — a missing payload just means the copy spawns without that attachment.
    const Light* light = nullptr;
    const ParticleEmitter* emitter = nullptr;
    auto lightIt = scene->getLights().find(src.lightIndex);
    if (lightIt != scene->getLights().end()) light = &lightIt->second;
    auto emitterIt = scene->getEmitters().find(src.particleIndex);
    if (emitterIt != scene->getEmitters().end()) emitter = &emitterIt->second;

    // Volumes are deliberately left out: duplicating a volume node has never carried one over.
    return spawnNode(src, src.parentIndex, relativePosition, light, emitter, nullptr);
}

uint32_t SceneGraph::spawnNode(const Node& src, uint32_t parentIndex, glm::vec3 relativePosition,
                               const Light* light, const ParticleEmitter* emitter, const Volume* volume) {
    // Snapshot the mesh before adding the copy: the fresh node starts with its handles cleared so
    // it can't alias the source's GPU resources, and assignMesh re-takes the reference below.
    uint32_t meshIndex = src.meshIndex;

    Node copy(src);
    copy.parentIndex = parentIndex;
    copy.firstChild = 0;                   // src may carry template-local links, which mean
    copy.nextSibling = 0;                  // nothing to the graph — addNode relinks it anyway
    copy.meshIndex = MAX_MESHES;
    copy.lightIndex = MAX_LIGHTS;          // don't alias the source's GPU light...
    copy.particleIndex = 0xFFFFFFFF;       // ...or its emitter pool
    copy.relativePosition = relativePosition;

    // addNode allocates a fresh model-matrix slot and recomputes world transform from
    // relativePosition, so the node is at its final spot before attachments are baked.
    uint32_t newIndex = addNode(copy);
    Node& newNode = nodes[newIndex];

    // need to do this for proper ref counting of mesh
    if (meshIndex < scene->assetManager.meshes.size()) {
        NodeOps::assignMesh(newNode,meshIndex,*scene); // properly increases ref count
    }
    if(newNode.materialIndex != 0xFFFFFFFF) {
        const Material& material = scene->getMaterials()[newNode.materialIndex];
        scene->addMeshToShader(newIndex, material.shaderSource, material);
    }
    if (light)   NodeOps::assignLight(newNode, *light, *scene, *bindless, buffers->lightBufferIndex);
    if (emitter) NodeOps::assignEmitter(newNode, *emitter, *scene, *bindless, *buffers);
    if (volume)  NodeOps::assignVolume(newNode, *volume, *scene);
    return newIndex;
}

// Flatten a subtree into a standalone template array. Nodes are appended depth-first, so a node
// always precedes its descendants, and parentIndex/firstChild/nextSibling are remapped to dst-local
// indices — dst[0] is the subtree root, which keeps 0 readable as "no node" like in the graph.
// Nothing is spawned into the scene, and the mesh takes a reference so it stays resident once the
// source node is gone. Attachment handles are left as the source's for Scene::addTemplate to
// resolve into payloads — including nodeIndex, which volumes are keyed by. Returns the dst index
// of the copied subtree root.
uint32_t SceneGraph::duplicateNode(const Node& src, std::vector<Node>& dst) {
    uint32_t dstIndex = static_cast<uint32_t>(dst.size());

    Node copy(src);
    copy.parentIndex = 0; // patched below for children; the subtree root stays parentless
    copy.firstChild = 0;
    copy.nextSibling = 0;
    copy.isSelected = false;
    dst.push_back(copy);

    if (src.meshIndex < scene->assetManager.meshes.size()) {
        scene->assetManager.meshes[src.meshIndex].refCount++;
    }

    // Walk the source's children — the copy's links were just cleared.
    uint32_t prevChild = 0;
    uint32_t child = src.firstChild;
    while (child != 0) {
        uint32_t childDst = duplicateNode(nodes[child], dst);
        dst[childDst].parentIndex = dstIndex;
        if (prevChild == 0) dst[dstIndex].firstChild = childDst;
        else                dst[prevChild].nextSibling = childDst;
        prevChild = childDst;
        child = nodes[child].nextSibling;
    }
    return dstIndex;
}

void SceneGraph::killNode(uint32_t idx) {
    Node& node = nodes[idx];
    if (!node.alive) return;

    LightInfluence::onNodeRemoved(node, scene->getLightsMutable());

    node.alive = false;
    unlinkChild(idx);
    deallocateNodeGPU(node);

    if(node.meshIndex < scene->assetManager.meshes.size()) {
        scene->assetManager.meshes[node.meshIndex].refCount--;
    }

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
