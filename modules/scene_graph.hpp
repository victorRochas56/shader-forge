#pragma once
#include <array>
#include <optional>
#include <queue>
#include <stdexcept>
#include <stack>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "constants.hpp"
#include "render_buffers.hpp"
#include "scene_elements.hpp"
#include "transform_system.hpp"

class Scene;
class BindlessSystem;

// NOTE: members are suffixed to dodge <windows.h> macros (DELETE is a winnt.h
// macro) — this header gets pulled in after windows.h in some translation units.
enum NodeOpType {
    MUTATE_NODE,
    ADD_NODE,
    DELETE_NODE,
};

struct NodeOp {

    NodeOp(NodeOpType op, Node& node) {
        this->node = node;
        this->op = op;
    }
    NodeOpType op;
    std::optional<Node> node;        
};


class SceneGraph {
  public:

    SceneGraph() { nodes.reserve(MAX_NODES); }
    ~SceneGraph() = default;
    void init(Scene& scene, BindlessSystem& bindless, const RenderBuffers& buffers);
    // Wipes node storage and re-creates the null sentinel + root using the
    // resources cached at init(). Caller is responsible for clearing any
    // bindless slots tied to dead nodes before calling.
    void reset();

    static constexpr uint32_t ROOT_INDEX = 1;
    uint32_t selectedNode = 0; // 0 = none selected

    Node& getRootNode() { return nodes[ROOT_INDEX]; }
    std::vector<Node>& getNodes() { return nodes; }
    Node& getNode(uint32_t idx) { return nodes[idx]; }
    const Node& getNode(uint32_t idx) const { return nodes[idx]; }
    uint32_t getLastNode() { return lastNode; }
    bool isNodeValid(uint32_t idx) { return idx > 0 && idx < nodes.size() && nodes[idx].alive; }

    uint32_t addNode(bool internal = false, uint32_t parentIndex = ROOT_INDEX, glm::vec3 position = glm::vec3(0.0f),
                     glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f), glm::vec3 scale = glm::vec3(1.0f));
    void removeNode(uint32_t index);
    void syncDirtyNodes();

    void selectNode(uint32_t nodeIndex) {
        if (isNodeValid(nodeIndex)) {
            selectedNode = nodeIndex;
            nodes[nodeIndex].isSelected = true;
        }
    }

    void deSelectNode() {
        if (isNodeValid(selectedNode)){
            nodes[selectedNode].isSelected = false;
        }
        selectedNode = 0;
    }

    // Returns a name not currently used by any alive node, appending a Blender-style ".NNN" suffix
    // on collision. Used when spawning instances so the node tree (and ImGui IDs) stay unambiguous.
    std::string makeUniqueNodeName(const std::string& base) {
        auto nameTaken = [&](const std::string& candidate) {
            for (const Node& n : nodes) {
                if (n.alive && n.name == candidate) return true;
            }
            return false;
        };
        if (!nameTaken(base)) return base;
        for (uint32_t i = 1; i < 100000; i++) {
            std::string num = std::to_string(i);
            if (num.size() < 3) num = std::string(3 - num.size(), '0') + num;
            std::string candidate = base + "." + num;
            if (!nameTaken(candidate)) return candidate;
        }
        return base;
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

        child.transformDirty = true;
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

    void pushUndo(NodeOpType op, Node& node) {
        NodeOp nodeOp = NodeOp(op,node);
    }

    void popUndo() {
        
    }
  private:

    Scene* scene = nullptr;
    BindlessSystem* bindless = nullptr;
    // Non-owning view of the renderer-owned RenderBuffers. Read every access
    // (do not cache the indices locally) so any future buffer recreation in
    // the renderer is observed automatically.
    const RenderBuffers* buffers = nullptr;

    std::vector<Node> nodes;
    uint32_t lastNode = 0;
    std::queue<uint32_t> freeSlots;
    std::stack<Node> undoStack;

    // Allocate GPU model matrix buffer for a node
    void allocateNodeGPU(Node& node);
    // Free GPU model matrix buffer for a node
    void deallocateNodeGPU(Node& node);
    // Kill a single node (mark dead, unlink, free GPU, remove from render list)
    void killNode(uint32_t idx);
};
