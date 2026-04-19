#pragma once
#include <map>
#include <vector>

#include "constants.hpp"
#include "scene_elements.hpp"
#include "structs.hpp"

/*
Owns the invariant:
    For every point light L and every meshed node N with a valid AABB,
    L.influencedNodes.contains(N.nodeIndex) iff sphere(L) overlaps aabb(N).

Whenever the set flips membership OR a node is already inside and its AABB
changed, L.shadowDirty is raised so the point-shadow pass rerenders.

All callers feed this module through exactly two entry points:
  - onMeshedNodeTouched / rebuildLightSet: called from TransformSystem::syncToGPU,
    which is the single funnel for any AABB / light-position change.
  - onNodeRemoved: called from SceneGraph::killNode.

Callers never mutate Light::influencedNodes or Light::shadowDirty for
reconciliation purposes — setting node.transformDirty is enough.
*/
namespace LightInfluence {

    inline bool sphereIntersectsAABB(const glm::vec3& center, float radius,
                                     const glm::vec3& mn, const glm::vec3& mx) {
        glm::vec3 closest = glm::clamp(center, mn, mx);
        glm::vec3 d = closest - center;
        return glm::dot(d, d) <= radius * radius;
    }

    inline glm::vec3 lightWorldPos(const Light& L, const std::vector<Node>& nodes) {
        return glm::vec3(nodes[L.nodeIndex].worldTransform[3]);
    }

    // Reconcile a single meshed node against every shadow-casting point light.
    // Called once per node-with-AABB update inside syncToGPU.
    inline void onMeshedNodeTouched(const Node& node, const std::vector<Node>& nodes,
                                    std::map<uint32_t, Light>& lights) {
        if (node.meshIndex >= MAX_MESHES || !node.boundingBoxValid) return;
        for (auto& [id, L] : lights) {
            if (L.type != LightType::Point || !L.castsShadows) continue;
            if (L.nodeIndex == 0 || L.nodeIndex >= nodes.size()) continue;

            glm::vec3 lp = lightWorldPos(L, nodes);
            bool overlaps = sphereIntersectsAABB(lp, L.range, node.boundingBoxMin, node.boundingBoxMax);
            bool was      = L.influencedNodes.count(node.nodeIndex) != 0;

            if (overlaps && !was) {
                L.influencedNodes.insert(node.nodeIndex);
                L.shadowDirty = true;
            } else if (!overlaps && was) {
                L.influencedNodes.erase(node.nodeIndex);
                L.shadowDirty = true;
            } else if (was) {
                // Still overlapping but AABB changed — shadow needs rerender.
                L.shadowDirty = true;
            }
        }
    }

    // Rebuild one light's influence set from scratch by scanning all meshed nodes.
    // Called when the node carrying a point light is touched (position, range,
    // or shadow-enable changed).
    inline void rebuildLightSet(Light& L, const std::vector<Node>& nodes) {
        if (L.type != LightType::Point) return;
        L.influencedNodes.clear();
        L.shadowDirty = true;
        if (!L.castsShadows) return;
        if (L.nodeIndex == 0 || L.nodeIndex >= nodes.size()) return;

        glm::vec3 lp = lightWorldPos(L, nodes);
        for (const Node& n : nodes) {
            if (!n.alive || n.meshIndex >= MAX_MESHES || !n.boundingBoxValid) continue;
            if (sphereIntersectsAABB(lp, L.range, n.boundingBoxMin, n.boundingBoxMax))
                L.influencedNodes.insert(n.nodeIndex);
        }
    }

    // Called from killNode. Drops the node from every light's set and, if the
    // node carried a point light, clears that light's set.
    inline void onNodeRemoved(const Node& node, std::map<uint32_t, Light>& lights) {
        for (auto& [id, L] : lights) {
            if (L.type != LightType::Point) continue;
            if (L.influencedNodes.erase(node.nodeIndex) != 0) L.shadowDirty = true;
        }
        if (node.lightIndex != MAX_LIGHTS) {
            auto it = lights.find(node.lightIndex);
            if (it != lights.end() && it->second.type == LightType::Point) {
                it->second.influencedNodes.clear();
            }
        }
    }

}
