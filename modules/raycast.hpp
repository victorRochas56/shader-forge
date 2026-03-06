#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <limits>
#include <optional>
#include <array>

#include "constants.hpp"
#include "structs.hpp"
#include "scene_elements.hpp"

namespace Raycast {

struct MeshHit {
    uint32_t nodeIndex = MAX_NODES;
    uint32_t submeshLocalIndex = 0;
    float distance = std::numeric_limits<float>::max();
};

inline bool rayIntersectsAABB(glm::vec3 origin, glm::vec3 dir, glm::vec3 bmin, glm::vec3 bmax) {
    glm::vec3 invDir = 1.0f / dir;
    glm::vec3 t1 = (bmin - origin) * invDir;
    glm::vec3 t2 = (bmax - origin) * invDir;
    glm::vec3 tmin = glm::min(t1, t2);
    glm::vec3 tmax = glm::max(t1, t2);
    float enter = glm::max(glm::max(tmin.x, tmin.y), tmin.z);
    float exit = glm::min(glm::min(tmax.x, tmax.y), tmax.z);
    return exit >= glm::max(enter, 0.0f);
}

// Möller–Trumbore ray-triangle intersection, returns distance t or -1 if no hit
inline float rayTriangle(glm::vec3 origin, glm::vec3 dir, glm::vec3 v0, glm::vec3 v1, glm::vec3 v2) {
    glm::vec3 e1 = v1 - v0;
    glm::vec3 e2 = v2 - v0;
    glm::vec3 h = glm::cross(dir, e2);
    float a = glm::dot(e1, h);
    if (a > -1e-6f && a < 1e-6f) return -1.0f;

    float f = 1.0f / a;
    glm::vec3 tvec = origin - v0;
    float u = f * glm::dot(tvec, h);
    if (u < 0.0f || u > 1.0f) return -1.0f;

    glm::vec3 q = glm::cross(tvec, e1);
    float v = f * glm::dot(dir, q);
    if (v < 0.0f || u + v > 1.0f) return -1.0f;

    float t = f * glm::dot(e2, q);
    return t > 1e-6f ? t : -1.0f;
}

// Raycast against node bounding positions (existing proximity-based approach)
inline std::vector<uint32_t> castNodes(glm::vec3 origin, glm::vec3 direction,
                                       std::array<std::optional<Node>, MAX_NODES>& nodes, uint32_t lastNode) {
    float margin = 0.05f;
    glm::vec3 dir = glm::normalize(direction);
    std::vector<uint32_t> foundNodes;

    for (uint32_t i = 1; i <= lastNode; i++) {
        if (!nodes[i].has_value()) continue;
        glm::vec3 nodeWorldLoc = glm::vec3(nodes[i]->getWorldPosition());
        glm::vec3 toNode = nodeWorldLoc - origin;
        float projectionLength = glm::dot(toNode, dir);
        if (projectionLength < 0) continue;
        glm::vec3 closestPointOnRay = origin + dir * projectionLength;
        float distanceToRay = glm::distance(nodeWorldLoc, closestPointOnRay);
        if (distanceToRay < margin) {
            foundNodes.push_back(i);
        }
    }
    return foundNodes;
}

// Raycast against actual mesh triangles, returns closest hit with node and submesh indices
inline MeshHit castMeshes(glm::vec3 origin, glm::vec3 direction,
                          std::array<std::optional<Node>, MAX_NODES>& nodes, uint32_t lastNode,
                          std::vector<Mesh>& meshes, std::vector<SubMesh>& subMeshes) {
    glm::vec3 dir = glm::normalize(direction);
    MeshHit result;

    for (uint32_t i = 1; i <= lastNode; i++) {
        if (!nodes[i].has_value()) continue;
        auto& node = *nodes[i];
        uint32_t meshIdx = node.getMeshIndex();
        if (meshIdx >= meshes.size()) continue;
        auto& mesh = meshes[meshIdx];
        if (mesh.freed) continue;

        glm::mat4 worldTransform = node.getTransform();

        for (uint32_t s = 0; s < mesh.subMeshes.size(); s++) {
            auto& sub = subMeshes[mesh.subMeshes[s]];
            if (sub.cpuIndices.empty() || sub.cpuPositions.empty()) continue;

            // AABB early-out in world space
            glm::vec3 worldMin(std::numeric_limits<float>::max());
            glm::vec3 worldMax(std::numeric_limits<float>::lowest());
            glm::vec3 corners[8] = {
                {sub.boundingBoxMin.x, sub.boundingBoxMin.y, sub.boundingBoxMin.z},
                {sub.boundingBoxMax.x, sub.boundingBoxMin.y, sub.boundingBoxMin.z},
                {sub.boundingBoxMin.x, sub.boundingBoxMax.y, sub.boundingBoxMin.z},
                {sub.boundingBoxMax.x, sub.boundingBoxMax.y, sub.boundingBoxMin.z},
                {sub.boundingBoxMin.x, sub.boundingBoxMin.y, sub.boundingBoxMax.z},
                {sub.boundingBoxMax.x, sub.boundingBoxMin.y, sub.boundingBoxMax.z},
                {sub.boundingBoxMin.x, sub.boundingBoxMax.y, sub.boundingBoxMax.z},
                {sub.boundingBoxMax.x, sub.boundingBoxMax.y, sub.boundingBoxMax.z},
            };
            for (auto& c : corners) {
                glm::vec3 wc = glm::vec3(worldTransform * glm::vec4(c, 1.0f));
                worldMin = glm::min(worldMin, wc);
                worldMax = glm::max(worldMax, wc);
            }
            if (!rayIntersectsAABB(origin, dir, worldMin, worldMax)) continue;

            for (uint32_t idx = 0; idx + 2 < sub.cpuIndices.size(); idx += 3) {
                glm::vec3 v0 = glm::vec3(worldTransform * glm::vec4(sub.cpuPositions[sub.cpuIndices[idx]], 1.0f));
                glm::vec3 v1 = glm::vec3(worldTransform * glm::vec4(sub.cpuPositions[sub.cpuIndices[idx + 1]], 1.0f));
                glm::vec3 v2 = glm::vec3(worldTransform * glm::vec4(sub.cpuPositions[sub.cpuIndices[idx + 2]], 1.0f));

                float t = rayTriangle(origin, dir, v0, v1, v2);
                if (t > 0.0f && t < result.distance) {
                    result.distance = t;
                    result.nodeIndex = i;
                    result.submeshLocalIndex = s;
                }
            }
        }
    }
    return result;
}

} // namespace Raycast