#pragma once
#include "scene_elements.hpp"
#include "descriptor_sets.hpp"
#include "light_influence.hpp"
#include "structs.hpp"
#include <map>
#include <vector>

class AssetManager;

namespace TransformSystem {

    // Recompute world transforms for a node and all descendants (pure math, no GPU)
    inline void recomputeTransforms(Node& node, std::vector<Node>& nodes) {
        node.localTransform = makeTransform(node.relativePosition, node.relativeRotation, node.relativeScale);
        if (node.parentIndex != 0) {
            Node& parent = nodes[node.parentIndex];
            node.worldTransform = parent.worldTransform * node.localTransform;
            glm::quat worldRotation = parent.getWorldRotation() * node.relativeRotation;
            node.worldRotationEuler = glm::eulerAngles(worldRotation);
        } else {
            node.worldTransform = node.localTransform;
            node.worldRotationEuler = glm::eulerAngles(node.relativeRotation);
        }

        uint32_t child = node.firstChild;
        while (child != 0) {
            recomputeTransforms(nodes[child], nodes);
            child = nodes[child].nextSibling;
        }
    }

    // Sync a node's world transform to GPU buffers, update light direction, and recompute bounding box
    template<typename MeshArray>
    void syncToGPU(Node& node, std::vector<Node>& nodes, DescriptorSet& ds, uint32_t modelMatrixBufferIndex,
                   const MeshArray& meshes, std::unordered_map<uint32_t, Light>& lights) {
        // Update model matrix for all frames
        glm::mat4 offsetTransform = makeTransform(node.getWorldPosition(), node.getWorldRotation(), node.getWorldScale());
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            ds.updateFixedBufferWithOffset(modelMatrixBufferIndex, node.modelMatrixIndices[i], offsetTransform, i);
        }

        // Update world-space bounding box if node has a mesh.
        // Done before light reconciliation so the new AABB is what gets tested.
        if (node.meshIndex < meshes.size() && node.boundingBoxValid) {
            const auto& mesh = meshes[node.meshIndex];
            glm::vec3 corners[8] = {
                glm::vec3(mesh.boundingBoxMin.x, mesh.boundingBoxMin.y, mesh.boundingBoxMin.z), glm::vec3(mesh.boundingBoxMax.x, mesh.boundingBoxMin.y, mesh.boundingBoxMin.z),
                glm::vec3(mesh.boundingBoxMin.x, mesh.boundingBoxMax.y, mesh.boundingBoxMin.z), glm::vec3(mesh.boundingBoxMax.x, mesh.boundingBoxMax.y, mesh.boundingBoxMin.z),
                glm::vec3(mesh.boundingBoxMin.x, mesh.boundingBoxMin.y, mesh.boundingBoxMax.z), glm::vec3(mesh.boundingBoxMax.x, mesh.boundingBoxMin.y, mesh.boundingBoxMax.z),
                glm::vec3(mesh.boundingBoxMin.x, mesh.boundingBoxMax.y, mesh.boundingBoxMax.z), glm::vec3(mesh.boundingBoxMax.x, mesh.boundingBoxMax.y, mesh.boundingBoxMax.z)};

            node.boundingBoxMin = glm::vec3(std::numeric_limits<float>::max());
            node.boundingBoxMax = glm::vec3(std::numeric_limits<float>::lowest());

            for (const auto& corner : corners) {
                glm::vec3 worldCorner = glm::vec3(node.worldTransform * glm::vec4(corner, 1.0f));
                node.boundingBoxMin = glm::min(node.boundingBoxMin, worldCorner);
                node.boundingBoxMax = glm::max(node.boundingBoxMax, worldCorner);
            }
        }

        // Update light direction if node has a light, and rebuild that light's
        // influence set since its world position may have moved.
        if (node.lightIndex != MAX_LIGHTS && lights.contains(node.lightIndex)) {
            Light& L = lights[node.lightIndex];
            L.direction = glm::normalize(node.worldTransform[2]);
            L.shadowDirty = true;
            L.gpuDirtyFrames = MAX_FRAMES_IN_FLIGHT;
            LightInfluence::rebuildLightSet(L, nodes);
        }

        // Reconcile this node's AABB against every shadow-casting point light.
        LightInfluence::onMeshedNodeTouched(node, nodes, lights);

        uint32_t child = node.firstChild;
        while (child != 0) {
            syncToGPU(nodes[child], nodes, ds, modelMatrixBufferIndex, meshes, lights);
            child = nodes[child].nextSibling;
        }
    }

    // Full update: recompute transforms then sync to GPU (replaces Node::update())
    template<typename MeshArray>
    void updateAll(Node& node, std::vector<Node>& nodes, DescriptorSet& ds, uint32_t modelMatrixBufferIndex,
                   const MeshArray& meshes, std::unordered_map<uint32_t, Light>& lights) {
        recomputeTransforms(node, nodes);
        syncToGPU(node, nodes, ds, modelMatrixBufferIndex, meshes, lights);
    }

}
