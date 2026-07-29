#include "node_ops.hpp"
#include "scene.hpp"

namespace NodeOps {

void assignMesh(Node& node, uint32_t meshIndex, Scene& scene) {
    if (node.meshIndex < scene.assetManager.meshes.size()) {
        scene.assetManager.meshes[node.meshIndex].refCount--;
        if (scene.assetManager.meshes[node.meshIndex].refCount <= 0) {
            scene.assetManager.meshes[node.meshIndex].freed = true;
        }
    }
    node.meshIndex = meshIndex;
    scene.assetManager.meshes[meshIndex].refCount++;

    // Calculate world-space bounding box from mesh local-space bounding box
    const auto& mesh = scene.assetManager.meshes[meshIndex];
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
    node.boundingBoxValid = true;
    // Flow through syncDirtyNodes so LightInfluence reconciles this new AABB
    // against every point light.
    node.transformDirty = true;
}

void assignMaterial(Node& node, uint32_t materialIndex, Scene& scene) {
    if (node.meshIndex >= scene.assetManager.meshes.size()) {
        throw std::runtime_error("tried assigning material to a node without a mesh!");
    }
    // Remove old shader mapping if a material was previously assigned
    if (node.materialIndex != 0xFFFFFFFF) {
        scene.removeMeshFromShader(node.nodeIndex, scene.getMaterials()[node.materialIndex].shaderSource,
                                    scene.getMaterials()[node.materialIndex]);
    }
    node.materialIndex = materialIndex;
    scene.addMeshToShader(node.nodeIndex, scene.getMaterials()[materialIndex].shaderSource,
                           scene.getMaterials()[materialIndex]);
}

void enableLightShadows(Light& light, const std::string& nodeName, Scene& scene) {
    light.castsShadows = 1;
    light.shadowDirty = true;
    switch (light.type) {
    case LightType::Directional:
        for (int i = 0; i < light.numCascades; i++) {
            scene.shadowAtlas.allocateShadowMap(light.shadowResolution, light.cascades[i].shadowAtlasTile, light.cascades[i].shadowAtlasUVRange);
        }
        scene.shadowAtlas.allocateShadowMap(VXGI_DIRECTIONAL_SHADOW_RESOLUTION, light.shadowMaps[0].shadowAtlasTile, light.shadowMaps[0].shadowAtlasUVRange);
        break;
    case LightType::Point:
        for (int i = 0; i < 6; i++) {
            scene.shadowAtlas.allocateShadowMap(light.shadowResolution, light.shadowMaps[i].shadowAtlasTile, light.shadowMaps[i].shadowAtlasUVRange);
        }
        break;
    default:
        break;
    }
}

void disableLightShadows(Light& light, Scene& scene) {
    light.castsShadows = 0;
    switch (light.type) {
    case LightType::Directional:
        for (int i = 0; i < light.numCascades; i++) {
            scene.shadowAtlas.freeShadowMap(light.cascades[i].shadowAtlasTile);
        }
        break;
    case LightType::Point:
        for (int i = 0; i < 6; i++) {
            scene.shadowAtlas.freeShadowMap(light.shadowMaps[i].shadowAtlasTile);
        }
        break;
    default:
        break;
    }
}

void assignLight(Node& node, Light light, Scene& scene, BindlessSystem& bindless, uint32_t lightBufferIndex) {
    light.nodeIndex = node.nodeIndex;
    light.modelMatrixIndex = node.modelMatrixIndices[0];

    if (light.castsShadows == 1) {
        light.castsShadows = 0; // reset so enableLightShadows sets it
        enableLightShadows(light, node.name, scene);
    }

    node.lightIndex = scene.addLight(bindless, lightBufferIndex, light, light.toGPU(node.getWorldPosition(), node.forward()));
    std::cout << "added light at index : " << node.lightIndex << std::endl;
    node.transformDirty = true;
}

void assignVolume(Node& node, Volume volume, Scene& scene) {
    volume.nodeIndex = node.nodeIndex;
    scene.addVolume(node.nodeIndex, volume);
}

void assignBillboard(Node& node, Billboard billboard, Scene& scene) {
    billboard.nodeIndex = node.nodeIndex;
    scene.addBillboard(node.nodeIndex,billboard);
}

void assignEmitter(Node& node, ParticleEmitter emitter, Scene& scene, BindlessSystem& bindless, const RenderBuffers& buffers) {
    emitter.nodeIndex = node.nodeIndex;
    node.particleIndex = scene.addEmitter(bindless, emitter, buffers);
}

}
