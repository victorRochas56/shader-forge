#include "node_ops.hpp"
#include "renderer.hpp"

namespace NodeOps {

void assignMesh(Node& node, uint32_t meshIndex, Renderer& renderer) {
    if (node.meshIndex < renderer.scene.assetManager.meshes.size()) {
        renderer.scene.assetManager.meshes[node.meshIndex].refCount--;
        if (renderer.scene.assetManager.meshes[node.meshIndex].refCount <= 0) {
            renderer.scene.assetManager.meshes[node.meshIndex].freed = true;
        }
    }
    node.meshIndex = meshIndex;
    renderer.scene.assetManager.meshes[meshIndex].refCount++;

    // Calculate world-space bounding box from mesh local-space bounding box
    const auto& mesh = renderer.scene.assetManager.meshes[meshIndex];
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

void assignMaterial(Node& node, uint32_t materialIndex, Renderer& renderer) {
    if (node.meshIndex >= renderer.scene.assetManager.meshes.size()) {
        throw std::runtime_error("tried assigning material to a node without a mesh!");
    }
    // Remove old shader mapping if a material was previously assigned
    if (node.materialIndex != 0xFFFFFFFF) {
        renderer.removeMeshFromShader(node.nodeIndex, renderer.scene.getMaterials()[node.materialIndex].shaderSource,
                                       renderer.scene.getMaterials()[node.materialIndex]);
    }
    node.materialIndex = materialIndex;
    renderer.addMeshToShader(node.nodeIndex, renderer.scene.getMaterials()[materialIndex].shaderSource,
                              renderer.scene.getMaterials()[materialIndex]);
}

void enableLightShadows(Light& light, const std::string& nodeName, Renderer& renderer) {
    light.castsShadows = 1;
    light.shadowDirty = true;
    switch (light.type) {
    case LightType::Directional:
        for (int i = 0; i < light.numCascades; i++) {
            renderer.scene.shadowAtlas.allocateShadowMap(light.shadowResolution, light.cascades[i].shadowAtlasTile, light.cascades[i].shadowAtlasUVRange);
        }
        break;
    case LightType::Point:
        for (int i = 0; i < 6; i++) {
            renderer.scene.shadowAtlas.allocateShadowMap(light.shadowResolution, light.cubeMapIndices[i].shadowAtlasTile, light.cubeMapIndices[i].shadowAtlasUVRange);
        }
        break;
    default:
        break;
    }
}

void disableLightShadows(Light& light, Renderer& renderer) {
    light.castsShadows = 0;
    switch (light.type) {
    case LightType::Directional:
        for (int i = 0; i < light.numCascades; i++) {
            renderer.scene.shadowAtlas.freeShadowMap(light.cascades[i].shadowAtlasTile);
        }
        break;
    case LightType::Point:
        for (int i = 0; i < 6; i++) {
            renderer.scene.shadowAtlas.freeShadowMap(light.cubeMapIndices[i].shadowAtlasTile);
        }
        break;
    default:
        break;
    }
}

void assignLight(Node& node, Light light, Renderer& renderer) {
    light.nodeIndex = node.nodeIndex;
    light.modelMatrixIndex = node.modelMatrixIndices[0];

    if (light.castsShadows == 1) {
        light.castsShadows = 0; // reset so enableLightShadows sets it
        enableLightShadows(light, node.name, renderer);
    }

    node.lightIndex = (*renderer.bindless.descriptorSet).allocateFixedBuffer<GPULight>(renderer.getLightBufferIndex(), light.toGPU(node.getWorldPosition(), node.forward()));
    renderer.scene.addLight(node.lightIndex, light);
    std::cout << "added light at index : " << node.lightIndex << std::endl;
    node.transformDirty = true;
}

}
