#pragma once

#include <map>
#include <vector>

#include "asset_manager.hpp"
#include "bindless_system.hpp"
#include "constants.hpp"
#include "scene_elements.hpp"
#include "scene_graph.hpp"
#include "structs.hpp"

/*
Scene is the "what to draw" container: nodes, meshes, materials, lights,
camera, shadow atlas — the authoring-time state of the world. Renderer
holds a Scene& and decides *how* to draw it; external tools (GUI, scene
loaders, node ops) read and mutate Scene directly.

Scene does NOT own or depend on rendering passes. It does, however, need
the BindlessSystem + the light buffer index at clearLights time so it can
free the light slots it allocated. That's passed in rather than stored.
*/
class Scene {
  public:
    Scene() = default;
    ~Scene() = default;
    Scene(const Scene&) = delete;
    Scene& operator=(const Scene&) = delete;

    SceneGraph                 sceneGraph;
    AssetManager               assetManager;
    Camera                     activeCamera;
    ShadowAtlas                shadowAtlas;
    std::map<uint32_t, Light>  lights;
    std::map<uint32_t, Volume> volumes;
    std::vector<Material>      materials;
    Shader                     fallbackLitShader;
    uint32_t                   fallbackDefaultMaterialIndex = 0;
    uint32_t                   skyboxIndex = 0;

    void init(BindlessSystem& bindless, uint32_t vertexBufferIndex, uint32_t indexBufferIndex) {
        assetManager.init(bindless.resourceManager.get(), bindless.descriptorSet.get(), vertexBufferIndex, indexBufferIndex);
    }

    // --- materials -----------------------------------------------------
    std::vector<Material>& getMaterials() { return materials; }
    uint32_t addMaterial(Material material) {
        material.materialID = static_cast<uint32_t>(std::hash<Material>{}(material));
        for (uint32_t i = 0; i < materials.size(); i++) {
            if (materials[i] == material) return i;
        }
        materials.push_back(material);
        return materials.size() - 1;
    }

    // --- lights --------------------------------------------------------
    const std::map<uint32_t, Light>& getLights() const { return lights; }
    std::map<uint32_t, Light>&       getLightsMutable() { return lights; }
    void                             addLight(uint32_t index, Light light) { lights[index] = light; }
    Light&                           getLight(uint32_t index) { return lights[index]; }

    // Shader loop bound for the (sparse) light buffer. Slots are never compacted,
    // so the highest live key defines how far the shader must iterate. Gaps in
    // the range are kept skippable by writing a disabled GPULight sentinel
    // (intensity == 0) in removeLight — see below.
    uint32_t getLightLoopBound() const {
        return lights.empty() ? 0u : lights.rbegin()->first + 1u;
    }

    // Clears lights AND frees every shadow-atlas tile those lights held.
    // Caller supplies the bindless system and the light buffer index so Scene
    // doesn't have to store either.
    void clearLights(BindlessSystem& bindless, uint32_t lightBufferIndex) {
        for (auto& [idx, light] : lights) {
            if (!light.castsShadows) continue;
            switch (light.type) {
            case LightType::Directional:
                for (int i = 0; i < light.numCascades; i++)
                    shadowAtlas.freeShadowMap(light.cascades[i].shadowAtlasTile);
                break;
            case LightType::Point:
                for (int i = 0; i < 6; i++)
                    shadowAtlas.freeShadowMap(light.cubeMapIndices[i].shadowAtlasTile);
                break;
            default:
                break;
            }
        }
        bindless.descriptorSet->clearFixedBuffer(lightBufferIndex);
        lights.clear();
    }

    void removeLight(BindlessSystem& bindless, uint32_t lightBufferIndex, uint32_t lightIndex) {
        Light& light = lights[lightIndex];

        // Stamp a disabled sentinel into every frame slice. freeFixedBuffer only
        // touches CPU bookkeeping; without this write, the slot keeps its last
        // GPULight contents and the shader (which iterates 0..maxSlot) would
        // still light the scene from the dead slot.
        GPULight disabled{};
        disabled.intensity = 0.0f;
        disabled.castsShadows = 0;
        for (uint32_t frame = 0; frame < MAX_FRAMES_IN_FLIGHT; frame++) {
            bindless.descriptorSet->updateFixedBufferWithOffset<GPULight>(lightBufferIndex, lightIndex, disabled, frame);
        }
        bindless.descriptorSet->freeFixedBuffer(lightBufferIndex, lightIndex);
        if (light.castsShadows) {
            switch (light.type) {
                case LightType::Directional:
                    for (int i = 0; i < light.numCascades; i++)
                        shadowAtlas.freeShadowMap(light.cascades[i].shadowAtlasTile);
                    break;
                case LightType::Point:
                    for (int i = 0; i < 6; i++)
                        shadowAtlas.freeShadowMap(light.cubeMapIndices[i].shadowAtlasTile);
                    break;
                default:
                    break;
            }
        }
        lights.erase(lightIndex);
    }

    // --- volumes -------------------------------------------------------
    const std::map<uint32_t, Volume>& getVolumes() const { return volumes; }
    std::map<uint32_t, Volume>&       getVolumesMutable() { return volumes; }
    void                              addVolume(uint32_t index, Volume volume) { volumes[index] = volume; }
    Volume&                           getVolume(uint32_t index) { return volumes[index]; }

    // Same rationale as getLightLoopBound — volumes share the sparse-slot model.
    uint32_t getVolumeLoopBound() const {
        return volumes.empty() ? 0u : volumes.rbegin()->first + 1u;
    }

    void clearVolumes(BindlessSystem& bindless, uint32_t volumeBufferIndex) {
        bindless.descriptorSet->clearFixedBuffer(volumeBufferIndex);
        volumes.clear();
    }

    void removeVolume(BindlessSystem& bindless, uint32_t volumeBufferIndex, uint32_t volumeIndex) {
        // Same dead-slot trick as removeLight: density == 0 makes the shader skip.
        Volume disabled{};
        disabled.density = 0.0f;
        disabled.radius = 0.0f;
        bindless.descriptorSet->updateFixedBuffer<Volume>(volumeBufferIndex, volumeIndex, disabled);
        bindless.descriptorSet->freeFixedBuffer(volumeBufferIndex,volumeIndex);
        volumes.erase(volumeIndex);
    }

    // --- defaults / skybox ---------------------------------------------
    Shader   getFallBackShader() const    { return fallbackLitShader; }
    uint32_t getFallBackMaterial() const  { return fallbackDefaultMaterialIndex; }
    void     setSkyBox(uint32_t idx)      { skyboxIndex = idx; }
    uint32_t getSkyBox() const            { return skyboxIndex; }
};
