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
    std::unordered_map<uint32_t, Light>  lights;
    std::unordered_map<uint32_t, Volume> volumes;
    std::unordered_map<uint32_t, Billboard> billboards;
    std::vector<Material>      materials;
    Shader                     fallbackLitShader;
    uint32_t                   fallbackDefaultMaterialIndex = 0;
    uint32_t                   skyboxIndex = 0;

    // Render list — derived index of (node, material, shader) tuples that the
    // renderer iterates each frame. Lives on Scene because every mutation
    // (assignMaterial, node delete, scene clear) is already a scene-level
    // operation; the renderer only reads + sorts during draw.
    struct RenderEntry {
        uint32_t nodeIndex;
        uint32_t meshIndex;
        uint32_t materialIndex;       // index into materials vector
        uint32_t shaderPipelineIndex;
    };
    struct ShaderDrawRange {
        uint32_t pipelineIndex;
        uint32_t firstCommand;
        uint32_t commandCount;
    };
    std::vector<RenderEntry>     renderEntries;
    std::vector<ShaderDrawRange> shaderDrawRanges;
    bool                         renderListDirty = false;

    // --- materials -----------------------------------------------------
    std::vector<Material>& getMaterials() { return materials; }
    uint32_t addMaterial(Material material) {
        material.materialID = static_cast<uint32_t>(std::hash<Material>{}(material));
        for (uint32_t i = 0; i < materials.size(); i++) {
            if (materials[i].name == material.name) return i;
        }
        materials.push_back(material);
        return materials.size() - 1;
    }

    // --- lights --------------------------------------------------------
    const std::unordered_map<uint32_t, Light>& getLights() const { return lights; }
    std::unordered_map<uint32_t, Light>&       getLightsMutable() { return lights; }
    // Allocates a bindless slot for the GPULight payload and stores the CPU-side
    // Light against the same key. Returns the assigned index. Callers don't have
    // to manage the bindless allocation separately — same rationale as removeLight.
    uint32_t addLight(BindlessSystem& bindless, uint32_t lightBufferIndex, Light light, GPULight gpuLight) {
        uint32_t idx = bindless.descriptorSet->allocateFixedBuffer<GPULight>(lightBufferIndex, gpuLight);
        lights[idx] = light;
        return idx;
    }
    Light&                                     getLight(uint32_t index) { return lights[index]; }

    // Shader loop bound for the (sparse) light buffer. Slots are never compacted,
    // so the highest live key defines how far the shader must iterate. Gaps in
    // the range are kept skippable by writing a disabled GPULight sentinel
    // (intensity == 0) in removeLight — see below.
    uint32_t getLightLoopBound() const {
        return lights.empty() ? 0u : lights.size();
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
    const std::unordered_map<uint32_t, Volume>& getVolumes() const { return volumes; }
    std::unordered_map<uint32_t, Volume>&       getVolumesMutable() { return volumes; }
    // Same shape as addLight: allocates the bindless slot and stores the volume
    // under the assigned index, which is returned to the caller.
    uint32_t addVolume(BindlessSystem& bindless, uint32_t volumeBufferIndex, Volume volume) {
        uint32_t idx = bindless.descriptorSet->allocateFixedBuffer<Volume>(volumeBufferIndex, volume);
        volumes[idx] = volume;
        return idx;
    }
    Volume&                                     getVolume(uint32_t index) { return volumes[index]; }

    // Same rationale as getLightLoopBound — volumes share the sparse-slot model.
    uint32_t getVolumeLoopBound() const {
        return volumes.empty() ? 0u : volumes.size();
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

    const std::unordered_map<uint32_t, Billboard>& getBillboards() const { return billboards; }
    std::unordered_map<uint32_t, Billboard>&       getBillboardsMutable() { return billboards; }
    void addBillboard(uint32_t index, Billboard billboard) { billboards[index] = billboard; }
    Billboard& getBillboard(uint32_t index) { return billboards[index]; }
    void removeBillboard(uint32_t index) { billboards.erase(index); }
    void clearBillboards() { billboards.clear(); }

    // --- render list ---------------------------------------------------
    void addMeshToShader(uint32_t nodeIndex, Shader shader, Material material) {
        uint32_t matIdx = 0;
        for (uint32_t i = 0; i < materials.size(); i++) {
            if (materials[i] == material) { matIdx = i; break; }
        }
        for (const auto& e : renderEntries) {
            if (e.nodeIndex == nodeIndex && e.materialIndex == matIdx && e.shaderPipelineIndex == shader.pipelineIndex && e.meshIndex == sceneGraph.getNode(nodeIndex).meshIndex)
                return;
        }
        renderEntries.push_back({nodeIndex, sceneGraph.getNode(nodeIndex).meshIndex, matIdx, shader.pipelineIndex});
        renderListDirty = true;
    }

    void removeMeshFromShader(uint32_t nodeIndex, Shader shader, Material material) {
        uint32_t matIdx = 0;
        for (uint32_t i = 0; i < materials.size(); i++) {
            if (materials[i] == material) { matIdx = i; break; }
        }
        for (size_t i = 0; i < renderEntries.size(); ++i) {
            if (renderEntries[i].nodeIndex == nodeIndex && renderEntries[i].materialIndex == matIdx && 
                renderEntries[i].shaderPipelineIndex == shader.pipelineIndex && renderEntries[i].meshIndex == sceneGraph.getNode(nodeIndex).meshIndex) {
                renderEntries[i] = renderEntries.back();
                renderEntries.pop_back();
                renderListDirty = true;
                return;
            }
        }
    }

    void removeNodeFromRenderList(uint32_t nodeIndex) {
        for (size_t i = 0; i < renderEntries.size();) {
            if (renderEntries[i].nodeIndex == nodeIndex) {
                renderEntries[i] = renderEntries.back();
                renderEntries.pop_back();
                renderListDirty = true;
            } else {
                ++i;
            }
        }
    }

    void clearRenderList() {
        renderEntries.clear();
        shaderDrawRanges.clear();
        renderListDirty = false;
    }

    // --- defaults / skybox ---------------------------------------------
    Shader   getFallBackShader() const    { return fallbackLitShader; }
    uint32_t getFallBackMaterial() const  { return fallbackDefaultMaterialIndex; }
    void     setSkyBox(uint32_t idx)      { skyboxIndex = idx; }
    uint32_t getSkyBox() const            { return skyboxIndex; }
};
