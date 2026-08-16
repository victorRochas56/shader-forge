#pragma once

#include <cstring>
#include <map>
#include <vector>

#include "asset_manager.hpp"
#include "bindless_system.hpp"
#include "constants.hpp"
#include "render_buffers.hpp"
#include "scene_elements.hpp"
#include "scene_graph.hpp"
#include "structs.hpp"

/*
Scene is the "what to draw" container: nodes, meshes, materials, lights,
camera, shadow atlas
Scene does NOT own or depend on rendering passes. 
It does need the BindlessSystem + the light buffer index at clearLights time so it can
free the light slots it allocated.
*/
class Scene {
  public:
    Scene() = default;
    ~Scene() = default;
    Scene(const Scene&) = delete;
    Scene& operator=(const Scene&) = delete;

    SceneGraph                                      sceneGraph;
    AssetManager                                    assetManager;
    Camera                                          activeCamera;
    ShadowAtlas                                     shadowAtlas;

    std::unordered_map<uint32_t, Light>             lights;
    std::unordered_map<uint32_t, ParticleEmitter>   particleEmitters;
    std::unordered_map<uint32_t, Volume>            volumes;
    std::unordered_map<uint32_t, Billboard>         billboards;
    std::vector<Material>                           materials;
    
    Shader                                          fallbackLitShader;
    uint32_t                                        fallbackDefaultMaterialIndex = 0;
    uint32_t                                        skyboxIndex = 0;

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

    // Fills the hot-light mirror for a lit frame UBO. Zeroed slots read as disabled
    // sentinels (intensity 0), matching the sparse GPULight buffer. Returns the loop bound.
    //
    // Shadow-casting directional lights also get their 3 cascades copied into the flat
    // cascade pool, with the base slot stored in typeFlags.w — that keeps the lit pass's
    // shadow lookup out of the BDA Light buffer entirely. Directional casters past the pool
    // are demoted to non-casting rather than left pointing at another light's cascades.
    uint32_t fillHotLights(GPULightHot* dst, GPUCascadeHot* cascadeDst) {
        std::memset(dst, 0, sizeof(GPULightHot) * MAX_UBO_LIGHTS);
        std::memset(cascadeDst, 0, sizeof(GPUCascadeHot) * MAX_UBO_CASCADES);
        uint32_t bound = 0;
        uint32_t cascadeSlot = 0;
        for (auto& [idx, light] : lights) {
            if (idx >= MAX_UBO_LIGHTS) continue;
            auto& node = sceneGraph.getNode(light.nodeIndex);
            glm::vec3 pos = node.getWorldPosition();
            glm::vec3 dir = node.forward();

            uint32_t castsShadows = static_cast<uint32_t>(light.castsShadows);
            uint32_t cascadeBase = 0;
            if (castsShadows && light.type == LightType::Directional) {
                uint32_t count = std::min(light.numCascades, 3u);
                if (cascadeSlot + count <= MAX_UBO_CASCADES) {
                    cascadeBase = cascadeSlot;
                    for (uint32_t c = 0; c < count; c++) {
                        GPUCascadeHot& ch = cascadeDst[cascadeSlot + c];
                        ch.lightSpaceMatrix = light.cascades[c].lightSpaceMatrix;
                        ch.shadowAtlasRange = light.cascades[c].shadowAtlasUVRange;
                        ch.texelSizes = glm::vec4(light.cascades[c].texelSize, light.cascades[c].worldTexelSize, 0.0f, 0.0f);
                    }
                    cascadeSlot += count;
                } else {
                    castsShadows = 0; // pool exhausted; light stays lit rather than sampling a stale tile
                }
            }

            GPULightHot& h = dst[idx];
            h.positionRange = glm::vec4(pos, light.range);
            h.direction = glm::vec4(dir, 0.0f);
            h.colorIntensity = glm::vec4(glm::vec3(light.color), light.intensity);
            h.typeFlags = glm::uvec4(static_cast<uint32_t>(light.type), castsShadows, light.numCascades, cascadeBase);
            h.cascadeSplits = glm::vec4(light.cascades[0].splitDistance, light.cascades[1].splitDistance, light.cascades[2].splitDistance, 0.0f);
            bound = std::max(bound, idx + 1);
        }
        return bound;
    }

    // Cascade-visualization is a debug path; when any light has it on, the renderer
    // falls back to the uber lit shader that still compiles it in.
    bool anyLightShowsCascades() const {
        for (const auto& [idx, light] : lights)
            if (light.showCascades > 0) return true;
        return false;
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
                    shadowAtlas.freeShadowMap(light.shadowMaps[i].shadowAtlasTile);
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
                        shadowAtlas.freeShadowMap(light.shadowMaps[i].shadowAtlasTile);
                    break;
                default:
                    break;
            }
        }
        lights.erase(lightIndex);
    }

    // --- volumes -------------------------------------------------------
    // Streamed like billboards: keyed by owning node index, rebuilt into the volume buffer every
    // frame by VolumetricsPass. No bindless slot lifecycle — add/remove are pure map operations.
    const std::unordered_map<uint32_t, Volume>& getVolumes() const { return volumes; }
    std::unordered_map<uint32_t, Volume>&       getVolumesMutable() { return volumes; }
    void addVolume(uint32_t nodeIndex, Volume volume) { volumes[nodeIndex] = volume; }
    Volume&                                     getVolume(uint32_t nodeIndex) { return volumes[nodeIndex]; }
    void clearVolumes() { volumes.clear(); }
    void removeVolume(uint32_t nodeIndex) { volumes.erase(nodeIndex); }

    const std::unordered_map<uint32_t, Billboard>& getBillboards() const { return billboards; }
    std::unordered_map<uint32_t, Billboard>&       getBillboardsMutable() { return billboards; }
    void addBillboard(uint32_t index, Billboard billboard) { billboards[index] = billboard; }
    Billboard& getBillboard(uint32_t index) { return billboards[index]; }
    void removeBillboard(uint32_t index) { billboards.erase(index); }
    void clearBillboards() { billboards.clear(); }

    uint32_t addEmitter(BindlessSystem& bindless, ParticleEmitter emitter, const RenderBuffers& buffers) {
        auto& descriptorSet = *bindless.descriptorSet;

        emitter.particleCapacity = emitter.capacity();
        Particle dead{};
        dead.age = -1.0f;
        uint32_t byteOffset = descriptorSet.allocateVariableBuffer<Particle>(std::vector<Particle>(emitter.particleCapacity, dead), buffers.particlePoolBufferIndex);
        emitter.particleOffset = byteOffset / static_cast<uint32_t>(sizeof(Particle));

        Node& node = sceneGraph.getNode(emitter.nodeIndex);
        glm::quat worldRot = node.getWorldRotation();
        glm::vec3 spawnPos = node.getWorldPosition() + worldRot * emitter.positionOffset;
        glm::quat spawnRot = worldRot * emitter.rotationOffset;

        // allocate the descriptor slot (fanned across every frame slice by allocateFixedBuffer).
        uint32_t idx = descriptorSet.allocateFixedBuffer<GPUParticleEmitter>(buffers.emitterBufferIndex, emitter.toGPU(spawnPos, spawnRot));

        // Zero the GPU-owned runtime slot at the same index
        if (auto* runtime = descriptorSet.getFixedBufferMappedData<EmitterRuntime>(buffers.emitterRuntimeBufferIndex))
            runtime[idx] = EmitterRuntime{};

        particleEmitters[idx] = emitter;
        return idx;
    }

    void removeEmitter(BindlessSystem& bindless, const RenderBuffers& buffers, uint32_t emitterIndex) {
        auto& ds = *bindless.descriptorSet;
        auto it = particleEmitters.find(emitterIndex);
        if (it == particleEmitters.end()) return;

        // Release the pool sub-range (any live particles in it simply vanish with the range).
        ds.freeVariableBuffer(buffers.particlePoolBufferIndex, it->second.particleOffset * static_cast<uint32_t>(sizeof(Particle)));

        // Stamp a disabled sentinel into every frame slice before freeing, same as removeLight.
        GPUParticleEmitter disabled{};
        disabled.emissionRate = 0.0f;
        for (uint32_t frame = 0; frame < MAX_FRAMES_IN_FLIGHT; frame++) {
            ds.updateFixedBufferWithOffset<GPUParticleEmitter>(buffers.emitterBufferIndex, emitterIndex, disabled, frame);
        }
        ds.freeFixedBuffer(buffers.emitterBufferIndex, emitterIndex);
        particleEmitters.erase(it);
    }

    // Free every emitter's pool sub-range and descriptor slot, then drop the map.
    // Used on scene clear/reload so emitters don't orphan GPU slots.
    void clearEmitters(BindlessSystem& bindless, const RenderBuffers& buffers) {
        auto& ds = *bindless.descriptorSet;
        GPUParticleEmitter disabled{};
        disabled.emissionRate = 0.0f;
        for (auto& [idx, emitter] : particleEmitters) {
            ds.freeVariableBuffer(buffers.particlePoolBufferIndex, emitter.particleOffset * static_cast<uint32_t>(sizeof(Particle)));
            for (uint32_t frame = 0; frame < MAX_FRAMES_IN_FLIGHT; frame++) {
                ds.updateFixedBufferWithOffset<GPUParticleEmitter>(buffers.emitterBufferIndex, idx, disabled, frame);
            }
            ds.freeFixedBuffer(buffers.emitterBufferIndex, idx);
        }
        particleEmitters.clear();
    }

    const std::unordered_map<uint32_t, ParticleEmitter>& getEmitters() const { return particleEmitters; }

    // Re-reserve an emitter's pool sub-range after an edit to emissionRate/lifeTime (the fields that
    // feed capacity()). Keeps the emitter's descriptor slot (so node.particleIndex stays valid); only
    // the variable-buffer range moves. No-op when the rounded capacity is unchanged.
    void resizeEmitterPool(BindlessSystem& bindless, const RenderBuffers& buffers, uint32_t emitterIndex) {
        auto it = particleEmitters.find(emitterIndex);
        if (it == particleEmitters.end()) return;
        auto& emitter = it->second;

        uint32_t newCapacity = emitter.capacity();
        if (newCapacity == emitter.particleCapacity) return; // rounding left the ring size unchanged

        auto& ds = *bindless.descriptorSet;
        // Release the old range, then reserve a fresh one filled with dead sentinels.
        ds.freeVariableBuffer(buffers.particlePoolBufferIndex, emitter.particleOffset * static_cast<uint32_t>(sizeof(Particle)));
        Particle dead{};
        dead.age = -1.0f;
        uint32_t byteOffset = ds.allocateVariableBuffer<Particle>(std::vector<Particle>(newCapacity, dead), buffers.particlePoolBufferIndex);
        emitter.particleOffset   = byteOffset / static_cast<uint32_t>(sizeof(Particle));
        emitter.particleCapacity = newCapacity;

        // Range moved and old particles are gone — reset the ring bookkeeping so the head/accumulator
        // don't index past the new capacity. record() re-uploads the descriptor (with the new
        // offset/capacity) next frame, so no explicit emitter re-upload here.
        if (auto* runtime = ds.getFixedBufferMappedData<EmitterRuntime>(buffers.emitterRuntimeBufferIndex))
            runtime[emitterIndex] = EmitterRuntime{};
    }
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

    // --- lit / lit-derived shaders (selectable per material) -----------
    // Registered at startup by the renderer. All share the LIT_GEOMETRY pipeline + LitPushConstants
    // interface, so a material can point shaderSource at any of them interchangeably.
    std::vector<Shader> litShaders;
    void registerLitShader(const Shader& shader) { litShaders.push_back(shader); }
    const std::vector<Shader>& getLitShaders() const { return litShaders; }

    // Resolves a shader source path (as stored in a .scn) back to a registered shader so the
    // pipelineIndex is correct on reload; falls back to the default lit shader if unknown.
    Shader resolveLitShader(const std::string& sourceFile) const {
        for (const auto& s : litShaders) {
            if (s.sourceFile == sourceFile) return s;
        }
        return fallbackLitShader;
    }

    // --- defaults / skybox ---------------------------------------------
    Shader   getFallBackShader() const    { return fallbackLitShader; }
    uint32_t getFallBackMaterial() const  { return fallbackDefaultMaterialIndex; }
    void     setSkyBox(uint32_t idx)      { skyboxIndex = idx; }
    uint32_t getSkyBox() const            { return skyboxIndex; }
};
