# Plan: Layered refactor of `Renderer`

**Scope:** 107 call sites across 4 files (`app.hpp`, `node_gui.cpp`, `node_ops.cpp`, `scenes.hpp`) + `renderer.cpp` (1725 lines). Manageable, but do it in staged phases — each phase must compile and run before the next.

---

## Target shape

```
App (owns everything, wires the graph)
  ├─ GpuContext       "talk to GPU"       Instance, Device, Swapchain, CommandPool, Sync, msaaSamples
  ├─ BindlessSystem   "GPU resources"     DescriptorSet, ResourceManager, PipelineManager
  ├─ Scene            "what to draw"      SceneGraph, AssetManager, lights, materials, Camera, ShadowAtlas
  └─ Renderer         "how to draw"       RenderFeatures, per-pass state, draw logic
        ↑ holds GpuContext&, BindlessSystem&, Scene& (non-owning)
```

Dependency rule: **downward only.** `Renderer → Scene → BindlessSystem → GpuContext`. No back-pointers.

---

## Phase 1 — Extract `GpuContext` (lowest risk, biggest immediate payoff)

**Goal:** bundle the low-level Vulkan primitives so subsystems take one reference instead of three.

1. Create `modules/gpu_context.hpp` with a struct holding: `Instance`, `Device` (owned `unique_ptr`), `Swapchain` (owned), `CommandPool`, `commandBuffers`, sync objects (`presentComplete/renderFinished/inFlightFences`), `graphicsIndex`, `msaaSamples`, `currentFrame`/`totalFrames`, `window*`.
2. Move their `create*` methods (`createInstance`, `createSurface`, `createCommandPool`, `createCommandBuffers`, `createSyncObjects`, debug messenger) out of `Renderer` and into `GpuContext::init(width, height)`.
3. In `Renderer`, replace those members with `GpuContext& gpu;` held by reference. Update every `device->...` / `commandBuffers[...]` inside `renderer.cpp` — mostly mechanical find/replace.
4. `App` now owns `GpuContext` and passes `gpu` by reference into `Renderer`.
5. Keep `renderer.getDevice()` etc. as thin forwarders for one phase to avoid touching every external call site yet.

**Exit criteria:** compiles, renders identically. `renderer.hpp` drops ~40 lines.

---

## Phase 2 — Extract `BindlessSystem`

**Goal:** group GPU resource ownership.

1. Create `modules/bindless_system.hpp` owning `ResourceManager`, `DescriptorSet`, `PipelineManager` (all currently `unique_ptr` in `Renderer`).
2. Constructor takes `GpuContext&`. Move `DescriptorSet::createDescriptorSet()` orchestration here.
3. Move the default buffer/sampler/texture indices currently in `Renderer` (`vertexBufferIndex`, `indexBufferIndex`, `defaultSamplerIndex`, `depthSamplerIndex`, `shadowSamplerIndex`, `defaultNormalIndex`, `modelMatrixBufferIndex`, `lightBufferIndex`, the draw-data buffer indices) into `BindlessSystem`. These are shared infrastructure, not renderer-private.
4. `Renderer` now holds `BindlessSystem& bindless;`. Pass accessors through temporarily.

**Exit criteria:** compiles, renders identically. The 7 `std::unique_ptr`s in renderer.hpp collapse to two references.

---

## Phase 3 — Extract `Scene`

**Goal:** lift scene state out of the renderer. This is the phase that touches external call sites.

1. Create `modules/scene.hpp` owning: `SceneGraph`, `AssetManager`, `std::map<uint32_t, Light> lights`, `std::vector<Material> materials`, `Camera activeCamera`, `ShadowAtlas shadowAtlas`, `fallbackLitShader`, `fallbackDefaultMaterialIndex`, skybox index.
2. Move these methods off `Renderer` onto `Scene`: `addMaterial`, `getMaterials`, `addLight/getLight/clearLights/getLights[Mutable]`, `getFallBackShader/Material`, `setSkyBox`.
3. `Scene::init(BindlessSystem&)` — `AssetManager::init` call moves here.
4. **Update the 4 external files** (`app.hpp`, `node_gui.cpp`, `node_ops.cpp`, `scenes.hpp`):
   - `renderer.sceneGraph` → `scene.sceneGraph`
   - `renderer.assetManager` → `scene.assetManager`
   - `renderer.getMaterials()` → `scene.getMaterials()`
   - `renderer.activeCamera` → `scene.activeCamera`
   - etc.
5. `Renderer` holds `Scene& scene;`. The `addMeshToShader`/`removeMeshFromShader`/`renderEntries` list stays on `Renderer` — that's render-list state, not scene state.

**Exit criteria:** 107 call sites migrated. `Renderer` is now clearly "drawing logic + per-pass resources."

---

## Phase 4 — Clean up `Renderer`

What remains in `Renderer` should be just:
- `RenderFeatures features`
- Render-list state (`renderEntries`, `shaderDrawRanges`, `renderListDirty`)
- Pass-local resources (SSAO/SSR/HiZ/SDF texture indices, pipeline indices, MRT images, blur scratch, indirect draw buffers)
- `drawFrame`, `record*Pass`, `createOrResize*` methods
- References to `gpu`, `bindless`, `scene`

Delete the forwarding accessors added in Phase 1 (`getDevice`, `getResourceManager`, `getDescriptorSet`) — callers use `gpu`/`bindless` directly now.

**Exit criteria:** `renderer.hpp` shrinks from 327 → ~150 lines. No god class.

---

## Risks and ordering notes

- **Do phases in order.** Don't be tempted to start with Phase 3 — the external call-site churn is much worse if `BindlessSystem`/`GpuContext` aren't already settled, because `AssetManager::init` signature changes depend on them.
- **One phase per commit.** Each phase is a ~300-500 line diff; bisectable if something breaks.
- **The `PipelineManager` placement is debatable.** I put it in `BindlessSystem` because it's GPU-resource-ish, but if pipelines are created/owned per-renderer-pass it might belong in `Renderer`. Check `pipelines.hpp` before Phase 2 — if pipelines reference `DescriptorSet` layouts heavily, they belong with `BindlessSystem`.
- **`ShadowAtlas` is a judgment call** — it's scene-ish (driven by lights) but it's also a GPU render target. If it turns out to have pass-local-feeling state, leave it on `Renderer` instead of `Scene`.
- **Forward declarations matter.** `Renderer` currently forward-declares `Swapchain`/`PipelineManager` to avoid include bloat. `GpuContext`/`BindlessSystem`/`Scene` headers should do the same so `renderer.hpp` stays light.

---

## Estimate

- Phase 1: ~1-2 hours, self-contained, zero external churn.
- Phase 2: ~1-2 hours, self-contained.
- Phase 3: ~2-3 hours, 107 external edits (mostly mechanical).
- Phase 4: ~30 min cleanup.
