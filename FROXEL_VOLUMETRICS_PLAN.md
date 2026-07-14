# Froxel Volumetrics — Implementation Plan

Goal: replace the current per-pixel analytic ray-march with a **frustum-aligned
3D "froxel" grid** as the single medium representation. Analytic `Volume` shapes
*and* volumetric particles both **inject density** into the grid; the grid is lit
and integrated once (independent of screen resolution and source count), and the
composite pass does a single depth-based sample. This unifies fog, analytic
volumes, and particles into one consistent, scalable path.

## Where we are today

- `shaders/volumetrics.slang` is a fullscreen **per-pixel march**: reconstruct a
  ray per pixel (`fragMain` lines 99-193), intersect analytic spheres
  (`rayIntersectSphere` 46-81), march with shadowed lights + HG `phase()` (93-97).
  Cost is `O(pixels × steps × volumes × lights)`.
- `VolumetricsPass` (`volumetrics_pass.hpp`) streams `GPUVolume`s into a per-frame
  fixed buffer (record 58-66), runs the march into a half-res 2D texture (68-84),
  blurs it (86), and `volumetrics_apply.slang` composites it (sample at line 29).
- `EMITTER_FLAG_VOLUMETRIC` exists but is **unused** — there is no density texture
  for particles to write into. This plan gives them (and the analytic volumes) one.

## What changes conceptually

The 2D screen-space march result becomes a **3D view-aligned volume** produced by a
chain of compute passes:

```
[clear] -> inject analytic volumes -> inject particles -> light scatter -> integrate -> composite sample
 (A0)          (A)                        (B)                 (C)             (D)            (E)
```

The per-pixel `rayIntersectSphere` march is **deleted**; its light/phase math moves
into the per-froxel lighting pass (C).

## Reused infrastructure (already in place)

| Need                    | Status |
| ----------------------- | ------ |
| Compute pipelines       | `createComputePipeline<T>` used by `ParticlePass` (`particle_pass.hpp:44-49`); dispatch+barrier pattern at `particle_pass.hpp:101-120`. |
| Storage images (2D)     | `eStorageImage` binding + `allocateStorageImage(view)` (`descriptor_sets.hpp:260-269, 375`). **Needs a 3D extension** — see Prereq 1. |
| 3D image creation       | `resources.hpp` `createTexture` already takes an `imageType`/`viewType` param (`:103, :223`) — pass `e3D`. |
| Volume streaming        | `GPUVolume` fixed-buffer + BDA (`volumetrics_pass.hpp:20, 58-66`). Keep as-is; it feeds pass A. |
| Shadow + phase          | `calculateShadow(...)` and `phase()` reused verbatim in pass C. |
| Particle pool by BDA    | `ParticleComputePushConstants` (pool/emitter BDA + high-water-mark count) — pass B reuses these. |

---

## The froxel grid

Two `RGBA16F` 3D textures (each allocated as **both** a storage image, for compute
writes, and a sampled image, for reads/composite):

| Texture        | After which pass | Contents (rgb, a) |
| -------------- | ---------------- | ----------------- |
| `mediaVol`     | A + B            | `(scattering.rgb, extinction)` — injected density. |
| `scatterVol`   | C                | `(inScatteredLight.rgb, extinction)` — lit, still per-froxel (not accumulated). |
| `integratedVol`| D                | `(accumInScatter.rgb, transmittance)` — front-to-back accumulated; **this is what E samples**. |

`scatterVol` can alias `mediaVol` (lighting in place) to save memory; keep separate
for v1 clarity. `integratedVol` must be its own texture (D reads C while writing).

**Dimensions:** start at `160 × 90 × 64` (configurable via `features.volumetrics`).
x/y map to screen tiles; z is distance along the view ray.

**Non-linear Z:** pack more slices near the camera. Exponential distribution between
`nearPlane` and a chosen `gridFar` (a new setting, distinct from today's `maxDist`):

```slang
// froxel.slang — shared mapping module (imported by every froxel pass + composite)
float sliceToViewZ(float slice /*0..D, can be fractional*/, uint D, float near, float gridFar) {
    return near * pow(gridFar / near, slice / float(D));       // exponential
}
float viewZToSlice(float viewZ, uint D, float near, float gridFar) {
    return float(D) * log(viewZ / near) / log(gridFar / near); // inverse of the above
}

// Froxel cell center -> world position, using the *center* view depth of the slab.
float3 froxelToWorld(uint3 coord, uint3 dims, float near, float gridFar, float4x4 invViewProj) {
    float2 uv  = (float2(coord.xy) + 0.5) / float2(dims.xy);
    float2 ndc = uv * 2.0 - 1.0;
    float  vz  = sliceToViewZ(float(coord.z) + 0.5, dims.z, near, gridFar);
    // Reverse-Z reconstruction, matching volumetrics.slang:112-116.
    // Cheapest: unproject the ray and step vz along it (see notes).
    ...
}
```

> Note: reconstructing world pos from a *linear view depth* via an unproject needs
> a ray + distance, not a single `invViewProj` mul (that expects an NDC z). The
> ray-and-step form mirrors `volumetrics.slang:112-116`; factor it into `froxel.slang`.

---

## Prerequisite 1 — 3D storage images

The bindless set already has an `eStorageImage` binding and `allocateStorageImage`
(`descriptor_sets.hpp:375`), but only 2D images are created today. Add a helper that
creates a **3D** image (`vk::ImageType::e3D`) + 3D view (`vk::ImageViewType::e3D`)
with usage `eStorage | eSampled`, then registers **two** bindless indices for it:
one storage-image slot (compute write) and one sampled slot (reads/composite).
Slang side uses `RWTexture3D<float4>` for writes and `Texture3D<float4>` for reads —
both ride the existing bindings, no new descriptor types.

---

## Pass A0 — clear (compute, 1 thread/froxel)

`[numthreads(4,4,4)]`, dispatch `ceil(W/4, H/4, D/4)`. Zero `mediaVol`. (Phase 2:
replace with temporal reprojection of last frame's `integratedVol` — see below.)

## Pass A — analytic volume injection (compute, 1 thread/froxel)

Each thread owns one froxel, evaluates every `Volume` at the froxel center, and
accumulates. This is where `rayIntersectSphere` is **replaced by a point-inside
test** — far simpler than the current chord math:

```slang
[numthreads(4,4,4)]
[shader("compute")]
void injectVolumesMain(uint3 id : SV_DispatchThreadID) {
    if (any(id >= pc.dims)) return;
    float3 wpos = froxelToWorld(id, pc.dims, pc.near, pc.gridFar, pc.invViewProj);

    float3 scattering = 0; float extinction = 0;
    for (uint i = 0; i < pc.volumeCount; i++) {
        Volume* v = &pc.vols[i];
        if (v.density <= 0.0) continue;
        if (dot(wpos - v.center, wpos - v.center) < v.radius * v.radius) { // sphere; box analogous
            extinction += v.density;
            scattering += v.density * v.albedo.rgb;   // needs an albedo/color on GPUVolume
        }
    }
    pc.mediaVol[id] = float4(scattering, extinction);
}
```

`O(froxels × volumes)` — trivial: ~920k froxels × a few volumes.

## Pass B — particle injection (compute, 1 thread/particle, **scatter**)

Only particles whose emitter has `EMITTER_FLAG_VOLUMETRIC` participate.
Each thread reads a particle, finds its froxel, and **additively** writes density.
Reuse the sim's pool BDA + high-water-mark count so dead slots are skipped cheaply.

```slang
[numthreads(64,1,1)]
[shader("compute")]
void injectParticlesMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.particleCount) return;
    Particle p = pc.particles[tid.x];
    if (p.age < 0.0) return;
    ParticleEmitter e = pc.emitters[p.emitterIdx];
    if ((e.flags & EMITTER_FLAG_VOLUMETRIC) == 0u) return;

    // world -> view depth -> froxel coord
    float4 clip = mul(pc.viewProj, float4(p.position, 1.0));
    if (clip.w <= 0.0) return;
    float2 uv    = (clip.xy / clip.w) * 0.5 + 0.5;
    uint3  coord = uint3(uv * float2(pc.dims.xy),
                         viewZToSlice(clip.w, pc.dims.z, pc.near, pc.gridFar));
    if (any(coord >= pc.dims)) return;

    float density = lerp(e.densityRange.x, e.densityRange.y, /*per-particle*/ ...);
    // ATOMIC: multiple particles hit one froxel — see decision below.
    imageAtomicAdd(pc.densityAtomicVol, coord, density);   // requires an R32 atomic target
}
```

**The one real design decision — additive overlap.** `RGBA16F` has no atomic add,
so overlapping particles racing on `mediaVol` would lose contributions. Options:

- **(rec) Separate `R32UI`/`R32F` atomic density volume**, `imageAtomicAdd` here,
  then a tiny merge pass folds it into `mediaVol.a`. Correct, cheap, one extra texture.
- **Rasterize particles as point sprites into volume slices** with additive blend
  (geometry approach) — heavier, avoids compute atomics.
- **Accept races for v1** (last write wins) — fine to prove the pipeline, wrong for
  dense smoke. `log()` this limitation so it isn't mistaken for done.

Per-particle density source: `emitter.densityRange` (already exists). For colored
smoke, add a scattering color to the emitter and inject `density * color`.

## Pass C — light scattering (compute, 1 thread/froxel)

Read `mediaVol`, run the **exact** light loop from today's `volumetrics.slang`
(`fragMain` 158-186) but once per froxel instead of per march step: for each light,
`calculateShadow(...)`, HG `phase()`, accumulate in-scattered radiance. Write
`(inScatter, extinction)` to `scatterVol`.

```slang
float3 wpos = froxelToWorld(id, ...);
float4 media = pc.mediaVol[id];             // (scattering, extinction)
float3 inscatter = 0;
for (uint li = 0; li < pc.lightCount; li++) { /* shadow + phase, reuse existing math */ }
pc.scatterVol[id] = float4(inscatter * media.rgb, media.a);
```

## Pass D — integration (compute, 1 thread per x,y **column**)

`[numthreads(8,8,1)]`, dispatch `ceil(W/8, H/8)`. Each thread marches z front-to-back
accumulating scattering + transmittance, writing per slice:

```slang
float3 accum = 0; float transmittance = 1.0;
for (uint z = 0; z < pc.dims.z; z++) {
    float4 s = pc.scatterVol[uint3(id.xy, z)];             // (inScatter, extinction)
    float  thickness = sliceToViewZ(z+1,...) - sliceToViewZ(z,...);
    float  t = exp(-s.a * thickness);
    // energy-conserving analytic integration over the slab:
    float3 sIntegrated = (s.rgb - s.rgb * t) / max(s.a, 1e-5);
    accum += transmittance * sIntegrated;
    transmittance *= t;
    pc.integratedVol[uint3(id.xy, z)] = float4(accum, transmittance);
}
```

## Pass E — composite (fullscreen, replaces `volumetrics_apply.slang`)

Per pixel: linearize scene depth → view Z → fractional slice, trilinearly sample
`integratedVol`, blend. Replaces the flat 2D sample at `volumetrics_apply.slang:29`.

```slang
float sceneDepth = textures[pc.depthTex].Sample(...).r;
float viewZ      = linearize(sceneDepth, pc.near, pc.far);          // formula from image_view.slang:81
float slice      = viewZToSlice(min(viewZ, pc.gridFar), pc.dims.z, pc.near, pc.gridFar);
float4 v = textures[pc.integratedVol].Sample(sampler, float3(input.texCoord, slice / pc.dims.z));
// v = (inScatter, transmittance)
return float4(v.rgb, 1.0);      // blend: dst*transmittance + inScatter (see blend-state note)
```

Pixels past `gridFar` clamp to the last slice (fully integrated). Blend must apply
`sceneColor * transmittance + inScatter` — either a dual-source/custom blend, or
sample the scene color in-shader; today's pass is `POSTPROCESS_ALPHA_BLEND`, so plan
the blend state accordingly (transmittance in alpha, inScatter in rgb).

---

## `VolumetricsPass::record` — orchestration

Replace the single `drawFullscreenPass` march (`volumetrics_pass.hpp:68-84`) with the
dispatch chain, **each separated by a compute→compute image barrier**
(`eShaderWrite → eShaderRead`, `General` layout), per COMPUTE_PIPELINE_PLAN gotcha #2:

```
stream GPUVolumes (unchanged, 58-66)
A0 clear            -> barrier
A  inject volumes   -> barrier
B  inject particles -> (merge atomic vol) -> barrier
C  light scatter    -> barrier
D  integrate        -> barrier (compute -> fragment; transition integratedVol to ShaderReadOnly)
E  composite (fullscreen, onto compositeColorTextureIndex)
```

Drop `blurAttachment` (86) — the grid + temporal jitter replace the blur.

## To delete / rework

- `volumetrics.slang`: remove `rayIntersectSphere` + the march `fragMain`. Keep
  `phase()`; move the light loop into pass C.
- `volumetrics_apply.slang`: rework to the depth→slice sample above.
- `VolumetricsPass`: swap the 2D march texture for the 3D volumes; swap the two
  graphics pipelines for 4-5 compute pipelines + the composite pipeline.

## Data-structure changes

- **`GPUVolume`**: add `float4 albedo`/scattering color if you want colored media
  (injection currently only has scalar `density`).
- **Emitter**: reuse `densityRange` for per-particle density; optionally add a
  scattering color for colored smoke.
- **`features.volumetrics`**: add `froxelDims (uvec3)`, `gridFar`. Keep `numSteps`
  only if you still want it as the integration slice count (else drop).
- **Push constants**: each compute pass needs `dims`, `near`, `gridFar`,
  `invViewProj`/`viewProj`, `cameraPos`, storage-image indices, and its data BDAs
  (volumes for A; pool+emitter BDA + count for B; lights + shadowAtlas for C).

## Temporal reprojection + jitter (Phase 2, recommended)

The grid is low-res, so per-frame **jitter** the froxel sample depth and **blend**
this frame's `integratedVol` (or `mediaVol`) with last frame's reprojected by the
previous `viewProj`. This is what hides froxel blockiness in Killzone/AC4. Needs one
history texture + the previous-frame VP (the camera already tracks
`prevViewProjection`).

## Suggested order

1. **Prereq 1** (3D storage images) + allocate the volumes; composite samples a
   constant — proves the 3D plumbing end to end with no math.
2. **A + D + E** with density-only (no lighting): analytic fog appears as gray
   in-scatter. Proves grid → integrate → composite.
3. **C** — light scattering (reuse existing shadow/phase). Now it matches today's look.
4. **B** — particle injection (start with the race-accepting v1, then the atomic
   density volume).
5. **Phase 2** — temporal reprojection + jitter.

## Open decisions

- Particle overlap: atomic density volume vs. rasterized splat vs. accept-races-v1.
- `gridFar` vs. today's `maxDist` (volumes beyond the grid won't render).
- Colored scattering (per-volume / per-emitter albedo) — worth it now or later?
- Froxel resolution vs. perf — profile pass C (lighting) first; it's the heaviest.
