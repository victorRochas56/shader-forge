// Froxel volumetrics — density injection into the view-aligned media grid.
//   clearMain         (Pass A0): zero mediaVol.
//   injectVolumesMain (Pass A):  evaluate every analytic Volume at each froxel center and write
//                                (scattering.rgb, extinction) into mediaVol.
// mediaVol is an RGBA16F 3D storage image, bound as RWTexture3D at the bindless storage binding (4).
import modules.froxel;

// Layout matches GPUVolume (structs.hpp) and Volume (volumetrics.slang).
struct Volume {
    uint nodeIndex;
    float density;
    float phase;
    int shape;          // 0 = sphere, 1 = box
    float3 center;
    float radius;
    float3 dimensions;  // box half-extents
};

struct FroxelInjectPushConstants {
    Volume* vols;
    uint volumeCount;
    uint mediaVolIndex;
    uint3 dims;
    float nearZ;
    float3 cameraPos;
    float gridFar;
    float3 cameraForward;
    uint mediaPhaseIndex;
    float4x4 invViewProjection;
    uint frame;
}

[[vk::push_constant]] ConstantBuffer<FroxelInjectPushConstants> pc;

// Compute-writable froxel volumes. Must be in vk::ImageLayout::eGeneral while written.
[[vk::binding(4, 0)]]
RWTexture3D<float4> storageImages3D[];
// Single-channel storage (R32F) at the same bindless storage binding — the density-weighted phase.
[[vk::binding(4, 0)]]
RWTexture3D<float> storageImagesR[];

// Pass A0 — clear. One thread per froxel.
[NumThreads(4, 4, 4)]
[shader("compute")]
void clearMain(uint3 id : SV_DispatchThreadID) {
    if (any(id >= pc.dims)) return;
    storageImages3D[pc.mediaVolIndex][int3(id)] = float4(0, 0, 0, 0);
    storageImagesR[pc.mediaPhaseIndex][int3(id)] = 0.0;
}

// Pass A — analytic volume injection. One thread per froxel; evaluates every Volume at the froxel
// center. The chord math of the per-pixel march is replaced by a point-inside test.
[NumThreads(4, 4, 4)]
[shader("compute")]
void injectVolumesMain(uint3 id : SV_DispatchThreadID) {
    if (any(id >= pc.dims)) return;

    float3 wpos = froxelToWorld(id, pc.dims, pc.nearZ, pc.gridFar, pc.cameraPos, pc.cameraForward, pc.invViewProjection, froxelZJitter(id.xy));

    float3 scattering = float3(0);
    float  extinction = 0.0;
    float  phaseSum   = 0.0;   // density-weighted phase (Sum density_i * phase_i)
    for (uint i = 0; i < pc.volumeCount; i++) {
        Volume* v = &pc.vols[i];
        if (v.density <= 0.0) continue;

        bool inside;
        if (v.shape == 1) {                       // box: inside if within half-extents on every axis
            inside = all(abs(wpos - v.center) <= v.dimensions);
        } else {                                   // sphere
            float3 d = wpos - v.center;
            inside = dot(d, d) < v.radius * v.radius;
        }
        if (!inside) continue;

        extinction += v.density;
        // No per-volume albedo yet (plan: optional GPUVolume.albedo) — treat media as neutral so
        // scattering tracks density. Swap float3(1) for v.albedo.rgb once the field is added.
        scattering += v.density * float3(1.0);
        phaseSum   += v.density * v.phase;        // per-volume HG asymmetry (normalized in pass C)
    }

    storageImages3D[pc.mediaVolIndex][int3(id)] = float4(scattering, extinction);
    storageImagesR[pc.mediaPhaseIndex][int3(id)] = phaseSum;
}
