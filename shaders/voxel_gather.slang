// Ambient-cube gather: for each occupied voxel, cone trace the radiance volume along the six axes
// and store per-face irradiance, so the lit shader reads GI with three trilinear taps instead of
// marching cones per pixel. Faces store (Reinhard(irradiance) * valid, valid); empty voxels write
// zeros, so the full-grid dispatch doubles as the per-frame clear.
//
// The irradiance grid may be coarser than the radiance grid (both cover the same world cube). The
// cone trace still runs at radiance resolution; only the gather positions and storage are coarse.
import modules.voxel;

struct VoxelGatherPushConstants {
    float4x4 worldToGridClip;
    float4x4 gridClipToWorld;
    uint  radianceTextureIndex;
    uint  samplerIndex;
    uint  radianceResolution;   // cone-trace grid: voxel size and mip cap
    uint  irradianceResolution; // dispatch/storage grid; divides radianceResolution
    float worldExtent;
    uint  faceStorageIndices[6]; // VOXEL_FACE_DIRS order
    uint  sideCones;
    uint  maxSteps;
    uint  fetchBatch;
    uint  skyEnvMapIndex;
    float skyIntensity; // cone-miss sky radiance scale; 0 disables
}

[[vk::push_constant]] ConstantBuffer<VoxelGatherPushConstants> gpc;

[[vk::binding(4, 0)]]
RWTexture3D<float4> storageImages3D[];

[NumThreads(4, 4, 4)]
[shader("compute")]
void gatherMain(uint3 id : SV_DispatchThreadID) {
    if (any(id >= gpc.irradianceResolution)) return;

    // One irradiance voxel covers ratio^3 radiance voxels; its occupancy lives ratio's mip up the
    // chain, where coverage is the partial-occupancy fraction — any nonzero child counts.
    uint ratioMip = firstbithigh(gpc.radianceResolution / gpc.irradianceResolution);
    if (voxelLoadCoverage(gpc.radianceTextureIndex, id, ratioMip) == 0.0) {
        [unroll]
        for (uint f = 0; f < 6; f++)
            storageImages3D[gpc.faceStorageIndices[f]][int3(id)] = float4(0);
        return;
    }

    float3 worldPos = voxelCenterWorld(id, gpc.irradianceResolution, gpc.gridClipToWorld);
    for (uint f = 0; f < 6; f++) {
        float3 irr = coneTraceVXGI(gpc.radianceTextureIndex, gpc.samplerIndex, gpc.worldToGridClip,
                                   gpc.radianceResolution, gpc.worldExtent, worldPos, VOXEL_FACE_DIRS[f],
                                   gpc.sideCones, gpc.maxSteps, gpc.fetchBatch,
                                   gpc.skyEnvMapIndex, gpc.skyIntensity);
        storageImages3D[gpc.faceStorageIndices[f]][int3(id)] = float4(voxelToneMap(irr), 1.0);
    }
}
