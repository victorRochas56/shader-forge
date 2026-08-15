// Ambient-cube gather: for each occupied voxel, cone trace the radiance volume along the six axes
// and store per-face irradiance, so the lit shader reads GI with three trilinear taps instead of
// marching cones per pixel. Faces store (Reinhard(irradiance) * valid, valid).
//
// The irradiance grid may be coarser than the radiance grid (both cover the same world cube). The
// cone trace still runs at radiance resolution; only the gather positions and storage are coarse.
//
// Temporally amortized. Re-voxelizing every frame rebins triangles as the grid recentres, so a
// from-scratch gather flickers; instead each voxel blends its fresh trace into last frame's result
// by blendWeight, and only re-traces on the frame its phase comes up (other frames it carries
// history forward). The faces ping-pong — history is the set the lit pass read last frame, output
// is the other — so the recentre is an integer read offset rather than a copy pass.
import modules.voxel;

struct VoxelGatherPushConstants {
    float4x4 worldToGridClip;
    float4x4 gridClipToWorld;
    uint  radianceTextureIndex;
    uint  samplerIndex;
    uint  radianceResolution;   // cone-trace grid: voxel size and mip cap
    uint  irradianceResolution; // dispatch/storage grid; divides radianceResolution
    float worldExtent;
    uint  faceStorageIndices[6];   // VOXEL_FACE_DIRS order; written this frame
    uint  historyTextureIndices[6]; // last frame's faces, sampled slots
    int   historyOffset[3];        // history coord = id + offset; whole voxels, the grid snaps
    float blendWeight;             // weight of a fresh trace; 1 = replace, no usable history
    uint  phaseMask;               // updatePhases-1; 0 traces every voxel every frame
    uint  phase;                   // which phase updates this frame
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
void gatherMain(uint3 id : SV_DispatchThreadID, uint3 groupId : SV_GroupID) {
    if (any(id >= gpc.irradianceResolution)) return;

    int3 histCoord = int3(id) + int3(gpc.historyOffset[0], gpc.historyOffset[1], gpc.historyOffset[2]);
    bool hasHistory = gpc.blendWeight < 1.0 && all(histCoord >= 0) && all(histCoord < int3(gpc.irradianceResolution));

    // Phase is per workgroup, never per voxel: interleaving it inside the group puts every phase in
    // every wave, and one tracing lane costs the wave a full trace — the skipped lanes just idle
    // through it and the amortization saves nothing. Whole groups skipping is what actually skips
    // the work; the price is a coarser update pattern (4-voxel blocks, spaced by the phase count).
    // Voxels the grid just moved onto have nothing to carry, so they trace whatever the phase says.
    uint groupPhase = (groupId.x & 1) | ((groupId.y & 1) << 1) | ((groupId.z & 1) << 2);
    if (hasHistory && (groupPhase & gpc.phaseMask) != gpc.phase) {
        [unroll]
        for (uint f = 0; f < 6; f++)
            storageImages3D[gpc.faceStorageIndices[f]][int3(id)] = voxelLoadTexel(gpc.historyTextureIndices[f], uint3(histCoord), 0);
        return;
    }
    float weight = hasHistory ? gpc.blendWeight : 1.0;

    // One irradiance voxel covers ratio^3 radiance voxels; its occupancy lives ratio's mip up the
    // chain, where coverage is the partial-occupancy fraction — any nonzero child counts. Empty
    // voxels target zero and fade there over the same blend, rather than snapping.
    uint ratioMip = firstbithigh(gpc.radianceResolution / gpc.irradianceResolution);
    bool occupied = voxelLoadCoverage(gpc.radianceTextureIndex, id, ratioMip) > 0.0;

    float3 worldPos = voxelCenterWorld(id, gpc.irradianceResolution, gpc.gridClipToWorld);
    for (uint f = 0; f < 6; f++) {
        float4 target = float4(0);
        if (occupied) {
            float3 irr = coneTraceVXGI(gpc.radianceTextureIndex, gpc.samplerIndex, gpc.worldToGridClip,
                                       gpc.radianceResolution, gpc.worldExtent, worldPos, VOXEL_FACE_DIRS[f],
                                       gpc.sideCones, gpc.maxSteps, gpc.fetchBatch,
                                       gpc.skyEnvMapIndex, gpc.skyIntensity);
            target = float4(voxelToneMap(irr), 1.0);
        }
        // Premultiplied by `valid`, so the whole float4 blends linearly and occupancy fades with it.
        float4 hist = hasHistory ? voxelLoadTexel(gpc.historyTextureIndices[f], uint3(histCoord), 0) : float4(0);
        storageImages3D[gpc.faceStorageIndices[f]][int3(id)] = lerp(hist, target, weight);
    }
}
