// Ambient-cube gather: for each occupied voxel, cone trace the radiance volume along the six axes
// and store per-face irradiance, so the lit shader reads GI with three trilinear taps instead of
// marching cones per pixel. Faces store (Reinhard(irradiance) * valid, valid).
//
// One dispatch per clip level. The level's irradiance grid is coarser than its radiance grid (both
// cover the same cube); the cone trace still runs at radiance resolution, only the gather positions
// and storage are coarse. Its slab in the stacked face volumes is (id.xy, id.z + level*irrRes), and
// the same coordinate in the radiance volume's ratio mip — whose slabs are exactly irrRes deep — is
// where its occupancy lives.
//
// Temporally amortized. Re-voxelizing every frame rebins triangles as the level recentres, so a
// from-scratch gather flickers; instead each voxel blends its fresh trace into last frame's result
// by blendWeight, and only re-traces on the frame its phase comes up (other frames it carries
// history forward). The faces ping-pong — history is the set the lit pass read last frame, output
// is the other — so the recentre is an integer read offset rather than a copy pass.
//
// A voxel with no history of its own (the recentre just brought it into the level, or the level
// restarted) seeds from the next level out, which held the same spot last frame: it starts from the
// coarse value and converges to its own instead of stepping, so the lit pass's cross-level blend
// never sees a fresh voxel disagree with the coarse one it is fading out of.
import modules.voxel;

struct VoxelGatherPushConstants {
    VoxelClipmap* clipmap;         // this frame's levels
    uint  level;                   // clip level this dispatch gathers
    uint  radianceTextureIndex;
    uint  samplerIndex;
    uint  faceStorageIndices[6];   // VOXEL_FACE_DIRS order; written this frame
    uint  historyTextureIndices[6]; // last frame's faces, sampled slots
    int   historyOffset[3];        // history coord = id + offset; whole voxels, the level snaps
    float blendWeight;             // weight of a fresh trace against history or seed
    uint  phaseMask;               // updatePhases-1; 0 traces every voxel every frame
    uint  phase;                   // which phase updates this frame
    uint  sideCones;
    uint  maxSteps;
    uint  fetchBatch;
    uint  skyEnvMapIndex;
    float skyIntensity;     // cone-miss sky radiance scale; 0 disables
    float maxTraceDistance; // cone range cap (m)
    uint  historyValid;     // 1 when this level's own history slab reprojects (historyOffset is exact)
    float seedCenter[3];    // centre the next level out gathered its history under
    uint  seedFromParent;   // 1 when that slab is a valid gather at the current extent
}

[[vk::push_constant]] ConstantBuffer<VoxelGatherPushConstants> gpc;

[[vk::binding(4, 0)]]
RWTexture3D<float4> storageImages3D[];

[NumThreads(4, 4, 4)]
[shader("compute")]
void gatherMain(uint3 id : SV_DispatchThreadID, uint3 groupId : SV_GroupID) {
    VoxelClipmap cm = *gpc.clipmap;
    uint irrRes = cm.irradianceResolution;
    if (any(id >= irrRes)) return;
    uint3 dst = clipStackedCoord(id, gpc.level, irrRes);

    int3 histCoord = int3(id) + int3(gpc.historyOffset[0], gpc.historyOffset[1], gpc.historyOffset[2]);
    bool hasHistory = gpc.historyValid != 0 && all(histCoord >= 0) && all(histCoord < int3(irrRes));
    uint3 hist = clipStackedCoord(uint3(max(histCoord, int3(0))), gpc.level, irrRes);

    // Phase is per workgroup, never per voxel: interleaving it inside the group puts every phase in
    // every wave, and one tracing lane costs the wave a full trace — the skipped lanes just idle
    // through it and the amortization saves nothing. Whole groups skipping is what actually skips
    // the work; the price is a coarser update pattern (4-voxel blocks, spaced by the phase count).
    // Voxels the level just moved onto have nothing to carry, so they trace whatever the phase says.
    uint groupPhase = (groupId.x & 1) | ((groupId.y & 1) << 1) | ((groupId.z & 1) << 2);
    if (hasHistory && (groupPhase & gpc.phaseMask) != gpc.phase) {
        [unroll]
        for (uint f = 0; f < 6; f++)
            storageImages3D[gpc.faceStorageIndices[f]][int3(dst)] = voxelLoadTexel(gpc.historyTextureIndices[f], hist, 0);
        return;
    }

    float3 worldPos = clipVoxelCenterWorld(cm, gpc.level, irrRes, id);

    // No history of our own: seed from the next level out's history slab, sampled where this voxel
    // sits under the centre that slab was gathered around (it was not reprojected).
    bool seed = !hasHistory && gpc.seedFromParent != 0;
    float3 seedUVW = float3(0.0);
    if (seed) {
        uint parent = gpc.level + 1;
        float3 seedCenter = float3(gpc.seedCenter[0], gpc.seedCenter[1], gpc.seedCenter[2]);
        seedUVW = clipStackedUVW((worldPos - seedCenter) / cm.levels[parent].w + 0.5, parent, 0.5 / float(irrRes));
    }
    float weight = (hasHistory || seed) ? gpc.blendWeight : 1.0;

    // One irradiance voxel covers ratio^3 radiance voxels; its occupancy lives ratio's mip up the
    // chain, where coverage is the partial-occupancy fraction — any nonzero child counts. Empty
    // voxels target zero and fade there over the same blend, rather than snapping.
    uint ratioMip = firstbithigh(cm.resolution / irrRes);
    bool occupied = voxelLoadCoverage(gpc.radianceTextureIndex, dst, ratioMip) > 0.0;

    for (uint f = 0; f < 6; f++) {
        float4 target = float4(0);
        if (occupied) {
            float3 irr = coneTraceVXGI(gpc.radianceTextureIndex, gpc.samplerIndex, cm, gpc.level,
                                       worldPos, VOXEL_FACE_DIRS[f],
                                       gpc.sideCones, gpc.maxSteps, gpc.fetchBatch,
                                       gpc.skyEnvMapIndex, gpc.skyIntensity, gpc.maxTraceDistance);
            target = float4(voxelToneMap(irr), 1.0);
        }
        // Premultiplied by `valid`, so the whole float4 blends linearly and occupancy fades with it.
        float4 history = float4(0);
        if (hasHistory)  history = voxelLoadTexel(gpc.historyTextureIndices[f], hist, 0);
        else if (seed)   history = voxelSampleVolume(gpc.historyTextureIndices[f], gpc.samplerIndex, seedUVW);
        storageImages3D[gpc.faceStorageIndices[f]][int3(dst)] = lerp(history, target, weight);
    }
}
