// Froxel volumetrics — Pass B bin: bucket each volumetric particle into the screen tiles its sphere
// overlaps (1 thread/particle). The gather pass then reads these per-tile lists. This is the cluster
// build of a tiled/clustered injection — it moves the heavy work off the per-particle thread so a
// screen-filling particle only touches a handful of tiles (<= tilesX*tilesY) instead of millions of
// froxels. Tile = FROXEL_TILE_SIZE x FROXEL_TILE_SIZE froxels in screen XY, spanning all Z.
import modules.common;   // Particle
import modules.froxel;   // froxelToWorld, viewZToSlice

static const uint EMITTER_FLAG_VOLUMETRIC = 1u << 2;
static const int  TILE_SIZE = 8; // must match FROXEL_TILE_SIZE in structs.hpp

// Only the emitter fields the bin/gather passes read (layout must match GPUParticleEmitter).
struct ParticleEmitter {
    float3 position; float emissionRate;
    float4 spawnRotation;
    float speedMin; float lifetimeMin; float speedMax; float lifetimeMax;
    float2 angularVelocityRandom; float spreadAngle; float drag;
    float2 densityRange;
    uint particleOffset; uint particleCapacity;
    uint textureIndex; uint numFrames; uint flags;
    float2 sizeRandom; float softRadius; float2 _pad;
};

struct FroxelBinPushConstants {
    Particle* particles;
    ParticleEmitter* emitters;
    uint* tileCounts;
    uint* tileList;
    uint particleCount;
    uint tilesX;
    uint tilesY;
    uint maxPerTile;
    uint3 dims;
    float nearZ;
    float3 cameraPos;
    float gridFar;
    float3 cameraForward;
    float billboardScale;
    float4x4 viewProj;
    float4x4 invViewProjection;
}

[[vk::push_constant]] ConstantBuffer<FroxelBinPushConstants> bpc;

[NumThreads(64, 1, 1)]
[shader("compute")]
void binMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= bpc.particleCount) return;
    Particle p = bpc.particles[tid.x];
    if (p.age < 0.0) return;                                   // dead slot
    ParticleEmitter e = bpc.emitters[p.emitterIdx];
    if ((e.flags & EMITTER_FLAG_VOLUMETRIC) == 0u) return;

    float radius = 0.5 * p.size * bpc.billboardScale;
    if (radius <= 0.0) return;

    float3 toCenter = p.position - bpc.cameraPos;
    float centerVz = dot(toCenter, bpc.cameraForward);
    if (centerVz + radius < bpc.nearZ) return;                  // fully behind the near plane
    if (centerVz - radius > bpc.gridFar) return;                // fully beyond the grid

    // Tile AABB the sphere overlaps.
    int tx0, tx1, ty0, ty1;
    if (centerVz > radius) {
        // Sphere in front: project center, then take a conservative screen extent from the froxel
        // world size at the nearest slice the sphere reaches (smallest froxels -> largest extent).
        float4 clip = mul(bpc.viewProj, float4(p.position, 1.0));
        int cx = int((clip.x / clip.w * 0.5 + 0.5) * float(bpc.dims.x));
        int cy = int((clip.y / clip.w * 0.5 + 0.5) * float(bpc.dims.y));
        int zn = clamp(int(viewZToSlice(clamp(centerVz - radius, bpc.nearZ, bpc.gridFar), bpc.dims.z, bpc.nearZ, bpc.gridFar)), 0, int(bpc.dims.z) - 1);
        uint3 nc = uint3(uint(clamp(cx, 0, int(bpc.dims.x) - 1)), uint(clamp(cy, 0, int(bpc.dims.y) - 1)), uint(zn));
        // No jitter: these measure grid geometry, so neighbours must share the same slice depth.
        float3 c0  = froxelToWorld(nc,                  bpc.dims, bpc.nearZ, bpc.gridFar, bpc.cameraPos, bpc.cameraForward, bpc.invViewProjection, 0.0);
        float3 cxw = froxelToWorld(nc + uint3(1, 0, 0), bpc.dims, bpc.nearZ, bpc.gridFar, bpc.cameraPos, bpc.cameraForward, bpc.invViewProjection, 0.0);
        float3 cyw = froxelToWorld(nc + uint3(0, 1, 0), bpc.dims, bpc.nearZ, bpc.gridFar, bpc.cameraPos, bpc.cameraForward, bpc.invViewProjection, 0.0);
        int xr = int(ceil(radius / max(length(cxw - c0), 1e-4))) + 1;   // +1 tile of margin
        int yr = int(ceil(radius / max(length(cyw - c0), 1e-4))) + 1;
        tx0 = max((cx - xr) / TILE_SIZE, 0);              tx1 = min((cx + xr) / TILE_SIZE, int(bpc.tilesX) - 1);
        ty0 = max((cy - yr) / TILE_SIZE, 0);              ty1 = min((cy + yr) / TILE_SIZE, int(bpc.tilesY) - 1);
    } else {
        // Camera inside / center near or behind: sphere can cover the whole screen -> all tiles.
        tx0 = 0; tx1 = int(bpc.tilesX) - 1;
        ty0 = 0; ty1 = int(bpc.tilesY) - 1;
    }

    for (int ty = ty0; ty <= ty1; ty++)
    for (int tx = tx0; tx <= tx1; tx++) {
        uint tileIdx = uint(ty) * bpc.tilesX + uint(tx);
        uint slot;
        InterlockedAdd(bpc.tileCounts[tileIdx], 1u, slot);
        if (slot < bpc.maxPerTile)
            bpc.tileList[tileIdx * bpc.maxPerTile + slot] = tid.x;   // store the pool index
    }
}
