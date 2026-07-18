// Froxel volumetrics — Pass B gather: one thread per froxel sums the density of the particles in its
// screen tile (built by froxel_particle_bin.slang) and adds it to mediaVol. Because each froxel owns
// its write, there's no atomic and no race, and per-thread work is bounded by the tile's particle
// count — a screen-filling particle costs each covered froxel one extra test, spread over millions of
// threads, instead of one thread scanning millions of froxels.
import modules.common;   // Particle, textures[], samplers[]
import modules.froxel;   // froxelToWorld
import modules.particles;

static const uint EMITTER_FLAG_VOLUMETRIC    = 1u << 2;
static const uint EMITTER_FLAG_VOLUME_SPHERE = 1u << 4;
static const int  TILE_SIZE = 8; // must match FROXEL_TILE_SIZE in structs.hpp

struct FroxelGatherPushConstants {
    Particle* particles;
    ParticleEmitter* emitters;
    uint* tileCounts;
    uint* tileList;
    uint mediaVolIndex;
    uint tilesX;
    uint tilesY;
    uint maxPerTile;
    uint samplerIndex;
    uint frame;
    uint mediaPhaseIndex;
    uint _padC;
    uint3 dims;
    float nearZ;
    float3 cameraPos;
    float gridFar;
    float3 cameraForward;
    float billboardScale;
    float3 cameraRight;
    float _padD;
    float3 cameraUp;
    float _padE;
    float4x4 invViewProjection;
}

[[vk::push_constant]] ConstantBuffer<FroxelGatherPushConstants> gpc;

[[vk::binding(4, 0)]]
RWTexture3D<float4> storageImages3D[];
// Same bindless storage binding, R32F view — density-weighted phase accumulator (matches froxel_inject).
[[vk::binding(4, 0)]]
RWTexture3D<float> storageImagesR[];

[NumThreads(4, 4, 4)]
[shader("compute")]
void gatherMain(uint3 id : SV_DispatchThreadID) {
    if (any(id >= gpc.dims)) return;

    uint tileIdx = (id.y / uint(TILE_SIZE)) * gpc.tilesX + (id.x / uint(TILE_SIZE));
    uint count = min(gpc.tileCounts[tileIdx], gpc.maxPerTile);
    if (count == 0u) return;                                    // no particles touch this froxel's tile

    float3 wpos = froxelToWorld(id, gpc.dims, gpc.nearZ, gpc.gridFar, gpc.cameraPos, gpc.cameraForward, gpc.invViewProjection, froxelZJitter(id.xy,gpc.frame));

    float accum = 0.0;
    float phaseAccum = 0.0;   // density-weighted HG phase (Sum contrib_i * phase_i), normalized in pass C
    for (uint i = 0; i < count; i++) {
        uint pIdx = gpc.tileList[tileIdx * gpc.maxPerTile + i];
        Particle p = gpc.particles[pIdx];
        if (p.age < 0.0) continue;

        float radius = 0.5 * p.size * gpc.billboardScale;
        if (radius <= 0.0) continue;
        float R2 = radius * radius;

        float3 offset = wpos - p.position;
        float d2 = dot(offset, offset);
        if (d2 >= R2) continue;                                 // this froxel is outside the sphere

        ParticleEmitter e = gpc.emitters[p.emitterIdx];
        float ageFrac = p.age / max(p.lifespan, 1e-4);
        float ageFade = 1.0 - smoothstep(0.8, 1.0, ageFrac);

        if ((e.flags & EMITTER_FLAG_VOLUME_SPHERE) != 0u) {
            // View-independent radial density (correct from any angle / inside).
            float tau = lerp(e.densityRange.x, e.densityRange.y, ageFade);
            float rho = 1.0 - d2 / R2;                          // parabolic soft sphere
            float contrib = tau * rho * (3.0 / (4.0 * radius));  // normalized: integrates to tau along center
            accum += contrib;
            phaseAccum += contrib * e.phase;
        } else {
            // View-aligned textured billboard: sample the sprite alpha in the camera-facing plane.
            float2 planar = float2(dot(offset, gpc.cameraRight), dot(offset, gpc.cameraUp));
            float s2 = dot(planar, planar);
            float halfChord = sqrt(max(R2 - s2, 1e-8));
            float sr = sin(p.rotation), cr = cos(p.rotation);
            float2 local = float2(planar.x * cr + planar.y * sr, -planar.x * sr + planar.y * cr);
            float2 tc = local / (2.0 * radius) + 0.5;
            if (e.numFrames > 0u) {
                uint frame = min(uint(ageFrac * float(e.numFrames)), e.numFrames - 1u);
                tc.x = (tc.x + float(frame)) / float(e.numFrames);
            }
            float texA = textures[e.textureIndex].SampleLevel(samplers[gpc.samplerIndex], tc, 0).a;
            float tau  = lerp(e.densityRange.x, e.densityRange.y, texA * ageFade);
            float contrib = tau / (2.0 * halfChord);            // uniform along the chord -> integrates to tau
            accum += contrib;
            phaseAccum += contrib * e.phase;
        }
    }

    if (accum > 0.0) {
        float4 cur = storageImages3D[gpc.mediaVolIndex][int3(id)];
        storageImages3D[gpc.mediaVolIndex][int3(id)] = cur + float4(accum, accum, accum, accum);
        // Add particle phase into the same density-weighted accumulator pass A seeded (pass C divides by media.a).
        float curPhase = storageImagesR[gpc.mediaPhaseIndex][int3(id)];
        storageImagesR[gpc.mediaPhaseIndex][int3(id)] = curPhase + phaseAccum;
    }
}
