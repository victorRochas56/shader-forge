import modules.common;

struct ParticleEmitter {
    float3 position;
    float emissionRate;
    float4 spawnRotation;
    float speedMin;
    float lifetimeMin;
    float speedMax;
    float lifetimeMax;
    float2 angularVelocityRandom;
    float spreadAngle;
    float drag;
    float2 densityRange;

    uint particleOffset;
    uint particleCapacity;

    uint textureIndex;
    uint numFrames;
    uint flags;
    float2 sizeRandom; // per-particle size range (min, max)
    float softRadius;  // EMITTER_FLAG_SOFT depth-fade distance (unused here, keeps layout in sync)
    float2 _pad;
};

struct EmitterRuntime {
    uint ringHead;
    float spawnAccumulator;
    uint aliveCount;
    uint _pad;
};

struct ParticleComputePushConstants {
    EmitterRuntime* runtimes;
    ParticleEmitter* emitters;
    Particle* particles;
    uint emitterCount;
    uint particleCount;
    float dt;
};

[[vk::push_constant]] ConstantBuffer<ParticleComputePushConstants> particlePc;

static const float PI = 3.14159265;

// PCG hash + [0,1) RNG (same as modules/common.slang; inlined to keep this shader self-contained).
uint pcgHash(uint x) {
    uint state = x * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
float rand(inout uint seed) {
    seed = pcgHash(seed);
    return float(seed) * (1.0 / 4294967296.0);
}

// Rotate a vector by a quaternion (x,y,z,w).
float3 quatRotate(float4 q, float3 v) {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

// Spawn: one thread per emitter. Owns the serial ring-buffer bookkeeping (spawnAccumulator +
// ringHead) and writes newly spawned particles into its sub-range.
[NumThreads(64, 1, 1)]
[shader("compute")]
void spawnMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= particlePc.emitterCount) return;

    ParticleEmitter emitter = particlePc.emitters[tid.x];
    // Skip disabled (emissionRate = 0).
    if (emitter.emissionRate <= 0.0 || emitter.particleCapacity == 0) return;

    EmitterRuntime rt = particlePc.runtimes[tid.x];

    // Accumulate fractional particles across frames
    float acc = rt.spawnAccumulator + emitter.emissionRate * particlePc.dt;
    uint spawnCount = uint(acc);
    acc -= float(spawnCount);
    // Never overfill the ring in a single frame
    spawnCount = min(spawnCount, emitter.particleCapacity);

    uint base = emitter.particleOffset;
    float cosMax = cos(emitter.spreadAngle); // spreadAngle is the cone half-angle (radians)

    for (uint i = 0; i < spawnCount; i++) {
        uint slot = (rt.ringHead + i) % emitter.particleCapacity;
        uint idx = base + slot;

        // Per-particle RNG: mix the recycled slot's old seed with the emitter
        uint seed = particlePc.particles[idx].seed ^ pcgHash(tid.x * 32452843u + rt.ringHead + i + 1u);

        // Direction: uniform sample within a cone of half-angle spreadAngle around local +Z, then
        // rotate into world by the emitter's spawn rotation.
        float cosTheta = lerp(cosMax, 1.0, rand(seed));
        float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
        float phi = 2.0 * PI * rand(seed);
        float3 dirLocal = float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
        float3 dir = quatRotate(emitter.spawnRotation, dirLocal);

        // Speed: interpolate between the min/max velocity magnitudes.
        float speed = lerp(emitter.speedMin, emitter.speedMax, rand(seed));

        Particle p;
        p.position        = emitter.position;
        p.age             = 0.0;
        p.velocity        = dir * speed;
        p.lifespan        = lerp(emitter.lifetimeMin, emitter.lifetimeMax, rand(seed));
        p.rotation        = rand(seed) * 2.0 * PI;
        p.angularVelocity = lerp(emitter.angularVelocityRandom.x, emitter.angularVelocityRandom.y, rand(seed));
        p.size            = lerp(emitter.sizeRandom.x, emitter.sizeRandom.y, rand(seed));
        p.seed            = seed;
        p.emitterIdx      = tid.x;

        particlePc.particles[idx] = p;
    }

    // Commit ring state for next frame.
    particlePc.runtimes[tid.x].ringHead = (rt.ringHead + spawnCount) % emitter.particleCapacity;
    particlePc.runtimes[tid.x].spawnAccumulator = acc;
}

// Integrate: one thread per particle over the whole pool. Ages, applies drag/velocity, retires.
[NumThreads(64, 1, 1)]
[shader("compute")]
void integrateMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= particlePc.particleCount) return;
    Particle p = particlePc.particles[tid.x];
    ParticleEmitter em = particlePc.emitters[p.emitterIdx];

    if(p.age < 0.0) return;

    p.age += particlePc.dt;
    if (p.age >= p.lifespan) { 
        p.age = -1.0; 
        particlePc.particles[tid.x] = p; 
        return; 
    }

    p.velocity *= exp(-em.drag * particlePc.dt);
    p.position += p.velocity * particlePc.dt;
    p.rotation += p.angularVelocity * particlePc.dt;

    particlePc.particles[tid.x] = p;
}
