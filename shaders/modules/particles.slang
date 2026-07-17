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
