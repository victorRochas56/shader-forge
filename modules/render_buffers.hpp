#pragma once
#include <cstdint>

// Bundle of bindless slot indices the renderer owns and shares with subsystems
struct RenderBuffers {
    uint32_t modelMatrixBufferIndex = 0;
    uint32_t lightBufferIndex = 0;
    uint32_t volumeBufferIndex = 0;
    uint32_t nodeTextureIndex = 0;

    // Particle system (owned by ParticlePass). See modules/render_passes/particle_pass.hpp.
    uint32_t emitterBufferIndex = 0;      // per-frame GPUParticleEmitter descriptors
    uint32_t particlePoolBufferIndex = 0;  // shared Particle pool, range-allocated per emitter
    uint32_t emitterRuntimeBufferIndex = 0;// persistent GPU-owned EmitterRuntime state
};
