#pragma once
#include <cstdint>

// Bundle of bindless slot indices the renderer owns and shares with subsystems
// (SceneGraph today; node ops / scene loader candidates next). Renderer holds
// one by value; consumers hold a `const RenderBuffers&` so the renderer stays
// the single source of truth — no stale copies to invalidate if a buffer is
// ever recreated.
struct RenderBuffers {
    uint32_t modelMatrixBufferIndex = 0;
    uint32_t lightBufferIndex = 0;
    uint32_t volumeBufferIndex = 0;
    uint32_t nodeTextureIndex = 0;
};
