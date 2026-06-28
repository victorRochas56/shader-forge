#pragma once
#include <cstdint>

// Bundle of bindless slot indices the renderer owns and shares with subsystems
struct RenderBuffers {
    uint32_t modelMatrixBufferIndex = 0;
    uint32_t lightBufferIndex = 0;
    uint32_t volumeBufferIndex = 0;
    uint32_t nodeTextureIndex = 0;
};
