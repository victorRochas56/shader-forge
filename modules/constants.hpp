#pragma once
#include <cstdlib>
#include <stdint.h>

//parameters for scene size limits this 
//TODO expand this into a settings menu / system and handle buffer resizing for limits

constexpr int MAX_FRAMES_IN_FLIGHT = 2;

constexpr uint32_t MAX_MESHES = 2048;
constexpr uint32_t MAX_LIGHTS = 2048;
constexpr uint32_t MAX_NODES = 2048;
constexpr uint32_t MAX_GIZMO_LINES = 2048;

constexpr uint32_t MAX_TEXTURES = 2048;
constexpr uint32_t MAX_CUBEMAPS = 2048;
constexpr uint32_t MAX_SAMPLERS = 2048;

constexpr uint32_t MAX_FIXED_BUFFER = 2048;
constexpr uint32_t MAX_VARIABLE_BUFFER = 2048;

constexpr uint32_t DEFAULT_SHADOW_RESOLUTION = 1024;
const float CASCADE_OVERLAP_FACTOR = 1.1f; 
const float FOV_EDGE_PADDING = 1.15f; 

constexpr bool ENABLE_SUBMESH_CULLING = true;
constexpr float CULLING_EPSILON = 1.0f;
