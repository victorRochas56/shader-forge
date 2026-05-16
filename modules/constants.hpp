#pragma once
#include <cstdlib>
#include <stdint.h>

//parameters for scene size limits this 
//TODO expand this into a settings menu / system and handle buffer resizing for limits

constexpr int MAX_FRAMES_IN_FLIGHT = 2;

constexpr uint32_t MAX_MESHES = 2048;
constexpr uint32_t MAX_LIGHTS = 2048;
constexpr uint32_t MAX_NODES = 32384; // node is currently ~304 bytes so this is <10mb of node data
constexpr uint32_t MAX_GIZMO_LINES = 32384; //100mb of gizmo buffer

constexpr uint32_t MAX_TEXTURES = 2048;
constexpr uint32_t MAX_CUBEMAPS = 2048;
constexpr uint32_t MAX_SAMPLERS = 2048;

constexpr uint32_t MAX_FIXED_BUFFER = 2048;
constexpr uint32_t MAX_VARIABLE_BUFFER = 2048;
constexpr uint32_t MAX_INDIRECT_COMMANDS = 10000; // per frame slot count for indirect draw buffers
constexpr uint32_t MAX_SHADOW_CASTERS = 8; // per-frame slot count for shadow indirect/draw-data buffers

constexpr uint32_t SHADOW_ATLAS_SIZE = 8192;
constexpr uint32_t SHADOW_ATLAS_MIN_TILE = 256;
constexpr uint32_t SHADOW_ATLAS_LEAVES_PER_SIDE = SHADOW_ATLAS_SIZE / SHADOW_ATLAS_MIN_TILE;
constexpr uint32_t SHADOW_ATLAS_LEAF_COUNT     = SHADOW_ATLAS_LEAVES_PER_SIDE * SHADOW_ATLAS_LEAVES_PER_SIDE;
constexpr uint32_t SHADOW_ATLAS_QUADTREE_COUNT   = (4u * SHADOW_ATLAS_LEAF_COUNT - 1u) / 3u;

constexpr uint32_t DEFAULT_CSM_SHADOW_RESOLUTION = 2048;
constexpr uint32_t DEFAULT_SHADOW_RESOLUTION = 512;
const float CASCADE_OVERLAP_FACTOR = 1.1f; 
const float FOV_EDGE_PADDING = 1.15f; 

constexpr bool ENABLE_SUBMESH_CULLING = true;
constexpr float CULLING_EPSILON = 1.0f;
