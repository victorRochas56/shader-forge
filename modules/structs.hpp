#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <unordered_set>

#include "constants.hpp"

#define GLM_FORCE_RADIANS
#define GLM_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>


enum MaterialFlags : uint32_t
{
    MAT_NONE =      0,
    FLIP_NORMAL =   1 << 0,
    HAS_ALBEDO  =   1 << 1,
    HAS_ROUGHNESS = 1 << 2,
    HAS_METALLIC =  1 << 3,
    HAS_NORMAL =    1 << 4,
    ALPHA_CLIP =    1 << 5
};

/*
data only structs for push constants and various scene elements
*/
struct PushConstants {
    uint32_t vertexAllocationOffset; // Allocation handle (byte offset key) into the vertex variable buffer
    uint32_t vertexOffset;           // Byte offset in vertex buffer
    uint32_t vertexStride;          // Size of each vertex (e.g., sizeof(Vertex))
    uint32_t modelMatrixIndex;      // Index into model matrices

    uint32_t albedoTextureIndex;    // Index into textures
    uint32_t roughnessTextureIndex;
    uint32_t metallicTextureIndex;
    uint32_t normalTextureIndex;
    
    uint32_t environmentMapIndex;
    uint32_t samplerIndex; // Index into samplers
    uint32_t lightCount;
    uint32_t shadowSamplerIndex;
    
    glm::vec3 cameraPosition;
    MaterialFlags materialFlags;

    uint32_t elementOffsetModel;  // Element offset for this frame's model matrices
    uint32_t elementOffsetLight;  // Element offset for this frame's lights
    float metallic;
    float roughness;

    glm::mat4 viewProjection;
};

struct LinePushConstants {
    uint64_t lineVertsAddress;
    uint32_t depthTextureIndex;
    uint32_t depthSamplerIndex;
    glm::mat4 viewProjection;
};

struct SkyBoxPushConstants {
    uint32_t skyboxIndex;
    float blur;
    float intensity; // radiance multiplier on the cubemap sample
    uint32_t padding3;
    glm::mat4 invViewProjMatrix;
};

// Mirror of VoxelizationPushConstants in shaders/voxelization.slang.
// 200 bytes — above the 128-byte Vulkan floor, so this needs maxPushConstantsSize >= 256 (NVIDIA
// has it, most AMD/Intel report 128). To get back under, fold vpm*model into one mvp on the CPU.
struct VoxelizationPushConstants {
    glm::mat4 vpm;                  // orthographic voxel-grid view-projection
    glm::mat4 model;                // node world transform
    uint64_t  vertexBufferAddress;
    uint64_t  voxelAlbedoAddress;   // uint[res^3], luma-keyed packed RGB, InterlockedMax
    uint64_t  voxelRadianceAddress; // uint[res^3], same encoding
    uint64_t  lightBufferAddress;
    uint32_t  lightCount;
    uint32_t  shadowAtlasIndex;
    uint32_t  vertexStride;
    uint32_t  vertexOffset;         // byte offset of this mesh in the shared vertex buffer
    uint32_t  albedoTextureIndex;   // node material's albedo, bindless sampled slot
    uint32_t  samplerIndex;
    uint32_t  voxelResolution;      // cubic grid side; index = x + res*(y + res*z)
    uint32_t  skyEnvMapIndex;
    float     skyInjection;         // sky irradiance scale; 0 disables the injection
    uint32_t  packedColor;          // material base colour, RGBA8 linear
};
static_assert(sizeof(VoxelizationPushConstants) <= 256, "voxelization push constants exceed the 256-byte device limit");

// Mirror of VoxelResolvePushConstants in shaders/voxel_resolve.slang. Shared by both entry points:
// resolveMain unpacks the atomic buffers into mip 0, downsampleMain folds src -> dst one level down.
// Both mips are addressed as storage slots rather than sampled — the 2x2x2 fold wants exact texel
// fetches, and it sidesteps needing the image in a sampleable layout while it's being written.
struct VoxelResolvePushConstants {
    uint64_t voxelAlbedoAddress;   // unused by downsampleMain
    uint64_t voxelRadianceAddress; // unused by downsampleMain
    uint32_t dstStorageIndex;      // RWTexture3D<float4> slot being written
    uint32_t srcStorageIndex;      // RWTexture3D<float4> slot being read (unused by resolveMain)
    uint32_t dstResolution;        // side length of the destination mip
    uint32_t voxelResolution;      // mip-0 side length, for indexing the atomic buffers
};

// Keep in sync with the VOXDBG_* constants in shaders/voxel_debug.slang.
enum class VoxelDebugMode : uint32_t { Radiance = 0, Occupancy = 1, Albedo = 2, Depth = 3 };

struct VoxelDebugSettings {
    // Kill switch for the whole per-frame voxel build (clear/raster/resolve/mips), independent of the
    // debug views. For bisecting frame cost: if fps doesn't recover with this off, the cost isn't voxel code.
    bool           voxelizeScene = true;
    bool           enabled  = false;
    bool           drawCubes = false; // cubes via voxel_cubes.slang instead of the fullscreen ray march
    VoxelDebugMode mode     = VoxelDebugMode::Radiance; // ray-march only; cubes always show radiance
    int            volumeSelect = 0;  // 0 = radiance volume, 1..6 = irradiance face (+X,-X,+Y,-Y,+Z,-Z)
    uint32_t       mipLevel = 0;      // which level of the chain to visualize — good for eyeballing the fold
    float          alphaScale = 1.0f; // ray-march: raise to make a sparse grid readable
    float          cubeThreshold = 0.05f; // cubes: min coverage for a voxel to get a cube
    // Runaway guard for grazing rays. A full diagonal of a 128^3 grid is ~222 steps, so 256 covers the
    // worst honest case; this is a full-res per-pixel march, so it is the most expensive thing here.
    uint32_t       maxSteps = 256;
};

// Runtime VXGI cone-trace tuning, copied into LitPassData each frame. Ints for GUI sliders;
// caps live in shaders/modules/voxel.slang (MAX_HEMISPHERE_RAYS, MAX_FETCH_BATCH).
struct VXGISettings {
    int hemisphereRays = 5; // side cones at 60° (0..5); the normal cone always runs
    int maxSteps = 8;       // taps per cone
    int fetchBatch = 4;     // taps issued per batch
    int mode = 1;           // 0 = per-pixel cone trace, 1 = ambient-cube lookup
    float strength = 1.0f;  // scales the GI term in both modes; 1 = untouched
    // Sky seen by GI, relative to skyboxIntensity. The two paths are separate knobs because they
    // behave differently: the cone-miss term is occlusion-aware (only cones that leave the grid
    // collect it), while injection assumes visibility 1 and so adds a flat, occlusion-blind ambient
    // to every voxelized surface. Driving both from one slider makes sky look uniform — the flat
    // term rises with the directional one and swamps the contrast.
    float skyStrength = 0.5f;    // cone-miss sky: direct sky visibility, occlusion-aware
    float skyInjection = 0.1f;   // sky injected into voxel radiance: buys sky bounce, flat
    // Gather-pass tuning. Runs per occupied voxel, so it can afford more steps than the per-pixel path.
    int gatherSideCones = 5;
    int gatherSteps = 24;
    int gatherFetchBatch = 3;
    // Temporal amortization of the gather. blend is the weight a fresh trace gets against the
    // reprojected history — the flicker fix, since re-voxelizing rebins triangles every frame.
    // Below ~0.05 the unorm8 faces quantize the increment to nothing and convergence stalls.
    float temporalBlend = 0.25f; // 1 = no history
    int updatePhases = 2;        // power of two, 1..8: a voxel re-traces every N frames
};

// Mirror of VoxelGatherPushConstants in shaders/voxel_gather.slang.
struct VoxelGatherPushConstants {
    glm::mat4 worldToGridClip;
    glm::mat4 gridClipToWorld;
    uint32_t radianceTextureIndex;
    uint32_t samplerIndex;
    uint32_t radianceResolution;   // cone-trace grid: voxel size and mip cap
    uint32_t irradianceResolution; // dispatch/storage grid; divides radianceResolution
    float    worldExtent;
    std::array<uint32_t, 6> faceStorageIndices;   // VOXEL_FACE_DIRS order: +X,-X,+Y,-Y,+Z,-Z
    std::array<uint32_t, 6> historyTextureIndices; // last frame's faces, sampled slots
    std::array<int32_t, 3>  historyOffset;         // history coord = id + offset (grid recentre)
    float    blendWeight;  // weight of a fresh trace; 1 = replace, no usable history
    uint32_t phaseMask;    // updatePhases-1; 0 traces every voxel every frame
    uint32_t phase;        // which phase updates this frame
    uint32_t sideCones;
    uint32_t maxSteps;
    uint32_t fetchBatch;
    uint32_t skyEnvMapIndex;
    float    skyIntensity; // cone-miss sky radiance scale; 0 disables
};
// Scalar-packed to match Slang; already past the 128-byte spec floor, so it needs a device that
// reports 256 (desktop NVIDIA/Intel do; some AMD drivers cap at 128).
static_assert(sizeof(VoxelGatherPushConstants) == 240, "gather push constants must stay under 256 bytes");

// Mirror of VoxelCubePushConstants in shaders/voxel_cubes.slang. Shared by extractMain and the cube
// draw — one struct because two push-constant blocks in one module would collide.
struct VoxelCubePushConstants {
    glm::mat4 gridToClip;            // cameraVP * voxelCamInvVPM
    uint64_t  instanceBufferAddress;
    uint64_t  indirectBufferAddress;
    uint32_t  volumeTexIndex;
    uint32_t  mipLevel;
    uint32_t  mipRes;
    float     threshold;
};

// Mirror of VoxelDebugPushConstants in shaders/voxel_debug.slang. 112 bytes.
struct VoxelDebugPushConstants {
    glm::mat4 camNdcToGrid;  // voxelVPM * inverse(cameraViewProjection)
    glm::vec3 cameraPosGrid; // camera position in grid UVW
    uint32_t  mipLevel;
    uint32_t  volumeTexIndex;
    uint32_t  samplerIndex;
    uint32_t  depthTexIndex;
    uint32_t  depthSamplerIndex;
    uint32_t  resolution;
    uint32_t  mode;
    uint32_t  maxSteps;
    float     alphaScale;
};

enum ImageVisFlags : uint32_t 
{
    IMAGE_VIS_NONE =    0,
    B_W_IMAGE =         1 << 0,
    FLIP_VERTICAL =     1 << 1,
    LINEARIZE =         1 << 2,
    // Slicemap targets are R32_UINT bitmasks, so they're read through the uint alias of the texture
    // binding and mapped to grey: occupancy count by default, one slice when SLICEMAP_SLICE is also set.
    SLICEMAP =          1 << 3,
    SLICEMAP_SLICE =    1 << 4,
};

inline ImageVisFlags operator|(ImageVisFlags a, ImageVisFlags b) {
    return static_cast<ImageVisFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline ImageVisFlags& operator|=(ImageVisFlags& a, ImageVisFlags b) {
    return a = a | b;
}
inline ImageVisFlags operator&(ImageVisFlags a, ImageVisFlags b) {
    return static_cast<ImageVisFlags>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
inline ImageVisFlags& operator&=(ImageVisFlags& a, ImageVisFlags b) {
    return a = a & b;
}
inline ImageVisFlags operator^(ImageVisFlags a, ImageVisFlags b) {
    return static_cast<ImageVisFlags>(static_cast<uint32_t>(a) ^ static_cast<uint32_t>(b));
}
inline ImageVisFlags& operator^=(ImageVisFlags& a, ImageVisFlags b) {
    return a = a ^ b;
}

struct ImageVisPushConstants {
    uint32_t imageIndex;
    uint32_t samplerIndex;
    ImageVisFlags flags;
    float nearPlane;
    float farPlane;
    float imageAspect;
    float screenAspect;
    int mipLevel;
};

struct ImageVisSettings {
    uint32_t imageIndex = 0xFFFFFFFF;
    ImageVisFlags flags = ImageVisFlags::IMAGE_VIS_NONE;
    int mipLevel = 0;
};

struct SSAOSettings {
    bool enabled = true;
    float radius = 0.3f;
    float bias = 0.1f;
    float power = 2.0f;
    float resolutionScale = 0.5f;
};

struct SSRSettings {
    bool enabled = true;
    float resolutionScale = 1.0f;
    float roughnessThreshold = 0.8f;
    int maxSteps = 32;
    float thickness = 0.1f;
    float temporalBlend = 0.2;
    bool resolutionDirty = false;
    // Tiled trace: compute classification compacts trace-worthy pixels, then a dispatch-indirect
    // trace runs with full warps. Off = legacy fullscreen fragment path.
    bool tiledTrace = true;
    int hiZStartLevel = 2; // lower = cheaper nearby hits, more ascent for far ones
    int hiZMaxLevel = 6;   // coarser than ~6 the max-reduced plane rarely lets the ray advance
};

struct VolumetricSettings {
    bool enabled = true;
    float resolutionScale = 0.8f;
    float blurRadius = 2.0f;
    int numSteps = 16;
    float maxDist = 35.0f;
    // Froxel grid (FROXEL_VOLUMETRICS_PLAN.md). Screen-independent 3D media grid: x/y tiles, z
    // exponential slices between the near plane and gridFar. gridFar is distinct from maxDist.
    glm::uvec3 froxelDims = glm::uvec3(240, 135, 128);
    float gridFar = 50.0f;
    // Froxel debug overlay: 0 = off, else visualizes a grid volume (see FroxelDebugMode in
    // volumetrics_apply.slang). Per-volume scattering phase comes from each Volume's own `phase`.
    int debugView = 0;
};

struct TonemapSettings {
    float ev100 = 6.0f;        // manual exposure; higher = darker (used when autoExposure off)
    uint32_t op = 0;           // 0 = Reinhard, 1 = ACES, 2 = none (clamp)
    bool autoExposure = true; // meter scene average luminance and expose automatically
    float exposureComp = 0.0f; // EV bias on auto exposure; + = brighter
    float minEV = -1.0f;       // clamp the auto-metered EV100
    float maxEV = 3.0f;
    float adaptationSpeed = 2.5f; // eye-adaptation rate (1/s); higher = snappier
};

// EV100-based exposure factor (Lagarde/Frostbite). Multiply scene radiance by this.
inline float computeExposure(const TonemapSettings& t) {
    return 1.0f / (1.2f * std::exp2(t.ev100));
}

struct RenderFeatures {
    ImageVisSettings imageVis;
    SSAOSettings ssao;
    SSRSettings ssr;
    VolumetricSettings volumetrics;
    TonemapSettings tonemap;
    VoxelDebugSettings voxelDebug;
    VXGISettings vxgi;
    float skyboxIntensity = 5.0f; // radiance multiplier on the skybox draw
    bool showGizmos = true;
    bool showBBoxes = false;
    // Set by the Materials window each frame; MaterialThumbnailPass skips itself when it's false.
    bool materialPreviewsVisible = false;
};

struct TonemapPushConstants {
    uint32_t hdrTextureIndex;
    uint32_t samplerIndex;
    float exposure;          // manual exposure factor (used when autoExposure == 0)
    uint32_t op;
    uint32_t autoExposure;   // 0/1
    uint32_t lumTextureIndex; // metering source (mipped); smallest mip ≈ scene average
    uint32_t lumMipLevel;
    float exposureComp;
    float minEV;
    float maxEV;
};

struct LumExtractPushConstants {
    uint32_t inputTextureIndex; // scene color to meter (colorResolve)
    uint32_t samplerIndex;
    uint32_t padding[2];
};

struct ExposureAdaptPushConstants {
    uint32_t currentLumIndex;   // mipped log-luminance target
    uint32_t currentLumMip;     // smallest mip = avg log-luminance
    uint32_t prevAdaptedIndex;  // previous frame's adapted luminance (1x1)
    uint32_t samplerIndex;
    float dt;
    float speed;
    uint32_t initialized;       // 0 on first frame -> snap instead of lerp
    uint32_t padding;
};

struct ShadowPushConstants {
    uint64_t positionBufferAddress;         // 8
    uint64_t shadowInstanceDataAddress;     // 8  (pre-offset to current face's block)
    uint64_t shadowMeshDrawDataAddress;     // 8  (pre-offset to current light slot)
};  // Total: 24 bytes

struct ShadowMeshDrawData {
    uint32_t positionBufferOffset;
    uint32_t positionBufferStride;
    uint32_t firstInstance; // matches DrawIndexedIndirectCommand.firstInstance; shader uses mesh.firstInstance + SV_InstanceID to index ShadowInstanceData
    uint32_t _pad1;
}; // 16 bytes

struct ShadowInstanceData {
    glm::mat4 MMxLSM; // lightSpaceMatrix * worldTransform, baked per face per instance
}; // 64 bytes

struct BlurPushConstants {
    uint32_t inputTextureIndex;
    uint32_t samplerIndex;
    int32_t isHorizontal;
    float blurRadius;
    glm::uvec2 resolution;
    uint32_t mipLevel;
    uint32_t padding1;
};

struct SSAOPushConstants {
    glm::mat4 invProjection;    // 64 bytes
    uint32_t depthIndex;        // 4
    uint32_t depthSamplerIndex; // 4
    uint32_t noiseIndex;        // 4
    uint32_t noiseSamplerIndex; // 4
    glm::uvec2 resolution;     // 8
    float radius;               // 4
    float bias;                 // 4
    float power;                // 4
    uint32_t kernelSize;        // 4
    // Total: 100 bytes — fits in 128
};

struct SSRPushConstants {
    uint64_t ssrPassDataAddress;
    uint64_t rayHeaderAddress = 0; // SSRRayHeader (dispatch args + count); tiled path only
    uint64_t rayPixelsAddress = 0; // packed pixel list, rayHeaderAddress + 16
};

struct SSRPassData {
    glm::mat4 invViewProj;
    glm::mat4 viewProj;
    glm::vec3 cameraPos;
    uint32_t depthIndex;
    uint32_t depthSamplerIndex;
    uint32_t colorIndex;
    uint32_t colorSamplerIndex;
    uint32_t roughnessMetalIndex;
    uint32_t roughnessMetalSamplerIndex;
    uint32_t normalIndex;
    uint32_t normalSamplerIndex;
    uint32_t _pad_resolution;   // align uvec2 to 8 bytes (std430)
    glm::uvec2 resolution;
    uint32_t hiZIndex;      // Hi-Z pyramid texture index
    uint32_t hiZMipLevels;  // number of mip levels in Hi-Z pyramid
    float thickness;        // depth tolerance for hit detection
    float roughnessThreshold; // skip SSR for fragments above this roughness
    int maxSteps;             // max ray march iterations
    uint32_t frameIndex;
    uint32_t hiZStopLevel;    // finest Hi-Z mip to refine to (derived from resolution scale)
    // Tiled/tunable trace additions (mirrored in ssr.slang).
    uint32_t hiZStartLevel = 2;
    uint32_t hiZMaxLevel = 6;
    uint32_t outputImageIndex = 0;  // storage slot traceMain writes (ssr_current)
    glm::uvec2 ssrResolution = glm::uvec2(0); // trace target size in pixels
    uint32_t rayCapacity = 0;       // ray list element capacity
    uint32_t _pad2 = 0;
};

struct HiZPushConstants {
    uint32_t    inputTextureIndex;
    uint32_t    samplerIndex;
    uint32_t    inputMipLevel;
    uint32_t    reduceMode; // 0 = copy from depth, 1 = 2x2 min reduction
    glm::uvec2  inputResolution;
    uint32_t padding0;
    uint32_t padding1;
};

struct SSAOApplyPushConstants {
    uint32_t ssaoTextureIndex;
    uint32_t samplerIndex;
    uint32_t padding[2];
};

struct SSRAccumulatePushConstants {
    uint32_t currentSSRIndex;
    uint32_t historySSRIndex;
    uint32_t motionVectorIndex;
    uint32_t samplerIndex;
    float    temporalBlend;
    uint32_t historyValid;          // 0 = no valid history (first frame / resize)
    uint32_t padding[2];
};

struct SSRApplyPushConstants {
    uint32_t samplerIndex;
    uint32_t ssrTextureIndex;
};

enum SDFType {
    SPHERE,
    CYLINDER,
    CONE,
    PYRAMID
};

struct BillboardPushConstants {
    glm::mat4 invViewProj;
    uint64_t billboardBufferAddress;
    uint32_t billboardCount;
    uint32_t samplerIndex;
    glm::uvec2 resolution;
    uint32_t depthTextureIndex;
    uint32_t depthSamplerIndex;
    uint32_t depthTest;
    uint32_t _pad;
};

struct GPUBillboard {
    glm::vec3 position;
    uint32_t screenSpace;
    float size;
    uint32_t textureIndex;
    bool alphaBlend;
    float clipThreshold;
};

// ===== GUI =====
// One screen-space quad. Mirrors GPUGuiQuad in shaders/gui.slang; std430-clean (vec2s at
// 8-byte offsets, vec4 at 32, 64 bytes total) so the layouts match without a scalar-layout flag.
// Rects are in pixels with a top-left origin — same convention as GLFW cursor coords and Vulkan
// NDC, so hit-testing later needs no flip.
struct GPUGuiQuad {
    glm::vec2 minPx;
    glm::vec2 maxPx;
    glm::vec2 uvMin;
    glm::vec2 uvMax;
    glm::vec4 color;        // multiplied against the atlas sample
    uint32_t  textureIndex;
    // Per quad rather than per draw: a font atlas wants Nearest (crisp glyphs, no bleed across
    // atlas cells) while a 128px material thumbnail scaled to fit wants Linear, and both land in
    // the same single draw call. Costs a pad word, so the struct stays 64 B.
    uint32_t  samplerIndex;
    uint32_t  _pad1;
    uint32_t  _pad2;
};

struct GUIPushConstants {
    uint64_t   quadBufferAddress;
    glm::uvec2 resolution;
    uint32_t   quadCount;
    uint32_t   _pad0;
};

static_assert(sizeof(GPUGuiQuad) == 64, "must match GPUGuiQuad in gui.slang (std430)");
static_assert(sizeof(GUIPushConstants) == 24, "must match GUIPushConstants in gui.slang");

// ===== Particle system =====
// One element of the shared particle pool. All emitters live in a single pool buffer
// (see Renderer::particlePoolBufferIndex); each emitter owns a contiguous sub-range that it
// runs as a ring buffer.
struct Particle {
    glm::vec3 position;   // world-space
    float     age;        // seconds alive; age < 0 marks a dead/free slot
    glm::vec3 velocity;
    float     lifeSpan;   // total lifetime assigned at spawn (age reaches this -> death)
    float     rotation;   // current angular position, radians
    float     angularVel;
    float     size;
    uint32_t  seed;       // per-particle RNG state (also drives atlas frame for animated)
    uint32_t  emitterIdx;
};

// Emitter-flag bits packed into GPUParticleEmitter::flags.
constexpr uint32_t EMITTER_FLAG_ANIMATED   = 1u << 0;
constexpr uint32_t EMITTER_FLAG_LIT        = 1u << 1;
constexpr uint32_t EMITTER_FLAG_VOLUMETRIC = 1u << 2;
constexpr uint32_t EMITTER_FLAG_SOFT       = 1u << 3; // depth-fade near intersecting geometry
constexpr uint32_t EMITTER_FLAG_VOLUME_SPHERE = 1u << 4; // volumetric injection: view-independent sphere (else textured billboard)

// GPU-side emitter descriptor consumed by the compute sim + render passes.
struct GPUParticleEmitter {
    glm::vec3 position;            // world spawn origin (node world pos + rotated offset)
    float     emissionRate;       // particles / second

    glm::vec4 spawnRotation;      // quat (x,y,z,w): node world rot * rotationOffset

    float     speedMin;
    float     lifeTimeMin;
    float     speedMax;
    float     lifeTimeMax;

    glm::vec2 angularVelocityRandom;
    float     spreadAngle;        // radians, half-angle of the emission cone
    float     drag;

    glm::vec2 densityRange;
    float     phase;
    uint32_t  particleOffset;     // base index into the shared pool (Particle units)
    
    uint32_t  particleCapacity;   // ring size = ceil(lifeTimeMax * rate), workgroup-rounded
    uint32_t  textureIndex;
    uint32_t  numFrames;          // atlas frame count when EMITTER_FLAG_ANIMATED, else 0
    uint32_t  flags;              // EMITTER_FLAG_*

    glm::vec2 sizeRandom;         // per-particle size range (min, max)
    float     softRadius;         // EMITTER_FLAG_SOFT: depth-fade distance (view-space units)
    glm::vec2 emissiveRange;             
    uint32_t  _pad;
};

// Per-emitter mutable state, owned exclusively by the GPU sim after CPU zero-init on creation.
// Kept out of GPUParticleEmitter so re-uploading emitter params never races/clobbers it.
struct EmitterRuntime {
    uint32_t ringHead;          // next slot to (over)write, mod particleCapacity
    float    spawnAccumulator;  // fractional particle carried between frames
    uint32_t aliveCount;        // live particles this frame (diagnostics / indirect draw)
    uint32_t _pad;
};

struct ParticleComputePushConstants {
    uint64_t runtimeBDA;
    uint64_t emittersBDA;
    uint64_t particlesBDA;
    uint32_t emitterCount;
    uint32_t particleCount;
    float dt;
};

// Drawn per-emitter: instanceCount = particleCapacity, 6 verts/quad. The vertex shader reads
// particles[particleOffset + instanceID] by device address and degenerates dead slots (age < 0).
// Depth test is done in-shader against the resolved depth, like billboards.
// Slang compiles this push constant with std430 rules: uvec2 aligns to 8 and
// vec3 to 16. The _pad* fields make the C++ offsets match the shader exactly
// (offsets in comments; verified against particle_draw.spv). Do not reorder
// without re-checking the SPIR-V, or resolution/camera data will be misread.
struct ParticleDrawPushConstants {
    glm::mat4  viewProjection;     

    uint64_t   particlesBDA;       
    uint64_t   emittersBDA;        
    
    uint64_t   lightsBDA;          
    uint32_t   lightCount;         
    uint32_t   samplerIndex;       
    
    uint32_t   depthTextureIndex;  
    uint32_t   depthSamplerIndex;  
    glm::vec2  invResolution;         
        
    glm::vec3  cameraPos;          
    float      farPlane;           
    
    glm::vec3  cameraForward;      
    float      nearPlane;          
    
    uint32_t   shadowAtlasIndex;   
    uint32_t   emitterIndex;
    float      sphereRoundness;    
    float      opacity;            
};

struct SDF {
    glm::mat4 worldTransform;
    glm::mat4 invWorldTransform;
    glm::vec4 color;
    uint32_t type;
    float radius;
    float height;
    float padding;
};


struct SDFPushConstants {
    uint64_t sdfDataAddress;
    uint32_t sdfCount;
    uint32_t depthTextureIndex;
    uint32_t depthSamplerIndex;
    uint32_t padding;
    uint32_t _align[2];          // std430 aligns float3 to 16 bytes
    glm::vec3 cameraPos;
    float padding1;
    glm::mat4 invViewProjection;
};

struct SDFApplyPushConstants {
    uint32_t sdfTextureIndex;
    uint32_t samplerIndex;
    uint32_t padding[2];
};

struct VolumetricPushConstants {
    uint64_t lightsAddress;            // 0
    uint64_t volumeBufferAddress;        // 56
    uint32_t lightCount;               // 8
    uint32_t shadowAtlasIndex;         // 12
    uint32_t depthTextureIndex;        // 28
    uint32_t depthSamplerIndex;        // 44
    glm::vec3 cameraPos;               // 16
    uint32_t numSteps;                 // 64
    glm::vec3 cameraDir;               // 32
    uint32_t volumeCount;              // 80
    uint32_t screenSize[2];            // 48
    float    maxDist;                    // 68 (uint2 _pad aligns to 8)
    uint32_t _pad;                 // 72  shader: uint2 _pad
    glm::mat4 invViewProjection;       // 96, ends at 160
};

// Froxel composite (Pass E). Depth -> view Z -> fractional slice, trilinearly sample integratedVol,
// then blend with POSTPROCESS_VOLUMETRIC. vec3s on 16-byte boundaries to match std430.
// When debugMode != 0 the shader overwrites the scene with a visualization of a grid volume.
struct VolumetricApplyPushConstants {
    uint32_t   integratedTexIndex;   // 0   sampled 3D slot of integratedVol
    uint32_t   samplerIndex;         // 4
    uint32_t   depthTexIndex;        // 8
    uint32_t   depthSamplerIndex;    // 12
    glm::uvec3 dims;                 // 16
    float      nearZ;                // 28
    glm::vec3  cameraPos;            // 32
    float      gridFar;              // 44
    glm::vec3  cameraForward;        // 48
    uint32_t   debugMode;            // 60  0 = normal composite
    glm::mat4  invViewProjection;    // 64
    uint32_t   mediaTexIndex;        // 128 debug: sampled mediaVol
    uint32_t   scatterTexIndex;      // 132 debug: sampled scatterVol
    uint32_t   mediaPhaseTexIndex;   // 136 debug: sampled mediaPhase
    uint32_t   frame;                // 140 (ends 144)
};

// Froxel density injection — shared by passes A0 (clear) and A (analytic volume inject).
// See FROXEL_VOLUMETRICS_PLAN.md. vec3 members sit on 16-byte boundaries to match the slang
// std430 push-constant layout (mirror of FroxelInjectPushConstants in shaders/froxel_inject.slang).
struct FroxelInjectPushConstants {
    uint64_t   volsAddress;          // 0   Volume* (BDA into the per-frame volume buffer)
    uint32_t   volumeCount;          // 8
    uint32_t   mediaVolIndex;        // 12  storage-image slot of mediaVol
    glm::uvec3 dims;                 // 16  froxel grid dimensions
    float      nearZ;                // 28
    glm::vec3  cameraPos;            // 32
    float      gridFar;              // 44
    glm::vec3  cameraForward;        // 48  normalized camera look dir
    uint32_t   mediaPhaseIndex;      // 60  R32F storage slot: density-weighted phase accumulator
    glm::mat4  invViewProjection;    // 64 (ends 128)
    uint32_t   frame;
};

// Pass B — volumetric particle injection via froxel-parallel gather (clustered, like tiled lighting).
// The bin pass buckets each volumetric particle into the screen tiles its sphere overlaps; the gather
// pass runs one thread per froxel, reads its tile's particle list, and writes the summed density.
constexpr uint32_t FROXEL_TILE_SIZE = 8;             // froxels per tile edge (screen XY)
constexpr uint32_t MAX_PARTICLES_PER_TILE = 256;     // per-tile list capacity (overflow dropped)

// Bin pass (1 thread/particle): append particle index to each overlapped tile's list.
struct FroxelBinPushConstants {
    uint64_t   particlesAddress;     // 0   Particle* pool
    uint64_t   emittersAddress;      // 8   ParticleEmitter* (this frame's slice)
    uint64_t   tileCountsAddress;    // 16  uint per tile (atomic counter)
    uint64_t   tileListAddress;      // 24  uint[tiles * maxPerTile]
    uint32_t   particleCount;        // 32
    uint32_t   tilesX;               // 36
    uint32_t   tilesY;               // 40
    uint32_t   maxPerTile;           // 44
    glm::uvec3 dims;                 // 48
    float      nearZ;                // 60
    glm::vec3  cameraPos;            // 64
    float      gridFar;              // 76
    glm::vec3  cameraForward;        // 80
    float      billboardScale;       // 92
    glm::mat4  viewProj;             // 96
    glm::mat4  invViewProjection;    // 160 (ends 224)
};

// Gather pass (1 thread/froxel): sum density from the froxel's tile particle list into mediaVol.
struct FroxelGatherPushConstants {
    uint64_t   particlesAddress;     // 0
    uint64_t   emittersAddress;      // 8
    uint64_t   tileCountsAddress;    // 16
    uint64_t   tileListAddress;      // 24
    uint32_t   mediaVolIndex;        // 32
    uint32_t   tilesX;               // 36
    uint32_t   tilesY;               // 40
    uint32_t   maxPerTile;           // 44
    uint32_t   samplerIndex;         // 48
    uint32_t   frame;                // 52
    uint32_t   mediaPhaseIndex;      // 56  R32F storage slot: density-weighted phase accumulator (write)
    uint32_t   _padC;                // 60
    glm::uvec3 dims;                 // 64
    float      nearZ;                // 76
    glm::vec3  cameraPos;            // 80
    float      gridFar;              // 92
    glm::vec3  cameraForward;        // 96
    float      billboardScale;       // 108
    glm::vec3  cameraRight;          // 112
    float      _padD;                // 124
    glm::vec3  cameraUp;             // 128
    float      _padE;                // 140
    glm::mat4  invViewProjection;    // 144 (ends 208)
};

// Pass C — light scattering (1 thread/froxel). Reads mediaVol, writes lit scatterVol.
// Mirror of FroxelLightPushConstants in shaders/froxel_light.slang.
struct FroxelLightPushConstants {
    uint64_t   lightsAddress;        // 0   Light* (this frame's slice)
    uint32_t   lightCount;           // 8
    uint32_t   shadowAtlasIndex;     // 12
    glm::uvec3 dims;                 // 16
    uint32_t   mediaVolIndex;        // 28  storage slot read
    uint32_t   scatterVolIndex;      // 32  storage slot written
    float      nearZ;                // 36
    float      gridFar;              // 40
    uint32_t   mediaPhaseIndex;      // 44  R32F storage slot: density-weighted phase (read)
    glm::vec3  cameraPos;            // 48
    uint32_t   debugMode;            // 60  matches VolumetricApplyPushConstants::debugMode
    glm::vec3  cameraForward;        // 64
    uint32_t   frame;                // 76
    glm::mat4  invViewProjection;    // 80 (ends 144)
};

// Pass D — integration (1 thread per x,y column). Front-to-back marches scatterVol into integratedVol.
// Mirror of FroxelIntegratePushConstants in shaders/froxel_integrate.slang.
struct FroxelIntegratePushConstants {
    glm::uvec3 dims;                 // 0
    uint32_t   scatterVolIndex;      // 12
    uint32_t   integratedVolIndex;   // 16
    float      nearZ;                // 20
    float      gridFar;              // 24
};

struct LitMeshDrawData {
    uint32_t vertexAllocationOffset;
    uint32_t vertexOffset;
    uint32_t vertexStride;
    uint32_t firstInstance; // matches DrawIndexedIndirectCommand.firstInstance; shader uses this + SV_InstanceID to index per-instance SSBO
};

struct LitInstanceData {
    uint32_t modelMatrixIndex;
    uint32_t albedoTextureIndex;
    uint32_t roughnessTextureIndex;
    uint32_t metallicTextureIndex;
    uint32_t normalTextureIndex;
    uint32_t environmentMapIndex;
    float    maxEnvMips;
    uint32_t materialFlags;
    float    metallic;
    float    roughness;
    float    alphaCutoff;
    uint32_t packedColor; // material base colour, RGBA8 linear (reuses what was padding)
};

// Material base colour into LitInstanceData.packedColor. Linear, no sRGB encode — the shader
// multiplies it onto an already-decoded albedo sample. Mirrors unpackColorRGBA8() in common.slang.
inline uint32_t packColorRGBA8(const glm::vec4& c) {
    auto q = [](float v) { return static_cast<uint32_t>(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f); };
    return (q(c.r) << 24) | (q(c.g) << 16) | (q(c.b) << 8) | q(c.a);
}

// Frame-level push constants for the lit indirect pass (per-draw data moved to LitDrawData buffer)
struct LitPushConstants {
    uint64_t vertexBufferAddress;
    uint64_t modelMatricesAddress;
    uint64_t lightsAddress;
    uint64_t litInstanceDataAddress;
    uint64_t litMeshDrawDataAddress;
    uint64_t litPassDataAddress;
    // Added to SV_DrawIndex when indexing per-draw data. SV_DrawIndex resets to 0 per indirect call,
    // so the lit pass (one indirect call per pipeline range) sets this to the range's first command
    // index; the prepass draws all commands in one call and leaves it 0.
    uint32_t drawIDOffset = 0;
    float time = 0.0f;
    // Slot into the bindless uniform-buffer binding holding this pass's GPULitFrameUniforms.
    uint32_t frameUniformsIndex = 0;
    uint32_t _pad0 = 0;
};

// Field order matches ThumbnailPush in thumbnail.slang (mat4, then 8-byte address, then uints).
struct ThumbnailPushConstants {
    glm::mat4 mvp;
    uint64_t  vertexBufferAddress;
    uint32_t  vertexStride;
    uint32_t  vertexOffset;
};

struct LitPassData {
    uint32_t samplerIndex;
    uint32_t lightCount;
    uint32_t shadowSamplerIndex;
    uint32_t shadowAtlasIndex;
    glm::vec3 cameraPosition;
    uint32_t voxelTextureIndex;
    glm::vec3 cameraForward;
    uint32_t voxelSamplerIndex;
    glm::mat4 viewProjection;
    glm::mat4 prevViewProjection;
    glm::mat4 voxelViewProjection; // world -> voxel grid clip (VoxelizationPass::gridViewProjection)
    uint32_t voxelResolution;
    float voxelWorldExtent;
    // VXGI cone-trace tuning (VXGISettings sliders); defaults match the old compile-time values
    uint32_t giHemisphereRays = 5;
    uint32_t giMaxSteps = 9;
    uint32_t giFetchBatch = 3;
    uint32_t giMode = 0; // 0 = per-pixel cone trace, 1 = ambient-cube lookup
    std::array<uint32_t, 6> giIrradianceIndices{}; // per-face irradiance volumes, VOXEL_FACE_DIRS order
    float giStrength = 1.0f;
    float giSkyIntensity = 0.0f; // cone-miss sky radiance scale (skyboxIntensity * vxgi.skyStrength)
};

// Hot per-light data mirrored into the lit frame UBO (LightHot in common.slang).
// vec4-only members so std140 layout == this struct byte-for-byte.
struct GPULightHot {
    glm::vec4 positionRange = glm::vec4(0);   // xyz world position, w range
    glm::vec4 direction = glm::vec4(0);       // xyz normalized direction, w unused
    glm::vec4 colorIntensity = glm::vec4(0);  // rgb color, w intensity (<= 0 -> disabled slot)
    glm::uvec4 typeFlags = glm::uvec4(0);     // x LightType, y castsShadows, z numCascades, w cascade base slot
    glm::vec4 cascadeSplits = glm::vec4(0);   // CSM split distances (xyz)
};
static_assert(sizeof(GPULightHot) == 80, "must match LightHot in common.slang (std140)");

// Per-cascade shadow data mirrored into the lit frame UBO (CascadeHot in common.slang).
// Keeps the lit pass's shadow lookup off the BDA Light buffer: reading lightSpaceMatrix /
// atlasRange / texel sizes through a device-address pointer at dynamic stride 96 put a global
// load directly in front of the compare-fetch, so the two memory waits serialized. From the
// UBO these are uniform-per-light constant-cache reads the compiler can hoist and scalarize.
// vec4-only members so std140 layout == this struct byte-for-byte.
struct GPUCascadeHot {
    glm::mat4 lightSpaceMatrix = glm::mat4(1.0f);
    glm::vec4 shadowAtlasRange = glm::vec4(0);  // xy atlas UV min, zw atlas UV max
    glm::vec4 texelSizes = glm::vec4(0);        // x texelSize (tile-normalized), y worldTexelSize, zw unused
};
static_assert(sizeof(GPUCascadeHot) == 96, "must match CascadeHot in common.slang (std140)");

// Mirror of LitFrameUniforms in common.slang: the lit pass's hot pass/light data, read through
// the constant cache instead of per-fragment BDA loads. One UBO slot per frame in flight; the
// LitPassData / GPULight BDA buffers stay authoritative for every other pass.
struct GPULitFrameUniforms {
    glm::mat4 voxelViewProjection = glm::mat4(1.0f);
    glm::vec4 cameraPosExtent = glm::vec4(0);  // xyz cameraPos, w voxelWorldExtent
    glm::vec4 cameraForwardGI = glm::vec4(0);  // xyz cameraForward, w giStrength
    glm::uvec4 indicesA = glm::uvec4(0);       // sampler, lightCount (<= MAX_UBO_LIGHTS), shadowSampler, shadowAtlas
    glm::uvec4 indicesB = glm::uvec4(0);       // voxelTexture, voxelSampler, voxelResolution, giMode
    glm::uvec4 giParams = glm::uvec4(0);       // giHemisphereRays, giMaxSteps, giFetchBatch, unused
    glm::uvec4 giIrradianceA = glm::uvec4(0);  // irradiance volume indices 0-3
    glm::uvec4 giIrradianceB = glm::uvec4(0);  // xy irradiance indices 4-5, zw unused
    glm::vec4 giFloats = glm::vec4(0);         // x giSkyIntensity, yzw unused
    // Flat pool of directional cascades; GPULightHot::typeFlags.w is a light's base slot.
    GPUCascadeHot cascades[MAX_UBO_CASCADES];
    GPULightHot lights[MAX_UBO_LIGHTS];

    void setPassData(const LitPassData& pd) {
        voxelViewProjection = pd.voxelViewProjection;
        cameraPosExtent = glm::vec4(pd.cameraPosition, pd.voxelWorldExtent);
        cameraForwardGI = glm::vec4(pd.cameraForward, pd.giStrength);
        indicesA = glm::uvec4(pd.samplerIndex, std::min(pd.lightCount, MAX_UBO_LIGHTS), pd.shadowSamplerIndex, pd.shadowAtlasIndex);
        indicesB = glm::uvec4(pd.voxelTextureIndex, pd.voxelSamplerIndex, pd.voxelResolution, pd.giMode);
        giParams = glm::uvec4(pd.giHemisphereRays, pd.giMaxSteps, pd.giFetchBatch, 0u);
        giIrradianceA = glm::uvec4(pd.giIrradianceIndices[0], pd.giIrradianceIndices[1], pd.giIrradianceIndices[2], pd.giIrradianceIndices[3]);
        giIrradianceB = glm::uvec4(pd.giIrradianceIndices[4], pd.giIrradianceIndices[5], 0u, 0u);
        giFloats = glm::vec4(pd.giSkyIntensity, 0.0f, 0.0f, 0.0f);
    }
};
static_assert(sizeof(GPULitFrameUniforms) == 192 + sizeof(GPUCascadeHot) * MAX_UBO_CASCADES + sizeof(GPULightHot) * MAX_UBO_LIGHTS,
              "must match LitFrameUniforms in common.slang (std140)");

// matches VkDrawIndexedIndirectCommand)
struct DrawIndexedIndirectCommand {
    uint32_t indexCount;
    uint32_t instanceCount;
    uint32_t firstIndex;
    int32_t vertexOffset;
    uint32_t firstInstance;
};

struct Line {
    glm::vec3 startPoint;
    glm::vec3 endPoint;
    glm::vec4 color;
};

struct LineVertex {
    glm::vec4 color;
    glm::vec4 position;
};

struct Shader {
    std::string sourceFile;
    uint32_t refCount;
    uint32_t pipelineType = 1;
    uint32_t pipelineIndex;

    bool operator<(const Shader& other) const {
        if (sourceFile != other.sourceFile) {
            return sourceFile < other.sourceFile;
        }
        return pipelineIndex < other.pipelineIndex;
    }
};
struct Material {
    std::string name; // Material name from .mtl file or assigned name
    Shader shaderSource;
    MaterialFlags flags;
    // 2st bit : hasAlbedo
    // 3nd bit : hasRoughness
    // 4rd bit : hasMetallic
    // 5th bit : hasNormal
    uint32_t usageCount;
    uint32_t materialID;
    // Base colour factor, multiplied onto the albedo map in the lit pass. White = untinted; GLM
    // leaves vec4 default-uninitialized, so without this a default-constructed Material tints by
    // whatever was on the stack.
    glm::vec4 color = glm::vec4(1.0f);
    uint32_t albedoTextureIndex;
    float metallic;
    uint32_t metallicTextureIndex;
    float roughness;
    uint32_t roughnessTextureIndex;
    uint32_t normalTextureIndex; // should be set to default normal if not present
    uint32_t environmentMapIndex;
    float alphaCutoff = 0.5f;
    bool alphaClip = false;
    // GUI preview thumbnail; MaterialThumbnailPass renders into this. Not hashed/compared.
    uint32_t thumbnailTextureIndex = 0xFFFFFFFF;

    bool operator<(const Material& other) const {
        if (name != other.name)
            return name < other.name;
        if (shaderSource < other.shaderSource)
            return true;
        if (other.shaderSource < shaderSource)
            return false;

        if (flags != other.flags)
            return flags < other.flags;
        if (metallic != other.metallic)
            return metallic < other.metallic;
        if (roughness != other.roughness)
            return roughness < other.roughness;
        if (normalTextureIndex != other.normalTextureIndex)
            return normalTextureIndex < other.normalTextureIndex;
        if (albedoTextureIndex != other.albedoTextureIndex)
            return albedoTextureIndex < other.albedoTextureIndex;
        if (metallicTextureIndex != other.metallicTextureIndex)
            return metallicTextureIndex < other.metallicTextureIndex;
        if (roughnessTextureIndex != other.roughnessTextureIndex)
            return roughnessTextureIndex < other.roughnessTextureIndex;
        return false;
    }
    bool operator==(const Material& other) const {
        return name == other.name && shaderSource.sourceFile == other.shaderSource.sourceFile && flags == other.flags && metallic == other.metallic &&
               roughness == other.roughness && color == other.color && albedoTextureIndex == other.albedoTextureIndex && metallicTextureIndex == other.metallicTextureIndex &&
               roughnessTextureIndex == other.roughnessTextureIndex && normalTextureIndex == other.normalTextureIndex;
    }


};

struct Image {
    vk::raii::Image image = nullptr;
    vk::raii::DeviceMemory memory = nullptr;
    vk::raii::ImageView view = nullptr;
};

struct Mesh {
    std::string sourceFile;
    std::string name = "";

    uint32_t vertexAllocationOffset = 0;
    vk::DeviceSize vertexOffset = 0;
    uint32_t vertexCount = 0;
    uint32_t vertexStride = 0;

    uint32_t indexAllocationOffset = 0;
    uint32_t indexOffset = 0;
    uint32_t indexCount = 0;

    // index count per LOD [LOD0, LOD1, ...]; consecutive ranges in the index allocation
    std::vector<uint32_t> LODs;
    // LOD0 surface area in model space; drives screen-space triangle-size LOD selection
    float surfaceArea = 0.0f;
    // LOD picked by the last buildGeometryDrawCommands pass (debug: bbox tint, wireframe)
    uint32_t currentLOD = 0;
    uint32_t lodIndexCount(uint32_t lod) const { return LODs.empty() ? indexCount : LODs[lod]; }
    // start of a LOD's range, in indices, relative to indexOffset
    uint32_t lodIndexStart(uint32_t lod) const {
        uint32_t start = 0;
        for (uint32_t i = 0; i < lod; i++) start += LODs[i];
        return start;
    }

    // Local-space AABB
    glm::vec3 boundingBoxMin = glm::vec3(0.0f);
    glm::vec3 boundingBoxMax = glm::vec3(0.0f);
    glm::vec3 center = glm::vec3(0.0f);

    uint32_t positionAllocationOffset = 0;
    uint32_t positionOffset = 0;
    uint32_t positionCount = 0;

    // for uniqueness heuristic
    float minRadius = 0.0f;
    float maxRadius = 0.0f;

    // Unit scale of the source file, baked into the geometry at import time (1.0 = as-authored).
    // Persisted per source file so scene reloads re-bake identically.
    float importScale = 1.0f;

    // Index of the source-file entry (shape/material group) this geometry was built from.
    // Persisted so scene loads can rebuild the mesh directly, skipping instance detection.
    uint32_t sourceEntryIndex = 0;

    // CPU-side geometry for raycasting
    std::vector<glm::vec3> cpuPositions;
    std::vector<glm::vec3> cpuNormals;
    std::vector<uint32_t> cpuIndices;

    bool freed = false;
    uint32_t refCount = 0;

    // GUI preview thumbnail; thumbnail pass renders into thumbnailTextureIndex when dirty.
    uint32_t thumbnailTextureIndex = 0xFFFFFFFF;
    bool     thumbnailDirty = true;
};

enum class QuadTileState : uint8_t { Free, Split, Occupied };

struct ShadowAtlasQuadTreeTile {
    uint32_t parent = 0xFFFFFFFFu;
    uint32_t size = 0;
    QuadTileState state = QuadTileState::Free;
    std::array<uint32_t,4> children = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
};

struct ShadowAtlas {
    uint32_t textureIndex = 0xFFFFFFFF;
    std::array<ShadowAtlasQuadTreeTile,SHADOW_ATLAS_QUADTREE_COUNT> quadTree;

    void init() {
        quadTree[0].size = SHADOW_ATLAS_SIZE;
        for (uint32_t i = 0; i < SHADOW_ATLAS_QUADTREE_COUNT; ++i) {
            uint32_t firstChild = 4u * i + 1u;
            if (firstChild >= SHADOW_ATLAS_QUADTREE_COUNT) continue;
            uint32_t childSize = quadTree[i].size / 2u;
            for (uint32_t c = 0; c < 4; ++c) {
                uint32_t childIdx = firstChild + c;
                quadTree[i].children[c] = childIdx;
                quadTree[childIdx].parent = i;
                quadTree[childIdx].size = childSize;
            }
        }
    }

    bool allocateShadowMap(uint32_t size, uint32_t& outTile, glm::vec4& outUVRange) {
        uint32_t target = SHADOW_ATLAS_MIN_TILE;
        while (target < size) target <<= 1;
        if (target > SHADOW_ATLAS_SIZE) return false;

        // Best-fit: pick the smallest Free node with size >= target. Descending only
        // through Split nodes means we naturally prefer Free slots already nested inside
        // Split subtrees (clustered packing) over fragmenting a fresh large Free area.
        struct Frame { uint32_t idx, x, y; };
        Frame stack[64];
        uint32_t top = 0;
        stack[top++] = {0u, 0u, 0u};

        uint32_t foundIdx  = 0xFFFFFFFFu;
        uint32_t foundSize = 0xFFFFFFFFu;
        uint32_t foundX = 0, foundY = 0;

        while (top) {
            Frame f = stack[--top];
            auto& node = quadTree[f.idx];
            if (node.state == QuadTileState::Occupied) continue;
            if (node.size < target) continue;

            if (node.state == QuadTileState::Free) {
                if (node.size < foundSize) {
                    foundIdx  = f.idx;
                    foundSize = node.size;
                    foundX    = f.x;
                    foundY    = f.y;
                    if (foundSize == target) break; // exact — can't beat this
                }
                continue; // don't descend into Free (whole subtree is already free)
            }

            // Split: descend into children
            uint32_t half = node.size / 2u;
            for (uint32_t c = 0; c < 4; ++c) {
                uint32_t ch = node.children[c];
                if (ch == 0xFFFFFFFFu) continue;
                uint32_t cx = f.x + ((c & 1u) ? half : 0u);
                uint32_t cy = f.y + ((c & 2u) ? half : 0u);
                stack[top++] = {ch, cx, cy};
            }
        }

        if (foundIdx == 0xFFFFFFFFu) return false;

        // Descend TL splitting Free nodes. (child[0] offsets x/y by 0.)
        uint32_t idx = foundIdx;
        while (quadTree[idx].size > target) {
            quadTree[idx].state = QuadTileState::Split;
            idx = quadTree[idx].children[0];
        }
        quadTree[idx].state = QuadTileState::Occupied;

        outTile = idx;
        const float inv = 1.0f / float(SHADOW_ATLAS_SIZE);
        outUVRange = glm::vec4(
            float(foundX) * inv,
            float(foundY) * inv,
            float(foundX + target) * inv,
            float(foundY + target) * inv
        );
        return true;
    }

    void freeShadowMap(uint32_t tile) {
        if (tile >= SHADOW_ATLAS_QUADTREE_COUNT) return;
        if (quadTree[tile].state != QuadTileState::Occupied) return;
        quadTree[tile].state = QuadTileState::Free;

        // Roll up: if all siblings are Free, parent collapses back to Free.
        uint32_t p = quadTree[tile].parent;
        while (p != 0xFFFFFFFFu) {
            bool allFree = true;
            for (uint32_t c = 0; c < 4; ++c) {
                uint32_t ch = quadTree[p].children[c];
                if (ch != 0xFFFFFFFFu && quadTree[ch].state != QuadTileState::Free) {
                    allFree = false;
                    break;
                }
            }
            if (!allFree) break;
            quadTree[p].state = QuadTileState::Free;
            p = quadTree[p].parent;
        }
    }
};

enum class LightType { Point, Directional, Spot, Area, COUNT };

struct GPUCascade {
    glm::mat4 lightSpaceMatrix;
    glm::vec4 shadowAtlasRange;
    float splitDistance;
    float texelSize;
    float worldTexelSize;
    float _padding; // forces sizeof to match the Slang side's 96-byte stride under scalarBlockLayout
};

struct GPUShadowMap {
    glm::mat4 lightSpaceMatrix;
    glm::vec4 shadowAtlasRange;
};

struct GPULight {
    LightType type = LightType::Point;              
    glm::vec3 position = glm::vec3(0,0,0);
    glm::vec3 direction = glm::vec3(0,0,0);
    float range = 10.0f;            
    float intensity = 1.0f;
    uint32_t padding[3];         
    glm::vec4 color = glm::vec4(0, 0, 0, 1);
    int castsShadows = 0;           
    int showCascades = 0;           
    uint32_t numCascades = 3;       
    uint32_t shadowResolution = DEFAULT_SHADOW_RESOLUTION;
    GPUCascade cascades[3]; // for CSM directional lights
    GPUShadowMap shadowMaps[6]; // more generic than cascades up to 6 for point lights
};

enum class VolumeShape {
    SPHERE,
    BOX
};

// GPU-side volume payload — layout must match `struct Volume` in shaders/volumetrics.slang.
struct GPUVolume {
    uint32_t nodeIndex = 0;
    float density = 0.8f;
    float phase = 0.8f;
    VolumeShape shape = VolumeShape::SPHERE;
    glm::vec3 center = glm::vec3(0,0,0);
    float radius = 1.0f;
    glm::vec3 dimensions = glm::vec3(1,1,1);
};

// CPU-side volume: authoring data only. Streamed into the volume buffer every frame (like
// Billboard), so it carries no persistent GPU slot or dirty-tracking state. The world center is
// pulled fresh from the owning node at stream time and passed into toGPU, mirroring Billboard.
struct Volume {
    uint32_t nodeIndex = 0;
    float density = 0.8f;
    float phase = 0.8f;
    VolumeShape shape = VolumeShape::SPHERE;
    float radius = 1.0f;
    glm::vec3 dimensions = glm::vec3(1,1,1);

    GPUVolume toGPU(const glm::vec3& worldCenter) const {
        return GPUVolume{nodeIndex, density, phase, shape, worldCenter, radius, dimensions};
    }
};

struct Vertex {
    glm::vec3 position;     // 0-11
    glm::vec3 normal;       // 12-23
    glm::vec3 tangent;      // 24-35
    uint32_t materialIndex; // 36-39
    glm::vec2 texCoord;     // 40-47

    bool operator==(const Vertex& other) const {
        return position == other.position && normal == other.normal && texCoord == other.texCoord && tangent == other.tangent && materialIndex == other.materialIndex;
    }
};

namespace std {
template <> struct hash<Vertex> {
    size_t operator()(Vertex const& vertex) const {
        return ((hash<glm::vec3>()(vertex.position) ^ (hash<glm::vec3>()(vertex.normal) << 1)) >> 1) ^ (hash<glm::vec2>()(vertex.texCoord) << 1);
    }
};

template <> struct hash<Material> {
    size_t operator()(Material const& material) const {
        size_t h0 = hash<string>()(material.name);
        size_t h1 = hash<string>()(material.shaderSource.sourceFile);
        size_t h2 = hash<uint32_t>()(material.flags);
        size_t h3 = hash<glm::vec4>()(material.color);
        size_t h4 = hash<uint32_t>()(material.albedoTextureIndex);
        size_t h5 = hash<float>()(material.metallic);
        size_t h6 = hash<uint32_t>()(material.metallicTextureIndex);
        size_t h7 = hash<float>()(material.roughness);
        size_t h8 = hash<uint32_t>()(material.roughnessTextureIndex);
        size_t h9 = hash<uint32_t>()(material.normalTextureIndex);
        size_t h10 = hash<uint32_t>()(material.environmentMapIndex);

        // Combine all hashes using XOR and bit shifting
        return h0 ^ (h1 << 1) ^ (h2 << 2) ^ (h3 << 3) ^ (h4 << 4) ^ (h5 << 5) ^
               (h6 << 6) ^ (h7 << 7) ^ (h8 << 8) ^ (h9 << 9) ^ (h10 << 10);
    }
};
} // namespace std

