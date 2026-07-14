#pragma once

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
    uint32_t padding2;
    uint32_t padding3;
    glm::mat4 invViewProjMatrix;
};

enum ImageVisFlags : uint32_t 
{
    IMAGE_VIS_NONE =    0,
    B_W_IMAGE =         1 << 0,
    FLIP_VERTICAL =     1 << 1,
    LINEARIZE =         1 << 2,
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
    float resolutionScale = 0.5f;
    float roughnessThreshold = 0.9f;
    int maxSteps = 32;
    float thickness = 0.1f;
    float temporalBlend = 0.333f;
    bool resolutionDirty = false;
};

struct VolumetricSettings {
    bool enabled = true;
    float resolutionScale = 0.5f;
    float blurRadius = 2.0f;
    int numSteps = 16;
    float maxDist = 35.0f;
};

struct TonemapSettings {
    float ev100 = 6.0f;        // manual exposure; higher = darker (used when autoExposure off)
    uint32_t op = 1;           // 0 = Reinhard, 1 = ACES, 2 = none (clamp)
    bool autoExposure = true; // meter scene average luminance and expose automatically
    float exposureComp = 0.0f; // EV bias on auto exposure; + = brighter
    float minEV = -1.5f;       // clamp the auto-metered EV100
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
    bool showGizmos = true;
    bool showBBoxes = false;
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

struct ComputeTestPushConstants {
    uint32_t outputImageIndex;  // storage-image descriptor index the compute shader writes to
    uint32_t width;
    uint32_t height;
    float time;
};

struct ComputePresentPushConstants {
    uint32_t textureIndex;      // sampled view of the compute output
    uint32_t samplerIndex;
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
    uint32_t  particleOffset;     // base index into the shared pool (Particle units)
    uint32_t  particleCapacity;   // ring size = ceil(lifeTimeMax * rate), workgroup-rounded

    uint32_t  textureIndex;
    uint32_t  numFrames;          // atlas frame count when EMITTER_FLAG_ANIMATED, else 0
    uint32_t  flags;              // EMITTER_FLAG_*
    glm::vec2 sizeRandom;         // per-particle size range (min, max)
    float     softRadius;         // EMITTER_FLAG_SOFT: depth-fade distance (view-space units)
    glm::vec2 _pad;               // pad to a 16-byte multiple (112 bytes total)
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
struct ParticleDrawPushConstants {
    glm::mat4  viewProjection;
    uint64_t   particlesBDA;       // pool base address
    uint32_t   particleOffset;     // this emitter's base index into the pool
    uint32_t   particleCount;      // instances to draw (emitter ring capacity)
    uint32_t   textureIndex;
    uint32_t   numFrames;          // atlas frames, 0 = static
    uint32_t   samplerIndex;
    uint32_t   depthTextureIndex;
    uint32_t   depthSamplerIndex;
    uint32_t   flags;              // EMITTER_FLAG_*
    glm::uvec2 resolution;
    float      softRadius;         // EMITTER_FLAG_SOFT depth-fade distance (view-space units)
    float      nearPlane;          // for linearizing the sampled scene depth
    float      farPlane;
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

struct VolumetricApplyPushConstants {
    uint32_t volumetricTextureIndex;
    uint32_t samplerIndex;
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
    uint32_t _padding;
};

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
    uint32_t padding1;
    glm::vec3 cameraForward;
    uint32_t padding2;
    glm::mat4 viewProjection;
    glm::mat4 prevViewProjection;
};

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
    glm::vec4 color;
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

    bool allocateShadowMap(uint32_t size, uint32_t& ouTile, glm::vec4& outUVRange) {
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

        ouTile = idx;
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

struct GPUPointFace {
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
    GPUCascade cascades[3];
    GPUPointFace pointFaces[6];
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

