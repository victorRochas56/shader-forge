#pragma once

#include <string>

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
    HAS_NORMAL =    1 << 4
};

/*
data only structs for push constants and various scene elements
*/
struct PushConstants {
    uint32_t vertexAllocationIndex; // Index into vertex allocations
    uint32_t vertexOffset;          // Byte offset in vertex buffer
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
    FLIP_VERTICAL =     1 << 1
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
    uint32_t padding0;
};

struct ShadowPushConstants {
    glm::mat4 lightSpaceMatrix;
    uint32_t elementOffsetModel;  // Element offset for this frame's model matrices
    uint32_t elementOffsetShadow; // Element offset for this frame's shadow draw data
};

struct ShadowDrawData {
    uint32_t vertexAllocationIndex;
    uint32_t vertexOffset;
    uint32_t vertexStride;
    uint32_t modelMatrixIndex;
};

struct BlurPushConstants {
    uint32_t inputTextureIndex;
    uint32_t samplerIndex;
    int32_t isHorizontal;
    float blurRadius;
    glm::uvec2 resolution;
    uint32_t padding1;
    uint32_t padding2;
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

struct SSAOApplyPushConstants {
    uint32_t ssaoTextureIndex;
    uint32_t samplerIndex;
    uint32_t padding[2];
};

struct LitDrawData {
    uint32_t vertexAllocationIndex;
    uint32_t vertexOffset;
    uint32_t vertexStride;
    uint32_t modelMatrixIndex;
    uint32_t albedoTextureIndex;
    uint32_t roughnessTextureIndex;
    uint32_t metallicTextureIndex;
    uint32_t normalTextureIndex;
    uint32_t environmentMapIndex;
    uint32_t materialFlags;
    float    metallic;
    float    roughness;
};

// Frame-level push constants for the lit indirect pass (per-draw data moved to LitDrawData buffer)
struct LitPushConstants {
    uint32_t  samplerIndex;
    uint32_t  lightCount;
    uint32_t  shadowSamplerIndex;
    uint32_t  elementOffsetModel;
    uint32_t  elementOffsetLight;
    uint32_t  elementOffsetLit;  // frame offset into the LitDrawData buffer
    uint32_t  padding0;
    uint32_t  padding1;
    glm::vec3 cameraPosition;
    uint32_t  padding2;
    glm::mat4 viewProjection;
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
    uint32_t padding3;

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

struct Mesh {
    std::string sourceFile;
    std::vector<uint32_t> subMeshes;
    std::vector<int> originalMaterialIds; // Original material ID from OBJ file for each submesh
    std::vector<std::string> originalMaterialNames; // Original material name from OBJ file for each submesh
    glm::vec3 boundingBoxMin = glm::vec3(0.0f); // AABB in local/model space
    glm::vec3 boundingBoxMax = glm::vec3(0.0f);
    bool freed; // set when the vertex allocation is freed, should be checked before trying to render a mesh
    uint32_t refCount = 0;
};

struct SubMesh {
    uint32_t vertexAllocationIndex;
    vk::DeviceSize vertexOffset;
    uint32_t vertexCount;
    uint32_t vertexStride;

    uint32_t indexAllocationIndex;
    uint32_t indexOffset;
    uint32_t indexCount;

    // Local-space AABB for per-submesh culling
    glm::vec3 boundingBoxMin = glm::vec3(0.0f);
    glm::vec3 boundingBoxMax = glm::vec3(0.0f);

    // CPU-side geometry for raycasting
    std::vector<glm::vec3> cpuPositions;
    std::vector<uint32_t> cpuIndices;
};

enum class LightType { Point, Directional, Spot, Area, COUNT };

struct Cascade {
    glm::mat4 lightSpaceMatrix;
    uint32_t shadowMapIndex;
    float splitDistance;
    float texelSize;
    float worldTexelSize;
};

struct Light {
    LightType type = LightType::Point;
    uint32_t modelMatrixIndex = 0;
    uint32_t nodeIndex = MAX_NODES;
    uint32_t padding1;
    float range = 10.0;
    float intensity = 1.0;
    uint32_t shadowMapIndex = 0xFFFFFFFF;
    uint32_t shadowResolution = DEFAULT_SHADOW_RESOLUTION;
    glm::vec4 color = glm::vec4(0, 0, 0, 1);
    glm::mat4 lightSpaceMatrix;
    glm::vec3 direction = glm::vec3(1, 0, 0);
    int castsShadows = 0;
    int showCascades = 0;
    uint32_t padding3;
    uint32_t padding4;
    uint32_t numCascades = 4;
    std::array<Cascade, 4> cascades;

    bool operator==(const Light& other) const {
        return type == other.type && modelMatrixIndex == other.modelMatrixIndex && range == other.range && intensity == other.intensity && shadowMapIndex == other.shadowMapIndex &&
               shadowResolution == other.shadowResolution && color == other.color && lightSpaceMatrix == other.lightSpaceMatrix && direction == other.direction &&
               castsShadows == other.castsShadows;
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

