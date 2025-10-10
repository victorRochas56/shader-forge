#pragma once

#include <string>

#include "constants.hpp"

#define GLM_FORCE_RADIANS
#define GLM_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

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
    uint32_t textureMask;
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

struct DepthVisPushConstants {
    uint32_t depthIndex;
    uint32_t depthSamplerIndex;
    uint32_t showShadowMap = 0xFFFFFFFF;
    uint32_t shadowMapSamplerIndex;
    float nearPlane;
    float farPlane;
    uint32_t linearize;
    uint32_t doDepthBuffering;
};

struct ShadowPushConstants {
    uint32_t vertexAllocationIndex;
    uint32_t vertexOffset;
    uint32_t vertexStride;
    uint32_t modelMatrixIndex;
    glm::mat4 lightSpaceMatrix;
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
    Shader shaderSource;
    uint32_t textureMask;
    // 1st bit : hasAlbedo
    // 2nd bit : hasRoughness
    // 3rd bit : hasMetallic
    // 4th bit : hasNormal
    uint32_t usageCount;
    uint32_t padding;
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
        if (shaderSource < other.shaderSource)
            return true;
        if (other.shaderSource < shaderSource)
            return false;

        if (textureMask != other.textureMask)
            return textureMask < other.textureMask;
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
        return shaderSource.sourceFile == other.shaderSource.sourceFile && textureMask == other.textureMask && metallic == other.metallic && roughness == other.roughness &&
               color == other.color && albedoTextureIndex == other.albedoTextureIndex && metallicTextureIndex == other.metallicTextureIndex &&
               roughnessTextureIndex == other.roughnessTextureIndex && normalTextureIndex == other.normalTextureIndex;
    }
};

struct Mesh {
    std::string sourceFile;
    std::vector<uint32_t> subMeshes;
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
};

enum class LightType { Point, Directional, Spot, Area, COUNT };

struct Cascade {
    glm::mat4 lightSpaceMatrix;
    uint32_t shadowMapIndex;
    float splitDistance;
    float texelSize;
    uint32_t padding;
};

struct Light {
    LightType type = LightType::Point;
    uint32_t modelMatrixIndex = 0;
    uint32_t padding;
    uint32_t padding1;
    float range = 10.0;
    float intensity = 1.0;
    uint32_t shadowMapIndex = 0xFFFFFFFF;
    uint32_t shadowResolution = DEFAULT_SHADOW_RESOLUTION;
    glm::vec4 color = glm::vec4(0, 0, 0, 1);
    glm::mat4 lightSpaceMatrix;
    glm::vec3 direction = glm::vec3(1, 0, 0);
    int castsShadows = 0;
    uint32_t padding2;
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
} // namespace std
