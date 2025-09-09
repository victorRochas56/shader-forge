#pragma once

#include <string>

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
    uint32_t samplerIndex; // Index into samplers
    uint32_t padding1;
    uint32_t padding2;
    uint32_t padding3;
    glm::mat4 viewProjection;
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
    uint32_t padding2;
    uint32_t padding3;

    bool operator<(const Material& other) const {
        if (shaderSource < other.shaderSource) return true;
        if (other.shaderSource < shaderSource) return false;
        
        if (textureMask != other.textureMask) return textureMask < other.textureMask;
        if (metallic != other.metallic) return metallic < other.metallic;
        if (roughness != other.roughness) return roughness < other.roughness;
        if (normalTextureIndex != other.normalTextureIndex) return normalTextureIndex < other.normalTextureIndex;
        if (albedoTextureIndex != other.albedoTextureIndex) return albedoTextureIndex < other.albedoTextureIndex;
        if (metallicTextureIndex != other.metallicTextureIndex) return metallicTextureIndex < other.metallicTextureIndex;
        if (roughnessTextureIndex != other.roughnessTextureIndex) return roughnessTextureIndex < other.roughnessTextureIndex;
        return false; 
    }
};


struct Mesh {
    std::string sourceFile;
    std::vector<uint32_t> subMeshes;
    bool freed; //set when the vertex allocation is freed, should be checked before trying to render a mesh 
};

struct SubMesh {
    uint32_t vertexAllocationIndex;
    vk::DeviceSize vertexOffset;
    uint32_t vertexCount;
    uint32_t vertexStride;
};

struct Light {
    uint32_t type = 0;
    uint32_t modelMatrixIndex;
    float range = 0;
    float intensity = 0;
    glm::vec4 color = glm::vec4(0, 0, 0, 1);
    uint32_t allocationIndex;
};

struct Vertex {
    glm::vec3 position;
    uint32_t padding0;
    glm::vec3 normal;
    uint32_t padding1;
    glm::vec3 tangent;
    uint32_t materialIndex;
    glm::vec2 texCoord;
    uint32_t padding2;
    uint32_t padding3;
};
