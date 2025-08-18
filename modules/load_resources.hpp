#pragma once
#ifndef VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1 // for raii
#endif
#ifndef VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_NO_CONSTRUCTORS 1 // for structs constructors
#endif
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <devices.hpp>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <resource_manager.hpp>
#include <stdexcept>
#include <structs.hpp>
#include <swapchain.hpp>
#include <unordered_map>
#include <utils.hpp>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#define GLFW_INCLUDE_VULKAN // REQUIRED only for GLFW CreateWindowSurface.
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#endif

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

class MeshLoader {
    ResourceManager* resourceManager;
    Devices* devices;

  public:
    MeshLoader(ResourceManager& manager, Devices& devices) : resourceManager(&manager), devices(&devices) {}

    // Load mesh from vertex data
    Mesh loadMesh(const std::vector<Vertex>& vertices, const std::string& albedoTexturePath, const std::string& roughnessTexturePath, const std::string& metallicTexturePath,
                  const std::string& normalTexturePath, glm::vec3 position = glm::vec3(0.0f), glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
                  glm::vec3 scale = glm::vec3(1.0f)) {

        // Allocate vertex buffer space
        auto vertexInfo = resourceManager->allocateVertexBuffer(vertices.data(), vertices.size() * sizeof(Vertex), static_cast<uint32_t>(vertices.size()), sizeof(Vertex));

        // Load textures
        uint32_t albedoTextureIndex;
        if (!albedoTexturePath.empty()) {
            auto albedoTextureData = loadTextureFromFileImpl(albedoTexturePath);
            albedoTextureIndex = resourceManager->allocateTexture(albedoTextureData.data, albedoTextureData.width, albedoTextureData.height);
        } else {
            albedoTextureIndex = resourceManager->getWhiteTextureIndex(); // Use default
        }
        uint32_t rougnessTextureIndex;
        if (!roughnessTexturePath.empty()) {
            auto roughnessTextureData = loadTextureFromFileImpl(roughnessTexturePath);
            rougnessTextureIndex = resourceManager->allocateTexture(roughnessTextureData.data, roughnessTextureData.width, roughnessTextureData.height);
        } else {
            rougnessTextureIndex = resourceManager->getWhiteTextureIndex(); // Use default
        }

        uint32_t metallicTextureIndex;
        if (!metallicTexturePath.empty()) {
            auto metallicTextureData = loadTextureFromFileImpl(metallicTexturePath);
            metallicTextureIndex = resourceManager->allocateTexture(metallicTextureData.data, metallicTextureData.width, metallicTextureData.height);
        } else {
            metallicTextureIndex = resourceManager->getBlackTextureIndex(); // Use default
        }

        uint32_t normalTextureIndex;
        if (!normalTexturePath.empty()) {
            auto normalTextureData = loadTextureFromFileImpl(normalTexturePath);
            normalTextureIndex = resourceManager->allocateTexture(normalTextureData.data, normalTextureData.width, normalTextureData.height, vk::Format::eR8G8B8A8Unorm);
        } else {
            normalTextureIndex = resourceManager->getDefaultNormalIndex(); // Use default
        }

        // Create model matrix
        uint32_t modelMatrixIndex = resourceManager->allocateModelMatrixBuffer(position, rotation, scale);

        // Use default sampler
        uint32_t samplerIndex = resourceManager->getDefaultSamplerIndex();

        // Create mesh struct
        return Mesh{.vertexAllocationIndex = vertexInfo.allocationIndex,
                    .vertexOffset = vertexInfo.offset,
                    .vertexCount = vertexInfo.vertexCount,
                    .vertexStride = sizeof(Vertex),
                    .modelMatrixIndex = modelMatrixIndex,
                    .albedoTextureIndex = albedoTextureIndex,
                    .roughnessTextureIndex = rougnessTextureIndex,
                    .metallicTextureIndex = metallicTextureIndex,
                    .normalTextureIndex = normalTextureIndex,
                    .samplerIndex = samplerIndex};
    }

    // load from file (OBJ, GLTF, etc.)
    Mesh loadFromFile(const std::string& meshPath, const std::string& albedoTexturePath = "", const std::string& roughnessTexturePath = "",
                      const std::string& metallicTexturePath = "", const std::string& normalTexturePath = "", glm::vec3 position = glm::vec3(0.0f),
                      glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f), glm::vec3 scale = glm::vec3(1.0f)) {

        auto meshData = loadMeshFromFile(meshPath);

        return loadMesh(meshData.vertices, albedoTexturePath, roughnessTexturePath, metallicTexturePath, normalTexturePath, position, rotation, scale);
    }

    // Update mesh transform
    void updateMeshTransform(Mesh& mesh, glm::vec3 position, glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f), glm::vec3 scale = glm::vec3(1.0f)) {

        resourceManager->updateModelMatrix(mesh.modelMatrixIndex, position, rotation, scale);
    }

    // Update mesh vertices
    void updateMeshVertices(Mesh& mesh, const std::vector<Vertex>& newVertices) {
        if (newVertices.size() != mesh.vertexCount) {
            throw std::runtime_error("Vertex count mismatch - cannot update");
        }

        resourceManager->updateVertexBuffer(mesh.vertexAllocationIndex, newVertices.data(), newVertices.size() * sizeof(Vertex));
    }

    // Clean up mesh resources
    void freeMesh(const Mesh& mesh) {
        resourceManager->freeVertexBuffer(mesh.vertexAllocationIndex);
        resourceManager->freeModelMatrix(mesh.modelMatrixIndex);

        // Don't free texture/sampler as they might be shared
    }

    uint32_t loadTextureFromFile(const std::string& path, vk::Format format = vk::Format::eR8G8B8A8Srgb, vk::ImageType imageType = vk::ImageType::e2D,
                                 vk::ImageViewType viewType = vk::ImageViewType::e2D) {
        auto textureData = loadTextureFromFileImpl(path);
        return resourceManager->allocateTexture(textureData.data, textureData.width, textureData.height, format, imageType, viewType);
    }

    uint32_t loadCubeMapFromFiles(uint32_t width, uint32_t height, std::string posX, std::string negX, std::string posY, std::string negY, std::string posZ,
                                  std::string negZ) {

        std::vector<std::string> faceFiles = {posX, negX, posY, negY, posZ, negZ};
        // Load all 6 face data into a single buffer
        size_t faceSize = width * height * 4;
        size_t totalSize = faceSize * 6;
        std::vector<unsigned char> allFaceData(totalSize);

        for (int face = 0; face < 6; face++) {
            int imgWidth, imgHeight, channels;
            unsigned char* imageData = stbi_load(faceFiles[face].c_str(), &imgWidth, &imgHeight, &channels, STBI_rgb_alpha);
            if (!imageData) {
                throw std::runtime_error("Failed to load face: " + faceFiles[face]);
            }

            // Copy face data to the combined buffer
            memcpy(allFaceData.data() + face * faceSize, imageData, faceSize);
            stbi_image_free(imageData);
        }
        return resourceManager->allocateCubemap(allFaceData.data(), width, height, vk::Format::eR8G8B8A8Srgb, vk::ImageType::e2D, vk::ImageViewType::eCube);
    }

  private:
    struct TextureData {
        unsigned char* data;
        int width, height;
        ~TextureData() {
            if (data)
                stbi_image_free(data);
        }
    };

    struct MeshData {
        std::vector<Vertex> vertices;
    };

    TextureData loadTextureFromFileImpl(const std::string& path) {
        int width, height, channels;
        unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, STBI_rgb_alpha);
        if (!data) {
            throw std::runtime_error("Failed to load texture: " + path);
        }
        return {data, width, height};
    }

    MeshData loadMeshFromFile(const std::string& meshPath) {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, meshPath.c_str())) {
            throw std::runtime_error(warn + err);
        }
        MeshData meshData = {};
        size_t indexOffset = meshData.vertices.size();

        for (const auto& shape : shapes) {
            for (const auto& index : shape.mesh.indices) {
                Vertex vertex{};

                // Load position (with bounds check)
                if (index.vertex_index >= 0 && index.vertex_index < attrib.vertices.size() / 3) {
                    vertex.position = {attrib.vertices[3 * index.vertex_index + 0], attrib.vertices[3 * index.vertex_index + 1], attrib.vertices[3 * index.vertex_index + 2]};
                }

                // Load normal
                if (index.normal_index >= 0 && index.normal_index < attrib.normals.size() / 3) {
                    vertex.normal = {attrib.normals[3 * index.normal_index + 0], attrib.normals[3 * index.normal_index + 1], attrib.normals[3 * index.normal_index + 2]};
                } else {
                    vertex.normal = {0.0f, 1.0f, 0.0f}; // Default up normal
                }

                // Load texture coordinates
                if (index.texcoord_index >= 0 && index.texcoord_index < attrib.texcoords.size() / 2) {
                    vertex.texCoord = {attrib.texcoords[2 * index.texcoord_index + 0], 1.0f - attrib.texcoords[2 * index.texcoord_index + 1]};
                } else {
                    vertex.texCoord = {0.0f, 0.0f};
                }

                meshData.vertices.push_back(vertex);
            }
            // Second pass: calculate tangents for triangles
            size_t vertexCount = meshData.vertices.size() - indexOffset;
            for (size_t i = 0; i < vertexCount; i += 3) {
                if (i + 2 >= vertexCount)
                    break; // Ensure we have a complete triangle

                size_t idx0 = indexOffset + i;
                size_t idx1 = indexOffset + i + 1;
                size_t idx2 = indexOffset + i + 2;

                Vertex& v0 = meshData.vertices[idx0];
                Vertex& v1 = meshData.vertices[idx1];
                Vertex& v2 = meshData.vertices[idx2];

                // Calculate triangle edges
                glm::vec3 edge1 = v1.position - v0.position;
                glm::vec3 edge2 = v2.position - v0.position;

                // Calculate UV deltas
                glm::vec2 deltaUV1 = v1.texCoord - v0.texCoord;
                glm::vec2 deltaUV2 = v2.texCoord - v0.texCoord;

                // Calculate tangent
                float f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);

                // Handle degenerate case
                if (std::isfinite(f)) {
                    glm::vec3 tangent = f * (deltaUV2.y * edge1 - deltaUV1.y * edge2);

                    // Accumulate tangent for all three vertices of the triangle
                    v0.tangent += tangent;
                    v1.tangent += tangent;
                    v2.tangent += tangent;
                }
            }

            // Third pass: normalize accumulated tangents and orthogonalize
            for (size_t i = indexOffset; i < meshData.vertices.size(); ++i) {
                Vertex& vertex = meshData.vertices[i];

                // Gram-Schmidt orthogonalize tangent against normal
                vertex.tangent = vertex.tangent - glm::dot(vertex.tangent, vertex.normal) * vertex.normal;

                // Normalize tangent
                if (glm::length(vertex.tangent) > 0.0f) {
                    vertex.tangent = glm::normalize(vertex.tangent);
                } else {
                    // Fallback tangent if calculation failed
                    vertex.tangent = glm::vec3(1.0f, 0.0f, 0.0f);
                }
            }
        }

        return meshData;
    }
};