#pragma once
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1 // for raii
#define VULKAN_HPP_NO_CONSTRUCTORS 1         // for structs constructors
#include <algorithm>
#include <array>
#include <bindless_resources.hpp>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <devices.hpp>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
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
    BindlessResourceManager* resourceManager;

  public:
    MeshLoader(BindlessResourceManager& manager) : resourceManager(&manager) {}

    // Load mesh from vertex data
    Mesh loadMesh(const std::vector<Vertex>& vertices, const std::string& texturePath, glm::vec3 position = glm::vec3(0.0f), glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
                  glm::vec3 scale = glm::vec3(1.0f)) {

        // Allocate vertex buffer space
        auto vertexInfo = resourceManager->allocateVertexBuffer(vertices.data(), vertices.size() * sizeof(Vertex), static_cast<uint32_t>(vertices.size()), sizeof(Vertex));

        // Load texture
        uint32_t textureIndex;
        if (!texturePath.empty()) {
            auto textureData = loadTextureFromFile(texturePath);
            textureIndex = resourceManager->loadTexture(textureData.data, textureData.width, textureData.height);
        } else {
            textureIndex = resourceManager->getWhiteTextureIndex(); // Use default
        }

        // Create model matrix
        uint32_t modelMatrixIndex = resourceManager->allocateModelMatrixBuffer(position, rotation, scale);

        // Use default sampler (or create custom one)
        uint32_t samplerIndex = resourceManager->getDefaultSamplerIndex();

        // Create mesh object
        return Mesh{.vertexAllocationIndex = vertexInfo.allocationIndex,
                    .vertexOffset = vertexInfo.offset,
                    .vertexCount = vertexInfo.vertexCount,
                    .vertexStride = sizeof(Vertex),
                    .modelMatrixIndex = modelMatrixIndex,
                    .textureIndex = textureIndex,
                    .samplerIndex = samplerIndex};
    }

    // load from file (OBJ, GLTF, etc.)
    Mesh loadFromFile(const std::string& meshPath, const std::string& texturePath = "", glm::vec3 position = glm::vec3(0.0f),
                      glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f), glm::vec3 scale = glm::vec3(1.0f)) {

        auto meshData = loadMeshFromFile(meshPath);

        return loadMesh(meshData.vertices, texturePath, position, rotation, scale);
    }

    // Update mesh transform
    void updateMeshTransform(Mesh& mesh, glm::vec3 position, glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f), glm::vec3 scale = glm::vec3(1.0f)) {

        resourceManager->updateModelMatrix(mesh.modelMatrixIndex, position, rotation, scale);
    }

    // Update mesh vertices (for dynamic meshes)
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

        // Don't free texture/sampler if they might be shared
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

    TextureData loadTextureFromFile(const std::string& path) {
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
        }
        return meshData;
    }
};