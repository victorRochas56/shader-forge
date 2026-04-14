#pragma once
#include <string>
#include <vector>
#include <map>
#include <queue>
#include <limits>
#include <iostream>

#include "structs.hpp"
#include "resources.hpp"
#include "descriptor_sets.hpp"

class AssetManager {
  public:
    std::vector<Mesh>                   meshes;
    std::queue<uint32_t>                freeMeshes;
    std::map<std::string, uint32_t>     loadedTextures;
    std::map<std::string, uint32_t>     loadedCubemaps;

    void init(ResourceManager* resourceManager, DescriptorSet* descriptorSet, uint32_t vertexBufferIndex, uint32_t indexBufferIndex) {
        this->resourceManager = resourceManager;
        this->descriptorSet = descriptorSet;
        this->vertexBufferIndex = vertexBufferIndex;
        this->indexBufferIndex = indexBufferIndex;
    }

    // Loads an OBJ and returns a vector of mesh indices (one per shape/material group).
    // Each mesh is a single draw unit with its own vertex/index data.
    std::vector<uint32_t> loadMeshFromFile(std::string filePath) {
        std::vector<uint32_t> result;

#if DEBUG == 1
        std::cout << "Loading mesh from " << filePath << std::endl;
#endif
        auto meshData = resourceManager->loadMeshFromFile(filePath);

        for (auto& entry : meshData.entries) {
            auto& vertices = entry.vertices;
            auto& indices = entry.indices;

            glm::vec3 bbMin(std::numeric_limits<float>::max());
            glm::vec3 bbMax(std::numeric_limits<float>::lowest());
            for (const auto& vertex : vertices) {
                bbMin = glm::min(bbMin, vertex.position);
                bbMax = glm::max(bbMax, vertex.position);
            }

            glm::vec3 center = (bbMax + bbMin) * 0.5f;
            // Center the vertices at origin
            for (auto& vertex : vertices) {
                vertex.position -= center;
            }
            // Update bounding box to be centered
            bbMin -= center;
            bbMax -= center;

            uint32_t vertexAllocIndex = descriptorSet->allocateVariableBuffer<Vertex>(vertices, vertexBufferIndex);
            VariableBufferAllocation vertexAlloc = descriptorSet->getVariableBufferAllocation(vertexBufferIndex, vertexAllocIndex);

            uint32_t indexAllocIndex = descriptorSet->allocateVariableBuffer<uint32_t>(indices, indexBufferIndex);
            VariableBufferAllocation indexAlloc = descriptorSet->getVariableBufferAllocation(indexBufferIndex, indexAllocIndex);

            std::vector<glm::vec3> cpuPositions;
            cpuPositions.reserve(vertices.size());
            for (const auto& v : vertices) {
                cpuPositions.push_back(v.position);
            }

            Mesh mesh;
            mesh.sourceFile = filePath;
            mesh.vertexAllocationIndex = vertexAllocIndex;
            mesh.vertexOffset = vertexAlloc.offset;
            mesh.vertexCount = vertexAlloc.count;
            mesh.vertexStride = vertexAlloc.stride;
            mesh.indexAllocationIndex = indexAllocIndex;
            mesh.indexOffset = indexAlloc.offset;
            mesh.indexCount = indexAlloc.count;
            mesh.boundingBoxMin = bbMin;
            mesh.boundingBoxMax = bbMax;
            mesh.center = center;
            mesh.cpuPositions = std::move(cpuPositions);
            mesh.cpuIndices = indices;

            meshes.push_back(std::move(mesh));
            result.push_back(static_cast<uint32_t>(meshes.size() - 1));
        }

        return result;
    }

    uint32_t loadTextureFromFile(std::string filePath, vk::Format format = vk::Format::eR8G8B8A8Srgb) {
        if (loadedTextures.contains(filePath)) {
            return loadedTextures[filePath];
        }

        auto [image, memory, view, texW, texH] = resourceManager->loadTextureFromFile(filePath, format);
        uint32_t allocIndex = descriptorSet->allocateTexture(std::move(image), std::move(memory), std::move(view), filePath, false, texW, texH);
        loadedTextures[filePath] = allocIndex;
        return allocIndex;
    }

    uint32_t loadCubemapFromFile(std::string posX, std::string posY, std::string posZ, std::string negX, std::string negY, std::string negZ, uint32_t width = 2048,
                                 uint32_t height = 2048) {
        std::string cubemapKey = posX + "|" + negX + "|" + posY + "|" + negY + "|" + posZ + "|" + negZ;

        if (loadedCubemaps.contains(cubemapKey)) {
            return loadedCubemaps[cubemapKey];
        }

        auto [image, memory, view] = resourceManager->loadCubeMapFromFile(posX, negX, posY, negY, posZ, negZ, width, height);
        uint32_t allocIndex = descriptorSet->allocateTexture(std::move(image), std::move(memory), std::move(view), cubemapKey, true);
        loadedCubemaps[cubemapKey] = allocIndex;
        return allocIndex;
    }

    std::string getTexturePathFromIndex(uint32_t textureIndex) {
        for (const auto& [path, index] : loadedTextures) {
            if (index == textureIndex) {
                return path;
            }
        }
        return "";
    }

    std::string getCubemapPathFromIndex(uint32_t cubemapIndex) {
        for (const auto& [path, index] : loadedCubemaps) {
            if (index == cubemapIndex) {
                return path;
            }
        }
        return "";
    }

    std::vector<Mesh>& getMeshes() { return meshes; }

  private:
    ResourceManager* resourceManager = nullptr;
    DescriptorSet* descriptorSet = nullptr;
    uint32_t vertexBufferIndex = 0;
    uint32_t indexBufferIndex = 0;
};
