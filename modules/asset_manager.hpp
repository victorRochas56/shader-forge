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
    std::vector<SubMesh>                subMeshes;
    std::queue<uint32_t>                freeSubMeshes;
    std::map<std::string, uint32_t>     loadedTextures;
    std::map<std::string, uint32_t>     loadedCubemaps;

    void init(ResourceManager* resourceManager, DescriptorSet* descriptorSet, uint32_t vertexBufferIndex, uint32_t indexBufferIndex) {
        this->resourceManager = resourceManager;
        this->descriptorSet = descriptorSet;
        this->vertexBufferIndex = vertexBufferIndex;
        this->indexBufferIndex = indexBufferIndex;
    }

    uint32_t loadMeshFromFile(std::string filePath) {
        for (int i = 0; i < meshes.size(); i++) {
            if (meshes[i].sourceFile == filePath) {
                return i;
            }
        }
#if DEBUG == 1
        std::cout << "Loading mesh from " << filePath << std::endl;
#endif
        auto meshData = resourceManager->loadMeshFromFile(filePath);
        Mesh mainMesh{.sourceFile = filePath, .originalMaterialIds = meshData.materialIds, .originalMaterialNames = meshData.materialNames};

        glm::vec3 bbMin(std::numeric_limits<float>::max());
        glm::vec3 bbMax(std::numeric_limits<float>::lowest());

        for (const auto& submesh : meshData.subMeshes) {
            for (const auto& vertex : submesh) {
                bbMin = glm::min(bbMin, vertex.position);
                bbMax = glm::max(bbMax, vertex.position);
            }
        }
        mainMesh.boundingBoxMin = bbMin;
        mainMesh.boundingBoxMax = bbMax;

        for (size_t i = 0; i < meshData.subMeshes.size(); i++) {
            auto& vertices = meshData.subMeshes[i];
            auto& indices = meshData.subMeshIndices[i];

            glm::vec3 subBBMin(std::numeric_limits<float>::max());
            glm::vec3 subBBMax(std::numeric_limits<float>::lowest());
            for (const auto& vertex : vertices) {
                subBBMin = glm::min(subBBMin, vertex.position);
                subBBMax = glm::max(subBBMax, vertex.position);
            }

            uint32_t vertexAllocIndex = descriptorSet->allocateVariableBuffer<Vertex>(vertices, vertexBufferIndex);
            VariableBufferAllocation vertexAlloc = descriptorSet->getVariableBufferAllocation(vertexBufferIndex, vertexAllocIndex);

            uint32_t indexAllocIndex = descriptorSet->allocateVariableBuffer<uint32_t>(indices, indexBufferIndex);
            VariableBufferAllocation indexAlloc = descriptorSet->getVariableBufferAllocation(indexBufferIndex, indexAllocIndex);

            std::vector<glm::vec3> cpuPositions;
            cpuPositions.reserve(vertices.size());
            for (const auto& v : vertices) {
                cpuPositions.push_back(v.position);
            }

            SubMesh subMesh = {.vertexAllocationIndex = vertexAllocIndex,
                               .vertexOffset = vertexAlloc.offset,
                               .vertexCount = vertexAlloc.count,
                               .vertexStride = vertexAlloc.stride,
                               .indexAllocationIndex = indexAllocIndex,
                               .indexOffset = indexAlloc.offset,
                               .indexCount = indexAlloc.count,
                               .boundingBoxMin = subBBMin,
                               .boundingBoxMax = subBBMax,
                               .cpuPositions = std::move(cpuPositions),
                               .cpuIndices = indices};

            subMeshes.push_back(subMesh);
            mainMesh.subMeshes.push_back(subMeshes.size() - 1);
        }

        meshes.push_back(mainMesh);
        return meshes.size() - 1;
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