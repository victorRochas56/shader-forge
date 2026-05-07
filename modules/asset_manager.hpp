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

struct MeshLoadResult {
    std::vector<uint32_t> meshIndices;
    // source material ID -> list of mesh indices that use it
    std::map<int, std::vector<uint32_t>> meshesByMaterial;
    // source material ID -> material name from the file
    std::map<int, std::string> materialNames;
};

class AssetManager {
  public:
    std::vector<Mesh>                   meshes;
    std::queue<uint32_t>                freeMeshes;
    std::map<std::string, uint32_t>     loadedTextures;
    std::map<std::string, uint32_t>     loadedCubemaps;
    std::map<std::string, MeshLoadResult> loadedMeshes;

    void init(ResourceManager* resourceManager, DescriptorSet* descriptorSet, uint32_t vertexBufferIndex, uint32_t indexBufferIndex, uint32_t positionBufferIndex) {
        this->resourceManager = resourceManager;
        this->descriptorSet = descriptorSet;
        this->vertexBufferIndex = vertexBufferIndex;
        this->indexBufferIndex = indexBufferIndex;
        this->positionBufferIndex = positionBufferIndex;
    }

    // Loads an OBJ and returns mesh indices plus material grouping info.
    // Each mesh is a single draw unit with its own vertex/index data.
    MeshLoadResult loadMeshFromFile(std::string filePath) {
        if (auto it = loadedMeshes.find(filePath); it != loadedMeshes.end()) {
            return it->second;
        }

        MeshLoadResult result;

#if DEBUG == 1
        std::cout << "Loading mesh from " << filePath << std::endl;
#endif
        MeshData meshData = resourceManager->loadMeshFromFile(filePath);

        for (auto& entry : meshData.entries) {
            auto& vertices = entry.vertices;
            auto& indices = entry.indices;

            glm::vec3 bbMin(std::numeric_limits<float>::max());
            glm::vec3 bbMax(std::numeric_limits<float>::lowest());

            glm::dvec3 averagePointAccum{0,0,0};
            for (const auto& vertex : vertices) {
                bbMin = glm::min(bbMin, vertex.position);
                bbMax = glm::max(bbMax, vertex.position);
                averagePointAccum += glm::dvec3(vertex.position);
            }
            glm::vec3 averagePoint = glm::vec3(averagePointAccum / static_cast<double>(vertices.size()));

            glm::vec3 center = (bbMax + bbMin) * 0.5f;
            averagePoint -= center;
            printf("AVERAGE POINT OF MESH : %f, %f, %f \n",averagePoint.x,averagePoint.y,averagePoint.z);
            // Center the vertices at origin
            for (auto& vertex : vertices) {
                vertex.position -= center;
            }

            float inscribedRadius = std::numeric_limits<float>::max();
            float circumscribedRadius = 0.0f;
            glm::vec3 avg = glm::vec3(averagePoint);
            for (const auto& vertex : vertices) {
                float d = glm::length(vertex.position - avg);
                inscribedRadius = std::min(inscribedRadius, d);
                circumscribedRadius = std::max(circumscribedRadius, d);
            }
            printf("INSCRIBED SPHERE RADIUS    : %f\n", inscribedRadius);
            printf("CIRCUMSCRIBED SPHERE RADIUS: %f\n", circumscribedRadius);
            // Update bounding box to be centered
            bbMin -= center;
            bbMax -= center;

            uint32_t vertexAllocOffset = descriptorSet->allocateVariableBuffer<Vertex>(vertices, vertexBufferIndex);
            VariableBufferAllocation vertexAlloc = descriptorSet->getVariableBufferAllocation(vertexBufferIndex, vertexAllocOffset);

            uint32_t indexAllocOffset = descriptorSet->allocateVariableBuffer<uint32_t>(indices, indexBufferIndex);
            VariableBufferAllocation indexAlloc = descriptorSet->getVariableBufferAllocation(indexBufferIndex, indexAllocOffset);


            std::vector<glm::vec3> cpuPositions;
            cpuPositions.reserve(vertices.size());
            for (const auto& v : vertices) {
                cpuPositions.push_back(v.position);
            }

            uint32_t positionAllocOffset = descriptorSet->allocateVariableBuffer<glm::vec3>(cpuPositions, positionBufferIndex);
            VariableBufferAllocation positionAlloc = descriptorSet->getVariableBufferAllocation(positionBufferIndex, positionAllocOffset);


            Mesh mesh;
            mesh.sourceFile = filePath;
            mesh.name = entry.shapeName;
            mesh.vertexAllocationOffset = vertexAllocOffset;
            mesh.vertexOffset = vertexAlloc.offset;
            mesh.vertexCount = vertexAlloc.count;
            mesh.vertexStride = vertexAlloc.stride;
            mesh.indexAllocationOffset = indexAllocOffset;
            mesh.indexOffset = indexAlloc.offset;
            mesh.indexCount = indexAlloc.count;
            mesh.boundingBoxMin = bbMin;
            mesh.boundingBoxMax = bbMax;
            mesh.center = center;
            mesh.positionAllocationOffset = positionAllocOffset;
            mesh.positionOffset = positionAlloc.offset;
            mesh.positionCount = positionAlloc.count;
            mesh.cpuPositions = std::move(cpuPositions);
            mesh.cpuIndices = indices;

            meshes.push_back(std::move(mesh));
            uint32_t meshIdx = static_cast<uint32_t>(meshes.size() - 1);
            result.meshIndices.push_back(meshIdx);
            result.meshesByMaterial[entry.materialId].push_back(meshIdx);
            if (result.materialNames.find(entry.materialId) == result.materialNames.end()) {
                result.materialNames[entry.materialId] = entry.materialName;
            }
        }

        loadedMeshes[filePath] = result;
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
        uint32_t allocIndex = descriptorSet->allocateTexture(std::move(image), std::move(memory), std::move(view), cubemapKey, true, width, height);
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
    uint32_t positionBufferIndex = 0;
};
