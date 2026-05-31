#pragma once
#include <string>
#include <vector>
#include <map>
#include <queue>
#include <limits>
#include <iostream>
#include <future>

#include "structs.hpp"
#include "resources.hpp"
#include "descriptor_sets.hpp"

struct MeshLoadResult {
    std::vector<uint32_t> meshIndices;
    // instance transforms
    std::vector<glm::mat4> transforms;
    // source material ID -> list of mesh indices that use it
    std::vector<uint32_t> materialIds;
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
    // source mesh index -> reflected geometry copy used by mirrored instances
    std::map<uint32_t, uint32_t>        mirrorVariants;

    void init(ResourceManager* resourceManager, DescriptorSet* descriptorSet, uint32_t vertexBufferIndex, uint32_t indexBufferIndex, uint32_t positionBufferIndex) {
        this->resourceManager = resourceManager;
        this->descriptorSet = descriptorSet;
        this->vertexBufferIndex = vertexBufferIndex;
        this->indexBufferIndex = indexBufferIndex;
        this->positionBufferIndex = positionBufferIndex;
    }

    // Allocates GPU buffers for a mesh's geometry, records a Mesh entry, and returns its index.
    uint32_t createMesh(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices,
                        const glm::vec3& bbMin, const glm::vec3& bbMax, const glm::vec3& center,
                        float minRadius, float maxRadius, const std::string& name, const std::string& sourceFile) {
        std::vector<Vertex> verts = vertices; // allocateVariableBuffer takes a mutable buffer
        std::vector<uint32_t> inds = indices;

        uint32_t vertexAllocOffset = descriptorSet->allocateVariableBuffer<Vertex>(verts, vertexBufferIndex);
        VariableBufferAllocation vertexAlloc = descriptorSet->getVariableBufferAllocation(vertexBufferIndex, vertexAllocOffset);

        uint32_t indexAllocOffset = descriptorSet->allocateVariableBuffer<uint32_t>(inds, indexBufferIndex);
        VariableBufferAllocation indexAlloc = descriptorSet->getVariableBufferAllocation(indexBufferIndex, indexAllocOffset);

        std::vector<glm::vec3> cpuPositions, cpuNormals;
        cpuPositions.reserve(verts.size());
        cpuNormals.reserve(verts.size());
        for (const auto& v : verts) {
            cpuPositions.push_back(v.position);
            cpuNormals.push_back(v.normal);
        }

        uint32_t positionAllocOffset = descriptorSet->allocateVariableBuffer<glm::vec3>(cpuPositions, positionBufferIndex);
        VariableBufferAllocation positionAlloc = descriptorSet->getVariableBufferAllocation(positionBufferIndex, positionAllocOffset);

        Mesh mesh;
        mesh.minRadius = minRadius;
        mesh.maxRadius = maxRadius;
        mesh.sourceFile = sourceFile;
        mesh.name = name;
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
        mesh.cpuNormals = std::move(cpuNormals);
        mesh.cpuIndices = std::move(inds);

        meshes.push_back(std::move(mesh));
        return static_cast<uint32_t>(meshes.size() - 1);
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

            uint32_t existingInstance = false;
            for(uint32_t i = 0; i < meshes.size(); i++) {
                if(std::abs(inscribedRadius - meshes[i].minRadius) < 0.001f && std::abs(circumscribedRadius - meshes[i].maxRadius) < 0.001f ) {
                    existingInstance = i;
                    break; 
                }
            }

            uint32_t meshIdx;
            glm::mat4 transform;
            if(existingInstance == 0) {
                meshIdx = createMesh(vertices, indices, bbMin, bbMax, center,
                                     inscribedRadius, circumscribedRadius, entry.shapeName, filePath);
                transform = makeTransform(center);
            }
            else {
                meshIdx = existingInstance;

                // Welded formats (OBJ/PLY) don't store instance transforms and reorder vertices,
                // so we can't assume cpuPositions[i] matches entry.vertices[i]. Recover the
                // transform with correspondence-free multi-start ICP instead (which also detects
                // reflections). Both point sets are centered at their own bbox center, so ICP gives
                // the rotation plus a residual centroid offset; adding `center` puts it in world space.
                if (entry.vertices.size() >= 3 && meshes[meshIdx].cpuPositions.size() >= 3) {
                    std::vector<glm::vec3> instancePositions, instanceNormals;
                    instancePositions.reserve(entry.vertices.size());
                    instanceNormals.reserve(entry.vertices.size());
                    for (const auto& v : entry.vertices) {
                        instancePositions.push_back(v.position);
                        instanceNormals.push_back(v.normal);
                    }

                    float tol = glm::length(bbMax - bbMin) * 0.02f; // 2% of the bbox diagonal

                    // Align the instance to the reference mesh. Reflection is allowed: a mirrored fit
                    // is handled below by a reflected geometry copy (not a negative scale, which the
                    // node system can't carry without shearing).
                    RigidFit fit = icpAlign(meshes[meshIdx].cpuPositions, meshes[meshIdx].cpuNormals,
                                            instancePositions, instanceNormals);

                    if (!fit.valid || fit.rmsd >= tol) {
                        // Not really the same mesh (false instance match) -> no rotation.
                        printf("Instance alignment failed for mesh '%s' (rmsd=%f, tol=%f); placing without rotation.\n",
                               meshes[meshIdx].name.c_str(), fit.rmsd, tol);
                        transform = makeTransform(center);
                    } else if (!fit.mirrored) {
                        transform = makeTransform(center + fit.translation, fit.rotation);
                    } else {
                        // Mirrored instance: render a reflected geometry copy with a proper rotation so
                        // winding/culling stay correct. The first mirror of a source becomes the canonical
                        // mirror mesh (its own geometry is already correctly wound); later mirrors align to it.
                        auto it = mirrorVariants.find(meshIdx);
                        if (it == mirrorVariants.end()) {
                            uint32_t mirrorIdx = createMesh(entry.vertices, entry.indices, bbMin, bbMax, center,
                                                            inscribedRadius, circumscribedRadius,
                                                            meshes[meshIdx].name + "_mirror", filePath);
                            mirrorVariants[meshIdx] = mirrorIdx;
                            meshIdx = mirrorIdx;
                            transform = makeTransform(center); // geometry already in its correct orientation
                        } else {
                            uint32_t mirrorIdx = it->second;
                            RigidFit mf = icpAlign(meshes[mirrorIdx].cpuPositions, meshes[mirrorIdx].cpuNormals,
                                                   instancePositions, instanceNormals, /*allowReflection=*/false);
                            meshIdx = mirrorIdx;
                            transform = (mf.valid && mf.rmsd < tol) ? makeTransform(center + mf.translation, mf.rotation)
                                                                    : makeTransform(center);
                        }
                    }
                } else {
                    transform = makeTransform(center);
                }
            }
            result.meshIndices.push_back(meshIdx);
            result.transforms.push_back(transform);
            result.materialIds.push_back(entry.materialId);
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
