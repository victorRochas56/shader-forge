#pragma once
#include <string>
#include <vector>
#include <map>
#include <queue>
#include <limits>
#include <iostream>
#include <future>
#include <thread>
#include <atomic>

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

    void init(resource::Context* resourceCtx, DescriptorSet* descriptorSet, uint32_t vertexBufferIndex, uint32_t indexBufferIndex, uint32_t positionBufferIndex) {
        this->resourceCtx = resourceCtx;
        this->descriptorSet = descriptorSet;
        this->vertexBufferIndex = vertexBufferIndex;
        this->indexBufferIndex = indexBufferIndex;
        this->positionBufferIndex = positionBufferIndex;
    }

    // Allocates GPU buffers for a mesh's geometry, records a Mesh entry, and returns its index.
    // indices may contain appended LOD ranges (see generateLODs).
    uint32_t createMesh(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices,
                        const std::vector<uint32_t>& LODs,
                        const glm::vec3& bbMin, const glm::vec3& bbMax, const glm::vec3& center,
                        float minRadius, float maxRadius, const std::string& name, const std::string& sourceFile,
                        float importScale = 1.0f, uint32_t sourceEntryIndex = 0) {
        
        std::vector<Vertex> verts = vertices;
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
        mesh.importScale = importScale;
        mesh.sourceEntryIndex = sourceEntryIndex;
        mesh.sourceFile = sourceFile;
        mesh.name = name;
        mesh.vertexAllocationOffset = vertexAllocOffset;
        mesh.vertexOffset = vertexAlloc.offset;
        mesh.vertexCount = vertexAlloc.count;
        mesh.vertexStride = vertexAlloc.stride;
        mesh.indexAllocationOffset = indexAllocOffset;
        mesh.indexOffset = indexAlloc.offset;
        mesh.indexCount = indexAlloc.count;
        mesh.LODs = LODs.empty() ? std::vector<uint32_t>{indexAlloc.count} : LODs;

        const uint32_t lod0IndexCount = mesh.LODs[0];
        float surfaceArea = 0.0f;
        for (uint32_t i = 0; i + 2 < lod0IndexCount; i += 3) {
            const glm::vec3& a = cpuPositions[inds[i]];
            const glm::vec3& b = cpuPositions[inds[i + 1]];
            const glm::vec3& c = cpuPositions[inds[i + 2]];
            surfaceArea += 0.5f * glm::length(glm::cross(b - a, c - a));
        }
        mesh.surfaceArea = surfaceArea;
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
    // importScale is baked into the geometry 
    MeshLoadResult loadMeshFromFile(std::string filePath, float importScale = 1.0f) {
        if (auto it = loadedMeshes.find(filePath); it != loadedMeshes.end()) {
            return it->second;
        }
        MeshLoadResult result;
        std::cout << "Loading mesh from " << filePath << " (import scale " << importScale << ")" << std::endl;
        MeshData meshData = resource::loadMeshFromFile(filePath);

        // Bake the file's unit scale into vertex positions. Uniform scaling leaves normal
        // directions unchanged, so normals need no renormalization.
        if (importScale != 1.0f) {
            for (auto& entry : meshData.entries) {
                for (auto& vertex : entry.vertices) {
                    vertex.position *= importScale;
                }
            }
        }

        const size_t entryCount = meshData.entries.size();
        std::vector<float> inscribedRadii(entryCount);
        std::vector<float> circumscribedRadii(entryCount);
        std::vector<glm::vec3> bbMins(entryCount);
        std::vector<glm::vec3> bbMaxes(entryCount);
        std::vector<glm::vec3> bbCenters(entryCount);

        auto processEntry = [&](size_t idx) {
            std::cout << "processing entry!" << std::endl;
            auto& vertices = meshData.entries[idx].vertices;

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
            bbCenters[idx] = center;
            averagePoint -= center;
            // Center the vertices at origin
            for (auto& vertex : vertices) {
                vertex.position -= center;
            }
            // Store the bounding box centered to match the centered vertices.
            bbMins[idx] = bbMin - center;
            bbMaxes[idx] = bbMax - center;

            float inscribedRadius = std::numeric_limits<float>::max();
            float circumscribedRadius = 0.0f;
            glm::vec3 avg = glm::vec3(averagePoint);
            for (const auto& vertex : vertices) {
                float d = glm::length(vertex.position - avg);
                inscribedRadius = std::min(inscribedRadius, d);
                circumscribedRadius = std::max(circumscribedRadius, d);
            }
            inscribedRadii[idx] = inscribedRadius;
            circumscribedRadii[idx] = circumscribedRadius;
        };
        if (entryCount > 1) {
            unsigned int hw = std::thread::hardware_concurrency();
            size_t workerCount = std::min<size_t>(entryCount, hw ? hw : 4);
            std::atomic<size_t> nextEntry{0};
            auto worker = [&]() {
                for (size_t idx = nextEntry.fetch_add(1); idx < entryCount; idx = nextEntry.fetch_add(1)) {
                    processEntry(idx);
                }
            };
            std::vector<std::thread> workers;
            workers.reserve(workerCount);
            for (size_t w = 0; w < workerCount; ++w) {
                workers.emplace_back(worker);
            }
            for (auto& t : workers) {
                t.join();
            }
        } else if (entryCount == 1) {
            processEntry(0);
        }

        // now we classify each entry as a new mesh or an instance of an existing one
        // for instances recover the placement transform via ICP.
        struct EntryResolution {
            uint32_t meshIdx = 0;
            glm::mat4 transform{1.0f};
            uint32_t refMeshIdx = 0;  // for instances: the matched reference mesh
            bool runIcp = false;      // instance that needs an ICP fit in phase B
            RigidFit fit{};           // filled by phase B
        };
        std::vector<EntryResolution> resolutions(entryCount);

        // (serial): classify entries and create all new base meshes.
        // Afterwards `meshes` is stable for every reference an instance could point at.
        // Mirror meshes are created later
        for (size_t idx = 0; idx < entryCount; ++idx) {
            auto& entry = meshData.entries[idx];
            float inscribedRadius = inscribedRadii[idx];
            float circumscribedRadius = circumscribedRadii[idx];
            const glm::vec3& bbMin = bbMins[idx];
            const glm::vec3& bbMax = bbMaxes[idx];
            const glm::vec3& center = bbCenters[idx];

            uint32_t existingInstance = false;
            for (uint32_t m = 0; m < meshes.size(); m++) {
                if ((std::abs(inscribedRadius - meshes[m].minRadius) < 0.0001f && std::abs(circumscribedRadius - meshes[m].maxRadius) < 0.001f) &&
                    (std::abs(glm::distance(bbMax, bbMin) - glm::distance(meshes[m].boundingBoxMax, meshes[m].boundingBoxMin)) < 0.001f)) {
                    existingInstance = m;
                    break;
                }
            }

            EntryResolution& r = resolutions[idx];
            if (existingInstance == 0) {
                std::cout << "loaded new mesh!" << std::endl;
                generateLODs(entry); // serial on purpose: global Simplify state
                r.meshIdx = createMesh(entry.vertices, entry.indices, entry.LODs, bbMin, bbMax, center,
                                       inscribedRadius, circumscribedRadius, entry.shapeName, filePath, importScale,
                                       static_cast<uint32_t>(idx));
                r.transform = makeTransform(center);
            } else {
                // Default placement for an instance; will be replaces if ICP match is known.
                r.refMeshIdx = existingInstance;
                r.meshIdx = existingInstance;
                r.transform = makeTransform(center);
                // Welded formats (OBJ/PLY) don't store instance transforms and reorder vertices,
                // so we can't assume cpuPositions[i] matches entry.vertices[i].
                // Recover the transform with correspondence-free multi-start ICP instead (which also detects reflections).
                // Both point sets are centered at their own bbox center, so ICP gives the rotation plus a residual centroid offset
                if (entry.vertices.size() >= 3 && meshes[existingInstance].cpuPositions.size() >= 3) {
                    r.runIcp = true;
                }
            }
        }

        // (parallel): run the first icpAlign for every instance against its reference mesh.
        // Read-only on `meshes` and icpAlign is pure, so parallelizeable
        {
            std::vector<size_t> icpEntries;
            for (size_t idx = 0; idx < entryCount; ++idx) {
                if (resolutions[idx].runIcp) icpEntries.push_back(idx);
            }
            auto runIcpFor = [&](size_t idx) {
                auto& entry = meshData.entries[idx];
                std::vector<glm::vec3> instancePositions, instanceNormals;
                instancePositions.reserve(entry.vertices.size());
                instanceNormals.reserve(entry.vertices.size());
                for (const auto& v : entry.vertices) {
                    instancePositions.push_back(v.position);
                    instanceNormals.push_back(v.normal);
                }
                uint32_t ref = resolutions[idx].refMeshIdx;
                // Reflection is allowed here 
                // reflected geometry is a copy (negative scale not supported currently).
                resolutions[idx].fit = icpAlign(meshes[ref].cpuPositions, meshes[ref].cpuNormals, instancePositions, instanceNormals);
            };
            if (icpEntries.size() > 1) {
                unsigned int hw = std::thread::hardware_concurrency();
                size_t workerCount = std::min<size_t>(icpEntries.size(), hw ? hw : 4);
                std::atomic<size_t> nextJob{0};
                auto worker = [&]() {
                    for (size_t j = nextJob.fetch_add(1); j < icpEntries.size(); j = nextJob.fetch_add(1)) {
                        runIcpFor(icpEntries[j]);
                    }
                };
                std::vector<std::thread> workers;
                workers.reserve(workerCount);
                for (size_t w = 0; w < workerCount; ++w) workers.emplace_back(worker);
                for (auto& t : workers) t.join();
            } else if (icpEntries.size() == 1) {
                runIcpFor(icpEntries[0]);
            }
        }

        // (serial): turn each instance's fit into a transform
        // creating mirror-variant meshes in order. 
        // The first mirror of a source becomes the canonical mirror mesh and later mirrors align to it
        for (size_t idx = 0; idx < entryCount; ++idx) {
            EntryResolution& r = resolutions[idx];
            if (!r.runIcp) continue;
            auto& entry = meshData.entries[idx];
            const glm::vec3& bbMin = bbMins[idx];
            const glm::vec3& bbMax = bbMaxes[idx];
            const glm::vec3& center = bbCenters[idx];
            uint32_t refMeshIdx = r.refMeshIdx;
            const RigidFit& fit = r.fit;
            float tol = glm::length(bbMax - bbMin) * 0.02f; // 2% of the bbox diagonal

            if (!fit.valid || fit.rmsd >= tol) {
                // Not really the same mesh (false instance match) -> no rotation.
                printf("Instance alignment failed for mesh '%s' (rmsd=%f, tol=%f); placing without rotation.\n",
                       meshes[refMeshIdx].name.c_str(), fit.rmsd, tol);
                fflush(stdout);
                r.transform = makeTransform(center);
            } else if (!fit.mirrored) {
                r.transform = makeTransform(center + fit.translation, fit.rotation);
            } else {
                // Mirrored instance: render a reflected geometry copy with a proper rotation so
                // winding/culling stay correct.
                auto it = mirrorVariants.find(refMeshIdx);
                if (it == mirrorVariants.end()) {
                    generateLODs(entry);
                    uint32_t mirrorIdx = createMesh(entry.vertices, entry.indices, entry.LODs, bbMin, bbMax, center,
                                                    inscribedRadii[idx], circumscribedRadii[idx],
                                                    meshes[refMeshIdx].name + "_mirror", filePath, importScale,
                                                    static_cast<uint32_t>(idx));
                    mirrorVariants[refMeshIdx] = mirrorIdx;
                    r.meshIdx = mirrorIdx;
                    r.transform = makeTransform(center); // geometry already in its correct orientation
                } else {
                    uint32_t mirrorIdx = it->second;
                    std::vector<glm::vec3> instancePositions, instanceNormals;
                    instancePositions.reserve(entry.vertices.size());
                    instanceNormals.reserve(entry.vertices.size());
                    for (const auto& v : entry.vertices) {
                        instancePositions.push_back(v.position);
                        instanceNormals.push_back(v.normal);
                    }
                    RigidFit mf = icpAlign(meshes[mirrorIdx].cpuPositions, meshes[mirrorIdx].cpuNormals,
                                           instancePositions, instanceNormals, /*allowReflection=*/false);
                    r.meshIdx = mirrorIdx;
                    r.transform = (mf.valid && mf.rmsd < tol) ? makeTransform(center + mf.translation, mf.rotation)
                                                              : makeTransform(center);
                }
            }
        }

        // Final assembly
        for (size_t idx = 0; idx < entryCount; ++idx) {
            auto& entry = meshData.entries[idx];
            result.meshIndices.push_back(resolutions[idx].meshIdx);
            result.transforms.push_back(resolutions[idx].transform);
            result.materialIds.push_back(entry.materialId);
            if (result.materialNames.find(entry.materialId) == result.materialNames.end()) {
                result.materialNames[entry.materialId] = entry.materialName;
            }
        }
 
        loadedMeshes[filePath] = result;
        return result;
    }

    // Fast scene-load path: build a single mesh from a specific source-file entry
    // Meshes are deduped by (file, entry) so the instances and mirror groups that shared one mesh on import share one again here.
    uint32_t loadSceneMesh(const std::string& filePath, uint32_t entryIndex, const std::string& meshName,
                           float importScale = 1.0f) {
        // Persists across scene clears so reloads reuse meshes; scale keyed since it bakes geometry.
        std::string key = filePath + "#" + std::to_string(entryIndex) + "@" + std::to_string(importScale);
        if (auto it = sceneMeshCache.find(key); it != sceneMeshCache.end()) {
            return it->second;
        }

        // Parse the source file once and keep it around for the rest of this scene load.
        auto rawIt = rawMeshDataCache.find(filePath);
        if (rawIt == rawMeshDataCache.end()) {
            rawIt = rawMeshDataCache.emplace(filePath, resource::loadMeshFromFile(filePath)).first;
        }
        const MeshData& meshData = rawIt->second;
        if (entryIndex >= meshData.entries.size()) {
            throw std::runtime_error("scene references mesh entry " + std::to_string(entryIndex) +
                                     " out of range in " + filePath);
        }

        // Work on copies so the cached raw data stays pristine for sibling entries.
        std::vector<Vertex> vertices = meshData.entries[entryIndex].vertices;
        std::vector<uint32_t> indices = meshData.entries[entryIndex].indices;

        // Bake the file's unit scale, then mirror the import path's per-entry processing exactly so
        // the geometry lines up with the saved node transforms.
        if (importScale != 1.0f) {
            for (auto& v : vertices) v.position *= importScale;
        }

        glm::vec3 bbMin(std::numeric_limits<float>::max());
        glm::vec3 bbMax(std::numeric_limits<float>::lowest());
        glm::dvec3 averageAccum{0, 0, 0};
        for (const auto& v : vertices) {
            bbMin = glm::min(bbMin, v.position);
            bbMax = glm::max(bbMax, v.position);
            averageAccum += glm::dvec3(v.position);
        }
        glm::vec3 center = (bbMax + bbMin) * 0.5f;
        glm::vec3 averagePoint = vertices.empty() ? glm::vec3(0.0f)
                               : glm::vec3(averageAccum / static_cast<double>(vertices.size())) - center;

        // Center geometry at origin (import does the same; node transforms encode the offset).
        for (auto& v : vertices) v.position -= center;
        bbMin -= center;
        bbMax -= center;

        float inscribedRadius = std::numeric_limits<float>::max();
        float circumscribedRadius = 0.0f;
        for (const auto& v : vertices) {
            float d = glm::length(v.position - averagePoint);
            inscribedRadius = std::min(inscribedRadius, d);
            circumscribedRadius = std::max(circumscribedRadius, d);
        }

        std::string name = meshName.empty() ? meshData.entries[entryIndex].shapeName : meshName;
        std::vector<uint32_t> LODs;
        generateLODs(vertices, indices, LODs);
        uint32_t meshIdx = createMesh(vertices, indices, LODs, bbMin, bbMax, center, inscribedRadius,
                                      circumscribedRadius, name, filePath, importScale, entryIndex);
        sceneMeshCache[key] = meshIdx;
        return meshIdx;
    }

    // Drop the heavy parsed-file cache after load; sceneMeshCache is kept so reloads reuse meshes.
    void clearSceneMeshLoadCache() {
        rawMeshDataCache.clear();
    }

    // colorSpace only matters the first time a source image is seen, when it drives the KTX encode.
    // Auto derives it from `format`; normal maps have to say so explicitly since they are linear
    // but need the encoder's RDO turned off.
    uint32_t loadTextureFromFile(std::string filePath, vk::Format format = vk::Format::eR8G8B8A8Srgb,
                                 textureconv::ColorSpace colorSpace = textureconv::ColorSpace::Auto) {
        if (loadedTextures.contains(filePath)) {
            return loadedTextures[filePath];
        }

        resource::LoadedTexture texture = resource::loadTextureFromFile(*resourceCtx, filePath, format, colorSpace);
        uint32_t allocIndex = descriptorSet->allocateTexture(std::move(texture.image), std::move(texture.memory), std::move(texture.view), filePath, false, texture.width,
                                                             texture.height, texture.ktxPath, texture.mipLevels);
        loadedTextures[filePath] = allocIndex;
        return allocIndex;
    }

    uint32_t loadCubemapFromFile(std::string posX, std::string posY, std::string posZ, std::string negX, std::string negY, std::string negZ) {
        std::string cubemapKey = posX + "|" + negX + "|" + posY + "|" + negY + "|" + posZ + "|" + negZ;

        if (loadedCubemaps.contains(cubemapKey)) {
            return loadedCubemaps[cubemapKey];
        }

        resource::LoadedTexture texture = resource::loadCubeMapFromFile(*resourceCtx, posX, negX, posY, negY, posZ, negZ);
        uint32_t allocIndex = descriptorSet->allocateTexture(std::move(texture.image), std::move(texture.memory), std::move(texture.view), cubemapKey, true, texture.width,
                                                             texture.height, texture.ktxPath, texture.mipLevels);
        loadedCubemaps[cubemapKey] = allocIndex;
        return allocIndex;
    }

    // Encodes any missing/stale .ktx2 caches for a batch of textures in parallel. Loading them one
    // at a time still works, it just serialises every encode on a cold import.
    void prewarmTextureCache(const std::vector<textureconv::Job>& jobs) { textureconv::convertBatch(jobs); }

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
    std::map<std::string, MeshData> rawMeshDataCache;  // transient parsed source files, freed after load
    std::map<std::string, uint32_t> sceneMeshCache;    // "file#entry@scale" -> meshIdx, persists across loads

    resource::Context* resourceCtx = nullptr;
    DescriptorSet* descriptorSet = nullptr;
    uint32_t vertexBufferIndex = 0;
    uint32_t indexBufferIndex = 0;
    uint32_t positionBufferIndex = 0;
};
