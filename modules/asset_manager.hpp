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

        const size_t entryCount = meshData.entries.size();
        std::vector<float> inscribedRadii(entryCount);
        std::vector<float> circumscribedRadii(entryCount);
        std::vector<glm::vec3> bbMins(entryCount);
        std::vector<glm::vec3> bbMaxes(entryCount);
        std::vector<glm::vec3> bbCenters(entryCount);

        // Each entry is independent: it only touches its own vertices and writes its
        // results to a fixed index, so the per-entry work can run on worker threads.
        auto processEntry = [&](size_t idx) {
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

        // The remaining per-entry work: classify each entry as a new mesh or an instance of an
        // existing one, and for instances recover the placement transform via ICP. This can't be
        // a flat parallel-for: instance dedup reads `meshes` while createMesh grows it (so
        // classification is order-dependent), createMesh allocates GPU buffers (not thread-safe),
        // and mirror-variant creation is order-dependent. Only icpAlign is both expensive and
        // pure, so we split into three phases and parallelize just the ICP.
        struct EntryResolution {
            uint32_t meshIdx = 0;
            glm::mat4 transform{1.0f};
            uint32_t refMeshIdx = 0;  // for instances: the matched reference mesh
            bool runIcp = false;      // instance that needs an ICP fit in phase B
            RigidFit fit{};           // filled by phase B
        };
        std::vector<EntryResolution> resolutions(entryCount);

        // Phase A (serial): classify entries and create all new base meshes. Afterwards `meshes`
        // is stable for every reference an instance could point at. Mirror meshes are created
        // later in phase C, but they share their source's radii so they never change which mesh
        // the dedup scan matches (it breaks on the earliest match).
        for (size_t idx = 0; idx < entryCount; ++idx) {
            auto& entry = meshData.entries[idx];
            float inscribedRadius = inscribedRadii[idx];
            float circumscribedRadius = circumscribedRadii[idx];
            const glm::vec3& bbMin = bbMins[idx];
            const glm::vec3& bbMax = bbMaxes[idx];
            const glm::vec3& center = bbCenters[idx];

            uint32_t existingInstance = false;
            for (uint32_t m = 0; m < meshes.size(); m++) {
                if (std::abs(inscribedRadius - meshes[m].minRadius) < 0.001f && std::abs(circumscribedRadius - meshes[m].maxRadius) < 0.001f) {
                    existingInstance = m;
                    break;
                }
            }

            EntryResolution& r = resolutions[idx];
            if (existingInstance == 0) {
                r.meshIdx = createMesh(entry.vertices, entry.indices, bbMin, bbMax, center,
                                       inscribedRadius, circumscribedRadius, entry.shapeName, filePath);
                r.transform = makeTransform(center);
            } else {
                // Default placement for an instance; refined in phase C once its ICP fit is known.
                r.refMeshIdx = existingInstance;
                r.meshIdx = existingInstance;
                r.transform = makeTransform(center);
                // Welded formats (OBJ/PLY) don't store instance transforms and reorder vertices,
                // so we can't assume cpuPositions[i] matches entry.vertices[i]. Recover the
                // transform with correspondence-free multi-start ICP instead (which also detects
                // reflections). Both point sets are centered at their own bbox center, so ICP gives
                // the rotation plus a residual centroid offset; adding `center` puts it in world space.
                if (entry.vertices.size() >= 3 && meshes[existingInstance].cpuPositions.size() >= 3) {
                    r.runIcp = true;
                }
            }
        }

        // Phase B (parallel): run the first (expensive) icpAlign for every instance against its
        // now-fixed reference mesh. Read-only on `meshes` and icpAlign is pure, so this fans out
        // safely across the bounded worker pool.
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
                // Reflection is allowed here; a mirrored fit is handled in phase C with a reflected
                // geometry copy (not a negative scale, which the node system can't carry without shearing).
                resolutions[idx].fit = icpAlign(meshes[ref].cpuPositions, meshes[ref].cpuNormals,
                                                instancePositions, instanceNormals);
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

        // Phase C (serial): turn each instance's fit into a transform, creating mirror-variant
        // meshes in order. The first mirror of a source becomes the canonical mirror mesh (its own
        // geometry is already correctly wound); later mirrors align to it, so this must stay serial.
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
                r.transform = makeTransform(center);
            } else if (!fit.mirrored) {
                r.transform = makeTransform(center + fit.translation, fit.rotation);
            } else {
                // Mirrored instance: render a reflected geometry copy with a proper rotation so
                // winding/culling stay correct.
                auto it = mirrorVariants.find(refMeshIdx);
                if (it == mirrorVariants.end()) {
                    uint32_t mirrorIdx = createMesh(entry.vertices, entry.indices, bbMin, bbMax, center,
                                                    inscribedRadii[idx], circumscribedRadii[idx],
                                                    meshes[refMeshIdx].name + "_mirror", filePath);
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

        // Final assembly (serial, in entry order).
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
