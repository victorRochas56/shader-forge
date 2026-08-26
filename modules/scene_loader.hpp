#pragma once
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "node_ops.hpp"
#include "scene.hpp"
#include "utils.hpp"

/*
parses scene data and saves it to a file
and
parses a file and loads a scene from it
*/

class SceneLoader {
  public:
    // clearScene/loadScene need bindless + the renderer-owned buffer indices
    // to free GPU slots. The SceneGraph re-creates root from resources cached
    // at its own init(), so the loader doesn't need a renderer pointer.
    void clearScene(Scene& scene, BindlessSystem& bindless,
                    const RenderBuffers& buffers,
                    uint32_t modelMatrixBufferIndex,
                    uint32_t lightBufferIndex) {
        clearSceneInternal(scene, bindless, buffers, modelMatrixBufferIndex, lightBufferIndex);
    }

    void saveScene(std::string filePath, Scene& scene) {

        ofs.open(filePath, std::ios::out | std::ios::trunc); // clears the file
        savedMaterialIDs.clear();

        // first collect all unique materials used by nodes
        Node& rootNode = scene.sceneGraph.getRootNode();
        auto& nodes = scene.sceneGraph.getNodes();
        uint32_t child = rootNode.firstChild;
        while (child != 0) {
            collectMaterials(nodes[child], scene);
            child = nodes[child].nextSibling;
        }
        // Templates carry their own material references — a template whose material no longer
        // appears anywhere in the graph still has to find it again on load.
        for (const auto& [name, tmpl] : scene.templates) {
            for (const Node& node : tmpl.nodes) {
                if (node.meshIndex == MAX_MESHES || node.materialIndex == 0xFFFFFFFF) continue;
                savedMaterialIDs.insert(scene.getMaterials()[node.materialIndex].materialID);
            }
        }
        writeMaterials(scene);

        // Meshes go before the nodes: it is node parsing that builds the meshes, so their cached
        // LOD chains have to already be in hand by the time it runs.
        writeMeshes(scene.assetManager);

        writeTemplates(scene);
        // then write all nodes
        child = rootNode.firstChild;
        while (child != 0) {
            writeNodes(nodes[child], scene, 0);
            child = nodes[child].nextSibling;
        }
        ofs.close();
    }

    void loadScene(std::string filePath, Scene& scene, BindlessSystem& bindless,
                   const RenderBuffers& buffers,
                   uint32_t modelMatrixBufferIndex,
                   uint32_t lightBufferIndex) {
        std::cout << "Loading scene from: " << filePath << std::endl;

        std::ifstream ifs(filePath);
        if (!ifs.is_open()) {
            std::cerr << "Failed to open scene file: " << filePath << std::endl;
            return;
        }

        // The per-material loads below are serial, so on a cold import every missing .ktx2 would be
        // encoded one after another. Scan the file for texture paths first and encode them in parallel.
        prewarmTextureCache(filePath, scene);

        // Clear existing scene (except root node)
        clearSceneInternal(scene, bindless, buffers, modelMatrixBufferIndex, lightBufferIndex);

        // Maps to track loaded resources
        std::unordered_map<uint32_t, uint32_t> materialIDToIndex; // materialID -> material index in scene
        meshLODCache.clear();

        std::string line;
        while (std::getline(ifs, line)) {
            trim(line);

            if (line == "Materials {") {
                parseMaterialsSection(ifs, scene, materialIDToIndex);
            } else if (line == "Mesh {") {
                parseMeshSection(ifs);
            } else if (line == "Template {") {
                // Has to consume its own Node blocks — left to the branch below they would load as
                // real scene nodes, duplicating whatever the template was made from.
                parseTemplateSection(ifs, scene, bindless, buffers, lightBufferIndex, materialIDToIndex);
            } else if (line == "Node {") {
                parseNode(ifs, scene, bindless, buffers, lightBufferIndex, SceneGraph::ROOT_INDEX, materialIDToIndex);
            }
        }

        ifs.close();
        scene.assetManager.clearSceneMeshLoadCache(); // free the transient raw-file cache used during load
        std::cout << "Scene loaded successfully!" << std::endl;
    }

  private:
    std::fstream ofs;
    std::unordered_set<uint32_t> savedMaterialIDs;
    // Saved LOD chains for the scene being loaded, keyed the same way loadSceneMesh keys a mesh.
    std::map<std::string, PrecomputedLODs> meshLODCache;

    // One key for both sides. The scale goes through std::to_string on each so the Node section's
    // MeshScale and the Mesh section's Source field — both written from the same importScale with
    // stream defaults — normalise to the same text.
    static std::string meshLODKey(const std::string& path, uint32_t entry, float scale) {
        return path + "#" + std::to_string(entry) + "@" + std::to_string(scale);
    }

    // LODIndices runs to one value per LOD corner — hundreds of thousands on a dense mesh — so it
    // is walked in place rather than through split(), which would allocate a string apiece.
    static void parseUintList(const std::string& value, std::vector<uint32_t>& out) {
        const char* p = value.c_str();
        while (*p != '\0') {
            char* end = nullptr;
            unsigned long parsed = std::strtoul(p, &end, 10);
            if (end == p) break; // no digits left
            out.push_back(static_cast<uint32_t>(parsed));
            p = end;
            while (*p == ';' || *p == ' ' || *p == ',') p++;
        }
    }

    // Mesh sections exist purely to carry the precomputed LOD chain; the geometry itself is still
    // rebuilt from the source file. Anything that fails to describe a usable chain is dropped here
    // rather than stored, so loadSceneMesh simply falls back to simplifying.
    void parseMeshSection(std::ifstream& ifs) {
        std::string line, key, value, cacheKey;
        PrecomputedLODs lods;

        while (std::getline(ifs, line)) {
            trim(line);
            if (line == "}") break;
            if (!parseKeyValue(line, key, value)) continue;

            if (key == "Source") {
                auto parts = split(value, ';');
                if (parts.size() >= 3)
                    cacheKey = meshLODKey(parts[0], std::stoul(parts[1]), std::stof(parts[2]));
            } else if (key == "vertexCount") {
                lods.vertexCount = static_cast<uint32_t>(std::stoul(value));
            } else if (key == "LODs") {
                parseUintList(value, lods.levelIndexCounts);
            } else if (key == "LODIndices") {
                parseUintList(value, lods.indices);
            }
        }

        // A single level is just LOD0 — nothing was simplified, so there is nothing to reuse.
        if (!cacheKey.empty() && lods.levelIndexCounts.size() > 1 && lods.vertexCount > 0)
            meshLODCache[cacheKey] = std::move(lods);
    }

    // Cheap pre-parse of the scene file that only looks for texture paths, so the KTX encoder can
    // work through the whole set at once before any of it is needed.
    void prewarmTextureCache(const std::string& filePath, Scene& scene) {
        std::ifstream scan(filePath);
        if (!scan.is_open())
            return;

        std::vector<textureconv::Job> jobs;
        std::string line, key, value;
        while (std::getline(scan, line)) {
            if (!parseKeyValue(line, key, value) || value.empty())
                continue;

            if (key == "AlbedoTexture" || key == "Texture") {
                jobs.push_back({{value}, textureconv::ColorSpace::Srgb});
            } else if (key == "MetallicTexture" || key == "RoughnessTexture") {
                jobs.push_back({{value}, textureconv::ColorSpace::Linear});
            } else if (key == "NormalTexture") {
                jobs.push_back({{value}, textureconv::ColorSpace::NormalMap});
            } else if (key == "EnvironmentMap") {
                // stored as posX|negX|posY|negY|posZ|negZ, which is already KTX face order
                auto parts = split(value, '|');
                if (parts.size() == 6)
                    jobs.push_back({parts, textureconv::ColorSpace::Srgb});
            }
        }
        scene.assetManager.prewarmTextureCache(jobs);
    }

    // trims whitespace from line
    void trim(std::string& str) {
        size_t start = str.find_first_not_of(" \t\r\n");
        size_t end = str.find_last_not_of(" \t\r\n");
        if (start == std::string::npos || end == std::string::npos) {
            str = "";
        } else {
            str = str.substr(start, end - start + 1);
        }
    }

    bool parseKeyValue(const std::string& line, std::string& key, std::string& value) {
        size_t colonPos = line.find(':');
        if (colonPos == std::string::npos) {
            return false;
        }
        key = line.substr(0, colonPos);
        value = line.substr(colonPos + 1);
        trim(key);
        trim(value);
        return true;
    }

    // parse comma separated values
    std::vector<std::string> split(const std::string& str, char delimiter) {
        std::vector<std::string> result;
        std::stringstream ss(str);
        std::string item;
        while (std::getline(ss, item, delimiter)) {
            trim(item);
            result.push_back(item);
        }
        return result;
    }

    void clearSceneInternal(Scene& scene, BindlessSystem& bindless,
                            const RenderBuffers& buffers,
                            uint32_t modelMatrixBufferIndex,
                            uint32_t lightBufferIndex) {
        // Textures remain loaded for reuse; the Free Unused window reclaims whatever the next scene
        // does not reference. Meshes do not — they go with the node graph below.
        std::cout << "Clearing scene..." << std::endl;

        auto& nodes = scene.sceneGraph.getNodes();
        Node& rootNode = scene.sceneGraph.getRootNode();

        // we don't delete root itself - clear all children recursively
        uint32_t child = rootNode.firstChild;
        while (child != 0) {
            clearNodeRecursive(nodes[child], scene);
            child = nodes[child].nextSibling;
        }
        rootNode.firstChild = 0;

        auto& materials = scene.getMaterials();
        if (materials.size() > 1) {
            Material defaultMaterial = materials[0]; // don't clear the default material
            // Free preview thumbnail slots; re-parsed materials reallocate them, orphaning these otherwise.
            for (size_t i = 1; i < materials.size(); i++) {
                if (materials[i].thumbnailTextureIndex != 0xFFFFFFFF)
                    bindless.descriptorSet->freeTexture(materials[i].thumbnailTextureIndex);
            }
            materials.clear();
            materials.push_back(defaultMaterial);
        }

        bindless.descriptorSet->clearFixedBuffer(modelMatrixBufferIndex);
        scene.clearBillboards();
        scene.clearRenderList();
        scene.clearLights(bindless, lightBufferIndex);
        scene.clearVolumes();
        scene.clearEmitters(bindless, buffers);
        scene.sceneGraph.reset();
        // Templates hold mesh references and index the mesh/material tables, so they go before the
        // release below — nothing survives a reload to point at the old indices.
        scene.clearTemplates();
        // After the graph reset nothing holds a mesh, so they all go. Deliberately not kept for
        // reuse: a surviving mesh table means every reload leaves behind the previous scene's
        // geometry, and the LOD chains in the scene file make the rebuild cheap anyway.
        scene.assetManager.releaseAllMeshes();
        std::cout << "Scene cleared successfully!" << std::endl;
    }

    void clearNodeRecursive(Node& node, Scene& scene) {
        auto& nodes = scene.sceneGraph.getNodes();
        uint32_t child = node.firstChild;
        while (child != 0) {
            clearNodeRecursive(nodes[child], scene);
            child = nodes[child].nextSibling;
        }

        // Remove mesh from shader rendering
        if (node.getMeshIndex() != MAX_MESHES && node.getMaterialIndex() != 0xFFFFFFFF) {
            uint32_t matIndex = node.getMaterialIndex();
            scene.removeMeshFromShader(node.getIndex(), scene.getMaterials()[matIndex].shaderSource, scene.getMaterials()[matIndex]);
        }
    }

    void parseMaterialsSection(std::ifstream& ifs, Scene& scene, std::unordered_map<uint32_t, uint32_t>& materialIDToIndex) {
        std::string line;
        while (std::getline(ifs, line)) {
            trim(line);

            if (line == "}") {
                break;
            }
            if (line == "Material {") {
                parseMaterial(ifs, scene, materialIDToIndex);
            }
        }
    }

    void parseMaterial(std::ifstream& ifs, Scene& scene, std::unordered_map<uint32_t, uint32_t>& materialIDToIndex) {
        Material material;
        material.shaderSource = scene.getFallBackShader();
        uint32_t materialID = 0;
        std::string albedoPath, metallicPath, roughnessPath, normalPath, environmentMapPath;

        std::string line;
        while (std::getline(ifs, line)) {
            trim(line);

            if (line == "}") {
                break; // End of Material
            }

            std::string key, value;
            if (!parseKeyValue(line, key, value)) {
                continue;
            }

            if (key == "ID") {
                materialID = std::stoul(value);
            } else if (key == "Name") {
                material.name = value;
            } else if (key == "Shader") {
                // resolve to a registered lit shader so pipelineIndex matches the saved source
                material.shaderSource = scene.resolveLitShader(value);
            } else if (key == "TextureMask" || key == "MaterialFlags") {
                material.flags = static_cast<MaterialFlags>(std::stoul(value));
            } else if (key == "Color") {
                auto parts = split(value, ',');
                if (parts.size() >= 4) {
                    material.color = glm::vec4(std::stof(parts[0]), std::stof(parts[1]), std::stof(parts[2]), std::stof(parts[3]));
                }
            } else if (key == "Metallic") {
                material.metallic = std::stof(value);
            } else if (key == "Roughness") {
                material.roughness = std::stof(value);
            } else if (key == "AlphaClip") {
                material.alphaClip = (std::stoi(value) != 0);
            } else if (key == "AlphaCutoff") {
                material.alphaCutoff = std::stof(value);
            } else if (key == "TriplanarScale") {
                material.triplanarScale = std::stof(value);
            } else if (key == "TriplanarBlend") {
                material.triplanarBlend = std::stof(value);
            } else if (key == "AlbedoTexture") {
                albedoPath = value;
            } else if (key == "MetallicTexture") {
                metallicPath = value;
            } else if (key == "RoughnessTexture") {
                roughnessPath = value;
            } else if (key == "NormalTexture") {
                normalPath = value;
            } else if (key == "EnvironmentMap") {
                environmentMapPath = value;
            }
        }

        // Load textures if paths are provided
        if (!albedoPath.empty()) {
            try {
                material.albedoTextureIndex = scene.assetManager.loadTextureFromFile(albedoPath);
            } catch (const std::exception& e) {
                std::cerr << "Failed to load albedo texture: " << albedoPath << " - " << e.what() << std::endl;
                material.albedoTextureIndex = scene.getMaterials()[scene.getFallBackMaterial()].albedoTextureIndex;
            }
        } else {
            material.albedoTextureIndex = scene.getMaterials()[scene.getFallBackMaterial()].albedoTextureIndex;
        }

        if (!metallicPath.empty()) {
            try {
                material.metallicTextureIndex = scene.assetManager.loadTextureFromFile(metallicPath, vk::Format::eR8G8B8A8Unorm);
            } catch (const std::exception& e) {
                std::cerr << "Failed to load metallic texture: " << metallicPath << " - " << e.what() << std::endl;
                material.metallicTextureIndex = scene.getMaterials()[scene.getFallBackMaterial()].metallicTextureIndex;
            }
        } else {
            material.metallicTextureIndex = scene.getMaterials()[scene.getFallBackMaterial()].metallicTextureIndex;
        }

        if (!roughnessPath.empty()) {
            try {
                material.roughnessTextureIndex = scene.assetManager.loadTextureFromFile(roughnessPath, vk::Format::eR8G8B8A8Unorm);
            } catch (const std::exception& e) {
                std::cerr << "Failed to load roughness texture: " << roughnessPath << " - " << e.what() << std::endl;
                material.roughnessTextureIndex = scene.getMaterials()[scene.getFallBackMaterial()].roughnessTextureIndex;
            }
        } else {
            material.roughnessTextureIndex = scene.getMaterials()[scene.getFallBackMaterial()].roughnessTextureIndex;
        }

        if (!normalPath.empty()) {
            try {
                material.normalTextureIndex = scene.assetManager.loadTextureFromFile(normalPath, vk::Format::eR8G8B8A8Unorm, textureconv::ColorSpace::NormalMap);
            } catch (const std::exception& e) {
                std::cerr << "Failed to load normal texture: " << normalPath << " - " << e.what() << std::endl;
                material.normalTextureIndex = scene.getMaterials()[scene.getFallBackMaterial()].normalTextureIndex;
            }
        } else {
            material.normalTextureIndex = scene.getMaterials()[scene.getFallBackMaterial()].normalTextureIndex;
        }

        if (!environmentMapPath.empty()) {
            try {
                auto parts = split(environmentMapPath, '|');
                if (parts.size() == 6) {
                    material.environmentMapIndex = scene.assetManager.loadCubemapFromFile(parts[0], // posX
                                                                                parts[2], // posY
                                                                                parts[4], // posZ
                                                                                parts[1], // negX
                                                                                parts[3], // negY
                                                                                parts[5]  // negZ
                    );
                    std::cout << "Loaded cubemap from: " << environmentMapPath << std::endl;
                } else {
                    std::cerr << "Invalid cubemap path format: " << environmentMapPath << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "Failed to load environment map: " << environmentMapPath << " - " << e.what() << std::endl;
            }
        }

        // Set HAS_* flags based on which textures were actually loaded
        uint32_t defaultAlbedo = scene.getMaterials()[scene.getFallBackMaterial()].albedoTextureIndex;
        uint32_t defaultRoughness = scene.getMaterials()[scene.getFallBackMaterial()].roughnessTextureIndex;
        uint32_t defaultMetallic = scene.getMaterials()[scene.getFallBackMaterial()].metallicTextureIndex;
        uint32_t defaultNormal = scene.getMaterials()[scene.getFallBackMaterial()].normalTextureIndex;
        if (!albedoPath.empty() && material.albedoTextureIndex != defaultAlbedo)
            material.flags = static_cast<MaterialFlags>(material.flags | HAS_ALBEDO);
        if (!roughnessPath.empty() && material.roughnessTextureIndex != defaultRoughness)
            material.flags = static_cast<MaterialFlags>(material.flags | HAS_ROUGHNESS);
        if (!metallicPath.empty() && material.metallicTextureIndex != defaultMetallic)
            material.flags = static_cast<MaterialFlags>(material.flags | HAS_METALLIC);
        if (!normalPath.empty() && material.normalTextureIndex != defaultNormal)
            material.flags = static_cast<MaterialFlags>(material.flags | HAS_NORMAL);
        if (material.alphaClip)
            material.flags = static_cast<MaterialFlags>(material.flags | ALPHA_CLIP);

        // Add material to renderer and store the mapping
        uint32_t materialIndex = scene.addMaterial(material);
        materialIDToIndex[materialID] = materialIndex;

        std::cout << "Loaded material ID: " << materialID << " -> index: " << materialIndex << std::endl;
    }

    // Returns the index of the node it created, so a caller that owns the block (the template
    // parser) can act on the finished subtree.
    uint32_t parseNode(std::ifstream& ifs, Scene& scene, BindlessSystem& bindless,
                   const RenderBuffers& buffers,
                   uint32_t lightBufferIndex,
                   uint32_t parentIndex, std::unordered_map<uint32_t, uint32_t>& materialIDToIndex) {
        std::string name = "Node";
        glm::vec3 position(0.0f);
        glm::quat rotation(1.0f, 0.0f, 0.0f, 0.0f);
        glm::vec3 scale(1.0f);
        std::string meshPath;
        std::string meshName;
        float meshScale = 1.0f;
        int meshEntry = -1; // -1 = not present (legacy scene) -> fall back to detection-based load
        std::vector<uint32_t> materialIDs;
        bool hasLight = false;
        bool hasVolume = false;
        bool hasEmitter = false;
        Light light;
        Volume vol;
        ParticleEmitter emitter;

        // Collect child node stream positions to parse after this node is created
        std::vector<std::streampos> childNodePositions;

        std::string line;
        int braceDepth = 0;
        while (std::getline(ifs, line)) {
            trim(line);

            if (line == "}") {
                if (braceDepth == 0) break;
                braceDepth--;
                continue;
            }

            if (line == "Node {") {
                if (braceDepth == 0) {
                    // Direct child — remember position to parse after this node is created
                    childNodePositions.push_back(ifs.tellg());
                }
                braceDepth++;
                continue;
            }

            if (braceDepth > 0) continue; // inside a child block, skip for now

            // Emitter is a nested block; consume it whole (including its own '}') so it
            // doesn't disturb this node's brace tracking.
            if (line == "Emitter {") {
                emitter = parseEmitter(ifs, scene);
                hasEmitter = true;
                continue;
            }

            std::string key, value;
            if (!parseKeyValue(line, key, value)) {
                continue;
            }

            if (key == "Name") {
                name = value;
            } else if (key == "Position") {
                auto parts = split(value, ',');
                if (parts.size() >= 3) {
                    position = glm::vec3(std::stof(parts[0]), std::stof(parts[1]), std::stof(parts[2]));
                }
            } else if (key == "Rotation") {
                auto parts = split(value, ',');
                if (parts.size() >= 4) {
                    rotation = glm::quat(std::stof(parts[0]), std::stof(parts[1]), std::stof(parts[2]), std::stof(parts[3]));
                }
            } else if (key == "Scale") {
                auto parts = split(value, ',');
                if (parts.size() >= 3) {
                    scale = glm::vec3(std::stof(parts[0]), std::stof(parts[1]), std::stof(parts[2]));
                }
            } else if (key == "Mesh") {
                meshPath = value;
            } else if (key == "MeshName") {
                meshName = value;
            } else if (key == "MeshEntry") {
                meshEntry = std::stoi(value);
            } else if (key == "MeshScale") {
                meshScale = std::stof(value);
            } else if (key == "MaterialID") {
                materialIDs.push_back(std::stoul(value));
            } else if (key == "Light") {
                hasLight = true;
                auto parts = split(value, ';');
                if (parts.size() >= 4) {
                    if (parts[0] == "Point") {
                        light.type = LightType::Point;
                    } else if (parts[0] == "Directional") {
                        light.type = LightType::Directional;
                    } else if (parts[0] == "Spot") {
                        light.type = LightType::Spot;
                    } else if (parts[0] == "Area") {
                        light.type = LightType::Area;
                    }

                    light.range = std::stof(parts[1]);
                    light.intensity = std::stof(parts[2]);

                    auto colorParts = split(parts[3], ',');
                    if (colorParts.size() >= 3) {
                        light.color = glm::vec4(std::stof(colorParts[0]), std::stof(colorParts[1]), std::stof(colorParts[2]), 1.0f);
                    }

                    if (parts.size() >= 5) {
                        light.castsShadows = std::stoi(parts[4]);
                    }
                    if (parts.size() >= 6) {
                        light.shadowResolution = std::stoul(parts[5]);
                    }
                    if (parts.size() >= 7) {
                        auto cascadeParts = split(parts[6], ',');
                        for (size_t i = 0; i < cascadeParts.size(); i++) {
                            if(i >= light.numCascades) break;
                            light.cascades[i].splitDistance = std::stof(cascadeParts[i]);
                        }
                    }
                }
            } else if (key == "Volume") {
                hasVolume = true;
                auto parts = split(value, ';');
                vol.density = std::stof(parts[0]);
                vol.phase = std::stof(parts[1]);
                vol.shape = static_cast<VolumeShape>(std::stoul(parts[2]));
                // parts[3] is a legacy center token — volumes now derive center from their node at
                // stream time, so it is parsed for format compatibility but discarded.
                vol.radius = std::stof(parts[4]);
                auto dimParts = split(parts[5], ',');
                vol.dimensions = glm::vec3(std::stof(dimParts[0]),std::stof(dimParts[1]),std::stof(dimParts[2]));
            }
        }

        // Create the node with the saved transform
        uint32_t nodeIndex = scene.sceneGraph.addNode(false, parentIndex, position, rotation, scale);
        scene.sceneGraph.getNodes()[nodeIndex].name = name;

        std::cout << "created node: " << name << " at index " << nodeIndex << std::endl;

        // Load mesh if specified
        if (!meshPath.empty()) {
            try {
                uint32_t meshIdx;
                if (meshEntry >= 0) {
                    // Fast path: rebuild this exact mesh from its source entry, no instance detection.
                    // A saved LOD chain skips the simplifier; loadSceneMesh re-derives it if the
                    // entry no longer matches what was serialised.
                    auto cached = meshLODCache.find(meshLODKey(meshPath, static_cast<uint32_t>(meshEntry), meshScale));
                    meshIdx = scene.assetManager.loadSceneMesh(meshPath, static_cast<uint32_t>(meshEntry), meshName, meshScale,
                                                               cached != meshLODCache.end() ? &cached->second : nullptr);
                } else {
                    // Legacy scene without entry indices — fall back to the full detection-based load.
                    auto loadResult = scene.assetManager.loadMeshFromFile(meshPath, meshScale);
                    auto& meshIndices = loadResult.meshIndices;
                    if (meshIndices.empty()) {
                        throw std::runtime_error("no geometry in " + meshPath);
                    }

                    // Find the specific sub-mesh by name, or fall back to the first one
                    meshIdx = meshIndices[0];
                    if (!meshName.empty()) {
                        for (uint32_t idx : meshIndices) {
                            if (scene.assetManager.getMeshes()[idx].name == meshName) {
                                meshIdx = idx;
                                break;
                            }
                        }
                    }
                }

                Node& n = scene.sceneGraph.getNodes()[nodeIndex];
                NodeOps::assignMesh(n, meshIdx, scene);
                uint32_t matIdx = scene.getFallBackMaterial();
                if (!materialIDs.empty() && materialIDToIndex.find(materialIDs[0]) != materialIDToIndex.end()) {
                    matIdx = materialIDToIndex[materialIDs[0]];
                }
                NodeOps::assignMaterial(n, matIdx, scene);
            } catch (const std::exception& e) {
                std::cerr << "Failed to load mesh: " << meshPath << " - " << e.what() << std::endl;
            }
        }

        // Add light if specified
        if (hasLight) {
            Node& n = scene.sceneGraph.getNodes()[nodeIndex];
            NodeOps::assignLight(n, light, scene, bindless, lightBufferIndex);
            std::cout << "Added light to node: " << name << std::endl;
        }

        // same for volume
        if (hasVolume) {
            Node& n = scene.sceneGraph.getNodes()[nodeIndex];
            NodeOps::assignVolume(n, vol, scene);
            std::cout << "Added volume to node: " << name << std::endl;
        }

        // and particle emitter (registers a pool sub-range + descriptor slot via Scene::addEmitter)
        if (hasEmitter) {
            Node& n = scene.sceneGraph.getNodes()[nodeIndex];
            NodeOps::assignEmitter(n, emitter, scene, bindless, buffers);
            std::cout << "Added emitter to node: " << name << std::endl;
        }

        // Now parse child nodes recursively
        for (auto& pos : childNodePositions) {
            ifs.seekg(pos);
            parseNode(ifs, scene, bindless, buffers, lightBufferIndex, nodeIndex, materialIDToIndex);
        }
        return nodeIndex;
    }

    // A Template block is loaded by materialising its nodes in the scene, snapshotting them into a
    // template, then deleting them again. That reuses parseNode's mesh/material/light/emitter
    // handling instead of duplicating it, and addTemplate lifts the attachment payloads out of the
    // scene before the temporaries go away, so nothing is left dangling.
    void parseTemplateSection(std::ifstream& ifs, Scene& scene, BindlessSystem& bindless,
                              const RenderBuffers& buffers, uint32_t lightBufferIndex,
                              std::unordered_map<uint32_t, uint32_t>& materialIDToIndex) {
        std::string name;
        uint32_t rootIndex = 0;

        std::string line, key, value;
        while (std::getline(ifs, line)) {
            trim(line);
            if (line == "}") break; // end of Template

            if (line == "Node {") {
                // A template has exactly one root; anything further is malformed, so drop it
                // rather than leave stray nodes in the scene.
                uint32_t idx = parseNode(ifs, scene, bindless, buffers, lightBufferIndex, SceneGraph::ROOT_INDEX, materialIDToIndex);
                if (rootIndex == 0) rootIndex = idx;
                else scene.sceneGraph.removeNode(idx);
                continue;
            }
            if (parseKeyValue(line, key, value) && key == "Name") name = value;
        }

        if (rootIndex == 0) return;
        if (!name.empty()) scene.addTemplate(rootIndex, name);
        else std::cerr << "Template block has no Name, dropping it" << std::endl;
        scene.sceneGraph.removeNode(rootIndex);
    }

    ParticleEmitter parseEmitter(std::ifstream& ifs, Scene& scene) {
        ParticleEmitter emitter;
        std::string texturePath;

        std::string line;
        while (std::getline(ifs, line)) {
            trim(line);

            if (line == "}") {
                break; // End of Emitter
            }

            std::string key, value;
            if (!parseKeyValue(line, key, value)) {
                continue;
            }

            if (key == "Texture") {
                texturePath = value;
            } else if (key == "PositionOffset") {
                auto p = split(value, ',');
                if (p.size() >= 3) emitter.positionOffset = glm::vec3(std::stof(p[0]), std::stof(p[1]), std::stof(p[2]));
            } else if (key == "RotationOffset") {
                auto p = split(value, ',');
                if (p.size() >= 4) emitter.rotationOffset = glm::quat(std::stof(p[0]), std::stof(p[1]), std::stof(p[2]), std::stof(p[3])); // w,x,y,z
            } else if (key == "EmissionRate") {
                emitter.emissionRate = std::stof(value);
            } else if (key == "LifeTime") {
                auto p = split(value, ',');
                if (p.size() >= 2) emitter.lifeTime = glm::vec2(std::stof(p[0]), std::stof(p[1]));
            } else if (key == "SpreadAngle") {
                emitter.spreadAngle = std::stof(value);
            } else if (key == "SpeedMin") {
                emitter.speedMin = std::stof(value);
            } else if (key == "SpeedMax") {
                emitter.speedMax = std::stof(value);
            } else if (key == "AngularVelocity") {
                auto p = split(value, ',');
                if (p.size() >= 2) emitter.angularVelocityRandom = glm::vec2(std::stof(p[0]), std::stof(p[1]));
            } else if (key == "Drag") {
                emitter.drag = std::stof(value);
            } else if (key == "SizeRandom") {
                auto p = split(value, ',');
                if (p.size() >= 2) emitter.sizeRandom = glm::vec2(std::stof(p[0]), std::stof(p[1]));
            } else if (key == "Animated") {
                emitter.animated = (std::stoi(value) != 0);
            } else if (key == "NumFrames") {
                emitter.numFrames = static_cast<uint8_t>(std::stoul(value));
            } else if (key == "Lit") {
                emitter.lit = (std::stoi(value) != 0);
            } else if (key == "Volumetric") {
                emitter.volumetric = (std::stoi(value) != 0);
            } else if (key == "VolumetricSphere") {
                emitter.volumetricSphere = (std::stoi(value) != 0);
            } else if (key == "SoftParticle") {
                emitter.softParticle = (std::stoi(value) != 0);
            } else if (key == "SoftRadius") {
                emitter.softRadius = std::stof(value);
            } else if (key == "SphereRoundness") {
                emitter.sphereRoundness = std::stof(value);
            } else if (key == "Opacity") {
                emitter.opacity = std::stof(value);
            } else if (key == "DensityRange") {
                auto p = split(value, ',');
                if (p.size() >= 2) emitter.densityRange = glm::vec2(std::stof(p[0]), std::stof(p[1]));
            } else if (key == "VolumePhase") {
                emitter.volumePhase = std::stof(value);
            } else if (key == "EmissiveRange") {
                auto p = split(value, ',');
                if (p.size() >= 2) emitter.emissiveRange = glm::vec2(std::stof(p[0]), std::stof(p[1]));
            }
        }

        if (!texturePath.empty()) {
            try {
                emitter.textureIndex = scene.assetManager.loadTextureFromFile(texturePath);
            } catch (const std::exception& e) {
                std::cerr << "Failed to load emitter texture: " << texturePath << " - " << e.what() << std::endl;
            }
        }

        return emitter;
    }

    void collectMaterials(Node& node, Scene& scene) {
        if (node.getMeshIndex() != MAX_MESHES && node.getMaterialIndex() != 0xFFFFFFFF) {
            Material& mat = scene.getMaterials()[node.getMaterialIndex()];
            savedMaterialIDs.insert(mat.materialID);
        }
        auto& nodes = scene.sceneGraph.getNodes();
        uint32_t child = node.firstChild;
        while (child != 0) {
            collectMaterials(nodes[child], scene);
            child = nodes[child].nextSibling;
        }
    }

    void writeMaterials(Scene& scene) {
        ofs << "Materials {" << std::endl;
        for (const Material& mat : scene.getMaterials()) {
            // Skip default material (index 0) as it's recreated on init
            if (mat.materialID == scene.getMaterials()[0].materialID) {
                continue;
            }

            ofs << "  Material {" << std::endl;
            ofs << "    ID : " << mat.materialID << std::endl;
            ofs << "    Name : " << mat.name << std::endl;
            ofs << "    Shader : " << mat.shaderSource.sourceFile << std::endl;
            ofs << "    MaterialFlags : " << mat.flags << std::endl;
            ofs << "    Color : " << mat.color.r << "," << mat.color.g << "," << mat.color.b << "," << mat.color.a << std::endl;
            ofs << "    Metallic : " << mat.metallic << std::endl;
            ofs << "    Roughness : " << mat.roughness << std::endl;
            ofs << "    AlphaClip : " << (mat.alphaClip ? 1 : 0) << std::endl;
            ofs << "    AlphaCutoff : " << mat.alphaCutoff << std::endl;
            ofs << "    TriplanarScale : " << mat.triplanarScale << std::endl;
            ofs << "    TriplanarBlend : " << mat.triplanarBlend << std::endl;

            // Save texture paths
            std::string albedoPath = scene.assetManager.getTexturePathFromIndex(mat.albedoTextureIndex);
            std::string metallicPath = scene.assetManager.getTexturePathFromIndex(mat.metallicTextureIndex);
            std::string roughnessPath = scene.assetManager.getTexturePathFromIndex(mat.roughnessTextureIndex);
            std::string normalPath = scene.assetManager.getTexturePathFromIndex(mat.normalTextureIndex);

            ofs << "    AlbedoTexture : " << albedoPath << std::endl;
            ofs << "    MetallicTexture : " << metallicPath << std::endl;
            ofs << "    RoughnessTexture : " << roughnessPath << std::endl;
            ofs << "    NormalTexture : " << normalPath << std::endl;

            // Save cubemap
            std::string cubemapPath = scene.assetManager.getCubemapPathFromIndex(mat.environmentMapIndex);
            ofs << "    EnvironmentMap : " << cubemapPath << std::endl;

            ofs << "  }" << std::endl;
        }
        ofs << "}" << std::endl << std::endl;
    }

    // Serialises one node and its subtree. `tmpl` switches both the child walk and the attachment
    // lookups over to a template's own arrays: a template node's links and payload keys are local
    // to that template and mean something else entirely in the scene graph.
    void writeNodes(const Node& node, Scene& scene, int depth,
                    const NodeTemplate* tmpl = nullptr, uint32_t localIndex = 0) {
        const Light* lightPtr = nullptr;
        const Volume* volumePtr = nullptr;
        const ParticleEmitter* emitterPtr = nullptr;
        if (tmpl) {
            auto l = tmpl->lights.find(localIndex);
            if (l != tmpl->lights.end()) lightPtr = &l->second;
            auto v = tmpl->volumes.find(localIndex);
            if (v != tmpl->volumes.end()) volumePtr = &v->second;
            auto e = tmpl->emitters.find(localIndex);
            if (e != tmpl->emitters.end()) emitterPtr = &e->second;
        } else {
            auto l = scene.lights.find(node.lightIndex);
            if (l != scene.lights.end()) lightPtr = &l->second;
            auto v = scene.volumes.find(node.nodeIndex);
            if (v != scene.volumes.end()) volumePtr = &v->second;
            auto e = scene.particleEmitters.find(node.particleIndex);
            if (e != scene.particleEmitters.end()) emitterPtr = &e->second;
        }

        std::string indent(depth * 2, ' ');
        ofs << indent << "Node {" << std::endl;
        ofs << indent << "  Name : " << node.name << std::endl;
        glm::vec3 pos = node.relativePosition;
        glm::quat rot = node.relativeRotation;
        glm::vec3 scale = node.relativeScale;
        ofs << indent << "  Position : " << pos.x << "," << pos.y << "," << pos.z << std::endl;
        ofs << indent << "  Rotation : " << rot.w << "," << rot.x << "," << rot.y << "," << rot.z << std::endl;
        ofs << indent << "  Scale : " << scale.x << "," << scale.y << "," << scale.z << std::endl;

        if (lightPtr) {
            const Light& light = *lightPtr;
            std::string type;
            std::string cascades = "";
            switch (light.type) {
            case LightType::Point:
                type = "Point";
                break;
            case LightType::Directional:
                type = "Directional";
                cascades = ";";
                for (size_t i = 0; i < light.cascades.size(); i++) {
                    cascades += std::to_string(light.cascades[i].splitDistance);
                    if (i < light.cascades.size() - 1) {
                        cascades += ",";
                    }
                }
                break;
            case LightType::Spot:
                type = "Spot";
                break;
            case LightType::Area:
                type = "Area";
                break;
            }
            ofs << indent << "  Light : " << type << ";" << light.range << ";" << light.intensity << ";" << light.color.r << "," << light.color.g << "," << light.color.b << ";"
                << light.castsShadows << ";" << light.shadowResolution << cascades << std::endl;
        }

        if (node.meshIndex != MAX_MESHES && node.meshIndex < scene.assetManager.meshes.size()) {
            Mesh& mesh = scene.assetManager.meshes[node.meshIndex];
            ofs << indent << "  Mesh : " << mesh.sourceFile << std::endl;
            if (!mesh.name.empty()) {
                ofs << indent << "  MeshName : " << mesh.name << std::endl;
            }
            // Source entry index lets scene load rebuild the mesh directly, skipping instance detection.
            ofs << indent << "  MeshEntry : " << mesh.sourceEntryIndex << std::endl;
            if (mesh.importScale != 1.0f) {
                ofs << indent << "  MeshScale : " << mesh.importScale << std::endl;
            }

            if (node.materialIndex != 0xFFFFFFFF) {
                Material& mat = scene.getMaterials()[node.materialIndex];
                ofs << indent << "  MaterialID : " << mat.materialID << std::endl;
            }
        }

        if (volumePtr) {
            const Volume& vol = *volumePtr;
            // Center is derived from the node at stream time; write a 0,0,0 placeholder to keep the
            // legacy token slot in the format (it is discarded on load).
            ofs << indent << " Volume : " << vol.density << ";" << vol.phase << ";" << static_cast<uint32_t>(vol.shape) << ";" << "0,0,0" << ";" << vol.radius << ";" << vol.dimensions.x << "," << vol.dimensions.y << "," << vol.dimensions.z << std::endl;
        }

        if (emitterPtr) {
            const ParticleEmitter& em = *emitterPtr;
            // Pool residency (particleOffset/particleCapacity) and nodeIndex are recomputed by
            // Scene::addEmitter on load, so only the authored fields are written here.
            ofs << indent << "  Emitter {" << std::endl;
            ofs << indent << "    Texture : " << scene.assetManager.getTexturePathFromIndex(em.textureIndex) << std::endl;
            ofs << indent << "    PositionOffset : " << em.positionOffset.x << "," << em.positionOffset.y << "," << em.positionOffset.z << std::endl;
            ofs << indent << "    RotationOffset : " << em.rotationOffset.w << "," << em.rotationOffset.x << "," << em.rotationOffset.y << "," << em.rotationOffset.z << std::endl;
            ofs << indent << "    EmissionRate : " << em.emissionRate << std::endl;
            ofs << indent << "    LifeTime : " << em.lifeTime.x << "," << em.lifeTime.y << std::endl;
            ofs << indent << "    SpreadAngle : " << em.spreadAngle << std::endl;
            ofs << indent << "    SpeedMin : " << em.speedMin << std::endl;
            ofs << indent << "    SpeedMax : " << em.speedMax << std::endl;
            ofs << indent << "    AngularVelocity : " << em.angularVelocityRandom.x << "," << em.angularVelocityRandom.y << std::endl;
            ofs << indent << "    Drag : " << em.drag << std::endl;
            ofs << indent << "    SizeRandom : " << em.sizeRandom.x << "," << em.sizeRandom.y << std::endl;
            ofs << indent << "    Animated : " << (em.animated ? 1 : 0) << std::endl;
            ofs << indent << "    NumFrames : " << static_cast<uint32_t>(em.numFrames) << std::endl;
            ofs << indent << "    Lit : " << (em.lit ? 1 : 0) << std::endl;
            ofs << indent << "    Volumetric : " << (em.volumetric ? 1 : 0) << std::endl;
            ofs << indent << "    VolumetricSphere : " << (em.volumetricSphere ? 1 : 0) << std::endl;
            ofs << indent << "    SoftParticle : " << (em.softParticle ? 1 : 0) << std::endl;
            ofs << indent << "    SoftRadius : " << em.softRadius << std::endl;
            ofs << indent << "    SphereRoundness : " << em.sphereRoundness << std::endl;
            ofs << indent << "    Opacity : " << em.opacity << std::endl;
            ofs << indent << "    DensityRange : " << em.densityRange.x << "," << em.densityRange.y << std::endl;
            ofs << indent << "    VolumePhase : " << em.volumePhase << std::endl;
            ofs << indent << "    EmissiveRange : " << em.emissiveRange.x << "," << em.emissiveRange.y << std::endl;
            ofs << indent << "  }" << std::endl;
        }

        // Write children, following whichever pool this node's links belong to
        uint32_t child = node.firstChild;
        if (tmpl) {
            while (child != 0) {
                writeNodes(tmpl->nodes[child], scene, depth + 1, tmpl, child);
                child = tmpl->nodes[child].nextSibling;
            }
        } else {
            auto& nodes = scene.sceneGraph.getNodes();
            while (child != 0) {
                writeNodes(nodes[child], scene, depth + 1);
                child = nodes[child].nextSibling;
            }
        }

        ofs << indent << "}" << std::endl;
    }

    void writeMeshes(AssetManager& assets) {

        for(Mesh& mesh : assets.meshes) {
            if (mesh.freed) continue; // slot released, its geometry is already gone
            ofs << "Mesh {" << std::endl;
            ofs << "  Source : " << mesh.sourceFile << ";" << mesh.sourceEntryIndex << ";" << mesh.importScale << std::endl;
            ofs << "  vertexCount : " << mesh.vertexCount << std::endl;
            ofs << "  indexCount : " << mesh.indexCount << std::endl;
            ofs << "  Center : " << mesh.center.x << ";" << mesh.center.y << ";" << mesh.center.z << std::endl;
            ofs << "  BBOX : " << mesh.boundingBoxMin.x << ";" << mesh.boundingBoxMin.y << ";" << mesh.boundingBoxMin.z << ";" << mesh.boundingBoxMax.x << ";" << mesh.boundingBoxMax.y << ";" << mesh.boundingBoxMax.z << std::endl;
            ofs << "  minRadius : " << mesh.minRadius << std::endl;
            ofs << "  maxRadius : " << mesh.maxRadius << std::endl;
            ofs << "  surfaceArea : " << mesh.surfaceArea << std::endl;
            ofs << "  LODs : ";
            for(uint32_t lod : mesh.LODs) {
                ofs << lod << ";";
            }
            ofs << std::endl;
            // The counts above are only the level lengths — they can't spare a reload the
            // simplification pass on their own. This is the simplifier's actual output: the LOD1..N
            // corners, which all index this mesh's own vertex buffer, so the levels cost nothing
            // beyond these uint32s. cpuIndices holds LOD0 first, then the levels back to back.
            if (!mesh.LODs.empty() && mesh.cpuIndices.size() > mesh.LODs[0]) {
                ofs << "  LODIndices : ";
                for (size_t i = mesh.LODs[0]; i < mesh.cpuIndices.size(); i++) ofs << mesh.cpuIndices[i] << ";";
                ofs << std::endl;
            }
            ofs << "}" << std::endl;
        }
    }

    // One block per template: its name, then its root node — writeNodes recurses through the
    // template's own child links from there, so there is no sibling walk to do here.
    void writeTemplates(Scene& scene) {
        for (const auto& [name, tmpl] : scene.templates) {
            if (tmpl.nodes.empty()) continue;
            ofs << "Template {" << std::endl;
            ofs << "  Name : " << name << std::endl;
            writeNodes(tmpl.nodes[0], scene, 1, &tmpl, 0);
            ofs << "}" << std::endl;
        }
    }
};
