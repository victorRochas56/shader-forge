#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "renderer.hpp"
#include "node_ops.hpp"
#include "utils.hpp"

/*
parses scene data and saves it to a file
and 
parses a file and loads a scene from it
*/

class SceneManager {
  public:
    void clearScene(Renderer& renderer) { clearSceneInternal(renderer); }

    void saveScene(std::string filePath, Renderer& renderer) {

        ofs.open(filePath, std::ios::out | std::ios::trunc); // clears the file
        savedMaterialIDs.clear();

        // first collect all unique materials used by nodes
        Node& rootNode = renderer.sceneGraph.getRootNode();
        auto& nodes = renderer.sceneGraph.getNodes();
        uint32_t child = rootNode.firstChild;
        while (child != 0) {
            collectMaterials(nodes[child], renderer);
            child = nodes[child].nextSibling;
        }
        writeMaterials(renderer);

        // then write all nodes
        child = rootNode.firstChild;
        while (child != 0) {
            writeNodes(nodes[child], renderer);
            child = nodes[child].nextSibling;
        }
        ofs.close();
    }

    void loadScene(std::string filePath, Renderer& renderer) {
        std::cout << "Loading scene from: " << filePath << std::endl;

        std::ifstream ifs(filePath);
        if (!ifs.is_open()) {
            std::cerr << "Failed to open scene file: " << filePath << std::endl;
            return;
        }

        // Clear existing scene (except root node)
        clearSceneInternal(renderer);

        // Maps to track loaded resources
        std::unordered_map<uint32_t, uint32_t> materialIDToIndex; // materialID -> material index in renderer
        std::unordered_map<std::string, uint32_t> loadedMeshes;   // mesh path -> mesh index

        std::string line;
        while (std::getline(ifs, line)) {
            trim(line);

            if (line == "Materials {") {
                parseMaterialsSection(ifs, renderer, materialIDToIndex);
            } else if (line == "Node {") {
                parseNode(ifs, renderer, SceneGraph::ROOT_INDEX, materialIDToIndex, loadedMeshes);
            }
        }

        ifs.close();
        std::cout << "Scene loaded successfully!" << std::endl;
    }

  private:
    std::fstream ofs;
    std::unordered_set<uint32_t> savedMaterialIDs;

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

    void clearSceneInternal(Renderer& renderer) { // meshes and textures remain loaded in memory for reuse (TODO check on load for unused?)
        std::cout << "Clearing scene..." << std::endl;

        auto& nodes = renderer.sceneGraph.getNodes();
        Node& rootNode = renderer.sceneGraph.getRootNode();

        // we don't delete root itself - clear all children recursively
        uint32_t child = rootNode.firstChild;
        while (child != 0) {
            clearNodeRecursive(nodes[child], renderer);
            child = nodes[child].nextSibling;
        }
        rootNode.firstChild = 0;

        auto& materials = renderer.getMaterials();
        if (materials.size() > 1) {
            Material defaultMaterial = materials[0]; // don't clear the default material
            materials.clear();
            materials.push_back(defaultMaterial);
        }

        renderer.sceneGraph.getNodes().clear();
        renderer.sceneGraph.resetLastNode();
        renderer.clearRenderList();
        renderer.clearLights();
        renderer.sceneGraph.init(&renderer);
        std::cout << "Scene cleared successfully!" << std::endl;
    }

    void clearNodeRecursive(Node& node, Renderer& renderer) {
        auto& nodes = renderer.sceneGraph.getNodes();
        uint32_t child = node.firstChild;
        while (child != 0) {
            clearNodeRecursive(nodes[child], renderer);
            child = nodes[child].nextSibling;
        }

        // Remove meshes from shader rendering maps
        if (node.getMeshIndex() != MAX_MESHES) {
            uint32_t meshIndex = node.getMeshIndex();
            auto& materialIndices = node.getMaterialIndices();
            for (size_t i = 0; i < node.materialIndexCount; i++) {
                uint32_t matIndex = materialIndices[i];
                if (i < renderer.assetManager.meshes[meshIndex].subMeshes.size()) {
                    uint32_t subMeshIndex = renderer.assetManager.meshes[meshIndex].subMeshes[i];
                    renderer.removeMeshFromShader(&node, subMeshIndex, renderer.getMaterials()[matIndex].shaderSource, renderer.getMaterials()[matIndex]);
                }
            }
        }
    }

    void parseMaterialsSection(std::ifstream& ifs, Renderer& renderer, std::unordered_map<uint32_t, uint32_t>& materialIDToIndex) {
        std::string line;
        while (std::getline(ifs, line)) {
            trim(line);

            if (line == "}") {
                break;
            }
            if (line == "Material {") {
                parseMaterial(ifs, renderer, materialIDToIndex);
            }
        }
    }

    void parseMaterial(std::ifstream& ifs, Renderer& renderer, std::unordered_map<uint32_t, uint32_t>& materialIDToIndex) {
        Material material;
        material.shaderSource = renderer.getFallBackShader();
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
                material.shaderSource.sourceFile = value;
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
                material.albedoTextureIndex = renderer.assetManager.loadTextureFromFile(albedoPath);
            } catch (const std::exception& e) {
                std::cerr << "Failed to load albedo texture: " << albedoPath << " - " << e.what() << std::endl;
                material.albedoTextureIndex = renderer.getMaterials()[renderer.getFallBackMaterial()].albedoTextureIndex;
            }
        } else {
            material.albedoTextureIndex = renderer.getMaterials()[renderer.getFallBackMaterial()].albedoTextureIndex;
        }

        if (!metallicPath.empty()) {
            try {
                material.metallicTextureIndex = renderer.assetManager.loadTextureFromFile(metallicPath, vk::Format::eR8G8B8A8Unorm);
            } catch (const std::exception& e) {
                std::cerr << "Failed to load metallic texture: " << metallicPath << " - " << e.what() << std::endl;
                material.metallicTextureIndex = renderer.getMaterials()[renderer.getFallBackMaterial()].metallicTextureIndex;
            }
        } else {
            material.metallicTextureIndex = renderer.getMaterials()[renderer.getFallBackMaterial()].metallicTextureIndex;
        }

        if (!roughnessPath.empty()) {
            try {
                material.roughnessTextureIndex = renderer.assetManager.loadTextureFromFile(roughnessPath, vk::Format::eR8G8B8A8Unorm);
            } catch (const std::exception& e) {
                std::cerr << "Failed to load roughness texture: " << roughnessPath << " - " << e.what() << std::endl;
                material.roughnessTextureIndex = renderer.getMaterials()[renderer.getFallBackMaterial()].roughnessTextureIndex;
            }
        } else {
            material.roughnessTextureIndex = renderer.getMaterials()[renderer.getFallBackMaterial()].roughnessTextureIndex;
        }

        if (!normalPath.empty()) {
            try {
                material.normalTextureIndex = renderer.assetManager.loadTextureFromFile(normalPath, vk::Format::eR8G8B8A8Unorm);
            } catch (const std::exception& e) {
                std::cerr << "Failed to load normal texture: " << normalPath << " - " << e.what() << std::endl;
                material.normalTextureIndex = renderer.getMaterials()[renderer.getFallBackMaterial()].normalTextureIndex;
            }
        } else {
            material.normalTextureIndex = renderer.getMaterials()[renderer.getFallBackMaterial()].normalTextureIndex;
        }
        
        if (!environmentMapPath.empty()) {
            try {
                auto parts = split(environmentMapPath, '|');
                if (parts.size() == 6) {
                    material.environmentMapIndex = renderer.assetManager.loadCubemapFromFile(parts[0], // posX
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
        uint32_t defaultAlbedo = renderer.getMaterials()[renderer.getFallBackMaterial()].albedoTextureIndex;
        uint32_t defaultRoughness = renderer.getMaterials()[renderer.getFallBackMaterial()].roughnessTextureIndex;
        uint32_t defaultMetallic = renderer.getMaterials()[renderer.getFallBackMaterial()].metallicTextureIndex;
        uint32_t defaultNormal = renderer.getMaterials()[renderer.getFallBackMaterial()].normalTextureIndex;
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
        uint32_t materialIndex = renderer.addMaterial(material);
        materialIDToIndex[materialID] = materialIndex;

        std::cout << "Loaded material ID: " << materialID << " -> index: " << materialIndex << std::endl;
    }

    void parseNode(std::ifstream& ifs, Renderer& renderer, uint32_t parentIndex, std::unordered_map<uint32_t, uint32_t>& materialIDToIndex,
                   std::unordered_map<std::string, uint32_t>& loadedMeshes) {
        std::string name = "Node";
        glm::vec3 position(0.0f);
        glm::quat rotation(1.0f, 0.0f, 0.0f, 0.0f);
        glm::vec3 scale(1.0f);
        std::string meshPath;
        std::vector<uint32_t> materialIDs;
        bool hasLight = false;
        Light light;

        std::string line;
        while (std::getline(ifs, line)) {
            trim(line);

            if (line == "}") {
                break;
            }

            if (line == "Node {") {
                // Recursively parse child node (we'll handle this after creating current node)
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
            } else if (key == "MaterialID") {
                materialIDs.push_back(std::stoul(value));
            } else if (key == "Light") {
                hasLight = true;
                auto parts = split(value, ';');
                if (parts.size() >= 4) {
                    // Parse light type
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

                    // Parse shadow settings (if present)
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
            }
        }

        // Create the node
        uint32_t nodeIndex = renderer.sceneGraph.addNode(parentIndex, position, rotation, scale);
        Node& newNode = renderer.sceneGraph.getNodes()[nodeIndex];
        newNode.name = name;

        std::cout << "created node: " << name << " at index " << nodeIndex << std::endl;

        // Load mesh if specified
        if (!meshPath.empty()) {
            try {
                uint32_t meshIndex;
                if (loadedMeshes.find(meshPath) != loadedMeshes.end()) {
                    meshIndex = loadedMeshes[meshPath];
                    std::cout << "Reusing mesh: " << meshPath << std::endl;
                } else {
                    meshIndex = renderer.assetManager.loadMeshFromFile(meshPath);
                    loadedMeshes[meshPath] = meshIndex;
                    std::cout << "Loaded mesh: " << meshPath << std::endl;
                }
                NodeOps::assignMesh(newNode, meshIndex, renderer);

                // Add materials to submeshes
                for (size_t i = 0; i < materialIDs.size(); i++) {
                    uint32_t materialID = materialIDs[i];
                    if (materialIDToIndex.find(materialID) != materialIDToIndex.end()) {
                        uint32_t materialIndex = materialIDToIndex[materialID];
                        NodeOps::assignMaterial(newNode, i, materialIndex, renderer);
                        std::cout << "  Added material " << materialID << " to submesh " << i << std::endl;
                    } else {
                        std::cerr << "Material ID " << materialID << " not found, using fallback" << std::endl;
                        NodeOps::assignMaterial(newNode, i, renderer.getFallBackMaterial(), renderer);
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "Failed to load mesh: " << meshPath << " - " << e.what() << std::endl;
            }
        }

        // Add light if specified
        if (hasLight) {
            NodeOps::assignLight(newNode, light, renderer);
            std::cout << "Added light to node: " << name << std::endl;
        }
    }

    void collectMaterials(Node& node, Renderer& renderer) {
        if (node.getMeshIndex() != MAX_MESHES) {
            auto& materialIndices = node.getMaterialIndices();
            for (int i = 0; i < node.materialIndexCount; i++) {
                Material& mat = renderer.getMaterials()[materialIndices[i]];
                savedMaterialIDs.insert(mat.materialID);
            }
        }
        auto& nodes = renderer.sceneGraph.getNodes();
        uint32_t child = node.firstChild;
        while (child != 0) {
            collectMaterials(nodes[child], renderer);
            child = nodes[child].nextSibling;
        }
    }

    void writeMaterials(Renderer& renderer) { 
        ofs << "Materials {" << std::endl;
        for (const Material& mat : renderer.getMaterials()) {
            // Skip default material (index 0) as it's recreated on init
            if (mat.materialID == renderer.getMaterials()[0].materialID) {
                continue;
            }
            // Save all non-default materials (including unassigned ones for reuse)

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

            // Save texture paths
            std::string albedoPath = renderer.assetManager.getTexturePathFromIndex(mat.albedoTextureIndex);
            std::string metallicPath = renderer.assetManager.getTexturePathFromIndex(mat.metallicTextureIndex);
            std::string roughnessPath = renderer.assetManager.getTexturePathFromIndex(mat.roughnessTextureIndex);
            std::string normalPath = renderer.assetManager.getTexturePathFromIndex(mat.normalTextureIndex);

            ofs << "    AlbedoTexture : " << albedoPath << std::endl;
            ofs << "    MetallicTexture : " << metallicPath << std::endl;
            ofs << "    RoughnessTexture : " << roughnessPath << std::endl;
            ofs << "    NormalTexture : " << normalPath << std::endl;

            // Save cubemap
            std::string cubemapPath = renderer.assetManager.getCubemapPathFromIndex(mat.environmentMapIndex);
            ofs << "    EnvironmentMap : " << cubemapPath << std::endl;

            ofs << "  }" << std::endl;
        }
        ofs << "}" << std::endl << std::endl;
    }

    void writeNodes(Node& node, Renderer& renderer) {
        ofs << "Node {" << std::endl;
        ofs << "  Name : " << node.name << std::endl;
        glm::vec3 pos = node.getWorldPosition();
        glm::quat rot = node.getWorldRotation();
        glm::vec3 scale = node.getWorldScale();
        ofs << "  Position : " << pos.x << "," << pos.y << "," << pos.z << std::endl;
        ofs << "  Rotation : " << rot.w << "," << rot.x << "," << rot.y << "," << rot.z << std::endl;
        ofs << "  Scale : " << scale.x << "," << scale.y << "," << scale.z << std::endl;

        if (node.getLightIndex() != MAX_LIGHTS) {
            Light light = renderer.getLight(node.getLightIndex());
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
            ofs << "  Light : " << type << ";" << light.range << ";" << light.intensity << ";" << light.color.r << "," << light.color.g << "," << light.color.b << ";"
                << light.castsShadows << ";" << light.shadowResolution << cascades << std::endl;
        }

        if (node.getMeshIndex() != MAX_MESHES) {
            Mesh mesh = renderer.assetManager.meshes[node.getMeshIndex()];
            ofs << "  Mesh : " << mesh.sourceFile << std::endl;

            // Reference materials by their matID
            auto& materialIndices = node.getMaterialIndices();
            for (int i = 0; i < node.materialIndexCount; i++) {
                Material& mat = renderer.getMaterials()[materialIndices[i]];
                ofs << "  MaterialID : " << mat.materialID << std::endl;
            }
        }
        ofs << "}" << std::endl;

        auto& nodes = renderer.sceneGraph.getNodes();
        uint32_t child = node.firstChild;
        while (child != 0) {
            writeNodes(nodes[child], renderer);
            child = nodes[child].nextSibling;
        }
        ofs << std::endl;
    }
};