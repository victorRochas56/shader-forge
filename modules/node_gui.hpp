#pragma once
#include "scene_elements.hpp"
#include <unordered_map>

class Renderer;
class Scene;
class BindlessSystem;

struct NodeGuiState {
    bool changingMesh = false;
    bool changingMaterials = false;
    char textBuffer[256] = {};
    std::vector<std::string> materialList;
    bool lightShadow = false;
    bool keepMaterialAssignments = false;
};

NodeGuiState& getNodeGuiState(uint32_t nodeIndex);

void showNodeInfo(Node& node, Scene& scene, BindlessSystem& bindless, uint32_t lightBufferIndex, uint32_t volumeBufferIndex);
void showNodeMeshInfo(Node& node, Scene& scene);
void showNodeMaterialDialog(Node& node, Scene& scene);
void showNodeLightInfo(Node& node, Scene& scene, BindlessSystem& bindless, uint32_t lightBufferIndex);
void showNodeVolumeInfo(Node& node, Scene& scene, BindlessSystem& bindless, uint32_t volumeBufferIndex);
void showNodeTransformInfo(Node& node, Scene& scene);

