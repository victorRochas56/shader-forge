#pragma once
#include "scene_elements.hpp"
#include <unordered_map>

class Renderer;

struct NodeGuiState {
    bool changingMesh = false;
    bool changingMaterials = false;
    char textBuffer[256] = {};
    std::vector<std::string> materialList;
    bool lightShadow = false;
    bool keepMaterialAssignments = false;
};

NodeGuiState& getNodeGuiState(uint32_t nodeIndex);

void showNodeInfo(Node& node, Renderer& renderer);
void showNodeMeshInfo(Node& node, Renderer& renderer);
void showNodeMaterialDialog(Node& node, Renderer& renderer);
void showNodeLightInfo(Node& node, Renderer& renderer);
void showNodeTransformInfo(Node& node, Renderer& renderer);

