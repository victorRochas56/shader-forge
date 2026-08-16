#pragma once
#include "scene_elements.hpp"
#include <unordered_map>

class Renderer;
class Scene;
class BindlessSystem;
class GUI;
struct RenderBuffers;

struct NodeGuiState {
    bool changingMesh = false;
    bool changingMaterials = false;
    char textBuffer[256] = {};
    std::vector<std::string> materialList;
    bool lightShadow = false;
    bool keepMaterialAssignments = false;
    char emitterTexBuffer[256] = {}; // emitter texture path (browse field)
    bool emitterTexInit = false;     // seeded emitterTexBuffer from the current texture yet?
    // Unit scale baked into the imported file's geometry (e.g. 0.01 for a cm-authored model).
    float importScale = 1.0f;
};

NodeGuiState& getNodeGuiState(uint32_t nodeIndex);

void showNodeInfo(GUI& gui, Node& node, Scene& scene, BindlessSystem& bindless, RenderBuffers& buffers);
void showTransformModifiers(GUI& gui);
void showNodeMeshInfo(GUI& gui, Node& node, Scene& scene);
void showNodeMaterialDialog(GUI& gui, Node& node, Scene& scene);
void showNodeLightInfo(GUI& gui, Node& node, Scene& scene, BindlessSystem& bindless, uint32_t lightBufferIndex);
void showNodeVolumeInfo(GUI& gui, Node& node, Scene& scene);
void showNodeEmitterInfo(GUI& gui, Node& node, Scene& scene, BindlessSystem& bindless, const RenderBuffers& buffers);
void showNodeTransformInfo(GUI& gui, Node& node, Scene& scene);

