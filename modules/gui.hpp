#pragma once
#include "GUI.h"

#include <string>
#include <filesystem>
#include <vulkan/vulkan.hpp>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <commdlg.h>
#endif

// Forward declarations
class Scene;
class SceneLoader;
class SceneGraph;
class BindlessSystem;
class DescriptorSet;
struct RenderBuffers;
class AssetManager;
class Device;
class Swapchain;
struct GLFWwindow;
struct Camera;
struct DrawIndexedIndirectCommand;
class Node;
class GUI;
struct MaterialEditorState;
struct RenderFeatures;

//helper functions for the GUI windows

#ifdef _WIN32
static std::string openFileDialog(const char* filter = "All Files\0*.*\0") {
    char filename[MAX_PATH] = "";
    OPENFILENAMEA ofn = {};
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFilter = filter;
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
    if (GetOpenFileNameA(&ofn)) {
        std::filesystem::path absPath(filename);
        std::filesystem::path relPath = std::filesystem::relative(absPath, std::filesystem::current_path());
        std::string result = relPath.generic_string(); // forward slashes
        return result;
    }
    return "";
}

static void browseButton(GUI& gui, const char* id, char* pathBuffer, size_t bufferSize) {
    gui.sameLine();
    std::string btnLabel = std::string("Browse##") + id;
    if (gui.button(btnLabel)) {
        std::string result = openFileDialog();
        if (!result.empty()) {
            strncpy(pathBuffer, result.c_str(), bufferSize);
            pathBuffer[bufferSize - 1] = '\0';
        }
    }
}
#endif
void traverseNodeTree(GUI& gui, Node& node, uint32_t level, uint32_t selectedNode, SceneGraph& sceneGraph);
void showMaterialEditor(GUI& gui, MaterialEditorState& state, Scene& scene, BindlessSystem& bindless);
void showImageViewList(GUI& gui, BindlessSystem& bindless, RenderFeatures& features);
void showActionMenu(GUI& gui, uint32_t context, GLFWwindow* window, Camera& camera, SceneGraph& sceneGraph, float posX, float posY);
void showToggles(GUI& gui, RenderFeatures& features);
void showScenesMenu(GUI& gui, Scene& scene, BindlessSystem& bindless, RenderBuffers& buffers,SceneLoader& sceneLoader);
void showBufferAllocs(GUI& gui, DescriptorSet& descriptorSet, AssetManager& assetManager, const std::vector<DrawIndexedIndirectCommand>& indirectDraws);
void showMeshList(GUI& gui, Scene& scene, BindlessSystem& bindless, uint32_t whiteTextureIndex);
void showTraces(GUI& gui);