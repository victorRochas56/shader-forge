#pragma once
#include "imgui.h"

#include <string>
#include <filesystem>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <commdlg.h>
#endif

// Forward declarations
class Renderer;
class SceneManager;
class Node;
struct MaterialEditorState;

//helper functions for IMGUI

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

static void browseButton(const char* id, char* pathBuffer, size_t bufferSize) {
    ImGui::SameLine();
    std::string btnLabel = std::string("Browse##") + id;
    if (ImGui::Button(btnLabel.c_str())) {
        std::string result = openFileDialog();
        if (!result.empty()) {
            strncpy(pathBuffer, result.c_str(), bufferSize);
            pathBuffer[bufferSize - 1] = '\0';
        }
    }
}
#endif
void initIMGUI(Renderer* renderer);
void traverseNodeTree(Node& node, uint32_t level, uint32_t selectedNode, Renderer* renderer);
void showMaterialEditor(MaterialEditorState& state, Renderer* renderer);
void showImageViewList(Renderer* renderer);
void showActionMenu(uint32_t context, Renderer* renderer, float posX, float posY);
void showToggles(Renderer* renderer);
void showScenesMenu(SceneManager* sceneManager);
void showBufferAllocs(Renderer* renderer);
void showDebugWindow(Renderer* renderer);
void showTraces();