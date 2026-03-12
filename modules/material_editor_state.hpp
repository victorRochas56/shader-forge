#pragma once
#include <string>
#include <sstream>
#include <cstring>
#include "renderer.hpp"

struct MaterialEditorState {
    bool showEditor = false;
    int selectedIndex = -1; // -1 = creating new, >= 0 = editing existing

    // editable fields
    char nameBuffer[128] = "New Material";
    float color[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float metallic = 0.0f;
    float roughness = 0.5f;
    char albedoPath[256] = "";
    char metallicPath[256] = "";
    char roughnessPath[256] = "";
    char normalPath[256] = "";
    bool flipNormal = false;
    bool alphaClip = false;
    float alphaCutoff = 0.5f;
    char envMapPaths[6][256] = {"", "", "", "", "", ""}; // posX, posY, posZ, negX, negY, negZ

    void resetToDefaults() {
        selectedIndex = -1;
        strncpy(nameBuffer, "New Material", sizeof(nameBuffer));
        color[0] = 1.0f; color[1] = 1.0f; color[2] = 1.0f; color[3] = 1.0f;
        metallic = 0.0f;
        roughness = 0.5f;
        albedoPath[0] = '\0';
        metallicPath[0] = '\0';
        roughnessPath[0] = '\0';
        normalPath[0] = '\0';
        flipNormal = false;
        alphaClip = false;
        alphaCutoff = 0.5f;
        for (int i = 0; i < 6; i++) envMapPaths[i][0] = '\0';
    }

    void loadFromMaterial(int index, Material& mat, Renderer* renderer) {
        selectedIndex = index;
        strncpy(nameBuffer, mat.name.empty() ? ("Material " + std::to_string(index)).c_str() : mat.name.c_str(), sizeof(nameBuffer));
        color[0] = mat.color.r; color[1] = mat.color.g; color[2] = mat.color.b; color[3] = mat.color.a;
        metallic = mat.metallic;
        roughness = mat.roughness;

        auto copyPath = [](char* dest, size_t destSize, const std::string& src) {
            strncpy(dest, src.c_str(), destSize);
            dest[destSize - 1] = '\0';
        };
        copyPath(albedoPath, sizeof(albedoPath), renderer->assetManager.getTexturePathFromIndex(mat.albedoTextureIndex));
        copyPath(metallicPath, sizeof(metallicPath), renderer->assetManager.getTexturePathFromIndex(mat.metallicTextureIndex));
        copyPath(roughnessPath, sizeof(roughnessPath), renderer->assetManager.getTexturePathFromIndex(mat.roughnessTextureIndex));
        copyPath(normalPath, sizeof(normalPath), renderer->assetManager.getTexturePathFromIndex(mat.normalTextureIndex));

        flipNormal = (mat.flags & MaterialFlags::FLIP_NORMAL) != 0;
        alphaClip = mat.alphaClip;
        alphaCutoff = mat.alphaCutoff;

        std::string cubemapPath = renderer->assetManager.getCubemapPathFromIndex(mat.environmentMapIndex);
        for (int i = 0; i < 6; i++) envMapPaths[i][0] = '\0';
        if (!cubemapPath.empty()) {
            // Format: "posX|negX|posY|negY|posZ|negZ"
            std::stringstream ss(cubemapPath);
            std::string part;
            int idx = 0;
            while (std::getline(ss, part, '|') && idx < 6) {
                strncpy(envMapPaths[idx], part.c_str(), sizeof(envMapPaths[idx]));
                envMapPaths[idx][sizeof(envMapPaths[idx]) - 1] = '\0';
                idx++;
            }
        }
    }
};