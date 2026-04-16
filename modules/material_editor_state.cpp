#include "material_editor_state.hpp"
#include "renderer.hpp"
#include <sstream>

void MaterialEditorState::loadFromMaterial(int index, Material& mat, Renderer* renderer) {
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