#pragma once
#include <string>
#include <cstring>

// Forward declarations
class Scene;
struct Material;

struct MaterialEditorState {
    bool showEditor = false;
    int selectedIndex = -1; // -1 = creating new, >= 0 = editing existing

    // editable fields
    int selectedShaderIndex = 0; // index into Scene::getLitShaders(); 0 = default lit
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
    bool triplanar = false;
    float triplanarScale = 1.0f;
    float triplanarBlend = 0.1f;
    char envMapPaths[6][256] = {"", "", "", "", "", ""}; // posX, posY, posZ, negX, negY, negZ

    void resetToDefaults() {
        selectedIndex = -1;
        selectedShaderIndex = 0;
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
        triplanar = false;
        triplanarScale = 1.0f;
        triplanarBlend = 0.1f;
        for (int i = 0; i < 6; i++) envMapPaths[i][0] = '\0';
    }

    void loadFromMaterial(int index, Material& mat, Scene& scene);
};