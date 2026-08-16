#include "gui.hpp"
#include "GUI.h"
#include "input.hpp"
#include "profiling.hpp"
#include "material_editor_state.hpp"
#include "pipelines.hpp"
#include "swapchain.hpp"
#include "scene_loader.hpp"
#include "scene.hpp"
#include "raycast.hpp"
#include "node_ops.hpp"
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <unordered_map>

void traverseNodeTree(GUI& gui, Node& node, uint32_t level, uint32_t selectedNode, SceneGraph& sceneGraph) {
    std::string displayText = "";
    GUITreeFlags flag = GUITreeFlags::TreeDefaultOpen;
    if(!node.internal) {
        if (node.getIndex() == selectedNode) {
            flag = flag | GUITreeFlags::TreeSelected;
        }
        // Append "##<index>" so the ID stays unique even if two nodes share a display name
        // (the text after ## is not drawn). Node names are uniquified on spawn, but this guards
        // legacy/default-named nodes too.
        displayText += node.name;
        displayText += "##" + std::to_string(node.getIndex());
        if(node.firstChild == 0) flag = flag | GUITreeFlags::TreeLeaf;
        bool open = gui.treeNodeEx(displayText, flag);
        // Read the click outside the open branch: a Leaf never opens, and nesting the test the way
        // the ImGui version did would make a childless node impossible to select.
        if(gui.isItemClicked())
            sceneGraph.selectNode(node.getIndex());
        if(open){
            auto& nodes = sceneGraph.getNodes();
            uint32_t child = node.firstChild;
            while (child != 0) {
                traverseNodeTree(gui, nodes[child], level + 1, selectedNode, sceneGraph);
                child = nodes[child].nextSibling;
            }
            gui.treePop();
        }
    }
}

void showMaterialEditor(GUI& gui, MaterialEditorState& state, Scene& scene, BindlessSystem& bindless, RenderFeatures& features) {
    auto& materials = scene.getMaterials();

    // This list is the only consumer of the material thumbnails, and the pass redraws every one of
    // them every frame. Tell it whether the list is actually on screen. drawGui runs before
    // drawFrame, so the pass reads this the same frame it is written.
    bool listVisible = gui.beginWindow("Materials", nullptr, glm::vec2(300, 420), glm::vec2(20, 20));
    features.materialPreviewsVisible = listVisible;
    if (!listVisible) return;

    // No thumbnail descriptor cache. The custom GUI draws any bindless texture by index, so the
    // per-image-view descriptor map this used to keep is simply gone.
    const std::vector<TextureResource>& textures = bindless.descriptorSet->getTextureResources();

    // material list
    for (int i = 0; i < static_cast<int>(materials.size()); i++) {
        uint32_t ti = materials[i].thumbnailTextureIndex;
        if (ti != 0xFFFFFFFF && ti < textures.size() && !textures[ti].isEmpty()) {
            gui.image(ti, glm::vec2(128.0f, 128.0f));
        }

        std::string label = materials[i].name.empty() ? ("Material " + std::to_string(i)) : materials[i].name;
        if (i == 0) label += " (default)";

        bool isSelected = (state.showEditor && state.selectedIndex == i);
        if (gui.selectable(label, isSelected)) {
            state.showEditor = true;
            state.loadFromMaterial(i, materials[i], scene);
            InputManager::getInstance().canMove = false;
        }
    }

    gui.separator();
    if (gui.button("Create New Material")) {
        state.showEditor = true;
        state.resetToDefaults();
        InputManager::getInstance().canMove = false;
    }
    if (gui.button("Pick from Mesh")) {
        InputManager::getInstance().materialPickMode = true;
        InputManager::getInstance().pickedMaterialIndex = -1;
    }
    if (InputManager::getInstance().materialPickMode) {
        gui.textColored("Click a mesh to select its material...", glm::vec4(1, 1, 0, 1));
    }

    // handle pick result
    auto& input = InputManager::getInstance();
    if (input.pickedMaterialIndex >= 0 && input.pickedMaterialIndex < static_cast<int>(materials.size())) {
        state.showEditor = true;
        state.loadFromMaterial(input.pickedMaterialIndex, materials[input.pickedMaterialIndex], scene);
        input.canMove = false;
        input.pickedMaterialIndex = -1;
    }

    gui.endWindow();

    if (!state.showEditor) return;

    bool open = true;
    std::string title = state.selectedIndex >= 0 ? "Edit Material" : "New Material";
    if (!gui.beginWindow(title.c_str(), &open, glm::vec2(440, 600), glm::vec2(420, 20))) return;
    if (!open) {
        state.showEditor = false;
        InputManager::getInstance().canMove = true;
        gui.endWindow();
        return;
    }

    InputManager::getInstance().canMove = false;

    gui.setNextItemWidth(200);
    gui.inputText("Name", state.nameBuffer, sizeof(state.nameBuffer));

    // shader selection — lists all registered lit / lit-derived shaders
    {
        const auto& litShaders = scene.getLitShaders();
        auto shaderLabel = [](const std::string& path) {
            size_t slash = path.find_last_of("/\\");
            std::string name = (slash == std::string::npos) ? path : path.substr(slash + 1);
            size_t dot = name.find_last_of('.');
            return (dot == std::string::npos) ? name : name.substr(0, dot);
        };
        if (state.selectedShaderIndex < 0 || state.selectedShaderIndex >= static_cast<int>(litShaders.size()))
            state.selectedShaderIndex = 0;
        std::string preview = litShaders.empty() ? "<none>" : shaderLabel(litShaders[state.selectedShaderIndex].sourceFile);
        gui.setNextItemWidth(200);
        if (gui.beginCombo("Shader##mat", preview)) {
            for (int i = 0; i < static_cast<int>(litShaders.size()); i++) {
                bool sel = (state.selectedShaderIndex == i);
                if (gui.comboItem(shaderLabel(litShaders[i].sourceFile), sel))
                    state.selectedShaderIndex = i;
            }
            gui.endCombo();
        }
    }

    gui.separator();
    gui.textf("Properties");
    gui.colorEdit4("Base Color", state.color);
    gui.sliderFloat("Metallic##mat", &state.metallic, 0.0f, 1.0f);
    gui.sliderFloat("Roughness##mat", &state.roughness, 0.0f, 1.0f);

    gui.separator();
    gui.textf("Textures (leave empty for defaults)");
    gui.setNextItemWidth(300);
    gui.inputText("Albedo", state.albedoPath, sizeof(state.albedoPath));
    browseButton(gui, "albedo", state.albedoPath, sizeof(state.albedoPath));
    gui.setNextItemWidth(300);
    gui.inputText("Metallic Tex", state.metallicPath, sizeof(state.metallicPath));
    browseButton(gui, "metallic", state.metallicPath, sizeof(state.metallicPath));
    gui.setNextItemWidth(300);
    gui.inputText("Roughness Tex", state.roughnessPath, sizeof(state.roughnessPath));
    browseButton(gui, "roughness", state.roughnessPath, sizeof(state.roughnessPath));
    gui.setNextItemWidth(300);
    gui.inputText("Normal", state.normalPath, sizeof(state.normalPath));
    browseButton(gui, "normal", state.normalPath, sizeof(state.normalPath));
    gui.checkbox("Flip Normal", &state.flipNormal);
    gui.checkbox("Alpha Clip", &state.alphaClip);
    if (state.alphaClip) {
        gui.sliderFloat("Alpha Cutoff", &state.alphaCutoff, 0.0f, 1.0f);
    }

    gui.separator();
    gui.textf("Environment Map (6 faces)");
    const char* faceLabels[] = {"Pos X", "Pos Y", "Pos Z", "Neg X", "Neg Y", "Neg Z"};
    for (int i = 0; i < 6; i++) {
        gui.setNextItemWidth(300);
        gui.inputText(faceLabels[i], state.envMapPaths[i], sizeof(state.envMapPaths[i]));
        browseButton(gui, faceLabels[i], state.envMapPaths[i], sizeof(state.envMapPaths[i]));
    }

    gui.separator();

    auto buildMaterial = [&](Material& mat) {
        const Material& defaultMat = materials[scene.getFallBackMaterial()];

        mat.name = state.nameBuffer;
        const auto& litShaders = scene.getLitShaders();
        if (state.selectedShaderIndex >= 0 && state.selectedShaderIndex < static_cast<int>(litShaders.size()))
            mat.shaderSource = litShaders[state.selectedShaderIndex];
        else
            mat.shaderSource = scene.getFallBackShader();
        mat.color = glm::vec4(state.color[0], state.color[1], state.color[2], state.color[3]);
        mat.metallic = state.metallic;
        mat.roughness = state.roughness;

        uint32_t matFlags = 0;

        // load textures
        if (strlen(state.albedoPath) > 0) {
            try {
                mat.albedoTextureIndex = scene.assetManager.loadTextureFromFile(state.albedoPath);
                matFlags |= MaterialFlags::HAS_ALBEDO;
            } catch (...) { mat.albedoTextureIndex = defaultMat.albedoTextureIndex; }
        } else {
            mat.albedoTextureIndex = defaultMat.albedoTextureIndex;
        }

        if (strlen(state.roughnessPath) > 0) {
            try {
                mat.roughnessTextureIndex = scene.assetManager.loadTextureFromFile(state.roughnessPath, vk::Format::eR8G8B8A8Unorm);
                matFlags |= MaterialFlags::HAS_ROUGHNESS;
            } catch (...) { mat.roughnessTextureIndex = defaultMat.roughnessTextureIndex; }
        } else {
            mat.roughnessTextureIndex = defaultMat.roughnessTextureIndex;
        }

        if (strlen(state.metallicPath) > 0) {
            try {
                mat.metallicTextureIndex = scene.assetManager.loadTextureFromFile(state.metallicPath, vk::Format::eR8G8B8A8Unorm);
                matFlags |= MaterialFlags::HAS_METALLIC;
            } catch (...) { mat.metallicTextureIndex = defaultMat.metallicTextureIndex; }
        } else {
            mat.metallicTextureIndex = defaultMat.metallicTextureIndex;
        }

        if (strlen(state.normalPath) > 0) {
            try {
                mat.normalTextureIndex = scene.assetManager.loadTextureFromFile(state.normalPath, vk::Format::eR8G8B8A8Unorm, textureconv::ColorSpace::NormalMap);
                matFlags |= MaterialFlags::HAS_NORMAL;
            } catch (...) { mat.normalTextureIndex = defaultMat.normalTextureIndex; }
        } else {
            mat.normalTextureIndex = defaultMat.normalTextureIndex;
        }
        if(state.flipNormal) matFlags |= MaterialFlags::FLIP_NORMAL;
        if(state.alphaClip) matFlags |= MaterialFlags::ALPHA_CLIP;

        mat.flags = static_cast<MaterialFlags>(matFlags);
        mat.alphaClip = state.alphaClip;
        mat.alphaCutoff = state.alphaCutoff;

        // environment map
        bool hasEnvMap = false;
        for (int i = 0; i < 6; i++) {
            if (strlen(state.envMapPaths[i]) > 0) { hasEnvMap = true; break; }
        }
        if (hasEnvMap) {
            try {
                // envMapPaths is in key order [posX, negX, posY, negY, posZ, negZ];
                // loadCubemapFromFile takes (posX, posY, posZ, negX, negY, negZ).
                mat.environmentMapIndex = scene.assetManager.loadCubemapFromFile(
                    state.envMapPaths[0],  // posX
                    state.envMapPaths[2],  // posY
                    state.envMapPaths[4],  // posZ
                    state.envMapPaths[1],  // negX
                    state.envMapPaths[3],  // negY
                    state.envMapPaths[5]); // negZ
            } catch (...) { mat.environmentMapIndex = defaultMat.environmentMapIndex; }
        } else {
            mat.environmentMapIndex = defaultMat.environmentMapIndex;
        }
    };

    if (state.selectedIndex >= 0) {
        // editing existing material
        if (gui.button("Apply Changes")) {
            Material& existingMat = materials[state.selectedIndex];
            Material oldMat = existingMat;

            Material newMat;
            buildMaterial(newMat);
            newMat.thumbnailTextureIndex = oldMat.thumbnailTextureIndex; // keep preview target; pass re-renders it
            newMat.materialID = static_cast<uint32_t>(std::hash<Material>{}(newMat));

            auto& nodes = scene.sceneGraph.getNodes();
            auto usesThisMaterial = [&](uint32_t i) {
                if (!scene.sceneGraph.isNodeValid(i)) return false;
                if (nodes[i].getMeshIndex() >= scene.assetManager.meshes.size()) return false;
                return nodes[i].getMaterialIndex() == static_cast<uint32_t>(state.selectedIndex);
            };

            // Remove the old render entries BEFORE overwriting materials[selectedIndex]: removeMeshFromShader
            // resolves the material index by value, so it must still match oldMat in the materials vector.
            for (uint32_t i = 1; i <= scene.sceneGraph.getLastNode(); i++) {
                if (usesThisMaterial(i)) scene.removeMeshFromShader(i, oldMat.shaderSource, oldMat);
            }

            existingMat = newMat; // commit the change

            // Re-register the nodes against the (possibly new) shader pipeline.
            for (uint32_t i = 1; i <= scene.sceneGraph.getLastNode(); i++) {
                if (usesThisMaterial(i)) scene.addMeshToShader(i, existingMat.shaderSource, existingMat);
            }

            state.showEditor = false;
            InputManager::getInstance().canMove = true;
        }
    } else {
        // creating new material
        if (gui.button("Create")) {
            Material newMat;
            buildMaterial(newMat);
            scene.addMaterial(newMat);

            state.showEditor = false;
            InputManager::getInstance().canMove = true;
        }
    }

    gui.sameLine();
    if (gui.button("Cancel")) {
        state.showEditor = false;
        InputManager::getInstance().canMove = true;
    }

    gui.endWindow();
}

void showImageViewList(GUI& gui, BindlessSystem& bindless, RenderFeatures& features) {
    if (!gui.beginWindow("Image Views", nullptr, glm::vec2(360, 280), glm::vec2(200, 200))) return;

    ImageVisSettings& vis = features.imageVis;
    if (gui.button("Clear Selection")) {
        vis.imageIndex = 0xFFFFFFFF;
    }

    int i = 0;
    for (const TextureResource& img : (*bindless.descriptorSet).getTextureResources()) {
        if(img.source.empty()){
            i++;
            continue;
        }
        bool selected = (vis.imageIndex == static_cast<uint32_t>(i));
        if (gui.buttonToggled(img.source, selected)) {
            vis.imageIndex = (selected) ? 0xFFFFFFFF : i;
        }
        if (selected) {
            gui.sameLine();
            if (gui.button("B&W")){
                vis.flags ^= ImageVisFlags::B_W_IMAGE;
            }
            gui.sameLine();
            if (gui.button("Flip Y")){
                vis.flags ^= ImageVisFlags::FLIP_VERTICAL;
            }
            gui.sameLine();
            if (gui.button("Slicemap")){
                vis.flags ^= ImageVisFlags::SLICEMAP;
            }
            if ((vis.flags & ImageVisFlags::SLICEMAP) != 0) {
                gui.sameLine();
                if (gui.button("Single Slice")){
                    vis.flags ^= ImageVisFlags::SLICEMAP_SLICE;
                }
            }
        }
        i++;
    }

    gui.endWindow();
}

void showActionMenu(GUI& gui, uint32_t context, GLFWwindow* window, Camera& camera,
                    SceneGraph& sceneGraph, float posX, float posY) {
    gui.setNextWindowPos(glm::vec2(posX, posY));
    if (!gui.beginWindow("Action", nullptr, glm::vec2(160, 80))) return;
    if (gui.button("Add Node")) {
        glm::vec3 origin, direction;
        int w = 0, h = 0;
        glfwGetWindowSize(window, &w, &h);
        camera.rayFromScreenCoords((posX / w) * 2.0 - 1.0, (posY / h) * 2.0 - 1.0, origin, direction);
        sceneGraph.addNode(false, SceneGraph::ROOT_INDEX, origin + direction);
    }
    gui.endWindow();
}

void showToggles(GUI& gui, RenderFeatures& f){
    if(!gui.beginWindow("Toggles", nullptr, glm::vec2(360, 560), glm::vec2(-510, 0), GUIAnchor::Top | GUIAnchor::Right, GUIWindowFixed | GUIWindowNoSavedSettings)) return;
    if(gui.button("Gizmos")){
        f.showGizmos = !f.showGizmos;
    }
    // In slicemap mode the same field selects one of the 32 slices instead of a mip.
    bool slicemap = (f.imageVis.flags & ImageVisFlags::SLICEMAP) != 0;
    gui.sliderInt(slicemap ? "Slice" : "ImageMip", &f.imageVis.mipLevel, 0, slicemap ? 31 : 6);
    if(gui.button("Depth Buffer")){
        f.imageVis.flags ^= ImageVisFlags::LINEARIZE;
    }
    if(gui.button("SSAO")){
        f.ssao.enabled = !f.ssao.enabled;
    }
    if(f.ssao.enabled){
        gui.sliderFloat("AO Radius", &f.ssao.radius, 0.01f, 10.0f);
        gui.sliderFloat("AO Bias", &f.ssao.bias, 0.01f, 0.1f);
        gui.sliderFloat("AO Power", &f.ssao.power, 0.01f, 5.0f);
    }
    if(gui.button("SSR")){
        f.ssr.enabled = !f.ssr.enabled;
    }
    if(f.ssr.enabled){
        gui.sliderInt("Max Steps", &f.ssr.maxSteps, 16, 128);
        gui.sliderFloat("Thickness", &f.ssr.thickness, 0.01f, 5.0f);
        gui.sliderFloat("Roughness Threshold", &f.ssr.roughnessThreshold, 0.0f, 1.0f);
        gui.sliderFloat("Temporal Blend", &f.ssr.temporalBlend, 0.01f, 1.0f);
        if(gui.sliderFloat("Resolution Scale", &f.ssr.resolutionScale, 0.25f, 1.0f)){
            f.ssr.resolutionDirty = true;
        }
    }
    if(gui.button("Volumetrics")){
        f.volumetrics.enabled = !f.volumetrics.enabled;
    }
    if(f.volumetrics.enabled){
        // Grid Far bounds the froxel volume (media beyond it isn't rendered). Scattering phase is a
        // per-volume property (edit it on each Volume node). Debug View overwrites the scene with a
        // false-color visualization of a grid volume to diagnose injection/lighting.
        gui.sliderFloat("Grid Far", &f.volumetrics.gridFar, 5.0f, 200.0f);
        const char* dbgItems[] = {"Off", "Extinction", "Scatter", "In-Scatter", "Transmittance", "Slice", "Phase g", "Shadow"};
        gui.combo("Debug View", &f.volumetrics.debugView, dbgItems, guiArraySize(dbgItems));
    }
    gui.checkbox("Voxelize Scene", &f.voxelDebug.voxelizeScene);
    if(gui.button("Voxel Grid")){
        f.voxelDebug.enabled = !f.voxelDebug.enabled;
    }
    if(f.voxelDebug.enabled){
        // Mip Level walks the chain the resolve pass builds — stepping up should blur the grid
        // coherently, not fade it toward black.
        gui.checkbox("Draw Cubes", &f.voxelDebug.drawCubes);
        const char* volItems[] = {"Radiance", "Irradiance +X", "Irradiance -X", "Irradiance +Y", "Irradiance -Y", "Irradiance +Z", "Irradiance -Z"};
        gui.combo("Volume", &f.voxelDebug.volumeSelect, volItems, guiArraySize(volItems));
        gui.setItemTooltip("Radiance volume, or one ambient-cube irradiance face (single mip).");
        int voxMip = static_cast<int>(f.voxelDebug.mipLevel);
        if(gui.sliderInt("Voxel Mip", &voxMip, 0, 7)){
            f.voxelDebug.mipLevel = static_cast<uint32_t>(voxMip);
        }
        if(f.voxelDebug.drawCubes){
            gui.sliderFloat("Cube Threshold", &f.voxelDebug.cubeThreshold, 0.001f, 1.0f, "%.3f");
            gui.setItemTooltip("Min coverage for a voxel to get a cube; lower to see partially-covered mips.");
        } else {
            const char* voxItems[] = {"Radiance", "Occupancy", "Albedo (normalized)", "First-hit Depth"};
            int voxMode = static_cast<int>(f.voxelDebug.mode);
            if(gui.combo("Voxel View", &voxMode, voxItems, guiArraySize(voxItems))){
                f.voxelDebug.mode = static_cast<VoxelDebugMode>(voxMode);
            }
            gui.sliderFloat("Voxel Opacity", &f.voxelDebug.alphaScale, 0.1f, 10.0f);
            gui.setItemTooltip("Multiplies per-voxel coverage; raise it to read a sparse grid.");
            int voxSteps = static_cast<int>(f.voxelDebug.maxSteps);
            if(gui.sliderInt("Voxel Max Steps", &voxSteps, 32, 1024)){
                f.voxelDebug.maxSteps = static_cast<uint32_t>(voxSteps);
            }
        }
    }
    gui.separatorText("VXGI");
    const char* giModes[] = {"Per-pixel cone trace", "Ambient cube lookup"};
    gui.combo("GI Mode", &f.vxgi.mode, giModes, guiArraySize(giModes));
    gui.setItemTooltip("Cone trace per pixel, or sample the per-voxel 6-face irradiance volumes.");
    gui.sliderFloat("GI Strength", &f.vxgi.strength, 0.0f, 4.0f);
    gui.sliderFloat("GI Sky Strength", &f.vxgi.skyStrength, 0.0f, 2.0f);
    gui.setItemTooltip("Sky collected by cones that leave the grid. Occlusion-aware: open areas gain, enclosed ones don't.");
    gui.sliderFloat("GI Sky Injection", &f.vxgi.skyInjection, 0.0f, 2.0f);
    gui.setItemTooltip("Sky added to every voxelized surface so sky-lit geometry bounces. Visibility is assumed 1, so this is a flat ambient — raising it washes out the contrast from Sky Strength.");
    if (f.vxgi.mode == 0) {
        gui.sliderInt("GI Side Cones", &f.vxgi.hemisphereRays, 0, 5);
        gui.sliderInt("GI Steps", &f.vxgi.maxSteps, 1, 64);
        gui.setItemTooltip("Rounds down to a multiple of GI fetch batch");
        gui.sliderInt("GI Fetch Batch", &f.vxgi.fetchBatch, 1, 8);
    } else {
        gui.sliderInt("Gather Side Cones", &f.vxgi.gatherSideCones, 0, 5);
        gui.sliderInt("Gather Steps", &f.vxgi.gatherSteps, 1, 64);
        gui.setItemTooltip("Rounds down to a multiple of Gather fetch batch.");
        gui.sliderInt("Gather Fetch Batch", &f.vxgi.gatherFetchBatch, 1, 8);
        gui.sliderFloat("Gather Blend", &f.vxgi.temporalBlend, 0.02f, 1.0f, "%.2f");
        gui.setItemTooltip("Weight a fresh gather gets against the reprojected history. 1 = no temporal filtering, which flickers as the grid rebins geometry; under ~0.05 the 8-bit faces stop converging.");
        const char* rateItems[] = {"Every frame", "1/2 per frame", "1/4 per frame", "1/8 per frame"};
        int rateSel = f.vxgi.updatePhases >= 8 ? 3 : f.vxgi.updatePhases >= 4 ? 2 : f.vxgi.updatePhases >= 2 ? 1 : 0;
        if(gui.combo("Gather Rate", &rateSel, rateItems, guiArraySize(rateItems))){
            f.vxgi.updatePhases = 1 << rateSel;
        }
        gui.setItemTooltip("Share of voxels re-traced per frame, in 4-voxel blocks; the rest carry history forward. Cuts gather cost, at proportionally slower GI response.");
    }

    if(gui.button("Show BBOXes")){
        f.showBBoxes = !f.showBBoxes;
    }

    gui.separatorText("Sky");
    gui.sliderFloat("Skybox Intensity", &f.skyboxIntensity, 0.0f, 20.0f);

    gui.separatorText("Tonemap / Exposure");
    const char* tonemapOps[] = {"Reinhard", "ACES", "None"};
    int op = static_cast<int>(f.tonemap.op);
    if(gui.combo("Operator", &op, tonemapOps, guiArraySize(tonemapOps))){
        f.tonemap.op = static_cast<uint32_t>(op);
    }
    gui.checkbox("Auto Exposure", &f.tonemap.autoExposure);
    if(f.tonemap.autoExposure){
        gui.sliderFloat("Exposure Comp (EV)", &f.tonemap.exposureComp, -8.0f, 8.0f, "%.2f");
        gui.dragFloatRange2("EV clamp", &f.tonemap.minEV, &f.tonemap.maxEV, 0.1f, -10.0f, 20.0f, "min %.1f", "max %.1f");
        gui.sliderFloat("Adaptation Speed", &f.tonemap.adaptationSpeed, 0.1f, 10.0f, "%.2f /s");
    } else {
        gui.sliderFloat("Exposure (EV100)", &f.tonemap.ev100, -2.0f, 20.0f, "%.2f");
        gui.textf("Exposure factor: %.3e", computeExposure(f.tonemap));
    }

    gui.endWindow();
}

void showScenesMenu(GUI& gui, Scene& scene, BindlessSystem& bindless, RenderBuffers& buffers,SceneLoader& sceneLoader){
    // Offset is signed screen px from the anchored corner, so a Right anchor insets with -x.
    if(!gui.beginWindow("Scene Manager", nullptr, glm::vec2(510, 90), glm::vec2(0, 0),
                        GUIAnchor::Top | GUIAnchor::Right, GUIWindowFixed | GUIWindowNoSavedSettings)) return;
    static char saveName[128] = "scene";
    gui.setNextItemWidth(150.0f);
    gui.inputTextWithHint("##saveName", "scene name", saveName, sizeof(saveName));
    gui.sameLine();
    if(gui.button("Save Scene") && saveName[0] != '\0'){
        std::string path(saveName);
        if(std::filesystem::path(path).extension() != ".scn") path += ".scn";
        sceneLoader.saveScene(path, scene);
    }
    gui.sameLine();
    if(gui.button("Load Scene")){
        std::string path = openFileDialog("Scene Files\0*.scn\0All Files\0*.*\0");
        if(!path.empty()){
            sceneLoader.loadScene(path, scene, bindless, buffers,
                                   buffers.modelMatrixBufferIndex,
                                   buffers.lightBufferIndex);
        }
    }
    gui.sameLine();
    if(gui.button("Clear Scene")){
        sceneLoader.clearScene(scene, bindless, buffers,
                                buffers.modelMatrixBufferIndex,
                                buffers.lightBufferIndex);
    }
    gui.endWindow();
}

// ---- skybox picker ----

// One cubemap on disk. Faces are in loadCubemapFromFile's argument order.
struct SkyboxSet {
    std::string name;     // path relative to the scan root, what the list shows
    std::string faces[6]; // posX, posY, posZ, negX, negY, negZ
};

// The key AssetManager files a loaded cubemap under. Its face order is not the load call's.
static std::string skyboxKey(const SkyboxSet& set) {
    return set.faces[0] + "|" + set.faces[3] + "|" + set.faces[1] + "|" + set.faces[4] + "|" + set.faces[2] + "|" + set.faces[5];
}

// Fills `faces` from `dir`, or returns false if any of the six is missing. Resolves to the source
// images; loadCubeMapFromFile picks up the .ktx2 cache next to them on its own.
static bool collectCubemapFaces(const std::filesystem::path& dir, std::string (&faces)[6]) {
    static const char* faceNames[6] = {"posx", "posy", "posz", "negx", "negy", "negz"};
    static const char* extensions[] = {".jpg", ".jpeg", ".png", ".tga", ".bmp", ".hdr"};
    for (int i = 0; i < 6; i++) {
        faces[i].clear();
        for (const char* ext : extensions) {
            std::filesystem::path candidate = dir / (std::string(faceNames[i]) + ext);
            std::error_code ec;
            if (std::filesystem::exists(candidate, ec)) {
                faces[i] = candidate.generic_string();
                break;
            }
        }
        if (faces[i].empty()) return false;
    }
    return true;
}

// Every folder under `root` (root itself included) holding a full set of faces.
static std::vector<SkyboxSet> scanSkyboxes(const std::string& root) {
    std::vector<SkyboxSet> sets;
    std::error_code rootEc;
    if (!std::filesystem::is_directory(root, rootEc)) return sets;

    auto consider = [&](const std::filesystem::path& dir) {
        SkyboxSet set;
        if (!collectCubemapFaces(dir, set.faces)) return;
        std::error_code ec;
        std::filesystem::path rel = std::filesystem::relative(dir, root, ec);
        set.name = (ec || rel.empty() || rel == ".") ? std::filesystem::path(root).generic_string() : rel.generic_string();
        sets.push_back(std::move(set));
    };

    consider(root);
    std::error_code walkEc;
    auto end = std::filesystem::recursive_directory_iterator();
    auto it = std::filesystem::recursive_directory_iterator(root, std::filesystem::directory_options::skip_permission_denied, walkEc);
    for (; !walkEc && it != end; it.increment(walkEc)) {
        std::error_code entryEc;
        if (!std::filesystem::is_directory(it->path(), entryEc) || entryEc) continue;
        if (it.depth() >= 2) it.disable_recursion_pending(); // asset trees get deep, sky folders do not
        consider(it->path());
    }

    std::sort(sets.begin(), sets.end(), [](const SkyboxSet& a, const SkyboxSet& b) { return a.name < b.name; });
    return sets;
}

// Folder of a stored cubemap key, ie. "textures/sky2".
static std::string skyboxLabelFromKey(const std::string& key) {
    if (key.empty()) return "<none>";
    std::string posX = key.substr(0, key.find('|'));
    std::string folder = std::filesystem::path(posX).parent_path().generic_string();
    return folder.empty() ? posX : folder;
}

void showSkyboxMenu(GUI& gui, Scene& scene) {
    if (!gui.beginWindow("Skybox", nullptr, glm::vec2(300, 260), glm::vec2(20, 790))) return;

    // Scanned once and again on demand — the folder list only changes when the user adds files.
    static std::vector<SkyboxSet> sets = scanSkyboxes("textures");
    static std::string status;

    std::string currentKey = scene.assetManager.getCubemapPathFromIndex(scene.getSkyBox());
    gui.textf("Current: %s", skyboxLabelFromKey(currentKey).c_str());
    gui.separator();

    // Drags along every material that was pointed at the old sky: the fallback material is wired
    // that way at startup and anything without its own cubemap inherits it. Custom ones are left be.
    auto applySkybox = [&](const SkyboxSet& set) {
        try {
            uint32_t index = scene.assetManager.loadCubemapFromFile(set.faces[0], set.faces[1], set.faces[2],
                                                                    set.faces[3], set.faces[4], set.faces[5]);
            uint32_t previous = scene.getSkyBox();
            scene.setSkyBox(index);
            for (Material& mat : scene.getMaterials()) {
                if (mat.environmentMapIndex == previous) mat.environmentMapIndex = index;
            }
            status = "Loaded " + set.name;
        } catch (const std::exception& e) {
            status = std::string("Failed: ") + e.what();
        } catch (...) {
            status = "Failed to load " + set.name;
        }
    };

    if (sets.empty()) gui.textColored("No cubemaps found under textures/", glm::vec4(1, 1, 0, 1));
    for (const SkyboxSet& set : sets) {
        if (gui.selectable(set.name, skyboxKey(set) == currentKey)) applySkybox(set);
        gui.setItemTooltip(set.faces[0]);
    }

    gui.separator();
    if (gui.button("Browse...")) {
        std::string face = openFileDialog("Cubemap Face\0*.jpg;*.jpeg;*.png;*.tga;*.bmp;*.hdr\0All Files\0*.*\0");
        if (!face.empty()) {
            std::filesystem::path dir = std::filesystem::path(face).parent_path();
            SkyboxSet set;
            if (collectCubemapFaces(dir, set.faces)) {
                set.name = dir.generic_string();
                applySkybox(set);
            } else {
                status = "No posx/negx/... set in " + dir.generic_string();
            }
        }
    }
    gui.setItemTooltip("Pick any face of a folder holding posx/negx/posy/negy/posz/negz.");
    gui.sameLine();
    if (gui.button("Rescan")) sets = scanSkyboxes("textures");

    // A sky seen for the first time is encoded to a .ktx2 cache here, which stalls for a few seconds.
    if (!status.empty()) gui.textf("%s", status.c_str());

    gui.endWindow();
}

// Format a byte count as B / KB / MB / GB. Picks the largest unit the value comfortably fits.
static std::string formatBytes(vk::DeviceSize bytes) {
    constexpr double KB = 1024.0;
    constexpr double MB = 1024.0 * 1024.0;
    constexpr double GB = 1024.0 * 1024.0 * 1024.0;
    char buf[32];
    if (bytes >= (vk::DeviceSize)GB)      std::snprintf(buf, sizeof(buf), "%.2f GB", bytes / GB);
    else if (bytes >= (vk::DeviceSize)MB) std::snprintf(buf, sizeof(buf), "%.2f MB", bytes / MB);
    else if (bytes >= (vk::DeviceSize)KB) std::snprintf(buf, sizeof(buf), "%.2f KB", bytes / KB);
    else                                   std::snprintf(buf, sizeof(buf), "%llu B", (unsigned long long)bytes);
    return std::string(buf);
}

// Geometric series for a full mip pyramid down to 1x1: sum of 4^-i for i=0..mipLevels-1.
// mipLevels=1 → 1.0 (no mip overhead), mipLevels→∞ → 4/3.
static float mipChainFactor(uint32_t mipLevels) {
    if (mipLevels <= 1) return 1.0f;
    return (1.0f - std::pow(0.25f, (float)mipLevels)) / 0.75f;
}

void showBufferAllocs(GUI& gui, DescriptorSet& descriptorSet, AssetManager& assetManager, const std::vector<DrawIndexedIndirectCommand>& indirectDraws) {

    if (!gui.beginWindow("Buffers", nullptr, glm::vec2(430, 400), glm::vec2(830, 320))) return;

    vk::DeviceSize bufferUsed = 0;      // bytes actually occupied by live allocations
    vk::DeviceSize bufferCommitted = 0; // bytes reserved on the GPU (full buffer capacity)

    if (gui.collapsingHeader("Variable Buffers", true)) {
        int i = 0;
        for (auto& buffer : descriptorSet.getVariableBuffers()) {
            vk::DeviceSize usedBytes = 0;
            for (const auto& [offset, alloc] : buffer->allocations)
                usedBytes += alloc.capacity;
            const char* label = buffer->name.empty() ? nullptr : buffer->name.c_str();
            if (label)
                gui.textf("%s [%d] : %d allocs", label, i, (int)buffer->allocations.size());
            else
                gui.textf("Buffer %d : %d allocs", i, (int)buffer->allocations.size());

            float fraction = buffer->bufferSize > 0 ? (float)((double)usedBytes / (double)buffer->bufferSize) : 0.0f;
            std::string overlay = formatBytes(usedBytes) + " / " + formatBytes(buffer->bufferSize);
            gui.progressBar(fraction, glm::vec2(-1, 0), overlay);

            bufferUsed += usedBytes;
            bufferCommitted += buffer->bufferSize;
            i++;
        }
    }

    if (gui.collapsingHeader("Fixed Buffers", true)) {
        int i = 0;
        for (auto& buffer : descriptorSet.getFixedBuffers()) {
            // Count slots tracked via allocateFixedBuffer + slots streamed via direct memcpy.
            // Streamed sites bump liveCount through writeFixedBuffer.
            uint32_t slotInUse = 0;
            for (const auto& alloc : buffer->allocations)
                if (alloc.inUse)
                    slotInUse++;
            uint32_t inUse = std::max(slotInUse, buffer->liveCount);
            bool isStreamed = (buffer->liveCount > 0 && slotInUse == 0);

            // For perFrame buffers, allocate-tracked and streamed counts are both per-frame logical
            // counts, but maxSize/bufferSize span all frame slices. Normalize for display so the
            // ratio reflects "this frame's usage vs this frame's budget".
            uint32_t displaySlots = buffer->perFrame ? (buffer->maxSize / MAX_FRAMES_IN_FLIGHT) : buffer->maxSize;
            vk::DeviceSize displayBufferSize = buffer->perFrame ? (buffer->bufferSize / MAX_FRAMES_IN_FLIGHT) : buffer->bufferSize;
            vk::DeviceSize usedBytes = (vk::DeviceSize)inUse * buffer->elementSize;

            const char* label = buffer->name.empty() ? nullptr : buffer->name.c_str();
            const char* tag = isStreamed ? (buffer->perFrame ? "  (streamed, per-frame)" : "  (streamed)") : buffer->perFrame ? "  (per-frame)" : "";
            if (label)
                gui.textf("%s [%d] : %u / %u slots (%d free)%s", label, i, inUse, displaySlots, (int)buffer->freeSlots.size(), tag);
            else
                gui.textf("Buffer %d : %u / %u slots (%d free)%s", i, inUse, displaySlots, (int)buffer->freeSlots.size(), tag);

            float fraction = displaySlots > 0 ? (float)inUse / (float)displaySlots : 0.0f;
            std::string overlay = formatBytes(usedBytes) + " / " + formatBytes(displayBufferSize);
            gui.progressBar(fraction, glm::vec2(-1, 0), overlay);

            // Grand totals use real GPU footprint, not the per-frame display values.
            bufferUsed += usedBytes;
            bufferCommitted += buffer->bufferSize;
            i++;
        }
    }

    // Estimated texture footprint. Assumes RGBA8 (4 bpp) — values for HDR / depth / BC will be off.
    vk::DeviceSize totalTexBytes = 0;
    int texCount = 0;
    for (const auto& tex : descriptorSet.getTextureResources()) {
        if (tex.isEmpty()) continue;
        vk::DeviceSize baseSize = (vk::DeviceSize)tex.width * tex.height * 4;
        totalTexBytes += (vk::DeviceSize)((double)baseSize * mipChainFactor(tex.mipLevels));
        texCount++;
    }

    if (gui.collapsingHeader("Textures & Samplers", true)) {
        gui.textf("Textures: %d | ~%s (RGBA8 estimate, includes mips)", texCount, formatBytes(totalTexBytes).c_str());
        gui.textf("Samplers: %d", (int)descriptorSet.getSamplerResources().size());
    }

    // Mesh stats are derived from variable-buffer allocs and overlap with the Variable Buffers totals
    // above — informational only, not added to grand totals to avoid double-counting.
    vk::DeviceSize totalVertexBytes = 0;
    vk::DeviceSize totalIndexBytes  = 0;
    int meshCount = 0;
    for (const auto& mesh : assetManager.meshes) {
        if (mesh.freed) continue;
        meshCount++;
        totalVertexBytes += (vk::DeviceSize)mesh.vertexCount * mesh.vertexStride;
        totalIndexBytes  += (vk::DeviceSize)mesh.indexCount  * sizeof(uint32_t);
    }

    if (gui.collapsingHeader("Meshes", true)) {
        gui.textf("%d meshes | Verts: %s | Idx: %s | Total: %s",
            meshCount,
            formatBytes(totalVertexBytes).c_str(),
            formatBytes(totalIndexBytes).c_str(),
            formatBytes(totalVertexBytes + totalIndexBytes).c_str());
    }

    uint64_t totalTris = 0;
    for (const auto& cmd : indirectDraws)
        totalTris += cmd.indexCount / 3;
    gui.separator();
    gui.textf("Triangles: %llu (%u draws)", (unsigned long long)totalTris, (uint32_t)indirectDraws.size());

    gui.separator();
    vk::DeviceSize liveUsed  = bufferUsed + totalTexBytes;
    vk::DeviceSize committed = bufferCommitted + totalTexBytes;
    gui.textf("GPU Memory  Used: ~%s | Committed: ~%s",
        formatBytes(liveUsed).c_str(),
        formatBytes(committed).c_str());

    gui.endWindow();
}

void showMeshList(GUI& gui, Scene& scene, BindlessSystem& bindless, uint32_t whiteTextureIndex) {
    AssetManager& assetManager = scene.assetManager;

    // No descriptor cache: a thumbnail is drawn straight from its bindless index, so a re-rendered
    // thumbnail needs no re-registration and there is nothing to key a cache on.
    const std::vector<TextureResource>& textures = bindless.descriptorSet->getTextureResources();

    // Drag-to-place state: pressing and holding a mesh thumbnail spawns a node carrying that
    // mesh, which is tracked under the cursor until the mouse is released (handled below).
    static uint32_t draggedNodeIdx = 0;
    static uint32_t draggedMeshIdx = 0;

    if (gui.beginWindow("Meshes", nullptr, glm::vec2(240, 400), glm::vec2(620, 20))) {
        for (uint32_t meshIdx = 0; meshIdx < assetManager.meshes.size(); meshIdx++) {
            const Mesh& mesh = assetManager.meshes[meshIdx];
            if (mesh.freed) continue;

            // Use the rendered thumbnail once available, otherwise the white texture as a placeholder.
            uint32_t texIndex = (mesh.thumbnailTextureIndex != 0xFFFFFFFF) ? mesh.thumbnailTextureIndex : whiteTextureIndex;
            if (texIndex < textures.size() && !textures[texIndex].isEmpty()) {
                gui.imageButton(mesh.name, texIndex, glm::vec2(64.0f, 64.0f));
                // On press: spawn a node with this mesh and start tracking it. The press-capture
                // keeps the button active while the mouse stays down, even once the cursor leaves
                // it for the viewport — which is the whole flow.
                if (draggedNodeIdx == 0 && gui.isItemActivated()) {
                    uint32_t idx = scene.sceneGraph.addNode(false, SceneGraph::ROOT_INDEX);
                    Node& node = scene.sceneGraph.getNode(idx);
                    node.name = scene.sceneGraph.makeUniqueNodeName(mesh.name);
                    NodeOps::assignMesh(node, meshIdx, scene);
                    NodeOps::assignMaterial(node, scene.getFallBackMaterial(), scene);
                    draggedNodeIdx = idx;
                    draggedMeshIdx = meshIdx;
                }
            }
            gui.textf("%s", mesh.name.c_str());
        }
        gui.endWindow();
    }

    // Track the dragged node. Deliberately outside the window scope: it keeps updating while the
    // cursor is over the viewport, and now also survives the window being collapsed or scrolled
    // away mid-drag, which would otherwise strand the node.
    if (draggedNodeIdx != 0) {
        if (gui.isMouseDown() && scene.sceneGraph.isNodeValid(draggedNodeIdx)) {
            // Same GLFW-derived NDC as the selection raycast, so the placement can't drift from
            // the cursor.
            glm::vec2 ndc = InputManager::getCurrentState().ndcMousePos;
            glm::vec3 origin, direction;
            scene.activeCamera.rayFromScreenCoords(ndc.x, ndc.y, origin, direction);

            Node& node = scene.sceneGraph.getNode(draggedNodeIdx);
            const Mesh& mesh = assetManager.meshes[draggedMeshIdx];

            // Skip the dragged node itself so the ray reaches the surface beneath it.
            Raycast::MeshHit hit = Raycast::castMeshes(origin, direction, scene.sceneGraph.getNodes(),
                                                       scene.sceneGraph.getLastNode(), assetManager.meshes, draggedNodeIdx);

            if (hit.nodeIndex != 0) {
                // Orient the node's up axis to the surface normal and rest the base of its
                // bounding box on the hit point. Geometry is centered at the origin on import, so
                // the bbox-base anchor comes from the (centered) bounding box itself — NOT mesh.center,
                // which stores the mesh's pre-centering authoring offset and would shove the node away.
                glm::quat rot = glm::rotation(glm::vec3(0.0f, 1.0f, 0.0f), hit.hitNormal);
                glm::vec3 bbCenter = (mesh.boundingBoxMin + mesh.boundingBoxMax) * 0.5f;
                glm::vec3 anchor(bbCenter.x, mesh.boundingBoxMin.y, bbCenter.z);
                node.relativePosition = hit.hitPoint - rot * anchor;
                node.relativeRotation = rot;
                node.relativeRotationEuler = glm::eulerAngles(rot);
            } else {
                // No surface under the cursor — float the node along the ray.
                node.relativePosition = origin + direction * 10.0f;
                node.relativeRotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
                node.relativeRotationEuler = glm::vec3(0.0f);
            }
            node.transformDirty = true;
        } else {
            // Mouse released (or the node was removed) — stop tracking.
            draggedNodeIdx = 0;
            draggedMeshIdx = 0;
        }
    }
}

void showTraces(GUI& gui) {
    if (!gui.beginWindow("CPU Timing", nullptr, glm::vec2(360, 20), glm::vec2(0, 0), GUIAnchor::Top | GUIAnchor::Left, GUIWindowNoMove | GUIWindowNoSavedSettings)) return;
    auto& traces = tracing::getTraces();
    for(auto& name : tracing::getOrder()){
        auto& trace = traces[name];
    // color each scope with a distinct hue via the golden-ratio sequence
        float hue = fmodf(static_cast<float>(trace.scope) * 0.61803398875f, 1.0f);
        glm::vec3 rgb = GUI::hsvToRgb(hue, 0.6f, 0.95f);
    // indent 2 spaces per scope level, clamped: a drifted scope counter must cost a wrong indent,
    // never a multi-gigabyte string allocation.
        std::string indent(std::min(trace.scope, 16u) * 2, ' ');
        gui.textColored(indent + name + " : " + std::to_string(trace.averageDuration.count() * 1000.0) + " ms", glm::vec4(rgb, 1.0f));
    }
    gui.endWindow();
};
