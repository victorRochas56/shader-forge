#include "gui.hpp"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"
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

static void check_vk_result(VkResult err) {
    if (err == 0)
        return;
    fprintf(stderr, "[vulkan] Error: VkResult = %d\n", err);
    if (err < 0)
        abort();
}

void initIMGUI(Device& device, vk::Instance instance, uint32_t graphicsQueueFamily,
               Swapchain& swapchain, GLFWwindow* window) {
    VkDescriptorPoolSize pool_sizes[] = {{VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
                                         {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
                                         {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
                                         {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
                                         {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
                                         {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
                                         {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
                                         {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
                                         {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
                                         {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
                                         {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000}};

    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 1000;
    pool_info.poolSizeCount = std::size(pool_sizes);
    pool_info.pPoolSizes = pool_sizes;

    VkDescriptorPool imguiPool;
    VkResult result = vkCreateDescriptorPool(*device.getDevice(), &pool_info, nullptr, &imguiPool);
    check_vk_result(result);

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForVulkan(window, true);

    // Get swapchain details
    uint32_t swapchainImageCount = swapchain.getSwapChainImages().size();
    vk::Format colorFormat = swapchain.getSwapChainImageFormat();
    vk::Format depthFormat = findDepthFormat(device);

    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.ApiVersion = VK_API_VERSION_1_4;
    init_info.Instance = instance;
    init_info.PhysicalDevice = *device.getPhysicalDevice();
    init_info.Device = *device.getDevice();
    init_info.QueueFamily = graphicsQueueFamily;
    init_info.Queue = *device.getGraphicsQueue();
    init_info.DescriptorPool = imguiPool;
    init_info.Subpass = 0;
    init_info.MinImageCount = swapchainImageCount;
    init_info.ImageCount = swapchainImageCount;
    init_info.MSAASamples = vk::SampleCountFlagBits::e1;
    init_info.Allocator = nullptr;
    init_info.CheckVkResultFn = check_vk_result;
    init_info.UseDynamicRendering = true;

    // Setup pipeline rendering create info for dynamic rendering
    vk::Format colorAttachmentFormat = colorFormat;
    VkPipelineRenderingCreateInfo pipelineRenderingInfo = {};
    pipelineRenderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    pipelineRenderingInfo.colorAttachmentCount = 1;
    pipelineRenderingInfo.pColorAttachmentFormats = reinterpret_cast<VkFormat*>(&colorAttachmentFormat);
    pipelineRenderingInfo.depthAttachmentFormat = VK_FORMAT_UNDEFINED;
    pipelineRenderingInfo.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

    init_info.PipelineRenderingCreateInfo = pipelineRenderingInfo;

    // Initialize Vulkan backend
    bool initResult = ImGui_ImplVulkan_Init(&init_info);
    if (!initResult) {
        throw std::runtime_error("Failed to initialize ImGui Vulkan backend");
    }
}

void traverseNodeTree(Node& node, uint32_t level, uint32_t selectedNode, SceneGraph& sceneGraph) {
    std::string displayText = "";
    ImGuiTreeNodeFlags flag = ImGuiTreeNodeFlags_DefaultOpen;
    if(!node.internal) {
        if (node.getIndex() == selectedNode) {
            flag |= ImGuiTreeNodeFlags_Selected;
        }
        // Append "##<index>" so the ImGui ID stays unique even if two nodes share a display name
        // (the text after ## is not drawn). Node names are uniquified on spawn, but this guards
        // legacy/default-named nodes too.
        displayText += node.name;
        displayText += "##" + std::to_string(node.getIndex());
        if(node.firstChild == 0) flag |= ImGuiTreeNodeFlags_Leaf;
        if(ImGui::TreeNodeEx(displayText.c_str(),flag)){
            if(ImGui::IsItemClicked())
                sceneGraph.selectNode(node.getIndex());
                
            auto& nodes = sceneGraph.getNodes();
            uint32_t child = node.firstChild;
            while (child != 0) {
                traverseNodeTree(nodes[child], level + 1, selectedNode, sceneGraph);
                child = nodes[child].nextSibling;
            }
            ImGui::TreePop();
        }
    }
}

void showMaterialEditor(MaterialEditorState& state, Scene& scene, BindlessSystem& bindless) {
    auto& materials = scene.getMaterials();

    ImGui::Begin("Materials");

    // Cache the ImGui descriptor per thumbnail image view (see showMeshList for the rationale).
    static std::unordered_map<VkImageView, ImTextureID> matThumbCache;
    const std::vector<TextureResource>& textures = bindless.descriptorSet->getTextureResources();
    const std::vector<SamplerResource>& samplers = bindless.descriptorSet->getSamplerResources();

    // material list
    for (int i = 0; i < static_cast<int>(materials.size()); i++) {
        uint32_t ti = materials[i].thumbnailTextureIndex;
        if (ti != 0xFFFFFFFF && ti < textures.size() && !textures[ti].isEmpty() && !samplers.empty()) {
            VkImageView view = static_cast<VkImageView>(**textures[ti].imageView);
            auto it = matThumbCache.find(view);
            ImTextureID texId;
            if (it != matThumbCache.end()) {
                texId = it->second;
            } else {
                texId = (ImTextureID)ImGui_ImplVulkan_AddTexture(static_cast<VkSampler>(**samplers[0].sampler), view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                matThumbCache.emplace(view, texId);
            }
            ImGui::Image(texId, ImVec2(128.0f, 128.0f));
        }

        std::string label = materials[i].name.empty() ? ("Material " + std::to_string(i)) : materials[i].name;
        if (i == 0) label += " (default)";

        bool isSelected = (state.showEditor && state.selectedIndex == i);
        if (ImGui::Selectable(label.c_str(), isSelected)) {
            state.showEditor = true;
            state.loadFromMaterial(i, materials[i], scene);
            InputManager::getInstance().canMove = false;
        }
    }

    ImGui::Separator();
    if (ImGui::Button("Create New Material")) {
        state.showEditor = true;
        state.resetToDefaults();
        InputManager::getInstance().canMove = false;
    }
    if (ImGui::Button("Pick from Mesh")) {
        InputManager::getInstance().materialPickMode = true;
        InputManager::getInstance().pickedMaterialIndex = -1;
    }
    if (InputManager::getInstance().materialPickMode) {
        ImGui::TextColored(ImVec4(1, 1, 0, 1), "Click a mesh to select its material...");
    }

    // handle pick result
    auto& input = InputManager::getInstance();
    if (input.pickedMaterialIndex >= 0 && input.pickedMaterialIndex < static_cast<int>(materials.size())) {
        state.showEditor = true;
        state.loadFromMaterial(input.pickedMaterialIndex, materials[input.pickedMaterialIndex], scene);
        input.canMove = false;
        input.pickedMaterialIndex = -1;
    }

    ImGui::End();

    if (!state.showEditor) return;

    bool open = true;
    std::string title = state.selectedIndex >= 0 ? "Edit Material" : "New Material";
    ImGui::Begin(title.c_str(), &open);
    if (!open) {
        state.showEditor = false;
        InputManager::getInstance().canMove = true;
        ImGui::End();
        return;
    }

    InputManager::getInstance().canMove = false;

    ImGui::SetNextItemWidth(200);
    ImGui::InputText("Name", state.nameBuffer, sizeof(state.nameBuffer));

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
        ImGui::SetNextItemWidth(200);
        if (ImGui::BeginCombo("Shader##mat", preview.c_str())) {
            for (int i = 0; i < static_cast<int>(litShaders.size()); i++) {
                bool sel = (state.selectedShaderIndex == i);
                if (ImGui::Selectable(shaderLabel(litShaders[i].sourceFile).c_str(), sel))
                    state.selectedShaderIndex = i;
                if (sel) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
    }

    ImGui::Separator();
    ImGui::Text("Properties");
    ImGui::ColorEdit4("Base Color", state.color);
    ImGui::SliderFloat("Metallic##mat", &state.metallic, 0.0f, 1.0f);
    ImGui::SliderFloat("Roughness##mat", &state.roughness, 0.0f, 1.0f);

    ImGui::Separator();
    ImGui::Text("Textures (leave empty for defaults)");
    ImGui::SetNextItemWidth(300);
    ImGui::InputText("Albedo", state.albedoPath, sizeof(state.albedoPath));
    browseButton("albedo", state.albedoPath, sizeof(state.albedoPath));
    ImGui::SetNextItemWidth(300);
    ImGui::InputText("Metallic Tex", state.metallicPath, sizeof(state.metallicPath));
    browseButton("metallic", state.metallicPath, sizeof(state.metallicPath));
    ImGui::SetNextItemWidth(300);
    ImGui::InputText("Roughness Tex", state.roughnessPath, sizeof(state.roughnessPath));
    browseButton("roughness", state.roughnessPath, sizeof(state.roughnessPath));
    ImGui::SetNextItemWidth(300);
    ImGui::InputText("Normal", state.normalPath, sizeof(state.normalPath));
    browseButton("normal", state.normalPath, sizeof(state.normalPath));
    ImGui::Checkbox("Flip Normal", &state.flipNormal);
    ImGui::Checkbox("Alpha Clip", &state.alphaClip);
    if (state.alphaClip) {
        ImGui::SliderFloat("Alpha Cutoff", &state.alphaCutoff, 0.0f, 1.0f);
    }

    ImGui::Separator();
    ImGui::Text("Environment Map (6 faces)");
    const char* faceLabels[] = {"Pos X", "Pos Y", "Pos Z", "Neg X", "Neg Y", "Neg Z"};
    for (int i = 0; i < 6; i++) {
        ImGui::SetNextItemWidth(300);
        ImGui::InputText(faceLabels[i], state.envMapPaths[i], sizeof(state.envMapPaths[i]));
        browseButton(faceLabels[i], state.envMapPaths[i], sizeof(state.envMapPaths[i]));
    }

    ImGui::Separator();

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
                mat.normalTextureIndex = scene.assetManager.loadTextureFromFile(state.normalPath, vk::Format::eR8G8B8A8Unorm);
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
        if (ImGui::Button("Apply Changes")) {
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
        if (ImGui::Button("Create")) {
            Material newMat;
            buildMaterial(newMat);
            scene.addMaterial(newMat);

            state.showEditor = false;
            InputManager::getInstance().canMove = true;
        }
    }

    ImGui::SameLine();
    if (ImGui::Button("Cancel")) {
        state.showEditor = false;
        InputManager::getInstance().canMove = true;
    }

    ImGui::End();
}

void showImageViewList(BindlessSystem& bindless, RenderFeatures& features) {
    ImGui::Begin("Image Views");

    ImageVisSettings& vis = features.imageVis;
    if (ImGui::Button("Clear Selection")) {
        vis.imageIndex = 0xFFFFFFFF;
    }

    int i = 0;
    for (const TextureResource& img : (*bindless.descriptorSet).getTextureResources()) {
        if(img.source.empty()){
            i++;
            continue;
        }
        bool selected = (vis.imageIndex == static_cast<uint32_t>(i));
        if (selected) ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
        if (ImGui::Button(img.source.c_str())) {
            vis.imageIndex = (selected) ? 0xFFFFFFFF : i;
        }
        if (selected) {
            ImGui::PopStyleColor();
            ImGui::SameLine();
            if (ImGui::Button("B&W")){
                vis.flags ^= ImageVisFlags::B_W_IMAGE;
            }
            ImGui::SameLine();
            if (ImGui::Button("Flip Y")){
                vis.flags ^= ImageVisFlags::FLIP_VERTICAL;
            }
        }
        i++;
    }

    ImGui::End();
}

void showActionMenu(uint32_t context, GLFWwindow* window, Camera& camera,
                    SceneGraph& sceneGraph, float posX, float posY) {
    ImGui::SetNextWindowPos({posX, posY});
    ImGui::Begin("Action");
    if (ImGui::Button("Add Node")) {
        glm::vec3 origin, direction;
        int w = 0, h = 0;
        glfwGetWindowSize(window, &w, &h);
        camera.rayFromScreenCoords((posX / w) * 2.0 - 1.0, (posY / h) * 2.0 - 1.0, origin, direction);
        sceneGraph.addNode(false, SceneGraph::ROOT_INDEX, origin + direction);
    }
    ImGui::End();
}

void showToggles(RenderFeatures& f){
    ImGui::Begin("Toggles");
    if(ImGui::Button("Gizmos")){
        f.showGizmos = !f.showGizmos;
    }
    ImGui::SliderInt("ImageMip", &f.imageVis.mipLevel, 0, 6);
    if(ImGui::Button("Depth Buffer")){
        f.imageVis.flags ^= ImageVisFlags::LINEARIZE;
    }
    if(ImGui::Button("SSAO")){
        f.ssao.enabled = !f.ssao.enabled;
    }
    if(f.ssao.enabled){
        ImGui::SliderFloat("AO Radius", &f.ssao.radius, 0.01f, 10.0f);
        ImGui::SliderFloat("AO Bias", &f.ssao.bias, 0.01f, 0.1f);
        ImGui::SliderFloat("AO Power", &f.ssao.power, 0.01f, 5.0f);
    }
    if(ImGui::Button("SSR")){
        f.ssr.enabled = !f.ssr.enabled;
    }
    if(f.ssr.enabled){
        ImGui::SliderInt("Max Steps", &f.ssr.maxSteps, 16, 128);
        ImGui::SliderFloat("Thickness", &f.ssr.thickness, 0.01f, 5.0f);
        ImGui::SliderFloat("Roughness Threshold", &f.ssr.roughnessThreshold, 0.0f, 1.0f);
        ImGui::SliderFloat("Temporal Blend", &f.ssr.temporalBlend, 0.01f, 1.0f);
        if(ImGui::SliderFloat("Resolution Scale", &f.ssr.resolutionScale, 0.25f, 1.0f)){
            f.ssr.resolutionDirty = true;
        }
    }
    if(ImGui::Button("Volumetrics")){
        f.volumetrics.enabled = !f.volumetrics.enabled;
    }
    if(f.volumetrics.enabled){
        ImGui::SliderInt("Steps", &f.volumetrics.numSteps,4,32);
        ImGui::SliderFloat("Max Distance", &f.volumetrics.maxDist, 5.0f, 50.0f);
        ImGui::SliderFloat("Blur Radius", &f.volumetrics.blurRadius, 0.0f, 4.0f);
    }
    if(ImGui::Button("Show BBOXes")){
        f.showBBoxes = !f.showBBoxes;
    }

    ImGui::SeparatorText("Tonemap / Exposure");
    const char* tonemapOps[] = {"Reinhard", "ACES", "None"};
    int op = static_cast<int>(f.tonemap.op);
    if(ImGui::Combo("Operator", &op, tonemapOps, IM_ARRAYSIZE(tonemapOps))){
        f.tonemap.op = static_cast<uint32_t>(op);
    }
    ImGui::Checkbox("Auto Exposure", &f.tonemap.autoExposure);
    if(f.tonemap.autoExposure){
        ImGui::SliderFloat("Exposure Comp (EV)", &f.tonemap.exposureComp, -8.0f, 8.0f, "%.2f");
        ImGui::SetItemTooltip("Bias on the metered exposure; + = brighter.");
        ImGui::DragFloatRange2("EV clamp", &f.tonemap.minEV, &f.tonemap.maxEV, 0.1f, -10.0f, 20.0f, "min %.1f", "max %.1f");
        ImGui::SliderFloat("Adaptation Speed", &f.tonemap.adaptationSpeed, 0.1f, 10.0f, "%.2f /s");
        ImGui::SetItemTooltip("Eye-adaptation rate; higher = snappier, lower = slower fade.");
    } else {
        ImGui::SliderFloat("Exposure (EV100)", &f.tonemap.ev100, -2.0f, 20.0f, "%.2f");
        ImGui::SetItemTooltip("Higher = darker. ~log2(directionalIntensity) - 3.6 keys whites near clipping.");
        ImGui::Text("Exposure factor: %.3e", computeExposure(f.tonemap));
    }

    ImGui::End();
}

void showScenesMenu(Scene& scene, BindlessSystem& bindless, RenderBuffers& buffers,SceneLoader& sceneLoader){
    ImGui::Begin("Scene Manager");
    if(ImGui::Button("Save Scene")){
        sceneLoader.saveScene("scene.scn", scene);
    }
    ImGui::SameLine();
    if(ImGui::Button("Load Scene")){
        sceneLoader.loadScene("scene.scn", scene, bindless,
                               buffers.modelMatrixBufferIndex,
                               buffers.lightBufferIndex,
                               buffers.volumeBufferIndex);
    }
    ImGui::SameLine();
    if(ImGui::Button("Clear Scene")){
        sceneLoader.clearScene(scene, bindless,
                                buffers.modelMatrixBufferIndex,
                                buffers.lightBufferIndex,
                                buffers.volumeBufferIndex);
    }
    ImGui::End();
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

void showBufferAllocs(DescriptorSet& descriptorSet, AssetManager& assetManager, const std::vector<DrawIndexedIndirectCommand>& indirectDraws) {

    ImGui::Begin("Buffers");

    vk::DeviceSize bufferUsed = 0;      // bytes actually occupied by live allocations
    vk::DeviceSize bufferCommitted = 0; // bytes reserved on the GPU (full buffer capacity)

    if (ImGui::CollapsingHeader("Variable Buffers", ImGuiTreeNodeFlags_DefaultOpen)) {
        int i = 0;
        for (auto& buffer : descriptorSet.getVariableBuffers()) {
            vk::DeviceSize usedBytes = 0;
            for (const auto& [offset, alloc] : buffer->allocations)
                usedBytes += alloc.capacity;
            const char* label = buffer->name.empty() ? nullptr : buffer->name.c_str();
            if (label)
                ImGui::Text("%s [%d] : %d allocs", label, i, (int)buffer->allocations.size());
            else
                ImGui::Text("Buffer %d : %d allocs", i, (int)buffer->allocations.size());

            float fraction = buffer->bufferSize > 0 ? (float)((double)usedBytes / (double)buffer->bufferSize) : 0.0f;
            std::string overlay = formatBytes(usedBytes) + " / " + formatBytes(buffer->bufferSize);
            ImGui::ProgressBar(fraction, ImVec2(-1, 0), overlay.c_str());

            bufferUsed += usedBytes;
            bufferCommitted += buffer->bufferSize;
            i++;
        }
    }

    if (ImGui::CollapsingHeader("Fixed Buffers", ImGuiTreeNodeFlags_DefaultOpen)) {
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
                ImGui::Text("%s [%d] : %u / %u slots (%d free)%s", label, i, inUse, displaySlots, (int)buffer->freeSlots.size(), tag);
            else
                ImGui::Text("Buffer %d : %u / %u slots (%d free)%s", i, inUse, displaySlots, (int)buffer->freeSlots.size(), tag);

            float fraction = displaySlots > 0 ? (float)inUse / (float)displaySlots : 0.0f;
            std::string overlay = formatBytes(usedBytes) + " / " + formatBytes(displayBufferSize);
            ImGui::ProgressBar(fraction, ImVec2(-1, 0), overlay.c_str());

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

    if (ImGui::CollapsingHeader("Textures & Samplers", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Textures: %d | ~%s (RGBA8 estimate, includes mips)", texCount, formatBytes(totalTexBytes).c_str());
        ImGui::Text("Samplers: %d", (int)descriptorSet.getSamplerResources().size());
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

    if (ImGui::CollapsingHeader("Meshes", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("%d meshes | Verts: %s | Idx: %s | Total: %s",
            meshCount,
            formatBytes(totalVertexBytes).c_str(),
            formatBytes(totalIndexBytes).c_str(),
            formatBytes(totalVertexBytes + totalIndexBytes).c_str());
    }

    uint64_t totalTris = 0;
    for (const auto& cmd : indirectDraws)
        totalTris += cmd.indexCount / 3;
    ImGui::Separator();
    ImGui::Text("Triangles: %llu (%u draws)", (unsigned long long)totalTris, (uint32_t)indirectDraws.size());

    ImGui::Separator();
    vk::DeviceSize liveUsed  = bufferUsed + totalTexBytes;
    vk::DeviceSize committed = bufferCommitted + totalTexBytes;
    ImGui::Text("GPU Memory  Used: ~%s | Committed: ~%s",
        formatBytes(liveUsed).c_str(),
        formatBytes(committed).c_str());

    ImGui::End();
}

void showMeshList(Scene& scene, BindlessSystem& bindless, uint32_t whiteTextureIndex) {
    ImGui::Begin("Meshes");

    AssetManager& assetManager = scene.assetManager;

    // ImGui_ImplVulkan_AddTexture allocates a descriptor per call, so cache it. Keyed by image
    // view so that a re-rendered thumbnail (new view) gets a fresh registration automatically.
    static std::unordered_map<VkImageView, ImTextureID> imguiTexCache;
    const std::vector<TextureResource>& textures = bindless.descriptorSet->getTextureResources();
    const std::vector<SamplerResource>& samplers = bindless.descriptorSet->getSamplerResources();

    // Drag-to-place state: pressing and holding a mesh thumbnail spawns a node carrying that
    // mesh, which is tracked under the cursor until the mouse is released (handled below).
    static uint32_t draggedNodeIdx = 0;
    static uint32_t draggedMeshIdx = 0;

    for (uint32_t meshIdx = 0; meshIdx < assetManager.meshes.size(); meshIdx++) {
        const Mesh& mesh = assetManager.meshes[meshIdx];
        if (mesh.freed) continue;

        // Use the rendered thumbnail once available, otherwise the white texture as a placeholder.
        uint32_t texIndex = (mesh.thumbnailTextureIndex != 0xFFFFFFFF) ? mesh.thumbnailTextureIndex : whiteTextureIndex;
        ImTextureID texId = ImTextureID_Invalid;
        if (texIndex < textures.size() && !textures[texIndex].isEmpty() && !samplers.empty()) {
            VkImageView view = static_cast<VkImageView>(**textures[texIndex].imageView);
            auto it = imguiTexCache.find(view);
            if (it != imguiTexCache.end()) {
                texId = it->second;
            } else {
                VkDescriptorSet ds = ImGui_ImplVulkan_AddTexture(static_cast<VkSampler>(**samplers[0].sampler),
                                                                 view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                texId = (ImTextureID)ds;
                imguiTexCache.emplace(view, texId);
            }
        }

        if (texId != ImTextureID_Invalid) {
            ImGui::ImageButton(mesh.name.c_str(), texId, ImVec2(64.0f, 64.0f));
            // On click & hold: spawn a node with this mesh and start tracking it. The button
            // stays "active" while the mouse stays down, even once the cursor leaves it.
            if (draggedNodeIdx == 0 && ImGui::IsItemActivated()) {
                uint32_t idx = scene.sceneGraph.addNode(false, SceneGraph::ROOT_INDEX);
                Node& node = scene.sceneGraph.getNode(idx);
                node.name = scene.sceneGraph.makeUniqueNodeName(mesh.name);
                NodeOps::assignMesh(node, meshIdx, scene);
                NodeOps::assignMaterial(node, scene.getFallBackMaterial(), scene);
                draggedNodeIdx = idx;
                draggedMeshIdx = meshIdx;
            }
        }
        ImGui::Text(mesh.name.c_str());
    }

    ImGui::End();

    // Track the dragged node. Done outside the window scope so it keeps updating while the
    // cursor is over the viewport, and stops the moment the mouse is released.
    if (draggedNodeIdx != 0) {
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left) && scene.sceneGraph.isNodeValid(draggedNodeIdx)) {
            // Use the same GLFW-derived NDC as the selection raycast. ImGui::GetMousePos() would be
            // wrong under multi-viewports (it reports desktop-global coords), causing the placement
            // to drift from the cursor by the window's screen position.
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

void showDebugWindow(uint32_t culledCount, float& cullFovScale){
    ImGui::Begin("Debug");
    ImGui::Text(("Culled : " + std::to_string(culledCount)).c_str());
    ImGui::SliderFloat("Cull FOV Scale", &cullFovScale, 0.1f, 1.0f);
    ImGui::End();
}

void showTraces() {
    ImGui::Begin("CPU Timing");
    auto& traces = Tracer::getTraces();
    for(auto& trace : traces){
        ImGui::Text((trace.first+" : "+std::to_string(trace.second.averageDuration.count() * 1000.0)+" ms").c_str());
    }
    ImGui::End();
};
