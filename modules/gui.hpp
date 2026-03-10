#pragma once
#ifndef VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1 // for raii
#endif
#ifndef VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_NO_CONSTRUCTORS 1 // for structs constructors
#endif
#include "include/imgui.h"
#include "include/imgui_impl_glfw.h"
#include "include/imgui_impl_vulkan.h"
#include "renderer.hpp"
#include "input.hpp"
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <filesystem>

#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>
#endif

//helper functions for IMGUI

#ifdef _WIN32
static std::string openFileDialog(const char* filter = "Image Files\0*.png;*.jpg;*.jpeg;*.bmp;*.tga;*.hdr\0All Files\0*.*\0") {
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

static void check_vk_result(VkResult err) {
    if (err == 0)
        return;
    fprintf(stderr, "[vulkan] Error: VkResult = %d\n", err);
    if (err < 0)
        abort();
}

void initIMGUI(Renderer* renderer) {
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
    VkResult result = vkCreateDescriptorPool(*renderer->getDevice().getDevice(), &pool_info, nullptr, &imguiPool);
    check_vk_result(result);

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForVulkan(renderer->getWindow(), true);

    // Get swapchain details
    uint32_t swapchainImageCount = renderer->getSwapchain().getSwapChainImages().size();
    vk::Format colorFormat = renderer->getSwapchain().getSwapChainImageFormat();
    vk::Format depthFormat = findDepthFormat(renderer->getDevice());

    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.ApiVersion = VK_API_VERSION_1_4;
    init_info.Instance = renderer->getInstance();
    init_info.PhysicalDevice = *renderer->getDevice().getPhysicalDevice();
    init_info.Device = *renderer->getDevice().getDevice();
    init_info.QueueFamily = renderer->getGraphicsIndex();
    init_info.Queue = *renderer->getDevice().getGraphicsQueue();
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

void traverseNodeTree(Node* node, uint32_t level, uint32_t selectedNode, Renderer* renderer) {
    std::string displayText = " ";
    for (int i = 0; i < level; i++) {
        displayText += " ";
    }
    if (node->getIndex() == selectedNode) {
        ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(200, 200, 0, 255));
        displayText += "|_[" + node->name + "]";
    } else {
        displayText += "|_ " + node->name;
    }
    ImGui::Text(displayText.c_str());
    if (node->getIndex() == selectedNode) {
        ImGui::PopStyleColor();
    }
    if (!node->getChildren().empty()) {
        for (auto* childNode : node->getChildren()) {
            traverseNodeTree(childNode, level + 1, selectedNode, renderer);
        }
    }
}

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

void showMaterialEditor(MaterialEditorState& state, Renderer* renderer) {
    auto& materials = renderer->getMaterials();

    ImGui::Begin("Materials");

    // material list
    for (int i = 0; i < static_cast<int>(materials.size()); i++) {
        std::string label = materials[i].name.empty() ? ("Material " + std::to_string(i)) : materials[i].name;
        if (i == 0) label += " (default)";

        bool isSelected = (state.showEditor && state.selectedIndex == i);
        if (ImGui::Selectable(label.c_str(), isSelected)) {
            state.showEditor = true;
            state.loadFromMaterial(i, materials[i], renderer);
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
        state.loadFromMaterial(input.pickedMaterialIndex, materials[input.pickedMaterialIndex], renderer);
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
        const Material& defaultMat = materials[renderer->getFallBackMaterial()];

        mat.name = state.nameBuffer;
        mat.shaderSource = renderer->getFallBackShader();
        mat.color = glm::vec4(state.color[0], state.color[1], state.color[2], state.color[3]);
        mat.metallic = state.metallic;
        mat.roughness = state.roughness;

        uint32_t matFlags;

        // load textures
        if (strlen(state.albedoPath) > 0) {
            try {
                mat.albedoTextureIndex = renderer->assetManager.loadTextureFromFile(state.albedoPath);
                matFlags |= MaterialFlags::HAS_ALBEDO;
            } catch (...) { mat.albedoTextureIndex = defaultMat.albedoTextureIndex; }
        } else {
            mat.albedoTextureIndex = defaultMat.albedoTextureIndex;
        }

        if (strlen(state.roughnessPath) > 0) {
            try {
                mat.roughnessTextureIndex = renderer->assetManager.loadTextureFromFile(state.roughnessPath, vk::Format::eR8G8B8A8Unorm);
                matFlags |= MaterialFlags::HAS_ROUGHNESS;
            } catch (...) { mat.roughnessTextureIndex = defaultMat.roughnessTextureIndex; }
        } else {
            mat.roughnessTextureIndex = defaultMat.roughnessTextureIndex;
        }

        if (strlen(state.metallicPath) > 0) {
            try {
                mat.metallicTextureIndex = renderer->assetManager.loadTextureFromFile(state.metallicPath, vk::Format::eR8G8B8A8Unorm);
                matFlags |= MaterialFlags::HAS_METALLIC;
            } catch (...) { mat.metallicTextureIndex = defaultMat.metallicTextureIndex; }
        } else {
            mat.metallicTextureIndex = defaultMat.metallicTextureIndex;
        }

        if (strlen(state.normalPath) > 0) {
            try {
                mat.normalTextureIndex = renderer->assetManager.loadTextureFromFile(state.normalPath, vk::Format::eR8G8B8A8Unorm);
                matFlags |= MaterialFlags::HAS_NORMAL;
            } catch (...) { mat.normalTextureIndex = defaultMat.normalTextureIndex; }
        } else {
            mat.normalTextureIndex = defaultMat.normalTextureIndex;
        }
        if(state.flipNormal) matFlags |= MaterialFlags::FLIP_NORMAL; 

        mat.flags = static_cast<MaterialFlags>(matFlags);

        // environment map
        bool hasEnvMap = false;
        for (int i = 0; i < 6; i++) {
            if (strlen(state.envMapPaths[i]) > 0) { hasEnvMap = true; break; }
        }
        if (hasEnvMap) {
            try {
                mat.environmentMapIndex = renderer->assetManager.loadCubemapFromFile(
                    state.envMapPaths[0], state.envMapPaths[1], state.envMapPaths[2],
                    state.envMapPaths[3], state.envMapPaths[4], state.envMapPaths[5]);
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

            // re-register all nodes that use this material with updated material data
            existingMat = newMat;
            existingMat.materialID = static_cast<uint32_t>(std::hash<Material>{}(newMat));

            // update shader mappings for all nodes using this material
            auto& nodes = renderer->sceneGraph.getNodes();
            for (uint32_t i = 1; i < renderer->sceneGraph.getNodeCount(); i++) {
                if (!nodes[i].has_value()) continue;
                auto& matIndices = nodes[i]->getMaterialIndices();
                auto meshIdx = nodes[i]->getMeshIndex();
                if (meshIdx >= renderer->assetManager.meshes.size()) continue;

                for (size_t j = 0; j < matIndices.size(); j++) {
                    if (matIndices[j] == static_cast<uint32_t>(state.selectedIndex)) {
                        uint32_t subMeshIndex = renderer->assetManager.meshes[meshIdx].subMeshes[j];
                        renderer->removeMeshFromShader(&*nodes[i], subMeshIndex, oldMat.shaderSource, oldMat);
                        renderer->addMeshToShader(&*nodes[i], subMeshIndex, existingMat.shaderSource, existingMat);
                    }
                }
            }

            state.showEditor = false;
            InputManager::getInstance().canMove = true;
        }
    } else {
        // creating new material
        if (ImGui::Button("Create")) {
            Material newMat;
            buildMaterial(newMat);
            renderer->addMaterial(newMat);

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

void showImageViewList(Renderer* renderer) {
    ImGui::Begin("Image Views");

    if (renderer->imageVisIndex != 0xFFFFFFFF) {
        if (ImGui::Button("Clear Selection")) {
            renderer->imageVisIndex = 0xFFFFFFFF;
        }
    }

    int i = 0;
    for (const TextureResource& img : renderer->getDescriptorSet().getTextureResources()) {
        if(img.source.empty()){
            i++;
            continue;
        }
        bool selected = (renderer->imageVisIndex == static_cast<uint32_t>(i));
        if (selected) ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
        if (ImGui::Button(img.source.c_str())) {
            renderer->imageVisIndex = (selected) ? 0xFFFFFFFF : i;
        }
        if (selected) ImGui::PopStyleColor();
        i++;
    }

    ImGui::End();
}

void showActionMenu(uint32_t context, Renderer* renderer, float posX, float posY) {
    ImGui::SetNextWindowPos(ImVec2{posX, posY});
    ImGui::Begin("Action");
    if (ImGui::Button("Add Node")) {
        glm::vec3 origin;
        glm::vec3 direction;
        int width = 0, height = 0;
        glfwGetWindowSize(renderer->getWindow(), &width, &height);
        renderer->activeCamera.rayFromScreenCoords((posX / width) * 2.0 - 1.0, (posY / height) * 2.0 - 1.0, &origin, &direction);
        renderer->sceneGraph.addNode(0, origin + direction);
    }

    ImGui::End();
}