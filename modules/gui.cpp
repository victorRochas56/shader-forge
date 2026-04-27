#include "gui.hpp"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"
#include "input.hpp"
#include "profiling.hpp"
#include "material_editor_state.hpp"
#include "pipelines.hpp"
#include "swapchain.hpp"
#include "scene_loader.hpp"
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

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
        displayText += node.name;
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

void showMaterialEditor(MaterialEditorState& state, Scene& scene) {
    auto& materials = scene.getMaterials();

    ImGui::Begin("Materials");

    // material list
    for (int i = 0; i < static_cast<int>(materials.size()); i++) {
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
        mat.shaderSource = scene.getFallBackShader();
        mat.color = glm::vec4(state.color[0], state.color[1], state.color[2], state.color[3]);
        mat.metallic = state.metallic;
        mat.roughness = state.roughness;

        uint32_t matFlags;

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
                mat.environmentMapIndex = scene.assetManager.loadCubemapFromFile(
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
            auto& nodes = scene.sceneGraph.getNodes();
            for (uint32_t i = 1; i <= scene.sceneGraph.getLastNode(); i++) {
                if (!scene.sceneGraph.isNodeValid(i)) continue;
                auto meshIdx = nodes[i].getMeshIndex();
                if (meshIdx >= scene.assetManager.meshes.size()) continue;

                if (nodes[i].getMaterialIndex() == static_cast<uint32_t>(state.selectedIndex)) {
                    scene.removeMeshFromShader(i, oldMat.shaderSource, oldMat);
                    scene.addMeshToShader(i, existingMat.shaderSource, existingMat);
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

void showBufferAllocs(DescriptorSet& descriptorSet, AssetManager& assetManager, const std::vector<DrawIndexedIndirectCommand>& indirectDraws){

    ImGui::Begin("Buffers");

    vk::DeviceSize grandTotal = 0;

    ImGui::Text("Variable Buffers");
    int i =0;
    for(auto& buffer : descriptorSet.getVariableBuffers()){
        vk::DeviceSize usedBytes = 0;
        for (const auto& alloc : buffer->allocations)
            usedBytes += alloc.size;
        float usedMB = usedBytes / (1024.0f * 1024.0f);
        float totalMB = buffer->bufferSize / (1024.0f * 1024.0f);
        ImGui::Text("Buffer %d : %d allocs | %.2f / %.2f MB", i, (int)buffer->allocations.size(), usedMB, totalMB);
        grandTotal += buffer->bufferSize;
        i++;
    }

    ImGui::Text("Fixed Buffers");
    i =0;
    for(auto& buffer : descriptorSet.getFixedBuffers()){
        uint32_t inUse = 0;
        for (const auto& alloc : buffer->allocations)
            if (alloc.inUse) inUse++;
        vk::DeviceSize usedBytes = inUse * buffer->elementSize;
        float usedMB = usedBytes / (1024.0f * 1024.0f);
        float totalMB = buffer->bufferSize / (1024.0f * 1024.0f);
        ImGui::Text("Buffer %d : %u / %d slots | %.2f / %.2f MB", i, inUse, (int)buffer->allocations.size(), usedMB, totalMB);
        ImGui::Text("  Free Slots : %d", (int)buffer->freeSlots.size());
        grandTotal += buffer->bufferSize;
        i++;
    }

    ImGui::Separator();
    ImGui::Text("Textures");
    vk::DeviceSize totalTexBytes = 0;
    int texCount = 0;
    for (const auto& tex : descriptorSet.getTextureResources()) {
        if (tex.isEmpty()) continue;
        // base RGBA8 size * ~1.33 for mip chain
        vk::DeviceSize baseSize = (vk::DeviceSize)tex.width * tex.height * 4;
        vk::DeviceSize withMips = baseSize * 4 / 3;
        totalTexBytes += withMips;
        texCount++;
    }
    ImGui::Text("%d textures | ~%.2f MB (with mips)", texCount, totalTexBytes / (1024.0f * 1024.0f));
    grandTotal += totalTexBytes;

    ImGui::Text("Samplers: %d", (int)descriptorSet.getSamplerResources().size());

    ImGui::Separator();
    ImGui::Text("Meshes");
    vk::DeviceSize totalVertexBytes = 0;
    vk::DeviceSize totalIndexBytes = 0;
    int meshCount = 0;
    for (const auto& mesh : assetManager.meshes) {
        if (mesh.freed) continue;
        meshCount++;
        totalVertexBytes += (vk::DeviceSize)mesh.vertexCount * mesh.vertexStride;
        totalIndexBytes += (vk::DeviceSize)mesh.indexCount * sizeof(uint32_t);
    }
    float vertMB = totalVertexBytes / (1024.0f * 1024.0f);
    float idxMB = totalIndexBytes / (1024.0f * 1024.0f);
    ImGui::Text("%d meshes | Verts: %.2f MB | Idx: %.2f MB | Total: %.2f MB", meshCount, vertMB, idxMB, vertMB + idxMB);

    ImGui::Separator();
    uint32_t totalTris = 0;
    for (const auto& cmd : indirectDraws)
        totalTris += cmd.indexCount / 3;
    ImGui::Text("Triangles: %u (%u draws)", totalTris, (uint32_t)indirectDraws.size());

    ImGui::Separator();
    float grandTotalMB = grandTotal / (1024.0f * 1024.0f);
    grandTotal += totalVertexBytes + totalIndexBytes;
    ImGui::Text("Total GPU Memory: ~%.2f MB", grandTotalMB);

    ImGui::End();
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
