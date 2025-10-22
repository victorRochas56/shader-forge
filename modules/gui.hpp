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
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

//helper functions for IMGUI

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
    init_info.MSAASamples = renderer->getMsaaSamples();
    init_info.Allocator = nullptr;
    init_info.CheckVkResultFn = check_vk_result;
    init_info.UseDynamicRendering = true;

    // Setup pipeline rendering create info for dynamic rendering
    vk::Format colorAttachmentFormat = colorFormat;
    VkPipelineRenderingCreateInfo pipelineRenderingInfo = {};
    pipelineRenderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    pipelineRenderingInfo.colorAttachmentCount = 1;
    pipelineRenderingInfo.pColorAttachmentFormats = reinterpret_cast<VkFormat*>(&colorAttachmentFormat);
    pipelineRenderingInfo.depthAttachmentFormat = static_cast<VkFormat>(depthFormat);
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

void showActionMenu(uint32_t context, Renderer* pRenderer, float posX, float posY) {
    ImGui::SetNextWindowPos(ImVec2{posX, posY});
    ImGui::Begin("Action");
    if (ImGui::Button("Add Node")) {
        glm::vec3 origin;
        glm::vec3 direction;
        int width = 0, height = 0;
        glfwGetWindowSize(pRenderer->getWindow(), &width, &height);
        pRenderer->activeCamera.rayFromScreenCoords((posX / width) * 2.0 - 1.0, (posY / height) * 2.0 - 1.0, &origin, &direction);
        pRenderer->addNode(0, origin + direction);
    }

    ImGui::End();
}