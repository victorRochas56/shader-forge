#pragma once
#include "include/imgui.h"
#include "include/imgui_impl_glfw.h"
#include "include/imgui_impl_vulkan.h"
#include <core.hpp>

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

void initIMGUI(Renderer* renderer) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForVulkan(renderer->getWindow(), false);
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.ApiVersion = VK_API_VERSION_1_4;
    init_info.Instance = renderer->getInstance();
    init_info.PhysicalDevice = *renderer->getDevices()->getPhysicalDevice();
    init_info.Device = *renderer->getDevices()->getLogicalDevice();
    init_info.QueueFamily = renderer->getGraphicsIndex();
    init_info.Queue = *(renderer->getDevices()->getGraphicsQueue());
    // init_info.PipelineCache = renderer->getResourceManager();
    init_info.DescriptorPool = *renderer->getResourceManager()->getDescriptorPool();
    init_info.Subpass = 0;
    init_info.MinImageCount = 2;
    init_info.ImageCount = 2;
    init_info.MSAASamples = renderer->getMsaaSamples();
    init_info.Allocator = nullptr;
    init_info.CheckVkResultFn = check_vk_result;
    init_info.UseDynamicRendering = true;
    vk::PipelineRenderingCreateInfo createInfo = {
        .colorAttachmentCount = 1, .pColorAttachmentFormats = &(renderer->getSwapChain()->getSwapChainImageFormat()), .depthAttachmentFormat = findDepthFormat(renderer->getDevices())};
    init_info.PipelineRenderingCreateInfo = createInfo;
    ImGui_ImplVulkan_Init(&init_info);
}

