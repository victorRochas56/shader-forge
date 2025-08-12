#pragma once
#ifndef VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1 // for raii
#endif
#ifndef VULKAN_HPP_NO_CONSTRUCTORS  
#define VULKAN_HPP_NO_CONSTRUCTORS 1 // for structs constructors
#endif

#include <core.hpp>
#include "include/imgui.h"
#include "include/imgui_impl_glfw.h"
#include "include/imgui_impl_vulkan.h"

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

    	//1: create descriptor pool for IMGUI
	// the size of the pool is very oversize, but it's copied from imgui demo itself.
	VkDescriptorPoolSize pool_sizes[] =
	{
		{ VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
		{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
		{ VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
	};

	VkDescriptorPoolCreateInfo pool_info = {};
	pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
	pool_info.maxSets = 1000;
	pool_info.poolSizeCount = std::size(pool_sizes);
	pool_info.pPoolSizes = pool_sizes;

	VkDescriptorPool imguiPool;
    VkResult result;
	result = vkCreateDescriptorPool(*renderer->getDevices()->getLogicalDevice(), &pool_info, nullptr, &imguiPool);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForVulkan(renderer->getWindow(), true);
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.ApiVersion = VK_API_VERSION_1_4;
    init_info.Instance = renderer->getInstance();
    init_info.PhysicalDevice = *renderer->getDevices()->getPhysicalDevice();
    init_info.Device = *renderer->getDevices()->getLogicalDevice();
    init_info.QueueFamily = renderer->getGraphicsIndex();
    init_info.Queue = *(renderer->getDevices()->getGraphicsQueue());
    // init_info.PipelineCache = renderer->getResourceManager();
    init_info.DescriptorPool = imguiPool;
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
