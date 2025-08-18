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
    // Create descriptor pool for IMGUI
    VkDescriptorPoolSize pool_sizes[] = {
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
    VkResult result = vkCreateDescriptorPool(*renderer->getDevices()->getLogicalDevice(), &pool_info, nullptr, &imguiPool);
    check_vk_result(result);
    
    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    
    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForVulkan(renderer->getWindow(), true);
    
    // Get swapchain details
    uint32_t swapchainImageCount = renderer->getSwapChain()->getSwapChainImages().size();
    vk::Format colorFormat = renderer->getSwapChain()->getSwapChainImageFormat();
    vk::Format depthFormat = findDepthFormat(renderer->getDevices());
    
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.ApiVersion = VK_API_VERSION_1_4;
    init_info.Instance = renderer->getInstance();
    init_info.PhysicalDevice = *renderer->getDevices()->getPhysicalDevice();
    init_info.Device = *renderer->getDevices()->getLogicalDevice();
    init_info.QueueFamily = renderer->getGraphicsIndex();
    init_info.Queue = *renderer->getDevices()->getGraphicsQueue();
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
    /*
    // Upload fonts to GPU 
    VkCommandBuffer command_buffer;
    VkCommandPool command_pool = renderer->getCommandPool(); 
    
    VkCommandBufferAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandPool = command_pool;
    alloc_info.commandBufferCount = 1;
    
    result = vkAllocateCommandBuffers(*renderer->getDevices()->getLogicalDevice(), &alloc_info, &command_buffer);
    check_vk_result(result);
    
    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    result = vkBeginCommandBuffer(command_buffer, &begin_info);
    check_vk_result(result);
    
    ImGui_ImplVulkan_CreateFontsTexture();
    
    
    result = vkEndCommandBuffer(command_buffer);
    check_vk_result(result);
    
    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;
    
    result = vkQueueSubmit(*renderer->getDevices()->getGraphicsQueue(), 1, &submit_info, VK_NULL_HANDLE);
    check_vk_result(result);
    
    result = vkQueueWaitIdle(*renderer->getDevices()->getGraphicsQueue());
    check_vk_result(result);
    
    ImGui_ImplVulkan_DestroyFontsTexture();
    
    vkFreeCommandBuffers(*renderer->getDevices()->getLogicalDevice(), command_pool, 1, &command_buffer);*/
}


void traverseNodeTree(Node& node, uint32_t level ,Renderer* renderer){
    std::string displayText = " ";
    for(int i=0;i<level;i++){
        displayText+= " ";
    }
    displayText += "|_ " + node.name;
    ImGui::Text(displayText.c_str());
    if(!node.children.empty()){
        for( auto childNode : node.children){
            traverseNodeTree(*childNode, level + 1, renderer);
        }
    }
}