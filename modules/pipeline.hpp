#pragma once
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1 // for raii
#define VULKAN_HPP_NO_CONSTRUCTORS 1         // for structs constructors
#include <algorithm>
#include <array>
#include <resource_manager.hpp>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <devices.hpp>
#include <fstream>
#include <iostream>
#include <limits>
#include <load_resources.hpp>
#include <memory>
#include <stdexcept>
#include <structs.hpp>
#include <swapchain.hpp>
#include <unordered_map>
#include <utils.hpp>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#define GLFW_INCLUDE_VULKAN // REQUIRED only for GLFW CreateWindowSurface.
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#endif

class PipelineBuilder {

  public:
    PipelineBuilder(const Devices* devices, const BindlessResourceManager* resources, SwapChain* swapChain) : devices(devices), resources(resources), swapChain(swapChain) {}

    const vk::raii::PipelineLayout& getPipelineLayout(int index) {
        if (index < pipelineLayouts.size()) {
            return pipelineLayouts[index];
        } else {
            throw std::runtime_error("trying to access non existent pipeline");
        }
    }

    vk::raii::Pipeline createGraphicsPipeline(vk::SampleCountFlagBits msaaSamples) {

        size_t index = pipelineLayouts.size();

        vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shaders/shader.spv"), &*devices);
        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{.stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "vertMain"};

        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{.stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain"};

        vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
            .vertexBindingDescriptionCount = 0, .pVertexBindingDescriptions = nullptr, .vertexAttributeDescriptionCount = 0, .pVertexAttributeDescriptions = nullptr};

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{.topology = vk::PrimitiveTopology::eTriangleList};
        vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1, .scissorCount = 1};

        vk::PipelineRasterizationStateCreateInfo rasterizer{.depthClampEnable = vk::False,
                                                            .rasterizerDiscardEnable = vk::False,
                                                            .polygonMode = vk::PolygonMode::eFill,
                                                            .cullMode = vk::CullModeFlagBits::eBack,
                                                            .frontFace = vk::FrontFace::eCounterClockwise,
                                                            .depthBiasEnable = vk::False,
                                                            .depthBiasSlopeFactor = 1.0f,
                                                            .lineWidth = 1.0f};

        vk::PipelineMultisampleStateCreateInfo multisampling{.rasterizationSamples = msaaSamples, .sampleShadingEnable = vk::True, .minSampleShading = 0.2f};

        vk::PipelineDepthStencilStateCreateInfo depthStencil{
            .depthTestEnable = vk::True, .depthWriteEnable = vk::True, .depthCompareOp = vk::CompareOp::eLess, .depthBoundsTestEnable = vk::False, .stencilTestEnable = vk::False};

        vk::PipelineColorBlendAttachmentState colorBlendAttachment{.blendEnable = vk::False,
                                                                   .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                                                                                     vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};

        vk::PipelineColorBlendStateCreateInfo colorBlending{.logicOpEnable = vk::False, .logicOp = vk::LogicOp::eCopy, .attachmentCount = 1, .pAttachments = &colorBlendAttachment};

        std::vector dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
        vk::PipelineDynamicStateCreateInfo dynamicState{.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data()};

        vk::PushConstantRange pushConstantRange{.stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, .offset = 0, .size = sizeof(PushConstants)};

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
            .setLayoutCount = 1, .pSetLayouts = &*resources->getDescriptorSetLayout(), .pushConstantRangeCount = 1, .pPushConstantRanges = &pushConstantRange};
        pipelineLayouts.emplace_back(vk::raii::PipelineLayout(devices->getLogicalDevice(), pipelineLayoutInfo));

        vk::Format depthFormat = findDepthFormat(devices);
        vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo{
            .colorAttachmentCount = 1, .pColorAttachmentFormats = &swapChain->getSwapChainImageFormat(), .depthAttachmentFormat = depthFormat};

        vk::GraphicsPipelineCreateInfo pipelineInfo{.pNext = &pipelineRenderingCreateInfo,
                                                    .stageCount = 2,
                                                    .pStages = shaderStages,
                                                    .pVertexInputState = &vertexInputInfo,
                                                    .pInputAssemblyState = &inputAssembly,
                                                    .pViewportState = &viewportState,
                                                    .pRasterizationState = &rasterizer,
                                                    .pMultisampleState = &multisampling,
                                                    .pDepthStencilState = &depthStencil,
                                                    .pColorBlendState = &colorBlending,
                                                    .pDynamicState = &dynamicState,
                                                    .layout = *pipelineLayouts[index],
                                                    .renderPass = nullptr, // is nullptr because we're using dynamic rendering
                                                    .basePipelineHandle = VK_NULL_HANDLE,
                                                    .basePipelineIndex = -1};

        return vk::raii::Pipeline(devices->getLogicalDevice(), nullptr, pipelineInfo);
    }

    vk::raii::Pipeline createLinePipeline(vk::SampleCountFlagBits msaaSamples) {

        size_t index = pipelineLayouts.size();

        vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shaders/line.spv"), &*devices);
        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{.stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "vertMain"};

        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{.stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain"};

        vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
            .vertexBindingDescriptionCount = 0, .pVertexBindingDescriptions = nullptr, .vertexAttributeDescriptionCount = 0, .pVertexAttributeDescriptions = nullptr};

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{.topology = vk::PrimitiveTopology::eLineList};
        vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1, .scissorCount = 1};

        vk::PipelineRasterizationStateCreateInfo rasterizer{.depthClampEnable = vk::False,
                                                            .rasterizerDiscardEnable = vk::False,
                                                            .polygonMode = vk::PolygonMode::eLine,
                                                            .cullMode = vk::CullModeFlagBits::eNone,
                                                            .frontFace = vk::FrontFace::eCounterClockwise,
                                                            .depthBiasEnable = vk::False,
                                                            .depthBiasSlopeFactor = 1.0f,
                                                            .lineWidth = 5.0f};

        vk::PipelineMultisampleStateCreateInfo multisampling{.rasterizationSamples = msaaSamples, .sampleShadingEnable = vk::True, .minSampleShading = 0.2f};

        vk::PipelineDepthStencilStateCreateInfo depthStencil{.depthTestEnable = vk::False,
                                                             .depthWriteEnable = vk::False,
                                                             .depthCompareOp = vk::CompareOp::eLess,
                                                             .depthBoundsTestEnable = vk::False,
                                                             .stencilTestEnable = vk::False};

        vk::PipelineColorBlendAttachmentState colorBlendAttachment{.blendEnable = vk::False,
                                                                   .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                                                                                     vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};

        vk::PipelineColorBlendStateCreateInfo colorBlending{.logicOpEnable = vk::False, .logicOp = vk::LogicOp::eCopy, .attachmentCount = 1, .pAttachments = &colorBlendAttachment};

        std::vector dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
        vk::PipelineDynamicStateCreateInfo dynamicState{.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data()};

        vk::PushConstantRange pushConstantRange{.stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, .offset = 0, .size = sizeof(Line)};

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
            .setLayoutCount = 1, 
            .pSetLayouts = &*resources->getDescriptorSetLayout(), 
            .pushConstantRangeCount = 1, 
            .pPushConstantRanges = &pushConstantRange};
        pipelineLayouts.emplace_back(vk::raii::PipelineLayout(devices->getLogicalDevice(), pipelineLayoutInfo));

        vk::Format depthFormat = findDepthFormat(devices);
        vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo{
            .colorAttachmentCount = 1, .pColorAttachmentFormats = &swapChain->getSwapChainImageFormat(), .depthAttachmentFormat = depthFormat};

        vk::GraphicsPipelineCreateInfo pipelineInfo{.pNext = &pipelineRenderingCreateInfo,
                                                    .stageCount = 2,
                                                    .pStages = shaderStages,
                                                    .pVertexInputState = &vertexInputInfo,
                                                    .pInputAssemblyState = &inputAssembly,
                                                    .pViewportState = &viewportState,
                                                    .pRasterizationState = &rasterizer,
                                                    .pMultisampleState = &multisampling,
                                                    .pDepthStencilState = &depthStencil,
                                                    .pColorBlendState = &colorBlending,
                                                    .pDynamicState = &dynamicState,
                                                    .layout = *pipelineLayouts[index],
                                                    .renderPass = nullptr, // is nullptr because we're using dynamic rendering
                                                    .basePipelineHandle = VK_NULL_HANDLE,
                                                    .basePipelineIndex = -1};

        return vk::raii::Pipeline(devices->getLogicalDevice(), nullptr, pipelineInfo);
    }

  private:
    vk::PipelineCache pipelineCache;
    const Devices* devices;
    const BindlessResourceManager* resources;
    SwapChain* swapChain;

    std::vector<vk::raii::PipelineLayout> pipelineLayouts;

    [[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char>& code, const Devices* devices) const {
        vk::ShaderModuleCreateInfo createInfo{
            .codeSize = code.size(),  // Already in bytes, don't multiply!
            .pCode = reinterpret_cast<const uint32_t*>(code.data())
        };
        vk::raii::ShaderModule shaderModule{devices->getLogicalDevice(), createInfo};
        return shaderModule;
    }
};