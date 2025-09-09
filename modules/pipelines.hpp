#pragma once
#ifndef VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1 // for raii
#endif
#ifndef VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_NO_CONSTRUCTORS 1 // for structs constructors
#endif

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#define GLFW_INCLUDE_VULKAN // REQUIRED only for GLFW CreateWindowSurface.
#include <GLFW/glfw3.h>

#include "descriptor_sets.hpp"
#include "devices.hpp"
#include "swapchain.hpp"
#include "utils.hpp"

constexpr uint32_t BEFORE_GEOMETRY = 0;
constexpr uint32_t GEOMETRY = 1;
constexpr uint32_t AFTER_GEOMETRY = 2;

struct PipelineBase {
    vk::raii::Pipeline pipeline = nullptr;
    vk::raii::PipelineLayout layout = nullptr;
    vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
    vk::raii::DescriptorSet descriptorSet = nullptr;

    virtual void pushConstants(vk::raii::CommandBuffer& cmd) = 0;
    virtual ~PipelineBase() = default;
};

template <typename T> struct Pipeline : public PipelineBase {
    T pushConstantData;

    void pushConstants(vk::raii::CommandBuffer& cmd) override {
        cmd.pushConstants<T>(layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pushConstantData);
    }
};

class PipelineManager {
  public:
    PipelineManager(Device& device, Swapchain& swapchain, vk::SampleCountFlagBits msaaSamples) : device(device), swapchain(swapchain), msaaSamples(msaaSamples) {}

    template <typename T> void createPipeline(uint32_t pipelineCategory, std::string shaderPath, vk::raii::DescriptorSetLayout& setLayout, vk::raii::DescriptorSet& descriptorSet) {

        vk::raii::ShaderModule shaderModule = createShaderModule(readFile(shaderPath));
        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{.stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "vertMain"};
        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{.stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain"};
        vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};
        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
            .vertexBindingDescriptionCount = 0, .pVertexBindingDescriptions = nullptr, .vertexAttributeDescriptionCount = 0, .pVertexAttributeDescriptions = nullptr};
        vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1, .scissorCount = 1};
        std::vector dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
        vk::PipelineDynamicStateCreateInfo dynamicState{.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data()};
        vk::Format depthFormat = findDepthFormat(device);
        vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo{
            .colorAttachmentCount = 1, .pColorAttachmentFormats = &swapchain.getSwapChainImageFormat(), .depthAttachmentFormat = depthFormat};

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly;
        vk::PipelineRasterizationStateCreateInfo rasterizer;
        vk::PipelineMultisampleStateCreateInfo multisampling;
        vk::PipelineDepthStencilStateCreateInfo depthStencil;
        vk::PipelineColorBlendAttachmentState colorBlendAttachment;
        vk::PipelineColorBlendStateCreateInfo colorBlending;
        vk::PushConstantRange pushConstantRange;
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo;

        vk::GraphicsPipelineCreateInfo createInfo;
        switch (pipelineCategory) {
        case BEFORE_GEOMETRY:
            break;

        case GEOMETRY: {
            inputAssembly = {.topology = vk::PrimitiveTopology::eTriangleList};
            rasterizer = {.depthClampEnable = vk::False,
                          .rasterizerDiscardEnable = vk::False,
                          .polygonMode = vk::PolygonMode::eFill,
                          .cullMode = vk::CullModeFlagBits::eBack,
                          .frontFace = vk::FrontFace::eCounterClockwise,
                          .depthBiasEnable = vk::False,
                          .depthBiasSlopeFactor = 1.0f,
                          .lineWidth = 1.0f};
            multisampling = {.rasterizationSamples = msaaSamples, .sampleShadingEnable = vk::True, .minSampleShading = 0.2f};
            depthStencil = {.depthTestEnable = vk::True,
                            .depthWriteEnable = vk::True,
                            .depthCompareOp = vk::CompareOp::eLess,
                            .depthBoundsTestEnable = vk::False,
                            .stencilTestEnable = vk::False};
            colorBlendAttachment = {.blendEnable = vk::False,
                                    .colorWriteMask =
                                        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};
            colorBlending = {.logicOpEnable = vk::False, .logicOp = vk::LogicOp::eCopy, .attachmentCount = 1, .pAttachments = &colorBlendAttachment};
            pushConstantRange = {.stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, .offset = 0, .size = sizeof(T)};
            vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
                .setLayoutCount = 1, .pSetLayouts = &*setLayout, .pushConstantRangeCount = 1, .pPushConstantRanges = &pushConstantRange};
            break;
        }
        case AFTER_GEOMETRY:
            break;

        default:
            throw std::range_error("invalid pipeline category specifier!");
        }

        auto pipeline = std::make_unique<Pipeline<T>>();
        pipeline->layout = vk::raii::PipelineLayout(device.getDevice(), pipelineLayoutInfo);
        pipeline->descriptorSetLayout = std::move(setLayout);
        pipeline->descriptorSet = std::move(descriptorSet);

        createInfo = {.pNext = &pipelineRenderingCreateInfo,
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
                      .layout = pipeline->layout,
                      .renderPass = nullptr, // is nullptr because we're using dynamic rendering
                      .basePipelineHandle = VK_NULL_HANDLE,
                      .basePipelineIndex = -1};

        pipeline->pipeline = vk::raii::Pipeline(device.getDevice(), nullptr, createInfo);

        switch (pipelineCategory) {
        case BEFORE_GEOMETRY:
            break;
        case GEOMETRY:
            geometryPipelines.push_back(std::move(pipeline));
            break;
        case AFTER_GEOMETRY:
            break;
        }
    }

    std::vector<std::unique_ptr<PipelineBase>>& getBeforeGeoPipelines() { return beforeGeometryPipelines; }
    std::vector<std::unique_ptr<PipelineBase>>& getGeoPipelines() { return geometryPipelines; }
    std::vector<std::unique_ptr<PipelineBase>>& getAfterGeoPipelines() { return afterGeometryPipelines; }

  private:
    Device& device;
    Swapchain& swapchain;
    vk::SampleCountFlagBits msaaSamples;

    std::vector<std::unique_ptr<PipelineBase>> beforeGeometryPipelines; // skybox etc.
    std::vector<std::unique_ptr<PipelineBase>> geometryPipelines;       // meshes & anything depth buffered
    std::vector<std::unique_ptr<PipelineBase>> afterGeometryPipelines;  // ui etc.

    [[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const {
        vk::ShaderModuleCreateInfo createInfo{.codeSize = code.size(), .pCode = reinterpret_cast<const uint32_t*>(code.data())};
        vk::raii::ShaderModule shaderModule{device.getDevice(), createInfo};
        return shaderModule;
    }
};