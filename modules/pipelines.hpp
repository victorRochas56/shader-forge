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
#include <typeindex>
#include <unordered_map>

#include "descriptor_sets.hpp"
#include "devices.hpp"
#include "swapchain.hpp"
#include "utils.hpp"

enum class PipelineCategory { BEFORE_GEOMETRY, GEOMETRY, AFTER_GEOMETRY };

class PipelineManager;

struct PipelineBase {
    vk::raii::Pipeline pipeline = nullptr;
    vk::raii::PipelineLayout layout = nullptr;
    vk::raii::DescriptorSetLayout* descriptorSetLayout = nullptr;
    vk::raii::DescriptorSet* descriptorSet = nullptr;
    std::string shaderPath;

    // Creation parameters needed for recreation
    PipelineCategory pipelineCategory;
    vk::PrimitiveTopology topology;
    vk::CullModeFlagBits cullMode;
    vk::Bool32 depthTestEnable;
    vk::Bool32 depthWriteEnable;

    std::type_index pushConstantType = std::type_index(typeid(void));

    PipelineBase() = default;

    virtual void pushConstants(vk::raii::CommandBuffer& cmd) = 0;
    virtual void recreateInternal(PipelineManager& manager) = 0;
    virtual ~PipelineBase() = default;
};

template <typename T> struct Pipeline : public PipelineBase {
    T pushConstantData;

    Pipeline() { pushConstantType = std::type_index(typeid(T)); }

    void pushConstants(vk::raii::CommandBuffer& cmd) override {
        cmd.pushConstants<T>(layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pushConstantData);
    }

    void recreateInternal(PipelineManager& manager) override;
};

class PipelineManager {
  public:
    PipelineManager(Device& device, Swapchain& swapchain, vk::SampleCountFlagBits msaaSamples) : device(device), swapchain(swapchain), msaaSamples(msaaSamples) {}

    template <typename T>
    uint32_t createPipeline(PipelineCategory pipelineCategory, vk::PrimitiveTopology topology, vk::CullModeFlagBits cullMode, vk::Bool32 depthTestEnable,
                            vk::Bool32 depthWriteEnable, std::string shaderPath, vk::raii::DescriptorSetLayout& setLayout, vk::raii::DescriptorSet& descriptorSet) {

        return createPipelineInternal<T>(pipelineCategory, topology, cullMode, depthTestEnable, depthWriteEnable, shaderPath, setLayout, descriptorSet, false, 0);
    }

    void checkForShaderUpdates() {
        for (const auto& [shaderPath, indices] : shaderPathToIndices) {
            if (hasFileChanged(shaderPath)) {
                for (const auto& [category, index] : indices) {
                    recreatePipelineAtIndex(category, index);
                }
            }
        }
    }

    std::vector<std::unique_ptr<PipelineBase>>& getBeforeGeoPipelines() { return beforeGeometryPipelines; }
    std::vector<std::unique_ptr<PipelineBase>>& getGeoPipelines() { return geometryPipelines; }
    std::vector<std::unique_ptr<PipelineBase>>& getAfterGeoPipelines() { return afterGeometryPipelines; }

    template <typename T>
    void recreatePipelineAtIndexInternal(PipelineCategory pipelineCategory, uint32_t index, const std::string& shaderPath, vk::PrimitiveTopology topology,
                                         vk::CullModeFlagBits cullMode, vk::Bool32 depthTestEnable, vk::Bool32 depthWriteEnable, vk::raii::DescriptorSetLayout& setLayout,
                                         vk::raii::DescriptorSet& descriptorSet) {
        createPipelineInternal<T>(pipelineCategory, topology, cullMode, depthTestEnable, depthWriteEnable, shaderPath, setLayout, descriptorSet, true, index);
    }

  private:
    Device& device;
    Swapchain& swapchain;
    vk::SampleCountFlagBits msaaSamples;

    std::unordered_map<std::string, std::vector<std::pair<PipelineCategory, uint32_t>>> shaderPathToIndices;

    std::vector<std::unique_ptr<PipelineBase>> beforeGeometryPipelines;
    std::vector<std::unique_ptr<PipelineBase>> geometryPipelines;
    std::vector<std::unique_ptr<PipelineBase>> afterGeometryPipelines;

    [[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const {
        vk::ShaderModuleCreateInfo createInfo{.codeSize = code.size(), .pCode = reinterpret_cast<const uint32_t*>(code.data())};
        vk::raii::ShaderModule shaderModule{device.getDevice(), createInfo};
        return shaderModule;
    }

    std::vector<std::unique_ptr<PipelineBase>>* getTargetVector(PipelineCategory pipelineCategory) {
        switch (pipelineCategory) {
        case PipelineCategory::BEFORE_GEOMETRY:
            return &beforeGeometryPipelines;
        case PipelineCategory::GEOMETRY:
            return &geometryPipelines;
        case PipelineCategory::AFTER_GEOMETRY:
            return &afterGeometryPipelines;
        default:
            return nullptr;
        }
    }

    void recreatePipelineAtIndex(PipelineCategory pipelineCategory, uint32_t index) {
        std::vector<std::unique_ptr<PipelineBase>>* targetVector = getTargetVector(pipelineCategory);
        if (!targetVector || index >= targetVector->size()) {
            throw std::out_of_range("Invalid pipeline index");
        }
        (*targetVector)[index]->recreateInternal(*this);
    }

    template <typename T>
    uint32_t createPipelineInternal(PipelineCategory pipelineCategory, vk::PrimitiveTopology topology, vk::CullModeFlagBits cullMode, vk::Bool32 depthTestEnable,
                                    vk::Bool32 depthWriteEnable, std::string shaderPath, vk::raii::DescriptorSetLayout& setLayout, vk::raii::DescriptorSet& descriptorSet,
                                    bool isRecreation, uint32_t recreateIndex) {

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

        auto pipeline = std::make_unique<Pipeline<T>>();

        // store parameters for recreation
        pipeline->pipelineCategory = pipelineCategory;
        pipeline->topology = topology;
        pipeline->cullMode = cullMode;
        pipeline->depthTestEnable = depthTestEnable;
        pipeline->depthWriteEnable = depthWriteEnable;
        pipeline->shaderPath = shaderPath;
        pipeline->descriptorSetLayout = &setLayout;
        pipeline->descriptorSet = &descriptorSet;

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly;
        vk::PipelineRasterizationStateCreateInfo rasterizer;
        vk::PipelineMultisampleStateCreateInfo multisampling;
        vk::PipelineDepthStencilStateCreateInfo depthStencil;
        vk::PipelineColorBlendAttachmentState colorBlendAttachment;
        vk::PipelineColorBlendStateCreateInfo colorBlending;
        vk::PushConstantRange pushConstantRange;

        switch (pipelineCategory) {
        case PipelineCategory::BEFORE_GEOMETRY: {
            pipelineRenderingCreateInfo = {.colorAttachmentCount = 0, .pColorAttachmentFormats = nullptr, .depthAttachmentFormat = vk::Format::eD32Sfloat};
            inputAssembly = {.topology = topology};
            rasterizer = {.depthClampEnable = vk::False,
                          .rasterizerDiscardEnable = vk::False,
                          .polygonMode = vk::PolygonMode::eFill,
                          .cullMode = cullMode,
                          .frontFace = vk::FrontFace::eCounterClockwise,
                          .depthBiasEnable = vk::False,
                          .depthBiasSlopeFactor = 1.0f};
            multisampling = {.rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False};
            depthStencil = {.depthTestEnable = depthTestEnable,
                            .depthWriteEnable = depthWriteEnable,
                            .depthCompareOp = vk::CompareOp::eLessOrEqual,
                            .depthBoundsTestEnable = vk::False,
                            .stencilTestEnable = vk::False};
            colorBlendAttachment = {.blendEnable = vk::False,
                                    .colorWriteMask =
                                        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};
            colorBlending = {.logicOpEnable = vk::False, .logicOp = vk::LogicOp::eCopy, .attachmentCount = 0, .pAttachments = nullptr};
            pushConstantRange = {.stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, .offset = 0, .size = sizeof(T)};
            vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
                .setLayoutCount = 1, .pSetLayouts = &*setLayout, .pushConstantRangeCount = 1, .pPushConstantRanges = &pushConstantRange};
            pipeline->layout = vk::raii::PipelineLayout(device.getDevice(), pipelineLayoutInfo);
            break;
        }
        case PipelineCategory::GEOMETRY: {
            inputAssembly = {.topology = topology};
            rasterizer = {.depthClampEnable = vk::False,
                          .rasterizerDiscardEnable = vk::False,
                          .polygonMode = vk::PolygonMode::eFill,
                          .cullMode = cullMode,
                          .frontFace = vk::FrontFace::eCounterClockwise,
                          .depthBiasEnable = vk::False,
                          .depthBiasSlopeFactor = 1.0f,
                          .lineWidth = 1.0f};
            multisampling = {.rasterizationSamples = msaaSamples, .sampleShadingEnable = vk::True, .minSampleShading = 0.2f};
            depthStencil = {.depthTestEnable = depthTestEnable,
                            .depthWriteEnable = depthWriteEnable,
                            .depthCompareOp = vk::CompareOp::eLessOrEqual,
                            .depthBoundsTestEnable = vk::False,
                            .stencilTestEnable = vk::False};
            colorBlendAttachment = {.blendEnable = vk::False,
                                    .colorWriteMask =
                                        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};
            colorBlending = {.logicOpEnable = vk::False, .logicOp = vk::LogicOp::eCopy, .attachmentCount = 1, .pAttachments = &colorBlendAttachment};
            pushConstantRange = {.stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, .offset = 0, .size = sizeof(T)};
            vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
                .setLayoutCount = 1, .pSetLayouts = &*setLayout, .pushConstantRangeCount = 1, .pPushConstantRanges = &pushConstantRange};
            pipeline->layout = vk::raii::PipelineLayout(device.getDevice(), pipelineLayoutInfo);
            break;
        }
        case PipelineCategory::AFTER_GEOMETRY: {
            inputAssembly = {.topology = topology};
            rasterizer = {.depthClampEnable = vk::False,
                          .rasterizerDiscardEnable = vk::False,
                          .polygonMode = vk::PolygonMode::eFill,
                          .cullMode = cullMode,
                          .frontFace = vk::FrontFace::eCounterClockwise,
                          .depthBiasEnable = vk::False,
                          .lineWidth = 1.0f};
            multisampling = {.rasterizationSamples = vk::SampleCountFlagBits::e1, // No MSAA for post-process
                             .sampleShadingEnable = vk::False};
            depthStencil = {.depthTestEnable = depthTestEnable,
                            .depthWriteEnable = depthWriteEnable,
                            .depthCompareOp = vk::CompareOp::eLess,
                            .depthBoundsTestEnable = vk::False,
                            .stencilTestEnable = vk::False};
            colorBlendAttachment = {.blendEnable = vk::False,
                                    .colorWriteMask =
                                        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};
            colorBlending = {.logicOpEnable = vk::False, .attachmentCount = 1, .pAttachments = &colorBlendAttachment};
            pushConstantRange = {.stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, .offset = 0, .size = sizeof(T)};
            vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
                .setLayoutCount = 1, .pSetLayouts = &*setLayout, .pushConstantRangeCount = 1, .pPushConstantRanges = &pushConstantRange};
            pipeline->layout = vk::raii::PipelineLayout(device.getDevice(), pipelineLayoutInfo);
            break;
        }
        default:
            throw std::range_error("invalid pipeline category specifier!");
        }

        vk::GraphicsPipelineCreateInfo createInfo = {.pNext = &pipelineRenderingCreateInfo,
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
                                                     .renderPass = nullptr,
                                                     .basePipelineHandle = VK_NULL_HANDLE,
                                                     .basePipelineIndex = -1};

        pipeline->pipeline = vk::raii::Pipeline(device.getDevice(), nullptr, createInfo);

        std::vector<std::unique_ptr<PipelineBase>>* targetVector = getTargetVector(pipelineCategory);

        if (isRecreation) {
            (*targetVector)[recreateIndex] = std::move(pipeline);
            return recreateIndex;
        } else {
            targetVector->push_back(std::move(pipeline));
            uint32_t newIndex = targetVector->size() - 1;
            shaderPathToIndices[shaderPath].push_back({pipelineCategory, newIndex});
            return newIndex;
        }
    }
};

template <typename T> void Pipeline<T>::recreateInternal(PipelineManager& manager) {
    manager.recreatePipelineAtIndexInternal<T>(pipelineCategory, 0, shaderPath, topology, cullMode, depthTestEnable, depthWriteEnable, *descriptorSetLayout, *descriptorSet);
}