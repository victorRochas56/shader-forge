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
/*
handles pipeline creation
BEFORE_GEOMETRY is only used by shadow rendering
GEOMETRY is the main pass with 2 color attachments (color + roughness/metallic)
POSTPROCESS are passes that process color attachments
*/
enum class PipelineCategory { BEFORE_GEOMETRY, LIT_GEOMETRY, ALPHA_GEOMETRY, POSTPROCESS, POSTPROCESS_MULTIPLY, POSTPROCESS_ALPHA_BLEND, POSTPROCESS_VOLUMETRIC, SHADOW, DEPTH_PREPASS, THUMBNAIL, MATERIAL_THUMBNAIL, VOXELIZATION };

class PipelineManager;

struct PipelineBase {
    vk::raii::Pipeline pipeline = nullptr;
    vk::raii::PipelineLayout layout = nullptr;
    vk::raii::DescriptorSetLayout* descriptorSetLayout = nullptr;
    vk::raii::DescriptorSet* descriptorSet = nullptr;
    std::string shaderPath;

    // store parameters for recreation
    PipelineCategory pipelineCategory;
    vk::PrimitiveTopology topology;
    vk::CullModeFlagBits cullMode;
    vk::Bool32 depthTestEnable;
    vk::Bool32 depthWriteEnable;
    // Required by callers. Used for POSTPROCESS / POSTPROCESS_MULTIPLY / POSTPROCESS_ALPHA_BLEND
    // categories whose target format isn't fixed by category. Ignored for SHADOW (depth-only),
    // DEPTH_PREPASS (depth-only), LIT_GEOMETRY (MRT, formats fixed), BEFORE_GEOMETRY (R32Sfloat fixed).
    vk::Format colorAttachmentFormat = vk::Format::eUndefined;

    std::type_index pushConstantType = std::type_index(typeid(void));

    PipelineBase() = default;

    virtual void pushConstants(vk::raii::CommandBuffer& cmd) = 0;
    virtual void recreateInternal(PipelineManager& manager, uint32_t index) = 0;
    virtual ~PipelineBase() = default;
};

// pipelines are templated with different push constants
template <typename T> struct Pipeline : public PipelineBase {
    T pushConstantData;

    Pipeline() { pushConstantType = std::type_index(typeid(T)); }

    void pushConstants(vk::raii::CommandBuffer& cmd) override {
        cmd.pushConstants<T>(layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pushConstantData);
    }

    void recreateInternal(PipelineManager& manager, uint32_t index) override;
};

struct ComputePipelineBase {
    vk::raii::Pipeline pipeline = nullptr;
    vk::raii::PipelineLayout layout = nullptr;
    vk::raii::DescriptorSet* descriptorSet = nullptr;
    vk::raii::DescriptorSetLayout* descriptorSetLayout = nullptr;
    std::string shaderPath;
    ComputePipelineBase() = default;
    virtual ~ComputePipelineBase() = default;
};

template <typename T> struct ComputePipeline : public ComputePipelineBase {
    T pushConstantData;

    void pushConstants(vk::raii::CommandBuffer& cmd) {
        cmd.pushConstants<T>(layout, vk::ShaderStageFlagBits::eCompute, 0, pushConstantData);
    } 
};

class PipelineManager {
  public:
    PipelineManager(Device& device, Swapchain& swapchain, vk::SampleCountFlagBits msaaSamples) : device(device), swapchain(swapchain), msaaSamples(msaaSamples) {}

    // colorAttachmentFormat is required. For POSTPROCESS* and VOXELIZATION it sets the pipeline's color target format.
    // For categories with a category-fixed format (SHADOW, DEPTH_PREPASS, LIT_GEOMETRY, BEFORE_GEOMETRY)
    // it is ignored — pass vk::Format::eUndefined.
    template <typename T>
    uint32_t createPipeline(PipelineCategory pipelineCategory, vk::PrimitiveTopology topology, vk::CullModeFlagBits cullMode, vk::Bool32 depthTestEnable,
                            vk::Bool32 depthWriteEnable, std::string shaderPath, vk::raii::DescriptorSetLayout& setLayout, vk::raii::DescriptorSet& descriptorSet,
                            vk::Format colorAttachmentFormat) {

        return createPipelineInternal<T>(pipelineCategory, topology, cullMode, depthTestEnable, depthWriteEnable, shaderPath, setLayout, descriptorSet, false, 0, colorAttachmentFormat);
    }

    // Watches each pipeline's .slang source (and the shared modules they import). On change it runs
    // slangc to regenerate the .spv, and only on a successful compile recreates the affected pipelines.
    void checkForShaderUpdates() {
        initShaderWatchState();

        // A shared module change invalidates every shader that imports it, so force-recompile all.
        bool modulesChanged = false;
        for (auto& [modPath, modTime] : moduleWriteTimes) {
            if (hasFileChanged(modPath, modTime)) modulesChanged = true;
        }

        for (auto& [spvPath, entry] : shaderPathToIndices) {
            std::string slangSrc = slangSourceForSpv(spvPath);
            bool srcChanged = std::filesystem::exists(slangSrc) && hasFileChanged(slangSrc, entry.first);
            if (!srcChanged && !modulesChanged) continue;

            if (!compileSlangToSpv(slangSrc, spvPath)) continue; // keep the working pipeline on a compile error

            // in-flight command buffers still reference these pipelines; destroying them mid-use is a device-lost
            device.getDevice().waitIdle();
            for (const auto& [category, index] : entry.second) {
                recreatePipelineAtIndex(category, index);
            }
        }
    }

    template <typename T>
    void recreatePipelineAtIndexInternal(PipelineCategory pipelineCategory, uint32_t index, const std::string& shaderPath, vk::PrimitiveTopology topology,
                                         vk::CullModeFlagBits cullMode, vk::Bool32 depthTestEnable, vk::Bool32 depthWriteEnable, vk::raii::DescriptorSetLayout& setLayout,
                                         vk::raii::DescriptorSet& descriptorSet, vk::Format colorAttachmentFormat) {
        createPipelineInternal<T>(pipelineCategory, topology, cullMode, depthTestEnable, depthWriteEnable, shaderPath, setLayout, descriptorSet, true, index, colorAttachmentFormat);
    }

    std::vector<std::unique_ptr<PipelineBase>>& getBeforeGeoPipelines() { return beforeGeometryPipelines; }
    std::vector<std::unique_ptr<PipelineBase>>& getGeoPipelines() { return geometryPipelines; }
    std::vector<std::unique_ptr<PipelineBase>>& getPostProcessPipelines() { return postProcessPipelines; }
    std::vector<std::unique_ptr<ComputePipelineBase>>& getComputePipelines() { return computePipelines; }

  private:
    Device& device;
    Swapchain& swapchain;
    vk::SampleCountFlagBits msaaSamples;

    // used for hot reloading shaders / recreating pipelines
                        //shader file               // last modified           // all pipelines associated (category & index)
    std::unordered_map<std::string, std::pair<std::filesystem::file_time_type,std::vector<std::pair<PipelineCategory, uint32_t>>>> shaderPathToIndices;

    // shared modules/*.slang imported by the top-level shaders; changing one invalidates every importer
    std::unordered_map<std::string, std::filesystem::file_time_type> moduleWriteTimes;
    bool shaderWatchInitialized = false;

    // Maps a runtime .spv path (e.g. "shaders/lit.spv") back to its source (SHADER_SRC_DIR/lit.slang).
    static std::string slangSourceForSpv(const std::string& spvPath) {
        return std::string(SHADER_SRC_DIR) + "/" + std::filesystem::path(spvPath).stem().string() + ".slang";
    }

    // Recompiles the .spv if it's missing or older than its .slang source. Covers shaders added after
    // the last CMake configure (whose .spv the build never produced) and any stale build output, so a
    // fresh launch always loads a pipeline built from current source. No-op in a shipped build (no sources).
    static void ensureSpvUpToDate(const std::string& spvPath) {
        std::string slangSrc = slangSourceForSpv(spvPath);
        if (!std::filesystem::exists(slangSrc)) return;
        if (!std::filesystem::exists(spvPath)) { compileSlangToSpv(slangSrc, spvPath); return; }

        auto spvTime = std::filesystem::last_write_time(spvPath);
        // Newest of the source and any shared module it might import (modules aren't tracked per shader,
        // so a module edit must invalidate every .spv — same reasoning as the hot-reload module check).
        auto newest = std::filesystem::last_write_time(slangSrc);
        std::filesystem::path modDir = std::filesystem::path(SHADER_SRC_DIR) / "modules";
        if (std::filesystem::exists(modDir)) {
            for (const auto& e : std::filesystem::directory_iterator(modDir)) {
                if (e.path().extension() == ".slang") newest = std::max(newest, e.last_write_time());
            }
        }
        if (newest > spvTime) compileSlangToSpv(slangSrc, spvPath);
    }

    // Records baseline mtimes for the shared modules so the first check doesn't see them all as changed.
    void initShaderWatchState() {
        if (shaderWatchInitialized) return;
        shaderWatchInitialized = true;
        std::filesystem::path modDir = std::filesystem::path(SHADER_SRC_DIR) / "modules";
        if (!std::filesystem::exists(modDir)) return;
        for (const auto& e : std::filesystem::directory_iterator(modDir)) {
            if (e.path().extension() == ".slang") moduleWriteTimes[e.path().string()] = e.last_write_time();
        }
    }

    std::vector<std::unique_ptr<PipelineBase>> beforeGeometryPipelines;
    std::vector<std::unique_ptr<PipelineBase>> geometryPipelines;
    std::vector<std::unique_ptr<PipelineBase>> postProcessPipelines;
    std::vector<std::unique_ptr<ComputePipelineBase>> computePipelines;

    void recreatePipelineAtIndex(PipelineCategory pipelineCategory, uint32_t index) {
        std::vector<std::unique_ptr<PipelineBase>>* targetVector = getTargetVector(pipelineCategory);
        if (!targetVector || index >= targetVector->size()) {
            throw std::out_of_range("Invalid pipeline index");
        }
        (*targetVector)[index]->recreateInternal(*this, index);
    }

    template <typename T>
    uint32_t createPipelineInternal(PipelineCategory pipelineCategory, vk::PrimitiveTopology topology, vk::CullModeFlagBits cullMode, vk::Bool32 depthTestEnable,
                                    vk::Bool32 depthWriteEnable, std::string shaderPath, vk::raii::DescriptorSetLayout& setLayout, vk::raii::DescriptorSet& descriptorSet,
                                    bool isRecreation, uint32_t recreateIndex, vk::Format colorAttachmentFormat) {

        ensureSpvUpToDate(shaderPath); // rebuild stale/missing .spv from source before loading it
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
        pipeline->colorAttachmentFormat = colorAttachmentFormat;

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly;
        vk::PipelineRasterizationStateCreateInfo rasterizer;
        vk::PipelineMultisampleStateCreateInfo multisampling;
        vk::PipelineDepthStencilStateCreateInfo depthStencil;
        vk::PipelineColorBlendAttachmentState colorBlendAttachment;
        vk::PipelineColorBlendStateCreateInfo colorBlending;
        vk::PushConstantRange pushConstantRange;

        // Format variables must persist beyond the switch scope since pointers to them are used
        vk::Format pcfFormat = vk::Format::eR32Sfloat; // PCF uses R32F for raw depth values
        vk::Format mrtFormats[3] = {swapchain.getHDRColorFormat(), vk::Format::eR8G8B8A8Unorm, vk::Format::eR8G8B8A8Unorm};
        vk::Format thumbMrtFormats[3] = {vk::Format::eR8G8B8A8Unorm, vk::Format::eR8G8B8A8Unorm, vk::Format::eR8G8B8A8Unorm};
        vk::PipelineColorBlendAttachmentState mrtBlendAttachments[3] = {
            {.blendEnable = vk::False, .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA},
            {.blendEnable = vk::False, .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA},
            {.blendEnable = vk::False, .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA}
        };

        switch (pipelineCategory) {
        case PipelineCategory::SHADOW: {
            depthFormat = vk::Format::eD32Sfloat;
            pipelineRenderingCreateInfo = {.colorAttachmentCount = 0, .pColorAttachmentFormats = nullptr, .depthAttachmentFormat = depthFormat};
            inputAssembly = {.topology = topology};
            rasterizer = {.depthClampEnable = vk::False,
                          .rasterizerDiscardEnable = vk::False,
                          .polygonMode = vk::PolygonMode::eFill,
                          .cullMode = cullMode,
                          .frontFace = vk::FrontFace::eCounterClockwise,
                          .depthBiasEnable = vk::True,
                          .depthBiasConstantFactor = 1.25f,
                          .depthBiasSlopeFactor = 1.75f};
            multisampling = {.rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False};
            depthStencil = {.depthTestEnable = depthTestEnable,
                            .depthWriteEnable = depthWriteEnable,
                            .depthCompareOp = vk::CompareOp::eLessOrEqual,
                            .depthBoundsTestEnable = vk::False,
                            .stencilTestEnable = vk::False};
            colorBlending = {.logicOpEnable = vk::False, .attachmentCount = 0, .pAttachments = nullptr};
            pushConstantRange = {.stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, .offset = 0, .size = sizeof(T)};
            vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
                .setLayoutCount = 1, .pSetLayouts = &*setLayout, .pushConstantRangeCount = 1, .pPushConstantRanges = &pushConstantRange};
            pipeline->layout = vk::raii::PipelineLayout(device.getDevice(), pipelineLayoutInfo);
            break;
        }
        case PipelineCategory::BEFORE_GEOMETRY: {
            // Only sets depth format if depth testing is enabled for this specific pipeline
            if (depthTestEnable == vk::True) {
                depthFormat = vk::Format::eD32Sfloat;
            } else {
                std::cout << "UNDEFINED PIPELINE DEPTH LAYOUT" << std::endl;
                depthFormat = vk::Format::eUndefined;
            }
            pipelineRenderingCreateInfo = {.colorAttachmentCount = 1, .pColorAttachmentFormats = &pcfFormat, .depthAttachmentFormat = depthFormat};
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
            colorBlending = {.logicOpEnable = vk::False, .logicOp = vk::LogicOp::eCopy, .attachmentCount = 1, .pAttachments = &colorBlendAttachment};
            pushConstantRange = {.stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, .offset = 0, .size = sizeof(T)};
            vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
                .setLayoutCount = 1, .pSetLayouts = &*setLayout, .pushConstantRangeCount = 1, .pPushConstantRanges = &pushConstantRange};
            pipeline->layout = vk::raii::PipelineLayout(device.getDevice(), pipelineLayoutInfo);
            break;
        }
        case PipelineCategory::DEPTH_PREPASS: {
            depthFormat = vk::Format::eD32Sfloat;
            pipelineRenderingCreateInfo = {.colorAttachmentCount = 0, .pColorAttachmentFormats = nullptr, .depthAttachmentFormat = depthFormat};
            inputAssembly = {.topology = topology};
            rasterizer = {.depthClampEnable = vk::False,
                          .rasterizerDiscardEnable = vk::False,
                          .polygonMode = vk::PolygonMode::eFill,
                          .cullMode = cullMode,
                          .frontFace = vk::FrontFace::eCounterClockwise,
                          .depthBiasEnable = vk::False,
                          .depthBiasSlopeFactor = 1.0f};
            multisampling = {.rasterizationSamples = msaaSamples, .sampleShadingEnable = vk::False, .minSampleShading = 0.2f};
            depthStencil = {.depthTestEnable = depthTestEnable,
                            .depthWriteEnable = depthWriteEnable,
                            .depthCompareOp = vk::CompareOp::eGreaterOrEqual, // reverse-Z
                            .depthBoundsTestEnable = vk::False,
                            .stencilTestEnable = vk::False};
            colorBlending = {.logicOpEnable = vk::False, .attachmentCount = 0, .pAttachments = nullptr};
            pushConstantRange = {.stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, .offset = 0, .size = sizeof(T)};
            vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
                .setLayoutCount = 1, .pSetLayouts = &*setLayout, .pushConstantRangeCount = 1, .pPushConstantRanges = &pushConstantRange};
            pipeline->layout = vk::raii::PipelineLayout(device.getDevice(), pipelineLayoutInfo);
            break;
        }
        case PipelineCategory::THUMBNAIL: {
            // Single-sample geometry pass into a small offscreen target for GUI mesh previews.
            depthFormat = vk::Format::eD32Sfloat;
            pipelineRenderingCreateInfo = {.colorAttachmentCount = 1, .pColorAttachmentFormats = &pipeline->colorAttachmentFormat, .depthAttachmentFormat = depthFormat};
            inputAssembly = {.topology = topology};
            rasterizer = {.depthClampEnable = vk::False,
                          .rasterizerDiscardEnable = vk::False,
                          .polygonMode = vk::PolygonMode::eFill,
                          .cullMode = cullMode,
                          .frontFace = vk::FrontFace::eCounterClockwise,
                          .depthBiasEnable = vk::False,
                          .lineWidth = 1.0f};
            multisampling = {.rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False};
            depthStencil = {.depthTestEnable = depthTestEnable,
                            .depthWriteEnable = depthWriteEnable,
                            .depthCompareOp = vk::CompareOp::eLessOrEqual,
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
        case PipelineCategory::LIT_GEOMETRY: {
            pipelineRenderingCreateInfo = {.colorAttachmentCount = 3, .pColorAttachmentFormats = mrtFormats, .depthAttachmentFormat = depthFormat};
            inputAssembly = {.topology = topology};
            rasterizer = {.depthClampEnable = vk::False,
                          .rasterizerDiscardEnable = vk::False,
                          .polygonMode = vk::PolygonMode::eFill,
                          .cullMode = cullMode,
                          .frontFace = vk::FrontFace::eCounterClockwise,
                          .depthBiasEnable = vk::False,
                          .depthBiasSlopeFactor = 1.0f,
                          .lineWidth = 1.0f};
            multisampling = {.rasterizationSamples = msaaSamples, .sampleShadingEnable = vk::False, .minSampleShading = 0.2f};
            depthStencil = {.depthTestEnable = depthTestEnable,
                            .depthWriteEnable = depthWriteEnable,
                            .depthCompareOp = vk::CompareOp::eGreaterOrEqual, // reverse-Z
                            .depthBoundsTestEnable = vk::False,
                            .stencilTestEnable = vk::False};
            colorBlending = {.logicOpEnable = vk::False, .logicOp = vk::LogicOp::eCopy, .attachmentCount = 3, .pAttachments = mrtBlendAttachments};
            pushConstantRange = {.stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, .offset = 0, .size = sizeof(T)};
            vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
                .setLayoutCount = 1, .pSetLayouts = &*setLayout, .pushConstantRangeCount = 1, .pPushConstantRanges = &pushConstantRange};
            pipeline->layout = vk::raii::PipelineLayout(device.getDevice(), pipelineLayoutInfo);
            break;
        }
        case PipelineCategory::MATERIAL_THUMBNAIL: {
            // Single-sample lit pass into small MRT targets for material GUI previews (reuses lit.spv).
            depthFormat = vk::Format::eD32Sfloat;
            pipelineRenderingCreateInfo = {.colorAttachmentCount = 3, .pColorAttachmentFormats = thumbMrtFormats, .depthAttachmentFormat = depthFormat};
            inputAssembly = {.topology = topology};
            rasterizer = {.depthClampEnable = vk::False,
                          .rasterizerDiscardEnable = vk::False,
                          .polygonMode = vk::PolygonMode::eFill,
                          .cullMode = cullMode,
                          .frontFace = vk::FrontFace::eCounterClockwise,
                          .depthBiasEnable = vk::False,
                          .lineWidth = 1.0f};
            multisampling = {.rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False};
            depthStencil = {.depthTestEnable = depthTestEnable,
                            .depthWriteEnable = depthWriteEnable,
                            .depthCompareOp = vk::CompareOp::eLessOrEqual,
                            .depthBoundsTestEnable = vk::False,
                            .stencilTestEnable = vk::False};
            colorBlending = {.logicOpEnable = vk::False, .attachmentCount = 3, .pAttachments = mrtBlendAttachments};
            pushConstantRange = {.stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, .offset = 0, .size = sizeof(T)};
            vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
                .setLayoutCount = 1, .pSetLayouts = &*setLayout, .pushConstantRangeCount = 1, .pPushConstantRanges = &pushConstantRange};
            pipeline->layout = vk::raii::PipelineLayout(device.getDevice(), pipelineLayoutInfo);
            break;
        }
        case PipelineCategory::ALPHA_GEOMETRY: {
            pipelineRenderingCreateInfo = {.colorAttachmentCount = 1, .pColorAttachmentFormats = &swapchain.getSwapChainImageFormat(), .depthAttachmentFormat = depthFormat};
            inputAssembly = {.topology = topology};
            rasterizer = {.depthClampEnable = vk::False,
                          .rasterizerDiscardEnable = vk::False,
                          .polygonMode = vk::PolygonMode::eFill,
                          .cullMode = cullMode,
                          .frontFace = vk::FrontFace::eCounterClockwise,
                          .depthBiasEnable = vk::False,
                          .depthBiasSlopeFactor = 1.0f,
                          .lineWidth = 1.0f};
            multisampling = {.rasterizationSamples = msaaSamples, .sampleShadingEnable = vk::False};
            depthStencil = {.depthTestEnable = depthTestEnable,
                            .depthWriteEnable = depthWriteEnable,
                            .depthCompareOp = vk::CompareOp::eGreaterOrEqual, // reverse-Z
                            .depthBoundsTestEnable = vk::False,
                            .stencilTestEnable = vk::False};
            colorBlendAttachment = {.blendEnable = vk::True,
                                    .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
                                    .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
                                    .colorBlendOp = vk::BlendOp::eAdd,
                                    .srcAlphaBlendFactor = vk::BlendFactor::eOne,
                                    .dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
                                    .alphaBlendOp = vk::BlendOp::eAdd,
                                    .colorWriteMask =
                                        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};
            colorBlending = {.logicOpEnable = vk::False, .logicOp = vk::LogicOp::eCopy, .attachmentCount = 1, .pAttachments = &colorBlendAttachment};
            pushConstantRange = {.stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, .offset = 0, .size = sizeof(T)};
            vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
                .setLayoutCount = 1, .pSetLayouts = &*setLayout, .pushConstantRangeCount = 1, .pPushConstantRanges = &pushConstantRange};
            pipeline->layout = vk::raii::PipelineLayout(device.getDevice(), pipelineLayoutInfo);
            break;
        }
        case PipelineCategory::POSTPROCESS_ALPHA_BLEND:
        case PipelineCategory::POSTPROCESS_VOLUMETRIC:
        case PipelineCategory::POSTPROCESS_MULTIPLY:
        case PipelineCategory::POSTPROCESS: {
            bool multiply = (pipelineCategory == PipelineCategory::POSTPROCESS_MULTIPLY);
            bool alphaBlend = (pipelineCategory == PipelineCategory::POSTPROCESS_ALPHA_BLEND);
            bool volumetric = (pipelineCategory == PipelineCategory::POSTPROCESS_VOLUMETRIC);
            assert(colorAttachmentFormat != vk::Format::eUndefined && "POSTPROCESS pipelines require an explicit colorAttachmentFormat");
            pipelineRenderingCreateInfo = {.colorAttachmentCount = 1, .pColorAttachmentFormats = &pipeline->colorAttachmentFormat, .depthAttachmentFormat = depthFormat};
            inputAssembly = {.topology = topology};
            rasterizer = {.depthClampEnable = vk::False,
                          .rasterizerDiscardEnable = vk::False,
                          .polygonMode = vk::PolygonMode::eFill,
                          .cullMode = cullMode,
                          .frontFace = vk::FrontFace::eCounterClockwise,
                          .depthBiasEnable = vk::False,
                          .lineWidth = 1.0f};
            multisampling = {.rasterizationSamples = vk::SampleCountFlagBits::e1,
                             .sampleShadingEnable = vk::False};
            depthStencil = {.depthTestEnable = (multiply || alphaBlend || volumetric) ? vk::False : depthTestEnable,
                            .depthWriteEnable = (multiply || alphaBlend || volumetric) ? vk::False : depthWriteEnable,
                            .depthCompareOp = vk::CompareOp::eNever,
                            .depthBoundsTestEnable = vk::False,
                            .stencilTestEnable = vk::False};
            if (alphaBlend) {
                colorBlendAttachment = {.blendEnable = vk::True,
                                        .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
                                        .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
                                        .colorBlendOp = vk::BlendOp::eAdd,
                                        .srcAlphaBlendFactor = vk::BlendFactor::eOne,
                                        .dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
                                        .alphaBlendOp = vk::BlendOp::eAdd,
                                        .colorWriteMask =
                                            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};
            } else if (volumetric) {
                // Froxel composite E: out.rgb = inScatter + sceneColor * transmittance.
                // Shader outputs rgb = accumulated in-scatter (premultiplied), a = transmittance.
                colorBlendAttachment = {.blendEnable = vk::True,
                                        .srcColorBlendFactor = vk::BlendFactor::eOne,
                                        .dstColorBlendFactor = vk::BlendFactor::eSrcAlpha,
                                        .colorBlendOp = vk::BlendOp::eAdd,
                                        .srcAlphaBlendFactor = vk::BlendFactor::eOne,
                                        .dstAlphaBlendFactor = vk::BlendFactor::eZero,
                                        .alphaBlendOp = vk::BlendOp::eAdd,
                                        .colorWriteMask =
                                            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};
            } else {
                colorBlendAttachment = {.blendEnable = multiply ? vk::True : vk::False,
                                        .srcColorBlendFactor = vk::BlendFactor::eDstColor,
                                        .dstColorBlendFactor = vk::BlendFactor::eZero,
                                        .colorBlendOp = vk::BlendOp::eAdd,
                                        .srcAlphaBlendFactor = vk::BlendFactor::eOne,
                                        .dstAlphaBlendFactor = vk::BlendFactor::eZero,
                                        .alphaBlendOp = vk::BlendOp::eAdd,
                                        .colorWriteMask =
                                            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};
            }
            colorBlending = {.logicOpEnable = vk::False, .attachmentCount = 1, .pAttachments = &colorBlendAttachment};
            pushConstantRange = {.stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, .offset = 0, .size = sizeof(T)};
            vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
                .setLayoutCount = 1, .pSetLayouts = &*setLayout, .pushConstantRangeCount = 1, .pPushConstantRanges = &pushConstantRange};
            pipeline->layout = vk::raii::PipelineLayout(device.getDevice(), pipelineLayoutInfo);
            break;
        }
        // Slicemap voxelization. The target is a UINT 2D image where bit k of a texel marks occupancy
        // of slice k along the sweep axis, so the fragment shader emits raw bits and overlapping
        // triangles merge with a fixed-function OR — no atomics, no storage image. Depth test/write
        // are forced off (every fragment must land) and there is no depth attachment.
        case PipelineCategory::VOXELIZATION: {
            assert(colorAttachmentFormat != vk::Format::eUndefined && "VOXELIZATION pipelines require an explicit UINT colorAttachmentFormat");
            pipelineRenderingCreateInfo = {
                .colorAttachmentCount = 1, .pColorAttachmentFormats = &pipeline->colorAttachmentFormat, .depthAttachmentFormat = vk::Format::eUndefined};
            inputAssembly = {.topology = topology};
            rasterizer = {.depthClampEnable = vk::False,
                          .rasterizerDiscardEnable = vk::False,
                          .polygonMode = vk::PolygonMode::eFill,
                          .cullMode = cullMode,
                          .frontFace = vk::FrontFace::eCounterClockwise,
                          .depthBiasEnable = vk::False,
                          .lineWidth = 1.0f};
            multisampling = {.rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False};
            depthStencil = {.depthTestEnable = vk::False,
                            .depthWriteEnable = vk::False,
                            .depthCompareOp = vk::CompareOp::eNever,
                            .depthBoundsTestEnable = vk::False,
                            .stencilTestEnable = vk::False};
            // logicOp replaces blending outright — the two are mutually exclusive, so blendEnable stays false.
            colorBlendAttachment = {.blendEnable = vk::False,
                                    .colorWriteMask =
                                        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};
            colorBlending = {.logicOpEnable = vk::True, .logicOp = vk::LogicOp::eOr, .attachmentCount = 1, .pAttachments = &colorBlendAttachment};
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
            // baseline the .slang source mtime so the first checkForShaderUpdates doesn't see it as "changed"
            std::string slangSrc = slangSourceForSpv(shaderPath);
            if (std::filesystem::exists(slangSrc)) {
                shaderPathToIndices[shaderPath].first = std::filesystem::last_write_time(slangSrc);
            }
            shaderPathToIndices[shaderPath].second.push_back({pipelineCategory, newIndex});
            return newIndex;
        }
    }

    [[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const {
        vk::ShaderModuleCreateInfo createInfo{.codeSize = code.size(), .pCode = reinterpret_cast<const uint32_t*>(code.data())};
        vk::raii::ShaderModule shaderModule{device.getDevice(), createInfo};
        return shaderModule;
    }

    std::vector<std::unique_ptr<PipelineBase>>* getTargetVector(PipelineCategory pipelineCategory) {
        switch (pipelineCategory) {
        case PipelineCategory::SHADOW:
        case PipelineCategory::BEFORE_GEOMETRY:
        case PipelineCategory::DEPTH_PREPASS:
        case PipelineCategory::THUMBNAIL:
        case PipelineCategory::MATERIAL_THUMBNAIL:
        case PipelineCategory::VOXELIZATION:
            return &beforeGeometryPipelines;
        case PipelineCategory::LIT_GEOMETRY:
        case PipelineCategory::ALPHA_GEOMETRY:
            return &geometryPipelines;
        case PipelineCategory::POSTPROCESS:
        case PipelineCategory::POSTPROCESS_MULTIPLY:
        case PipelineCategory::POSTPROCESS_ALPHA_BLEND:
        case PipelineCategory::POSTPROCESS_VOLUMETRIC:
            return &postProcessPipelines;
        default:
            return nullptr;
        }
    }

public:
    template <typename T>
    uint32_t createComputePipeline(std::string shaderPath, vk::raii::DescriptorSetLayout& setLayout, vk::raii::DescriptorSet& descriptorSet, const char* entryPoint = "computeMain") {
        ensureSpvUpToDate(shaderPath); // build .spv from .slang like the graphics path (pipelines.hpp:189)
        vk::raii::ShaderModule module = createShaderModule(readFile(shaderPath));
        // One VkPipeline binds one entry point. A multi-kernel .spv (e.g. particle spawn/integrate)
        // is turned into several pipelines by calling this once per entry point with the same path.
        vk::PipelineShaderStageCreateInfo stage{.stage = vk::ShaderStageFlagBits::eCompute, .module = module, .pName = entryPoint};
        vk::PushConstantRange pc{vk::ShaderStageFlagBits::eCompute, 0, sizeof(T)};
        vk::PipelineLayoutCreateInfo layoutInfo{
            .setLayoutCount = 1, .pSetLayouts = &*setLayout, .pushConstantRangeCount = 1, .pPushConstantRanges = &pc};

        std::unique_ptr<ComputePipeline<T>> pipeline = std::make_unique<ComputePipeline<T>>();
        pipeline->layout = vk::raii::PipelineLayout(device.getDevice(), layoutInfo);
        pipeline->descriptorSet = &descriptorSet;
        pipeline->descriptorSetLayout = &setLayout;
        vk::ComputePipelineCreateInfo info{.stage = stage, .layout = pipeline->layout};
        pipeline->pipeline = vk::raii::Pipeline(device.getDevice(), nullptr, info);

        computePipelines.push_back(std::move(pipeline));
        return computePipelines.size() - 1; // 0-based index, matching the graphics path
    }
};

template <typename T> void Pipeline<T>::recreateInternal(PipelineManager& manager, uint32_t index) {
    manager.recreatePipelineAtIndexInternal<T>(pipelineCategory, index, shaderPath, topology, cullMode, depthTestEnable, depthWriteEnable, *descriptorSetLayout, *descriptorSet, colorAttachmentFormat);
}