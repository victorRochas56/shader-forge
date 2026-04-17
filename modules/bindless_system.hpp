#pragma once
#ifndef VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#endif
#ifndef VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_NO_CONSTRUCTORS 1
#endif

#include <memory>

#include "descriptor_sets.hpp"
#include "gpu_context.hpp"
#include "pipelines.hpp"
#include "resources.hpp"

/*
BindlessSystem bundles the GPU-resource infrastructure: the ResourceManager that
allocates raw Vulkan resources, the DescriptorSet that manages the bindless
descriptor table, and the PipelineManager that compiles/caches pipelines.

These three objects are intertwined: DescriptorSet borrows ResourceManager,
PipelineManager is built against DescriptorSet layouts, and the Swapchain
(on GpuContext) needs both ResourceManager and DescriptorSet for its
color/depth attachments. Initialization is split into two steps because
PipelineManager needs the Swapchain to exist, while Swapchain needs
ResourceManager + DescriptorSet:

    bindless.initResources(gpu);                       // RM + DS
    gpu.initSwapchain(*bindless.resourceManager,
                      *bindless.descriptorSet);        // swapchain object
    bindless.initPipelineManager(gpu);                 // PM (needs swapchain)
    // ... app-level buffer/pipeline setup on bindless.descriptorSet ...
    bindless.descriptorSet->createDescriptorSet();
    gpu.createSwapchainAndSync();                       // swap images + sync

Subsystems hold a BindlessSystem& (non-owning). App owns the BindlessSystem.
*/
class BindlessSystem {
  public:
    BindlessSystem() = default;
    ~BindlessSystem() = default;
    BindlessSystem(const BindlessSystem&) = delete;
    BindlessSystem& operator=(const BindlessSystem&) = delete;

    std::unique_ptr<ResourceManager> resourceManager;
    std::unique_ptr<DescriptorSet>   descriptorSet;
    std::unique_ptr<PipelineManager> pipelineManager;

    void initResources(GpuContext& gpu) {
        resourceManager = std::make_unique<ResourceManager>(gpu.getDevice(), gpu.getCommandPool());
        descriptorSet   = std::make_unique<DescriptorSet>(gpu.getDevice(), *resourceManager, &gpu.getCommandPool());
    }

    void initPipelineManager(GpuContext& gpu) {
        pipelineManager = std::make_unique<PipelineManager>(gpu.getDevice(), gpu.getSwapchain(), gpu.getMsaaSamples());
    }
};
