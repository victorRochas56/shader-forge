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
BindlessSystem bundles the GPU-resource infrastructure: the resource::Context that
the functional resource module allocates raw Vulkan resources through, the
DescriptorSet that manages the bindless descriptor table, and the PipelineManager
that compiles/caches pipelines.
*/
class BindlessSystem {
  public:
    BindlessSystem() = default;
    ~BindlessSystem() = default;
    BindlessSystem(const BindlessSystem&) = delete;
    BindlessSystem& operator=(const BindlessSystem&) = delete;

    std::unique_ptr<resource::Context> resourceCtx;
    std::unique_ptr<DescriptorSet>     descriptorSet;
    std::unique_ptr<PipelineManager>   pipelineManager;

    void initResources(GpuContext& gpu) {
        resourceCtx   = std::make_unique<resource::Context>(gpu.getDevice(), gpu.getCommandPool());
        descriptorSet = std::make_unique<DescriptorSet>(gpu.getDevice(), &gpu.getCommandPool());
    }

    void initPipelineManager(GpuContext& gpu) {
        pipelineManager = std::make_unique<PipelineManager>(gpu.getDevice(), gpu.getSwapchain(), gpu.getMsaaSamples());
    }
};
