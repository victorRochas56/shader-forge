#pragma once
#include <vulkan/vulkan_raii.hpp>

class GpuContext;
class BindlessSystem;
class Scene;
struct RenderFeatures;

class RenderPass {
public:
    virtual ~RenderPass() = default;
    virtual void init(uint32_t width, uint32_t height) = 0;
    virtual void resize(uint32_t width, uint32_t height) = 0;
    virtual void record(vk::raii::CommandBuffer& cmd, uint32_t imageIndex) = 0;

protected:
    RenderPass(GpuContext& gpu, BindlessSystem& bindless, Scene& scene, RenderFeatures& features)
        : gpu(gpu), bindless(bindless), scene(scene), features(features) {}

    GpuContext&     gpu;
    BindlessSystem& bindless;
    Scene&          scene;
    RenderFeatures& features;
};