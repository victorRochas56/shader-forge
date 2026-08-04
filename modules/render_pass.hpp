#pragma once
#include <array>
#include <cstdint>
#include <vulkan/vulkan_raii.hpp>
#include "pipelines.hpp"

class GpuContext;
class BindlessSystem;
class Scene;
struct RenderFeatures;
struct RenderBuffers;

// Bindless indices (and other handles) that are created once by the app and
// shared across passes. Passes read from this; they do not own the resources.
struct RenderPassResources {
    uint32_t blurPipelineIndex = 0xFFFFFFFF;

    // 1x1 fallback material textures; instance data substitutes these for absent maps so the
    // lit shader can sample all four material textures unconditionally.
    uint32_t defaultAlbedoIndex = 0xFFFFFFFF;
    uint32_t defaultRoughnessIndex = 0xFFFFFFFF;
    uint32_t defaultMetallicIndex = 0xFFFFFFFF;
    uint32_t defaultNormalIndex = 0xFFFFFFFF;

    uint32_t depthSamplerIndex = 0xFFFFFFFF;
    uint32_t defaultSamplerIndex = 0xFFFFFFFF;
    // Linear clamp-to-edge, for screen-space textures: defaultSampler is eRepeat, which wraps
    // filter footprints at screen borders (visible on high mips of the color chain).
    uint32_t screenSamplerIndex = 0xFFFFFFFF;
    uint32_t shadowSamplerIndex = 0xFFFFFFFF;
    // Trilinear, clamped, anisotropy off. For volume textures: defaultSampler is eRepeat, so a uvw
    // that strays outside [0,1] wraps to the opposite face of the grid instead of clamping.
    uint32_t volumeSamplerIndex = 0xFFFFFFFF;

    // Global geometry buffers, for passes that draw meshes (thumbnails).
    uint32_t vertexBufferIndex = 0xFFFFFFFF;
    uint32_t indexBufferIndex = 0xFFFFFFFF;

    // gbuffer
    uint32_t colorResolveTextureIndex = 0xFFFFFFFF;
    // HDR composite target: post passes blend into this, then it's tonemapped to the swapchain.
    uint32_t compositeColorTextureIndex = 0xFFFFFFFF;
    uint32_t roughnessMetalTextureIndex = 0xFFFFFFFF;
    uint32_t normalTextureIndex = 0xFFFFFFFF;
    uint32_t motionVectorTextureIndex = 0xFFFFFFFF;
    RenderBuffers& buffers;
    // hiZ
    uint32_t hiZTextureIndex = 0xFFFFFFFF;
    uint32_t hiZMipLevels = 0;

    // blurred color mip chain (for cone tracing)
    uint32_t tempBlurTextureIndex = 0xFFFFFFFF;
    uint32_t fullscreenMipLevels = 1;
    std::vector<vk::raii::ImageView>* colorResolveMipViews = nullptr;
    std::vector<vk::raii::ImageView>* tempBlurMipViews = nullptr;
};

enum PassId {
    SSAO,
    SSR,
    VOLUMETRICS,
    PARTICLES,
    THUMBNAIL,
    MATERIAL_THUMBNAIL,
    VOXELIZATION,
};

class RenderPass {
public:
    virtual ~RenderPass() = default;
    void resize(uint32_t& index, uint32_t width, uint32_t height, vk::Format format, const char* debugName, vk::ImageUsageFlags extraUsage = {});
    // Allocates a 3D storage+sampled volume (froxel grid) and registers two bindless slots for one
    // view: textureIndex (sampled read) + storageIndex (compute write). Screen-independent, so unlike
    // resize() it's typically called once. Device must be idle. See FROXEL_VOLUMETRICS_PLAN.md prereq 1.
    void resize3DStorageImage(uint32_t& textureIndex, uint32_t& storageIndex, uint32_t width, uint32_t height, uint32_t depth,
                              vk::Format format, const char* debugName);
    // 2D counterpart of resize3DStorageImage: storage+sampled, two bindless slots over one view.
    // extraUsage adds flags the image needs beyond that — e.g. eColorAttachment when the same image is
    // also rasterized into. Device must be idle.
    void resize2DStorageImage(uint32_t& textureIndex, uint32_t& storageIndex, uint32_t width, uint32_t height,
                              vk::Format format, const char* debugName, vk::ImageUsageFlags extraUsage = {});
    virtual void init(uint32_t width, uint32_t height) = 0;
    virtual void record(vk::raii::CommandBuffer& cmd, uint32_t imageIndex) = 0;

protected:
    RenderPass(GpuContext& gpu, BindlessSystem& bindless, Scene& scene, RenderFeatures& features, RenderPassResources& shared)
        : gpu(gpu), bindless(bindless), scene(scene), features(features), shared(shared) {}

    GpuContext&          gpu;
    BindlessSystem&      bindless;
    Scene&               scene;
    RenderFeatures&      features;
    RenderPassResources& shared;
};

void setFullscreenViewport(vk::raii::CommandBuffer& cmd, vk::Extent2D extent);
void bindPipeline(vk::raii::CommandBuffer& cmd, PipelineBase& pipeline);

template <typename T>
void drawFullscreenPass(vk::raii::CommandBuffer& cmd, PipelineBase& pipeline, vk::ImageView targetView, vk::Extent2D extent,
                        const T& pushConstants, vk::AttachmentLoadOp loadOp = vk::AttachmentLoadOp::eClear,
                        std::array<float, 4> clearColor = {0.0f, 0.0f, 0.0f, 0.0f}) {
    vk::ClearValue clear{.color = vk::ClearColorValue(clearColor)};
    vk::RenderingAttachmentInfo colorAttachment{.imageView = targetView,
                                                .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                .loadOp = loadOp,
                                                .storeOp = vk::AttachmentStoreOp::eStore,
                                                .clearValue = clear};
    vk::RenderingInfo renderInfo{.renderArea = {{0, 0}, extent}, .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &colorAttachment};

    cmd.beginRendering(renderInfo);
    setFullscreenViewport(cmd, extent);
    bindPipeline(cmd, pipeline);
    cmd.pushConstants<T>(pipeline.layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pushConstants);
    cmd.draw(3, 1, 0, 0);
    cmd.endRendering();
}

void blurAttachment(BindlessSystem& bindless, vk::raii::CommandBuffer& cmd, uint32_t blurPipelineIndex,
                    uint32_t sourceTextureIndex, uint32_t tempTextureIndex, uint32_t width, uint32_t height,
                    float blurRadius = 1.0f, uint32_t samplerIndex = 0);
