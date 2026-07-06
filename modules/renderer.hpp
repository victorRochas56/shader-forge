#pragma once
#ifndef VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1 // for raii
#endif
#ifndef VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_NO_CONSTRUCTORS 1 // for structs constructors
#endif

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <unordered_set>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#if TRACY_ENABLE
#include <tracy/TracyVulkan.hpp> // GPU (Vulkan) profiling zones; TracyVkCtx is void* when disabled
#endif

#define GLM_FORCE_RADIANS
#define GLM_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#include "constants.hpp"
#include "structs.hpp"
#include "scene_elements.hpp"
#include "asset_manager.hpp"
#include "scene_graph.hpp"
#include "gpu_context.hpp"
#include "bindless_system.hpp"
#include "render_buffers.hpp"
#include "scene.hpp"
#include "render_pass.hpp"

// Forward declarations — full definitions only needed in renderer.cpp
struct GLFWwindow;
struct PipelineBase;

/*
main rendering engine holds the state of the scene, nodes, meshes, lights, materials etc...
handles vulkan initialization and the main render loop
*/

class Renderer {
#pragma region VARS
  public:
    GpuContext&                         gpu;
    Scene&                              scene;
    RenderBuffers                       buffers;
    RenderFeatures                      features;
    float                               cullFovScale = 1.0f;
    uint32_t                            culledCount = 0;
    uint32_t                            defaultAlbedoIndex;

  private:
    bool                                framebufferResized = false;
    BindlessSystem&                     bindless;
    RenderPassResources                 passResources;

    // Tracy GPU profiling context (per graphics queue). void* / no-op when TRACY_ENABLE is off.
    #if TRACY_ENABLE
    TracyVkCtx                          tracyCtx = nullptr;
    #endif

    std::map<PassId,std::unique_ptr<RenderPass>>   passes;
    //pipelines
    uint32_t                            skyboxPipelineIndex;
    uint32_t                            shadowPipelineIndex;
    uint32_t                            litPipelineIndex;
    uint32_t                            gizmoPipelineIndex;
    uint32_t                            imageViewPipelineIndex;
    uint32_t                            depthPipelineIndex;
    uint32_t                            billboardPipelineIndex;
    uint32_t                            tonemapPipelineIndex = 0xFFFFFFFF;

    // Auto-exposure (metering + eye adaptation)
    uint32_t                            lumExtractPipelineIndex = 0xFFFFFFFF;
    uint32_t                            exposureAdaptPipelineIndex = 0xFFFFFFFF;
    uint32_t                            avgLumTextureIndex = 0xFFFFFFFF;   // mipped log-luminance
    vk::raii::ImageView                 avgLumMip0View = nullptr;          // mip-0 render view
    uint32_t                            adaptedLumIndex[2] = {0xFFFFFFFF, 0xFFFFFFFF}; // ping-pong 1x1
    uint32_t                            adaptFlip = 0;
    bool                                adaptInitialized = false;
    float                               autoExposurePrevTime = 0.0f;

    //defaults
    uint32_t                            shadowSamplerIndex;
    uint32_t                            defaultNormalIndex;

    uint32_t                            vertexBufferIndex;
    uint32_t                            indexBufferIndex;
    uint32_t                            positionBufferIndex;
    uint32_t                            billboardBufferIndex;
    uint32_t                            shadowInstanceDataBufferIndex;
    uint32_t                            shadowMeshDrawDataBufferIndex;
    uint32_t                            litInstanceDataBufferIndex;
    uint32_t                            litMeshDrawDataBufferIndex;
    uint32_t                            litPassDataBufferIndex;

    // Persistent buffers for indirect drawing
    vk::raii::Buffer                    indirectDrawBuffer = nullptr;      // shadow pass
    vk::raii::DeviceMemory              indirectDrawBufferMemory = nullptr;
    void*                               indirectDrawBufferMapped = nullptr;
    vk::raii::Buffer                    litIndirectDrawBuffer = nullptr;   // lit geometry pass
    vk::raii::DeviceMemory              litIndirectDrawBufferMemory = nullptr;
    void*                               litIndirectDrawBufferMapped = nullptr;

    // Reusable staging vectors (cleared per draw recording)
    std::vector<DrawIndexedIndirectCommand> indirectCommands;
    std::vector<ShadowMeshDrawData>         shadowMeshDrawDataList;
    std::vector<ShadowInstanceData>         shadowInstanceDataList;
    std::vector<LitMeshDrawData>            litMeshDrawDataList;
    std::vector<LitInstanceData>            litInstanceDataList;

    // Roughness-metallic MRT from lit pass
    Image                               roughnessMetal;

    // World-space normals MRT from lit pass
    uint32_t                            normalMipLevels = 1;
    Image                               normalMSAA;

    Image                               motionVectors;

    // Hi-Z (hierarchical depth pyramid)
    uint32_t                            hiZPipelineIndex = 0xFFFFFFFF;
    std::vector<vk::raii::ImageView>    hiZMipViews;
    std::vector<vk::raii::ImageView>    colorResolveMipViews;

    // SDF rendering
    uint32_t                            sdfPipelineIndex = 0xFFFFFFFF;
    uint32_t                            sdfTextureIndex = 0xFFFFFFFF;
    uint32_t                            sdfPassDataBufferIndex = 0xFFFFFFFF;
    uint32_t                            sdfApplyPipelineIndex = 0xFFFFFFFF;

    // Volume rendering

  private:
    // Temporary texture for gaussian blur (mipmapped for per-mip blur passes)
    std::vector<vk::raii::ImageView>    tempBlurMipViews;
#pragma endregion



  public:

#pragma region INIT
    Renderer(GpuContext& gpu, BindlessSystem& bindless, Scene& scene);
    ~Renderer();

    void initVulkan(uint32_t startWidth, uint32_t startHeight);
#pragma endregion




#pragma region DRAWFRAME
    void drawFrame();
#pragma endregion




#pragma region GET/SET
    uint32_t getModelMatrixBufferIndex();
    uint32_t getLightBufferIndex();
    uint32_t getVolumeBufferIndex();

    void clearLights();
    void clearVolumes();

    void toggleVsync();

    const std::vector<DrawIndexedIndirectCommand>& getIndirectCommands() const { return indirectCommands; }
#pragma endregion

    void handleSwapchainResize();

    void blurAttachment(vk::raii::CommandBuffer& cmd, uint32_t sourceTextureIndex, uint32_t tempTextureIndex,
                        uint32_t width, uint32_t height, float blurRadius = 1.0f, uint32_t samplerIndex = 0);

  private:
#pragma region CREATE RESOURCES
    void createShadowAtlas(uint32_t resolution);
    void createRoughnessMetalResources(uint32_t width, uint32_t height);
    void createNormalResources(uint32_t width, uint32_t height);
    void createColorResolveResources(uint32_t width, uint32_t height);
    void createMotionVectorResources(uint32_t width, uint32_t height);
    void createHiZResources(uint32_t width, uint32_t height);
    void createSDFResources(uint32_t width, uint32_t height);
#pragma endregion




#pragma region RENDERING
    void recordCommandBuffer(uint32_t imageIndex);

    template <typename PerMeshFn, typename PerInstanceFn>
    void buildGeometryDrawCommands(const std::array<Plane, 6>& frustumPlanes, bool doCulling, PerMeshFn&& perMeshFn, PerInstanceFn&& perInstanceFn,
                                   const std::function<bool(const Node&)>& nodeFilter = {});

    void recordHiZPass(vk::raii::CommandBuffer& cmd);
    void recordGeometryPass(vk::raii::CommandBuffer& cmd, uint32_t imageIndex);
    void recordBillboardBlendPass(vk::raii::CommandBuffer& cmd, uint32_t imageIndex);
    void recordResolveToCompositeCopy(vk::raii::CommandBuffer& cmd);
    uint32_t recordAutoExposure(vk::raii::CommandBuffer& cmd); // returns adapted-luminance texture index
    void recordTonemapPass(vk::raii::CommandBuffer& cmd, uint32_t imageIndex);
    void recordOverlayPass(vk::raii::CommandBuffer& cmd, uint32_t imageIndex);
    void recordImageVisPass(vk::raii::CommandBuffer& cmd, uint32_t imageIndex);
    void recordShadowPass(vk::raii::CommandBuffer& cmd, Light& light, uint32_t shadowSlot);
    void recordSDFPass(vk::raii::CommandBuffer& cmd, uint32_t imageIndex);

    void createOrResizeRenderTarget(uint32_t& index, uint32_t width, uint32_t height,
                                     vk::Format format, const char* debugName,
                                     vk::ImageUsageFlags extraUsage = {});
    void createOrResizeMSAATarget(Image& target, uint32_t width, uint32_t height, vk::Format format);

    void setFullscreenViewport(vk::raii::CommandBuffer& cmd, vk::Extent2D extent);
    void bindPipeline(vk::raii::CommandBuffer& cmd, PipelineBase& pipeline);

    template <typename T>
    void drawFullscreenPass(vk::raii::CommandBuffer& cmd, PipelineBase& pipeline,
                            vk::ImageView targetView, vk::Extent2D extent, const T& pushConstants,
                            vk::AttachmentLoadOp loadOp = vk::AttachmentLoadOp::eClear,
                            std::array<float, 4> clearColor = {0.0f, 0.0f, 0.0f, 0.0f});
#pragma endregion
};
