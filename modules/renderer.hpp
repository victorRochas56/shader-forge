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
#include "scene.hpp"

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
    BindlessSystem&                     bindless;
    Scene&                              scene;
    RenderFeatures                      features;
    float                               cullFovScale = 1.0f;
    uint32_t                            culledCount = 0;

  private:
    bool                                framebufferResized = false;

    //pipelines
    uint32_t                            skyboxPipelineIndex;
    uint32_t                            shadowPipelineIndex;
    uint32_t                            litPipelineIndex;
    uint32_t                            gizmoPipelineIndex;
    uint32_t                            imageViewPipelineIndex;
    uint32_t                            blurPipelineIndex;
    uint32_t                            depthPipelineIndex;

    //defaults
    uint32_t                            defaultSamplerIndex;
    uint32_t                            depthSamplerIndex;
    uint32_t                            shadowSamplerIndex;
    uint32_t                            defaultNormalIndex;

    struct RenderEntry {
        uint32_t nodeIndex;
        uint32_t materialIndex;       // index into materials vector
        uint32_t shaderPipelineIndex;
    };
    struct ShaderDrawRange {
        uint32_t pipelineIndex;
        uint32_t firstCommand;
        uint32_t commandCount;
    };
    std::vector<RenderEntry>            renderEntries;
    bool                                renderListDirty = false;
    std::vector<ShaderDrawRange>        shaderDrawRanges;

    uint32_t                            vertexBufferIndex;
    uint32_t                            indexBufferIndex;
    uint32_t                            modelMatrixBufferIndex;
    uint32_t                            lightBufferIndex;
    uint32_t                            shadowDrawDataBufferIndex;
    uint32_t                            litDrawDataBufferIndex;
    uint32_t                            litPassDataBufferIndex;
    uint32_t                            ssrPassDataBufferIndex;

    // Persistent buffers for indirect drawing
    vk::raii::Buffer                    indirectDrawBuffer = nullptr;      // shadow pass
    vk::raii::DeviceMemory              indirectDrawBufferMemory = nullptr;
    void*                               indirectDrawBufferMapped = nullptr;
    vk::raii::Buffer                    litIndirectDrawBuffer = nullptr;   // lit geometry pass
    vk::raii::DeviceMemory              litIndirectDrawBufferMemory = nullptr;
    void*                               litIndirectDrawBufferMapped = nullptr;

    // Reusable staging vectors (cleared per draw recording)
    std::vector<DrawIndexedIndirectCommand> indirectCommands;
    std::vector<ShadowDrawData>             drawDataList;
    std::vector<LitDrawData>                litDrawDataList;

    //SSAO
    uint32_t                            ssaoTextureIndex = 0xFFFFFFFF;
    uint32_t                            ssaoBlurTextureIndex = 0xFFFFFFFF;
    uint32_t                            ssaoNoiseTextureIndex = 0xFFFFFFFF;
    uint32_t                            ssaoNoiseSamplerIndex = 0xFFFFFFFF;
    uint32_t                            ssaoPipelineIndex = 0xFFFFFFFF;
    uint32_t                            ssaoApplyPipelineIndex = 0xFFFFFFFF;

    //number of mips for a full resolution fullscreen image
    uint32_t                            fullscreenMipLevels = 1;

    // Roughness-metallic MRT from lit pass
    uint32_t                            roughnessMetalTextureIndex = 0xFFFFFFFF;
    Image                               roughnessMetal;

    // World-space normals MRT from lit pass
    uint32_t                            normalTextureIndex = 0xFFFFFFFF;
    uint32_t                            normalMipLevels = 1;
    Image                               normalMSAA;

    uint32_t                            motionVectorTextureIndex = 0xFFFFFFFF;
    Image                               motionVectors;

    //SSR
    uint32_t                            ssrCurrentTextureIndex = 0xFFFFFFFF;
    uint32_t                            ssrHistoryTextureIndices[2] = {0xFFFFFFFF, 0xFFFFFFFF};
    uint32_t                            ssrHistoryFlip = 0;
    bool                                ssrHistoryInvalid = true;
    uint32_t                            ssrPipelineIndex = 0xFFFFFFFF;
    uint32_t                            ssrAccumulatePipelineIndex = 0xFFFFFFFF;
    uint32_t                            ssrApplyPipelineIndex = 0xFFFFFFFF;
    uint32_t                            colorResolveTextureIndex = 0xFFFFFFFF;
    std::vector<vk::raii::ImageView>    colorResolveMipViews;


    // Hi-Z (hierarchical depth pyramid)
    uint32_t                            hiZTextureIndex = 0xFFFFFFFF;
    uint32_t                            hiZMipLevels = 0;
    uint32_t                            hiZPipelineIndex = 0xFFFFFFFF;
    std::vector<vk::raii::ImageView>    hiZMipViews;

    // SDF rendering
    uint32_t                            sdfPipelineIndex = 0xFFFFFFFF;
    uint32_t                            sdfTextureIndex = 0xFFFFFFFF;
    uint32_t                            sdfPassDataBufferIndex = 0xFFFFFFFF;
    uint32_t                            sdfApplyPipelineIndex = 0xFFFFFFFF;

    // Volume rendering
    uint32_t                            volPipelineIndex = 0xFFFFFFFF;
    uint32_t                            volTextureIndex = 0xFFFFFFFF;
    uint32_t                            volPassDataBufferIndex = 0xFFFFFFFF;
    uint32_t                            volApplyPipelineIndex = 0xFFFFFFFF;

  private:
    // Temporary texture for gaussian blur (mipmapped for per-mip blur passes)
    uint32_t                            tempBlurTextureIndex = 0xFFFFFFFF;
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
    uint32_t getShadowDrawDataBufferIndex();

    void addMeshToShader(uint32_t nodeIndex, Shader shader, Material material);
    void removeMeshFromShader(uint32_t nodeIndex, Shader shader, Material material);
    void removeNodeFromRenderList(uint32_t nodeIndex);

    void clearRenderList();
    void clearLights();

    void toggleVsync();

    const std::vector<DrawIndexedIndirectCommand>& getIndirectCommands() const { return indirectCommands; }
#pragma endregion

    void handleSwapchainResize();

    void blurAttachment(vk::raii::CommandBuffer& cmd, uint32_t sourceTextureIndex, uint32_t tempTextureIndex,
                        uint32_t width, uint32_t height, float blurRadius = 1.0f, uint32_t samplerIndex = 0);

  private:
#pragma region CREATE RESOURCES
    void createShadowAtlas(uint32_t resolution);
    void createSSAOResources(uint32_t width, uint32_t height);
    void createRoughnessMetalResources(uint32_t width, uint32_t height);
    void createNormalResources(uint32_t width, uint32_t height);
    void createColorResolveResources(uint32_t width, uint32_t height);
    void createMotionVectorResources(uint32_t width, uint32_t height);
    void createSSRResources(uint32_t width, uint32_t height);
    void createHiZResources(uint32_t width, uint32_t height);
    void createSDFResources(uint32_t width, uint32_t height);
    void createVolumetricResources(uint32_t width, uint32_t height);
#pragma endregion




#pragma region RENDERING
    void recordCommandBuffer(uint32_t imageIndex);

    template <typename PerMeshFn>
    void buildGeometryDrawCommands(const std::array<Plane, 6>& frustumPlanes, bool doCulling, PerMeshFn&& perMeshFn,
                                   const std::function<bool(const Node&)>& nodeFilter = {});

    void recordHiZPass(vk::raii::CommandBuffer& cmd);
    void recordGeometryPass(vk::raii::CommandBuffer& cmd, uint32_t imageIndex);
    void recordOverlayPass(vk::raii::CommandBuffer& cmd, uint32_t imageIndex);
    void recordSSAOPass(vk::raii::CommandBuffer& cmd, uint32_t imageIndex);
    void recordSSRPass(vk::raii::CommandBuffer& cmd, uint32_t imageIndex);
    void recordImageVisPass(vk::raii::CommandBuffer& cmd, uint32_t imageIndex);
    void recordShadowPass(vk::raii::CommandBuffer& cmd, Light& light, uint32_t shadowSlot);
    void recordVolumetricsPass(vk::raii::CommandBuffer& cmd, uint32_t imageIndex);
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
