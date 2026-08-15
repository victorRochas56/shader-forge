#include "renderer.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>

#include <stb_image.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
 

#include "gizmo.hpp"
#include "GUI.h"
#include "node_ops.hpp"
#include "pipelines.hpp"
#include "profiling.hpp"
#include "swapchain.hpp"

#include "ssao_pass.hpp"
#include "ssr_pass.hpp"
#include "volumetrics_pass.hpp"
#include "particle_pass.hpp"
#include "thumbnail_pass.hpp"
#include "material_thumbnail_pass.hpp"
#include "voxelization_pass.hpp"

// TODO clustered lights? (forward +)
//      pass the N nearest lights to the lit shader
// TODO spot and area lights
// TODO multithread command buffer recording

/**
 * TODO : make adding new shaders to lit pipeline easy and declarative 
 * avoid having to write much if any renderer code for simple new shaders ideally (ie. water shader, toon, etc.)
 * 
 * declare resources needed & associate indices, expose shader on material options, passData / push constants
 * along with this i can do hot shader reloading too
 */

glm::mat4 calculateLightSpaceMatrix(Light& light, Camera& camera);

void calculatePointLightFaceMatrices(Light& light, const glm::vec3& lightPos);

void calculateCascadedLightSpaceMatrices(Light& light, Camera& camera, Renderer* renderer);

Renderer::Renderer(GpuContext& gpu, BindlessSystem& bindless, Scene& scene, GUI& gui)
    : scene(scene), gpu(gpu), bindless(bindless), gui(gui), passResources{.buffers = buffers} {}
Renderer::~Renderer() {
    // Device is idle here (App waits before teardown). No-op when Tracy is disabled.
    #if TRACY_ENABLE
    TracyVkDestroy(tracyCtx);
    #endif
}

/////=================================================INIT=================================================/////

void Renderer::initVulkan(uint32_t startWidth, uint32_t startHeight) {
    // GpuContext::initCore() must have been called by App already.
    bindless.initResources(gpu);
    gpu.initSwapchain(*bindless.resourceCtx, *bindless.descriptorSet);
    bindless.initPipelineManager(gpu);

    // Tracy GPU profiling context: calibrates timestamps via a transient submit on the graphics
    // queue (safe here — the queue is idle during init). No-op when TRACY_ENABLE is off.
    #if TRACY_ENABLE
    tracyCtx = TracyVkContext(*gpu.getDevice().getPhysicalDevice(), *gpu.getDevice().getDevice(),
                              *gpu.getDevice().getGraphicsQueue(), *gpu.getCommandBuffer(0));
    #endif

    // initializing default camera
    scene.activeCamera = Camera{.position = glm::vec3(1, 1, 1),
                          .target = glm::vec3(0, 0, 0),
                          .fov = 45.0,
                          .aspectRatio = static_cast<float>(startWidth) / static_cast<float>(startHeight),
                          .nearPlane = 0.1,
                          .farPlane = 500.0};
    scene.activeCamera.calculateViewProjectionMatrix();

    /////=====================================DESCRIPTOR SET BUFFERS=================================================/////
    vertexBufferIndex   = bindless.descriptorSet->createVariableBuffer(256 * 1024 * 1024, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress, false, "Vertex");
    indexBufferIndex    = bindless.descriptorSet->createVariableBuffer(128 * 1024 * 1024, vk::BufferUsageFlagBits::eIndexBuffer, false, "Index");
    positionBufferIndex = bindless.descriptorSet->createVariableBuffer(128 * 1024 * 1024, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress, false, "Position");
    scene.assetManager.init(bindless.resourceCtx.get(), bindless.descriptorSet.get(), vertexBufferIndex, indexBufferIndex, positionBufferIndex);
    passResources.vertexBufferIndex = vertexBufferIndex;
    passResources.indexBufferIndex  = indexBufferIndex;

    billboardBufferIndex   = bindless.descriptorSet->createFixedBuffer<GPUBillboard>(MAX_FRAMES_IN_FLIGHT * MAX_FIXED_BUFFER, true, "Billboard");
    sdfPassDataBufferIndex = bindless.descriptorSet->createFixedBuffer<SDF>(MAX_FIXED_BUFFER, true, "SDF");


    // these buffers store the data once per frame in flight since they are usually accessed every frame by the CPU
    buffers.modelMatrixBufferIndex         = bindless.descriptorSet->createFixedBuffer<glm::mat4>(MAX_FRAMES_IN_FLIGHT * MAX_FIXED_BUFFER, true, "ModelMatrix");
    shadowInstanceDataBufferIndex          = bindless.descriptorSet->createFixedBuffer<ShadowInstanceData>(MAX_FRAMES_IN_FLIGHT * MAX_SHADOW_CASTERS * MAX_FIXED_BUFFER, true, "ShadowInstanceData");
    shadowMeshDrawDataBufferIndex          = bindless.descriptorSet->createFixedBuffer<ShadowMeshDrawData>(MAX_FRAMES_IN_FLIGHT * MAX_SHADOW_CASTERS * MAX_FIXED_BUFFER, true, "ShadowMeshDrawData");
    passResources.buffers.lightBufferIndex = bindless.descriptorSet->createFixedBuffer<GPULight>(MAX_LIGHTS * MAX_FRAMES_IN_FLIGHT, true, "Light");
    litPassDataBufferIndex                 = bindless.descriptorSet->createFixedBuffer<LitPassData>(MAX_FRAMES_IN_FLIGHT * MAX_FIXED_BUFFER, true, "LitPassData");
    litMeshDrawDataBufferIndex = bindless.descriptorSet->createFixedBuffer<LitMeshDrawData>(MAX_FRAMES_IN_FLIGHT * MAX_FIXED_BUFFER, true, "LitMeshDrawData");
    litInstanceDataBufferIndex = bindless.descriptorSet->createFixedBuffer<LitInstanceData>(MAX_FRAMES_IN_FLIGHT * MAX_FIXED_BUFFER, true, "LitInstanceDrawData");
    // sets the frame offsets for each buffer
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        bindless.descriptorSet->setBufferFrameOffset(buffers.modelMatrixBufferIndex, i, MAX_FIXED_BUFFER * i);
        bindless.descriptorSet->setBufferFrameOffset(passResources.buffers.lightBufferIndex, i, MAX_LIGHTS * i);
        bindless.descriptorSet->setBufferFrameOffset(shadowInstanceDataBufferIndex, i, MAX_SHADOW_CASTERS * MAX_FIXED_BUFFER * i);
        bindless.descriptorSet->setBufferFrameOffset(shadowMeshDrawDataBufferIndex, i, MAX_SHADOW_CASTERS * MAX_FIXED_BUFFER * i);
        bindless.descriptorSet->setBufferFrameOffset(litPassDataBufferIndex,i, MAX_FIXED_BUFFER * i);
        bindless.descriptorSet->setBufferFrameOffset(billboardBufferIndex, i, MAX_FIXED_BUFFER * i);
        bindless.descriptorSet->setBufferFrameOffset(litMeshDrawDataBufferIndex, i, MAX_FIXED_BUFFER * i);
        bindless.descriptorSet->setBufferFrameOffset(litInstanceDataBufferIndex, i, MAX_FIXED_BUFFER * i);
    }
    
    // indirect draw buffers (separate for shadow and lit passes)
    // Shadow indirect buffer needs one slot per shadow-casting light per frame so multiple lights
    // recorded into the same command buffer don't overwrite each other's draw commands.
    std::tie(indirectDrawBuffer, indirectDrawBufferMemory, indirectDrawBufferMapped)          = resource::createIndirectDrawBuffer(*bindless.resourceCtx, MAX_SHADOW_CASTERS);
    std::tie(litIndirectDrawBuffer, litIndirectDrawBufferMemory, litIndirectDrawBufferMapped) = resource::createIndirectDrawBuffer(*bindless.resourceCtx);
    
    //init gizmos
    Gizmos::init(MAX_GIZMO_LINES, &*bindless.descriptorSet, sdfPassDataBufferIndex);

    // after having created all our desire buffers we can initialize the descriptor set
    bindless.descriptorSet->createDescriptorSet();

    gpu.createSwapchainAndSync();

    passResources.colorResolveMipViews = &colorResolveMipViews;
    passResources.tempBlurMipViews = &tempBlurMipViews;
    passResources.buffers = buffers;

    passes.emplace(PassId::SSAO,std::make_unique<SSAOPass>(gpu, bindless, scene, features, passResources));
    passes.emplace(PassId::SSR,std::make_unique<SSRPass>(gpu, bindless, scene, features, passResources));
    passes.emplace(PassId::PARTICLES,std::make_unique<ParticlePass>(gpu, bindless, scene, features, passResources));
    passes.emplace(PassId::VOLUMETRICS,std::make_unique<VolumetricsPass>(gpu, bindless, scene, features, passResources));
    passes.emplace(PassId::THUMBNAIL,std::make_unique<ThumbnailPass>(gpu, bindless, scene, features, passResources));
    passes.emplace(PassId::MATERIAL_THUMBNAIL,std::make_unique<MaterialThumbnailPass>(gpu, bindless, scene, features, passResources));

    passes.emplace(PassId::VOXELIZATION,std::make_unique<VoxelizationPass>(gpu, bindless, scene, features, passResources));

    createShadowAtlas(SHADOW_ATLAS_SIZE);
    createRoughnessMetalResources(startWidth, startHeight);
    createNormalResources(startWidth, startHeight);
    createMotionVectorResources(startWidth,startHeight);
    createColorResolveResources(startWidth, startHeight);
    createHiZResources(startWidth, startHeight);
    createSDFResources(startWidth,startHeight);

    for(auto& pass : passes) {
        pass.second->init(startWidth, startHeight);
    }

#if DEBUG == 1
    bindless.descriptorSet->debugDescriptorSet("after_createDescriptorSet");
#endif

    /////S=================================================DEFAULTS=================================================/////

    passResources.defaultSamplerIndex = bindless.descriptorSet->allocateSampler(vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear, vk::SamplerAddressMode::eRepeat, VK_TRUE, 16.0, VK_FALSE,
                                                         vk::CompareOp::eLessOrEqual, vk::BorderColor::eFloatOpaqueBlack);
    passResources.depthSamplerIndex = bindless.descriptorSet->allocateSampler(vk::Filter::eNearest, vk::SamplerMipmapMode::eNearest, vk::SamplerAddressMode::eClampToEdge, VK_FALSE, 16.0, VK_FALSE,
                                                       vk::CompareOp::eLessOrEqual, vk::BorderColor::eFloatOpaqueBlack);
    passResources.screenSamplerIndex = bindless.descriptorSet->allocateSampler(vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear, vk::SamplerAddressMode::eClampToEdge, VK_FALSE, 1.0, VK_FALSE,
                                                        vk::CompareOp::eLessOrEqual, vk::BorderColor::eFloatOpaqueBlack);
    // volumeSamplerIndex is allocated by VoxelizationPass::init (border-black, for cone tracing).
    shadowSamplerIndex = bindless.descriptorSet->allocateSampler(vk::Filter::eNearest,
                                                        vk::SamplerMipmapMode::eNearest,
                                                        vk::SamplerAddressMode::eClampToBorder,
                                                        VK_FALSE,
                                                        1.0f,
                                                        VK_FALSE,
                                                        vk::CompareOp::eLessOrEqual,
                                                        vk::BorderColor::eFloatOpaqueWhite
    );
    passResources.shadowSamplerIndex = shadowSamplerIndex;
    // Dedicated hardware-PCF comparison sampler — lives at its own binding.
    bindless.descriptorSet->allocateShadowCompareSampler(vk::Filter::eLinear,
                                                        vk::SamplerAddressMode::eClampToBorder,
                                                        vk::CompareOp::eLess,
                                                        vk::BorderColor::eFloatOpaqueWhite);
    // Default albedo (white)
    std::array<uint8_t, 4> whiteColor = {255, 255, 255, 255};
    auto [albedoImage, albedoMemory, albedoImageView] = resource::createTexture(*bindless.resourceCtx, whiteColor.data(), 1, 1, vk::Format::eR8G8B8A8Srgb);
    defaultAlbedoIndex = bindless.descriptorSet->allocateTexture(std::move(albedoImage), std::move(albedoMemory), std::move(albedoImageView));

    // Default normal (flat normal = 0.5, 0.5, 1.0 in RGB)
    std::array<uint8_t, 4> normalColor = {128, 128, 255, 255};
    auto [normalImage, normalMemory, normalImageView] = resource::createTexture(*bindless.resourceCtx, normalColor.data(), 1, 1, vk::Format::eR8G8B8A8Unorm);
    defaultNormalIndex = bindless.descriptorSet->allocateTexture(std::move(normalImage), std::move(normalMemory), std::move(normalImageView));

    // Default roughness = 0.5
    std::array<uint8_t, 4> roughnessColor = {128, 128, 128, 255};
    auto [roughnessImage, roughnessMemory, roughnessImageView] = resource::createTexture(*bindless.resourceCtx, roughnessColor.data(), 1, 1, vk::Format::eR8G8B8A8Unorm);
    defaultRoughnessIndex = bindless.descriptorSet->allocateTexture(std::move(roughnessImage), std::move(roughnessMemory), std::move(roughnessImageView));

    // Default metallic = 0.0
    std::array<uint8_t, 4> metallicColor = {0, 0, 0, 255};
    auto [metallicImage, metallicMemory, metallicImageView] = resource::createTexture(*bindless.resourceCtx, metallicColor.data(), 1, 1, vk::Format::eR8G8B8A8Unorm);
    defaultMetallicIndex = bindless.descriptorSet->allocateTexture(std::move(metallicImage), std::move(metallicMemory), std::move(metallicImageView));

    // Shared with passes (material thumbnails substitute the same defaults for absent maps).
    passResources.defaultAlbedoIndex    = defaultAlbedoIndex;
    passResources.defaultRoughnessIndex = defaultRoughnessIndex;
    passResources.defaultMetallicIndex  = defaultMetallicIndex;
    passResources.defaultNormalIndex    = defaultNormalIndex;

    // Per-frame lit frame uniforms (hot pass + light data; see GPULitFrameUniforms).
    litFrameStaging = std::make_unique<GPULitFrameUniforms>();
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        litFrameUniformsIndex[i] = bindless.descriptorSet->allocateUniformBuffer(sizeof(GPULitFrameUniforms), "LitFrameUniforms" + std::to_string(i));
    }

    /////S=================================================PIPELINES=================================================/////
    skyboxPipelineIndex =
        bindless.pipelineManager->createPipeline<SkyBoxPushConstants>(PipelineCategory::LIT_GEOMETRY, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone, vk::False,
                                                             vk::False, "shaders/skybox.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet(),
                                                             vk::Format::eUndefined);

    shadowPipelineIndex = bindless.pipelineManager->createPipeline<ShadowPushConstants>(PipelineCategory::SHADOW, vk::PrimitiveTopology::eTriangleList,
                                                                               vk::CullModeFlagBits::eNone, vk::True, vk::True, "shaders/shadow_geometry.spv",
                                                                               bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet(),
                                                                               vk::Format::eUndefined);
    // Blur is reused for many targets; callers re-bind the pipeline against the active attachment.
    // Most blur sources are color-resolve mips (HDR), so create with that.
    passResources.blurPipelineIndex =
        bindless.pipelineManager->createPipeline<BlurPushConstants>(PipelineCategory::POSTPROCESS, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone, vk::False,
                                                           vk::False, "shaders/blur.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet(),
                                                           gpu.getSwapchain().getHDRColorFormat());

    depthPipelineIndex =
        bindless.pipelineManager->createPipeline<LitPushConstants>(PipelineCategory::DEPTH_PREPASS, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eBack,vk::True,
                                                                        vk::True,"shaders/depth_prepass.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet(),
                                                                        vk::Format::eUndefined);

    // Creates a LIT_GEOMETRY pipeline for a lit / lit-derived shader and registers it so materials
    // can select it from the material editor. Returns the pipeline index.
    auto addLitShader = [&](const std::string& spvPath) {
        uint32_t idx = bindless.pipelineManager->createPipeline<LitPushConstants>(
            PipelineCategory::LIT_GEOMETRY, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eBack, vk::True,
            vk::True, spvPath, bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet(),
            vk::Format::eUndefined);
        scene.registerLitShader(Shader{.sourceFile = spvPath, .pipelineIndex = idx});
        return idx;
    };

    litPipelineIndex = addLitShader("shaders/lit.spv");

    // Specialized lit variants: compile-time giMode, no cascade-debug path. Dropping the unused
    // GI path from the fragment shader roughly halves its register pressure, which is what makes
    // the pass's memory-latency stalls hideable. Not registered as material shaders — the draw
    // loop remaps litPipelineIndex to one of these unless cascade debug is active.
    litPipelineGIConeIndex = bindless.pipelineManager->createPipeline<LitPushConstants>(
        PipelineCategory::LIT_GEOMETRY, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eBack, vk::True,
        vk::True, "shaders/lit.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet(),
        vk::Format::eUndefined, nullptr, "fragMainGICone");
    litPipelineGICubeIndex = bindless.pipelineManager->createPipeline<LitPushConstants>(
        PipelineCategory::LIT_GEOMETRY, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eBack, vk::True,
        vk::True, "shaders/lit.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet(),
        vk::Format::eUndefined, nullptr, "fragMainGICube");

    addLitShader("shaders/water.spv");

    // lit-derived variants are added declaratively here, e.g.:
    // addLitShader("shaders/lit_toon.spv");

    // Billboards alpha-blend into the HDR composite target after SSR/SSAO. Depth test is
    // performed in-shader by sampling the resolved depth.
    billboardPipelineIndex = bindless.pipelineManager->createPipeline<BillboardPushConstants>(PipelineCategory::POSTPROCESS_ALPHA_BLEND, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone, vk::False,
                                                                                            vk::False,"shaders/billboard.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet(),
                                                                                            gpu.getSwapchain().getHDRColorFormat());
    gizmoPipelineIndex =
        bindless.pipelineManager->createPipeline<LinePushConstants>(PipelineCategory::POSTPROCESS, vk::PrimitiveTopology::eLineList, vk::CullModeFlagBits::eNone, vk::False, vk::False,
                                                           "shaders/line.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet(),
                                                           gpu.getSwapchain().getSwapChainImageFormat());
    imageViewPipelineIndex =
        bindless.pipelineManager->createPipeline<ImageVisPushConstants>(PipelineCategory::POSTPROCESS, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone, vk::False,
                                                               vk::False, "shaders/image_view.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet(),
                                                               gpu.getSwapchain().getSwapChainImageFormat());
    // Tonemap: samples the HDR composite, writes the sRGB swapchain (hardware does sRGB encode).
    tonemapPipelineIndex =
        bindless.pipelineManager->createPipeline<TonemapPushConstants>(PipelineCategory::POSTPROCESS, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone, vk::False,
                                                               vk::False, "shaders/tonemap.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet(),
                                                               gpu.getSwapchain().getSwapChainImageFormat());
    // Auto-exposure: log-luminance extract + temporal adaptation (both write R16F).
    lumExtractPipelineIndex =
        bindless.pipelineManager->createPipeline<LumExtractPushConstants>(PipelineCategory::POSTPROCESS, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone, vk::False,
                                                               vk::False, "shaders/lum_extract.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet(),
                                                               vk::Format::eR16Sfloat);
    exposureAdaptPipelineIndex =
        bindless.pipelineManager->createPipeline<ExposureAdaptPushConstants>(PipelineCategory::POSTPROCESS, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone, vk::False,
                                                               vk::False, "shaders/exposure_adapt.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet(),
                                                               vk::Format::eR16Sfloat);

    /**
     * post process pipelines added declaratively are added here?
     */

    // default litshader / material
    scene.fallbackLitShader = Shader{.sourceFile = "shaders/lit.spv", .pipelineIndex = litPipelineIndex};
    MaterialFlags defaultTexMask = MaterialFlags::MAT_NONE; // see the material struct definition
    // texMask |= (1U << 0);
    // texMask |= (1U << 1);
    // texMask |= (1U << 3);
    Material defaultMaterial = Material{.shaderSource = scene.fallbackLitShader,
                                        .flags = defaultTexMask,
                                        .color = glm::vec4(0.5, 0.5, 0.5, 1),
                                        .albedoTextureIndex = defaultAlbedoIndex,
                                        .metallic = 0.0,
                                        .metallicTextureIndex = defaultMetallicIndex,
                                        .roughness = 0.5,
                                        .roughnessTextureIndex = defaultRoughnessIndex,
                                        .normalTextureIndex = defaultNormalIndex};
    scene.fallbackDefaultMaterialIndex = scene.addMaterial(defaultMaterial);

#if DEBUG == 1
    bindless.descriptorSet->debugDescriptorSet("after_pipeline_creation");
#endif

    // create the root node - end of initialization. SceneGraph captures
    // const RenderBuffers& — nodeTextureIndex is filled in later by App, the
    // ref will pick that up automatically.
    scene.sceneGraph.init(scene, bindless, buffers);
    bindless.descriptorSet->allocateFixedBuffer(litPassDataBufferIndex, LitPassData{.samplerIndex = passResources.defaultSamplerIndex,
                                                                           .lightCount = 0,
                                                                           .shadowSamplerIndex = shadowSamplerIndex,
                                                                           .shadowAtlasIndex = scene.shadowAtlas.textureIndex,
                                                                           .cameraPosition = scene.activeCamera.position,
                                                                           .voxelTextureIndex = static_cast<VoxelizationPass*>(passes.at(PassId::VOXELIZATION).get())->getVolumeTextureIndex(),
                                                                           .cameraForward = glm::vec3(1, 0, 0),
                                                                           .voxelSamplerIndex = passResources.volumeSamplerIndex,
                                                                           .viewProjection = scene.activeCamera.viewProjection,
                                                                           .prevViewProjection = scene.activeCamera.viewProjection,
                                                                           .voxelViewProjection = VoxelizationPass::gridViewProjection(scene.activeCamera.position),
                                                                           .voxelResolution = VoxelizationPass::VOXEL_RESOLUTION,
                                                                           .voxelWorldExtent = VoxelizationPass::VOXEL_WORLD_EXTENT,
                                                                           .giIrradianceIndices = static_cast<VoxelizationPass*>(passes.at(PassId::VOXELIZATION).get())->getIrradianceTextureIndices()});
}

/////=================================================DRAW FRAME=================================================/////

void Renderer::drawFrame() {
    #if TRACY_ENABLE
    ZoneScoped; // Tracy CPU zone for the whole frame
    #endif
    // Slow-frame breakdown: fenceWait = GPU execution of a previous frame; record = command recording
    // (+ command-time validation); submit/present = queue submit (+ sync validation's analysis).
    using frameClk = std::chrono::steady_clock;
    auto phaseMs = [](frameClk::time_point a, frameClk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    frameClk::time_point tStart = frameClk::now();

    tracing::startTrace("draw frame");
    tracing::startTrace("wait for fences");
    gpu.getDevice().getDevice().waitForFences(*gpu.getInFlightFence(gpu.currentFrame), vk::True, UINT64_MAX);
    tracing::endTrace("wait for fences");
    frameClk::time_point tFence = frameClk::now();

    // This slot's prior submission has retired — safe to destroy any textures it referenced.
    bindless.descriptorSet->processDeferredTextureFrees();
    //TODO make all full screen passes have a resolution scale that can be set dirty when changed/ needs to recreate
    if (features.ssr.resolutionDirty) {
        features.ssr.resolutionDirty = false;
        gpu.getDevice().getDevice().waitIdle();
        int w = 0, h = 0;
        glfwGetFramebufferSize(gpu.getWindow(), &w, &h);
        if (w > 0 && h > 0) {
            // init() reads features.ssr.resolutionScale and recreates the SSR textures at the new size.
            passes.at(PassId::SSR)->init(static_cast<uint32_t>(w), static_cast<uint32_t>(h));
        }
    }

    tracing::startTrace("acquire next image");
    auto [result, imageIndex] = gpu.getSwapchain().getSwapChain().acquireNextImage(UINT64_MAX, *gpu.getPresentCompleteSemaphore(gpu.currentFrame), nullptr);
    tracing::endTrace("acquire next image");

    if (result == vk::Result::eErrorOutOfDateKHR) {
        gpu.recreateSwapchain();
        handleSwapchainResize();
        tracing::endTrace("draw frame"); // early return must not leak the zone — scope drifts otherwise
        return;
    }
    if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
        throw std::runtime_error("failed to acquire swap chain image!");
    }
    tracing::startTrace("wait image in flight");
    if (gpu.getImagesInFlight()[imageIndex] != VK_NULL_HANDLE) {
        vk::Result waitResult = gpu.getDevice().getDevice().waitForFences(gpu.getImagesInFlight()[imageIndex], vk::True, UINT64_MAX);
        if (waitResult != vk::Result::eSuccess) {
            throw std::runtime_error("failed to wait for image fence!");
        }
    }
    tracing::endTrace("wait image in flight");
    frameClk::time_point tAcquire = frameClk::now();

    tracing::startTrace("reset fences");
    gpu.getImagesInFlight()[imageIndex] = *gpu.getInFlightFence(gpu.currentFrame);
    gpu.getDevice().getDevice().resetFences(*gpu.getInFlightFence(gpu.currentFrame));
    tracing::endTrace("reset fences");

    // Fan out model-matrix writes for recently-moved nodes into this frame's slice. Runs here
    // (after the fence wait above) so the slice being written is no longer read by the GPU.
    scene.sceneGraph.uploadDirtyTransforms(gpu.currentFrame);

    for (auto& [id, light] : scene.lights) {
        glm::vec3 lightDir = scene.sceneGraph.getNode(light.nodeIndex).forward();
        glm::vec3 lightPos = scene.sceneGraph.getNode(light.nodeIndex).getWorldPosition();

        bool matricesUpdated = false;
        if (light.castsShadows == 1) {
            if (light.type == LightType::Directional) {
                // CSM depends on camera — always recalculate.
                calculateCascadedLightSpaceMatrices(light, scene.activeCamera, this);
                matricesUpdated = true;
            } else if (light.type == LightType::Point && light.shadowDirty) {
                calculatePointLightFaceMatrices(light, lightPos);
                matricesUpdated = true;
            }
        }
        if (matricesUpdated) light.gpuDirtyFrames = MAX_FRAMES_IN_FLIGHT;

        // Fan out the GPULight write across every frame-in-flight slice so the
        // per-frame buffer stays coherent instead of one slice winning the race.
        if (light.gpuDirtyFrames > 0) {
            bindless.descriptorSet->updateFixedBufferWithOffset<GPULight>(passResources.buffers.lightBufferIndex, id, light.toGPU(lightPos, lightDir), gpu.currentFrame);
            light.gpuDirtyFrames--;
        }
    }
    tracing::startTrace("record command buffer");
    gpu.getCommandBuffer(gpu.currentFrame).reset();
    recordCommandBuffer(imageIndex);
    tracing::endTrace("record command buffer");
    frameClk::time_point tRecord = frameClk::now();

    tracing::startTrace("submit & present");
    vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
    const vk::SubmitInfo submitInfo{.waitSemaphoreCount = 1,
                                    .pWaitSemaphores = &*gpu.getPresentCompleteSemaphore(gpu.currentFrame),
                                    .pWaitDstStageMask = &waitDestinationStageMask,
                                    .commandBufferCount = 1,
                                    .pCommandBuffers = &*gpu.getCommandBuffer(gpu.currentFrame),
                                    .signalSemaphoreCount = 1,
                                    .pSignalSemaphores = &*gpu.getRenderFinishedSemaphore(imageIndex)};

    gpu.getDevice().getGraphicsQueue().submit(submitInfo, gpu.getInFlightFence(gpu.currentFrame));

    const vk::PresentInfoKHR presentInfoKHR{.waitSemaphoreCount = 1,
                                            .pWaitSemaphores = &*gpu.getRenderFinishedSemaphore(imageIndex),
                                            .swapchainCount = 1,
                                            .pSwapchains = &*gpu.getSwapchain().getSwapChain(),
                                            .pImageIndices = &imageIndex};

    try {
        result = gpu.getDevice().getPresentQueue().presentKHR(presentInfoKHR);
    } catch (const vk::OutOfDateKHRError&) {
        result = vk::Result::eErrorOutOfDateKHR;
    }

    if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || framebufferResized) {
        framebufferResized = false;
        gpu.recreateSwapchain();
        handleSwapchainResize();
    } else if (result != vk::Result::eSuccess) {
        throw std::runtime_error("failed to present swap chain image!");
    }
    tracing::endTrace("submit & present");
    frameClk::time_point tPresent = frameClk::now();

    gpu.currentFrame = (gpu.currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    gpu.totalFrames++;
    if(gpu.totalFrames % 60 == 0) {
        bindless.pipelineManager->checkForShaderUpdates();
    }

    // outsideFrame = time between the end of the previous drawFrame and the start of this one — the
    // main loop (GUI, events, scene updates). A stall there never shows up in the phases.
    frameClk::time_point tEnd = frameClk::now();
    static frameClk::time_point lastFrameEnd{};
    double outsideMs = (lastFrameEnd == frameClk::time_point{}) ? 0.0 : phaseMs(lastFrameEnd, tStart);
    lastFrameEnd = tEnd;

    if (phaseMs(tStart, tEnd) > 200.0 || outsideMs > 200.0) {
        static frameClk::time_point lastSlowPrint{};
        if (tEnd - lastSlowPrint > std::chrono::seconds(1)) {
            lastSlowPrint = tEnd;
            std::cout << "[slow frame] total " << phaseMs(tStart, tEnd)
                      << "ms | fenceWait " << phaseMs(tStart, tFence)
                      << "ms | acquire " << phaseMs(tFence, tAcquire)
                      << "ms | record " << phaseMs(tAcquire, tRecord)
                      << "ms | submit/present " << phaseMs(tRecord, tPresent)
                      << "ms | shaderCheck " << phaseMs(tPresent, tEnd)
                      << "ms | outsideFrame " << outsideMs << "ms" << std::endl;
        }
    }
    tracing::endTrace("draw frame");
    #if TRACY_ENABLE
    FrameMark; // Tracy frame boundary
    #endif
}

/////=================================================GET/SET=================================================/////

uint32_t Renderer::getModelMatrixBufferIndex() { return buffers.modelMatrixBufferIndex; }
uint32_t Renderer::getLightBufferIndex() { return passResources.buffers.lightBufferIndex; }
uint32_t Renderer::getParticleBufferIndex() { return buffers.emitterBufferIndex; }

void Renderer::clearLights() { scene.clearLights(bindless, passResources.buffers.lightBufferIndex); }
void Renderer::clearVolumes() { scene.clearVolumes(); }

void Renderer::toggleVsync() {
    gpu.vSync = !gpu.vSync;
    gpu.recreateSwapchain();
    handleSwapchainResize();
}


void Renderer::handleSwapchainResize() {
    int width = 0, height = 0;
    glfwGetFramebufferSize(gpu.getWindow(), &width, &height);
    if (width > 0 && height > 0) {
        createRoughnessMetalResources(width, height);
        createNormalResources(width, height);
        createMotionVectorResources(width,height);
        createColorResolveResources(width, height);
        createHiZResources(width, height);
        createSDFResources(width, height);

        for(auto& pass : passes) {
            pass.second->init(width,height);
        }
    }
    scene.activeCamera.aspectRatio = static_cast<float>(width) / static_cast<float>(height);
    scene.activeCamera.calculateViewProjectionMatrix();
}

void Renderer::blurAttachment(vk::raii::CommandBuffer& cmd, uint32_t sourceTextureIndex, uint32_t tempTextureIndex, uint32_t width, uint32_t height, float blurRadius,
                    uint32_t samplerIndex) {

    auto& blurPipeline = *bindless.pipelineManager->getPostProcessPipelines()[passResources.blurPipelineIndex];
    auto& sourceTexture = bindless.descriptorSet->getTextureResource(sourceTextureIndex);
    auto& tempTexture = bindless.descriptorSet->getTextureResource(tempTextureIndex);
    vk::Extent2D extent{width, height};

    // Horizontal blur (source -> temp)
    resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *tempTexture.image, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eColorAttachmentOptimal);
    drawFullscreenPass(cmd, blurPipeline, *tempTexture.imageView, extent,
        BlurPushConstants{.inputTextureIndex = sourceTextureIndex, .samplerIndex = samplerIndex, .isHorizontal = 1, .blurRadius = blurRadius, .resolution = glm::uvec2(width, height)});
    resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *tempTexture.image, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

    // Vertical blur (temp -> source)
    resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *sourceTexture.image, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eColorAttachmentOptimal);
    drawFullscreenPass(cmd, blurPipeline, *sourceTexture.imageView, extent,
        BlurPushConstants{.inputTextureIndex = tempTextureIndex, .samplerIndex = samplerIndex, .isHorizontal = 0, .blurRadius = blurRadius, .resolution = glm::uvec2(width, height)});
    resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *sourceTexture.image, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
}

/////=================================================CREATE RESOURCES=================================================/////

void Renderer::createShadowAtlas(uint32_t resolution) {
    scene.shadowAtlas.init();
    vk::Format format = vk::Format::eD32Sfloat;
    vk::raii::Image image = nullptr;
    vk::raii::DeviceMemory memory = nullptr;
    resource::createImage(*bindless.resourceCtx, resolution,resolution, 1, vk::SampleCountFlagBits::e1, format,
                                vk::ImageTiling::eOptimal,
                                vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
                                vk::MemoryPropertyFlagBits::eDeviceLocal, image, memory, 1);

    vk::raii::ImageView view = resource::createImageView(*bindless.resourceCtx, image,format,vk::ImageAspectFlagBits::eDepth,1);

    // Atlas rests in eShaderReadOnlyOptimal so it matches its bindless descriptor's recorded layout —
    // required for the froxel light pass to sample it correctly from compute (VolumetricsPass C).
    // Seed via depth-read-only (Undefined can't go straight to shader-read for a depth aspect here).
    resource::transitionImageLayout(*bindless.resourceCtx, nullptr, *image, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthReadOnlyOptimal);
    resource::transitionImageLayout(*bindless.resourceCtx, nullptr, *image, vk::ImageLayout::eDepthReadOnlyOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

    scene.shadowAtlas.textureIndex = bindless.descriptorSet->allocateTexture(std::move(image),std::move(memory),std::move(view),"internal/scene.shadowAtlas",false,resolution,resolution);
}

void Renderer::createRoughnessMetalResources(uint32_t width, uint32_t height) {
    createOrResizeMSAATarget(roughnessMetal, width, height, vk::Format::eR8G8B8A8Unorm);
    createOrResizeRenderTarget(passResources.roughnessMetalTextureIndex, width, height, vk::Format::eR8G8B8A8Unorm, "internal/roughness_metal");
}

void Renderer::createNormalResources(uint32_t width, uint32_t height) {
    createOrResizeMSAATarget(normalMSAA, width, height, vk::Format::eR8G8B8A8Unorm);
    // Create with mip levels for SSR normal pre-filtering
    normalMipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1;
    if (passResources.normalTextureIndex != 0xFFFFFFFF) {
        bindless.descriptorSet->freeTexture(passResources.normalTextureIndex);
    }
    vk::raii::Image image = nullptr;
    vk::raii::DeviceMemory memory = nullptr;
    resource::createImage(*bindless.resourceCtx, width, height, normalMipLevels, vk::SampleCountFlagBits::e1, vk::Format::eR8G8B8A8Unorm, vk::ImageTiling::eOptimal,
                                 vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled |
                                 vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst,
                                 vk::MemoryPropertyFlagBits::eDeviceLocal, image, memory, 1);
    auto view = resource::createImageView(*bindless.resourceCtx, image, vk::Format::eR8G8B8A8Unorm, vk::ImageAspectFlagBits::eColor, normalMipLevels);
    resource::transitionImageLayout(*bindless.resourceCtx, nullptr, image, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal, 0, normalMipLevels);
    passResources.normalTextureIndex = bindless.descriptorSet->allocateTexture(std::move(image), std::move(memory), std::move(view), "internal/normals", false, width, height);
}

void Renderer::createColorResolveResources(uint32_t width, uint32_t height) {
    colorResolveMipViews.clear();

    passResources.fullscreenMipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1;

    if (passResources.colorResolveTextureIndex != 0xFFFFFFFF) {
        bindless.descriptorSet->freeTexture(passResources.colorResolveTextureIndex);
    }

    auto format = gpu.getSwapchain().getHDRColorFormat();

    vk::raii::Image image = nullptr;
    vk::raii::DeviceMemory memory = nullptr;
    resource::createImage(*bindless.resourceCtx, width, height, passResources.fullscreenMipLevels, vk::SampleCountFlagBits::e1, format, vk::ImageTiling::eOptimal,
                                 vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc,
                                 vk::MemoryPropertyFlagBits::eDeviceLocal, image, memory);

    // Create per-mip image views for rendering to individual levels
    for (uint32_t mip = 0; mip < passResources.fullscreenMipLevels; ++mip) {
        vk::ImageViewCreateInfo viewInfo{.image = image,
                                         .viewType = vk::ImageViewType::e2D,
                                         .format = format,
                                         .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = mip, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1}};
        colorResolveMipViews.emplace_back(gpu.getDevice().getDevice(), viewInfo);
    }

    // Create a full-chain view for sampling
    auto fullView = resource::createImageView(*bindless.resourceCtx, image, format, vk::ImageAspectFlagBits::eColor, passResources.fullscreenMipLevels);
    resource::transitionImageLayout(*bindless.resourceCtx, nullptr, image, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal, 0, passResources.fullscreenMipLevels);
    passResources.colorResolveTextureIndex = bindless.descriptorSet->allocateTexture(std::move(image), std::move(memory), std::move(fullView), "internal/color_resolve", false, width, height);

    // Temp texture for separable blur passes (mipmapped, matching color resolve)
    tempBlurMipViews.clear();

    if (passResources.tempBlurTextureIndex != 0xFFFFFFFF) {
        bindless.descriptorSet->freeTexture(passResources.tempBlurTextureIndex);
    }

    vk::raii::Image tempImage = nullptr;
    vk::raii::DeviceMemory tempMemory = nullptr;
    resource::createImage(*bindless.resourceCtx, width, height, passResources.fullscreenMipLevels, vk::SampleCountFlagBits::e1, format, vk::ImageTiling::eOptimal,
                                 vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
                                 vk::MemoryPropertyFlagBits::eDeviceLocal, tempImage, tempMemory);

    for (uint32_t mip = 0; mip < passResources.fullscreenMipLevels; ++mip) {
        vk::ImageViewCreateInfo viewInfo{.image = tempImage,
                                         .viewType = vk::ImageViewType::e2D,
                                         .format = format,
                                         .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = mip, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1}};
        tempBlurMipViews.emplace_back(gpu.getDevice().getDevice(), viewInfo);
    }

    auto tempFullView = resource::createImageView(*bindless.resourceCtx, tempImage, format, vk::ImageAspectFlagBits::eColor, passResources.fullscreenMipLevels);
    resource::transitionImageLayout(*bindless.resourceCtx, nullptr, tempImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal, 0, passResources.fullscreenMipLevels);
    passResources.tempBlurTextureIndex = bindless.descriptorSet->allocateTexture(std::move(tempImage), std::move(tempMemory), std::move(tempFullView), "internal/blur_temp", false, width, height);

    // HDR composite target seeded from colorResolve via copy, so it needs TransferDst.
    createOrResizeRenderTarget(passResources.compositeColorTextureIndex, width, height, format, "internal/composite_color", vk::ImageUsageFlagBits::eTransferDst);

    // Auto-exposure metering: mipped log-luminance target (box-averaged to 1x1 = geometric mean).
    avgLumMip0View = nullptr;
    if (avgLumTextureIndex != 0xFFFFFFFF) {
        bindless.descriptorSet->freeTexture(avgLumTextureIndex);
    }
    vk::raii::Image lumImage = nullptr;
    vk::raii::DeviceMemory lumMemory = nullptr;
    resource::createImage(*bindless.resourceCtx, width, height, passResources.fullscreenMipLevels, vk::SampleCountFlagBits::e1, vk::Format::eR16Sfloat, vk::ImageTiling::eOptimal,
                                 vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst,
                                 vk::MemoryPropertyFlagBits::eDeviceLocal, lumImage, lumMemory);
    vk::ImageViewCreateInfo lumMip0Info{.image = lumImage, .viewType = vk::ImageViewType::e2D, .format = vk::Format::eR16Sfloat,
                                        .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1}};
    avgLumMip0View = vk::raii::ImageView(gpu.getDevice().getDevice(), lumMip0Info);
    auto lumFullView = resource::createImageView(*bindless.resourceCtx, lumImage, vk::Format::eR16Sfloat, vk::ImageAspectFlagBits::eColor, passResources.fullscreenMipLevels);
    resource::transitionImageLayout(*bindless.resourceCtx, nullptr, lumImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal, 0, passResources.fullscreenMipLevels);
    avgLumTextureIndex = bindless.descriptorSet->allocateTexture(std::move(lumImage), std::move(lumMemory), std::move(lumFullView), "internal/avg_luminance", false, width, height);

    // 1x1 ping-pong adapted-luminance targets (created once; persist across resizes for adaptation state).
    if (adaptedLumIndex[0] == 0xFFFFFFFF) {
        createOrResizeRenderTarget(adaptedLumIndex[0], 1, 1, vk::Format::eR16Sfloat, "internal/adapted_lum0");
        createOrResizeRenderTarget(adaptedLumIndex[1], 1, 1, vk::Format::eR16Sfloat, "internal/adapted_lum1");
        adaptInitialized = false;
    }
}

void Renderer::createMotionVectorResources(uint32_t width, uint32_t height) {
    createOrResizeMSAATarget(motionVectors,width,height, vk::Format::eR16G16Sfloat);
    createOrResizeRenderTarget(passResources.motionVectorTextureIndex, width, height, vk::Format::eR16G16Sfloat,"internal/motion_vectors");
}

void Renderer::createHiZResources(uint32_t width, uint32_t height) {
    hiZMipViews.clear();

    // Calculate mip levels for the Hi-Z pyramid
    passResources.hiZMipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1;

    // Free previous Hi-Z texture if it exists
    if (passResources.hiZTextureIndex != 0xFFFFFFFF) {
        bindless.descriptorSet->freeTexture(passResources.hiZTextureIndex);
    }

    // Create mipmapped R32Sfloat image
    vk::raii::Image image = nullptr;
    vk::raii::DeviceMemory memory = nullptr;
    resource::createImage(*bindless.resourceCtx, width, height, passResources.hiZMipLevels, vk::SampleCountFlagBits::e1, vk::Format::eR32Sfloat, vk::ImageTiling::eOptimal,
                                 vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
                                 vk::MemoryPropertyFlagBits::eDeviceLocal, image, memory);

    // Create per-mip image views for rendering to individual levels
    for (uint32_t mip = 0; mip < passResources.hiZMipLevels; ++mip) {
        vk::ImageViewCreateInfo viewInfo{.image = image,
                                         .viewType = vk::ImageViewType::e2D,
                                         .format = vk::Format::eR32Sfloat,
                                         .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = mip, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1}};
        hiZMipViews.emplace_back(gpu.getDevice().getDevice(), viewInfo);
    }

    // Create a full-chain view for sampling
    auto fullView = resource::createImageView(*bindless.resourceCtx, image, vk::Format::eR32Sfloat, vk::ImageAspectFlagBits::eColor, passResources.hiZMipLevels);
    resource::transitionImageLayout(*bindless.resourceCtx, nullptr, image, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal, 0, passResources.hiZMipLevels);
    passResources.hiZTextureIndex = bindless.descriptorSet->allocateTexture(std::move(image), std::move(memory), std::move(fullView), "internal/hiZ", false, width, height);

    // Hi-Z pipeline (only created once)
    if (hiZPipelineIndex == 0xFFFFFFFF) {
        hiZPipelineIndex = bindless.pipelineManager->createPipeline<HiZPushConstants>(
            PipelineCategory::BEFORE_GEOMETRY, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone,
            vk::False, vk::False, "shaders/hiz_reduce.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet(),
            vk::Format::eUndefined);
    }
}

void Renderer::createSDFResources(uint32_t width, uint32_t height) {
    createOrResizeRenderTarget(sdfTextureIndex,width,height,gpu.getSwapchain().getSwapChainImageFormat(),"internal/sdf");

    if (sdfPipelineIndex == 0xFFFFFFFF) {
        sdfPipelineIndex =
        bindless.pipelineManager->createPipeline<SDFPushConstants>(PipelineCategory::POSTPROCESS, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone, vk::False,
                                                           vk::False, "shaders/sdf.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet(),
                                                           gpu.getSwapchain().getSwapChainImageFormat());
    }
    // SDF apply pipeline — composites onto the HDR composite target
    if (sdfApplyPipelineIndex == 0xFFFFFFFF) {
        sdfApplyPipelineIndex = bindless.pipelineManager->createPipeline<SDFApplyPushConstants>(
            PipelineCategory::POSTPROCESS_ALPHA_BLEND, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone,
            vk::False, vk::False, "shaders/sdf_apply.spv", bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet(),
            gpu.getSwapchain().getHDRColorFormat());
    }
}

/////=================================================RENDERING=================================================/////

void Renderer::recordCommandBuffer(uint32_t imageIndex) {
    auto& cmd = gpu.getCommandBuffer(gpu.currentFrame);
    cmd.begin({});

    tracing::startTrace("record shadow pass");
    
    resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *bindless.descriptorSet->getTextureResource(scene.shadowAtlas.textureIndex).image, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eDepthStencilAttachmentOptimal);
    uint32_t shadowSlot = 0;
    for (auto& [lightId, light] : scene.lights) {
        if (light.castsShadows != 1) continue;
        // Skip point lights whose shadow maps are already up to date
        if (light.type == LightType::Point && !light.shadowDirty) continue;
        if (shadowSlot >= MAX_SHADOW_CASTERS) {
            std::cerr << "Warning: more than MAX_SHADOW_CASTERS shadow-casting lights in a frame; dropping extras" << std::endl;
            break;
        }
        recordShadowPass(cmd, light, shadowSlot);
        if (light.type == LightType::Point) light.shadowDirty = false;
        shadowSlot++;
    }
    resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *bindless.descriptorSet->getTextureResource(scene.shadowAtlas.textureIndex).image, vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

    tracing::endTrace("record shadow pass");

    // Voxelization records before the geometry pass so the lit shader samples volumes built with
    // this frame's grid center — sampling last frame's contents through this frame's matrix goes
    // black for one frame whenever the grid recenters. Only needs the shadow atlas, recorded above.
    passes.at(PassId::VOXELIZATION)->record(cmd, imageIndex);

    tracing::startTrace("record geo pass");
    recordGeometryPass(cmd, imageIndex);
    tracing::endTrace("record geo pass");

    recordResolveToCompositeCopy(cmd);

    tracing::startTrace("record passes");
    for(auto& pass : passes) {
        if (pass.first == PassId::VOXELIZATION) continue; // recorded before the geometry pass
        pass.second->record(cmd, imageIndex);
    }
    tracing::endTrace("record passes");

    // Voxel grid overlay (features.voxelDebug). Separate from the loop because it draws over the HDR
    // composite, so it has to run once every pass above has finished contributing to it.
    static_cast<VoxelizationPass*>(passes.at(PassId::VOXELIZATION).get())->recordDebugOverlay(cmd);

    recordBillboardBlendPass(cmd, imageIndex);

    tracing::startTrace("record SDF pass");
    if(sdfPipelineIndex != 0xFFFFFFFF)
        recordSDFPass(cmd, imageIndex);
    tracing::endTrace("record SDF pass");

    // Tonemap HDR composite -> swapchain. Scene passes above run in HDR; UI passes below run in LDR.
    recordTonemapPass(cmd, imageIndex);

    tracing::startTrace("record image vis pass");
    if (features.imageVis.imageIndex != 0xFFFFFFFF)
        recordImageVisPass(cmd, imageIndex);
    tracing::endTrace("record image vis pass");

    recordOverlayPass(cmd, imageIndex);

    resource::transitionImageLayout(*bindless.resourceCtx, &cmd, gpu.getSwapchain().getSwapChainImages()[imageIndex], vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR);
    cmd.end();
}

// Finest LOD whose average triangle still covers kMinAvgTrianglePixels on screen 
// Foreshortening and backfaces shrink real coverage further; the target constant absorbs that on average.
static uint32_t selectLOD(const Mesh& mesh, float areaToPixels2) {
    constexpr float kMinAvgTrianglePixels = 8.0f;
    if (mesh.surfaceArea <= 0.0f) return 0;
    uint32_t lod = 0;
    while (lod + 1 < mesh.LODs.size()) {
        float triCount = std::max(1.0f, static_cast<float>(mesh.LODs[lod] / 3));
        if (mesh.surfaceArea * areaToPixels2 / triCount >= kMinAvgTrianglePixels) break;
        lod++;
    }
    return lod;
}

template <typename PerMeshFn, typename PerInstanceFn>
void Renderer::buildGeometryDrawCommands(const std::array<Plane, 6>& frustumPlanes, bool doCulling, PerMeshFn&& perMeshFn, PerInstanceFn&& perInstanceFn,
                                         const std::function<bool(const Node&)>& nodeFilter) {
    if (scene.renderListDirty) {
        std::sort(scene.renderEntries.begin(), scene.renderEntries.end(), [](const Scene::RenderEntry& a, const Scene::RenderEntry& b) {
            return std::tie(a.shaderPipelineIndex,a.meshIndex) < std::tie(b.shaderPipelineIndex, b.meshIndex);
        });
        scene.renderListDirty = false;
    }

    indirectCommands.clear();
    scene.shaderDrawRanges.clear();
    if (scene.renderEntries.empty()) return;

    std::unordered_set<uint32_t> freedMeshes;

    uint32_t groupPipelineIdx = UINT32_MAX;
    uint32_t groupMeshIdx     = UINT32_MAX;
    Mesh*           groupMesh          = nullptr;
    Node*           groupFirstNode     = nullptr;
    const Material* groupFirstMaterial = nullptr;

    uint32_t currentInstanceCount = 0;
    uint32_t instanceWriteCursor  = 0;
    // screen pixels per model-space unit at distance 1; |proj[1][1]| = 1/tan(fovY/2)
    const float pixelsPerUnit = 0.5f * static_cast<float>(gpu.getSwapchain().getSwapChainExtent().height) *
                                std::abs(scene.activeCamera.projectionMatrix[1][1]);
    const float pixelsPerUnit2 = pixelsPerUnit * pixelsPerUnit;
    // largest scale²/distance² among the group's instances: its closest/biggest view drives the LOD
    float groupMaxAreaScale2 = 0.0f;
    // world AABBs of the group's instances (showBBoxes); drawn at flush, tinted by the group's LOD
    std::vector<std::pair<glm::vec3, glm::vec3>> groupBBoxes;

    auto flushGroup = [&]() {
        if (currentInstanceCount == 0 || groupMesh == nullptr) {
            currentInstanceCount = 0;
            groupMaxAreaScale2 = 0.0f;
            groupBBoxes.clear();
            return;
        }
        uint32_t lod = selectLOD(*groupMesh, pixelsPerUnit2 * groupMaxAreaScale2);
        groupMesh->currentLOD = lod;
        static const glm::vec4 lodColors[4] = {{0, 1, 0, 1}, {1, 1, 0, 1}, {1, 0.5f, 0, 1}, {1, 0, 0, 1}}; // green -> red
        for (const auto& [bbMin, bbMax] : groupBBoxes)
            Gizmos::drawBox(bbMin, bbMax, lodColors[std::min<uint32_t>(lod, 3)]);
        groupBBoxes.clear();
        indirectCommands.push_back({.indexCount    = groupMesh->lodIndexCount(lod),
                                    .instanceCount = currentInstanceCount,
                                    .firstIndex    = static_cast<uint32_t>(groupMesh->indexOffset / sizeof(uint32_t)) + groupMesh->lodIndexStart(lod),
                                    .vertexOffset  = 0,
                                    .firstInstance = instanceWriteCursor});
        groupMaxAreaScale2 = 0.0f;
        perMeshFn(*groupMesh, *groupFirstNode, *groupFirstMaterial);
        if (!scene.shaderDrawRanges.empty())
            scene.shaderDrawRanges.back().commandCount++;
        instanceWriteCursor += currentInstanceCount;
        currentInstanceCount = 0;
    };

    for (const auto& entry : scene.renderEntries) {
        Node& node = scene.sceneGraph.getNode(entry.nodeIndex);
        const Material& material = scene.materials[entry.materialIndex];
        auto& mesh = scene.assetManager.meshes[entry.meshIndex];

        if (mesh.freed) {
            if (freedMeshes.insert(entry.meshIndex).second) {
                bindless.descriptorSet->freeVariableBuffer(vertexBufferIndex, mesh.vertexAllocationOffset);
                bindless.descriptorSet->freeVariableBuffer(indexBufferIndex,  mesh.indexAllocationOffset);
                scene.assetManager.freeMeshes.push(entry.meshIndex);
            }
            continue;
        }
        if (nodeFilter && !nodeFilter(node)) continue;

        bool pendingBBox = false;
        glm::vec3 bbWorldMin{}, bbWorldMax{};
        if (node.isBoundingBoxValid() && doCulling) {
            glm::vec3 worldMin, worldMax;
            transformAABBToWorldSpace(mesh.boundingBoxMin, mesh.boundingBoxMax, node.getTransform(), worldMin, worldMax);
            if (!isAABBInFrustum(worldMin, worldMax, frustumPlanes)) {
                culledCount++;
                continue;
            }
            if (features.showBBoxes) {
                // deferred to flushGroup: the group's LOD (its color) isn't known yet
                pendingBBox = true;
                bbWorldMin = worldMin;
                bbWorldMax = worldMax;
            }
        }

        // Group boundary: flush the previous group, then open a new one.
        if (entry.meshIndex != groupMeshIdx || entry.shaderPipelineIndex != groupPipelineIdx) {
            flushGroup();

            if (entry.shaderPipelineIndex != groupPipelineIdx) {
                groupPipelineIdx = entry.shaderPipelineIndex;
                scene.shaderDrawRanges.push_back({groupPipelineIdx,
                                                  static_cast<uint32_t>(indirectCommands.size()),
                                                  0});
            }
            groupMeshIdx       = entry.meshIndex;
            groupMesh          = &mesh;
            groupFirstNode     = &node;
            groupFirstMaterial = &material;
        }

        if (pendingBBox)
            groupBBoxes.emplace_back(bbWorldMin, bbWorldMax);

        // LOD input: this instance's scale²/distance² (screen size of one unit of surface area)
        const glm::mat4& t = node.getTransform();
        float scale2 = std::max({glm::dot(glm::vec3(t[0]), glm::vec3(t[0])),
                                 glm::dot(glm::vec3(t[1]), glm::vec3(t[1])),
                                 glm::dot(glm::vec3(t[2]), glm::vec3(t[2]))});
        glm::vec3 toCam = glm::vec3(t[3]) - scene.activeCamera.position;
        float dist2 = std::max(glm::dot(toCam, toCam), 1e-6f);
        groupMaxAreaScale2 = std::max(groupMaxAreaScale2, scale2 / dist2);

        perInstanceFn(mesh, node, material);
        currentInstanceCount++;
    }
    flushGroup();
}

void Renderer::recordHiZPass(vk::raii::CommandBuffer& cmd) {
    if (passResources.hiZTextureIndex == 0xFFFFFFFF || hiZPipelineIndex == 0xFFFFFFFF) return;

    auto& hiZRes = bindless.descriptorSet->getTextureResource(passResources.hiZTextureIndex);
    auto& pipeline = *bindless.pipelineManager->getBeforeGeoPipelines()[hiZPipelineIndex];
    uint32_t w = hiZRes.width;
    uint32_t h = hiZRes.height;

    // Transition depth resolve to shader read for sampling
    resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *bindless.descriptorSet->getTextureResource(gpu.getSwapchain().getDepthResolveIndex()).image,
        vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

    for (uint32_t mip = 0; mip < passResources.hiZMipLevels; ++mip) {
        uint32_t mipW = std::max(1u, w >> mip);
        uint32_t mipH = std::max(1u, h >> mip);

        // Transition this mip to color attachment
        resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *hiZRes.image,
            vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eColorAttachmentOptimal, mip, 1);

        HiZPushConstants hizPC;
        if (mip == 0) {
            // Mip 0: copy from depth resolve
            hizPC = {.inputTextureIndex = gpu.getSwapchain().getDepthResolveIndex(),
                     .samplerIndex = passResources.depthSamplerIndex,
                     .inputMipLevel = 0,
                     .reduceMode = 0,
                     .inputResolution = glm::uvec2(mipW, mipH)};
        } else {
            // Mip N: min-reduce from mip N-1 of the Hi-Z texture itself
            hizPC = {.inputTextureIndex = passResources.hiZTextureIndex,
                     .samplerIndex = passResources.depthSamplerIndex,
                     .inputMipLevel = mip - 1,
                     .reduceMode = 1,
                     .inputResolution = glm::uvec2(std::max(1u, w >> (mip - 1)), std::max(1u, h >> (mip - 1)))};
        }

        vk::Extent2D mipExtent{mipW, mipH};
        drawFullscreenPass(cmd, pipeline, *hiZMipViews[mip], mipExtent, hizPC);

        // Transition this mip back to shader read
        resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *hiZRes.image,
            vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, mip, 1);
    }
    // hiZ empty space calculation pass?

    // Transition depth resolve back to depth attachment
    resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *bindless.descriptorSet->getTextureResource(gpu.getSwapchain().getDepthResolveIndex()).image,
        vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eDepthStencilAttachmentOptimal);
}

void Renderer::recordGeometryPass(vk::raii::CommandBuffer& cmd, uint32_t imageIndex) {
    resource::transitionImageLayouts(cmd, {
        {gpu.getSwapchain().getSwapChainImages()[imageIndex],                            vk::ImageLayout::eUndefined,              vk::ImageLayout::eColorAttachmentOptimal},
        {gpu.getSwapchain().getColorImage(),                                             vk::ImageLayout::eUndefined,              vk::ImageLayout::eColorAttachmentOptimal},
        {gpu.getSwapchain().getDepthImage(),                                             vk::ImageLayout::eUndefined,              vk::ImageLayout::eDepthStencilAttachmentOptimal},
        {roughnessMetal.image,                                                           vk::ImageLayout::eUndefined,              vk::ImageLayout::eColorAttachmentOptimal},
        {*bindless.descriptorSet->getTextureResource(passResources.roughnessMetalTextureIndex).image,   vk::ImageLayout::eShaderReadOnlyOptimal,  vk::ImageLayout::eColorAttachmentOptimal},
        {normalMSAA.image,                                                               vk::ImageLayout::eUndefined,              vk::ImageLayout::eColorAttachmentOptimal},
        {*bindless.descriptorSet->getTextureResource(passResources.normalTextureIndex).image,           vk::ImageLayout::eShaderReadOnlyOptimal,  vk::ImageLayout::eColorAttachmentOptimal},
        {motionVectors.image,                                                            vk::ImageLayout::eUndefined,              vk::ImageLayout::eColorAttachmentOptimal},
        {*bindless.descriptorSet->getTextureResource(passResources.motionVectorTextureIndex).image,     vk::ImageLayout::eShaderReadOnlyOptimal,  vk::ImageLayout::eColorAttachmentOptimal},
    });

    vk::ClearValue clearColor{.color = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f})};
    vk::ClearValue clearDepth{.depthStencil = vk::ClearDepthStencilValue{0.0f, 0}}; // reverse-Z: far = 0

    // Frustum cull and build draw commands + lit draw data
    Camera fakeCam = scene.activeCamera;
    fakeCam.fov = cullFovScale * scene.activeCamera.fov;
    fakeCam.calculateViewProjectionMatrix();
    std::array<Plane, 6> frustumPlanes = extractFrustumPlanes(fakeCam.viewProjection);
    culledCount = 0;
    litInstanceDataList.clear();
    litMeshDrawDataList.clear();
    buildGeometryDrawCommands(frustumPlanes, true, [&](const Mesh& mesh, Node& node, const Material& material) {
        litMeshDrawDataList.push_back({ .vertexAllocationOffset = mesh.vertexAllocationOffset,
                                        .vertexOffset = static_cast<uint32_t>(mesh.vertexOffset),
                                        .vertexStride = mesh.vertexStride});},
                                    
                                    [&](const Mesh& mesh, Node& node, const Material& material) {
        // Absent maps get the 1x1 defaults so the shader can sample all four textures
        // unconditionally (fetches issue back-to-back instead of serializing on branches).
        uint32_t mflags = static_cast<uint32_t>(material.flags);
        litInstanceDataList.push_back({ .modelMatrixIndex      = node.getModelMatrixIndex(),
                                        .albedoTextureIndex    = (mflags & HAS_ALBEDO)    ? material.albedoTextureIndex    : defaultAlbedoIndex,
                                        .roughnessTextureIndex = (mflags & HAS_ROUGHNESS) ? material.roughnessTextureIndex : defaultRoughnessIndex,
                                        .metallicTextureIndex  = (mflags & HAS_METALLIC)  ? material.metallicTextureIndex  : defaultMetallicIndex,
                                        .normalTextureIndex    = (mflags & HAS_NORMAL)    ? material.normalTextureIndex    : defaultNormalIndex,
                                        .environmentMapIndex   = material.environmentMapIndex,
                                        .maxEnvMips            = static_cast<float>(bindless.descriptorSet->getTextureMipLevels(material.environmentMapIndex) - 1),
                                        .materialFlags         = static_cast<uint32_t>(material.flags),
                                        .metallic              = material.metallic,
                                        .roughness             = material.roughness,
                                        .alphaCutoff           = material.alphaCutoff,
                                        .packedColor           = packColorRGBA8(material.color)});});

    // Backfill firstInstance into each LitMeshDrawData so shaders can compute the absolute instance
    // index as mesh.firstInstance + SV_InstanceID (independent of how Slang maps SV_InstanceID).
    for (size_t i = 0; i < indirectCommands.size() && i < litMeshDrawDataList.size(); ++i) {
        litMeshDrawDataList[i].firstInstance = indirectCommands[i].firstInstance;
    }

    vk::DeviceSize frameByteOffset = gpu.currentFrame * MAX_INDIRECT_COMMANDS * sizeof(DrawIndexedIndirectCommand);
    vk::Buffer indexBufferHandle = bindless.descriptorSet->getVariableBuffer(indexBufferIndex);

    glm::vec3 cameraForward = glm::normalize(scene.activeCamera.target - scene.activeCamera.position);

    LitPassData litPassData {
        .samplerIndex = passResources.defaultSamplerIndex,
        .lightCount = scene.getLightLoopBound(),
        .shadowSamplerIndex = shadowSamplerIndex,
        .shadowAtlasIndex = scene.shadowAtlas.textureIndex,
        .cameraPosition = scene.activeCamera.position,
        .voxelTextureIndex = static_cast<VoxelizationPass*>(passes.at(PassId::VOXELIZATION).get())->getVolumeTextureIndex(),
        .cameraForward = cameraForward,
        .voxelSamplerIndex = passResources.volumeSamplerIndex,
        .viewProjection = scene.activeCamera.viewProjection,
        .prevViewProjection = scene.activeCamera.prevViewProjection,
        .voxelViewProjection = VoxelizationPass::gridViewProjection(scene.activeCamera.position),
        .voxelResolution = VoxelizationPass::VOXEL_RESOLUTION,
        .voxelWorldExtent = VoxelizationPass::VOXEL_WORLD_EXTENT,
        .giHemisphereRays = static_cast<uint32_t>(features.vxgi.hemisphereRays),
        .giMaxSteps = static_cast<uint32_t>(features.vxgi.maxSteps),
        .giFetchBatch = static_cast<uint32_t>(features.vxgi.fetchBatch),
        .giMode = static_cast<uint32_t>(features.vxgi.mode),
        .giIrradianceIndices = static_cast<VoxelizationPass*>(passes.at(PassId::VOXELIZATION).get())->getIrradianceTextureIndices(),
        .giStrength = features.vxgi.strength,
        .giSkyIntensity = features.skyboxIntensity * features.vxgi.skyStrength
    };
    bindless.descriptorSet->updateFixedBufferWithOffset<LitPassData>(litPassDataBufferIndex,0,litPassData,gpu.currentFrame);

    // Mirror the hot pass + light data into this frame's UBO slot (constant-cache path for lit).
    GPULitFrameUniforms& uf = *litFrameStaging;
    uint32_t hotBound = scene.fillHotLights(uf.lights);
    uf.setPassData(litPassData);
    uf.indicesA.y = std::min(hotBound, MAX_UBO_LIGHTS);
    memcpy(bindless.descriptorSet->getUniformBufferMapped(litFrameUniformsIndex[gpu.currentFrame]), &uf, sizeof(uf));

    if (!indirectCommands.empty()) {
        // Upload indirect commands and per-draw data (shared between depth prepass and lit pass)
        memcpy(static_cast<char*>(litIndirectDrawBufferMapped) + frameByteOffset, indirectCommands.data(), indirectCommands.size() * sizeof(DrawIndexedIndirectCommand));

        bindless.descriptorSet->writeFixedBuffer<LitInstanceData>(litInstanceDataBufferIndex, litInstanceDataList.data(), static_cast<uint32_t>(litInstanceDataList.size()), gpu.currentFrame * MAX_FIXED_BUFFER, gpu.currentFrame);
        bindless.descriptorSet->writeFixedBuffer<LitMeshDrawData>(litMeshDrawDataBufferIndex, litMeshDrawDataList.data(), static_cast<uint32_t>(litMeshDrawDataList.size()), gpu.currentFrame * MAX_FIXED_BUFFER, gpu.currentFrame);

        LitPushConstants pushConstants = {.vertexBufferAddress    = bindless.descriptorSet->getVariableBuffers()[vertexBufferIndex]->address,
                                          .modelMatricesAddress   = bindless.descriptorSet->getFixedBuffers()[buffers.modelMatrixBufferIndex]->address + static_cast<vk::DeviceSize>(gpu.currentFrame) * MAX_FIXED_BUFFER * sizeof(glm::mat4),
                                          .lightsAddress          = bindless.descriptorSet->getFixedBuffers()[passResources.buffers.lightBufferIndex]->address + static_cast<vk::DeviceSize>(gpu.currentFrame) * MAX_FIXED_BUFFER * sizeof(GPULight),
                                          .litInstanceDataAddress = bindless.descriptorSet->getFixedBuffers()[litInstanceDataBufferIndex]->address + static_cast<vk::DeviceSize>(gpu.currentFrame) * MAX_FIXED_BUFFER * sizeof(LitInstanceData),
                                          .litMeshDrawDataAddress = bindless.descriptorSet->getFixedBuffers()[litMeshDrawDataBufferIndex]->address + static_cast<vk::DeviceSize>(gpu.currentFrame) * MAX_FIXED_BUFFER * sizeof(LitMeshDrawData),
                                          .litPassDataAddress     = bindless.descriptorSet->getFixedBuffers()[litPassDataBufferIndex]->address + static_cast<vk::DeviceSize>(gpu.currentFrame) * MAX_FIXED_BUFFER * sizeof(LitPassData),
                                          .frameUniformsIndex     = litFrameUniformsIndex[gpu.currentFrame]
                                        };

        // --- Depth prepass (depth-only, no color attachment) ---
        vk::RenderingAttachmentInfo depthPrepassAttachment = {.imageView = gpu.getSwapchain().getDepthImageView(),
                                                              .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                                              .resolveMode = vk::ResolveModeFlagBits::eMax, // reverse-Z: nearest sample = max depth
                                                              .resolveImageView = gpu.getSwapchain().getDepthResolveImageView(),
                                                              .resolveImageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                                              .loadOp = vk::AttachmentLoadOp::eClear,
                                                              .storeOp = vk::AttachmentStoreOp::eStore,
                                                              .clearValue = clearDepth};

        auto& motionVectorResolve = bindless.descriptorSet->getTextureResource(passResources.motionVectorTextureIndex);
        vk::ClearValue clearMotionVectors{.color = vk::ClearColorValue(std::array<float, 4>{0.0f,0.0f,0.0f,1.0f})};
        // SAMPLE_ZERO: averaging motion across an edge blends two unrelated motions — worse for
        // reprojection than picking one — and reads 1 sample instead of N in the resolve shader.
        vk::RenderingAttachmentInfo motionVectorAttachment = {  .imageView = motionVectors.view,
                                                                .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                                .resolveMode = vk::ResolveModeFlagBits::eSampleZero,
                                                                .resolveImageView = *motionVectorResolve.imageView,
                                                                .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                                .loadOp = vk::AttachmentLoadOp::eClear,
                                                                .storeOp = vk::AttachmentStoreOp::eStore,
                                                                .clearValue = clearMotionVectors};


        vk::RenderingInfo depthRenderingInfo = {.renderArea = {.offset = {0, 0}, .extent = gpu.getSwapchain().getSwapChainExtent()},
                                                .layerCount = 1,
                                                .colorAttachmentCount = 1,
                                                .pColorAttachments = &motionVectorAttachment,
                                                .pDepthAttachment = &depthPrepassAttachment};

        cmd.beginRendering(depthRenderingInfo);
        setFullscreenViewport(cmd, gpu.getSwapchain().getSwapChainExtent());

        auto& depthPipeline = bindless.pipelineManager->getBeforeGeoPipelines()[depthPipelineIndex];
        bindPipeline(cmd, *depthPipeline);
        cmd.bindIndexBuffer(indexBufferHandle, 0, vk::IndexType::eUint32);
        cmd.pushConstants<LitPushConstants>(depthPipeline->layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pushConstants);
        cmd.drawIndexedIndirect(*litIndirectDrawBuffer, frameByteOffset, static_cast<uint32_t>(indirectCommands.size()), sizeof(DrawIndexedIndirectCommand));

        cmd.endRendering();

        recordHiZPass(cmd);
    }

    // --- Lit geometry pass (2 color attachments: color + roughness/metallic) ---
    auto& colorResolve = bindless.descriptorSet->getTextureResource(passResources.colorResolveTextureIndex);
    resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *colorResolve.image,
        vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eColorAttachmentOptimal);

    vk::RenderingAttachmentInfo colorAttachment = {.imageView = gpu.getSwapchain().getColorImageView(),
                                                   .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                   .resolveMode = vk::ResolveModeFlagBits::eAverage,
                                                   .resolveImageView = *colorResolve.imageView,
                                                   .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                   .loadOp = vk::AttachmentLoadOp::eClear,
                                                   .storeOp = vk::AttachmentStoreOp::eStore,
                                                   .clearValue = clearColor};

    auto& roughnessMetalResolve = bindless.descriptorSet->getTextureResource(passResources.roughnessMetalTextureIndex);
    vk::ClearValue clearRoughMetal{.color = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f})};
    // SAMPLE_ZERO for the aux G-buffer targets: averaging two materials' roughness (or two surfaces'
    // normals, which also denormalizes) across an edge is meaningless for the 1x consumers (SSR/SSAO),
    // and the resolve reads 1 sample instead of N. Color keeps AVERAGE — that's the actual AA.
    vk::RenderingAttachmentInfo roughnessMetalAttachment = {.imageView = *roughnessMetal.view,
                                                             .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                             .resolveMode = vk::ResolveModeFlagBits::eSampleZero,
                                                             .resolveImageView = *roughnessMetalResolve.imageView,
                                                             .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                             .loadOp = vk::AttachmentLoadOp::eClear,
                                                             .storeOp = vk::AttachmentStoreOp::eStore,
                                                             .clearValue = clearRoughMetal};

    auto& normalResolve = bindless.descriptorSet->getTextureResource(passResources.normalTextureIndex);
    vk::ClearValue clearNormal{.color = vk::ClearColorValue(std::array<float, 4>{0.5f, 0.5f, 1.0f, 1.0f})};
    vk::RenderingAttachmentInfo normalAttachment = {.imageView = *normalMSAA.view,
                                                     .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                     .resolveMode = vk::ResolveModeFlagBits::eSampleZero,
                                                     .resolveImageView = *normalResolve.imageView,
                                                     .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                     .loadOp = vk::AttachmentLoadOp::eClear,
                                                     .storeOp = vk::AttachmentStoreOp::eStore,
                                                     .clearValue = clearNormal};

    // No resolve here: the prepass already max-resolved depth, and the lit pass draws the same
    // geometry with the same alpha clip, so depth cannot change — re-resolving was a full-res
    // MSAA read+write for identical data every frame.
    vk::RenderingAttachmentInfo depthAttachmentInfo = {.imageView = gpu.getSwapchain().getDepthImageView(),
                                                       .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                                       .resolveMode = vk::ResolveModeFlagBits::eNone,
                                                       .loadOp = vk::AttachmentLoadOp::eLoad,
                                                       .storeOp = vk::AttachmentStoreOp::eDontCare};

    std::array<vk::RenderingAttachmentInfo, 3> colorAttachments = {colorAttachment, roughnessMetalAttachment, normalAttachment};
    vk::RenderingInfo renderingInfo = {.renderArea = {.offset = {0, 0}, .extent = gpu.getSwapchain().getSwapChainExtent()},
                                       .layerCount = 1,
                                       .colorAttachmentCount = 3,
                                       .pColorAttachments = colorAttachments.data(),
                                       .pDepthAttachment = &depthAttachmentInfo};

    cmd.beginRendering(renderingInfo);
    setFullscreenViewport(cmd, gpu.getSwapchain().getSwapChainExtent());

    // skybox
    auto& skyboxPipeline = bindless.pipelineManager->getGeoPipelines()[skyboxPipelineIndex];
    bindPipeline(cmd, *skyboxPipeline);
    SkyBoxPushConstants skyboxConstants = {.skyboxIndex = scene.skyboxIndex, .blur = 0.5, .intensity = features.skyboxIntensity, .invViewProjMatrix = glm::inverse(scene.activeCamera.viewProjection)};
    cmd.pushConstants<SkyBoxPushConstants>(*skyboxPipeline->layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, skyboxConstants);
    cmd.draw(3, 1, 0, 0);

    // lit geometry — reuses the same indirect buffer from the prepass
    if (!indirectCommands.empty()) {
        cmd.bindIndexBuffer(indexBufferHandle, 0, vk::IndexType::eUint32);
        auto& geoPipelines = bindless.pipelineManager->getGeoPipelines();

        LitPushConstants pushConstants = {.vertexBufferAddress    = bindless.descriptorSet->getVariableBuffers()[vertexBufferIndex]->address,
                                          .modelMatricesAddress   = bindless.descriptorSet->getFixedBuffers()[buffers.modelMatrixBufferIndex]->address + static_cast<vk::DeviceSize>(gpu.currentFrame) * MAX_FIXED_BUFFER * sizeof(glm::mat4),
                                          .lightsAddress          = bindless.descriptorSet->getFixedBuffers()[passResources.buffers.lightBufferIndex]->address + static_cast<vk::DeviceSize>(gpu.currentFrame) * MAX_FIXED_BUFFER * sizeof(GPULight),
                                          .litInstanceDataAddress = bindless.descriptorSet->getFixedBuffers()[litInstanceDataBufferIndex]->address + static_cast<vk::DeviceSize>(gpu.currentFrame) * MAX_FIXED_BUFFER * sizeof(LitInstanceData),
                                          .litMeshDrawDataAddress = bindless.descriptorSet->getFixedBuffers()[litMeshDrawDataBufferIndex]->address + static_cast<vk::DeviceSize>(gpu.currentFrame) * MAX_FIXED_BUFFER * sizeof(LitMeshDrawData),
                                          .litPassDataAddress     = bindless.descriptorSet->getFixedBuffers()[litPassDataBufferIndex]->address + static_cast<vk::DeviceSize>(gpu.currentFrame) * MAX_FIXED_BUFFER * sizeof(LitPassData),
                                          .time                   = gpu.time,
                                          .frameUniformsIndex     = litFrameUniformsIndex[gpu.currentFrame]
                                        };

        // Cascade debug needs the uber fragMain; otherwise bind the GI-specialized variant
        // (compile-time giMode -> roughly half the register pressure of the uber shader).
        const bool cascadeDebugActive = scene.anyLightShowsCascades();
        const uint32_t litVariantIndex = (features.vxgi.mode == 1) ? litPipelineGICubeIndex : litPipelineGIConeIndex;

        for (const auto& range : scene.shaderDrawRanges) {
            uint32_t pipelineIdx = range.pipelineIndex;
            if (pipelineIdx == litPipelineIndex && !cascadeDebugActive)
                pipelineIdx = litVariantIndex;
            auto currentPipeline = &(geoPipelines[pipelineIdx]);
            bindPipeline(cmd, **currentPipeline);
            // SV_DrawIndex restarts at 0 for this range's indirect call, but per-draw data is indexed by
            // absolute command index, so offset the shader's lookups by the range's first command.
            pushConstants.drawIDOffset = range.firstCommand;
            cmd.pushConstants<LitPushConstants>((*currentPipeline)->layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pushConstants);
            vk::DeviceSize rangeOffset = frameByteOffset + range.firstCommand * sizeof(DrawIndexedIndirectCommand);
            cmd.drawIndexedIndirect(*litIndirectDrawBuffer, rangeOffset, range.commandCount, sizeof(DrawIndexedIndirectCommand));
        }
    }

    cmd.endRendering();

    // Transition: color resolve to shader readable, roughness-metal to shader readable for SSR.
    // colorResolve is copied into the HDR composite target by recordResolveToCompositeCopy().
    resource::transitionImageLayouts(cmd, {
        {*colorResolve.image,                                                            vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal},
        {*bindless.descriptorSet->getTextureResource(passResources.roughnessMetalTextureIndex).image,  vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal},
    });

    // Generate normal mips inline for SSR pre-filtering
    auto& normalRes = bindless.descriptorSet->getTextureResource(passResources.normalTextureIndex);
    resource::generateMipmaps(*bindless.resourceCtx, *normalRes.image, vk::Format::eR8G8B8A8Unorm,
        static_cast<int32_t>(normalRes.width), static_cast<int32_t>(normalRes.height),
        normalMipLevels, 1, &cmd, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

    // Transition motion vecs to shader read only
    resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *bindless.descriptorSet->getTextureResource(passResources.motionVectorTextureIndex).image,
                                           vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
}

void Renderer::recordBillboardBlendPass(vk::raii::CommandBuffer& cmd, uint32_t imageIndex) {

    if(scene.billboards.empty())
        return;

    auto extent = gpu.getSwapchain().getSwapChainExtent();

    // Depth test is done in-shader by sampling the resolved depth, so make it shader-readable.
    auto& depthResolveTex = bindless.descriptorSet->getTextureResource(gpu.getSwapchain().getDepthResolveIndex());
    resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *depthResolveTex.image,
        vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

    vk::RenderingAttachmentInfo colorAttachment = {.imageView = *bindless.descriptorSet->getTextureResource(passResources.compositeColorTextureIndex).imageView,
                                                   .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                   .loadOp = vk::AttachmentLoadOp::eLoad,
                                                   .storeOp = vk::AttachmentStoreOp::eStore};
    vk::RenderingInfo renderInfo = {.renderArea = {.offset = {0,0}, .extent = extent},
                                    .layerCount = 1,
                                    .colorAttachmentCount = 1,
                                    .pColorAttachments = &colorAttachment};


    std::vector<std::pair<float, GPUBillboard>> depthTested;
    std::vector<std::pair<float, GPUBillboard>> noDepthTest;
    depthTested.reserve(scene.billboards.size());
    noDepthTest.reserve(scene.billboards.size());
    for (auto& kv : scene.billboards) {
        const uint32_t& nodeIdx = kv.first;
        Billboard& billboard = kv.second;
        if(billboard.hidden || !scene.sceneGraph.isNodeValid(billboard.nodeIndex))
            continue;
        glm::vec3 nodePos = scene.sceneGraph.getNode(nodeIdx).getWorldPosition();
        glm::vec3 d = scene.activeCamera.position - nodePos;
        float dist2 = glm::dot(d, d);

        auto& bucket = billboard.depthTest ? depthTested : noDepthTest;
        bucket.emplace_back(dist2, billboard.toGPU(nodePos));
    }
    auto sortFarToNear = [](const auto& a, const auto& b) { return a.first > b.first; };
    std::sort(depthTested.begin(), depthTested.end(), sortFarToNear);
    std::sort(noDepthTest.begin(), noDepthTest.end(), sortFarToNear);

    cmd.beginRendering(renderInfo);
    setFullscreenViewport(cmd, extent);

    uint32_t frameOffset = gpu.currentFrame * MAX_FIXED_BUFFER;
    std::vector<GPUBillboard> billboardWriteBuf;
    billboardWriteBuf.reserve(depthTested.size() + noDepthTest.size());
    for (const auto& p : depthTested) billboardWriteBuf.push_back(p.second);
    for (const auto& p : noDepthTest) billboardWriteBuf.push_back(p.second);
    bindless.descriptorSet->writeFixedBuffer<GPUBillboard>(billboardBufferIndex, billboardWriteBuf.data(), static_cast<uint32_t>(billboardWriteBuf.size()), frameOffset, gpu.currentFrame);

    auto& pipeline = bindless.pipelineManager->getPostProcessPipelines()[billboardPipelineIndex];
    bindPipeline(cmd, *pipeline);

    auto drawGroup = [&](uint32_t groupOffset, uint32_t groupCount, uint32_t depthTest) {
        if (groupCount == 0) return;
        BillboardPushConstants pc = {
            .invViewProj = scene.activeCamera.viewProjection,
            .billboardBufferAddress = bindless.descriptorSet->getFixedBuffers()[billboardBufferIndex]->address + (frameOffset + groupOffset) * sizeof(GPUBillboard),
            .billboardCount = groupCount,
            .samplerIndex = passResources.defaultSamplerIndex,
            .resolution = glm::uvec2(extent.width, extent.height),
            .depthTextureIndex = gpu.getSwapchain().getDepthResolveIndex(),
            .depthSamplerIndex = passResources.depthSamplerIndex,
            .depthTest = depthTest,
        };
        cmd.pushConstants<BillboardPushConstants>(*pipeline->layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pc);
        cmd.draw(6, groupCount, 0, 0);
    };

    drawGroup(0, static_cast<uint32_t>(depthTested.size()), 1);
    drawGroup(static_cast<uint32_t>(depthTested.size()), static_cast<uint32_t>(noDepthTest.size()), 0);

    cmd.endRendering();

    resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *depthResolveTex.image,
        vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eDepthStencilAttachmentOptimal);

}

void Renderer::recordResolveToCompositeCopy(vk::raii::CommandBuffer& cmd) {
    // Seed the HDR composite target with the lit scene color; post passes blend on top of it.
    auto& colorResolve = bindless.descriptorSet->getTextureResource(passResources.colorResolveTextureIndex);
    auto& composite = bindless.descriptorSet->getTextureResource(passResources.compositeColorTextureIndex);
    auto extent = gpu.getSwapchain().getSwapChainExtent();

    resource::transitionImageLayouts(cmd, {
        {*colorResolve.image, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eTransferSrcOptimal},
        {*composite.image,    vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eTransferDstOptimal},
    });

    vk::ImageCopy copyRegion{
        .srcSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .layerCount = 1},
        .dstSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .layerCount = 1},
        .extent = {extent.width, extent.height, 1}
    };
    cmd.copyImage(*colorResolve.image, vk::ImageLayout::eTransferSrcOptimal,
                  *composite.image, vk::ImageLayout::eTransferDstOptimal,
                  copyRegion);

    resource::transitionImageLayouts(cmd, {
        {*colorResolve.image, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eShaderReadOnlyOptimal},
        {*composite.image,    vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eColorAttachmentOptimal},
    });
}

// Meters the lit scene's log-average luminance and adapts toward it over time.
// Returns the 1x1 adapted-luminance texture index for the tonemap pass to sample.
uint32_t Renderer::recordAutoExposure(vk::raii::CommandBuffer& cmd) {
    auto extent = gpu.getSwapchain().getSwapChainExtent();
    auto& avgLum = bindless.descriptorSet->getTextureResource(avgLumTextureIndex);

    // 1. Extract log2(luminance) of the lit scene into avgLum mip 0.
    resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *avgLum.image,
        vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eColorAttachmentOptimal, 0, 1);
    drawFullscreenPass(cmd, *bindless.pipelineManager->getPostProcessPipelines()[lumExtractPipelineIndex],
        *avgLumMip0View, extent,
        LumExtractPushConstants{.inputTextureIndex = passResources.colorResolveTextureIndex, .samplerIndex = passResources.defaultSamplerIndex});

    // 2. Box-average down to 1x1 (the geometric mean, since values are log2). Leaves all mips shader-readable.
    resource::generateMipmaps(*bindless.resourceCtx, *avgLum.image, vk::Format::eR16Sfloat,
        avgLum.width, avgLum.height, passResources.fullscreenMipLevels, 1, &cmd,
        vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

    // 3. Temporally adapt toward the metered value (ping-pong since we read prev + write new).
    uint32_t readIdx = adaptedLumIndex[adaptFlip];
    uint32_t writeIdx = adaptedLumIndex[1 - adaptFlip];
    auto& writeTex = bindless.descriptorSet->getTextureResource(writeIdx);

    float dt = std::clamp(gpu.time - autoExposurePrevTime, 0.0f, 0.1f);
    autoExposurePrevTime = gpu.time;

    resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *writeTex.image,
        vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eColorAttachmentOptimal);
    drawFullscreenPass(cmd, *bindless.pipelineManager->getPostProcessPipelines()[exposureAdaptPipelineIndex],
        *writeTex.imageView, vk::Extent2D{1, 1},
        ExposureAdaptPushConstants{
            .currentLumIndex = avgLumTextureIndex,
            .currentLumMip = passResources.fullscreenMipLevels - 1,
            .prevAdaptedIndex = readIdx,
            .samplerIndex = passResources.defaultSamplerIndex,
            .dt = dt,
            .speed = features.tonemap.adaptationSpeed,
            .initialized = adaptInitialized ? 1u : 0u,
        });
    resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *writeTex.image,
        vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

    adaptFlip = 1 - adaptFlip;
    adaptInitialized = true;
    return writeIdx;
}

void Renderer::recordTonemapPass(vk::raii::CommandBuffer& cmd, uint32_t imageIndex) {
    auto extent = gpu.getSwapchain().getSwapChainExtent();
    auto& composite = bindless.descriptorSet->getTextureResource(passResources.compositeColorTextureIndex);

    uint32_t lumIndex = avgLumTextureIndex; // unused when auto off
    if (features.tonemap.autoExposure) {
        lumIndex = recordAutoExposure(cmd);
    }

    // Composite finished as a color attachment; make it sampleable, then resolve to the swapchain.
    resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *composite.image,
        vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

    drawFullscreenPass(cmd, *bindless.pipelineManager->getPostProcessPipelines()[tonemapPipelineIndex],
        *gpu.getSwapchain().getSwapChainImageViews()[imageIndex], extent,
        TonemapPushConstants{
            .hdrTextureIndex = passResources.compositeColorTextureIndex,
            .samplerIndex = passResources.defaultSamplerIndex,
            .exposure = computeExposure(features.tonemap),
            .op = features.tonemap.op,
            .autoExposure = features.tonemap.autoExposure ? 1u : 0u,
            .lumTextureIndex = lumIndex,
            .lumMipLevel = 0,
            .exposureComp = features.tonemap.exposureComp,
            .minEV = features.tonemap.minEV,
            .maxEV = features.tonemap.maxEV,
        });
}

void Renderer::recordOverlayPass(vk::raii::CommandBuffer& cmd, uint32_t imageIndex) {
    auto extent = gpu.getSwapchain().getSwapChainExtent();
    vk::RenderingAttachmentInfo colorAttachment = {.imageView = gpu.getSwapchain().getSwapChainImageViews()[imageIndex],
                                                   .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                   .loadOp = vk::AttachmentLoadOp::eLoad,
                                                   .storeOp = vk::AttachmentStoreOp::eStore};
    vk::RenderingInfo renderInfo = {.renderArea = {.offset = {0, 0}, .extent = extent},
                                    .layerCount = 1,
                                    .colorAttachmentCount = 1,
                                    .pColorAttachments = &colorAttachment};

    cmd.beginRendering(renderInfo);
    setFullscreenViewport(cmd, extent);

    // gizmos
    if(features.showGizmos){
        for(auto& line : Gizmos::getNoDiscardLines()){
            Gizmos::drawLine(line.second);
        }
        auto& gizmoPipeline = bindless.pipelineManager->getPostProcessPipelines()[gizmoPipelineIndex];
        bindPipeline(cmd, *gizmoPipeline);
        LinePushConstants lineConstants = {.lineVertsAddress = Gizmos::getLineBufferAddress(),
                                        .depthTextureIndex = gpu.getSwapchain().getDepthResolveIndex(),
                                        .depthSamplerIndex = passResources.depthSamplerIndex,
                                        .viewProjection = scene.activeCamera.viewProjection};
        cmd.pushConstants<LinePushConstants>(*gizmoPipeline->layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, lineConstants);
        cmd.draw(Gizmos::getVertexCount(), 1, 0, 0);
    }
    // the whole overlay: one instanced draw for every rect and glyph on screen
    gui.record(bindless, cmd, gpu.currentFrame, glm::uvec2(extent.width, extent.height));
    cmd.endRendering();
}

void Renderer::recordImageVisPass(vk::raii::CommandBuffer& cmd, uint32_t imageIndex) {
    if (features.imageVis.imageIndex == 0xFFFFFFFF)
        return;

    auto extent = gpu.getSwapchain().getSwapChainExtent();
    auto& visTexture = bindless.descriptorSet->getTextureResource(features.imageVis.imageIndex);
    float imgAspect = (visTexture.width > 0 && visTexture.height > 0)
                          ? static_cast<float>(visTexture.width) / static_cast<float>(visTexture.height)
                          : static_cast<float>(extent.width) / static_cast<float>(extent.height);

    drawFullscreenPass(cmd, *bindless.pipelineManager->getPostProcessPipelines()[imageViewPipelineIndex], *gpu.getSwapchain().getSwapChainImageViews()[imageIndex],
                       extent,
                       ImageVisPushConstants{.imageIndex = features.imageVis.imageIndex,
                                             .samplerIndex = passResources.defaultSamplerIndex,
                                             .flags = features.imageVis.flags,
                                             .nearPlane = scene.activeCamera.nearPlane,
                                             .farPlane = scene.activeCamera.farPlane,
                                             .imageAspect = imgAspect,
                                             .screenAspect = static_cast<float>(extent.width) / static_cast<float>(extent.height),
                                             .mipLevel = features.imageVis.mipLevel},
                       vk::AttachmentLoadOp::eLoad);
}

void Renderer::recordShadowPass(vk::raii::CommandBuffer& cmd, Light& light, uint32_t shadowSlot) {
    auto& currentPipeline = bindless.pipelineManager->getBeforeGeoPipelines()[shadowPipelineIndex];
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, currentPipeline->pipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, currentPipeline->layout, 0, {*currentPipeline->descriptorSet}, {});
    cmd.bindIndexBuffer(bindless.descriptorSet->getVariableBuffer(indexBufferIndex), 0, vk::IndexType::eUint32);

    // Determine face count up front — draw data is baked per-face below.
    uint32_t faceCount = 0;
    if (light.type == LightType::Directional) {
        faceCount = light.numCascades + 1; // +1 for VXGI radiance as it needs to encapsulate a different view of the scene
    } else if (light.type == LightType::Point) {
        faceCount = 6;
    }

    // Build indirect draw commands once — identical across all faces/cascades since culling is off.
    // Capture per-instance world transforms so we can re-bake `lightSpaceMatrix * worldTransform` per face
    // without re-walking the scene graph.
    std::array<Plane, 6> dummyPlanes{};
    shadowMeshDrawDataList.clear();
    shadowInstanceDataList.clear();
    // Point lights only need to draw casters inside their range — LightInfluence
    // keeps that set current, so we filter by it here. Directional lights skip
    // the filter (they affect all geometry).
    std::function<bool(const Node&)> nodeFilter;
    if (light.type == LightType::Point) {
        nodeFilter = [&light](const Node& n) { return light.influencedNodes.count(n.nodeIndex) != 0; };
    }
    std::vector<glm::mat4> instanceWorldTransforms;
    buildGeometryDrawCommands(dummyPlanes, false,
        [&](const Mesh& mesh, Node& node, const auto& material) {
            shadowMeshDrawDataList.push_back({.positionBufferOffset = mesh.positionOffset,
                                              .positionBufferStride = static_cast<uint32_t>(sizeof(glm::vec3))});
        },
        [&](const Mesh& mesh, Node& node, const Material& material) {
            instanceWorldTransforms.push_back(node.worldTransform);
        },
        nodeFilter);

    uint32_t instancesPerFace = static_cast<uint32_t>(instanceWorldTransforms.size());

    // Backfill firstInstance into each ShadowMeshDrawData so the shadow shader can compute the
    // absolute instance index as mesh.firstInstance + SV_InstanceID (driver/Slang-agnostic).
    for (size_t i = 0; i < indirectCommands.size() && i < shadowMeshDrawDataList.size(); ++i) {
        shadowMeshDrawDataList[i].firstInstance = indirectCommands[i].firstInstance;
    }

    // Bake per-face MMxLSM into shadowInstanceDataList laid out as [face0 instances..., face1 instances..., ...].
    shadowInstanceDataList.reserve(static_cast<size_t>(instancesPerFace) * faceCount);
    for (uint32_t f = 0; f < faceCount; ++f) {
        glm::mat4 lsm;
        if(light.type == LightType::Directional) {
            if(f < light.numCascades) {
                lsm = light.cascades[f].lightSpaceMatrix;
            } else {
                lsm = light.shadowMaps[0].lightSpaceMatrix;
            }
        } else {
            lsm = light.shadowMaps[f].lightSpaceMatrix;
        }
        for (const auto& wt : instanceWorldTransforms) {
            shadowInstanceDataList.push_back({.MMxLSM = lsm * wt});
        }
    }

    // Per-light slot within the frame so multiple shadow-casting lights don't stomp on each
    // other's indirect commands / draw data before the GPU reads them.
    uint32_t slotIdx = gpu.currentFrame * MAX_SHADOW_CASTERS + shadowSlot;
    vk::DeviceSize frameByteOffset = static_cast<vk::DeviceSize>(slotIdx) * MAX_INDIRECT_COMMANDS * sizeof(DrawIndexedIndirectCommand);

    // Upload indirect commands, mesh draw data (one per group), and per-face baked instance data
    if (!indirectCommands.empty()) {
        memcpy(static_cast<char*>(indirectDrawBufferMapped) + frameByteOffset, indirectCommands.data(), indirectCommands.size() * sizeof(DrawIndexedIndirectCommand));

        if (shadowInstanceDataList.size() > MAX_FIXED_BUFFER) {
            std::cerr << "Warning: shadow instance data (" << shadowInstanceDataList.size()
                      << ") exceeds MAX_FIXED_BUFFER (" << MAX_FIXED_BUFFER << "); truncating" << std::endl;
        }
        if (shadowMeshDrawDataList.size() > MAX_FIXED_BUFFER) {
            std::cerr << "Warning: shadow mesh draw data (" << shadowMeshDrawDataList.size()
                      << ") exceeds MAX_FIXED_BUFFER (" << MAX_FIXED_BUFFER << "); truncating" << std::endl;
        }
        uint32_t slotElementOffset = slotIdx * MAX_FIXED_BUFFER;
        uint32_t instanceCopyCount = static_cast<uint32_t>(std::min<size_t>(shadowInstanceDataList.size(), MAX_FIXED_BUFFER));
        uint32_t meshCopyCount     = static_cast<uint32_t>(std::min<size_t>(shadowMeshDrawDataList.size(), MAX_FIXED_BUFFER));
        bindless.descriptorSet->writeFixedBuffer<ShadowInstanceData>(shadowInstanceDataBufferIndex, shadowInstanceDataList.data(), instanceCopyCount, slotElementOffset, gpu.currentFrame);
        bindless.descriptorSet->writeFixedBuffer<ShadowMeshDrawData>(shadowMeshDrawDataBufferIndex, shadowMeshDrawDataList.data(), meshCopyCount, slotElementOffset, gpu.currentFrame);
    }

    // Bind the atlas once; each face/cascade is rendered into its tile via viewport+scissor.
    auto& atlasTex = bindless.descriptorSet->getTextureResource(scene.shadowAtlas.textureIndex);
    vk::RenderingAttachmentInfo depthAttachment{.imageView   = *atlasTex.imageView,
                                                .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                                .loadOp      = vk::AttachmentLoadOp::eLoad,
                                                .storeOp     = vk::AttachmentStoreOp::eStore};
    vk::RenderingInfo renderInfo{.renderArea           = {{0, 0}, {SHADOW_ATLAS_SIZE, SHADOW_ATLAS_SIZE}},
                                 .layerCount           = 1,
                                 .colorAttachmentCount = 0,
                                 .pColorAttachments    = nullptr,
                                 .pDepthAttachment     = &depthAttachment};
    cmd.beginRendering(renderInfo);

    for (uint32_t i = 0; i < faceCount; i++) {
        glm::vec4 uvRange;
        if (light.type == LightType::Directional) {
            if(i < light.numCascades) {
                uvRange = light.cascades[i].shadowAtlasUVRange;
            } else {
                uvRange = light.shadowMaps[0].shadowAtlasUVRange;
            }
        } else {
            uvRange = light.shadowMaps[i].shadowAtlasUVRange;
        }

        int32_t  tx = static_cast<int32_t>(uvRange.x * SHADOW_ATLAS_SIZE);
        int32_t  ty = static_cast<int32_t>(uvRange.y * SHADOW_ATLAS_SIZE);
        uint32_t tw = static_cast<uint32_t>((uvRange.z - uvRange.x) * SHADOW_ATLAS_SIZE);
        uint32_t th = static_cast<uint32_t>((uvRange.w - uvRange.y) * SHADOW_ATLAS_SIZE);
        if (tw == 0 || th == 0) continue;

        vk::Rect2D tileRect{{tx, ty}, {tw, th}};

        // Clear just this tile (the atlas was loaded, not cleared).
        vk::ClearAttachment clearInfo{.aspectMask = vk::ImageAspectFlagBits::eDepth,
                                      .colorAttachment = 0,
                                      .clearValue = vk::ClearValue{.depthStencil = vk::ClearDepthStencilValue{1.0f, 0}}};
        vk::ClearRect clearRect{.rect = tileRect, .baseArrayLayer = 0, .layerCount = 1};
        cmd.clearAttachments(clearInfo, clearRect);

        cmd.setViewport(0, vk::Viewport(static_cast<float>(tx), static_cast<float>(ty),
                                        static_cast<float>(tw), static_cast<float>(th), 0.0f, 1.0f));
        cmd.setScissor(0, tileRect);

        if (!indirectCommands.empty()) {
            // Instance pointer slides into this face's [face i] block; mesh pointer is shared across faces.
            ShadowPushConstants pushConstants = {
                .positionBufferAddress      = bindless.descriptorSet->getVariableBuffers()[positionBufferIndex]->address,
                .shadowInstanceDataAddress  = bindless.descriptorSet->getFixedBuffers()[shadowInstanceDataBufferIndex]->address
                                              + (static_cast<vk::DeviceSize>(slotIdx) * MAX_FIXED_BUFFER + static_cast<vk::DeviceSize>(i) * instancesPerFace) * sizeof(ShadowInstanceData),
                .shadowMeshDrawDataAddress  = bindless.descriptorSet->getFixedBuffers()[shadowMeshDrawDataBufferIndex]->address
                                              + static_cast<vk::DeviceSize>(slotIdx) * MAX_FIXED_BUFFER * sizeof(ShadowMeshDrawData),
            };
            cmd.pushConstants<ShadowPushConstants>(*currentPipeline->layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pushConstants);

            cmd.drawIndexedIndirect(*indirectDrawBuffer, frameByteOffset, static_cast<uint32_t>(indirectCommands.size()), sizeof(DrawIndexedIndirectCommand));
        }
    }

    cmd.endRendering();
}

void Renderer::recordSDFPass(vk::raii::CommandBuffer& cmd, uint32_t imageIndex) {
    if (sdfTextureIndex == 0xFFFFFFFF)  
        return;

    auto extent = gpu.getSwapchain().getSwapChainExtent();

    auto& sdfAllocations = bindless.descriptorSet->getFixedBufferAllocations(sdfPassDataBufferIndex);
    uint32_t sdfCount = 0;
    for (const auto& alloc : sdfAllocations) {
        if (alloc.inUse) sdfCount++;
    }

    if (sdfCount == 0) return;

    drawFullscreenPass(cmd,*bindless.pipelineManager->getPostProcessPipelines()[sdfPipelineIndex], *bindless.descriptorSet->getTextureResource(sdfTextureIndex).imageView,
                       extent,
                       SDFPushConstants{.sdfDataAddress = bindless.descriptorSet->getFixedBuffers()[sdfPassDataBufferIndex]->address,
                                        .sdfCount = sdfCount,
                                        .depthTextureIndex = gpu.getSwapchain().getDepthResolveIndex(),
                                        .depthSamplerIndex = passResources.depthSamplerIndex,
                                        .cameraPos = scene.activeCamera.position,
                                        .invViewProjection = glm::inverse(scene.activeCamera.viewProjection)
                                        }, vk::AttachmentLoadOp::eClear);

    drawFullscreenPass(cmd,*bindless.pipelineManager->getPostProcessPipelines()[sdfApplyPipelineIndex], *bindless.descriptorSet->getTextureResource(passResources.compositeColorTextureIndex).imageView,
                       extent,
                       SDFApplyPushConstants{.sdfTextureIndex = sdfTextureIndex,
                                             .samplerIndex = passResources.defaultSamplerIndex},
                                            vk::AttachmentLoadOp::eLoad);
}

void Renderer::createOrResizeRenderTarget(uint32_t& index, uint32_t width, uint32_t height, vk::Format format, const char* debugName,
                                          vk::ImageUsageFlags extraUsage) {
    if (index != 0xFFFFFFFF) {
        bindless.descriptorSet->freeTexture(index);
    }
    vk::raii::Image image = nullptr;
    vk::raii::DeviceMemory memory = nullptr;
    resource::createImage(*bindless.resourceCtx, width, height, 1, vk::SampleCountFlagBits::e1, format, vk::ImageTiling::eOptimal,
                                 vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled | extraUsage,
                                 vk::MemoryPropertyFlagBits::eDeviceLocal, image, memory, 1);
    auto view = resource::createImageView(*bindless.resourceCtx, image, format, vk::ImageAspectFlagBits::eColor);
    resource::transitionImageLayout(*bindless.resourceCtx, nullptr, image, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal);
    index = bindless.descriptorSet->allocateTexture(std::move(image), std::move(memory), std::move(view), debugName, false, width, height);
}

void Renderer::createOrResize3DStorageImage(uint32_t& textureIndex, uint32_t& storageIndex, uint32_t width, uint32_t height, uint32_t depth, vk::Format format,
                                          const char* debugName, vk::ImageUsageFlags extraUsage) {
    // Recycle old slots (device is idle here — init/resize only).
    if (storageIndex != 0xFFFFFFFF) bindless.descriptorSet->freeStorageImage(storageIndex);
    if (textureIndex != 0xFFFFFFFF) bindless.descriptorSet->freeTexture(textureIndex);

    vk::raii::Image image = nullptr;
    vk::raii::DeviceMemory memory = nullptr;
    resource::create3DImage(*bindless.resourceCtx, width, height, depth, 1, vk::SampleCountFlagBits::e1, format, vk::ImageTiling::eOptimal,
                                 vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | extraUsage,
                                 vk::MemoryPropertyFlagBits::eDeviceLocal, image, memory, 1);
    auto view = resource::create3DImageView(*bindless.resourceCtx, image, format, vk::ImageAspectFlagBits::eColor);
    // Sampled slot reads it in eShaderReadOnlyOptimal; compute passes transition to eGeneral before writing.
    resource::transitionImageLayout(*bindless.resourceCtx, nullptr, image, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal);

    // Register the same view twice: once sampled (owns image/memory/view), once as a storage descriptor.
    vk::ImageView rawView = *view;
    textureIndex = bindless.descriptorSet->allocateTexture(std::move(image), std::move(memory), std::move(view), debugName, false, width, height);
    storageIndex = bindless.descriptorSet->allocateStorageImage(rawView);
}

void Renderer::createOrResizeMSAATarget(Image& target, uint32_t width, uint32_t height, vk::Format format) {
    target.view = nullptr; // destroy view before image
    resource::createImage(*bindless.resourceCtx, width, height, 1, gpu.getMsaaSamples(),format, vk::ImageTiling::eOptimal,
                                 vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment,
                                 vk::MemoryPropertyFlagBits::eDeviceLocal, target.image, target.memory);
    target.view = resource::createImageView(*bindless.resourceCtx, target.image, format, vk::ImageAspectFlagBits::eColor, 1);
    resource::transitionImageLayout(*bindless.resourceCtx, nullptr, target.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal);
}

void Renderer::setFullscreenViewport(vk::raii::CommandBuffer& cmd, vk::Extent2D extent) {
    cmd.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.0f, 1.0f));
    cmd.setScissor(0, vk::Rect2D({0, 0}, extent));
}

void Renderer::bindPipeline(vk::raii::CommandBuffer& cmd, PipelineBase& pipeline) {
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.pipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline.layout, 0, {**pipeline.descriptorSet}, {});
}

void Renderer::bindComputePipeline(vk::raii::CommandBuffer& cmd, ComputePipelineBase& pipeline) {
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline.layout, 0, {**pipeline.descriptorSet}, {});
}

template <typename T>
void Renderer::drawFullscreenPass(vk::raii::CommandBuffer& cmd, PipelineBase& pipeline, vk::ImageView targetView, vk::Extent2D extent,
                                  const T& pushConstants, vk::AttachmentLoadOp loadOp, std::array<float, 4> clearColor) {

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


glm::mat4 calculateLightSpaceMatrix(Light& light, Camera& camera) {

    glm::mat4 lightProjection = glm::ortho(-light.range, light.range, -light.range, light.range, camera.nearPlane, camera.farPlane);
    glm::vec3 lightPos = camera.position - light.direction * 0.5f * camera.farPlane;
    glm::mat4 lightView = glm::lookAt(lightPos, lightPos + light.direction, glm::vec3(0.0f, 1.0f, 0.0f));
    return lightProjection * lightView;
}

void calculatePointLightFaceMatrices(Light& light, const glm::vec3& lightPos) {
    float nearPlane = 0.05f;
    float farPlane = light.range;
    glm::mat4 projection = glm::perspective(glm::radians(90.0f), 1.0f, nearPlane, farPlane);

    // 6 cubemap faces: +X, -X, +Y, -Y, +Z, -Z
    const glm::vec3 directions[6] = {
        { 1.0f,  0.0f,  0.0f},  // +X
        {-1.0f,  0.0f,  0.0f},  // -X
        { 0.0f,  1.0f,  0.0f},  // +Y
        { 0.0f, -1.0f,  0.0f},  // -Y
        { 0.0f,  0.0f,  1.0f},  // +Z
        { 0.0f,  0.0f, -1.0f},  // -Z
    };
    const glm::vec3 ups[6] = {
        { 0.0f, -1.0f,  0.0f},  // +X
        { 0.0f, -1.0f,  0.0f},  // -X
        { 0.0f,  0.0f,  1.0f},  // +Y
        { 0.0f,  0.0f, -1.0f},  // -Y
        { 0.0f, -1.0f,  0.0f},  // +Z
        { 0.0f, -1.0f,  0.0f},  // -Z
    };

    for (int i = 0; i < 6; i++) {
        glm::mat4 view = glm::lookAt(lightPos, lightPos + directions[i], ups[i]);
        light.shadowMaps[i].lightSpaceMatrix = projection * view;
    }
}

void calculateCascadedLightSpaceMatrices(Light& light, Camera& camera, Renderer* renderer) {
    glm::vec3 lightDir = glm::normalize(light.direction);

    // Stable up vector that avoids degeneracy when light is near-vertical
    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
    if (glm::abs(glm::dot(lightDir, up)) > 0.99f) {
        up = glm::vec3(1.0f, 0.0f, 0.0f);
    }
    // Unproject the full camera frustum to world space (Vulkan NDC: z in [0,1])
    // Reverse-Z: near is at NDC z=1, far at z=0. Use (1-z) so indices 0-3 = near plane, 4-7 = far plane.
    glm::mat4 invCamVP = glm::inverse(camera.viewProjection);
    glm::vec3 fullCorners[8];
    int idx = 0;
    for (int z = 0; z < 2; ++z)
        for (int y = 0; y < 2; ++y)
            for (int x = 0; x < 2; ++x) {
                glm::vec4 c = invCamVP * glm::vec4(2.f * x - 1.f, 2.f * y - 1.f, static_cast<float>(1 - z), 1.f);
                fullCorners[idx++] = glm::vec3(c / c.w);
            }

    float lastSplitDist = 0.0f;

    for (uint32_t i = 0; i < light.numCascades; i++) {

        // Cascade splits are in world-space units
        float splitDist;
        if (light.cascades[i].splitDistance > 0.0f) {
            splitDist = (light.cascades[i].splitDistance - camera.nearPlane) / (camera.farPlane - camera.nearPlane);
            splitDist = glm::clamp(splitDist, lastSplitDist + 0.001f, 1.0f);
        } else {
            splitDist = static_cast<float>(i + 1) / static_cast<float>(light.numCascades);
            light.cascades[i].splitDistance = camera.nearPlane + splitDist * (camera.farPlane - camera.nearPlane);
        }

        // Slice the full frustum into this cascade's sub-frustum
        glm::vec3 corners[8];
        for (int j = 0; j < 4; j++) {
            glm::vec3 ray = fullCorners[j + 4] - fullCorners[j];
            corners[j]     = fullCorners[j] + ray * lastSplitDist;
            corners[j + 4] = fullCorners[j] + ray * splitDist;
        }

        // Sub-frustum center
        glm::vec3 center(0.0f);
        for (const auto& c : corners) center += c;
        center /= 8.0f;

        // Build light view matrix looking at the frustum center
        float zPullBack = 500.0f;
        glm::mat4 lightView = glm::lookAt(
            center - lightDir * zPullBack,
            center,
            up
        );

        // Compute tight AABB in light space from the frustum corners
        glm::vec3 lsMin(std::numeric_limits<float>::max());
        glm::vec3 lsMax(std::numeric_limits<float>::lowest());
        for (const auto& c : corners) {
            glm::vec3 ls = glm::vec3(lightView * glm::vec4(c, 1.0f));
            lsMin = glm::min(lsMin, ls);
            lsMax = glm::max(lsMax, ls);
        }

        float extentX = lsMax.x - lsMin.x;
        float extentY = lsMax.y - lsMin.y;
        float maxExtent = glm::max(extentX, extentY);

        // Expand AABB for cascade overlap
        float overlapMargin = maxExtent * 0.1f;
        lsMin.x -= overlapMargin;
        lsMin.y -= overlapMargin;
        lsMax.x += overlapMargin;
        lsMax.y += overlapMargin;

        // Near=0.1 captures shadow casters between the light eye and the frustum.
        // Far extends just past the farthest frustum corner in light space.
        float orthoNear = 0.1f;
        float orthoFar  = -lsMin.z + 10.0f;

        glm::mat4 lightProj = glm::ortho(
            lsMin.x, lsMax.x,
            lsMin.y, lsMax.y,
            orthoNear, orthoFar
        );

        // Snap the shadow matrix translation to texel boundaries so the
        // shadow map stays locked to a fixed world-space grid as the camera moves.
        glm::mat4 shadowMatrix = lightProj * lightView;
        float halfRes = static_cast<float>(light.shadowResolution) * 0.5f;
        glm::vec4 shadowOrigin = shadowMatrix * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        shadowOrigin *= halfRes;
        glm::vec4 rounded = glm::round(shadowOrigin);
        glm::vec4 roundOffset = rounded - shadowOrigin;
        roundOffset /= halfRes;
        lightProj[3][0] += roundOffset.x;
        lightProj[3][1] += roundOffset.y;

        float finalExtent = glm::max(lsMax.x - lsMin.x, lsMax.y - lsMin.y);
        light.cascades[i].lightSpaceMatrix = lightProj * lightView;
        light.cascades[i].texelSize = 1.0f / static_cast<float>(light.shadowResolution);
        light.cascades[i].worldTexelSize = finalExtent / static_cast<float>(light.shadowResolution);

        lastSplitDist = splitDist;
    }

    // shadowMaps[0] is the VXGI shadow map: tight ortho bounds around the voxel volume
    // (the camera-snapped cube VoxelizationPass rasterizes) instead of a camera sub-frustum.
    {
        glm::vec3 gridCenter = VoxelizationPass::snappedGridCenter(camera.position);
        constexpr float half = VoxelizationPass::VOXEL_WORLD_EXTENT * 0.5f;

        float zPullBack = 500.0f;
        glm::mat4 lightView = glm::lookAt(gridCenter - lightDir * zPullBack, gridCenter, up);

        // Light-space AABB of the volume's 8 corners
        glm::vec3 lsMin(std::numeric_limits<float>::max());
        glm::vec3 lsMax(std::numeric_limits<float>::lowest());
        for (int j = 0; j < 8; j++) {
            glm::vec3 corner = gridCenter + half * glm::vec3(j & 1 ? 1.0f : -1.0f,
                                                             j & 2 ? 1.0f : -1.0f,
                                                             j & 4 ? 1.0f : -1.0f);
            glm::vec3 ls = glm::vec3(lightView * glm::vec4(corner, 1.0f));
            lsMin = glm::min(lsMin, ls);
            lsMax = glm::max(lsMax, ls);
        }

        // Same near/far convention as the cascades: near starts at the light eye to catch
        // casters outside the volume, far ends just past its back face.
        glm::mat4 lightProj = glm::ortho(lsMin.x, lsMax.x, lsMin.y, lsMax.y, 0.1f, -lsMin.z + 10.0f);

        // Texel snap at the VXGI tile resolution — the grid snaps in voxel steps, not texel steps.
        glm::mat4 shadowMatrix = lightProj * lightView;
        float halfRes = static_cast<float>(VXGI_DIRECTIONAL_SHADOW_RESOLUTION) * 0.5f;
        glm::vec4 shadowOrigin = shadowMatrix * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f) * halfRes;
        glm::vec4 roundOffset = (glm::round(shadowOrigin) - shadowOrigin) / halfRes;
        lightProj[3][0] += roundOffset.x;
        lightProj[3][1] += roundOffset.y;

        light.shadowMaps[0].lightSpaceMatrix = lightProj * lightView;
    }
}