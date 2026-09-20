#pragma once
#include "render_pass.hpp"
#include "bindless_system.hpp"
#include "profiling.hpp"
#include "scene_elements.hpp"
#include "scene.hpp"

#include <cstring>

#ifndef VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_NO_CONSTRUCTORS 1 // for structs constructors
#endif


// Voxelizes the scene into a clipmap of RGBA8 volumes in three stages:
//
//   [raster scatter] -> [resolve to mip 0] -> [downsample mip chain]
//
// A clip level is a CLIP_RESOLUTION^3 cube around the camera whose voxels are 2^level times the size
// of level 0's, so each level out doubles the covered radius for the same voxel count. Every volume
// (radiance, the twelve irradiance faces) is one 3D texture with the levels stacked along Z; see
// shaders/modules/voxel_clipmap.slang for the layout. The raster stage does per-triangle dominant-axis
// selection in a geometry shader and runs once per level, with coarser mesh LODs on coarser levels
// and that level's own VXGI shadow tile. The gather runs per level too, outer levels amortized
// harder; cones and the lit pass's lookup climb through the levels as they need coverage.
class VoxelizationPass : public RenderPass {
public:
    static constexpr uint32_t CLIP_RESOLUTION       = VOXEL_CLIP_RESOLUTION; // radiance voxels a side, per level
    static constexpr uint32_t IRRADIANCE_RESOLUTION = 64;  // irradiance voxels a side, per level
    static constexpr uint32_t MIPS_PER_LEVEL        = 4;   // mips 0..3 — the range traceCone reads (CONE_MAX_MIP)
    static constexpr uint32_t STACKED_LEVELS        = MAX_VOXEL_CLIP_LEVELS; // slabs allocated per volume
    // Level centres snap to this many of their own voxels (== CLIP_SNAP_VOXELS in voxel_clipmap.slang).
    // Keeps mips 0-3 phase-stable in world space; the shaders measure a level's reach from the camera
    // minus this step, so a recentre never changes what a point reads.
    static constexpr uint32_t SNAP_VOXELS           = 8;
    static_assert(CLIP_RESOLUTION % IRRADIANCE_RESOLUTION == 0, "gather maps irradiance voxels onto whole radiance mips");
    static_assert(((CLIP_RESOLUTION / IRRADIANCE_RESOLUTION) & (CLIP_RESOLUTION / IRRADIANCE_RESOLUTION - 1)) == 0, "irradiance ratio must be a radiance mip");
    static_assert(CLIP_RESOLUTION % (1u << (MIPS_PER_LEVEL - 1)) == 0, "level slabs must stay 2x2x2-aligned through every mip");
    // The snap must land on whole irradiance voxels, or the gather's history offset stops being an
    // exact integer and every recentre throws the temporal history away.
    static_assert(SNAP_VOXELS * IRRADIANCE_RESOLUTION % CLIP_RESOLUTION == 0, "grid snap must be a whole number of irradiance voxels");
    // Levels snap independently; level l-1 stays inside level l while the snap step is under a
    // quarter of the level extent.
    static_assert(SNAP_VOXELS * 4 <= CLIP_RESOLUTION, "snap step must be under a quarter of the level extent for the levels to nest");

    static uint32_t activeLevels(const RenderFeatures& f) {
        return static_cast<uint32_t>(std::clamp(f.vxgi.clipLevels, 1, static_cast<int>(STACKED_LEVELS)));
    }
    static float levelExtent(const RenderFeatures& f, uint32_t level) {
        return f.vxgi.voxelSize0 * static_cast<float>(CLIP_RESOLUTION) * static_cast<float>(1u << level);
    }
    // Level centre snapped to SNAP_VOXELS of its voxels, which is also a whole number of irradiance voxels.
    static glm::vec3 levelCenter(const RenderFeatures& f, const glm::vec3& cameraPos, uint32_t level) {
        float snap = static_cast<float>(SNAP_VOXELS) * f.vxgi.voxelSize0 * static_cast<float>(1u << level);
        return glm::floor(cameraPos / snap) * snap;
    }
    // Deterministic from the camera and settings, so the renderer rebuilds it for the lit pass rather
    // than depending on record() having run first.
    static GPUVoxelClipmap buildClipmap(const RenderFeatures& f, const glm::vec3& cameraPos) {
        GPUVoxelClipmap cm{};
        cm.levelCount = activeLevels(f);
        cm.resolution = CLIP_RESOLUTION;
        cm.irradianceResolution = IRRADIANCE_RESOLUTION;
        cm.mipLevels = MIPS_PER_LEVEL;
        cm.anchor = glm::vec4(cameraPos, 0.0f);
        for (uint32_t l = 0; l < cm.levelCount; l++)
            cm.levels[l] = glm::vec4(levelCenter(f, cameraPos, l), levelExtent(f, l));
        return cm;
    }

    // World -> level clip: xy in [-1,1], z in [0,1], grid axes = world axes. A pure scale/translate
    // rather than a lookAt, so level-local UVW is (world - centre) / extent + 0.5 everywhere and the
    // frustum-plane extraction still sees an ordinary ZO ortho matrix.
    static glm::mat4 levelViewProjection(const glm::vec4& level) {
        glm::vec3 c = glm::vec3(level);
        float e = level.w;
        glm::mat4 m(0.0f);
        m[0][0] = 2.0f / e;
        m[1][1] = 2.0f / e;
        m[2][2] = 1.0f / e;
        m[3] = glm::vec4(-2.0f * c.x / e, -2.0f * c.y / e, 0.5f - c.z / e, 1.0f);
        return m;
    }
    // Same mapping as clipLevelUVW() in voxel_clipmap.slang.
    static glm::vec3 levelUVW(const glm::vec4& level, const glm::vec3& worldPos) {
        return (worldPos - glm::vec3(level)) / level.w + 0.5f;
    }

private:
    static constexpr vk::Format VOXEL_VOLUME_FORMAT = vk::Format::eR8G8B8A8Unorm;
    static constexpr vk::DeviceSize LEVEL_VOXELS = static_cast<vk::DeviceSize>(CLIP_RESOLUTION) * CLIP_RESOLUTION * CLIP_RESOLUTION;
    static constexpr vk::DeviceSize LEVEL_SCATTER_BYTES = LEVEL_VOXELS * sizeof(uint32_t);
    static constexpr uint32_t WG = 4; // [numthreads(4,4,4)] in every voxel compute kernel

    // Scatter targets: uint[STACKED_LEVELS * CLIP_RESOLUTION^3] each, one contiguous block per level.
    // The active levels are cleared every frame.
    uint32_t albedoBufferIndex   = 0xFFFFFFFF;
    uint32_t radianceBufferIndex = 0xFFFFFFFF;

    // This frame's levels: built at the top of record(), mirrored into clipmapBufferIndex (one slot
    // per frame in flight) for the compute passes. The lit pass gets its copy through LitFrameUniforms.
    GPUVoxelClipmap clipmap{};
    std::array<glm::mat4, STACKED_LEVELS> levelVPM{};    // world -> level clip, for the raster and debug views
    std::array<glm::mat4, STACKED_LEVELS> levelInvVPM{};
    uint32_t clipmapBufferIndex = 0xFFFFFFFF;

    // The filterable volume: CLIP_RESOLUTION^2 x (CLIP_RESOLUTION * STACKED_LEVELS), MIPS_PER_LEVEL
    // mips. voxelVolumeTextureIndex is the sampled full-chain slot cone tracing reads;
    // volumeMipStorageIndices[i] is a single-mip storage slot the resolve/downsample write through
    // (an RWTexture3D binding must resolve to exactly one mip).
    uint32_t voxelVolumeTextureIndex = 0xFFFFFFFF;
    std::vector<uint32_t> volumeMipStorageIndices;
    std::vector<vk::raii::ImageView> volumeMipViews; // owns the views the storage slots point at
    uint32_t volumeMipLevels = 0;

    // Ambient-cube irradiance volumes (VOXEL_FACE_DIRS order: +X,-X,+Y,-Y,+Z,-Z), stacked per level
    // like the radiance volume. Written by the gather pass, sampled by the lit shader in giMode 1.
    //
    // Ping-ponged for the temporal blend: the gather reads set 1-irradianceSet as history and writes
    // the other, so a level recentre costs an integer read offset instead of a copy pass. Everything
    // downstream (lit, debug views) reads whatever was written last.
    static constexpr std::array<uint32_t, 6> UNSET_FACES{0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF};
    std::array<std::array<uint32_t, 6>, 2> irradianceTextureIndices{UNSET_FACES, UNSET_FACES};
    std::array<std::array<uint32_t, 6>, 2> irradianceStorageIndices{};
    uint32_t irradianceSet = 0;
    // Per level: the centre and extent its history was gathered under, and whether that history can
    // be reprojected onto this frame.
    std::array<glm::vec3, STACKED_LEVELS> prevGatherCenter{};
    std::array<float, STACKED_LEVELS> prevGatherExtent{};
    std::array<bool, STACKED_LEVELS> irradianceHistoryValid{};
    uint32_t gatherFrame = 0;           // drives which amortization phase updates
    uint32_t gatherPipelineIndex = 0xFFFFFFFF;

    // How far a level recentred since its last gather, in irradiance voxels: history coord = id + this.
    // The snap keeps it whole; the residual check is the guard for when it somehow isn't, and the
    // extent check drops the history outright when Voxel Size changed under it.
    struct GridShift { glm::ivec3 voxels{0}; bool usable = false; };
    GridShift gridShiftSinceGather(uint32_t level) const {
        const glm::vec4& L = clipmap.levels[level];
        float irradianceVoxel = L.w / static_cast<float>(IRRADIANCE_RESOLUTION);
        glm::vec3 shiftF = (glm::vec3(L) - prevGatherCenter[level]) / irradianceVoxel;
        GridShift s;
        s.voxels = glm::ivec3(glm::round(shiftF));
        glm::vec3 residual = glm::abs(shiftF - glm::vec3(s.voxels));
        s.usable = irradianceHistoryValid[level]
                && prevGatherExtent[level] == L.w
                && std::max({residual.x, residual.y, residual.z}) < 0.01f
                && std::max({std::abs(s.voxels.x), std::abs(s.voxels.y), std::abs(s.voxels.z)}) < static_cast<int>(IRRADIANCE_RESOLUTION);
        return s;
    }

    // Whether recordGather runs this frame — the faces are only worth feeding back if they're being
    // kept current. Mirrors the condition in record().
    bool gatherEnabled() const {
        return features.vxgi.mode == 1 || (features.voxelDebug.enabled && features.voxelDebug.volumeSelect > 0);
    }

    bool volumeBlanked = false;  // volume zeroed since voxelizeScene went off
    bool volumesCleared = false; // every slab zeroed once, so a level enabled later never reads uninitialised memory

    uint32_t rasterPipelineIndex     = 0xFFFFFFFF;
    uint32_t resolvePipelineIndex    = 0xFFFFFFFF;
    uint32_t downsamplePipelineIndex = 0xFFFFFFFF;
    uint32_t debugPipelineIndex      = 0xFFFFFFFF; // features.voxelDebug ray-march overlay
    // Cube debug view: extract compacts occupied voxels into cubeInstanceBuffer and bumps the
    // instanceCount in cubeIndirectBuffer; cubeDraw renders them with one indirect draw.
    uint32_t cubeExtractPipelineIndex = 0xFFFFFFFF;
    uint32_t cubeDrawPipelineIndex    = 0xFFFFFFFF;
    uint32_t cubeInstanceBufferIndex  = 0xFFFFFFFF;
    uint32_t cubeIndirectBufferIndex  = 0xFFFFFFFF;

public:
    VoxelizationPass(GpuContext& gpu, BindlessSystem& bindless, Scene& scene, RenderFeatures& features, RenderPassResources& shared)
        : RenderPass(gpu, bindless, scene, features, shared) {}

    uint32_t getVolumeTextureIndex() const { return voxelVolumeTextureIndex; }
    uint32_t getVolumeMipLevels() const { return volumeMipLevels; }
    std::array<uint32_t, 6> getIrradianceTextureIndices() const { return irradianceTextureIndices[irradianceSet]; }
    const GPUVoxelClipmap& getClipmap() const { return clipmap; }

    // Debug views: selected volume (0 = radiance, 1..6 = irradiance face), its mip cap, its per-level
    // side, and which level's slab is shown.
    uint32_t debugVolumeTexIndex() const {
        int sel = features.voxelDebug.volumeSelect;
        return sel == 0 ? voxelVolumeTextureIndex : irradianceTextureIndices[irradianceSet][sel - 1];
    }
    uint32_t debugVolumeMaxMip() const {
        return features.voxelDebug.volumeSelect == 0 ? volumeMipLevels - 1 : 0; // faces are single-mip
    }
    uint32_t debugVolumeResolution() const {
        return features.voxelDebug.volumeSelect == 0 ? CLIP_RESOLUTION : IRRADIANCE_RESOLUTION;
    }
    uint32_t debugLevel() const {
        return std::min(features.voxelDebug.clipLevel, clipmap.levelCount - 1);
    }

    void init(uint32_t width, uint32_t height) override {
        (void)width;
        (void)height;

        // Screen-independent, so allocated once.
        if (albedoBufferIndex == 0xFFFFFFFF) {
            constexpr vk::DeviceSize voxelCount = LEVEL_VOXELS * STACKED_LEVELS;
            // eTransferDst for the per-frame fillBuffer clear. Device-local is not optional: the
            // fragment shader does two atomics per fragment on these, and on the default host-visible
            // allocation every one of them crosses PCIe.
            auto usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eTransferDst;
            static_assert(voxelCount * sizeof(uint32_t) <= 0xFFFFFFFFull, "voxel buffer exceeds createVariableBuffer's uint32 size");
            constexpr uint32_t bufferBytes = static_cast<uint32_t>(voxelCount * sizeof(uint32_t));
            albedoBufferIndex   = bindless.descriptorSet->createVariableBuffer(bufferBytes, usage, false, "VoxelAlbedo", true);
            radianceBufferIndex = bindless.descriptorSet->createVariableBuffer(bufferBytes, usage, false, "VoxelRadiance", true);

            // Cube debug view shows one level at a time, so instance capacity is one level's worst
            // case (every mip-0 voxel occupied) and the extract's InterlockedAdd can never run past the end.
            cubeInstanceBufferIndex = bindless.descriptorSet->createVariableBuffer(static_cast<uint32_t>(LEVEL_VOXELS * 2 * sizeof(uint32_t)),
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress, false, "VoxelCubeInstances", true);
            cubeIndirectBufferIndex = bindless.descriptorSet->createVariableBuffer(4 * sizeof(uint32_t),
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress |
                vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eTransferDst, false, "VoxelCubeIndirect", true);

            // Per-frame clipmap descriptor for the compute passes: host-visible, rewritten every frame.
            clipmapBufferIndex = bindless.descriptorSet->createVariableBuffer(static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT * sizeof(GPUVoxelClipmap)),
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress, false, "VoxelClipmap", false);
        }

        if (voxelVolumeTextureIndex == 0xFFFFFFFF) createVolume();
        if (irradianceTextureIndices[0][0] == 0xFFFFFFFF) createIrradianceVolumes();

        // Trilinear for the cone-widening mips; border-black so cones leaving a level through its x/y
        // faces read zero radiance/opacity instead of smearing the edge voxel or wrapping
        // (defaultSampler is eRepeat). The z faces border other levels' slabs, so those clamp instead.
        if (shared.volumeSamplerIndex == 0xFFFFFFFF)
            shared.volumeSamplerIndex = bindless.descriptorSet->allocateSampler(
                vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear, vk::SamplerAddressMode::eClampToBorder,
                VK_FALSE, 1.0f, VK_FALSE, vk::CompareOp::eLessOrEqual, vk::BorderColor::eFloatTransparentBlack);

        auto& setLayout = bindless.descriptorSet->getDescriptorSetLayout();
        auto& set = bindless.descriptorSet->getDescriptorSet();
        if (rasterPipelineIndex == 0xFFFFFFFF)
            rasterPipelineIndex = bindless.pipelineManager->createPipeline<VoxelizationPushConstants>(
                PipelineCategory::VOXELIZATION, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone,
                vk::False, vk::False, "shaders/voxelization.spv", setLayout, set, vk::Format::eUndefined, "geomMain");
        if (resolvePipelineIndex == 0xFFFFFFFF)
            resolvePipelineIndex = bindless.pipelineManager->createComputePipeline<VoxelResolvePushConstants>("shaders/voxel_resolve.spv", setLayout, set, "resolveMain");
        if (downsamplePipelineIndex == 0xFFFFFFFF)
            downsamplePipelineIndex = bindless.pipelineManager->createComputePipeline<VoxelResolvePushConstants>("shaders/voxel_resolve.spv", setLayout, set, "downsampleMain");
        if (gatherPipelineIndex == 0xFFFFFFFF)
            gatherPipelineIndex = bindless.pipelineManager->createComputePipeline<VoxelGatherPushConstants>("shaders/voxel_gather.spv", setLayout, set, "gatherMain");
        // Overlay onto the HDR composite, so it sits alongside the other post passes.
        if (debugPipelineIndex == 0xFFFFFFFF)
            debugPipelineIndex = bindless.pipelineManager->createPipeline<VoxelDebugPushConstants>(
                PipelineCategory::POSTPROCESS_ALPHA_BLEND, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone,
                vk::False, vk::False, "shaders/voxel_debug.spv", setLayout, set, gpu.getSwapchain().getHDRColorFormat());
        if (cubeExtractPipelineIndex == 0xFFFFFFFF)
            cubeExtractPipelineIndex = bindless.pipelineManager->createComputePipeline<VoxelCubePushConstants>("shaders/voxel_cubes.spv", setLayout, set, "extractMain");
        // Depth test AND write (reverse-Z GE): cube mode replaces the scene, so the cubes only need to
        // occlude each other — recordCubes clears both attachments before the draw.
        if (cubeDrawPipelineIndex == 0xFFFFFFFF)
            cubeDrawPipelineIndex = bindless.pipelineManager->createPipeline<VoxelCubePushConstants>(
                PipelineCategory::POSTPROCESS, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone,
                vk::True, vk::True, "shaders/voxel_cubes.spv", setLayout, set, gpu.getSwapchain().getHDRColorFormat());
    }

    void record(vk::raii::CommandBuffer& cmd, uint32_t imageIndex) override {
        (void)imageIndex;
        // Levels first: the blank path below and the debug views later in the frame both need them.
        updateClipmap();
        if (!volumesCleared) {
            clearVolumes(cmd);
            volumesCleared = true;
        }

        // On the disable edge, resolve once over zeroed scatter buffers: the volume reads black
        // instead of feeding stale radiance to cone tracing and the debug views. After that,
        // skipping is safe — the volume rests in ShaderReadOnly with all transitions paired here.
        if (!features.voxelDebug.voxelizeScene) {
            if (!volumeBlanked) {
                irradianceHistoryValid.fill(false); // snap the faces to black instead of fading into it
                clearScatterBuffers(cmd);
                recordResolve(cmd);
                recordGather(cmd); // gathers off the blank volume, so the faces zero too
                volumeBlanked = true;
            }
            return;
        }
        if (volumeBlanked) irradianceHistoryValid.fill(false); // the history is a gather off a blank volume
        volumeBlanked = false;
        tracing::startTrace("voxelization");

        clearScatterBuffers(cmd);
        recordRaster(cmd);
        recordResolve(cmd);
        // Also gather when a debug view is inspecting a face, so it never shows stale data.
        if (gatherEnabled()) recordGather(cmd);

        tracing::endTrace("voxelization");
    }

    // Called separately from record() — this draws into the HDR composite, so it has to run after the
    // geometry and lighting passes have filled it, not inside the voxel build at the top of the frame.
    void recordDebugOverlay(vk::raii::CommandBuffer& cmd) {
        if (!features.voxelDebug.enabled || voxelVolumeTextureIndex == 0xFFFFFFFF) return;
        if (features.voxelDebug.drawCubes) { recordCubes(cmd); return; }
        tracing::startTrace("voxel debug");

        // Camera NDC -> level clip in one matrix, so the fragment shader can march straight through the
        // level's cube without ever going back to world space.
        uint32_t level = debugLevel();
        glm::mat4 camNdcToGrid = levelVPM[level] * glm::inverse(scene.activeCamera.viewProjection);
        glm::vec3 cameraPosGrid = levelUVW(clipmap.levels[level], scene.activeCamera.position);

        // The depth resolve rests in eDepthStencilAttachmentOptimal; sampling it requires the shader
        // read layout, same as the SSAO and particle passes do around their own depth reads.
        auto& depthResolveTex = bindless.descriptorSet->getTextureResource(gpu.getSwapchain().getDepthResolveIndex());
        resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *depthResolveTex.image,
                                        vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

        drawFullscreenPass(cmd, *bindless.pipelineManager->getPostProcessPipelines()[debugPipelineIndex],
            *bindless.descriptorSet->getTextureResource(shared.compositeColorTextureIndex).imageView,
            gpu.getSwapchain().getSwapChainExtent(),
            VoxelDebugPushConstants{
                .camNdcToGrid      = camNdcToGrid,
                .cameraPosGrid     = cameraPosGrid,
                .mipLevel          = std::min(features.voxelDebug.mipLevel, debugVolumeMaxMip()),
                .volumeTexIndex    = debugVolumeTexIndex(),
                .samplerIndex      = shared.volumeSamplerIndex, // clamped; defaultSampler's eRepeat wraps the grid
                .depthTexIndex     = gpu.getSwapchain().getDepthResolveIndex(),
                .depthSamplerIndex = shared.depthSamplerIndex,
                .resolution        = debugVolumeResolution(),
                .mode              = static_cast<uint32_t>(features.voxelDebug.mode),
                .maxSteps          = features.voxelDebug.maxSteps,
                .alphaScale        = features.voxelDebug.alphaScale,
                .level             = level,
            }, vk::AttachmentLoadOp::eLoad);

        resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *depthResolveTex.image,
                                        vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eDepthStencilAttachmentOptimal);

        tracing::endTrace("voxel debug");
    }

private:
    // Build this frame's levels from the camera and upload them for the compute passes.
    void updateClipmap() {
        clipmap = buildClipmap(features, scene.activeCamera.position);
        for (uint32_t l = 0; l < clipmap.levelCount; l++) {
            levelVPM[l]    = levelViewProjection(clipmap.levels[l]);
            levelInvVPM[l] = glm::inverse(levelVPM[l]);
        }
        auto& buffer = *bindless.descriptorSet->getVariableBuffers()[clipmapBufferIndex];
        std::memcpy(static_cast<char*>(buffer.mappedData) + clipmapFrameOffset(), &clipmap, sizeof(GPUVoxelClipmap));
    }
    vk::DeviceSize clipmapFrameOffset() const { return static_cast<vk::DeviceSize>(gpu.currentFrame) * sizeof(GPUVoxelClipmap); }
    vk::DeviceAddress clipmapAddress() const {
        return bindless.descriptorSet->getVariableBuffers()[clipmapBufferIndex]->address + clipmapFrameOffset();
    }

    // Zero every slab of every volume once. Only the active levels get resolved and gathered each
    // frame, so a level enabled later would otherwise show whatever its allocation held.
    void clearVolumes(vk::raii::CommandBuffer& cmd) {
        auto clear = [&](uint32_t textureIndex, uint32_t mips) {
            auto& tex = bindless.descriptorSet->getTextureResource(textureIndex);
            resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *tex.image,
                                            vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eTransferDstOptimal, 0, mips);
            cmd.clearColorImage(*tex.image, vk::ImageLayout::eTransferDstOptimal,
                                vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f}),
                                vk::ImageSubresourceRange{.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = mips, .baseArrayLayer = 0, .layerCount = 1});
            resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *tex.image,
                                            vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, 0, mips);
        };
        clear(voxelVolumeTextureIndex, volumeMipLevels);
        for (const auto& set : irradianceTextureIndices)
            for (uint32_t idx : set) clear(idx, 1);
    }

    // Cube view: extract occupied voxels of the selected level (compute) -> one indirect draw of
    // shaded cubes into the HDR composite, depth-tested against the scene. The volume rests in
    // ShaderReadOnly, which is what the extract's Load needs, and the depth resolve rests in
    // DepthStencilAttachmentOptimal, which is what the draw binds — so no layout transitions at all.
    void recordCubes(vk::raii::CommandBuffer& cmd) {
        tracing::startTrace("voxel debug cubes");

        uint32_t level = debugLevel();
        uint32_t mip = std::min(features.voxelDebug.mipLevel, debugVolumeMaxMip());
        uint32_t mipRes = std::max(1u, debugVolumeResolution() >> mip);
        VoxelCubePushConstants pc{
            .gridToClip            = scene.activeCamera.viewProjection * levelInvVPM[level],
            .instanceBufferAddress = bindless.descriptorSet->getVariableBuffers()[cubeInstanceBufferIndex]->address,
            .indirectBufferAddress = bindless.descriptorSet->getVariableBuffers()[cubeIndirectBufferIndex]->address,
            .volumeTexIndex        = debugVolumeTexIndex(),
            .mipLevel              = mip,
            .mipRes                = mipRes,
            .threshold             = features.voxelDebug.cubeThreshold,
            .zOffset               = level * mipRes,
        };

        // Last frame's indirect draw must finish reading these buffers before this frame's reset
        // overwrites them (WAR — execution ordering only, no data to flush).
        vk::MemoryBarrier warBarrier{};
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eDrawIndirect | vk::PipelineStageFlagBits::eVertexShader,
                            vk::PipelineStageFlagBits::eTransfer, {}, warBarrier, {}, {});

        // Reset the indirect args (36 verts/cube, instanceCount 0), then let the extract's atomics see it.
        const std::array<uint32_t, 4> drawArgs{36, 0, 0, 0};
        cmd.updateBuffer<uint32_t>(bindless.descriptorSet->getVariableBuffer(cubeIndirectBufferIndex), 0, drawArgs);
        vk::MemoryBarrier resetBarrier{.srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                                       .dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite};
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, resetBarrier, {}, {});

        auto& extract = static_cast<ComputePipeline<VoxelCubePushConstants>&>(*bindless.pipelineManager->getComputePipelines()[cubeExtractPipelineIndex]);
        extract.pushConstantData = pc;
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, extract.pipeline);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, extract.layout, 0, {**extract.descriptorSet}, {});
        extract.pushConstants(cmd);
        uint32_t groups = (mipRes + WG - 1) / WG;
        cmd.dispatch(groups, groups, groups);

        // Instance data to the vertex shader, instanceCount to the indirect fetch.
        vk::MemoryBarrier extractBarrier{.srcAccessMask = vk::AccessFlagBits::eShaderWrite,
                                         .dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead | vk::AccessFlagBits::eShaderRead};
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eDrawIndirect | vk::PipelineStageFlagBits::eVertexShader, {}, extractBarrier, {}, {});

        // Cube mode replaces the scene: clear the composite to a dark backdrop and depth to far
        // (reverse-Z: 0), then depth-write so cubes occlude each other. The rendered geometry is
        // hidden, not skipped — depth/normal targets stay valid for the other passes. Anything after
        // this (billboards, SDF, gizmos) now depth-tests against the voxel world, which is the point.
        vk::RenderingAttachmentInfo colorAttachment{.imageView = *bindless.descriptorSet->getTextureResource(shared.compositeColorTextureIndex).imageView,
                                                    .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                    .loadOp = vk::AttachmentLoadOp::eClear,
                                                    .storeOp = vk::AttachmentStoreOp::eStore,
                                                    .clearValue = {.color = vk::ClearColorValue(std::array<float, 4>{0.02f, 0.025f, 0.035f, 1.0f})}};
        vk::RenderingAttachmentInfo depthAttachment{.imageView = *gpu.getSwapchain().getDepthResolveImageView(),
                                                    .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                                    .loadOp = vk::AttachmentLoadOp::eClear,
                                                    .storeOp = vk::AttachmentStoreOp::eStore,
                                                    .clearValue = {.depthStencil = {0.0f, 0}}};
        vk::Extent2D extent = gpu.getSwapchain().getSwapChainExtent();
        vk::RenderingInfo renderInfo{.renderArea = {{0, 0}, extent}, .layerCount = 1,
                                     .colorAttachmentCount = 1, .pColorAttachments = &colorAttachment,
                                     .pDepthAttachment = &depthAttachment};

        cmd.beginRendering(renderInfo);
        auto& pipeline = *bindless.pipelineManager->getPostProcessPipelines()[cubeDrawPipelineIndex];
        bindPipeline(cmd, pipeline);
        setFullscreenViewport(cmd, extent);
        cmd.pushConstants<VoxelCubePushConstants>(pipeline.layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pc);
        cmd.drawIndirect(bindless.descriptorSet->getVariableBuffer(cubeIndirectBufferIndex), 0, 1, sizeof(vk::DrawIndirectCommand));
        cmd.endRendering();

        tracing::endTrace("voxel debug cubes");
    }

    // Mipped storage+sampled volume with every level's slab stacked along z. Built by hand rather
    // than through resize3DStorageImage because that helper is single-mip and each level here needs
    // its own storage slot. eTransferDst for the one-time clear.
    void createVolume() {
        volumeMipLevels = MIPS_PER_LEVEL;

        vk::raii::Image image = nullptr;
        vk::raii::DeviceMemory memory = nullptr;
        resource::create3DImage(*bindless.resourceCtx, CLIP_RESOLUTION, CLIP_RESOLUTION, CLIP_RESOLUTION * STACKED_LEVELS, volumeMipLevels,
                                vk::SampleCountFlagBits::e1, VOXEL_VOLUME_FORMAT, vk::ImageTiling::eOptimal,
                                vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
                                vk::MemoryPropertyFlagBits::eDeviceLocal, image, memory, 1);

        // Per-mip storage views first — they capture the VkImage handle, which survives the move into
        // the texture slot below.
        volumeMipViews.reserve(volumeMipLevels);
        for (uint32_t mip = 0; mip < volumeMipLevels; mip++)
            volumeMipViews.push_back(resource::create3DImageView(*bindless.resourceCtx, image, VOXEL_VOLUME_FORMAT, vk::ImageAspectFlagBits::eColor, mip, 1));

        auto sampledView = resource::create3DImageView(*bindless.resourceCtx, image, VOXEL_VOLUME_FORMAT, vk::ImageAspectFlagBits::eColor, 0, volumeMipLevels);
        // Resting layout is sampled; record() transitions the whole chain to eGeneral to write it.
        resource::transitionImageLayout(*bindless.resourceCtx, nullptr, image, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal, 0, volumeMipLevels);

        voxelVolumeTextureIndex = bindless.descriptorSet->allocateTexture(std::move(image), std::move(memory), std::move(sampledView),
                                                                         "internal/voxel_clipmap", false, CLIP_RESOLUTION, CLIP_RESOLUTION);
        volumeMipStorageIndices.reserve(volumeMipLevels);
        for (auto& view : volumeMipViews)
            volumeMipStorageIndices.push_back(bindless.descriptorSet->allocateStorageImage(*view));
    }

    // Two sets of six single-mip volumes for the ambient-cube gather, levels stacked along z — the
    // temporal history is the set the gather isn't writing. One view serves both slots:
    // allocateStorageImage captures the handle before the raii view moves into the texture slot.
    void createIrradianceVolumes() {
        static constexpr const char* faceNames[6] = {"px", "nx", "py", "ny", "pz", "nz"};
        for (uint32_t set = 0; set < 2; set++) {
            for (uint32_t f = 0; f < 6; f++) {
                vk::raii::Image image = nullptr;
                vk::raii::DeviceMemory memory = nullptr;
                resource::create3DImage(*bindless.resourceCtx, IRRADIANCE_RESOLUTION, IRRADIANCE_RESOLUTION, IRRADIANCE_RESOLUTION * STACKED_LEVELS, 1,
                                        vk::SampleCountFlagBits::e1, VOXEL_VOLUME_FORMAT, vk::ImageTiling::eOptimal,
                                        vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
                                        vk::MemoryPropertyFlagBits::eDeviceLocal, image, memory, 1);
                auto view = resource::create3DImageView(*bindless.resourceCtx, image, VOXEL_VOLUME_FORMAT, vk::ImageAspectFlagBits::eColor, 0, 1);
                // Resting layout is sampled; recordGather transitions to eGeneral to write.
                resource::transitionImageLayout(*bindless.resourceCtx, nullptr, image, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal, 0, 1);
                irradianceStorageIndices[set][f] = bindless.descriptorSet->allocateStorageImage(*view);
                irradianceTextureIndices[set][f] = bindless.descriptorSet->allocateTexture(std::move(image), std::move(memory), std::move(view),
                    std::string("internal/voxel_irradiance_") + faceNames[f] + (set == 0 ? "_a" : "_b"), false, IRRADIANCE_RESOLUTION, IRRADIANCE_RESOLUTION);
            }
        }
    }

    // Zero the active levels' blocks of both scatter buffers, then make the fill visible to the
    // fragment shader's atomics.
    void clearScatterBuffers(vk::raii::CommandBuffer& cmd) {
        // The previous frame's raster atomics and resolve reads must finish before this frame's clear
        // overwrites the buffers (WAR/WAW — execution ordering only).
        vk::MemoryBarrier warBarrier{};
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eFragmentShader | vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eTransfer, {}, warBarrier, {}, {});

        vk::DeviceSize activeBytes = LEVEL_SCATTER_BYTES * clipmap.levelCount;
        cmd.fillBuffer(bindless.descriptorSet->getVariableBuffer(albedoBufferIndex), 0, activeBytes, 0u);
        cmd.fillBuffer(bindless.descriptorSet->getVariableBuffer(radianceBufferIndex), 0, activeBytes, 0u);
        vk::MemoryBarrier fillBarrier{.srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                                      .dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite};
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, fillBarrier, {}, {});
    }

    // Attachment-less rasterization, one sweep per level. renderArea is what drives coverage — there
    // is nothing bound to write to, so the fragment shader's atomics are the only output. Every level
    // is the same voxel count, so the sweeps share one render pass and viewport; what changes per
    // level is the matrix and which block of the scatter buffers the fragments land in.
    void recordRaster(vk::raii::CommandBuffer& cmd) {
        VoxelizationPushConstants pc{};
        pc.vertexBufferAddress   = bindless.descriptorSet->getVariableBuffers()[shared.vertexBufferIndex]->address;
        pc.lightBufferAddress    = bindless.descriptorSet->getFixedBuffers()[shared.buffers.lightBufferIndex]->address + static_cast<vk::DeviceSize>(gpu.currentFrame) * MAX_FIXED_BUFFER * sizeof(GPULight);
        pc.lightCount            = scene.getLightLoopBound();
        pc.shadowAtlasIndex      = scene.shadowAtlas.textureIndex;
        pc.samplerIndex          = shared.defaultSamplerIndex;
        pc.voxelResolution       = CLIP_RESOLUTION;
        pc.skyEnvMapIndex        = scene.getSkyBox();
        pc.skyInjection          = features.skyboxIntensity * features.vxgi.skyInjection;
        const vk::DeviceAddress albedoBase   = bindless.descriptorSet->getVariableBuffers()[albedoBufferIndex]->address;
        const vk::DeviceAddress radianceBase = bindless.descriptorSet->getVariableBuffers()[radianceBufferIndex]->address;

        vk::Extent2D gridExtent{CLIP_RESOLUTION, CLIP_RESOLUTION};
        vk::RenderingInfo renderingInfo = {.renderArea = {.offset = {0, 0}, .extent = gridExtent},
                                           .layerCount = 1,
                                           .colorAttachmentCount = 0};
        cmd.beginRendering(renderingInfo);

        auto& pipeline = *bindless.pipelineManager->getBeforeGeoPipelines()[rasterPipelineIndex];
        bindPipeline(cmd, pipeline);
        setFullscreenViewport(cmd, gridExtent); // viewport/scissor are dynamic state
        cmd.bindIndexBuffer(bindless.descriptorSet->getVariableBuffer(shared.indexBufferIndex), 0, vk::IndexType::eUint32);

        for (uint32_t level = 0; level < clipmap.levelCount; level++) {
            float voxel = clipmap.levels[level].w / static_cast<float>(CLIP_RESOLUTION);
            pc.vpm                  = levelVPM[level];
            pc.voxelAlbedoAddress   = albedoBase   + level * LEVEL_SCATTER_BYTES;
            pc.voxelRadianceAddress = radianceBase + level * LEVEL_SCATTER_BYTES;
            pc.shadowTile           = level;          // the tile fit around this level's cube
            pc.shadowNormalOffset   = 0.8f * voxel;   // ~1.6 of the tile's texels, as before
            float cullBelow = features.vxgi.smallNodeCull * voxel;

            // One draw per node inside the level. Culling is against the level's cube itself, so
            // anything outside is skipped rather than clipped.
            std::array<Plane, 6> frustumPlanes = extractFrustumPlanes(pc.vpm);
            for (Node& node : scene.sceneGraph.getNodes()) {
                if (node.meshIndex == MAX_MESHES || !node.alive) continue;
                // Only cull on a bbox the scene graph has actually filled in — an unset one is all zeroes.
                if (node.isBoundingBoxValid()) {
                    if (!isAABBInFrustum(node.boundingBoxMin, node.boundingBoxMax, frustumPlanes)) continue;
                    // Optional: nodes far below the voxel size can't register reliably (VXGISettings::smallNodeCull).
                    glm::vec3 size = node.boundingBoxMax - node.boundingBoxMin;
                    if (cullBelow > 0.0f && std::max({size.x, size.y, size.z}) < cullBelow) continue;
                }

                const Mesh& mesh = scene.assetManager.meshes[node.meshIndex];
                if (mesh.freed || mesh.indexCount == 0) continue;

                uint32_t matIdx = node.getMaterialIndex();
                const Material& material = (matIdx < scene.materials.size()) ? scene.materials[matIdx]
                                                                             : scene.materials[scene.getFallBackMaterial()];
                // Coarser levels rasterize coarser LODs: their voxels are far above LOD0's triangle
                // size, and the outer levels see most of the scene.
                uint32_t lod = mesh.LODs.empty() ? 0u : std::min<uint32_t>(level, static_cast<uint32_t>(mesh.LODs.size()) - 1);
                pc.model              = node.getTransform();
                pc.vertexStride       = mesh.vertexStride;
                pc.vertexOffset       = static_cast<uint32_t>(mesh.vertexOffset);
                pc.albedoTextureIndex = material.albedoTextureIndex;
                pc.packedColor        = packColorRGBA8(material.color);
                cmd.pushConstants<VoxelizationPushConstants>(pipeline.layout,
                    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eGeometry | vk::ShaderStageFlagBits::eFragment, 0, pc);
                cmd.drawIndexed(mesh.lodIndexCount(lod), 1, static_cast<uint32_t>(mesh.indexOffset / sizeof(uint32_t)) + mesh.lodIndexStart(lod), 0, 0);
            }
        }

        cmd.endRendering();
    }

    // Unpack the scatter buffers into mip 0, then fold the chain, over every active level's slab at
    // once. Each mip depends on the previous one, so every dispatch is separated by a compute->compute barrier.
    void recordResolve(vk::raii::CommandBuffer& cmd) {
        tracing::startTrace("voxel resolve");
        auto& volumeTex = bindless.descriptorSet->getTextureResource(voxelVolumeTextureIndex);

        // The raster pass wrote through storage buffers from the fragment stage — that has to land
        // before the resolve reads it.
        vk::MemoryBarrier scatterBarrier{.srcAccessMask = vk::AccessFlagBits::eShaderWrite, .dstAccessMask = vk::AccessFlagBits::eShaderRead};
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eFragmentShader, vk::PipelineStageFlagBits::eComputeShader, {}, scatterBarrier, {}, {});

        resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *volumeTex.image,
                                        vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eGeneral, 0, volumeMipLevels);

        // Workgroups over a side x side x (side * levels) box.
        auto groupsFor = [&](uint32_t side) {
            uint32_t g = (side + WG - 1) / WG;
            return glm::uvec3(g, g, (side * clipmap.levelCount + WG - 1) / WG);
        };
        auto dispatchCompute = [&](uint32_t pipeIdx, const VoxelResolvePushConstants& pc, glm::uvec3 groups) {
            auto& pipe = static_cast<ComputePipeline<VoxelResolvePushConstants>&>(*bindless.pipelineManager->getComputePipelines()[pipeIdx]);
            pipe.pushConstantData = pc;
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipe.pipeline);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipe.layout, 0, {**pipe.descriptorSet}, {});
            pipe.pushConstants(cmd);
            cmd.dispatch(groups.x, groups.y, groups.z);
        };
        auto computeBarrier = [&]() {
            vk::MemoryBarrier b{.srcAccessMask = vk::AccessFlagBits::eShaderWrite,
                                .dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite};
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, b, {}, {});
        };

        VoxelResolvePushConstants pc{
            .voxelAlbedoAddress   = bindless.descriptorSet->getVariableBuffers()[albedoBufferIndex]->address,
            .voxelRadianceAddress = bindless.descriptorSet->getVariableBuffers()[radianceBufferIndex]->address,
            .dstStorageIndex      = volumeMipStorageIndices[0],
            .srcStorageIndex      = volumeMipStorageIndices[0],
            .dstResolution        = CLIP_RESOLUTION,
            .voxelResolution      = CLIP_RESOLUTION,
            .levelCount           = clipmap.levelCount,
        };
        dispatchCompute(resolvePipelineIndex, pc, groupsFor(CLIP_RESOLUTION));

        for (uint32_t mip = 1; mip < volumeMipLevels; mip++) {
            computeBarrier();
            pc.srcStorageIndex = volumeMipStorageIndices[mip - 1];
            pc.dstStorageIndex = volumeMipStorageIndices[mip];
            pc.dstResolution   = std::max(1u, CLIP_RESOLUTION >> mip);
            dispatchCompute(downsamplePipelineIndex, pc, groupsFor(pc.dstResolution));
        }

        // Back to the sampled resting layout; this also makes the writes visible to cone tracing.
        resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *volumeTex.image,
                                        vk::ImageLayout::eGeneral, vk::ImageLayout::eShaderReadOnlyOptimal, 0, volumeMipLevels);
        tracing::endTrace("voxel resolve");
    }

    // Ambient-cube gather: six axis cone traces per occupied voxel into the per-face irradiance
    // volumes, one dispatch per gathered level. Reads the radiance volume through its sampled slot —
    // recordResolve's final transition already ordered and flushed those writes — and writes through
    // the storage slots, so only the face volumes being written flip to eGeneral around the
    // dispatches. The history set stays in its sampled resting layout, which is what the shader's
    // Load() needs.
    void recordGather(vk::raii::CommandBuffer& cmd) {
        tracing::startTrace("voxel gather");
        uint32_t dstSet = 1 - irradianceSet;
        for (uint32_t idx : irradianceTextureIndices[dstSet])
            resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *bindless.descriptorSet->getTextureResource(idx).image,
                                            vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eGeneral);

        auto& pipe = static_cast<ComputePipeline<VoxelGatherPushConstants>&>(*bindless.pipelineManager->getComputePipelines()[gatherPipelineIndex]);
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipe.pipeline);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipe.layout, 0, {**pipe.descriptorSet}, {});

        for (uint32_t level = 0; level < clipmap.levelCount; level++) {
            GridShift shift = gridShiftSinceGather(level);
            bool useHistory = shift.usable;

            // Amortization needs somewhere to carry the untouched voxels from, so without history every
            // voxel traces. Rounded down to a power of two — the shader phase-tests with a mask.
            uint32_t phases = 1;
            if (useHistory) {
                int p = std::clamp(features.vxgi.updatePhases, 1, 8);
                phases = p >= 8 ? 8u : p >= 4 ? 4u : p >= 2 ? 2u : 1u;
                // Outer levels change slowly and cost the same per voxel, so each level out re-traces
                // half as often, down to 1/8. "Every frame" stays every frame everywhere.
                if (phases > 1) phases = std::min(8u, phases << std::min(level, 3u));
            }

            // Voxels the recentre just brought into this level have no history of their own; the next
            // level out held the same spot last frame, so they start from that value and converge
            // instead of stepping. Its validity here is still last frame's — this loop hasn't reached
            // it yet — which is exactly the slab in the history set.
            uint32_t parent = level + 1;
            bool seedFromParent = parent < clipmap.levelCount && irradianceHistoryValid[parent]
                               && prevGatherExtent[parent] == clipmap.levels[parent].w;
            glm::vec3 seedCenter = seedFromParent ? prevGatherCenter[parent] : glm::vec3(0.0f);

            pipe.pushConstantData = VoxelGatherPushConstants{
                .clipmapAddress        = clipmapAddress(),
                .level                 = level,
                .radianceTextureIndex  = voxelVolumeTextureIndex,
                .samplerIndex          = shared.volumeSamplerIndex,
                .faceStorageIndices    = irradianceStorageIndices[dstSet],
                .historyTextureIndices = irradianceTextureIndices[irradianceSet],
                .historyOffset         = {shift.voxels.x, shift.voxels.y, shift.voxels.z},
                .blendWeight           = std::clamp(features.vxgi.temporalBlend, 0.01f, 1.0f),
                .phaseMask             = phases - 1,
                .phase                 = gatherFrame & (phases - 1),
                .sideCones             = static_cast<uint32_t>(features.vxgi.gatherSideCones),
                .maxSteps              = static_cast<uint32_t>(features.vxgi.gatherSteps),
                .fetchBatch            = static_cast<uint32_t>(features.vxgi.gatherFetchBatch),
                .skyEnvMapIndex        = scene.getSkyBox(),
                .skyIntensity          = features.skyboxIntensity * features.vxgi.skyStrength,
                .maxTraceDistance      = features.vxgi.maxTraceDistance,
                .historyValid          = useHistory ? 1u : 0u,
                .seedCenter            = {seedCenter.x, seedCenter.y, seedCenter.z},
                .seedFromParent        = seedFromParent ? 1u : 0u,
            };
            pipe.pushConstants(cmd);
            uint32_t groups = (IRRADIANCE_RESOLUTION + WG - 1) / WG;
            cmd.dispatch(groups, groups, groups);

            prevGatherCenter[level]      = glm::vec3(clipmap.levels[level]);
            prevGatherExtent[level]      = clipmap.levels[level].w;
            irradianceHistoryValid[level] = true;
        }

        // Back to sampled; also makes the gather writes visible to the lit pass.
        for (uint32_t idx : irradianceTextureIndices[dstSet])
            resource::transitionImageLayout(*bindless.resourceCtx, &cmd, *bindless.descriptorSet->getTextureResource(idx).image,
                                            vk::ImageLayout::eGeneral, vk::ImageLayout::eShaderReadOnlyOptimal);

        // What was just written is what everything downstream reads, and next frame's history.
        irradianceSet = dstSet;
        gatherFrame++;
        tracing::endTrace("voxel gather");
    }
};
