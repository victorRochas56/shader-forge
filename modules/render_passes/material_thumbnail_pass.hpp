#pragma once
#include "render_pass.hpp"
#include "bindless_system.hpp"
#include "scene.hpp"
#include "structs.hpp"
#include "constants.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/constants.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// Renders each material onto a sphere with the real lit pipeline (lit.spv) into its own bindless
// target for GUI previews. The sphere is placed at the live camera position and lit by the scene's
// actual lights + shadow atlas, re-rendered every frame so previews track the camera.
class MaterialThumbnailPass : public RenderPass {

    static constexpr uint32_t SIZE = 128;
    static constexpr vk::Format COLOR_FORMAT = vk::Format::eR8G8B8A8Unorm;
    static constexpr uint32_t MAX_MATS = 256; // preview cap per frame

    uint32_t pipelineIndex = 0xFFFFFFFF;

    // Shared, reused across all materials each frame.
    vk::raii::Image        depthImage  = nullptr;
    vk::raii::DeviceMemory depthMemory = nullptr;
    vk::raii::ImageView    depthView   = nullptr;
    uint32_t scratchAIndex = 0xFFFFFFFF; // MRT slot 1 (roughness/metal), never sampled
    uint32_t scratchBIndex = 0xFFFFFFFF; // MRT slot 2 (world normal), never sampled

    // Unit sphere geometry in the shared vertex/index buffers.
    uint32_t sphereVertexAlloc = 0, sphereVertexOffset = 0, sphereVertexStride = 0;
    uint32_t sphereIndexFirst = 0, sphereIndexCount = 0;

    // Per-frame lit buffers (own slot region per frame to avoid frames-in-flight races).
    uint32_t instanceBufferIndex = 0xFFFFFFFF;
    uint32_t meshDrawBufferIndex = 0xFFFFFFFF;
    uint32_t modelBufferIndex    = 0xFFFFFFFF;
    uint32_t passBufferIndex     = 0xFFFFFFFF;
    // Own lit frame UBOs: the preview uses its own camera, so it can't share the main pass's slots.
    uint32_t frameUniformsIndex[MAX_FRAMES_IN_FLIGHT] = {0xFFFFFFFF, 0xFFFFFFFF};
    std::unique_ptr<GPULitFrameUniforms> frameUniformsStaging;

public:
    MaterialThumbnailPass(GpuContext& gpu, BindlessSystem& bindless, Scene& scene, RenderFeatures& features, RenderPassResources& shared)
        : RenderPass(gpu, bindless, scene, features, shared) {}

    void init(uint32_t, uint32_t) override {
        if (pipelineIndex == 0xFFFFFFFF) {
            pipelineIndex = bindless.pipelineManager->createPipeline<LitPushConstants>(
                PipelineCategory::MATERIAL_THUMBNAIL, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone,
                vk::True, vk::True, "shaders/lit.spv", bindless.descriptorSet->getDescriptorSetLayout(),
                bindless.descriptorSet->getDescriptorSet(), vk::Format::eUndefined);
        }
        if (*depthImage == VK_NULL_HANDLE) {
            resource::createImage(*bindless.resourceCtx, SIZE, SIZE, 1, vk::SampleCountFlagBits::e1, vk::Format::eD32Sfloat,
                vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment,
                vk::MemoryPropertyFlagBits::eDeviceLocal, depthImage, depthMemory, 1);
            depthView = resource::createImageView(*bindless.resourceCtx, depthImage, vk::Format::eD32Sfloat, vk::ImageAspectFlagBits::eDepth);
            resource::transitionImageLayout(*bindless.resourceCtx, nullptr, *depthImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal);
        }
        resize(scratchAIndex, SIZE, SIZE, COLOR_FORMAT, "internal/matThumbScratchA");
        resize(scratchBIndex, SIZE, SIZE, COLOR_FORMAT, "internal/matThumbScratchB");

        if (instanceBufferIndex == 0xFFFFFFFF) {
            buildSphere();

            instanceBufferIndex = bindless.descriptorSet->createFixedBuffer<LitInstanceData>(MAX_FRAMES_IN_FLIGHT * MAX_MATS, false, "MatThumbInstance");
            meshDrawBufferIndex = bindless.descriptorSet->createFixedBuffer<LitMeshDrawData>(MAX_FRAMES_IN_FLIGHT, false, "MatThumbMeshDraw");
            modelBufferIndex    = bindless.descriptorSet->createFixedBuffer<glm::mat4>(MAX_FRAMES_IN_FLIGHT, false, "MatThumbModel");
            passBufferIndex     = bindless.descriptorSet->createFixedBuffer<LitPassData>(MAX_FRAMES_IN_FLIGHT, false, "MatThumbPass");

            frameUniformsStaging = std::make_unique<GPULitFrameUniforms>();
            for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                frameUniformsIndex[i] = bindless.descriptorSet->allocateUniformBuffer(sizeof(GPULitFrameUniforms), "MatThumbFrameUniforms" + std::to_string(i));
            }
        }
    }

    void record(vk::raii::CommandBuffer& cmd, uint32_t) override {

        tracing::startTrace("material thumb pass");
        auto& materials = scene.getMaterials();
        // A full lit draw per material per frame, and nothing samples the targets while the
        // Materials list is collapsed — so skip the whole pass. Targets stay in
        // eShaderReadOnlyOptimal, so they are still valid to sample the frame the list reopens.
        if (materials.empty() || !features.materialPreviewsVisible) {
            tracing::endTrace("material thumb pass");
            return;
        }
        uint32_t count = std::min<uint32_t>(static_cast<uint32_t>(materials.size()), MAX_MATS);
        if (materials.size() > MAX_MATS) {
            static bool warned = false;
            if (!warned) { std::cerr << "MaterialThumbnailPass: capping previews at " << MAX_MATS << std::endl; warned = true; }
        }

        uint32_t frame = gpu.currentFrame;

        // Sphere at the live camera position; a fixed orbit cam frames it for the preview.
        glm::vec3 spherePos = scene.activeCamera.position;
        const float R = 1.0f;
        glm::mat4 model = glm::translate(glm::mat4(1.0f), spherePos) * glm::scale(glm::mat4(1.0f), glm::vec3(R));

        const float fov = glm::radians(40.0f);
        float dist = R / std::sin(fov * 0.5f) * 1.6f;
        // View the sphere along the scene camera's look direction so the preview rotates with the camera.
        glm::vec3 camForward = -scene.activeCamera.getLookDir();
        glm::vec3 eye = spherePos - camForward * dist;
        glm::mat4 view = glm::lookAt(eye, spherePos, glm::vec3(0.0f, 1.0f, 0.0f));
        // Small near plane (like the main camera): glm here emits GL [-1,1] depth, so a large near
        // would push the effective Vulkan near-clip into the sphere. Keep it tiny to avoid that.
        glm::mat4 proj = glm::perspective(fov, 1.0f, 0.1f, dist + R * 2.0f);
        proj[1][1] *= -1.0f;
        glm::mat4 vp = proj * view;

        // Per-frame shared data.
        bindless.descriptorSet->writeFixedBuffer<glm::mat4>(modelBufferIndex, &model, 1, frame, frame);
        LitMeshDrawData md{ sphereVertexAlloc, sphereVertexOffset, sphereVertexStride, 0 };
        bindless.descriptorSet->writeFixedBuffer<LitMeshDrawData>(meshDrawBufferIndex, &md, 1, frame, frame);
        LitPassData pd{ .samplerIndex = shared.defaultSamplerIndex,
                        .lightCount = scene.getLightLoopBound(),
                        .shadowSamplerIndex = shared.shadowSamplerIndex,
                        .shadowAtlasIndex = scene.shadowAtlas.textureIndex,
                        .cameraPosition = eye,
                        .cameraForward = glm::normalize(spherePos - eye),
                        .viewProjection = vp,
                        .prevViewProjection = vp };
        bindless.descriptorSet->writeFixedBuffer<LitPassData>(passBufferIndex, &pd, 1, frame, frame);

        // Mirror the pass + hot lights into this pass's own frame UBO (lit reads hot data there).
        GPULitFrameUniforms& uf = *frameUniformsStaging;
        uint32_t hotBound = scene.fillHotLights(uf.lights, uf.cascades);
        uf.setPassData(pd);
        uf.indicesA.y = std::min(hotBound, MAX_UBO_LIGHTS);
        std::memcpy(bindless.descriptorSet->getUniformBufferMapped(frameUniformsIndex[frame]), &uf, sizeof(uf));

        std::vector<LitInstanceData> insts(count);
        for (uint32_t i = 0; i < count; i++) {
            const Material& m = materials[i];
            uint32_t mflags = static_cast<uint32_t>(m.flags);
            insts[i] = { .modelMatrixIndex = 0,
                         .albedoTextureIndex = (mflags & HAS_ALBEDO) ? m.albedoTextureIndex : shared.defaultAlbedoIndex,
                         .roughnessTextureIndex = (mflags & HAS_ROUGHNESS) ? m.roughnessTextureIndex : shared.defaultRoughnessIndex,
                         .metallicTextureIndex = (mflags & HAS_METALLIC) ? m.metallicTextureIndex : shared.defaultMetallicIndex,
                         .normalTextureIndex = (mflags & HAS_NORMAL) ? m.normalTextureIndex : shared.defaultNormalIndex,
                         .environmentMapIndex = m.environmentMapIndex,
                         .maxEnvMips = static_cast<float>(bindless.descriptorSet->getTextureMipLevels(m.environmentMapIndex) - 1),
                         .materialFlags = static_cast<uint32_t>(m.flags),
                         .metallic = m.metallic,
                         .roughness = m.roughness,
                         .alphaCutoff = m.alphaCutoff,
                         .packedColor = packColorRGBA8(m.color),
                         .triplanarScale = m.triplanarScale,
                         .triplanarBlend = m.triplanarBlend };
        }
        uint32_t frameInstBase = frame * MAX_MATS;
        bindless.descriptorSet->writeFixedBuffer<LitInstanceData>(instanceBufferIndex, insts.data(), count, frameInstBase, frame);

        // Buffer addresses (per-frame regions).
        auto fb = [&](uint32_t idx) -> vk::DeviceAddress { return bindless.descriptorSet->getFixedBuffers()[idx]->address; };
        uint64_t vertexAddr   = bindless.descriptorSet->getVariableBuffers()[shared.vertexBufferIndex]->address;
        uint64_t modelAddr    = fb(modelBufferIndex)    + static_cast<vk::DeviceSize>(frame) * sizeof(glm::mat4);
        uint64_t meshDrawAddr = fb(meshDrawBufferIndex) + static_cast<vk::DeviceSize>(frame) * sizeof(LitMeshDrawData);
        uint64_t passAddr     = fb(passBufferIndex)     + static_cast<vk::DeviceSize>(frame) * sizeof(LitPassData);
        uint64_t instBase     = fb(instanceBufferIndex) + static_cast<vk::DeviceSize>(frameInstBase) * sizeof(LitInstanceData);
        // Match the main lit pass's light-buffer addressing so previews see the same per-frame lights.
        uint64_t lightsAddr   = fb(shared.buffers.lightBufferIndex) + static_cast<vk::DeviceSize>(frame) * MAX_FIXED_BUFFER * sizeof(GPULight);

        // Allocate any missing targets FIRST — resize() can reallocate the texture-resource vector,
        // which would dangle references/handles captured below.
        for (uint32_t i = 0; i < count; i++) {
            if (materials[i].thumbnailTextureIndex == 0xFFFFFFFF)
                resize(materials[i].thumbnailTextureIndex, SIZE, SIZE, COLOR_FORMAT, std::string("internal/matThumb" + std::to_string(i)).c_str());
        }

        // Capture handles by value now that the vector is stable for the rest of this frame.
        vk::Image     saImage = *bindless.descriptorSet->getTextureResource(scratchAIndex).image;
        vk::Image     sbImage = *bindless.descriptorSet->getTextureResource(scratchBIndex).image;
        vk::ImageView saView  = *bindless.descriptorSet->getTextureResource(scratchAIndex).imageView;
        vk::ImageView sbView  = *bindless.descriptorSet->getTextureResource(scratchBIndex).imageView;
        resource::transitionImageLayout(*bindless.resourceCtx, &cmd, saImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal);
        resource::transitionImageLayout(*bindless.resourceCtx, &cmd, sbImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal);

        bindPipeline(cmd, *bindless.pipelineManager->getBeforeGeoPipelines()[pipelineIndex]);
        cmd.bindIndexBuffer(bindless.descriptorSet->getVariableBuffer(shared.indexBufferIndex), 0, vk::IndexType::eUint32);
        setFullscreenViewport(cmd, {SIZE, SIZE});

        vk::ClearValue clearColor{.color = vk::ClearColorValue(std::array<float, 4>{0.10f, 0.10f, 0.12f, 1.0f})};
        vk::ClearValue clearDepth{.depthStencil = vk::ClearDepthStencilValue{1.0f, 0}};

        for (uint32_t i = 0; i < count; i++) {
            vk::Image     targetImage = *bindless.descriptorSet->getTextureResource(materials[i].thumbnailTextureIndex).image;
            vk::ImageView targetView  = *bindless.descriptorSet->getTextureResource(materials[i].thumbnailTextureIndex).imageView;
            resource::transitionImageLayout(*bindless.resourceCtx, &cmd, targetImage, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eColorAttachmentOptimal);

            vk::RenderingAttachmentInfo colorAttachments[3] = {
                {.imageView = targetView, .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                 .loadOp = vk::AttachmentLoadOp::eClear, .storeOp = vk::AttachmentStoreOp::eStore, .clearValue = clearColor},
                {.imageView = saView, .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                 .loadOp = vk::AttachmentLoadOp::eClear, .storeOp = vk::AttachmentStoreOp::eDontCare, .clearValue = clearColor},
                {.imageView = sbView, .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                 .loadOp = vk::AttachmentLoadOp::eClear, .storeOp = vk::AttachmentStoreOp::eDontCare, .clearValue = clearColor},
            };
            vk::RenderingAttachmentInfo depthAttachment{.imageView = *depthView,
                                                        .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                                        .loadOp = vk::AttachmentLoadOp::eClear,
                                                        .storeOp = vk::AttachmentStoreOp::eDontCare,
                                                        .clearValue = clearDepth};
            vk::RenderingInfo renderInfo{.renderArea = {{0, 0}, {SIZE, SIZE}}, .layerCount = 1,
                                         .colorAttachmentCount = 3, .pColorAttachments = colorAttachments, .pDepthAttachment = &depthAttachment};

            cmd.beginRendering(renderInfo);
            LitPushConstants pc{.vertexBufferAddress = vertexAddr,
                                .modelMatricesAddress = modelAddr,
                                .lightsAddress = lightsAddr,
                                .litInstanceDataAddress = instBase + static_cast<vk::DeviceSize>(i) * sizeof(LitInstanceData),
                                .litMeshDrawDataAddress = meshDrawAddr,
                                .litPassDataAddress = passAddr,
                                .frameUniformsIndex = frameUniformsIndex[frame]};
            cmd.pushConstants<LitPushConstants>(bindless.pipelineManager->getBeforeGeoPipelines()[pipelineIndex]->layout,
                                                vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pc);
            cmd.drawIndexed(sphereIndexCount, 1, sphereIndexFirst, 0, 0);
            cmd.endRendering();

            resource::transitionImageLayout(*bindless.resourceCtx, &cmd, targetImage, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
        }
        tracing::endTrace("material thumb pass");
    }

private:
    // Generate a unit UV sphere matching the engine Vertex layout, upload to the shared buffers.
    void buildSphere() {
        const uint32_t stacks = 24, slices = 48;
        std::vector<Vertex> verts;
        std::vector<uint32_t> indices;
        for (uint32_t i = 0; i <= stacks; i++) {
            float phi = glm::pi<float>() * static_cast<float>(i) / stacks;     // 0..pi
            float cp = std::cos(phi), sp = std::sin(phi);
            for (uint32_t j = 0; j <= slices; j++) {
                float theta = glm::two_pi<float>() * static_cast<float>(j) / slices;
                float ct = std::cos(theta), st = std::sin(theta);
                glm::vec3 n(sp * ct, cp, sp * st);
                Vertex v{};
                v.position = n;
                v.normal = n;
                v.tangent = glm::vec3(-st, 0.0f, ct);
                v.materialIndex = 0;
                v.texCoord = glm::vec2(static_cast<float>(j) / slices, static_cast<float>(i) / stacks);
                verts.push_back(v);
            }
        }
        uint32_t ring = slices + 1;
        for (uint32_t i = 0; i < stacks; i++) {
            for (uint32_t j = 0; j < slices; j++) {
                uint32_t a = i * ring + j, b = a + ring;
                indices.insert(indices.end(), {a, b, a + 1, a + 1, b, b + 1});
            }
        }

        sphereVertexAlloc = bindless.descriptorSet->allocateVariableBuffer<Vertex>(verts, shared.vertexBufferIndex);
        uint32_t indexAlloc = bindless.descriptorSet->allocateVariableBuffer<uint32_t>(indices, shared.indexBufferIndex);
        auto vAlloc = bindless.descriptorSet->getVariableBufferAllocation(shared.vertexBufferIndex, sphereVertexAlloc);
        auto iAlloc = bindless.descriptorSet->getVariableBufferAllocation(shared.indexBufferIndex, indexAlloc);
        sphereVertexOffset = vAlloc.offset;
        sphereVertexStride = vAlloc.stride;
        sphereIndexFirst = iAlloc.offset / sizeof(uint32_t);
        sphereIndexCount = iAlloc.count;
    }
};
