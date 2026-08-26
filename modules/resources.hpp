#pragma once
#ifndef VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1 // for raii
#endif
#ifndef VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_NO_CONSTRUCTORS 1 // for structs constructors
#endif

#include <tiny_obj_loader.h>
// tiny_gltf is header-only and its one translation unit is modules/tinygltf_implementation.cpp,
// which reaches the header through this file so the switches below apply there too. They have to
// live here rather than in that .cpp: TINYGLTF_NO_STB_IMAGE_WRITE changes a default member
// initialiser inside TinyGLTF itself, so a TU that disagrees is an ODR violation, not a link error.
#define TINYGLTF_NO_STB_IMAGE_WRITE // nothing writes glTF, and stb_image_write has no TU here
#include <tiny_gltf.h>
#include <vulkan/vulkan.hpp>
#include <ktx.h>

#define GLM_FORCE_RADIANS
#define GLM_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#include "devices.hpp"
#include "structs.hpp"
#include "texture_converter.hpp"
#include "utils.hpp"

struct MeshEntry {
    // Pivot for this entry, in the source file's own space. The loader picks it, both load paths
    // recentre the vertices on it, and the placement they hand back is built from it — so import
    // and scene reload agree by construction instead of by two copies of the same arithmetic.
    glm::vec3 origin{0.0f};
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    std::vector<uint32_t> LODs;
    int materialId = -1;
    std::string materialName = "default_material";
    std::string shapeName;

    // The object this entry came out of, before the split by material. Entries sharing an id were
    // one thing in the file — an OBJ shape, a glTF node placement — and import parents them under a
    // node of `objectName` instead of scattering the pieces as siblings. Loaders emit an object's
    // entries consecutively, which is what lets import walk them in one pass.
    uint32_t sourceObject = 0;
    std::string objectName;

    // Instancing the file states outright: this entry is entries[instanceOf] placed again at
    // `instanceTransform`, relative to where that entry sits. It carries no geometry of its own, so
    // it costs no simplification and no ICP fit. -1 on anything the file didn't say is a copy —
    // every OBJ entry, the first placement of a glTF primitive, and any copy whose transform a node
    // can't express (a mirror or a resize), which is baked and left to instance detection instead.
    int instanceOf = -1;
    glm::mat4 instanceTransform{1.0f};
};

struct MeshData {
    std::vector<MeshEntry> entries; // one per material group per shape
};

// LOD.cpp — appends simplified index sets to `indices`, fills `LODs` with per-LOD index counts.
// Not thread-safe (global Simplify state).
void generateLODs(const std::vector<Vertex>& vertices, std::vector<uint32_t>& indices, std::vector<uint32_t>& LODs);
inline void generateLODs(MeshEntry& mesh) { generateLODs(mesh.vertices, mesh.indices, mesh.LODs); }

/*
namespace for GPU resource operations: loading textures/meshes, creating textures
at runtime (shadow maps and other procedural textures), and image layout transitions.
*/
namespace resource {

// context struct shared by many of the resource operations
struct Context {
    Device& device;
    const vk::raii::CommandPool& commandPool;
};

struct ImageTransitionInfo {
    vk::Image     image;
    vk::ImageLayout oldLayout;
    vk::ImageLayout newLayout;
    uint32_t baseMipLevel   = 0;
    uint32_t mipLevelCount  = 1;
    uint32_t baseArrayLayer = 0;
    uint32_t layerCount     = 1;
};

inline vk::raii::CommandBuffer beginSingleTimeCommands(const Context& ctx) {
    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo.commandPool = *ctx.commandPool;
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandBufferCount = 1;

    vk::raii::CommandBuffers commandBuffers(ctx.device.getDevice(), allocInfo);
    vk::raii::CommandBuffer commandBuffer = std::move(commandBuffers[0]);

    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
    commandBuffer.begin(beginInfo);

    return commandBuffer;
}

inline void endSingleTimeCommands(const Context& ctx, vk::raii::CommandBuffer& commandBuffer) {
    commandBuffer.end();

    vk::SubmitInfo submitInfo{};
    submitInfo.setCommandBuffers(*commandBuffer);

    ctx.device.getGraphicsQueue().submit(submitInfo);
    ctx.device.getGraphicsQueue().waitIdle();
}

inline vk::ImageMemoryBarrier2 buildBarrier(const vk::Image& image, const vk::ImageLayout oldLayout, const vk::ImageLayout newLayout,
                                            uint32_t baseMipLevel, uint32_t mipLevelCount, uint32_t baseArrayLayer, uint32_t layerCount) {
    vk::ImageMemoryBarrier2 barrier{.oldLayout = oldLayout,
                                    .newLayout = newLayout,
                                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                    .image = image,
                                    .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                                                         .baseMipLevel = baseMipLevel,
                                                         .levelCount = mipLevelCount,
                                                         .baseArrayLayer = baseArrayLayer,
                                                         .layerCount = layerCount}};

    if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
        barrier.srcAccessMask = {};
        barrier.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
        barrier.srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
        barrier.dstStageMask = vk::PipelineStageFlagBits2::eTransfer;
    }

    else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
        barrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
        barrier.srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
        barrier.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader;
    }

    else if (oldLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
        // ColorAttachmentOutput in src: the MSAA depth *resolve* at endRendering counts as a
        // color-attachment write, not a late-fragment-tests write.
        barrier.srcAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite | vk::AccessFlagBits2::eColorAttachmentWrite;
        barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
        barrier.srcStageMask = vk::PipelineStageFlagBits2::eLateFragmentTests | vk::PipelineStageFlagBits2::eColorAttachmentOutput;
        // Compute in dst: the shadow atlas is sampled by the froxel light pass (VolumetricsPass C)
        // from compute, so the depth writes must be visible there too — not just to fragment.
        barrier.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader | vk::PipelineStageFlagBits2::eComputeShader;
        barrier.subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eDepth, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1};
    }

    else if (oldLayout == vk::ImageLayout::eShaderReadOnlyOptimal && newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
        // ColorAttachmentOutput in dst: the next write to a resolved depth image is the MSAA resolve
        // at endRendering, which lands at ColorAttachmentOutput — not early-fragment-tests.
        barrier.srcAccessMask = vk::AccessFlagBits2::eShaderRead;
        barrier.dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite | vk::AccessFlagBits2::eColorAttachmentWrite;
        // Compute in src: next frame's shadow writes must wait for this frame's compute reads (WAR).
        barrier.srcStageMask = vk::PipelineStageFlagBits2::eFragmentShader | vk::PipelineStageFlagBits2::eComputeShader;
        barrier.dstStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eColorAttachmentOutput;
        barrier.subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eDepth, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1};
    }

    // Depth-only read layout — preserves Z-compression on NV/AMD for faster sampled-depth reads.
    // Includes the compute stage: the froxel light pass (VolumetricsPass C) samples the shadow
    // atlas from a compute shader, so its reads must be ordered after the depth writes.
    else if (oldLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal && newLayout == vk::ImageLayout::eDepthReadOnlyOptimal) {
        // ColorAttachmentOutput in src: same MSAA-depth-resolve reasoning as the ShaderReadOnly case.
        barrier.srcAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite | vk::AccessFlagBits2::eColorAttachmentWrite;
        barrier.dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eShaderRead;
        barrier.srcStageMask = vk::PipelineStageFlagBits2::eLateFragmentTests | vk::PipelineStageFlagBits2::eColorAttachmentOutput;
        barrier.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader | vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eComputeShader;
        barrier.subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eDepth, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1};
    }

    else if (oldLayout == vk::ImageLayout::eDepthReadOnlyOptimal && newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
        // ColorAttachmentOutput in dst: same MSAA-depth-resolve reasoning as the restore case above.
        barrier.srcAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eShaderRead;
        barrier.dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite | vk::AccessFlagBits2::eColorAttachmentWrite;
        // Compute in src: next frame's shadow writes must wait for this frame's compute reads (WAR).
        barrier.srcStageMask = vk::PipelineStageFlagBits2::eFragmentShader | vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eComputeShader;
        barrier.dstStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eColorAttachmentOutput;
        barrier.subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eDepth, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1};
    }

    // Depth-read-only <-> generic shader-read-only. Read->read: contents preserved, only the
    // layout/visibility changes. Needed when a depth image's descriptor records the generic
    // layout; the shadow atlas no longer does (its descriptor is eDepthReadOnlyOptimal, which
    // is valid for sampling in any stage and keeps Z-compression), so these are currently unused.
    else if (oldLayout == vk::ImageLayout::eDepthReadOnlyOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
        barrier.srcAccessMask = vk::AccessFlagBits2::eShaderRead;
        barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
        barrier.srcStageMask = vk::PipelineStageFlagBits2::eFragmentShader | vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests;
        barrier.dstStageMask = vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eFragmentShader;
        barrier.subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eDepth, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1};
    }

    else if (oldLayout == vk::ImageLayout::eShaderReadOnlyOptimal && newLayout == vk::ImageLayout::eDepthReadOnlyOptimal) {
        barrier.srcAccessMask = vk::AccessFlagBits2::eShaderRead;
        barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eDepthStencilAttachmentRead;
        barrier.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eFragmentShader;
        barrier.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader | vk::PipelineStageFlagBits2::eEarlyFragmentTests;
        barrier.subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eDepth, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1};
    }

    else if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eColorAttachmentOptimal) {
        barrier.srcAccessMask = {};
        barrier.dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
        // ColorAttachmentOutput, not TopOfPipe: the swapchain acquire semaphore is waited at
        // ColorAttachmentOutput, and the transition must chain after it.
        barrier.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
        barrier.dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
        barrier.subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1};
    }

    else if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
        barrier.srcAccessMask = {};
        barrier.dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
        barrier.srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
        barrier.dstStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests;
        barrier.subresourceRange = {
            .aspectMask = vk::ImageAspectFlagBits::eDepth, .baseMipLevel = 0, .levelCount = mipLevelCount, .baseArrayLayer = 0, .layerCount = layerCount};
    }

    else if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eDepthReadOnlyOptimal) {
        barrier.srcAccessMask = {};
        barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
        barrier.srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
        // Compute in dst: this seeds the shadow atlas, which the froxel light pass samples from compute.
        barrier.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader | vk::PipelineStageFlagBits2::eComputeShader;
        barrier.subresourceRange = {
            .aspectMask = vk::ImageAspectFlagBits::eDepth, .baseMipLevel = 0, .levelCount = mipLevelCount, .baseArrayLayer = 0, .layerCount = layerCount};
    }

    else if (oldLayout == vk::ImageLayout::eColorAttachmentOptimal && newLayout == vk::ImageLayout::ePresentSrcKHR) {
        barrier.srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
        barrier.dstAccessMask = {};
        barrier.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
        barrier.dstStageMask = vk::PipelineStageFlagBits2::eBottomOfPipe;
        barrier.subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1};
    }

    else if (oldLayout == vk::ImageLayout::eColorAttachmentOptimal && newLayout == vk::ImageLayout::eTransferSrcOptimal) {
        barrier.srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
        barrier.dstAccessMask = vk::AccessFlagBits2::eTransferRead;
        barrier.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
        barrier.dstStageMask = vk::PipelineStageFlagBits2::eTransfer;
        barrier.subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1};
    }

    else if (oldLayout == vk::ImageLayout::eColorAttachmentOptimal && newLayout == vk::ImageLayout::eTransferDstOptimal) {
        barrier.srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
        barrier.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
        barrier.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
        barrier.dstStageMask = vk::PipelineStageFlagBits2::eTransfer;
        barrier.subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1};
    }

    else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eColorAttachmentOptimal) {
        barrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite;
        barrier.srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
        barrier.dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
        barrier.subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1};
    }

    else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eTransferSrcOptimal) {
        barrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits2::eTransferRead;
        barrier.srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
        barrier.dstStageMask = vk::PipelineStageFlagBits2::eTransfer;
    }

    else if (oldLayout == vk::ImageLayout::eTransferSrcOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
        barrier.srcAccessMask = vk::AccessFlagBits2::eTransferRead;
        barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
        barrier.srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
        barrier.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader;
    }

    else if (oldLayout == vk::ImageLayout::eShaderReadOnlyOptimal && newLayout == vk::ImageLayout::eColorAttachmentOptimal) {
        // ColorAttachmentOutput in src: if no pass sampled the target since it was last rendered
        // (a skipped post pass), the access to order against is the attachment write itself.
        barrier.srcAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eColorAttachmentWrite;
        barrier.dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
        barrier.srcStageMask = vk::PipelineStageFlagBits2::eFragmentShader | vk::PipelineStageFlagBits2::eColorAttachmentOutput;
        barrier.dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
        barrier.subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1};
    }

    else if (oldLayout == vk::ImageLayout::eColorAttachmentOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
        barrier.srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
        barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
        barrier.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
        barrier.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader;
        barrier.subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1};
    }

    else if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
        barrier.srcAccessMask = {};
        barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
        barrier.srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
        barrier.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader;
        barrier.subresourceRange = {
            .aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = mipLevelCount, .baseArrayLayer = 0, .layerCount = layerCount};
    }

    else if (oldLayout == vk::ImageLayout::eShaderReadOnlyOptimal && newLayout == vk::ImageLayout::eTransferDstOptimal) {
        // ColorAttachmentOutput in src: thumbnails are rendered, then immediately transitioned for
        // mip generation — the attachment write must finish before the transfer overwrites.
        barrier.srcAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eColorAttachmentWrite;
        barrier.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
        barrier.srcStageMask = vk::PipelineStageFlagBits2::eFragmentShader | vk::PipelineStageFlagBits2::eColorAttachmentOutput;
        barrier.dstStageMask = vk::PipelineStageFlagBits2::eTransfer;
    }

    else if (oldLayout == vk::ImageLayout::eShaderReadOnlyOptimal && newLayout == vk::ImageLayout::eTransferSrcOptimal) {
        barrier.srcAccessMask = vk::AccessFlagBits2::eShaderRead;
        barrier.dstAccessMask = vk::AccessFlagBits2::eTransferRead;
        barrier.srcStageMask = vk::PipelineStageFlagBits2::eFragmentShader;
        barrier.dstStageMask = vk::PipelineStageFlagBits2::eTransfer;
    }

    // Sampled read -> compute storage access. Used for compute output images. Keeps the caller's
    // subresource range — the voxel volume transitions its whole mip chain, not just level 0.
    // Compute in src: the voxel cube extract samples the volume from compute (WAR).
    else if (oldLayout == vk::ImageLayout::eShaderReadOnlyOptimal && newLayout == vk::ImageLayout::eGeneral) {
        barrier.srcAccessMask = vk::AccessFlagBits2::eShaderRead;
        barrier.dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite;
        barrier.srcStageMask = vk::PipelineStageFlagBits2::eFragmentShader | vk::PipelineStageFlagBits2::eComputeShader;
        barrier.dstStageMask = vk::PipelineStageFlagBits2::eComputeShader;
    }

    // Compute storage write -> sampled read. Used after a compute output dispatch. Compute in dst:
    // the voxel cube extract samples from compute.
    else if (oldLayout == vk::ImageLayout::eGeneral && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
        barrier.srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite;
        barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
        barrier.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader;
        barrier.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader | vk::PipelineStageFlagBits2::eComputeShader;
    }

    else {
        throw std::invalid_argument("unsupported layout transition!");
    }

    return barrier;
}

inline void executeImageTransition(vk::raii::CommandBuffer& cmd, const vk::Image& image, const vk::ImageLayout oldLayout, const vk::ImageLayout newLayout, uint32_t baseMipLevel,
                                   uint32_t mipLevelCount, uint32_t baseArrayLayer, uint32_t layerCount) {
    auto barrier = buildBarrier(image, oldLayout, newLayout, baseMipLevel, mipLevelCount, baseArrayLayer, layerCount);
    vk::DependencyInfo dependency_info = {.dependencyFlags = {}, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier};
    cmd.pipelineBarrier2(dependency_info);
}

inline void transitionImageLayout(const Context& ctx, vk::raii::CommandBuffer* commandBuffer, const vk::Image& image, const vk::ImageLayout oldLayout, const vk::ImageLayout newLayout,
                                  uint32_t baseMipLevel = 0, uint32_t mipLevelCount = 1, uint32_t baseArrayLayer = 0, uint32_t layerCount = 1) {
    if (commandBuffer == nullptr) {
        auto singleTimeCmdBuffer = beginSingleTimeCommands(ctx);
        executeImageTransition(singleTimeCmdBuffer, image, oldLayout, newLayout, baseMipLevel, mipLevelCount, baseArrayLayer, layerCount);
        endSingleTimeCommands(ctx, singleTimeCmdBuffer);
    } else {
        executeImageTransition(*commandBuffer, image, oldLayout, newLayout, baseMipLevel, mipLevelCount, baseArrayLayer, layerCount);
    }
}

inline void transitionImageLayouts(vk::raii::CommandBuffer& cmd, const std::vector<ImageTransitionInfo>& transitions) {
    std::vector<vk::ImageMemoryBarrier2> barriers;
    barriers.reserve(transitions.size());
    for (const auto& t : transitions) {
        barriers.push_back(buildBarrier(t.image, t.oldLayout, t.newLayout, t.baseMipLevel, t.mipLevelCount, t.baseArrayLayer, t.layerCount));
    }
    vk::DependencyInfo depInfo = {.dependencyFlags = {}, .imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size()), .pImageMemoryBarriers = barriers.data()};
    cmd.pipelineBarrier2(depInfo);
}

inline void copyBufferToImage(const Context& ctx, const vk::raii::Buffer& srcBuffer, const vk::raii::Image& dstImage, uint32_t width, uint32_t height, vk::BufferImageCopy* customRegion = nullptr) {
    vk::raii::CommandBuffer commandBuffer = beginSingleTimeCommands(ctx);
    if (customRegion == nullptr) {
        vk::BufferImageCopy region{.bufferOffset = 0,
                                   .bufferRowLength = 0,
                                   .bufferImageHeight = 0,
                                   .imageSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
                                   .imageOffset = {0, 0, 0},
                                   .imageExtent = {width, height, 1}};

        commandBuffer.copyBufferToImage(*srcBuffer, *dstImage, vk::ImageLayout::eTransferDstOptimal, region);
    } else {
        commandBuffer.copyBufferToImage(*srcBuffer, *dstImage, vk::ImageLayout::eTransferDstOptimal, *customRegion);
    }
    endSingleTimeCommands(ctx, commandBuffer);
}

inline void copyBufferToImageCubemap(const Context& ctx, const vk::raii::Buffer& buffer, const vk::raii::Image& image, uint32_t width, uint32_t height, uint32_t bytesPerPixel) {
    vk::raii::CommandBuffer commandBuffer = beginSingleTimeCommands(ctx);

    vk::DeviceSize faceSize = static_cast<vk::DeviceSize>(width) * height * bytesPerPixel;
    std::vector<vk::BufferImageCopy> regions;

    for (uint32_t face = 0; face < 6; face++) {
        vk::BufferImageCopy region{.bufferOffset = face * faceSize,
                                   .bufferRowLength = 0,
                                   .bufferImageHeight = 0,
                                   .imageSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = face, .layerCount = 1},
                                   .imageOffset = {0, 0, 0},
                                   .imageExtent = {width, height, 1}};
        regions.push_back(region);
    }

    commandBuffer.copyBufferToImage(*buffer, *image, vk::ImageLayout::eTransferDstOptimal, regions);

    endSingleTimeCommands(ctx, commandBuffer);
}

inline std::tuple<vk::raii::Buffer, vk::raii::DeviceMemory, void*> createIndirectDrawBuffer(const Context& ctx, uint32_t slotsPerFrame = 1) {

    vk::raii::Buffer indirectDrawBuffer = nullptr;
    vk::raii::DeviceMemory indirectDrawBufferMemory = nullptr;
    // Persistent indirect draw buffer: slotsPerFrame * MAX_FRAMES_IN_FLIGHT contiguous slots.
    // Callers needing multiple independent recordings per frame (e.g. per-light shadow passes)
    // pass slotsPerFrame > 1 so each recording writes into its own slot.
    vk::DeviceSize indirectBufferSize = sizeof(DrawIndexedIndirectCommand) * MAX_INDIRECT_COMMANDS * slotsPerFrame * MAX_FRAMES_IN_FLIGHT;
    vk::BufferCreateInfo indirectBufferInfo{.size = indirectBufferSize, .usage = vk::BufferUsageFlagBits::eIndirectBuffer, .sharingMode = vk::SharingMode::eExclusive};

    indirectDrawBuffer = vk::raii::Buffer(ctx.device.getDevice(), indirectBufferInfo);
    vk::MemoryRequirements indirectMemReqs = indirectDrawBuffer.getMemoryRequirements();
    uint32_t indirectMemType = findMemoryType(indirectMemReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, ctx.device);

    vk::MemoryAllocateFlagsInfo allocFlags{.flags = vk::MemoryAllocateFlagBits::eDeviceAddress };
    vk::MemoryAllocateInfo indirectAllocInfo{.pNext = allocFlags, .allocationSize = indirectMemReqs.size, .memoryTypeIndex = indirectMemType};

    indirectDrawBufferMemory = vk::raii::DeviceMemory(ctx.device.getDevice(), indirectAllocInfo);
    indirectDrawBuffer.bindMemory(*indirectDrawBufferMemory, 0);
    void* mappedPtr = indirectDrawBufferMemory.mapMemory(0, indirectBufferSize);
    return std::make_tuple(std::move(indirectDrawBuffer), std::move(indirectDrawBufferMemory), mappedPtr);
}

inline void createImage(const Context& ctx, uint32_t width, uint32_t height, uint32_t mipLevels, vk::SampleCountFlagBits numSamples, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage,
                        vk::MemoryPropertyFlags properties, vk::raii::Image& image, vk::raii::DeviceMemory& imageMemory, uint32_t arrayLayers = 1,
                        vk::ImageCreateFlagBits createFlags = vk::ImageCreateFlagBits{}) {

    vk::ImageCreateInfo imageInfo{.flags = createFlags,
                                  .imageType = vk::ImageType::e2D,
                                  .format = format,
                                  .extent = {width, height, 1},
                                  .mipLevels = mipLevels,
                                  .arrayLayers = arrayLayers,
                                  .samples = numSamples,
                                  .tiling = tiling,
                                  .usage = usage,
                                  .sharingMode = vk::SharingMode::eExclusive,
                                  .initialLayout = vk::ImageLayout::eUndefined};

    image = vk::raii::Image(ctx.device.getDevice(), imageInfo);
    vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
    vk::MemoryAllocateInfo allocInfo{.allocationSize = memRequirements.size, .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties, ctx.device)};

    imageMemory = vk::raii::DeviceMemory(ctx.device.getDevice(), allocInfo);
    image.bindMemory(imageMemory, 0);
}

inline void create3DImage(const Context& ctx, uint32_t width, uint32_t height, uint32_t depth, uint32_t mipLevels, vk::SampleCountFlagBits numSamples, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage,
                          vk::MemoryPropertyFlags properties, vk::raii::Image& image, vk::raii::DeviceMemory& imageMemory, uint32_t arrayLayers = 1,
                          vk::ImageCreateFlagBits createFlags = vk::ImageCreateFlagBits{}) {

    vk::ImageCreateInfo imageInfo{.flags = createFlags,
                                  .imageType = vk::ImageType::e3D,
                                  .format = format,
                                  .extent = {width, height, depth},
                                  .mipLevels = mipLevels,
                                  .arrayLayers = arrayLayers,
                                  .samples = numSamples,
                                  .tiling = tiling,
                                  .usage = usage,
                                  .sharingMode = vk::SharingMode::eExclusive,
                                  .initialLayout = vk::ImageLayout::eUndefined};

    image = vk::raii::Image(ctx.device.getDevice(), imageInfo);
    vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
    vk::MemoryAllocateInfo allocInfo{.allocationSize = memRequirements.size, .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties, ctx.device)};

    imageMemory = vk::raii::DeviceMemory(ctx.device.getDevice(), allocInfo);
    image.bindMemory(imageMemory, 0);
}

[[nodiscard]] inline vk::raii::ImageView createImageView(const Context& ctx, vk::raii::Image& image, vk::Format format, vk::ImageAspectFlags aspectFlags, uint32_t mipLevels = 1) {
    vk::ImageViewCreateInfo viewInfo{.image = image,
                                     .viewType = vk::ImageViewType::e2D,
                                     .format = format,
                                     .subresourceRange = {.aspectMask = aspectFlags, .baseMipLevel = 0, .levelCount = mipLevels, .baseArrayLayer = 0, .layerCount = 1}};
    return vk::raii::ImageView(ctx.device.getDevice(), viewInfo);
}

// 3D view over a volume. Used for froxel volumes bound as both RWTexture3D (storage) and Texture3D
// (sampled). Defaults to a single-mip view; the voxel grid passes levelCount for the sampled
// full-chain view and one baseMipLevel per level for the storage views the downsample writes through
// (an RWTexture3D binding must resolve to exactly one mip).
[[nodiscard]] inline vk::raii::ImageView create3DImageView(const Context& ctx, vk::raii::Image& image, vk::Format format, vk::ImageAspectFlags aspectFlags,
                                                           uint32_t baseMipLevel = 0, uint32_t levelCount = 1) {
    vk::ImageViewCreateInfo viewInfo{.image = image,
                                     .viewType = vk::ImageViewType::e3D,
                                     .format = format,
                                     .subresourceRange = {.aspectMask = aspectFlags, .baseMipLevel = baseMipLevel, .levelCount = levelCount, .baseArrayLayer = 0, .layerCount = 1}};
    return vk::raii::ImageView(ctx.device.getDevice(), viewInfo);
}

inline void uploadTextureData(const Context& ctx, const vk::raii::Image& image, const void* data, uint32_t width, uint32_t height, vk::Format format, bool isCubemap = false) {
    uint32_t bytesPerPixel = getBytesPerPixel(format);
    vk::DeviceSize imageSize = static_cast<vk::DeviceSize>(width) * height * bytesPerPixel;
    vk::DeviceSize totalSize = isCubemap ? imageSize * 6 : imageSize;

    // Create staging buffer
    vk::BufferCreateInfo bufferInfo{.size = totalSize, .usage = vk::BufferUsageFlagBits::eTransferSrc, .sharingMode = vk::SharingMode::eExclusive};
    vk::raii::Buffer stagingBuffer(ctx.device.getDevice(), bufferInfo);

    vk::MemoryRequirements memRequirements = stagingBuffer.getMemoryRequirements();
    vk::MemoryAllocateInfo allocInfo{
        .allocationSize = memRequirements.size,
        .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, ctx.device)};
    vk::raii::DeviceMemory stagingBufferMemory(ctx.device.getDevice(), allocInfo);
    stagingBuffer.bindMemory(*stagingBufferMemory, 0);

    // Copy data to staging buffer
    void* mappedData = stagingBufferMemory.mapMemory(0, totalSize);
    memcpy(mappedData, data, totalSize);
    stagingBufferMemory.unmapMemory();

    uint32_t layerCount = isCubemap ? 6 : 1;
    uint32_t mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1;

    transitionImageLayout(ctx, nullptr, image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, 0, mipLevels, 0, layerCount);

    // Copy buffer to image
    if (isCubemap) {
        copyBufferToImageCubemap(ctx, stagingBuffer, image, width, height, getBytesPerPixel(format));
    } else {
        copyBufferToImage(ctx, stagingBuffer, image, width, height);
    }
}

inline void generateMipmaps(const Context& ctx, vk::raii::Image& image, vk::Format imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels, uint32_t layerCount,
                            vk::raii::CommandBuffer* commandBuffer = nullptr, vk::ImageLayout srcLayout = vk::ImageLayout::eTransferDstOptimal,
                            vk::ImageLayout dstMipLayout = vk::ImageLayout::eTransferDstOptimal) {

    vk::FormatProperties formatProperties = ctx.device.getPhysicalDevice().getFormatProperties(imageFormat);
    if (!(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) {
        throw std::runtime_error("Texture image format does not support linear blitting!");
    }

    bool ownCommandBuffer = (commandBuffer == nullptr);
    vk::raii::CommandBuffer ownedCmd = nullptr;
    if (ownCommandBuffer) {
#if DEBUG == 1
        std::cout << "=== GENERATING MIPMAPS ===" << std::endl;
        std::cout << "Size: " << texWidth << "x" << texHeight << std::endl;
        std::cout << "Mip levels: " << mipLevels << std::endl;
        std::cout << "Layers: " << layerCount << std::endl;
#endif
        ownedCmd = beginSingleTimeCommands(ctx);
        commandBuffer = &ownedCmd;
    }

    vk::raii::CommandBuffer& cmd = *commandBuffer;
    int32_t mipWidth = texWidth;
    int32_t mipHeight = texHeight;

    for (uint32_t i = 1; i < mipLevels; i++) {
        // First iteration transitions from srcLayout, subsequent from eTransferDstOptimal
        vk::ImageLayout fromLayout = (i == 1) ? srcLayout : vk::ImageLayout::eTransferDstOptimal;
        transitionImageLayout(ctx, &cmd, *image, fromLayout, vk::ImageLayout::eTransferSrcOptimal, i - 1, 1, 0,
                              layerCount);

        // Transition destination mip to transfer dst if needed
        if (dstMipLayout != vk::ImageLayout::eTransferDstOptimal) {
            transitionImageLayout(ctx, &cmd, *image, dstMipLayout, vk::ImageLayout::eTransferDstOptimal, i, 1, 0, layerCount);
        }

        // Create separate blit for each layer
        std::vector<vk::ImageBlit> blits;
        for (uint32_t layer = 0; layer < layerCount; layer++) {
            vk::ImageBlit blit;
            blit.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
            blit.srcSubresource.mipLevel = i - 1;
            blit.srcSubresource.baseArrayLayer = layer;
            blit.srcSubresource.layerCount = 1;
            blit.srcOffsets[0] = vk::Offset3D{0, 0, 0};
            blit.srcOffsets[1] = vk::Offset3D{mipWidth, mipHeight, 1};

            blit.dstSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
            blit.dstSubresource.mipLevel = i;
            blit.dstSubresource.baseArrayLayer = layer;
            blit.dstSubresource.layerCount = 1;
            blit.dstOffsets[0] = vk::Offset3D{0, 0, 0};
            blit.dstOffsets[1] = vk::Offset3D{mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1};

            blits.push_back(blit);
        }

        cmd.blitImage(*image, vk::ImageLayout::eTransferSrcOptimal, *image, vk::ImageLayout::eTransferDstOptimal, blits, vk::Filter::eLinear);

        // Transition previous mip level (all layers) to shader read
        transitionImageLayout(ctx, &cmd, *image, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, i - 1, 1, 0, layerCount);

        if (mipWidth > 1)
            mipWidth /= 2;
        if (mipHeight > 1)
            mipHeight /= 2;
    }
    // finally transition the last mip level
    transitionImageLayout(ctx, &cmd, *image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, mipLevels - 1, 1, 0, layerCount);

    if (ownCommandBuffer) {
        endSingleTimeCommands(ctx, ownedCmd);
        ctx.device.getGraphicsQueue().waitIdle();
#if DEBUG == 1
        std::cout << "=== MIPMAPS COMPLETE ===" << std::endl;
#endif
    }
}

inline std::tuple<vk::raii::Image, vk::raii::DeviceMemory, vk::raii::ImageView> createTexture(const Context& ctx, const void* data, uint32_t width, uint32_t height, vk::Format format,
                                                                                              vk::ImageType imageType = vk::ImageType::e2D,
                                                                                              vk::ImageViewType viewType = vk::ImageViewType::e2D,
                                                                                              vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1, bool genMips = true) {
    bool isCubemap = (viewType == vk::ImageViewType::eCube);

    uint32_t mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1;
    // Create image
    vk::ImageCreateInfo imageInfo{.flags = viewType == vk::ImageViewType::eCube ? vk::ImageCreateFlagBits::eCubeCompatible : vk::ImageCreateFlags{},
                                  .imageType = imageType,
                                  .format = format,
                                  .extent = {width, height, 1},
                                  .mipLevels = mipLevels,
                                  .arrayLayers = viewType == vk::ImageViewType::eCube ? static_cast<uint32_t>(6) : static_cast<uint32_t>(1),
                                  .samples = samples,
                                  .tiling = vk::ImageTiling::eOptimal,
                                  .usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc,
                                  .sharingMode = vk::SharingMode::eExclusive,
                                  .initialLayout = vk::ImageLayout::eUndefined};

    vk::raii::Image image(ctx.device.getDevice(), imageInfo);

    // Allocate memory
    vk::MemoryRequirements memRequirements = image.getMemoryRequirements();

    vk::MemoryAllocateInfo allocInfo{.allocationSize = memRequirements.size,
                                     .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal, ctx.device)};

    vk::raii::DeviceMemory imageMemory(ctx.device.getDevice(), allocInfo);
    image.bindMemory(*imageMemory, 0);
    uploadTextureData(ctx, image, data, width, height, format, isCubemap);
    if (genMips) {
        generateMipmaps(ctx, image, format, width, height, mipLevels, isCubemap ? 6 : 1);
    }
    vk::ImageViewCreateInfo viewInfo{.image = *image,
                                     .viewType = viewType,
                                     .format = format,
                                     .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                                                          .baseMipLevel = 0,
                                                          .levelCount = mipLevels,
                                                          .baseArrayLayer = 0,
                                                          .layerCount = viewType == vk::ImageViewType::eCube ? static_cast<uint32_t>(6) : static_cast<uint32_t>(1)}};

    vk::raii::ImageView imageView(ctx.device.getDevice(), viewInfo);
    return std::make_tuple(std::move(image), std::move(imageMemory), std::move(imageView));
}

// A texture loaded from disk. mipLevels/ktxPath come from the .ktx2 cache; ktxPath is empty when
// the cache was unusable and the source image had to be decoded directly.
struct LoadedTexture {
    vk::raii::Image image = nullptr;
    vk::raii::DeviceMemory memory = nullptr;
    vk::raii::ImageView view = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t mipLevels = 1;
    std::string ktxPath;
};

inline bool isSrgbFormat(vk::Format format) {
    switch (format) {
    case vk::Format::eR8G8B8A8Srgb:
    case vk::Format::eB8G8R8A8Srgb:
    case vk::Format::eBc7SrgbBlock:
        return true;
    default:
        return false;
    }
}

// BC7 is core Vulkan but optional in the feature set, so a device can legitimately lack it.
// Checked once per load; the transcoder falls back to plain RGBA8 when it's missing.
inline bool supportsBC7(const Context& ctx) {
    vk::FormatProperties props = ctx.device.getPhysicalDevice().getFormatProperties(vk::Format::eBc7UnormBlock);
    return static_cast<bool>(props.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImage);
}

// Owns a ktxTexture2 for the duration of an upload.
struct KtxHandle {
    ktxTexture2* tex = nullptr;
    KtxHandle() = default;
    KtxHandle(const KtxHandle&) = delete;
    KtxHandle& operator=(const KtxHandle&) = delete;
    ~KtxHandle() {
        if (tex)
            ktxTexture_Destroy(ktxTexture(tex));
    }
};

// Transcodes a supercompressed KTX into something the device can sample and returns the format to
// create the image with. The caller's requested format only decides sRGB vs UNORM — the base format
// comes from what the file actually holds.
inline vk::Format transcodeToDeviceFormat(const Context& ctx, ktxTexture2* tex, vk::Format requested) {
    const bool wantSrgb = isSrgbFormat(requested);

    if (ktxTexture2_NeedsTranscoding(tex)) {
        const bool bc7 = supportsBC7(ctx);
        KTX_error_code result = ktxTexture2_TranscodeBasis(tex, bc7 ? KTX_TTF_BC7_RGBA : KTX_TTF_RGBA32, 0);
        if (result != KTX_SUCCESS) {
            throw std::runtime_error(std::string("ktx transcode failed: ") + ktxErrorString(result));
        }
        if (bc7)
            return wantSrgb ? vk::Format::eBc7SrgbBlock : vk::Format::eBc7UnormBlock;
        return wantSrgb ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm;
    }

    // Already a plain format — a hand-written .ktx2, or one from an uncompressed cache.
    switch (static_cast<vk::Format>(tex->vkFormat)) {
    case vk::Format::eR8G8B8A8Unorm:
    case vk::Format::eR8G8B8A8Srgb:
        return wantSrgb ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm;
    case vk::Format::eBc7UnormBlock:
    case vk::Format::eBc7SrgbBlock:
        return wantSrgb ? vk::Format::eBc7SrgbBlock : vk::Format::eBc7UnormBlock;
    default:
        return static_cast<vk::Format>(tex->vkFormat);
    }
}

// Uploads every level and face of an already-transcoded ktxTexture2 in one command buffer. The KTX
// carries a full mip chain, so unlike createTexture nothing is blitted here.
inline LoadedTexture uploadKtxTexture(const Context& ctx, ktxTexture2* tex, vk::Format format, vk::ImageViewType viewType) {
    const bool isCubemap = (viewType == vk::ImageViewType::eCube);
    const uint32_t layerCount = tex->numFaces; // 6 for a cubemap, 1 otherwise
    const uint32_t mipLevels = tex->numLevels;

    vk::ImageCreateInfo imageInfo{.flags = isCubemap ? vk::ImageCreateFlagBits::eCubeCompatible : vk::ImageCreateFlags{},
                                  .imageType = vk::ImageType::e2D,
                                  .format = format,
                                  .extent = {tex->baseWidth, tex->baseHeight, 1},
                                  .mipLevels = mipLevels,
                                  .arrayLayers = layerCount,
                                  .samples = vk::SampleCountFlagBits::e1,
                                  .tiling = vk::ImageTiling::eOptimal,
                                  .usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc,
                                  .sharingMode = vk::SharingMode::eExclusive,
                                  .initialLayout = vk::ImageLayout::eUndefined};

    vk::raii::Image image(ctx.device.getDevice(), imageInfo);
    vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
    vk::MemoryAllocateInfo allocInfo{.allocationSize = memRequirements.size,
                                     .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal, ctx.device)};
    vk::raii::DeviceMemory imageMemory(ctx.device.getDevice(), allocInfo);
    image.bindMemory(*imageMemory, 0);

    // Stage the whole level/face pyramid in one buffer — it is already laid out contiguously.
    const vk::DeviceSize dataSize = ktxTexture_GetDataSize(ktxTexture(tex));
    vk::BufferCreateInfo bufferInfo{.size = dataSize, .usage = vk::BufferUsageFlagBits::eTransferSrc, .sharingMode = vk::SharingMode::eExclusive};
    vk::raii::Buffer stagingBuffer(ctx.device.getDevice(), bufferInfo);
    vk::MemoryRequirements stagingRequirements = stagingBuffer.getMemoryRequirements();
    vk::MemoryAllocateInfo stagingAllocInfo{
        .allocationSize = stagingRequirements.size,
        .memoryTypeIndex = findMemoryType(stagingRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, ctx.device)};
    vk::raii::DeviceMemory stagingMemory(ctx.device.getDevice(), stagingAllocInfo);
    stagingBuffer.bindMemory(*stagingMemory, 0);

    void* mappedData = stagingMemory.mapMemory(0, dataSize);
    memcpy(mappedData, ktxTexture_GetData(ktxTexture(tex)), dataSize);
    stagingMemory.unmapMemory();

    // bufferRowLength/bufferImageHeight stay 0: each image is tightly packed, so the driver derives
    // the pitch from imageExtent — which is what block-compressed mips below 4x4 need.
    std::vector<vk::BufferImageCopy> regions;
    regions.reserve(static_cast<size_t>(mipLevels) * layerCount);
    for (uint32_t level = 0; level < mipLevels; level++) {
        for (uint32_t face = 0; face < layerCount; face++) {
            ktx_size_t offset = 0;
            if (ktxTexture_GetImageOffset(ktxTexture(tex), level, 0, face, &offset) != KTX_SUCCESS) {
                throw std::runtime_error("ktx: failed to resolve image offset");
            }
            regions.push_back(vk::BufferImageCopy{
                .bufferOffset = offset,
                .bufferRowLength = 0,
                .bufferImageHeight = 0,
                .imageSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = level, .baseArrayLayer = face, .layerCount = 1},
                .imageOffset = {0, 0, 0},
                .imageExtent = {std::max(1u, tex->baseWidth >> level), std::max(1u, tex->baseHeight >> level), 1}});
        }
    }

    vk::raii::CommandBuffer commandBuffer = beginSingleTimeCommands(ctx);
    executeImageTransition(commandBuffer, *image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, 0, mipLevels, 0, layerCount);
    commandBuffer.copyBufferToImage(*stagingBuffer, *image, vk::ImageLayout::eTransferDstOptimal, regions);
    executeImageTransition(commandBuffer, *image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, 0, mipLevels, 0, layerCount);
    endSingleTimeCommands(ctx, commandBuffer);

    vk::ImageViewCreateInfo viewInfo{
        .image = *image,
        .viewType = viewType,
        .format = format,
        .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = mipLevels, .baseArrayLayer = 0, .layerCount = layerCount}};

    LoadedTexture loaded;
    loaded.image = std::move(image);
    loaded.memory = std::move(imageMemory);
    loaded.view = vk::raii::ImageView(ctx.device.getDevice(), viewInfo);
    loaded.width = tex->baseWidth;
    loaded.height = tex->baseHeight;
    loaded.mipLevels = mipLevels;
    return loaded;
}

inline LoadedTexture loadKtxFile(const Context& ctx, const std::string& ktxPath, vk::Format format, vk::ImageViewType viewType) {
    KtxHandle handle;
    KTX_error_code result = ktxTexture2_CreateFromNamedFile(ktxPath.c_str(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &handle.tex);
    if (result != KTX_SUCCESS) {
        throw std::runtime_error("failed to open " + ktxPath + ": " + ktxErrorString(result));
    }
    vk::Format imageFormat = transcodeToDeviceFormat(ctx, handle.tex, format);
    LoadedTexture loaded = uploadKtxTexture(ctx, handle.tex, imageFormat, viewType);
    loaded.ktxPath = ktxPath;
    return loaded;
}

/*
Loads a texture through its .ktx2 cache, converting the source image first if the cache is missing
or stale. A cache that exists but won't load is re-encoded once; if that fails too the source is
decoded straight to an RGBA8 texture so a broken encoder still leaves the scene renderable.
*/
inline LoadedTexture loadTextureFromFile(const Context& ctx, const std::string& path, vk::Format format = vk::Format::eR8G8B8A8Srgb,
                                         textureconv::ColorSpace colorSpace = textureconv::ColorSpace::Auto) {
    if (colorSpace == textureconv::ColorSpace::Auto) {
        colorSpace = isSrgbFormat(format) ? textureconv::ColorSpace::Srgb : textureconv::ColorSpace::Linear;
    }
    const std::string ktxPath = textureconv::ktxPathFor(path);
    const int attempts = textureconv::isKtxPath(path) ? 1 : 2;

    for (int attempt = 0; attempt < attempts; attempt++) {
        const bool force = (attempt > 0);
        try {
            if (force || textureconv::needsConversion(path)) {
                textureconv::convert(path, colorSpace, force);
            }
            return loadKtxFile(ctx, ktxPath, format, vk::ImageViewType::e2D);
        } catch (const std::exception& e) {
            std::cerr << "[ktx] " << path << ": " << e.what() << (attempt + 1 < attempts ? " - re-converting" : " - decoding source directly") << std::endl;
        }
    }

    textureconv::SourceImage source = textureconv::loadSourceImage(path);
    auto [image, memory, imageView] = createTexture(ctx, source.pixels.data(), source.width, source.height, format);

    LoadedTexture loaded;
    loaded.image = std::move(image);
    loaded.memory = std::move(memory);
    loaded.view = std::move(imageView);
    loaded.width = source.width;
    loaded.height = source.height;
    loaded.mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(source.width, source.height)))) + 1;
    return loaded;
}

// Faces are given in Vulkan/KTX layer order: +X -X +Y -Y +Z -Z. Dimensions come from the file, so
// unlike the old stb path there is nothing to enter by hand.
inline LoadedTexture loadCubeMapFromFile(const Context& ctx, const std::string& posX, const std::string& negX, const std::string& posY, const std::string& negY,
                                         const std::string& posZ, const std::string& negZ) {
    const std::vector<std::string> faceFiles = {posX, negX, posY, negY, posZ, negZ};
    const std::string ktxPath = textureconv::cubemapKtxPathFor(posX);
    const int attempts = textureconv::isKtxPath(posX) ? 1 : 2;

    for (int attempt = 0; attempt < attempts; attempt++) {
        const bool force = (attempt > 0);
        try {
            if (force || textureconv::cubemapNeedsConversion(faceFiles)) {
                textureconv::convertCubemap(faceFiles, textureconv::ColorSpace::Srgb, force);
            }
            return loadKtxFile(ctx, ktxPath, vk::Format::eR8G8B8A8Srgb, vk::ImageViewType::eCube);
        } catch (const std::exception& e) {
            std::cerr << "[ktx] " << posX << ": " << e.what() << (attempt + 1 < attempts ? " - re-converting" : " - decoding faces directly") << std::endl;
        }
    }

    // Fallback: decode the 6 faces into one buffer and use the runtime mipmap path.
    std::vector<unsigned char> allFaceData;
    uint32_t width = 0;
    uint32_t height = 0;
    for (int face = 0; face < 6; face++) {
        textureconv::SourceImage source = textureconv::loadSourceImage(faceFiles[face]);
        if (face == 0) {
            width = source.width;
            height = source.height;
            allFaceData.resize(static_cast<size_t>(width) * height * 4 * 6);
        } else if (source.width != width || source.height != height) {
            throw std::runtime_error("Cubemap face size mismatch: " + faceFiles[face]);
        }
        memcpy(allFaceData.data() + static_cast<size_t>(face) * width * height * 4, source.pixels.data(), source.pixels.size());
    }
    auto [image, memory, imageView] = createTexture(ctx, allFaceData.data(), width, height, vk::Format::eR8G8B8A8Srgb, vk::ImageType::e2D, vk::ImageViewType::eCube);

    LoadedTexture loaded;
    loaded.image = std::move(image);
    loaded.memory = std::move(memory);
    loaded.view = std::move(imageView);
    loaded.width = width;
    loaded.height = height;
    loaded.mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1;
    return loaded;
}

// Case-insensitive suffix test, so Model.GLTF and model.gltf reach the same parser.
inline bool hasExtension(const std::string& path, const std::string& extension) {
    if (path.size() < extension.size()) return false;
    const size_t offset = path.size() - extension.size();
    for (size_t i = 0; i < extension.size(); i++) {
        char a = path[offset + i];
        char b = extension[i];
        if (a >= 'A' && a <= 'Z') a = static_cast<char>(a - 'A' + 'a');
        if (b >= 'A' && b <= 'Z') b = static_cast<char>(b - 'A' + 'a');
        if (a != b) return false;
    }
    return true;
}

// Midpoint of the vertex AABB — what both loaders use for MeshEntry::origin.
inline glm::vec3 boundsCenter(const std::vector<Vertex>& vertices) {
    if (vertices.empty()) return glm::vec3(0.0f);
    glm::vec3 min(std::numeric_limits<float>::max());
    glm::vec3 max(std::numeric_limits<float>::lowest());
    for (const Vertex& vertex : vertices) {
        min = glm::min(min, vertex.position);
        max = glm::max(max, vertex.position);
    }
    return (min + max) * 0.5f;
}

// Per-vertex tangent from the UV gradient across each adjoining triangle, area-weighted by the
// accumulation. Degenerate UVs contribute +X so nothing is left with a zero-length tangent.
inline void generateTangents(std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices) {
    for (auto& vertex : vertices) {
        vertex.tangent = glm::vec3(0.0f);
    }

    for (size_t i = 0; i + 2 < indices.size(); i += 3) {
        uint32_t idx0 = indices[i];
        uint32_t idx1 = indices[i + 1];
        uint32_t idx2 = indices[i + 2];
        if (idx0 >= vertices.size() || idx1 >= vertices.size() || idx2 >= vertices.size()) continue;

        Vertex& v0 = vertices[idx0];
        Vertex& v1 = vertices[idx1];
        Vertex& v2 = vertices[idx2];

        glm::vec3 edge1 = v1.position - v0.position;
        glm::vec3 edge2 = v2.position - v0.position;

        glm::vec2 deltaUV1 = v1.texCoord - v0.texCoord;
        glm::vec2 deltaUV2 = v2.texCoord - v0.texCoord;

        float denominator = deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y;

        glm::vec3 tangent;
        if (abs(denominator) < 0.0001f) {
            tangent = glm::vec3(1.0f, 0.0f, 0.0f);
        } else {
            float f = 1.0f / denominator;
            tangent.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
            tangent.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
            tangent.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);
            tangent = normalize(tangent);
        }

        v0.tangent += tangent;
        v1.tangent += tangent;
        v2.tangent += tangent;
    }

    for (auto& vertex : vertices) {
        if (glm::length(vertex.tangent) > 0.0001f) {
            vertex.tangent = normalize(vertex.tangent);
        } else {
            vertex.tangent = glm::vec3(1.0f, 0.0f, 0.0f);
        }
    }
}

inline MeshData loadMeshFromFileOBJ(const std::string& meshPath) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    // Extract directory path for loading .mtl file
    std::string mtlBaseDir = meshPath.substr(0, meshPath.find_last_of("/\\") + 1);

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, meshPath.c_str(), mtlBaseDir.c_str())) {
        std::cerr << "[obj] failed to load " << meshPath << ": " << warn << err << std::endl;
        return {};
    }
    if (!warn.empty()) std::cerr << "[obj] " << meshPath << ": " << warn << std::endl;
    MeshData meshData = {};

    // Process each shape and split by material
    for (size_t shapeIdx = 0; shapeIdx < shapes.size(); shapeIdx++) {
        const auto& shape = shapes[shapeIdx];

        // Group faces by material ID
        std::map<int, std::vector<size_t>> facesPerMaterial;
        size_t numFaces = shape.mesh.material_ids.size();
        for (size_t faceIdx = 0; faceIdx < numFaces; faceIdx++) {
            int matId = shape.mesh.material_ids.empty() ? -1 : shape.mesh.material_ids[faceIdx];
            facesPerMaterial[matId].push_back(faceIdx);
        }

        // Create a mesh entry for each material group
        for (const auto& [matId, faceIndices] : facesPerMaterial) {
            MeshEntry entry;
            entry.materialId = matId;
            entry.shapeName = shape.name;
            // The shape is the object; the split below is ours, not the file's.
            entry.sourceObject = static_cast<uint32_t>(shapeIdx);
            entry.objectName = shape.name;
            std::unordered_map<Vertex, uint32_t> uniqueVertices{};

            if (matId >= 0 && matId < static_cast<int>(materials.size())) {
                entry.materialName = materials[matId].name;
            }

            for (size_t faceIdx : faceIndices) {
                for (size_t v = 0; v < 3; v++) {
                    size_t indexIdx = faceIdx * 3 + v;
                    if (indexIdx >= shape.mesh.indices.size())
                        continue;

                    const auto& index = shape.mesh.indices[indexIdx];
                    Vertex vertex{};

                    if (index.vertex_index >= 0 && index.vertex_index < static_cast<int>(attrib.vertices.size() / 3)) {
                        vertex.position = {attrib.vertices[3 * index.vertex_index + 0], attrib.vertices[3 * index.vertex_index + 1],
                                           attrib.vertices[3 * index.vertex_index + 2]};
                    }

                    if (index.normal_index >= 0 && index.normal_index < static_cast<int>(attrib.normals.size() / 3)) {
                        vertex.normal = {attrib.normals[3 * index.normal_index + 0], attrib.normals[3 * index.normal_index + 1], attrib.normals[3 * index.normal_index + 2]};
                    } else {
                        vertex.normal = {0.0f, 1.0f, 0.0f};
                    }

                    if (index.texcoord_index >= 0 && index.texcoord_index < static_cast<int>(attrib.texcoords.size() / 2)) {
                        vertex.texCoord = {attrib.texcoords[2 * index.texcoord_index + 0], 1.0f - attrib.texcoords[2 * index.texcoord_index + 1]};
                    } else {
                        vertex.texCoord = {0.0f, 0.0f};
                    }

                    if (uniqueVertices.count(vertex) == 0) {
                        uniqueVertices[vertex] = static_cast<uint32_t>(entry.vertices.size());
                        entry.vertices.push_back(vertex);
                    }

                    entry.indices.push_back(uniqueVertices[vertex]);
                }
            }

            // Calculate tangents for this entry
            generateTangents(entry.vertices, entry.indices);
            // OBJ has no pivots of its own, so the geometry's own centre is the best one available.
            entry.origin = boundsCenter(entry.vertices);

            meshData.entries.push_back(std::move(entry));
        }
    }

    return meshData;
}

/*
glTF reading. The entries this produces look exactly like OBJ entries — geometry already sitting
where the file puts it, one entry per material group — so the importer's centring, LOD generation
and instance detection all run unchanged. What glTF adds is that it *says* when two placements are
the same geometry, and those entries come back marked (MeshEntry::instanceOf) with no vertices
attached, so the importer can skip straight to the placement instead of rediscovering it.
*/
namespace gltf {

// tiny_gltf refuses a file whose images it can't load, and its stock loader would decode every
// texture in the model — all of which we throw away, since materials reach the engine as paths and
// go through the KTX cache. Claiming success without touching the bytes skips both.
inline bool skipImageLoad(tinygltf::Image*, const int, std::string*, std::string*, int, int, const unsigned char*, int, void*) { return true; }

// glTF hands attributes over as strided, arbitrarily typed views into a buffer. This resolves one
// to the bytes it addresses; `stride` is not the element size when a view interleaves attributes.
struct AccessorView {
    const unsigned char* data = nullptr;
    size_t count = 0;
    int stride = 0;
    int components = 0;
    int componentType = 0;
    bool normalized = false;
};

inline bool resolveAccessor(const tinygltf::Model& model, int accessorIndex, AccessorView& out) {
    if (accessorIndex < 0 || accessorIndex >= static_cast<int>(model.accessors.size())) return false;
    const tinygltf::Accessor& accessor = model.accessors[accessorIndex];
    if (accessor.bufferView < 0 || accessor.bufferView >= static_cast<int>(model.bufferViews.size())) return false;
    const tinygltf::BufferView& view = model.bufferViews[accessor.bufferView];
    if (view.buffer < 0 || view.buffer >= static_cast<int>(model.buffers.size())) return false;
    const tinygltf::Buffer& buffer = model.buffers[view.buffer];

    const int components = tinygltf::GetNumComponentsInType(static_cast<uint32_t>(accessor.type));
    const int componentSize = tinygltf::GetComponentSizeInBytes(static_cast<uint32_t>(accessor.componentType));
    const int stride = accessor.ByteStride(view);
    if (components <= 0 || componentSize <= 0 || stride <= 0) return false;

    // Everything past this point indexes raw bytes, so the whole span is bounds-checked once here.
    const size_t start = view.byteOffset + accessor.byteOffset;
    const size_t span = accessor.count == 0 ? 0
                      : (accessor.count - 1) * static_cast<size_t>(stride) + static_cast<size_t>(componentSize) * components;
    if (start > buffer.data.size() || span > buffer.data.size() - start) return false;

    out = {buffer.data.data() + start, accessor.count, stride, components, accessor.componentType, accessor.normalized};
    return true;
}

// Reads an accessor as `wantComponents` packed floats per element, zero-filling the components a
// narrower accessor doesn't carry. Normalised integer types are the ones KHR_mesh_quantization
// emits; without this they would arrive as raw counts.
inline bool readAccessorFloats(const tinygltf::Model& model, int accessorIndex, int wantComponents, std::vector<float>& out) {
    AccessorView view;
    if (!resolveAccessor(model, accessorIndex, view)) return false;

    out.assign(view.count * static_cast<size_t>(wantComponents), 0.0f);
    const int copy = std::min(view.components, wantComponents);
    const int componentSize = tinygltf::GetComponentSizeInBytes(static_cast<uint32_t>(view.componentType));
    for (size_t i = 0; i < view.count; i++) {
        const unsigned char* element = view.data + i * static_cast<size_t>(view.stride);
        for (int c = 0; c < copy; c++) {
            const unsigned char* raw = element + c * componentSize;
            float value = 0.0f;
            switch (view.componentType) {
            case TINYGLTF_COMPONENT_TYPE_FLOAT: { float v; memcpy(&v, raw, 4); value = v; break; }
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: { uint8_t v; memcpy(&v, raw, 1); value = view.normalized ? v / 255.0f : static_cast<float>(v); break; }
            case TINYGLTF_COMPONENT_TYPE_BYTE: { int8_t v; memcpy(&v, raw, 1); value = view.normalized ? std::max(v / 127.0f, -1.0f) : static_cast<float>(v); break; }
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: { uint16_t v; memcpy(&v, raw, 2); value = view.normalized ? v / 65535.0f : static_cast<float>(v); break; }
            case TINYGLTF_COMPONENT_TYPE_SHORT: { int16_t v; memcpy(&v, raw, 2); value = view.normalized ? std::max(v / 32767.0f, -1.0f) : static_cast<float>(v); break; }
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: { uint32_t v; memcpy(&v, raw, 4); value = static_cast<float>(v); break; }
            default: return false;
            }
            out[i * wantComponents + c] = value;
        }
    }
    return true;
}

inline bool readAccessorIndices(const tinygltf::Model& model, int accessorIndex, std::vector<uint32_t>& out) {
    AccessorView view;
    if (!resolveAccessor(model, accessorIndex, view) || view.components != 1) return false;

    out.assign(view.count, 0u);
    for (size_t i = 0; i < view.count; i++) {
        const unsigned char* raw = view.data + i * static_cast<size_t>(view.stride);
        switch (view.componentType) {
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: { uint8_t v; memcpy(&v, raw, 1); out[i] = v; break; }
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: { uint16_t v; memcpy(&v, raw, 2); out[i] = v; break; }
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: { uint32_t v; memcpy(&v, raw, 4); out[i] = v; break; }
        default: return false;
        }
    }
    return true;
}

// A node gives either a full matrix or a TRS triple. Both are column-major the way glm is; only
// the quaternion differs, glTF storing xyzw where glm's constructor takes w first.
inline glm::mat4 nodeLocalTransform(const tinygltf::Node& node) {
    if (node.matrix.size() == 16) {
        glm::mat4 matrix(1.0f);
        for (int column = 0; column < 4; column++) {
            for (int row = 0; row < 4; row++) matrix[column][row] = static_cast<float>(node.matrix[column * 4 + row]);
        }
        return matrix;
    }
    glm::vec3 translation(0.0f);
    glm::quat rotation(1.0f, 0.0f, 0.0f, 0.0f);
    glm::vec3 scale(1.0f);
    if (node.translation.size() == 3)
        translation = {static_cast<float>(node.translation[0]), static_cast<float>(node.translation[1]), static_cast<float>(node.translation[2])};
    if (node.rotation.size() == 4)
        rotation = {static_cast<float>(node.rotation[3]), static_cast<float>(node.rotation[0]), static_cast<float>(node.rotation[1]), static_cast<float>(node.rotation[2])};
    if (node.scale.size() == 3)
        scale = {static_cast<float>(node.scale[0]), static_cast<float>(node.scale[1]), static_cast<float>(node.scale[2])};
    return makeTransform(translation, rotation, scale);
}

// True when the linear part of `m` is a plain rotation — no scale, no shear, no reflection. Those
// are the only copies a node can carry, since a node holds a position and a rotation and the
// importer drops the rest. Anything else has to be baked into geometry.
inline bool isRigidNoFlip(const glm::mat4& m, float epsilon = 1e-4f) {
    const glm::mat3 linear(m);
    if (glm::determinant(linear) <= 0.0f) return false;
    const glm::mat3 orthogonality = glm::transpose(linear) * linear;
    for (int column = 0; column < 3; column++) {
        for (int row = 0; row < 3; row++) {
            if (std::abs(orthogonality[column][row] - (column == row ? 1.0f : 0.0f)) > epsilon) return false;
        }
    }
    return true;
}

// Strips alternate winding every other triangle; fans all share corner 0. Both are rare but legal,
// and unrolling them here keeps everything downstream on plain triangle lists.
inline std::vector<uint32_t> trianglesFromStrip(const std::vector<uint32_t>& strip) {
    std::vector<uint32_t> triangles;
    if (strip.size() < 3) return triangles;
    triangles.reserve((strip.size() - 2) * 3);
    for (size_t i = 0; i + 2 < strip.size(); i++) {
        triangles.push_back(strip[i]);
        triangles.push_back(strip[i + (i % 2 == 0 ? 1 : 2)]);
        triangles.push_back(strip[i + (i % 2 == 0 ? 2 : 1)]);
    }
    return triangles;
}

inline std::vector<uint32_t> trianglesFromFan(const std::vector<uint32_t>& fan) {
    std::vector<uint32_t> triangles;
    if (fan.size() < 3) return triangles;
    triangles.reserve((fan.size() - 2) * 3);
    for (size_t i = 1; i + 1 < fan.size(); i++) {
        triangles.push_back(fan[0]);
        triangles.push_back(fan[i]);
        triangles.push_back(fan[i + 1]);
    }
    return triangles;
}

// One placement of one glTF mesh: the world matrix the node graph resolved to, plus the name to
// hang off the entries it produces.
struct MeshPlacement {
    int meshIndex = -1;
    glm::mat4 world{1.0f};
    std::string nodeName;
};

// EXT_mesh_gpu_instancing puts many copies of a node's mesh in TRANSLATION/ROTATION/SCALE
// accessors. Without it a node is simply one copy at its own transform.
inline std::vector<glm::mat4> instanceTransforms(const tinygltf::Model& model, const tinygltf::Node& node, const glm::mat4& world) {
    auto extension = node.extensions.find("EXT_mesh_gpu_instancing");
    if (extension == node.extensions.end() || !extension->second.Has("attributes")) return {world};
    const tinygltf::Value& attributes = extension->second.Get("attributes");

    std::vector<float> translations, rotations, scales;
    size_t count = 0;
    auto read = [&](const char* name, int components, std::vector<float>& values) {
        if (!attributes.Has(name)) return;
        if (!readAccessorFloats(model, attributes.Get(name).GetNumberAsInt(), components, values)) {
            values.clear();
            return;
        }
        count = std::max(count, values.size() / static_cast<size_t>(components));
    };
    read("TRANSLATION", 3, translations);
    read("ROTATION", 4, rotations);
    read("SCALE", 3, scales);
    if (count == 0) return {world};

    std::vector<glm::mat4> transforms;
    transforms.reserve(count);
    for (size_t i = 0; i < count; i++) {
        glm::vec3 translation(0.0f);
        glm::quat rotation(1.0f, 0.0f, 0.0f, 0.0f);
        glm::vec3 scale(1.0f);
        if (i * 3 + 2 < translations.size()) translation = {translations[i * 3], translations[i * 3 + 1], translations[i * 3 + 2]};
        if (i * 4 + 3 < rotations.size()) rotation = {rotations[i * 4 + 3], rotations[i * 4], rotations[i * 4 + 1], rotations[i * 4 + 2]};
        if (i * 3 + 2 < scales.size()) scale = {scales[i * 3], scales[i * 3 + 1], scales[i * 3 + 2]};
        transforms.push_back(world * makeTransform(translation, rotation, scale));
    }
    return transforms;
}

// Walks the node graph accumulating transforms. `visited` guards against the cycles a malformed
// file can contain — invalid glTF, but cheap to survive rather than recurse forever on.
inline void collectPlacements(const tinygltf::Model& model, int nodeIndex, const glm::mat4& parent,
                              std::vector<char>& visited, std::vector<MeshPlacement>& out) {
    if (nodeIndex < 0 || nodeIndex >= static_cast<int>(model.nodes.size()) || visited[nodeIndex]) return;
    visited[nodeIndex] = 1;

    const tinygltf::Node& node = model.nodes[nodeIndex];
    const glm::mat4 world = parent * nodeLocalTransform(node);
    if (node.mesh >= 0 && node.mesh < static_cast<int>(model.meshes.size())) {
        for (const glm::mat4& transform : instanceTransforms(model, node, world)) {
            out.push_back({node.mesh, transform, node.name});
        }
    }
    for (int child : node.children) collectPlacements(model, child, world, visited, out);
}

// Builds one entry from a primitive with `world` baked into the vertices, so it reaches the
// importer looking like an OBJ shape. Normals go through the inverse transpose; a reflecting
// `world` reverses the winding, which is what glTF asks for and what keeps culling right on
// mirrored copies.
inline bool buildPrimitiveEntry(const tinygltf::Model& model, const tinygltf::Primitive& primitive,
                                const glm::mat4& world, MeshEntry& entry) {
    if (primitive.mode != TINYGLTF_MODE_TRIANGLES && primitive.mode != TINYGLTF_MODE_TRIANGLE_STRIP &&
        primitive.mode != TINYGLTF_MODE_TRIANGLE_FAN) {
        return false; // points and lines have nothing to draw down this pipeline
    }

    auto attribute = [&](const char* name) {
        auto it = primitive.attributes.find(name);
        return it == primitive.attributes.end() ? -1 : it->second;
    };

    std::vector<float> positions;
    if (!readAccessorFloats(model, attribute("POSITION"), 3, positions) || positions.size() < 3) return false;
    const size_t vertexCount = positions.size() / 3;

    std::vector<float> normals, texCoords, tangents;
    const bool hasNormals = readAccessorFloats(model, attribute("NORMAL"), 3, normals) && normals.size() >= vertexCount * 3;
    const bool hasTexCoords = readAccessorFloats(model, attribute("TEXCOORD_0"), 2, texCoords) && texCoords.size() >= vertexCount * 2;
    // TANGENT is a vec4 whose w is the bitangent's handedness; the pipeline only carries xyz.
    const bool hasTangents = readAccessorFloats(model, attribute("TANGENT"), 4, tangents) && tangents.size() >= vertexCount * 4;

    std::vector<uint32_t> indices;
    if (primitive.indices >= 0) {
        if (!readAccessorIndices(model, primitive.indices, indices)) return false;
    } else {
        indices.resize(vertexCount);
        for (size_t i = 0; i < vertexCount; i++) indices[i] = static_cast<uint32_t>(i);
    }
    if (primitive.mode == TINYGLTF_MODE_TRIANGLE_STRIP) indices = trianglesFromStrip(indices);
    else if (primitive.mode == TINYGLTF_MODE_TRIANGLE_FAN) indices = trianglesFromFan(indices);
    indices.resize(indices.size() - indices.size() % 3);
    if (indices.empty()) return false;
    for (uint32_t index : indices) {
        if (index >= vertexCount) return false; // never hand the GPU a corner past the vertex buffer
    }

    const glm::mat3 linear(world);
    const glm::mat3 normalMatrix = glm::transpose(glm::inverse(linear));

    entry.vertices.assign(vertexCount, Vertex{});
    for (size_t i = 0; i < vertexCount; i++) {
        Vertex& vertex = entry.vertices[i];
        vertex.position = glm::vec3(world * glm::vec4(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2], 1.0f));

        vertex.normal = glm::vec3(0.0f, 1.0f, 0.0f);
        if (hasNormals) {
            glm::vec3 normal = normalMatrix * glm::vec3(normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2]);
            if (glm::length(normal) > 0.0001f) vertex.normal = glm::normalize(normal);
        }

        // glTF texture coordinates already run top-left down, like Vulkan's — no flip here, unlike
        // the OBJ path.
        vertex.texCoord = hasTexCoords ? glm::vec2(texCoords[i * 2], texCoords[i * 2 + 1]) : glm::vec2(0.0f);

        if (hasTangents) {
            glm::vec3 tangent = linear * glm::vec3(tangents[i * 4], tangents[i * 4 + 1], tangents[i * 4 + 2]);
            vertex.tangent = glm::length(tangent) > 0.0001f ? glm::normalize(tangent) : glm::vec3(1.0f, 0.0f, 0.0f);
        }
    }

    entry.indices = std::move(indices);
    if (glm::determinant(linear) < 0.0f) {
        for (size_t i = 0; i + 2 < entry.indices.size(); i += 3) std::swap(entry.indices[i + 1], entry.indices[i + 2]);
    }
    if (!hasTangents) generateTangents(entry.vertices, entry.indices);

    entry.materialId = primitive.material;
    if (primitive.material >= 0 && primitive.material < static_cast<int>(model.materials.size()) &&
        !model.materials[primitive.material].name.empty()) {
        entry.materialName = model.materials[primitive.material].name;
    }
    // The node's own world position — the pivot whoever authored the file placed. Keeping it is
    // the point of reading a format that has one: OBJ bakes its pivots into the coordinates and
    // leaves the bbox centre as the only thing recoverable, and a door imported that way hinges
    // about its middle. The cost is that a file parenting many primitives under one node gives
    // them all the same pivot, which is off-centre bounds rather than wrong ones.
    entry.origin = glm::vec3(world[3]);
    return true;
}

} // namespace gltf

inline MeshData loadMeshFromFileGLTF(const std::string& meshPath) {
    tinygltf::TinyGLTF loader;
    tinygltf::Model model;
    std::string warn, err;

    loader.SetImageLoader(gltf::skipImageLoad, nullptr);
    const bool loaded = hasExtension(meshPath, ".glb") ? loader.LoadBinaryFromFile(&model, &err, &warn, meshPath)
                                                       : loader.LoadASCIIFromFile(&model, &err, &warn, meshPath);
    if (!warn.empty()) std::cerr << "[gltf] " << meshPath << ": " << warn << std::endl;
    if (!loaded) {
        std::cerr << "[gltf] failed to load " << meshPath << ": " << err << std::endl;
        return {};
    }

    // The default scene is the authored one. A file with no scenes at all still has nodes, so fall
    // back to every node nobody claims as a child — walking all of them would re-enter subtrees
    // with the wrong parent transform.
    std::vector<int> roots;
    if (!model.scenes.empty()) {
        const int sceneIndex = (model.defaultScene >= 0 && model.defaultScene < static_cast<int>(model.scenes.size())) ? model.defaultScene : 0;
        roots = model.scenes[sceneIndex].nodes;
    } else {
        std::vector<char> isChild(model.nodes.size(), 0);
        for (const tinygltf::Node& node : model.nodes) {
            for (int child : node.children) {
                if (child >= 0 && child < static_cast<int>(isChild.size())) isChild[child] = 1;
            }
        }
        for (size_t i = 0; i < model.nodes.size(); i++) {
            if (!isChild[i]) roots.push_back(static_cast<int>(i));
        }
    }

    std::vector<gltf::MeshPlacement> placements;
    std::vector<char> visited(model.nodes.size(), 0);
    for (int root : roots) gltf::collectPlacements(model, root, glm::mat4(1.0f), visited, placements);

    MeshData meshData;
    // Entries that carry real geometry, per (mesh, primitive). A later placement of the same
    // primitive attaches to the first of these it can reach by a rotation alone — more than one is
    // possible because a mirrored copy gets baked and then serves as the anchor for its own copies.
    std::map<std::pair<int, int>, std::vector<size_t>> placedEntries;
    std::vector<glm::mat4> entryPlacement; // parallel to meshData.entries
    size_t statedInstances = 0;

    for (size_t placementIdx = 0; placementIdx < placements.size(); placementIdx++) {
        const gltf::MeshPlacement& placement = placements[placementIdx];
        const tinygltf::Mesh& mesh = model.meshes[placement.meshIndex];

        // One placement is one object. Its primitives are separate entries only because glTF
        // splits a mesh by material the same way the OBJ reader does, so they carry the placement's
        // index as their shared object and import hangs them off one node.
        const std::string objectName = !placement.nodeName.empty() ? placement.nodeName
                                     : (!mesh.name.empty() ? mesh.name : "mesh_" + std::to_string(placement.meshIndex));

        for (size_t p = 0; p < mesh.primitives.size(); p++) {
            const tinygltf::Primitive& primitive = mesh.primitives[p];

            std::string name = objectName;
            if (mesh.primitives.size() > 1) name += "_" + std::to_string(p);

            // Instancing the file states outright: reuse the entry that already holds this
            // primitive's geometry and record only where this copy sits.
            auto& candidates = placedEntries[{placement.meshIndex, static_cast<int>(p)}];
            bool stated = false;
            for (size_t candidate : candidates) {
                const glm::mat4 relative = placement.world * glm::inverse(entryPlacement[candidate]);
                if (!gltf::isRigidNoFlip(relative)) continue; // a mirror or a resize; bake it instead

                MeshEntry entry;
                entry.instanceOf = static_cast<int>(candidate);
                entry.instanceTransform = relative;
                // Carrying the source's pivot through `relative` lands on this node's own world
                // position, which is the pivot the file gave this copy.
                entry.origin = glm::vec3(relative * glm::vec4(meshData.entries[candidate].origin, 1.0f));
                entry.materialId = meshData.entries[candidate].materialId;
                entry.materialName = meshData.entries[candidate].materialName;
                entry.shapeName = std::move(name);
                entry.sourceObject = static_cast<uint32_t>(placementIdx);
                entry.objectName = objectName;
                meshData.entries.push_back(std::move(entry));
                entryPlacement.push_back(placement.world);
                statedInstances++;
                stated = true;
                break;
            }
            if (stated) continue;

            MeshEntry entry;
            if (!gltf::buildPrimitiveEntry(model, primitive, placement.world, entry)) continue;
            entry.shapeName = std::move(name);
            entry.sourceObject = static_cast<uint32_t>(placementIdx);
            entry.objectName = objectName;
            candidates.push_back(meshData.entries.size());
            meshData.entries.push_back(std::move(entry));
            entryPlacement.push_back(placement.world);
        }
    }

    std::cout << "[gltf] " << meshPath << ": " << meshData.entries.size() << " entries ("
              << statedInstances << " instanced by the file)" << std::endl;
    return meshData;
}

// Picks the parser by extension. Anything unrecognised goes to the OBJ reader, which is what every
// caller reached before glTF existed.
inline MeshData loadMeshFromFile(const std::string& meshPath) {
    if (hasExtension(meshPath, ".gltf") || hasExtension(meshPath, ".glb")) return loadMeshFromFileGLTF(meshPath);
    return loadMeshFromFileOBJ(meshPath);
}

inline void freeMesh(Mesh& mesh) {
    mesh.freed = true;
    // don't free texture/sampler as they might be shared
}

} // namespace resource
