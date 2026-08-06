#pragma once
#ifndef VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1 // for raii
#endif
#ifndef VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_NO_CONSTRUCTORS 1 // for structs constructors
#endif

#include <stb_image.h>
#include <tiny_obj_loader.h>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#define GLM_FORCE_RADIANS
#define GLM_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#include "devices.hpp"
#include "structs.hpp"
#include "utils.hpp"

struct MeshEntry {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    std::vector<uint32_t> LODs;
    int materialId = -1;
    std::string materialName = "default_material";
    std::string shapeName;
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

    // Depth-read-only -> generic shader-read-only (and back). Lets a compute pass sample a depth
    // image (the shadow atlas, VolumetricsPass pass C) with a descriptor recorded as
    // eShaderReadOnlyOptimal — eDepthReadOnlyOptimal mismatches that descriptor and is undefined in
    // compute on some drivers. Read->read: contents preserved, only the layout/visibility changes.
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
        barrier.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader;
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

struct TextureData {
    unsigned char* data;
    int width, height;
    ~TextureData() {
        if (data)
            stbi_image_free(data);
    }
};

inline TextureData loadTextureFromFileImpl(const std::string& path) {
    int width, height, channels;
    unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, STBI_rgb_alpha);
    if (!data) {
        throw std::runtime_error("Failed to load texture: " + path);
    }
    return {data, width, height};
}

inline std::tuple<vk::raii::Image, vk::raii::DeviceMemory, vk::raii::ImageView, uint32_t, uint32_t> loadTextureFromFile(const Context& ctx, const std::string& path, vk::Format format = vk::Format::eR8G8B8A8Srgb,
                                                                                                                        vk::ImageType imageType = vk::ImageType::e2D,
                                                                                                                        vk::ImageViewType viewType = vk::ImageViewType::e2D) {
    auto textureData = loadTextureFromFileImpl(path);
    auto [image, memory, imageView] = createTexture(ctx, textureData.data, textureData.width, textureData.height, format, imageType, viewType);
    return std::make_tuple(std::move(image), std::move(memory), std::move(imageView),
                           static_cast<uint32_t>(textureData.width), static_cast<uint32_t>(textureData.height));
}

inline std::tuple<vk::raii::Image, vk::raii::DeviceMemory, vk::raii::ImageView> loadCubeMapFromFile(const Context& ctx, std::string posX, std::string negX, std::string posY, std::string negY,
                                                                                                    std::string posZ, std::string negZ, uint32_t width, uint32_t height) {
    // TODO need to get imdgwidth and height from stbi load first so it doesnt have to be entered manually
    std::vector<std::string> faceFiles = {posX, negX, posY, negY, posZ, negZ};
    // Load all 6 face data into a single buffer
    size_t faceSize = width * height * 4;
    size_t totalSize = faceSize * 6;
    std::vector<unsigned char> allFaceData(totalSize);

    for (int face = 0; face < 6; face++) {
        int imgWidth, imgHeight, channels;
        unsigned char* imageData = stbi_load(faceFiles[face].c_str(), &imgWidth, &imgHeight, &channels, STBI_rgb_alpha);
        if (!imageData) {
            throw std::runtime_error("Failed to load face: " + faceFiles[face]);
        }

        // Copy face data to the combined buffer
        memcpy(allFaceData.data() + face * faceSize, imageData, faceSize);
        stbi_image_free(imageData);
    }
    auto [image, memory, imageView] = createTexture(ctx, allFaceData.data(), width, height, vk::Format::eR8G8B8A8Srgb, vk::ImageType::e2D, vk::ImageViewType::eCube);
#if DEBUG == 1
    uint32_t expectedMipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1;
    std::cout << "Cubemap should have " << expectedMipLevels << " mip levels" << std::endl;
#endif
    return std::make_tuple(std::move(image), std::move(memory), std::move(imageView));
}

// only handles obj for now (TODO expand this)
inline MeshData loadMeshFromFile(const std::string& meshPath) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    // Extract directory path for loading .mtl file
    std::string mtlBaseDir = meshPath.substr(0, meshPath.find_last_of("/\\") + 1);

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, meshPath.c_str(), mtlBaseDir.c_str())) {
        throw std::runtime_error(warn + err);
    }
    MeshData meshData = {};

    // Process each shape and split by material
    for (const auto& shape : shapes) {

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
            for (auto& vertex : entry.vertices) {
                vertex.tangent = glm::vec3(0.0f);
            }

            for (size_t i = 0; i < entry.indices.size(); i += 3) {
                uint32_t idx0 = entry.indices[i];
                uint32_t idx1 = entry.indices[i + 1];
                uint32_t idx2 = entry.indices[i + 2];

                Vertex& v0 = entry.vertices[idx0];
                Vertex& v1 = entry.vertices[idx1];
                Vertex& v2 = entry.vertices[idx2];

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

            for (auto& vertex : entry.vertices) {
                if (glm::length(vertex.tangent) > 0.0001f) {
                    vertex.tangent = normalize(vertex.tangent);
                } else {
                    vertex.tangent = glm::vec3(1.0f, 0.0f, 0.0f);
                }
            }

            meshData.entries.push_back(std::move(entry));
        }
    }

    return meshData;
}

inline void freeMesh(Mesh& mesh) {
    mesh.freed = true;
    // don't free texture/sampler as they might be shared
}

} // namespace resource
