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

struct MeshData {
    std::vector<std::vector<Vertex>> subMeshes;
};

class ResourceManager {
  public:
    ResourceManager(Device& device, vk::raii::CommandPool& commandPool) : commandPool(commandPool), device(device) {};

    void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, vk::SampleCountFlagBits numSamples, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage,
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

        image = vk::raii::Image(device.getDevice(), imageInfo);
        vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo{.allocationSize = memRequirements.size, .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties, device)};

        imageMemory = vk::raii::DeviceMemory(device.getDevice(), allocInfo);
        image.bindMemory(imageMemory, 0);
    }

    [[nodiscard]] vk::raii::ImageView createImageView(vk::raii::Image& image, vk::Format format, vk::ImageAspectFlags aspectFlags, uint32_t mipLevels = 1) const {
        vk::ImageViewCreateInfo viewInfo{.image = image,
                                         .viewType = vk::ImageViewType::e2D,
                                         .format = format,
                                         .subresourceRange = {.aspectMask = aspectFlags, .baseMipLevel = 0, .levelCount = mipLevels, .baseArrayLayer = 0, .layerCount = 1}};
        return vk::raii::ImageView(device.getDevice(), viewInfo);
    }

    std::tuple<vk::raii::Image, vk::raii::DeviceMemory, vk::raii::ImageView> createTexture(const void* data, uint32_t width, uint32_t height, vk::Format format,
                                                                                           vk::ImageType imageType = vk::ImageType::e2D,
                                                                                           vk::ImageViewType viewType = vk::ImageViewType::e2D) {
        bool isCubemap = (viewType == vk::ImageViewType::eCube);
        // Create image
        vk::ImageCreateInfo imageInfo{.flags = viewType == vk::ImageViewType::eCube ? vk::ImageCreateFlagBits::eCubeCompatible : vk::ImageCreateFlags{},
                                      .imageType = imageType,
                                      .format = format,
                                      .extent = {width, height, 1},
                                      .mipLevels = 1,
                                      .arrayLayers = viewType == vk::ImageViewType::eCube ? static_cast<uint32_t>(6) : static_cast<uint32_t>(1),
                                      .samples = vk::SampleCountFlagBits::e1,
                                      .tiling = vk::ImageTiling::eOptimal,
                                      .usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
                                      .sharingMode = vk::SharingMode::eExclusive,
                                      .initialLayout = vk::ImageLayout::eUndefined};

        vk::raii::Image image(device.getDevice(), imageInfo);

        // Allocate memory
        vk::MemoryRequirements memRequirements = image.getMemoryRequirements();

        vk::MemoryAllocateInfo allocInfo{.allocationSize = memRequirements.size,
                                         .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal, device)};

        vk::raii::DeviceMemory imageMemory(device.getDevice(), allocInfo);
        image.bindMemory(*imageMemory, 0);
        // upload the texture data
        uploadTextureData(image, data, width, height, format, isCubemap);
        // Create image view
        vk::ImageViewCreateInfo viewInfo{.image = *image,
                                         .viewType = viewType,
                                         .format = format,
                                         .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                                                              .baseMipLevel = 0,
                                                              .levelCount = 1,
                                                              .baseArrayLayer = 0,
                                                              .layerCount = viewType == vk::ImageViewType::eCube ? static_cast<uint32_t>(6) : static_cast<uint32_t>(1)}};

        vk::raii::ImageView imageView(device.getDevice(), viewInfo);
        return std::make_tuple(std::move(image), std::move(imageMemory), std::move(imageView));
    }

    void uploadTextureData(const vk::raii::Image& image, const void* data, uint32_t width, uint32_t height, vk::Format format, bool isCubemap = false) {
        uint32_t bytesPerPixel = getBytesPerPixel(format);
        vk::DeviceSize imageSize = static_cast<vk::DeviceSize>(width) * height * bytesPerPixel;
        vk::DeviceSize totalSize = isCubemap ? imageSize * 6 : imageSize;

        // Create staging buffer
        vk::BufferCreateInfo bufferInfo{.size = totalSize, .usage = vk::BufferUsageFlagBits::eTransferSrc, .sharingMode = vk::SharingMode::eExclusive};
        vk::raii::Buffer stagingBuffer(device.getDevice(), bufferInfo);

        vk::MemoryRequirements memRequirements = stagingBuffer.getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo{
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, device)};
        vk::raii::DeviceMemory stagingBufferMemory(device.getDevice(), allocInfo);
        stagingBuffer.bindMemory(*stagingBufferMemory, 0);

        // Copy data to staging buffer
        void* mappedData = stagingBufferMemory.mapMemory(0, totalSize);
        memcpy(mappedData, data, totalSize);
        stagingBufferMemory.unmapMemory();

        // Transition image layout for transfer
        transitionImageLayout(nullptr, image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, 1, isCubemap ? 6 : 1);

        // Copy buffer to image
        if (isCubemap) {
            copyBufferToImageCubemap(stagingBuffer, image, width, height, getBytesPerPixel(format));
        } else {
            copyBufferToImage(stagingBuffer, image, width, height);
        }
        // Transition image layout for shader access
        transitionImageLayout(nullptr, image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, 1, isCubemap ? 6 : 1);
    }

    
    void transitionImageLayout(vk::raii::CommandBuffer* commandBuffer, const vk::Image& image, const vk::ImageLayout oldLayout, const vk::ImageLayout newLayout,
                               uint32_t mipLevels = 1, uint32_t layerCount = 1) {
        if (commandBuffer == nullptr) {
            // Handle single-time command case
            auto singleTimeCmdBuffer = beginSingleTimeCommands();
            executeImageTransition(singleTimeCmdBuffer, image, oldLayout, newLayout, mipLevels, layerCount);
            endSingleTimeCommands(singleTimeCmdBuffer);
        } else {
            // Handle existing command buffer case
            executeImageTransition(*commandBuffer, image, oldLayout, newLayout, mipLevels, layerCount);
        }
    }

    void copyBufferToImage(const vk::raii::Buffer& srcBuffer, const vk::raii::Image& dstImage, uint32_t width, uint32_t height, vk::BufferImageCopy* customRegion = nullptr) {
        vk::raii::CommandBuffer commandBuffer = beginSingleTimeCommands();
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
        endSingleTimeCommands(commandBuffer);
    }

    void copyBufferToImageCubemap(const vk::raii::Buffer& buffer, const vk::raii::Image& image, uint32_t width, uint32_t height, uint32_t bytesPerPixel) {
        vk::raii::CommandBuffer commandBuffer = beginSingleTimeCommands();

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

        endSingleTimeCommands(commandBuffer);
    }

    // load from file (OBJ, GLTF, etc.)
    MeshData loadMeshFromFile(const std::string& meshPath) { return loadMeshFromFileImpl(meshPath); }

    // Clean up mesh resources
    void freeMesh(Mesh& mesh) {
        mesh.freed = true;
        // Don't free texture/sampler as they might be shared
    }

    std::tuple<vk::raii::Image, vk::raii::DeviceMemory, vk::raii::ImageView> loadTextureFromFile(const std::string& path, vk::Format format = vk::Format::eR8G8B8A8Srgb,
                                                                                                 vk::ImageType imageType = vk::ImageType::e2D,
                                                                                                 vk::ImageViewType viewType = vk::ImageViewType::e2D) {
        auto textureData = loadTextureFromFileImpl(path);
        auto [image, memory, imageView] = createTexture(textureData.data, textureData.width, textureData.height, format, imageType, viewType);
        return std::make_tuple(std::move(image), std::move(memory), std::move(imageView));
    }

    std::tuple<vk::raii::Image, vk::raii::DeviceMemory, vk::raii::ImageView> loadCubeMapFromFile(uint32_t width, uint32_t height, std::string posX, std::string negX,
                                                                                                 std::string posY, std::string negY, std::string posZ, std::string negZ) {

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
        auto [image, memory, imageView] = createTexture(allFaceData.data(), width, height, vk::Format::eR8G8B8A8Srgb, vk::ImageType::e2D, vk::ImageViewType::eCube);
        return std::make_tuple(std::move(image), std::move(memory), std::move(imageView));
    }

  private:
    Device& device;
    const vk::raii::CommandPool& commandPool;
    std::vector<Material> materials;
    std::vector<Shader> shaders;

    void executeImageTransition(vk::raii::CommandBuffer& cmd, const vk::Image& image, const vk::ImageLayout oldLayout, const vk::ImageLayout newLayout, uint32_t mipLevels,
                                uint32_t layerCount) {
        vk::ImageMemoryBarrier2 barrier{
            .oldLayout = oldLayout,
            .newLayout = newLayout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = mipLevels, .baseArrayLayer = 0, .layerCount = layerCount}};

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
            barrier.srcAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
            barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
            barrier.srcStageMask = vk::PipelineStageFlagBits2::eLateFragmentTests;
            barrier.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader;
            barrier.subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eDepth, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1};
        }

        else if (oldLayout == vk::ImageLayout::eShaderReadOnlyOptimal && newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
            barrier.srcAccessMask = vk::AccessFlagBits2::eShaderRead;
            barrier.dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
            barrier.srcStageMask = vk::PipelineStageFlagBits2::eFragmentShader;
            barrier.dstStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests;
            barrier.subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eDepth, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1};
        }

        else if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eColorAttachmentOptimal) {
            barrier.srcAccessMask = {};
            barrier.dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
            barrier.srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
            barrier.dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
            barrier.subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1};
        }

        else if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
            barrier.srcAccessMask = {};
            barrier.dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
            barrier.srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
            barrier.dstStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests;
        }

        else if (oldLayout == vk::ImageLayout::eColorAttachmentOptimal && newLayout == vk::ImageLayout::ePresentSrcKHR) {
            barrier.srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
            barrier.dstAccessMask = {};
            barrier.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
            barrier.dstStageMask = vk::PipelineStageFlagBits2::eBottomOfPipe;
            barrier.subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1};
        }

        else {
            throw std::invalid_argument("unsupported layout transition!");
        }
        vk::DependencyInfo dependency_info = {.dependencyFlags = {}, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier};
        cmd.pipelineBarrier2(dependency_info);
    }

    struct TextureData {
        unsigned char* data;
        int width, height;
        ~TextureData() {
            if (data)
                stbi_image_free(data);
        }
    };

    TextureData loadTextureFromFileImpl(const std::string& path) {
        int width, height, channels;
        unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, STBI_rgb_alpha);
        if (!data) {
            throw std::runtime_error("Failed to load texture: " + path);
        }
        return {data, width, height};
    }

    MeshData loadMeshFromFileImpl(const std::string& meshPath) {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, meshPath.c_str())) {
            throw std::runtime_error(warn + err);
        }
        MeshData meshData = {};

        meshData.subMeshes.reserve(shapes.size());
        uint32_t subMeshIndex = 0;
        for (const auto& shape : shapes) {

            for (const auto& index : shape.mesh.indices) {
                Vertex vertex{};

                // Load position (with bounds check)
                if (index.vertex_index >= 0 && index.vertex_index < attrib.vertices.size() / 3) {
                    vertex.position = {attrib.vertices[3 * index.vertex_index + 0], attrib.vertices[3 * index.vertex_index + 1], attrib.vertices[3 * index.vertex_index + 2]};
                }

                // Load normal
                if (index.normal_index >= 0 && index.normal_index < attrib.normals.size() / 3) {
                    vertex.normal = {attrib.normals[3 * index.normal_index + 0], attrib.normals[3 * index.normal_index + 1], attrib.normals[3 * index.normal_index + 2]};
                } else {
                    vertex.normal = {0.0f, 1.0f, 0.0f}; // Default up normal
                }

                // Load texture coordinates
                if (index.texcoord_index >= 0 && index.texcoord_index < attrib.texcoords.size() / 2) {
                    vertex.texCoord = {attrib.texcoords[2 * index.texcoord_index + 0], 1.0f - attrib.texcoords[2 * index.texcoord_index + 1]};
                } else {
                    vertex.texCoord = {0.0f, 0.0f};
                }
                meshData.subMeshes[subMeshIndex].push_back(vertex);
            }
            subMeshIndex++;
        }
        return meshData;
    }

    vk::raii::CommandBuffer beginSingleTimeCommands() {
        vk::CommandBufferAllocateInfo allocInfo{};
        allocInfo.commandPool = *commandPool;
        allocInfo.level = vk::CommandBufferLevel::ePrimary;
        allocInfo.commandBufferCount = 1;

        vk::raii::CommandBuffers commandBuffers(device.getDevice(), allocInfo);
        vk::raii::CommandBuffer commandBuffer = std::move(commandBuffers[0]);

        vk::CommandBufferBeginInfo beginInfo{};
        beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
        commandBuffer.begin(beginInfo);

        return commandBuffer;
    }

    void endSingleTimeCommands(vk::raii::CommandBuffer& commandBuffer) {
        commandBuffer.end();

        vk::SubmitInfo submitInfo{};
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &*commandBuffer;

        device.getGraphicsQueue().submit(submitInfo, nullptr);
        device.getGraphicsQueue().waitIdle();
    }
};