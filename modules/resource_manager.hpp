#pragma once
#ifndef VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1 // for raii
#endif
#ifndef VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_NO_CONSTRUCTORS 1 // for structs constructors
#endif
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <devices.hpp>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <structs.hpp>
#include <unordered_map>
#include <utils.hpp>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#define GLFW_INCLUDE_VULKAN // REQUIRED only for GLFW CreateWindowSurface.
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

constexpr uint32_t MAX_BINDLESS_TEXTURES = 256;
constexpr uint32_t MAX_BINDLESS_SAMPLERS = 16;
static constexpr vk::DeviceSize VERTEX_BUFFER_SIZE = 256 * 1024 * 1024; // max 256mb of vertex data
static constexpr uint32_t MAX_VERTEX_ALLOCATIONS = 2048;
constexpr uint32_t MAX_BINDLESS_MODEL_MATRICES = 2048;
constexpr uint32_t MAX_LIGHTS = 1024;
constexpr uint32_t MAX_CUBEMAPS = 64;

class ResourceManager {

  public:
    ResourceManager(Devices* devices, vk::raii::CommandPool* commandPool) : devices(devices), commandPool(commandPool) {
        descriptorSetLayout = createDescriptorSetLayout();
        descriptorPool = createDescriptorPool();
        bindlessDescriptorSet = allocateDescriptorSet();

        depthDescriptorSetLayout = createDepthDescriptorSetLayout();
        depthDescriptorPool = createDepthDescriptorPool();
        depthDescriptorSet = allocateDepthDescriptorSet();

        // Reserve space for maximum resources
        textureResources.reserve(MAX_BINDLESS_TEXTURES);
        samplerResources.reserve(MAX_BINDLESS_SAMPLERS);
        vertexAllocations.resize(MAX_VERTEX_ALLOCATIONS);
        modelMatrixSlots.resize(MAX_BINDLESS_MODEL_MATRICES);
        lightSlots.resize(MAX_LIGHTS);
        cubemapResources.reserve(MAX_CUBEMAPS);
    }

    void initializeDefaults() {

        initializeVertexBuffer();
        initializeModelMatrixBuffer();
        initializeLightBuffer();

        // Create default sampler
        vk::SamplerCreateInfo samplerInfo{};
        samplerInfo.magFilter = vk::Filter::eLinear;
        samplerInfo.minFilter = vk::Filter::eLinear;
        samplerInfo.addressModeU = vk::SamplerAddressMode::eClampToEdge;
        samplerInfo.addressModeV = vk::SamplerAddressMode::eClampToEdge;
        samplerInfo.addressModeW = vk::SamplerAddressMode::eClampToEdge;
        samplerInfo.anisotropyEnable = VK_TRUE;
        samplerInfo.maxAnisotropy = 16.0f;
        samplerInfo.borderColor = vk::BorderColor::eIntOpaqueBlack;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = vk::CompareOp::eAlways;
        samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;

        // Create depth default sampler
        vk::SamplerCreateInfo depthSamplerInfo{};
        samplerInfo.magFilter = vk::Filter::eNearest;
        samplerInfo.minFilter = vk::Filter::eNearest;
        samplerInfo.addressModeU = vk::SamplerAddressMode::eClampToEdge;
        samplerInfo.addressModeV = vk::SamplerAddressMode::eClampToEdge;
        samplerInfo.addressModeW = vk::SamplerAddressMode::eClampToEdge;
        samplerInfo.anisotropyEnable = VK_FALSE;
        samplerInfo.borderColor = vk::BorderColor::eFloatOpaqueWhite;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.mipmapMode = vk::SamplerMipmapMode::eNearest;

        vk::raii::Sampler defaultSampler(devices->getLogicalDevice(), samplerInfo);
        vk::raii::Sampler depthSampler(devices->getLogicalDevice(), depthSamplerInfo);
        updateSamplerDescriptor(*depthDescriptorSet, 1, *depthSampler);
        defaultSamplerIndex = allocateSamplerImpl(std::move(defaultSampler));

        // Create default textures (1x1 pixel images)
        whiteTextureIndex = createDefaultTexture({255, 255, 255, 255});
        blackTextureIndex = createDefaultTexture({0, 0, 0, 255});
        defaultNormalIndex = createDefaultTexture({128, 128, 255, 255});
    }

    const vk::raii::DescriptorSetLayout& getDescriptorSetLayout() const { return *descriptorSetLayout; }
    const vk::raii::DescriptorSet& getDescriptorSet() const { return *bindlessDescriptorSet; }
    vk::raii::DescriptorPool& getDescriptorPool() { return *descriptorPool; }

    const vk::raii::DescriptorSetLayout& getDepthDescriptorSetLayout() const { return *depthDescriptorSetLayout; }
    vk::raii::DescriptorSet& getDepthDescriptorSet() { return *depthDescriptorSet; }
    vk::raii::DescriptorPool& getDepthDescriptorPool() { return *depthDescriptorPool; }

    uint32_t getWhiteTextureIndex() const { return whiteTextureIndex; }
    uint32_t getBlackTextureIndex() const { return blackTextureIndex; }
    uint32_t getDefaultNormalIndex() const { return defaultNormalIndex; }
    uint32_t getDefaultSamplerIndex() const { return defaultSamplerIndex; }
    uint32_t getDefaultVertexBufferIndex() const { return defaultVertexBufferIndex; }
    uint32_t getLightCount() const { return lightBuffer.lightCount; }

    ///////////////////////////////////////////////////////////////////////////////////////
    // SAMPLERS
    ///////////////////////////////////////////////////////////////////////////////////////
    uint32_t allocateSampler(vk::raii::Sampler&& sampler) { return allocateSamplerImpl(std::move(sampler)); }

    void freeSampler(uint32_t index) {
        if (index >= samplerResources.size() || samplerResources[index].isEmpty()) {
            return; // Already freed or invalid index
        }

        // Don't allow freeing default sampler
        if (index == defaultSamplerIndex) {
            throw std::runtime_error("Cannot free default sampler");
        }

        samplerResources[index].reset();
        freeSamplerSlots.push(index);
        clearSamplerDescriptor(index);
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    // TEXTURES
    ///////////////////////////////////////////////////////////////////////////////////////
    uint32_t allocateTexture(const void* data, uint32_t width, uint32_t height, vk::Format format = vk::Format::eR8G8B8A8Srgb, vk::ImageType imageType = vk::ImageType::e2D,
                             vk::ImageViewType viewType = vk::ImageViewType::e2D) {
        if (!data) {
            throw std::invalid_argument("Texture data cannot be null");
        }
        if (width == 0 || height == 0) {
            throw std::invalid_argument("Texture dimensions must be greater than 0");
        }

        if (textureResources.size() >= MAX_BINDLESS_TEXTURES && freeTextureSlots.empty()) {
            throw std::runtime_error("Maximum bindless textures exceeded");
        }

        auto [image, memory, imageView] = createTexture(data, width, height, format, imageType, viewType);

        return allocateTextureImpl(std::move(image), std::move(memory), std::move(imageView));
    }

    void freeTexture(uint32_t index) {

        if (index >= textureResources.size() || textureResources[index].isEmpty()) {
            return; // Already freed or invalid index
        }

        // Don't allow freeing default textures
        if (index == whiteTextureIndex || index == blackTextureIndex || index == defaultNormalIndex) {
            throw std::runtime_error("Cannot free default textures");
        }

        textureResources[index].reset();
        freeTextureSlots.push(index);
        clearTextureDescriptor(index);
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    // VERTEX BUFFERS
    ///////////////////////////////////////////////////////////////////////////////////////
    struct VertexBufferInfo {
        uint32_t allocationIndex;
        vk::DeviceSize offset; // For shader use
        uint32_t vertexCount;
    };

    VertexBufferInfo allocateVertexBuffer(const void* data, vk::DeviceSize dataSize, uint32_t vertexCount, uint32_t vertexStride) {
        if (!data || dataSize == 0 || vertexCount == 0 || vertexStride == 0) {
            throw std::invalid_argument("Invalid vertex buffer parameters");
        }

        uint32_t index = allocateVertexBufferImpl(data, dataSize, vertexStride, vertexCount);

        return {.allocationIndex = index, .offset = vertexAllocations[index].offset, .vertexCount = vertexCount};
    }

    void updateVertexBuffer(uint32_t allocationIndex, const void* data, vk::DeviceSize dataSize, vk::DeviceSize offset = 0) {

        if (allocationIndex >= MAX_VERTEX_ALLOCATIONS || !vertexAllocations[allocationIndex].inUse) {
            throw std::invalid_argument("Invalid vertex allocation index");
        }

        auto& allocation = vertexAllocations[allocationIndex];
        if (dataSize + offset > allocation.size) {
            throw std::invalid_argument("Update would exceed allocation size");
        }

        // Direct memory copy into the big buffer
        uint8_t* bufferData = static_cast<uint8_t*>(vertexBuffer.mappedData);
        memcpy(bufferData + allocation.offset + offset, data, dataSize);
    }

    void freeVertexBuffer(uint32_t allocationIndex) {

        if (allocationIndex >= MAX_VERTEX_ALLOCATIONS || !vertexAllocations[allocationIndex].inUse) {
            return;
        }

        // Mark space as available (simple linear allocator for now)
        // TODO: Implement proper free space management
        vertexAllocations[allocationIndex].reset();
        freeVertexSlots.push(allocationIndex);
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    // MODEL MATRICES
    ///////////////////////////////////////////////////////////////////////////////////////
    uint32_t allocateModelMatrixBuffer(glm::vec3 position, glm::quat rotation, glm::vec3 scale) {
        glm::mat4 matrix = createModelMatrix(position, rotation, scale);
        return allocateModelMatrixBuffer(matrix);
    }

    uint32_t allocateModelMatrixBuffer(const glm::mat4& matrix) {

        uint32_t index;
        if (!freeModelMatrixSlots.empty()) {
            index = freeModelMatrixSlots.front();
            freeModelMatrixSlots.pop();
        } else {
            // Find first unused slot
            index = 0;
            while (index < MAX_BINDLESS_MODEL_MATRICES && modelMatrixSlots[index].inUse) {
                index++;
            }

            if (index >= MAX_BINDLESS_MODEL_MATRICES) {
                throw std::runtime_error("Maximum bindless model matrices exceeded");
            }
        }
        // Update CPU-side tracking
        modelMatrixBuffer.modelMatrixCount++;
        modelMatrixSlots[index].matrix = matrix;
        modelMatrixSlots[index].inUse = true;

        // Update GPU buffer directly (fast - no map/unmap)
        glm::mat4* gpuMatrices = static_cast<glm::mat4*>(modelMatrixBuffer.mappedData);
        gpuMatrices[index] = matrix;

        printf("allocating matrix: \n");
        for (int i = 0; i < 4; i++) {
            printf("%.2f %.2f %.2f %.2f\n", matrix[0][i], matrix[1][i], matrix[2][i], matrix[3][i]);
        }
        std::cout << "at index: " << index << std::endl;

        return index;
    }

    void updateModelMatrix(uint32_t index, glm::vec3 position, glm::quat rotation, glm::vec3 scale) {
        glm::mat4 matrix = createModelMatrix(position, rotation, scale);
        updateModelMatrix(index, matrix);
    }

    void updateModelMatrix(uint32_t index, const glm::mat4& matrix) {

        if (index >= MAX_BINDLESS_MODEL_MATRICES || !modelMatrixSlots[index].inUse) {
            throw std::invalid_argument("Invalid model matrix index");
        }
        // Update CPU copy
        modelMatrixSlots[index].matrix = matrix;

        // Update GPU buffer directly
        glm::mat4* gpuMatrices = static_cast<glm::mat4*>(modelMatrixBuffer.mappedData);
        gpuMatrices[index] = matrix;
    }

    void freeModelMatrix(uint32_t index) {

        if (index >= MAX_BINDLESS_MODEL_MATRICES || !modelMatrixSlots[index].inUse) {
            return;
        }

        modelMatrixBuffer.modelMatrixCount--;
        modelMatrixSlots[index].reset();
        freeModelMatrixSlots.push(index);

        // Reset to identity matrix in GPU buffer
        glm::mat4* gpuMatrices = static_cast<glm::mat4*>(modelMatrixBuffer.mappedData);
        gpuMatrices[index] = glm::mat4(1.0f);
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    // LIGHTS
    ///////////////////////////////////////////////////////////////////////////////////////
    uint32_t allocateLightBuffer(const Light light) {
        uint32_t index;
        if (!freeLightSlots.empty()) {
            index = freeLightSlots.front();
            freeLightSlots.pop();
        } else {
            // Find first unused slot
            index = 0;
            while (index < MAX_LIGHTS && lightSlots[index].inUse) {
                index++;
            }

            if (index >= MAX_LIGHTS) {
                throw std::runtime_error("Maximum point lights exceeded");
            }
        }

        // Update CPU-side tracking
        lightBuffer.lightCount++;
        lightSlots[index].light = light;
        lightSlots[index].inUse = true;

        // Update GPU buffer
        Light* bufferData = static_cast<Light*>(lightBuffer.mappedData);
        bufferData[index] = light;

        printf("allocating light: \n");
        std::cout << "position: " << light.position.x << " " << light.position.y << " " << light.position.z << std::endl;
        std::cout << "range: " << light.range << std::endl;
        std::cout << "intensity: " << light.intensity << std::endl;
        std::cout << "color: " << light.color.x << " " << light.color.y << " " << light.color.z << std::endl;
        std::cout << "at index: " << index << std::endl;

        return index;
    }

    void updateLight(uint32_t index, const Light& light) {
        if (index >= MAX_LIGHTS || !lightSlots[index].inUse) {
            throw std::invalid_argument("Invalid light index");
        }

        // Update CPU copy
        lightSlots[index].light = light;

        // Update GPU buffer directly
        Light* bufferData = static_cast<Light*>(lightBuffer.mappedData);
        bufferData[index] = light;
    }

    void freeLight(uint32_t index) {
        if (index >= MAX_LIGHTS || !lightSlots[index].inUse) {
            return;
        }

        lightBuffer.lightCount--;
        lightSlots[index].reset();
        freeLightSlots.push(index);

        // Reset to identity matrix in GPU buffer
        Light* bufferData = static_cast<Light*>(lightBuffer.mappedData);
        bufferData[index].reset();
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    // CUBEMAPS
    ///////////////////////////////////////////////////////////////////////////////////////
    uint32_t allocateCubemap(const void* data, uint32_t width, uint32_t height, vk::Format format = vk::Format::eR8G8B8A8Srgb, vk::ImageType imageType = vk::ImageType::e2D,
                             vk::ImageViewType viewType = vk::ImageViewType::eCube) {
        if (!data) {
            throw std::invalid_argument("Texture data cannot be null");
        }
        if (width == 0 || height == 0) {
            throw std::invalid_argument("Texture dimensions must be greater than 0");
        }

        if (textureResources.size() >= MAX_CUBEMAPS && freeCubemapSlots.empty()) {
            throw std::runtime_error("Maximum bindless textures exceeded");
        }

        auto [image, memory, imageView] = createTexture(data, width, height, format, imageType, viewType);

        return allocateCubeMapImpl(std::move(image), std::move(memory), std::move(imageView));
    }

    void freeCubemap(uint32_t index) {

        if (index >= cubemapResources.size() || cubemapResources[index].isEmpty()) {
            return; // Already freed or invalid index
        }

        cubemapResources[index].reset();
        freeCubemapSlots.push(index);
        clearCubemapDescriptor(index);
    }
    ///////////////////////////////////////////////////////////////////////////////////////

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

        image = vk::raii::Image(devices->getLogicalDevice(), imageInfo);
        vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo{.allocationSize = memRequirements.size, .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties, &*devices)};

        imageMemory = vk::raii::DeviceMemory(devices->getLogicalDevice(), allocInfo);
        image.bindMemory(imageMemory, 0);
    }

    [[nodiscard]] vk::raii::ImageView createImageView(vk::raii::Image& image, vk::Format format, vk::ImageAspectFlags aspectFlags, uint32_t mipLevels) const {
        vk::ImageViewCreateInfo viewInfo{.image = image,
                                         .viewType = vk::ImageViewType::e2D,
                                         .format = format,
                                         .subresourceRange = {.aspectMask = aspectFlags, .baseMipLevel = 0, .levelCount = mipLevels, .baseArrayLayer = 0, .layerCount = 1}};
        return vk::raii::ImageView(devices->getLogicalDevice(), viewInfo);
    }

    void transitionImageLayout(const vk::raii::Image& image, const vk::ImageLayout oldLayout, const vk::ImageLayout newLayout, uint32_t mipLevels, uint32_t layerCount = 1) {
        vk::raii::CommandBuffer commandBuffer = beginSingleTimeCommands();

        vk::ImageMemoryBarrier barrier{
            .oldLayout = oldLayout,
            .newLayout = newLayout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = *image,
            .subresourceRange = {.aspectMask = (oldLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal || newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal ||
                                                oldLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
                                                   ? vk::ImageAspectFlagBits::eDepth
                                                   : vk::ImageAspectFlagBits::eColor,
                                 .baseMipLevel = 0,
                                 .levelCount = mipLevels,
                                 .baseArrayLayer = 0,
                                 .layerCount = layerCount}};

        vk::PipelineStageFlags sourceStage;
        vk::PipelineStageFlags destinationStage;

        if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
            barrier.srcAccessMask = {};
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
            sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
            destinationStage = vk::PipelineStageFlagBits::eTransfer;
        } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
            sourceStage = vk::PipelineStageFlagBits::eTransfer;
            destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
        } else if (oldLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
            barrier.srcAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
            sourceStage = vk::PipelineStageFlagBits::eLateFragmentTests;
            destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
        } else if (oldLayout == vk::ImageLayout::eShaderReadOnlyOptimal && newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
            barrier.srcAccessMask = vk::AccessFlagBits::eShaderRead;
            barrier.dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentRead | vk::AccessFlagBits::eDepthStencilAttachmentWrite;
            sourceStage = vk::PipelineStageFlagBits::eFragmentShader;
            destinationStage = vk::PipelineStageFlagBits::eEarlyFragmentTests;
        } else if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
            barrier.srcAccessMask = {};
            barrier.dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentRead | vk::AccessFlagBits::eDepthStencilAttachmentWrite;
            sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
            destinationStage = vk::PipelineStageFlagBits::eEarlyFragmentTests;
        } else {
            throw std::invalid_argument("unsupported layout transition!");
        }

        commandBuffer.pipelineBarrier(sourceStage, destinationStage, {}, {}, {}, barrier);
        endSingleTimeCommands(commandBuffer);
    }

    void updateImageDescriptorSet(vk::raii::DescriptorSet& descriptorSet, uint32_t index, vk::ImageView imageView) {
        vk::DescriptorImageInfo imageInfo{.imageView = imageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};

        vk::WriteDescriptorSet write{.sType = vk::StructureType::eWriteDescriptorSet,
                                     .dstSet = *descriptorSet,
                                     .dstBinding = 0,
                                     .dstArrayElement = index,
                                     .descriptorCount = 1,
                                     .descriptorType = vk::DescriptorType::eSampledImage,
                                     .pImageInfo = &imageInfo};

        devices->getLogicalDevice().updateDescriptorSets(write, {});
    }

    struct ResourceStats {
        uint32_t texturesUsed;
        uint32_t texturesTotal;
        uint32_t samplersUsed;
        uint32_t samplersTotal;
        uint32_t vertexBuffersUsed;
        uint32_t vertexBuffersTotal;
        uint32_t modelMatricesUsed;
        uint32_t modelMatricesTotal;
    };

    ResourceStats getResourceStats() const {

        uint32_t texturesUsed = textureResources.size() - freeTextureSlots.size() - 3;
        uint32_t samplersUsed = samplerResources.size() - freeSamplerSlots.size();
        uint32_t vertexBuffersUsed = MAX_VERTEX_ALLOCATIONS - freeVertexSlots.size();
        uint32_t modelMatricesUsed = MAX_BINDLESS_MODEL_MATRICES - freeModelMatrixSlots.size();

        return {texturesUsed,      MAX_BINDLESS_TEXTURES,  samplersUsed,      MAX_BINDLESS_SAMPLERS,
                vertexBuffersUsed, MAX_VERTEX_ALLOCATIONS, modelMatricesUsed, MAX_BINDLESS_MODEL_MATRICES};
    }

  private:
    struct TextureResource {
        std::optional<vk::raii::ImageView> imageView;
        std::optional<vk::raii::Image> image;
        std::optional<vk::raii::DeviceMemory> memory;

        void reset() {
            imageView.reset();
            image.reset();
            memory.reset();
        }

        bool isEmpty() const { return !imageView.has_value(); }
    };

    struct CubemapResource {
        std::optional<vk::raii::ImageView> imageView;
        std::optional<vk::raii::Image> image;
        std::optional<vk::raii::DeviceMemory> memory;

        void reset() {
            imageView.reset();
            image.reset();
            memory.reset();
        }

        bool isEmpty() const { return !imageView.has_value(); }
    };

    struct SamplerResource {
        std::optional<vk::raii::Sampler> sampler;

        void reset() { sampler.reset(); }

        bool isEmpty() const { return !sampler.has_value(); }
    };

    struct VertexBufferResource {
        std::optional<vk::raii::Buffer> buffer;
        std::optional<vk::raii::DeviceMemory> memory;
        void* mappedData = nullptr;
        vk::DeviceSize totalSize;
        vk::DeviceSize usedSize = 0;

        void reset() {
            if (mappedData) {
                memory->unmapMemory();
                mappedData = nullptr;
            }
            buffer.reset();
            memory.reset();
            usedSize = 0;
        }

        bool isEmpty() const { return !buffer.has_value(); }
    };

    struct VertexAllocation {
        vk::DeviceSize offset; // Offset in the vertex buffer
        vk::DeviceSize size;   // Size of this allocation
        uint32_t vertexCount;  // Number of vertices
        uint32_t vertexStride; // Size per vertex
        bool inUse = false;

        void reset() {
            offset = 0;
            size = 0;
            vertexCount = 0;
            vertexStride = 0;
            inUse = false;
        }
    };

    struct ModelMatrixBufferResource {
        std::optional<vk::raii::Buffer> buffer;
        std::optional<vk::raii::DeviceMemory> memory;
        void* mappedData = nullptr;
        vk::DeviceSize bufferSize;
        uint32_t modelMatrixCount = 0;
        void reset() {
            if (mappedData) {
                memory->unmapMemory();
                mappedData = nullptr;
            }
            buffer.reset();
            memory.reset();
        }

        bool isEmpty() const { return !buffer.has_value(); }
    };

    struct ModelMatrixSlot {
        glm::mat4 matrix;
        bool inUse = false;

        void reset() {
            matrix = glm::mat4(1.0f);
            inUse = false;
        }
    };

    struct LightBufferResource {
        std::optional<vk::raii::Buffer> buffer;
        std::optional<vk::raii::DeviceMemory> memory;
        void* mappedData = nullptr;
        vk::DeviceSize bufferSize;
        uint32_t lightCount = 0;
        void reset() {
            if (mappedData) {
                memory->unmapMemory();
                mappedData = nullptr;
            }
            buffer.reset();
            memory.reset();
        }

        bool isEmpty() const { return !buffer.has_value(); }
    };

    struct LightSlot {
        Light light;
        bool inUse;

        void reset() {
            light.reset();
            inUse = false;
        }
    };

    std::vector<TextureResource> textureResources;
    std::vector<SamplerResource> samplerResources;
    std::vector<CubemapResource> cubemapResources;
    std::queue<uint32_t> freeTextureSlots;
    std::queue<uint32_t> freeSamplerSlots;
    std::queue<uint32_t> freeCubemapSlots;

    VertexBufferResource vertexBuffer;
    std::vector<VertexAllocation> vertexAllocations;
    std::queue<uint32_t> freeVertexSlots;

    ModelMatrixBufferResource modelMatrixBuffer;
    std::vector<ModelMatrixSlot> modelMatrixSlots;
    std::queue<uint32_t> freeModelMatrixSlots;

    LightBufferResource lightBuffer;
    std::vector<LightSlot> lightSlots;
    std::queue<uint32_t> freeLightSlots;

    std::optional<vk::raii::DescriptorSet> bindlessDescriptorSet;
    std::optional<vk::raii::DescriptorSetLayout> descriptorSetLayout;
    std::optional<vk::raii::DescriptorPool> descriptorPool;

    std::optional<vk::raii::DescriptorSet> depthDescriptorSet;
    std::optional<vk::raii::DescriptorSetLayout> depthDescriptorSetLayout;
    std::optional<vk::raii::DescriptorPool> depthDescriptorPool;

    const vk::raii::CommandPool* commandPool;
    const Devices* devices;

    uint32_t defaultSamplerIndex;
    uint32_t whiteTextureIndex;
    uint32_t blackTextureIndex;
    uint32_t defaultNormalIndex;
    uint32_t defaultVertexBufferIndex;

    // TODO : Thread safety
    mutable std::mutex resourceMutex;

    vk::raii::DescriptorSet allocateDescriptorSet() {
        std::array<uint32_t, 5> variableCounts = {MAX_BINDLESS_TEXTURES, MAX_BINDLESS_SAMPLERS, 1, 1, 1};

        vk::DescriptorSetVariableDescriptorCountAllocateInfo variableCountInfo{.descriptorSetCount = 1, .pDescriptorCounts = variableCounts.data()};
        vk::DescriptorSetAllocateInfo allocInfo{.pNext = &variableCountInfo, .descriptorPool = *descriptorPool, .descriptorSetCount = 1, .pSetLayouts = &(**descriptorSetLayout)};

        vk::raii::DescriptorSets sets(devices->getLogicalDevice(), allocInfo);
        return std::move(sets[0]);
    }

    vk::raii::DescriptorSetLayout createDescriptorSetLayout() {
        std::array<vk::DescriptorSetLayoutBinding, 6> bindings = {{// textures
                                                                   {.binding = 0,
                                                                    .descriptorType = vk::DescriptorType::eSampledImage,
                                                                    .descriptorCount = MAX_BINDLESS_TEXTURES,
                                                                    .stageFlags = vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eVertex,
                                                                    .pImmutableSamplers = nullptr},
                                                                   // samplers
                                                                   {.binding = 1,
                                                                    .descriptorType = vk::DescriptorType::eSampler,
                                                                    .descriptorCount = MAX_BINDLESS_SAMPLERS,
                                                                    .stageFlags = vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eVertex,
                                                                    .pImmutableSamplers = nullptr},
                                                                   // vertices
                                                                   {.binding = 2,
                                                                    .descriptorType = vk::DescriptorType::eStorageBuffer,
                                                                    .descriptorCount = 1,
                                                                    .stageFlags = vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eVertex,
                                                                    .pImmutableSamplers = nullptr},
                                                                   // model matrices
                                                                   {.binding = 3,
                                                                    .descriptorType = vk::DescriptorType::eStorageBuffer,
                                                                    .descriptorCount = 1,
                                                                    .stageFlags = vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eVertex,
                                                                    .pImmutableSamplers = nullptr},
                                                                   {.binding = 4,
                                                                    .descriptorType = vk::DescriptorType::eStorageBuffer,
                                                                    .descriptorCount = 1,
                                                                    .stageFlags = vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eVertex,
                                                                    .pImmutableSamplers = nullptr},
                                                                   {.binding = 5,
                                                                    .descriptorType = vk::DescriptorType::eSampledImage,
                                                                    .descriptorCount = MAX_CUBEMAPS,
                                                                    .stageFlags = vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eVertex,
                                                                    .pImmutableSamplers = nullptr}}};

        std::array<vk::DescriptorBindingFlags, 6> bindingFlags = {{vk::DescriptorBindingFlagBits::ePartiallyBound, vk::DescriptorBindingFlagBits::ePartiallyBound,
                                                                   vk::DescriptorBindingFlagBits::ePartiallyBound, vk::DescriptorBindingFlagBits::ePartiallyBound,
                                                                   vk::DescriptorBindingFlagBits::ePartiallyBound, vk::DescriptorBindingFlagBits::ePartiallyBound}};

        vk::DescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{.bindingCount = static_cast<uint32_t>(bindingFlags.size()), .pBindingFlags = bindingFlags.data()};

        vk::DescriptorSetLayoutCreateInfo layoutInfo{.pNext = &flagsInfo, .bindingCount = static_cast<uint32_t>(bindings.size()), .pBindings = bindings.data()};

        return vk::raii::DescriptorSetLayout(devices->getLogicalDevice(), layoutInfo);
    }

    vk::raii::DescriptorPool createDescriptorPool() {
        std::array<vk::DescriptorPoolSize, 6> poolSizes = {{{vk::DescriptorType::eSampledImage, MAX_BINDLESS_TEXTURES},
                                                            {vk::DescriptorType::eSampler, MAX_BINDLESS_SAMPLERS},
                                                            {vk::DescriptorType::eStorageBuffer, 1},
                                                            {vk::DescriptorType::eStorageBuffer, 1},
                                                            {vk::DescriptorType::eStorageBuffer, 1},
                                                            {vk::DescriptorType::eSampledImage, MAX_CUBEMAPS}}};

        vk::DescriptorPoolCreateInfo poolInfo{};
        poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
        poolInfo.maxSets = 1; // Only one bindless set
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();

        return vk::raii::DescriptorPool(devices->getLogicalDevice(), poolInfo);
    }

    vk::raii::DescriptorSetLayout createDepthDescriptorSetLayout() {
        std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {{{.binding = 0,
                                                                    .descriptorType = vk::DescriptorType::eSampledImage,
                                                                    .descriptorCount = 1,
                                                                    .stageFlags = vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eVertex,
                                                                    .pImmutableSamplers = nullptr},
                                                                   {.binding = 1,
                                                                    .descriptorType = vk::DescriptorType::eSampler,
                                                                    .descriptorCount = 1,
                                                                    .stageFlags = vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eVertex,
                                                                    .pImmutableSamplers = nullptr}}};

        std::array<vk::DescriptorBindingFlags, 2> bindingFlags = {{vk::DescriptorBindingFlagBits::ePartiallyBound, vk::DescriptorBindingFlagBits::ePartiallyBound}};

        vk::DescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{.bindingCount = static_cast<uint32_t>(bindingFlags.size()), .pBindingFlags = bindingFlags.data()};

        vk::DescriptorSetLayoutCreateInfo layoutInfo{.pNext = &flagsInfo, .bindingCount = static_cast<uint32_t>(bindings.size()), .pBindings = bindings.data()};

        return vk::raii::DescriptorSetLayout(devices->getLogicalDevice(), layoutInfo);
    }
    vk::raii::DescriptorSet allocateDepthDescriptorSet() {
        std::array<uint32_t, 2> variableCounts = {1, 1};

        vk::DescriptorSetVariableDescriptorCountAllocateInfo variableCountInfo{.descriptorSetCount = 1, .pDescriptorCounts = variableCounts.data()};
        vk::DescriptorSetAllocateInfo allocInfo{
            .pNext = &variableCountInfo, .descriptorPool = *depthDescriptorPool, .descriptorSetCount = 1, .pSetLayouts = &(**depthDescriptorSetLayout)};

        vk::raii::DescriptorSets sets(devices->getLogicalDevice(), allocInfo);
        return std::move(sets[0]);
    }

    vk::raii::DescriptorPool createDepthDescriptorPool() {
        std::array<vk::DescriptorPoolSize, 2> poolSizes = {{{vk::DescriptorType::eSampledImage, 1}, {vk::DescriptorType::eSampler, 1}}};

        vk::DescriptorPoolCreateInfo poolInfo{};
        poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();

        return vk::raii::DescriptorPool(devices->getLogicalDevice(), poolInfo);
    }

    void initializeModelMatrixBuffer() {

        // Create single large buffer for all matrices
        vk::DeviceSize bufferSize = sizeof(glm::mat4) * MAX_BINDLESS_MODEL_MATRICES;

        vk::BufferCreateInfo bufferInfo{.size = bufferSize, .usage = vk::BufferUsageFlagBits::eStorageBuffer, .sharingMode = vk::SharingMode::eExclusive};

        modelMatrixBuffer.buffer = vk::raii::Buffer(devices->getLogicalDevice(), bufferInfo);

        vk::MemoryRequirements memRequirements = modelMatrixBuffer.buffer->getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo{
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, devices)};

        modelMatrixBuffer.memory = vk::raii::DeviceMemory(devices->getLogicalDevice(), allocInfo);
        modelMatrixBuffer.buffer->bindMemory(**modelMatrixBuffer.memory, 0);
        modelMatrixBuffer.bufferSize = bufferSize;

        modelMatrixBuffer.mappedData = modelMatrixBuffer.memory->mapMemory(0, bufferSize);

        glm::mat4* matrices = static_cast<glm::mat4*>(modelMatrixBuffer.mappedData);
        for (uint32_t i = 0; i < MAX_BINDLESS_MODEL_MATRICES; ++i) {
            matrices[i] = glm::mat4(1.0f);
        }

        updateModelMatrixDescriptor();
    }

    void initializeVertexBuffer() {

        // Create single large buffer
        vk::BufferCreateInfo bufferInfo{
            .size = VERTEX_BUFFER_SIZE, .usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, .sharingMode = vk::SharingMode::eExclusive};

        vertexBuffer.buffer = vk::raii::Buffer(devices->getLogicalDevice(), bufferInfo);

        // Allocate memory
        vk::MemoryRequirements memRequirements = vertexBuffer.buffer->getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo{
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, devices)};

        vertexBuffer.memory = vk::raii::DeviceMemory(devices->getLogicalDevice(), allocInfo);
        vertexBuffer.buffer->bindMemory(**vertexBuffer.memory, 0);
        vertexBuffer.totalSize = VERTEX_BUFFER_SIZE;

        // Keep buffer persistently mapped
        vertexBuffer.mappedData = vertexBuffer.memory->mapMemory(0, VERTEX_BUFFER_SIZE);

        // Update descriptor to point to the single buffer
        updateVertexBufferDescriptor();
    }

    void initializeLightBuffer() {
        // Create single large buffer for all point light
        vk::DeviceSize bufferSize = sizeof(Light) * MAX_LIGHTS;

        vk::BufferCreateInfo bufferInfo{.size = bufferSize, .usage = vk::BufferUsageFlagBits::eStorageBuffer, .sharingMode = vk::SharingMode::eExclusive};

        lightBuffer.buffer = vk::raii::Buffer(devices->getLogicalDevice(), bufferInfo);

        vk::MemoryRequirements memRequirements = lightBuffer.buffer->getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo{
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, devices)};

        lightBuffer.memory = vk::raii::DeviceMemory(devices->getLogicalDevice(), allocInfo);
        lightBuffer.buffer->bindMemory(**lightBuffer.memory, 0);
        lightBuffer.bufferSize = bufferSize;

        lightBuffer.mappedData = lightBuffer.memory->mapMemory(0, bufferSize);

        Light* lights = static_cast<Light*>(lightBuffer.mappedData);
        for (uint32_t i = 0; i < MAX_LIGHTS; ++i) {
            lights[i] = Light{};
        }

        updateLightDescriptor();
    }

    uint32_t allocateTextureImpl(vk::raii::Image&& image, vk::raii::DeviceMemory&& memory, vk::raii::ImageView&& imageView) {
        if (textureResources.size() >= MAX_BINDLESS_TEXTURES && freeTextureSlots.empty()) {
            throw std::runtime_error("Maximum bindless textures exceeded");
        }

        uint32_t index;

        if (!freeTextureSlots.empty()) {
            index = freeTextureSlots.front();
            freeTextureSlots.pop();
            textureResources[index].imageView = std::move(imageView);
            textureResources[index].image = std::move(image);
            textureResources[index].memory = std::move(memory);
        } else {
            index = textureResources.size();
            TextureResource resource;
            resource.imageView = std::move(imageView);
            resource.image = std::move(image);
            resource.memory = std::move(memory);
            textureResources.emplace_back(std::move(resource));
        }

        updateImageDescriptorSet(*bindlessDescriptorSet, index, **textureResources[index].imageView);
        return index;
    }

    uint32_t allocateCubeMapImpl(vk::raii::Image&& image, vk::raii::DeviceMemory&& memory, vk::raii::ImageView&& imageView) {
        if (cubemapResources.size() >= MAX_CUBEMAPS && freeCubemapSlots.empty()) {
            throw std::runtime_error("Maximum bindless textures exceeded");
        }

        uint32_t index;

        if (!freeCubemapSlots.empty()) {
            index = freeCubemapSlots.front();
            freeCubemapSlots.pop();
            cubemapResources[index].imageView = std::move(imageView);
            cubemapResources[index].image = std::move(image);
            cubemapResources[index].memory = std::move(memory);
        } else {
            index = cubemapResources.size();
            CubemapResource resource;
            resource.imageView = std::move(imageView);
            resource.image = std::move(image);
            resource.memory = std::move(memory);
            cubemapResources.emplace_back(std::move(resource));
        }

        updateCubemapDescriptor(index, **cubemapResources[index].imageView);
        return index;
    }

    uint32_t allocateSamplerImpl(vk::raii::Sampler&& sampler) {
        if (samplerResources.size() >= MAX_BINDLESS_SAMPLERS && freeSamplerSlots.empty()) {
            throw std::runtime_error("Maximum bindless samplers exceeded");
        }

        uint32_t index;
        if (!freeSamplerSlots.empty()) {
            index = freeSamplerSlots.front();
            freeSamplerSlots.pop();
            samplerResources[index].sampler = std::move(sampler);
        } else {
            index = samplerResources.size();
            SamplerResource resource;
            resource.sampler = std::move(sampler);
            samplerResources.emplace_back(std::move(resource));
        }

        updateSamplerDescriptor(*bindlessDescriptorSet, index, **samplerResources[index].sampler);
        return index;
    }

    uint32_t allocateVertexBufferImpl(const void* data, vk::DeviceSize dataSize, uint32_t vertexStride, uint32_t vertexCount) {
        // Find free slot
        uint32_t index;
        if (!freeVertexSlots.empty()) {
            index = freeVertexSlots.front();
            freeVertexSlots.pop();
        } else {
            index = 0;
            while (index < MAX_VERTEX_ALLOCATIONS && vertexAllocations[index].inUse) {
                index++;
            }

            if (index >= MAX_VERTEX_ALLOCATIONS) {
                throw std::runtime_error("Maximum vertex allocations exceeded");
            }
        }

        // Check if we have space in the buffer
        if (vertexBuffer.usedSize + dataSize > vertexBuffer.totalSize) {
            throw std::runtime_error("Vertex buffer full");
        }

        // Allocate space
        vk::DeviceSize offset = vertexBuffer.usedSize;
        vertexBuffer.usedSize += dataSize;

        // Update allocation tracking
        vertexAllocations[index].offset = offset;
        vertexAllocations[index].size = dataSize;
        vertexAllocations[index].vertexCount = vertexCount;
        vertexAllocations[index].vertexStride = vertexStride;
        vertexAllocations[index].inUse = true;

        // Copy data to buffer
        uint8_t* bufferData = static_cast<uint8_t*>(vertexBuffer.mappedData);
        memcpy(bufferData + offset, data, dataSize);
        return index;
    }

    void updateSamplerDescriptor(vk::raii::DescriptorSet& descriptorSet, uint32_t index, VkSampler sampler) {
        vk::DescriptorImageInfo imageInfo{.sampler = sampler};

        vk::WriteDescriptorSet descriptorWrite{
            .dstSet = descriptorSet, .dstBinding = 1, .dstArrayElement = index, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampler, .pImageInfo = &imageInfo};

        devices->getLogicalDevice().updateDescriptorSets(descriptorWrite, {});
    }

    void updateVertexBufferDescriptor() {
        vk::DescriptorBufferInfo bufferInfo{.buffer = **vertexBuffer.buffer, .offset = 0, .range = vertexBuffer.totalSize};

        vk::WriteDescriptorSet write{.dstSet = *bindlessDescriptorSet,
                                     .dstBinding = 2, // binding 2 for vertex buffer
                                     .dstArrayElement = 0,
                                     .descriptorCount = 1, // Only ONE descriptor for entire buffer
                                     .descriptorType = vk::DescriptorType::eStorageBuffer,
                                     .pBufferInfo = &bufferInfo};

        devices->getLogicalDevice().updateDescriptorSets(write, {});
    }

    void updateModelMatrixDescriptor() {
        vk::DescriptorBufferInfo bufferInfo{.buffer = **modelMatrixBuffer.buffer, .offset = 0, .range = modelMatrixBuffer.bufferSize};

        vk::WriteDescriptorSet write{.dstSet = *bindlessDescriptorSet,
                                     .dstBinding = 3,
                                     .dstArrayElement = 0,
                                     .descriptorCount = 1, // Only one descriptor for the entire buffer
                                     .descriptorType = vk::DescriptorType::eStorageBuffer,
                                     .pBufferInfo = &bufferInfo};

        devices->getLogicalDevice().updateDescriptorSets(write, {});
    }

    void updateLightDescriptor() {
        vk::DescriptorBufferInfo bufferInfo{.buffer = **lightBuffer.buffer, .offset = 0, .range = lightBuffer.bufferSize};

        vk::WriteDescriptorSet write{.dstSet = *bindlessDescriptorSet,
                                     .dstBinding = 4,
                                     .dstArrayElement = 0,
                                     .descriptorCount = 1, // Only one descriptor for the entire buffer
                                     .descriptorType = vk::DescriptorType::eStorageBuffer,
                                     .pBufferInfo = &bufferInfo};

        devices->getLogicalDevice().updateDescriptorSets(write, {});
    }

    void updateCubemapDescriptor(uint32_t index, VkImageView imageView) {
        vk::DescriptorImageInfo imageInfo{.imageView = imageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};

        vk::WriteDescriptorSet write{.sType = vk::StructureType::eWriteDescriptorSet,
                                     .dstSet = *bindlessDescriptorSet,
                                     .dstBinding = 5,
                                     .dstArrayElement = index,
                                     .descriptorCount = 1,
                                     .descriptorType = vk::DescriptorType::eSampledImage,
                                     .pImageInfo = &imageInfo};

        devices->getLogicalDevice().updateDescriptorSets(write, {});
    }

    void clearTextureDescriptor(uint32_t index) {
        // Set to null descriptor
        vk::DescriptorImageInfo imageInfo{.imageView = VK_NULL_HANDLE, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
        vk::WriteDescriptorSet descriptorWrite{.dstSet = *bindlessDescriptorSet,
                                               .dstBinding = 0,
                                               .dstArrayElement = index,
                                               .descriptorCount = 1,
                                               .descriptorType = vk::DescriptorType::eSampledImage,
                                               .pImageInfo = &imageInfo};

        devices->getLogicalDevice().updateDescriptorSets(descriptorWrite, {});
    }

    void clearCubemapDescriptor(uint32_t index) {
        // Set to null descriptor
        vk::DescriptorImageInfo imageInfo{.imageView = VK_NULL_HANDLE, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
        vk::WriteDescriptorSet descriptorWrite{.dstSet = *bindlessDescriptorSet,
                                               .dstBinding = 5,
                                               .dstArrayElement = index,
                                               .descriptorCount = 1,
                                               .descriptorType = vk::DescriptorType::eSampledImage,
                                               .pImageInfo = &imageInfo};

        devices->getLogicalDevice().updateDescriptorSets(descriptorWrite, {});
    }

    void clearSamplerDescriptor(uint32_t index) {
        vk::DescriptorImageInfo imageInfo{.sampler = VK_NULL_HANDLE};
        vk::WriteDescriptorSet descriptorWrite{.dstSet = *bindlessDescriptorSet,
                                               .dstBinding = 1,
                                               .dstArrayElement = index,
                                               .descriptorCount = 1,
                                               .descriptorType = vk::DescriptorType::eSampler,
                                               .pImageInfo = &imageInfo};

        devices->getLogicalDevice().updateDescriptorSets(descriptorWrite, {});
    }

    glm::mat4 createModelMatrix(glm::vec3 position, glm::quat rotation, glm::vec3 scale) {
        glm::mat4 translationMatrix = glm::translate(glm::mat4(1.0f), position);
        glm::mat4 rotationMatrix = glm::mat4_cast(rotation);
        glm::mat4 scaleMatrix = glm::scale(glm::mat4(1.0f), scale);

        return translationMatrix * rotationMatrix * scaleMatrix;
    }

    uint32_t createDefaultTexture(std::array<uint8_t, 4> color) {
        // Create 1x1 texture with specified color
        auto [image, memory, imageView] = createTexture(color.data(), 1, 1, vk::Format::eR8G8B8A8Srgb);

        uint32_t index;
        if (!freeTextureSlots.empty()) {
            index = freeTextureSlots.front();
            freeTextureSlots.pop();
            textureResources[index].imageView = std::move(imageView);
            textureResources[index].image = std::move(image);
            textureResources[index].memory = std::move(memory);
        } else {
            index = textureResources.size();
            TextureResource resource;
            resource.imageView = std::move(imageView);
            resource.image = std::move(image);
            resource.memory = std::move(memory);
            textureResources.emplace_back(std::move(resource));
        }

        updateImageDescriptorSet(*bindlessDescriptorSet, index, **textureResources[index].imageView);
        return index;
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

        vk::raii::Image image(devices->getLogicalDevice(), imageInfo);

        // Allocate memory
        vk::MemoryRequirements memRequirements = image.getMemoryRequirements();

        vk::MemoryAllocateInfo allocInfo{.allocationSize = memRequirements.size,
                                         .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal, devices)};

        vk::raii::DeviceMemory imageMemory(devices->getLogicalDevice(), allocInfo);
        image.bindMemory(*imageMemory, 0);

        // Upload data via staging buffer
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

        vk::raii::ImageView imageView(devices->getLogicalDevice(), viewInfo);

        return std::make_tuple(std::move(image), std::move(imageMemory), std::move(imageView));
    }

    void uploadTextureData(const vk::raii::Image& image, const void* data, uint32_t width, uint32_t height, vk::Format format, bool isCubemap = false) {
        uint32_t bytesPerPixel = getBytesPerPixel(format);
        vk::DeviceSize imageSize = static_cast<vk::DeviceSize>(width) * height * bytesPerPixel;
        vk::DeviceSize totalSize = isCubemap ? imageSize * 6 : imageSize;

        // Create staging buffer
        vk::BufferCreateInfo bufferInfo{.size = totalSize, .usage = vk::BufferUsageFlagBits::eTransferSrc, .sharingMode = vk::SharingMode::eExclusive};
        vk::raii::Buffer stagingBuffer(devices->getLogicalDevice(), bufferInfo);

        vk::MemoryRequirements memRequirements = stagingBuffer.getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo{
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, devices)};
        vk::raii::DeviceMemory stagingBufferMemory(devices->getLogicalDevice(), allocInfo);
        stagingBuffer.bindMemory(*stagingBufferMemory, 0);

        // Copy data to staging buffer
        void* mappedData = stagingBufferMemory.mapMemory(0, totalSize);
        memcpy(mappedData, data, totalSize);
        stagingBufferMemory.unmapMemory();

        // Transition image layout for transfer
        transitionImageLayout(image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, 1, isCubemap ? 6 : 1);

        // Copy buffer to image
        if (isCubemap) {
            copyBufferToImageCubemap(stagingBuffer, image, width, height, getBytesPerPixel(format));
        } else {
            copyBufferToImage(stagingBuffer, image, width, height);
        }

        // Transition image layout for shader access
        transitionImageLayout(image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, 1, isCubemap ? 6 : 1);
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

    vk::raii::CommandBuffer beginSingleTimeCommands() {
        vk::CommandBufferAllocateInfo allocInfo{};
        allocInfo.commandPool = *commandPool;
        allocInfo.level = vk::CommandBufferLevel::ePrimary;
        allocInfo.commandBufferCount = 1;

        vk::raii::CommandBuffers commandBuffers(devices->getLogicalDevice(), allocInfo);
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

        devices->getGraphicsQueue().submit(submitInfo, nullptr);
        devices->getGraphicsQueue().waitIdle();
    }
};