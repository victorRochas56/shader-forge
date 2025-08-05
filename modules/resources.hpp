#pragma once
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1 // for raii
#define VULKAN_HPP_NO_CONSTRUCTORS 1         // for structs constructors
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
#include <swapchain.hpp>
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
constexpr uint32_t MAX_BINDLESS_SAMPLERS = 256;
static constexpr vk::DeviceSize VERTEX_BUFFER_SIZE = 256 * 1024 * 1024; // max 256mb of vertex data
static constexpr uint32_t MAX_VERTEX_ALLOCATIONS = 2048;
constexpr uint32_t MAX_BINDLESS_MODEL_MATRICES = 2048;

class BindlessResourceManager {

  public:
    BindlessResourceManager(Devices* devices, vk::raii::CommandPool* commandPool) : devices(devices), commandPool(commandPool) {
        descriptorSetLayout = createDescriptorSetLayout();
        descriptorPool = createDescriptorPool();
        bindlessDescriptorSet = allocateDescriptorSet();
        // Reserve space for maximum resources
        textureResources.reserve(MAX_BINDLESS_TEXTURES);
        samplerResources.reserve(MAX_BINDLESS_SAMPLERS);
        vertexAllocations.resize(MAX_VERTEX_ALLOCATIONS);
        modelMatrixSlots.resize(MAX_BINDLESS_MODEL_MATRICES);
    }

    void initializeDefaults() {
        

        initializeVertexBuffer();
        initializeModelMatrixBuffer();

        // Create default sampler
        vk::SamplerCreateInfo samplerInfo{};
        samplerInfo.magFilter = vk::Filter::eLinear;
        samplerInfo.minFilter = vk::Filter::eLinear;
        samplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
        samplerInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
        samplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat;
        samplerInfo.anisotropyEnable = VK_TRUE;
        samplerInfo.maxAnisotropy = 16.0f;
        samplerInfo.borderColor = vk::BorderColor::eIntOpaqueBlack;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = vk::CompareOp::eAlways;
        samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;

        vk::raii::Sampler defaultSampler(devices->getLogicalDevice(), samplerInfo);
        defaultSamplerIndex = allocateSamplerImpl(std::move(defaultSampler));

        // Create default textures (1x1 pixel images)
        whiteTextureIndex = createDefaultTexture({255, 255, 255, 255});
        blackTextureIndex = createDefaultTexture({0, 0, 0, 255});
        defaultNormalIndex = createDefaultTexture({128, 128, 255, 255});

        defaultModelMatrixIndex = allocateModelMatrixBuffer(createModelMatrix(glm::vec3(0.0f), glm::quat(1.0f, 0.0f, 0.0f, 0.0f), glm::vec3(1.0f)));
    }

    const vk::raii::DescriptorSetLayout& getDescriptorSetLayout() const { return *descriptorSetLayout; }
    const vk::raii::DescriptorSet& getDescriptorSet() const { return *bindlessDescriptorSet; }
    uint32_t getWhiteTextureIndex() const { return whiteTextureIndex; }
    uint32_t getBlackTextureIndex() const { return blackTextureIndex; }
    uint32_t getDefaultNormalIndex() const { return defaultNormalIndex; }
    uint32_t getDefaultSamplerIndex() const { return defaultSamplerIndex; }
    uint32_t getDefaultVertexBufferIndex() const { return defaultVertexBufferIndex; }
    uint32_t getDefaultModelMatrixIndex() const { return defaultModelMatrixIndex; }

    uint32_t allocateTexture(vk::raii::ImageView&& imageView) {
        
        return allocateTextureImpl(std::move(imageView));
    }

    uint32_t allocateSampler(vk::raii::Sampler&& sampler) {
        
        return allocateSamplerImpl(std::move(sampler));
    }

    uint32_t loadTexture(const void* data, uint32_t width, uint32_t height, vk::Format format = vk::Format::eR8G8B8A8Srgb) {
        if (!data) {
            throw std::invalid_argument("Texture data cannot be null");
        }
        if (width == 0 || height == 0) {
            throw std::invalid_argument("Texture dimensions must be greater than 0");
        }

        

        if (textureResources.size() >= MAX_BINDLESS_TEXTURES && freeTextureSlots.empty()) {
            throw std::runtime_error("Maximum bindless textures exceeded");
        }

        auto [image, memory, imageView] = createTexture(data, width, height, format);

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

        updateDescriptorSet(index, **textureResources[index].imageView);
        return index;
    }

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
        if (offset + dataSize > allocation.size) {
            throw std::invalid_argument("Update would exceed allocation size");
        }

        // Direct memory copy into the big buffer
        uint8_t* bufferData = static_cast<uint8_t*>(vertexBuffer.mappedData);
        memcpy(bufferData + allocation.offset + offset, data, dataSize);
    }

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
            index = 1; // Start from 1, 0 is reserved for default
            while (index < MAX_BINDLESS_MODEL_MATRICES && modelMatrixSlots[index].inUse) {
                index++;
            }

            if (index >= MAX_BINDLESS_MODEL_MATRICES) {
                throw std::runtime_error("Maximum bindless model matrices exceeded");
            }
        }

        // Update CPU-side tracking
        modelMatrixSlots[index].matrix = matrix;
        modelMatrixSlots[index].inUse = true;

        // Update GPU buffer directly (fast - no map/unmap)
        glm::mat4* gpuMatrices = static_cast<glm::mat4*>(modelMatrixBuffer.mappedData);
        gpuMatrices[index] = matrix;

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

    void freeVertexBuffer(uint32_t allocationIndex) {
        

        if (allocationIndex >= MAX_VERTEX_ALLOCATIONS || !vertexAllocations[allocationIndex].inUse) {
            return;
        }

        if (allocationIndex == defaultVertexAllocationIndex) {
            throw std::runtime_error("Cannot free default vertex allocation");
        }

        // Mark space as available (simple linear allocator for now)
        // TODO: Implement proper free space management
        vertexAllocations[allocationIndex].reset();
        freeVertexSlots.push(allocationIndex);
    }

    void freeModelMatrix(uint32_t index) {
        

        if (index >= MAX_BINDLESS_MODEL_MATRICES || !modelMatrixSlots[index].inUse) {
            return;
        }

        if (index == defaultModelMatrixIndex) {
            throw std::runtime_error("Cannot free default model matrix");
        }

        modelMatrixSlots[index].reset();
        freeModelMatrixSlots.push(index);

        // Reset to identity matrix in GPU buffer
        glm::mat4* gpuMatrices = static_cast<glm::mat4*>(modelMatrixBuffer.mappedData);
        gpuMatrices[index] = glm::mat4(1.0f);
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
        

        uint32_t texturesUsed = textureResources.size() - freeTextureSlots.size();
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

    std::vector<TextureResource> textureResources;
    std::vector<SamplerResource> samplerResources;
    std::queue<uint32_t> freeTextureSlots;
    std::queue<uint32_t> freeSamplerSlots;
    VertexBufferResource vertexBuffer;
    std::vector<VertexAllocation> vertexAllocations;
    std::queue<uint32_t> freeVertexSlots;
    uint32_t defaultVertexAllocationIndex;
    ModelMatrixBufferResource modelMatrixBuffer;
    std::vector<ModelMatrixSlot> modelMatrixSlots;
    std::queue<uint32_t> freeModelMatrixSlots;

    std::optional<vk::raii::DescriptorSet> bindlessDescriptorSet;
    std::optional<vk::raii::DescriptorSetLayout> descriptorSetLayout;
    std::optional<vk::raii::DescriptorPool> descriptorPool;

    vk::raii::CommandPool* commandPool;
    Devices* devices;

    uint32_t defaultSamplerIndex;
    uint32_t whiteTextureIndex;
    uint32_t blackTextureIndex;
    uint32_t defaultNormalIndex;
    uint32_t defaultVertexBufferIndex;
    uint32_t defaultModelMatrixIndex;

    // Thread safety
    mutable std::mutex resourceMutex;

    vk::raii::DescriptorSetLayout createDescriptorSetLayout() {
        std::array<vk::DescriptorSetLayoutBinding, 4> bindings = {{// textures
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
                                                                    .pImmutableSamplers = nullptr}}};

        std::array<vk::DescriptorBindingFlags, 4> bindingFlags = {{vk::DescriptorBindingFlagBits::ePartiallyBound,
                                                                   vk::DescriptorBindingFlagBits::ePartiallyBound,
                                                                   vk::DescriptorBindingFlagBits::ePartiallyBound, 
                                                                   vk::DescriptorBindingFlagBits::ePartiallyBound}};

        vk::DescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{.bindingCount = static_cast<uint32_t>(bindingFlags.size()), .pBindingFlags = bindingFlags.data()};

        vk::DescriptorSetLayoutCreateInfo layoutInfo{.pNext = &flagsInfo, .bindingCount = static_cast<uint32_t>(bindings.size()), .pBindings = bindings.data()};

        return vk::raii::DescriptorSetLayout(devices->getLogicalDevice(), layoutInfo);
    }

    vk::raii::DescriptorPool createDescriptorPool() {
        std::array<vk::DescriptorPoolSize, 4> poolSizes = {{{vk::DescriptorType::eSampledImage, MAX_BINDLESS_TEXTURES},
                                                            {vk::DescriptorType::eSampler, MAX_BINDLESS_SAMPLERS},
                                                            {vk::DescriptorType::eStorageBuffer, 1},
                                                            {vk::DescriptorType::eStorageBuffer, 1}}};

        vk::DescriptorPoolCreateInfo poolInfo{};
        poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
        poolInfo.maxSets = 1; // Only one bindless set
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

        defaultModelMatrixIndex = 0;
        modelMatrixSlots[0].matrix = glm::mat4(1.0f);
        modelMatrixSlots[0].inUse = true;
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

        // Create default empty allocation
        std::vector<uint8_t> emptyData(64, 0);
        defaultVertexAllocationIndex = allocateVertexBufferImpl(emptyData.data(), emptyData.size(), 16, 4); // 4 vertices, 16 bytes each
    }

    vk::raii::DescriptorSet allocateDescriptorSet() {
        std::array<uint32_t, 4> variableCounts = {MAX_BINDLESS_TEXTURES, MAX_BINDLESS_SAMPLERS, 1, 1};

        vk::DescriptorSetVariableDescriptorCountAllocateInfo variableCountInfo{.descriptorSetCount = 1, .pDescriptorCounts = variableCounts.data()};
        vk::DescriptorSetAllocateInfo allocInfo{.pNext = &variableCountInfo, .descriptorPool = *descriptorPool, .descriptorSetCount = 1, .pSetLayouts = &(**descriptorSetLayout)};

        vk::raii::DescriptorSets sets(devices->getLogicalDevice(), allocInfo);
        return std::move(sets[0]);
    }

    uint32_t allocateTextureImpl(vk::raii::ImageView&& imageView) {
        if (textureResources.size() >= MAX_BINDLESS_TEXTURES && freeTextureSlots.empty()) {
            throw std::runtime_error("Maximum bindless textures exceeded");
        }

        uint32_t index;
        if (!freeTextureSlots.empty()) {
            index = freeTextureSlots.front();
            freeTextureSlots.pop();
            textureResources[index].imageView = std::move(imageView);
        } else {
            index = textureResources.size();
            TextureResource resource;
            resource.imageView = std::move(imageView);
            textureResources.emplace_back(std::move(resource));
        }

        updateDescriptorSet(index, **textureResources[index].imageView);
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

        updateSamplerDescriptor(index, **samplerResources[index].sampler);
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

    void updateDescriptorSet(uint32_t index, VkImageView imageView) {
        vk::DescriptorImageInfo imageInfo{.imageView = imageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};

        vk::WriteDescriptorSet write{.sType = vk::StructureType::eWriteDescriptorSet,
                                     .dstSet = *bindlessDescriptorSet,
                                     .dstBinding = 0,
                                     .dstArrayElement = index,
                                     .descriptorCount = 1,
                                     .descriptorType = vk::DescriptorType::eSampledImage,
                                     .pImageInfo = &imageInfo};

        devices->getLogicalDevice().updateDescriptorSets(write, {});
    }

    void updateSamplerDescriptor(uint32_t index, VkSampler sampler) {
        vk::DescriptorImageInfo imageInfo{.sampler = sampler};

        vk::WriteDescriptorSet descriptorWrite{.dstSet = *bindlessDescriptorSet,
                                               .dstBinding = 1,
                                               .dstArrayElement = index,
                                               .descriptorCount = 1,
                                               .descriptorType = vk::DescriptorType::eSampler,
                                               .pImageInfo = &imageInfo};

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

        updateDescriptorSet(index, **textureResources[index].imageView);
        return index;
    }

    std::tuple<vk::raii::Image, vk::raii::DeviceMemory, vk::raii::ImageView> createTexture(const void* data, uint32_t width, uint32_t height, vk::Format format) {
        // Create image
        vk::ImageCreateInfo imageInfo{.imageType = vk::ImageType::e2D,
                                      .format = format,
                                      .extent = {width, height, 1},
                                      .mipLevels = 1,
                                      .arrayLayers = 1,
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
        uploadTextureData(image, data, width, height, format);

        // Create image view
        vk::ImageViewCreateInfo viewInfo{
            .image = *image,
            .viewType = vk::ImageViewType::e2D,
            .format = format,
            .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1}};

        vk::raii::ImageView imageView(devices->getLogicalDevice(), viewInfo);

        return std::make_tuple(std::move(image), std::move(imageMemory), std::move(imageView));
    }

    void uploadTextureData(const vk::raii::Image& image, const void* data, uint32_t width, uint32_t height, vk::Format format) {
        uint32_t bytesPerPixel = getBytesPerPixel(format);
        vk::DeviceSize imageSize = static_cast<vk::DeviceSize>(width) * height * bytesPerPixel;

        // Create staging buffer
        vk::BufferCreateInfo bufferInfo{.size = imageSize, .usage = vk::BufferUsageFlagBits::eTransferSrc, .sharingMode = vk::SharingMode::eExclusive};

        vk::raii::Buffer stagingBuffer(devices->getLogicalDevice(), bufferInfo);

        vk::MemoryRequirements memRequirements = stagingBuffer.getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo{
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, devices)};

        vk::raii::DeviceMemory stagingBufferMemory(devices->getLogicalDevice(), allocInfo);
        stagingBuffer.bindMemory(*stagingBufferMemory, 0);

        // Copy data to staging buffer
        void* mappedData = stagingBufferMemory.mapMemory(0, imageSize);
        memcpy(mappedData, data, imageSize);
        stagingBufferMemory.unmapMemory();

        // Transition image layout and copy data
        transitionImageLayout(image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, 1);
        copyBufferToImage(stagingBuffer, image, width, height);
        transitionImageLayout(image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, 1);
    }

    void transitionImageLayout(const vk::raii::Image& image, const vk::ImageLayout oldLayout, const vk::ImageLayout newLayout, uint32_t mipLevels) {
        vk::raii::CommandBuffer commandBuffer = beginSingleTimeCommands();

        vk::ImageMemoryBarrier barrier{
            .oldLayout = oldLayout,
            .newLayout = newLayout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = *image,
            .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = mipLevels, .baseArrayLayer = 0, .layerCount = 1}};

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
        } else {
            throw std::invalid_argument("unsupported layout transition!");
        }

        commandBuffer.pipelineBarrier(sourceStage, destinationStage, {}, {}, {}, barrier);
        endSingleTimeCommands(commandBuffer);
    }

    void copyBufferToImage(const vk::raii::Buffer& srcBuffer, const vk::raii::Image& dstImage, uint32_t width, uint32_t height) {
        vk::raii::CommandBuffer commandBuffer = beginSingleTimeCommands();

        vk::BufferImageCopy region{.bufferOffset = 0,
                                   .bufferRowLength = 0,
                                   .bufferImageHeight = 0,
                                   .imageSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
                                   .imageOffset = {0, 0, 0},
                                   .imageExtent = {width, height, 1}};

        commandBuffer.copyBufferToImage(*srcBuffer, *dstImage, vk::ImageLayout::eTransferDstOptimal, region);
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
/*


struct Material {
    uint32_t albedoTextureIndex;
    uint32_t normalTextureIndex;
    uint32_t roughnessTextureIndex;
    uint32_t samplerIndex;
};

class MaterialManager {
  private:
    vk::raii::Buffer materialBuffer;
    vk::raii::DeviceMemory materialBufferMemory;
    std::vector<Material> materials;
    Devices* devices;
    BindlessResourceManager* bindlessManager;
    size_t bufferSize;
    bool needsUpdate;
    mutable std::mutex materialMutex;

  public:
    MaterialManager(Devices& devices, BindlessResourceManager& bindlessManager, size_t maxMaterials = 1024)
        : devices(&devices), bindlessManager(&bindlessManager), bufferSize(maxMaterials * sizeof(Material)), needsUpdate(false) {

        materials.reserve(maxMaterials);
        createMaterialBuffer();
    }

    uint32_t createMaterial(uint32_t albedo = 0, uint32_t normal = 0, uint32_t roughness = 0, uint32_t sampler = 0) {
        std::lock_guard<std::mutex> lock(materialMutex);

        // Use defaults if not specified
        if (albedo == 0)
            albedo = bindlessManager->getWhiteTextureIndex();
        if (normal == 0)
            normal = bindlessManager->getDefaultNormalIndex();
        if (roughness == 0)
            roughness = bindlessManager->getWhiteTextureIndex();
        if (sampler == 0)
            sampler = bindlessManager->getDefaultSamplerIndex();

        uint32_t materialIndex = materials.size();
        materials.push_back({albedo, normal, roughness, sampler});

        needsUpdate = true;
        return materialIndex;
    }

    void updateMaterial(uint32_t index, uint32_t albedo, uint32_t normal, uint32_t roughness, uint32_t sampler) {
        std::lock_guard<std::mutex> lock(materialMutex);

        if (index >= materials.size()) {
            throw std::out_of_range("Material index out of range");
        }

        materials[index] = {albedo, normal, roughness, sampler};
        needsUpdate = true;
    }

    const Material& getMaterial(uint32_t index) const {
        std::lock_guard<std::mutex> lock(materialMutex);

        if (index >= materials.size()) {
            throw std::out_of_range("Material index out of range");
        }

        return materials[index];
    }

    size_t getMaterialCount() const {
        std::lock_guard<std::mutex> lock(materialMutex);
        return materials.size();
    }

    const vk::raii::Buffer& getMaterialBuffer() const { return materialBuffer; }

    void updateMaterialBuffer() {
        std::lock_guard<std::mutex> lock(materialMutex);

        if (!needsUpdate || materials.empty()) {
            return;
        }

        size_t dataSize = materials.size() * sizeof(Material);
        if (dataSize > bufferSize) {
            throw std::runtime_error("Material data exceeds buffer size");
        }

        // Map memory and copy data
        void* mappedData = materialBufferMemory.mapMemory(0, dataSize);
        memcpy(mappedData, materials.data(), dataSize);
        materialBufferMemory.unmapMemory();

        needsUpdate = false;
    }

    // Force update the buffer (useful for rendering loop)
    void flushMaterialBuffer() { updateMaterialBuffer(); }

  private:
    void createMaterialBuffer() {
        // Create buffer
        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = bufferSize;
        bufferInfo.usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eUniformBuffer;
        bufferInfo.sharingMode = vk::SharingMode::eExclusive;

        materialBuffer = vk::raii::Buffer(devices->getLogicalDevice(), bufferInfo);

        // Allocate memory
        vk::MemoryRequirements memRequirements = materialBuffer.getMemoryRequirements();

        vk::MemoryAllocateInfo allocInfo{};
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, *devices);

        materialBufferMemory = vk::raii::DeviceMemory(devices->getLogicalDevice(), allocInfo);
        materialBuffer.bindMemory(*materialBufferMemory, 0);
    }
};

// Example usage class showing how to integrate everything
class RenderSystem {
  private:
    std::unique_ptr<BindlessResourceManager> bindlessManager;
    std::unique_ptr<MaterialManager> materialManager;
    Devices* devices;

  public:
    RenderSystem(Devices& devices, vk::raii::CommandPool& commandPool) : devices(&devices) {

        bindlessManager = std::make_unique<BindlessResourceManager>(devices, commandPool);
        bindlessManager->initializeDefaults();

        materialManager = std::make_unique<MaterialManager>(devices, *bindlessManager);
    }

    // Load a texture from file data
    uint32_t loadTexture(const void* data, uint32_t width, uint32_t height, vk::Format format = vk::Format::eR8G8B8A8Srgb) {
        return bindlessManager->loadTexture(data, width, height, format);
    }

    // Create a custom sampler
    uint32_t createSampler(const vk::SamplerCreateInfo& samplerInfo) {
        vk::raii::Sampler sampler(devices->getLogicalDevice(), samplerInfo);
        return bindlessManager->allocateSampler(std::move(sampler));
    }

    // Create a material with specific textures
    uint32_t createMaterial(uint32_t albedoTexture, uint32_t normalTexture = 0, uint32_t roughnessTexture = 0, uint32_t sampler = 0) {
        return materialManager->createMaterial(albedoTexture, normalTexture, roughnessTexture, sampler);
    }

    // Get the descriptor set for binding
    const vk::raii::DescriptorSet& getBindlessDescriptorSet() const { return bindlessManager->getDescriptorSet(); }

    // Get the descriptor set layout for pipeline creation
    const vk::raii::DescriptorSetLayout& getBindlessDescriptorSetLayout() const { return bindlessManager->getDescriptorSetLayout(); }

    // Get material buffer for binding
    const vk::raii::Buffer& getMaterialBuffer() const { return materialManager->getMaterialBuffer(); }

    // Update material buffer before rendering
    void updateBuffers() { materialManager->flushMaterialBuffer(); }

    // Get resource usage statistics
    BindlessResourceManager::ResourceStats getResourceStats() const { return bindlessManager->getResourceStats(); }

    // Cleanup specific resources
    void freeTexture(uint32_t index) { bindlessManager->freeTexture(index); }

    void freeSampler(uint32_t index) { bindlessManager->freeSampler(index); }

    // Get default resource indices
    uint32_t getDefaultWhiteTexture() const { return bindlessManager->getWhiteTextureIndex(); }
    uint32_t getDefaultBlackTexture() const { return bindlessManager->getBlackTextureIndex(); }
    uint32_t getDefaultNormalTexture() const { return bindlessManager->getDefaultNormalIndex(); }
    uint32_t getDefaultSampler() const { return bindlessManager->getDefaultSamplerIndex(); }


};

// Initialize the system
RenderSystem renderSystem(devices, commandPool);

// Load textures
uint32_t albedoTex = renderSystem.loadTexture(albedoData, width, height);
uint32_t normalTex = renderSystem.loadTexture(normalData, width, height);

// Create material
uint32_t materialId = renderSystem.createMaterial(albedoTex, normalTex);

// Before rendering
renderSystem.updateBuffers();

// Bind descriptor set
commandBuffer.bindDescriptorSets(
    vk::PipelineBindPoint::eGraphics,
    pipelineLayout,
    0, 1,
    &*renderSystem.getBindlessDescriptorSet(),
    0, nullptr
);


layout(push_constant) uniform PushConstants {
    uint vertexOffset;      // Offset in bytes into the vertex buffer
    uint vertexStride;      // Size of each vertex
    uint modelMatrixIndex;
} pc;

layout(set = 0, binding = 2) readonly restrict buffer VertexBuffer {
    uint8_t data[];  // Raw vertex data
} vertexBuffer;

struct Vertex {
    vec3 position;
    vec3 normal;
    vec2 texCoord;
};

void main() {
    // Calculate vertex data location
    uint vertexIndex = gl_VertexIndex;
    uint byteOffset = pc.vertexOffset + (vertexIndex * pc.vertexStride);

    // Read vertex data (you'd have helper functions for this)
    Vertex vertex = unpackVertex(vertexBuffer.data, byteOffset);

    // Use the vertex data
    gl_Position = mvp * vec4(vertex.position, 1.0);
}*/