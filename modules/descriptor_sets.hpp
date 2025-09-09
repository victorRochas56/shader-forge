#pragma once
#ifndef VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1 // for raii
#endif
#ifndef VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_NO_CONSTRUCTORS 1 // for structs constructors
#endif

#include <optional>
#include <vector>
#include <queue>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include "devices.hpp"
#include "resources.hpp"
#include "utils.hpp"

constexpr uint32_t MAX_TEXTURES = 2048;
constexpr uint32_t MAX_CUBEMAPS = 2048;
constexpr uint32_t MAX_SAMPLERS = 2048;

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

// variable size allocations (ie: a set vertices of a mesh)

struct VariableBufferAllocation {
    uint32_t offset;
    uint32_t size;
    uint32_t stride;
    uint32_t count;
};

struct VariableBufferResource {
    vk::raii::Buffer buffer;
    vk::raii::DeviceMemory memory;
    void* mappedData = nullptr;
    vk::DeviceSize bufferSize;
    std::vector<VariableBufferAllocation> allocations;
    uint32_t bindingIndex = 0;
    uint32_t maxSize = 2048;

    VariableBufferResource(vk::raii::Buffer&& buf, vk::raii::DeviceMemory&& mem, void* mapped, vk::DeviceSize size, uint32_t max)
        : buffer(std::move(buf)), memory(std::move(mem)), mappedData(mapped), bufferSize(size), maxSize(max) {}
};

struct FixedBufferAllocation {
    uint32_t index;
    bool inUse = false;
};

// Base class for type-erased operations
struct FixedBufferResourceBase {
    uint32_t bindingIndex = 0;
    vk::raii::Buffer buffer;
    vk::raii::DeviceMemory memory;
    void* mappedData = nullptr;
    vk::DeviceSize bufferSize;
    std::vector<FixedBufferAllocation> allocations;
    std::queue<uint32_t> freeSlots;
    uint32_t maxSize = 2048;

    virtual ~FixedBufferResourceBase() = default;
    virtual uint32_t allocateImpl(const void* data) = 0;
    virtual void updateImpl(uint32_t index, const void* data) = 0;
    virtual size_t getElementSize() const = 0;

  protected:
    FixedBufferResourceBase(vk::raii::Buffer&& buf, vk::raii::DeviceMemory&& mem) : buffer(std::move(buf)), memory(std::move(mem)) {}
};

// Templated derived class
template <typename T> struct FixedBufferResource : FixedBufferResourceBase {
    std::vector<T> data;

    FixedBufferResource(vk::raii::Buffer&& buf, vk::raii::DeviceMemory&& mem) : FixedBufferResourceBase(std::move(buf), std::move(mem)) { data.reserve(maxSize); }

    uint32_t allocateImpl(const void* dataPtr) override {
        const T* typedData = static_cast<const T*>(dataPtr);
        return allocateTyped(*typedData);
    }

    void updateImpl(uint32_t index, const void* dataPtr) override {
        const T* typedData = static_cast<const T*>(dataPtr);
        updateTyped(index, *typedData);
    }

    size_t getElementSize() const override { return sizeof(T); }

    uint32_t allocateTyped(const T& newData) {
        uint32_t index;

        if (!freeSlots.empty()) {
            // Reuse freed slot
            index = freeSlots.front();
            freeSlots.pop();
            allocations[index] = FixedBufferAllocation{.index = index, .inUse = true};
            data[index] = newData;
        } else {
            // Allocate new slot
            if (data.size() >= maxSize) {
                throw std::runtime_error("Fixed buffer capacity exceeded");
            }
            index = static_cast<uint32_t>(data.size());
            allocations.push_back(FixedBufferAllocation{.index = index, .inUse = true});
            data.push_back(newData);
        }

        // Update GPU memory
        T* dataBuffer = static_cast<T*>(mappedData);
        dataBuffer[index] = newData;

        return index;
    }

    void updateTyped(uint32_t index, const T& newData) {
        if (index >= allocations.size() || !allocations[index].inUse) {
            throw std::runtime_error("Invalid buffer index or allocation not in use");
        }

        data[index] = newData;

        // Update GPU memory
        T* dataBuffer = static_cast<T*>(mappedData);
        dataBuffer[index] = newData;
    }

    void free(uint32_t index) {
        if (index >= allocations.size() || !allocations[index].inUse) {
            return; // Already freed or invalid
        }
        allocations[index].inUse = false;
        freeSlots.push(index);
    }
};

//
// ADD FREEING TEXTURES AND SAMPLERS
// NEXT IS MODEL LOADING I THINK
//

class DescriptorSet {
  public:
    DescriptorSet(Device& device, ResourceManager& resourceManager, vk::raii::CommandPool* commandPool)
        : device(device), resourceManager(resourceManager), commandPool(commandPool) {}

    void createDescriptorSet() {
        std::vector<vk::DescriptorSetLayoutBinding> layoutBindings;
        std::vector<vk::DescriptorPoolSize> poolSizes;
        std::vector<vk::DescriptorBindingFlags> bindingFlags;
        std::vector<uint32_t> variableCounts;
        uint32_t bindingIndex = 0;

        // textures
        layoutBindings.push_back(vk::DescriptorSetLayoutBinding{.binding = bindingIndex,
                                                                .descriptorType = vk::DescriptorType::eSampledImage,
                                                                .descriptorCount = MAX_TEXTURES,
                                                                .stageFlags = vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eVertex});
        poolSizes.push_back(vk::DescriptorPoolSize(vk::DescriptorType::eSampledImage, MAX_TEXTURES));
        bindingFlags.push_back(vk::DescriptorBindingFlagBits::ePartiallyBound);
        variableCounts.push_back(MAX_TEXTURES);
        bindingIndex++;

        // cubemaps
        layoutBindings.push_back(vk::DescriptorSetLayoutBinding{.binding = bindingIndex,
                                                                .descriptorType = vk::DescriptorType::eSampledImage,
                                                                .descriptorCount = MAX_CUBEMAPS,
                                                                .stageFlags = vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eVertex});
        poolSizes.push_back(vk::DescriptorPoolSize(vk::DescriptorType::eSampledImage, MAX_CUBEMAPS));
        bindingFlags.push_back(vk::DescriptorBindingFlagBits::ePartiallyBound);
        variableCounts.push_back(MAX_CUBEMAPS);
        bindingIndex++;

        // samplers
        layoutBindings.push_back(vk::DescriptorSetLayoutBinding{.binding = bindingIndex,
                                                                .descriptorType = vk::DescriptorType::eSampler,
                                                                .descriptorCount = MAX_SAMPLERS,
                                                                .stageFlags = vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eVertex});
        poolSizes.push_back(vk::DescriptorPoolSize(vk::DescriptorType::eSampler, MAX_SAMPLERS));
        bindingFlags.push_back(vk::DescriptorBindingFlagBits::ePartiallyBound);
        variableCounts.push_back(MAX_SAMPLERS);
        bindingIndex++;

        // variable size buffer (vertices)
        if (variableBuffer) {
            layoutBindings.push_back(vk::DescriptorSetLayoutBinding{.binding = bindingIndex,
                                                                    .descriptorType = vk::DescriptorType::eStorageBuffer,
                                                                    .descriptorCount = 1,
                                                                    .stageFlags = vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eVertex});
            poolSizes.push_back(vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 1));
            bindingFlags.push_back(vk::DescriptorBindingFlagBits::ePartiallyBound);
            variableCounts.push_back(1);
            (*variableBuffer).bindingIndex = bindingIndex;
            bindingIndex++;
        }

        for (auto& fixedBuffer : fixedBuffers) {
            layoutBindings.push_back(vk::DescriptorSetLayoutBinding{.binding = bindingIndex,
                                                                    .descriptorType = vk::DescriptorType::eStorageBuffer,
                                                                    .descriptorCount = 1,
                                                                    .stageFlags = vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eVertex});
            poolSizes.push_back(vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 1));
            bindingFlags.push_back(vk::DescriptorBindingFlagBits::ePartiallyBound);
            variableCounts.push_back(1);
            fixedBuffer->bindingIndex = bindingIndex;
            bindingIndex++;
        }

        vk::DescriptorPoolCreateInfo poolInfo{};
        poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
        poolInfo.maxSets = 1; // Only one bindless set
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        descriptorPool = vk::raii::DescriptorPool(device.getDevice(), poolInfo);

        vk::DescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{.bindingCount = static_cast<uint32_t>(bindingFlags.size()), .pBindingFlags = bindingFlags.data()};
        vk::DescriptorSetLayoutCreateInfo layoutInfo{.pNext = &flagsInfo, .bindingCount = static_cast<uint32_t>(layoutBindings.size()), .pBindings = layoutBindings.data()};
        descriptorSetLayout = vk::raii::DescriptorSetLayout(device.getDevice(), layoutInfo);

        vk::DescriptorSetVariableDescriptorCountAllocateInfo variableCountInfo{.descriptorSetCount = 1, .pDescriptorCounts = variableCounts.data()};
        vk::DescriptorSetAllocateInfo allocInfo{.pNext = &variableCountInfo, .descriptorPool = *descriptorPool, .descriptorSetCount = 1, .pSetLayouts = &**descriptorSetLayout};

        vk::raii::DescriptorSets sets(device.getDevice(), allocInfo);
        descriptorSet = std::move(sets[0]);
    }

    uint32_t allocateTexture(vk::raii::Image image, vk::raii::DeviceMemory memory, vk::raii::ImageView imageView, bool isCubeMap = false) {
        if (textureResources.size() >= MAX_TEXTURES && freeTextureSlots.empty()) {
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
        vk::DescriptorImageInfo imageInfo{.imageView = imageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
        vk::WriteDescriptorSet write{.sType = vk::StructureType::eWriteDescriptorSet,
                                     .dstSet = *descriptorSet,
                                     .dstBinding = isCubeMap ? cubemapBindingIndex : textureBindingIndex,
                                     .dstArrayElement = index,
                                     .descriptorCount = 1,
                                     .descriptorType = vk::DescriptorType::eSampledImage,
                                     .pImageInfo = &imageInfo};

        device.getDevice().updateDescriptorSets(write, {});
        return index;
    }
    void freeTexture(uint32_t index) {}

    uint32_t allocateSampler(vk::Filter filter, vk::SamplerMipmapMode mipmapMode, vk::SamplerAddressMode addressMode, vk::Bool32 anisotropyEnabled, float maxAnisotropy,
                             vk::Bool32 compareEnable, vk::CompareOp compareOp, vk::BorderColor borderColor) {

        vk::SamplerCreateInfo createInfo{
            .pNext = nullptr,
            .flags = {},
            .magFilter = filter,
            .minFilter = filter,
            .mipmapMode = mipmapMode,
            .addressModeU = addressMode,
            .addressModeV = addressMode,
            .addressModeW = addressMode,
            .anisotropyEnable = anisotropyEnabled,
            .maxAnisotropy = maxAnisotropy,
            .compareEnable = compareEnable,
            .compareOp = compareOp,
            .borderColor = borderColor,
        };
        vk::raii::Sampler sampler(device.getDevice(), createInfo);
        if (samplerResources.size() >= MAX_SAMPLERS && freeSamplerSlots.empty()) {
            throw std::runtime_error("Maximum samplers exceeded");
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
        }
        vk::DescriptorImageInfo imageInfo{.sampler = sampler};
        vk::WriteDescriptorSet write{.sType = vk::StructureType::eWriteDescriptorSet,
                                     .dstSet = *descriptorSet,
                                     .dstBinding = samplerBindingIndex,
                                     .dstArrayElement = index,
                                     .descriptorCount = 1,
                                     .descriptorType = vk::DescriptorType::eSampler,
                                     .pImageInfo = &imageInfo};
        device.getDevice().updateDescriptorSets(write, {});
        return index;
    }

    void freeSampler(uint32_t index) {}

    // Create a new fixed buffer of type T
    template <typename T> uint32_t createFixedBuffer(uint32_t maxElements = 2048) {
        // Create Vulkan buffer
        vk::DeviceSize bufferSize = maxElements * sizeof(T);

        vk::BufferCreateInfo bufferInfo{.size = bufferSize, .usage = vk::BufferUsageFlagBits::eStorageBuffer, .sharingMode = vk::SharingMode::eExclusive};

        vk::raii::Buffer buffer(device.getDevice(), bufferInfo);

        vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
        uint32_t memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, device);

        vk::MemoryAllocateInfo allocInfo{.allocationSize = memRequirements.size, .memoryTypeIndex = memoryTypeIndex};

        vk::raii::DeviceMemory memory(device.getDevice(), allocInfo);
        buffer.bindMemory(*memory, 0);
        void* mappedData = memory.mapMemory(0, bufferSize);

        // Create typed buffer resource
        auto fixedBuffer = std::make_unique<FixedBufferResource<T>>(std::move(buffer), std::move(memory));
        fixedBuffer->mappedData = mappedData;
        fixedBuffer->bufferSize = bufferSize;
        fixedBuffer->maxSize = maxElements;

        uint32_t bufferIndex = static_cast<uint32_t>(fixedBuffers.size());
        fixedBuffers.push_back(std::move(fixedBuffer));
        return bufferIndex;
    }

    template <typename T> uint32_t allocateFixedBuffer(uint32_t bufferIndex, const T& data) {
        if (bufferIndex >= fixedBuffers.size()) {
            throw std::runtime_error("Invalid buffer index");
        }
        return fixedBuffers[bufferIndex]->allocateImpl(&data);
    }

    void freeFixedBuffer(uint32_t bufferIndex, uint32_t index) {
        if (bufferIndex >= fixedBuffers.size()) {
            return;
        }

        // We need to cast to access the free method, but this is safe since we control the creation
        auto* basePtr = fixedBuffers[bufferIndex].get();
        // Since we can't make free virtual easily, we'll access allocations directly
        if (index < basePtr->allocations.size() && basePtr->allocations[index].inUse) {
            basePtr->allocations[index].inUse = false;
            basePtr->freeSlots.push(index);
        }
    }

    template <typename T> void updateFixedBuffer(uint32_t bufferIndex, uint32_t index, const T& data) {
        if (bufferIndex >= fixedBuffers.size()) {
            throw std::runtime_error("Invalid buffer index");
        }
        fixedBuffers[bufferIndex]->updateImpl(index, &data);
    }

    // Variable buffer operations
    void createVariableBuffer(uint32_t maxSizeBytes = 1024 * 1024) { // Default 1MB
        vk::BufferCreateInfo bufferInfo{.size = maxSizeBytes, .usage = vk::BufferUsageFlagBits::eStorageBuffer, .sharingMode = vk::SharingMode::eExclusive};

        vk::raii::Buffer buffer(device.getDevice(), bufferInfo);
        vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
        uint32_t memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, device);

        vk::MemoryAllocateInfo allocInfo{.allocationSize = memRequirements.size, .memoryTypeIndex = memoryTypeIndex};

        vk::raii::DeviceMemory memory(device.getDevice(), allocInfo);
        buffer.bindMemory(*memory, 0);
        void* mappedData = memory.mapMemory(0, maxSizeBytes);

        variableBuffer = VariableBufferResource{std::move(buffer), std::move(memory), mappedData, maxSizeBytes, maxSizeBytes};
    }

    template <typename T> uint32_t allocateVariableBuffer(const std::vector<T>& data) {
        if (!variableBuffer) {
            throw std::runtime_error("Variable buffer not created. Call createVariableBuffer() first.");
        }
        uint32_t elementSize = sizeof(T);
        uint32_t totalSize = static_cast<uint32_t>(data.size()) * elementSize;
        uint32_t count = static_cast<uint32_t>(data.size());
        // Find a suitable offset (simple linear allocation for now)
        uint32_t offset = findVariableBufferOffset(totalSize);
        if (offset + totalSize > variableBuffer->bufferSize) {
            throw std::runtime_error("Variable buffer out of space");
        }
        // Copy data to GPU memory
        T* gpuData = reinterpret_cast<T*>(static_cast<char*>(variableBuffer->mappedData) + offset);
        std::memcpy(gpuData, data.data(), totalSize);

        // Create allocation record
        uint32_t allocationIndex = static_cast<uint32_t>(variableBuffer->allocations.size());
        variableBuffer->allocations.push_back(VariableBufferAllocation{.offset = offset, .size = totalSize, .stride = elementSize, .count = count});
        return allocationIndex;
    }

    template <typename T> void updateVariableBuffer(uint32_t allocationIndex, const std::vector<T>& newData) {
        if (!variableBuffer || allocationIndex >= variableBuffer->allocations.size()) {
            throw std::runtime_error("Invalid variable buffer allocation");
        }
        auto& allocation = variableBuffer->allocations[allocationIndex];
        uint32_t elementSize = sizeof(T);
        uint32_t newTotalSize = static_cast<uint32_t>(newData.size()) * elementSize;
        // Check if new data fits in existing allocation
        if (newTotalSize > allocation.size) {
            throw std::runtime_error("New data too large for existing allocation. Consider reallocating.");
        }
        // Update allocation info
        allocation.count = static_cast<uint32_t>(newData.size());
        // Copy new data to GPU memory
        T* gpuData = static_cast<T*>(static_cast<char*>(variableBuffer->mappedData) + allocation.offset);
        std::memcpy(gpuData, newData.data(), newTotalSize);
    }

    void freeVariableBuffer(uint32_t allocationIndex) {
        if (!variableBuffer || allocationIndex >= variableBuffer->allocations.size()) {
            return;
        }
        // For now, just mark as invalid by setting size to 0
        // A more sophisticated implementation would have a free list and defragmentation
        variableBuffer->allocations.at(allocationIndex).size = 0;
    }

    VariableBufferAllocation getVariableBufferAllocation(uint32_t allocationIndex) const {
        if (!variableBuffer || allocationIndex >= variableBuffer->allocations.size()) {
            throw std::runtime_error("Invalid variable buffer allocation");
        }
        return variableBuffer->allocations.at(allocationIndex);
    }

    vk::raii::DescriptorSet& getDescriptorSet() { return *descriptorSet; }
    vk::raii::DescriptorSetLayout& getDescriptorSetLayout() { return *descriptorSetLayout; }

  private:
    Device& device;
    ResourceManager& resourceManager;
    vk::raii::CommandPool* commandPool;

    std::vector<SamplerResource> samplerResources;
    uint32_t samplerBindingIndex;
    std::queue<uint32_t> freeSamplerSlots;
    std::vector<TextureResource> textureResources;
    uint32_t textureBindingIndex;
    uint32_t cubemapBindingIndex;
    std::queue<uint32_t> freeTextureSlots;
    std::optional<VariableBufferResource> variableBuffer; // vertices
    std::vector<std::unique_ptr<FixedBufferResourceBase>> fixedBuffers;

    std::optional<vk::raii::DescriptorSetLayout> descriptorSetLayout;
    std::optional<vk::raii::DescriptorSet> descriptorSet;
    std::optional<vk::raii::DescriptorPool> descriptorPool;

    // Simple linear allocator for variable buffer
    uint32_t findVariableBufferOffset(uint32_t requestedSize) {
        if (!variableBuffer || variableBuffer->allocations.empty()) {
            return 0; // First allocation
        }

        // Sort allocations by offset (simple approach)
        std::vector<std::pair<uint32_t, uint32_t>> usedRanges; // {offset, size}
        for (const auto& alloc : variableBuffer->allocations) {
            if (alloc.size > 0) { // Active allocation
                usedRanges.push_back({alloc.offset, alloc.size});
            }
        }

        std::sort(usedRanges.begin(), usedRanges.end());

        // Find first gap that fits
        uint32_t currentOffset = 0;
        for (const auto& range : usedRanges) {
            if (range.first - currentOffset >= requestedSize) {
                return currentOffset; // Found a gap
            }
            currentOffset = range.first + range.second;
        }

        // No gap found, allocate at end
        return currentOffset;
    }
};