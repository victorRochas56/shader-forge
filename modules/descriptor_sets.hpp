#pragma once
#ifndef VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#endif
#ifndef VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_NO_CONSTRUCTORS 1
#endif

#include <optional>
#include <queue>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include "constants.hpp"
#include "devices.hpp"
#include "resources.hpp"
#include "utils.hpp"

/*
there are 4 kinds of resources:
Texture and samplers - self explanatory
Variable Buffers - buffer where each allocation can be a different size (meshes for example)
Fixed Buffers - allocations have a fixed size, 
some fixed buffers also have the perFrame flag meaning when they allocate they do so for every frame in flight 

finally the DescriptorSet class handles creation, allocation, updating and cleanup of these buffers
*/

struct TextureResource {
    std::optional<vk::raii::ImageView> imageView;
    std::optional<vk::raii::Image> image;
    std::optional<vk::raii::DeviceMemory> memory;
    std::string source;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t mipLevels = 1;
    void reset() {
        imageView.reset();
        image.reset();
        memory.reset();
        source.clear();
        width = 0;
        height = 0;
        mipLevels = 1;
    }
    bool isEmpty() const { return !imageView.has_value(); }
};

struct SamplerResource {
    std::optional<vk::raii::Sampler> sampler;
    void reset() { sampler.reset(); }
    bool isEmpty() const { return !sampler.has_value(); }
};

struct VariableBufferAllocation {
    uint32_t offset;
    uint32_t size;
    uint32_t stride;
    uint32_t count;
};

struct VariableBufferResource {
    vk::DeviceAddress address;
    vk::raii::Buffer buffer;
    vk::raii::DeviceMemory memory;
    void* mappedData = nullptr;
    vk::DeviceSize bufferSize;
    std::vector<VariableBufferAllocation> allocations;
    uint32_t bindingIndex = 0;
    uint32_t maxSize = MAX_VARIABLE_BUFFER;

    VariableBufferResource(vk::DeviceAddress add, vk::raii::Buffer&& buf, vk::raii::DeviceMemory&& mem, void* mapped, vk::DeviceSize size, uint32_t max)
        : address(add), buffer(std::move(buf)), memory(std::move(mem)), mappedData(mapped), bufferSize(size), maxSize(max) {}
};

struct FixedBufferAllocation {
    uint32_t index;
    bool inUse = false;
};

struct FixedBufferResourceBase {
    vk::DeviceAddress address;
    uint32_t bindingIndex = 0;
    vk::raii::Buffer buffer;
    vk::raii::DeviceMemory memory;
    void* mappedData = nullptr;
    vk::DeviceSize bufferSize = 0;
    std::vector<FixedBufferAllocation> allocations;
    std::queue<uint32_t> freeSlots;
    uint32_t maxSize = MAX_FIXED_BUFFER;
    bool perFrame;
    uint32_t elementSize = 0;  // Size of a single element
    uint32_t frameOffsets[MAX_FRAMES_IN_FLIGHT] = {};  // Offset for each frame in the buffer

    virtual ~FixedBufferResourceBase() = default;
    virtual uint32_t allocateImpl(const void* data) = 0;
    virtual void updateImpl(uint32_t index, const void* data) = 0;
    virtual size_t getElementSize() const = 0;

  protected:
    FixedBufferResourceBase(vk::raii::Buffer&& buf, vk::raii::DeviceMemory&& mem) : buffer(std::move(buf)), memory(std::move(mem)) {}
};

template <typename T> struct FixedBufferResource : FixedBufferResourceBase {
    std::vector<T> data;

    FixedBufferResource(vk::raii::Buffer&& buf, vk::raii::DeviceMemory&& mem) : FixedBufferResourceBase(std::move(buf), std::move(mem)) {
        data.reserve(maxSize);
        elementSize = sizeof(T);
    }

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
            index = freeSlots.front();
            freeSlots.pop();
            allocations[index] = FixedBufferAllocation{.index = index, .inUse = true};
            data[index] = newData;
        } else {
            if (data.size() >= maxSize) {
                throw std::runtime_error("Fixed buffer capacity exceeded");
            }
            index = static_cast<uint32_t>(data.size());
            allocations.push_back(FixedBufferAllocation{.index = index, .inUse = true});
            data.push_back(newData);
        }

        T* dataBuffer = static_cast<T*>(mappedData);
        dataBuffer[index] = newData;

        return index;
    }

    void updateTyped(uint32_t index, const T& newData) {
        if (index >= allocations.size() || !allocations[index].inUse) {
            throw std::runtime_error("Invalid buffer index or allocation not in use");
        }

        data[index] = newData;
        T* dataBuffer = static_cast<T*>(mappedData);
        dataBuffer[index] = newData;
    }

    // Updates with frame offset
    void updateTypedWithOffset(uint32_t index, const T& newData, uint32_t frameIndex) {
        if (frameIndex >= MAX_FRAMES_IN_FLIGHT) {
            throw std::runtime_error("Invalid frame index");
        }
        if (index >= allocations.size() || !allocations[index].inUse) {
            throw std::runtime_error("Invalid buffer index or allocation not in use");
        }
        // Calculate offset
        uint32_t byteOffset = frameOffsets[frameIndex] + (index * static_cast<uint32_t>(sizeof(T)));
        T* targetPtr = reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(mappedData) + byteOffset);
        *targetPtr = newData;
    }

    void free(uint32_t index) {
        if (index >= allocations.size() || !allocations[index].inUse) {
            return;
        }
        allocations[index].inUse = false;
        freeSlots.push(index);
    }
};

class DescriptorSet {
  public:
    DescriptorSet(Device& device, ResourceManager& resourceManager, vk::raii::CommandPool* commandPool)
        : device(device), resourceManager(resourceManager), commandPool(commandPool) {}

    void createDescriptorSet() {
        std::vector<vk::DescriptorSetLayoutBinding> layoutBindings;
        std::vector<vk::DescriptorPoolSize> poolSizes;
        std::vector<vk::DescriptorBindingFlags> bindingFlags;
        // variableCounts removed - no binding uses VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT
        uint32_t bindingIndex = 0;

        // textures
        layoutBindings.push_back(vk::DescriptorSetLayoutBinding{.binding = bindingIndex,
                                                                .descriptorType = vk::DescriptorType::eSampledImage,
                                                                .descriptorCount = MAX_TEXTURES,
                                                                .stageFlags = vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eVertex});
        poolSizes.push_back(vk::DescriptorPoolSize(vk::DescriptorType::eSampledImage, MAX_TEXTURES));
        bindingFlags.push_back(vk::DescriptorBindingFlagBits::ePartiallyBound);
        // variableCounts.push_back(MAX_TEXTURES);
        textureBindingIndex = bindingIndex;
        bindingIndex++;

        // cubemaps
        layoutBindings.push_back(vk::DescriptorSetLayoutBinding{.binding = bindingIndex,
                                                                .descriptorType = vk::DescriptorType::eSampledImage,
                                                                .descriptorCount = MAX_CUBEMAPS,
                                                                .stageFlags = vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eVertex});
        poolSizes.push_back(vk::DescriptorPoolSize(vk::DescriptorType::eSampledImage, MAX_CUBEMAPS));
        bindingFlags.push_back(vk::DescriptorBindingFlagBits::ePartiallyBound);
        // variableCounts.push_back(MAX_CUBEMAPS);
        cubemapBindingIndex = bindingIndex;
        bindingIndex++;

        // samplers
        layoutBindings.push_back(vk::DescriptorSetLayoutBinding{.binding = bindingIndex,
                                                                .descriptorType = vk::DescriptorType::eSampler,
                                                                .descriptorCount = MAX_SAMPLERS,
                                                                .stageFlags = vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eVertex});
        poolSizes.push_back(vk::DescriptorPoolSize(vk::DescriptorType::eSampler, MAX_SAMPLERS));
        bindingFlags.push_back(vk::DescriptorBindingFlagBits::ePartiallyBound);
        // variableCounts.push_back(MAX_SAMPLERS);
        samplerBindingIndex = bindingIndex;
        bindingIndex++;

        // Dedicated single-slot comparison sampler for hardware PCF shadow sampling.
        // Kept separate from the bindless sampler array so the shader can declare it
        // as SamplerComparisonState without aliasing a SamplerState binding.
        layoutBindings.push_back(vk::DescriptorSetLayoutBinding{.binding = bindingIndex,
                                                                .descriptorType = vk::DescriptorType::eSampler,
                                                                .descriptorCount = 1,
                                                                .stageFlags = vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eVertex});
        poolSizes.push_back(vk::DescriptorPoolSize(vk::DescriptorType::eSampler, 1));
        bindingFlags.push_back({});
        shadowCompareSamplerBindingIndex = bindingIndex;
        bindingIndex++;

        // Storage buffers are now accessed via Buffer Device Address (BDA) — no descriptor bindings needed

        vk::DescriptorPoolCreateInfo poolInfo{};
        poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        descriptorPool = vk::raii::DescriptorPool(device.getDevice(), poolInfo);

        vk::DescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{.bindingCount = static_cast<uint32_t>(bindingFlags.size()), .pBindingFlags = bindingFlags.data()};
        vk::DescriptorSetLayoutCreateInfo layoutInfo{.pNext = &flagsInfo, .bindingCount = static_cast<uint32_t>(layoutBindings.size()), .pBindings = layoutBindings.data()};
        descriptorSetLayout = vk::raii::DescriptorSetLayout(device.getDevice(), layoutInfo);

        vk::DescriptorSetAllocateInfo allocInfo{.descriptorPool = *descriptorPool, .descriptorSetCount = 1, .pSetLayouts = &**descriptorSetLayout};

        vk::raii::DescriptorSets sets(device.getDevice(), allocInfo);
        descriptorSet = std::move(sets[0]);

#if DEBUG == 1
        debugDescriptorSetState("after_descriptor_set_creation");
#endif

        // Storage buffers are accessed via BDA — no descriptor writes needed
    }

    vk::DeviceAddress getBufferAddress(const vk::raii::Device& device, const vk::raii::Buffer& buffer) {
        return device.getBufferAddress(vk::BufferDeviceAddressInfo{.buffer = *buffer});
    }

    //allocates a texture on the gpu in the bindless set, returns its index
    uint32_t allocateTexture(vk::raii::Image image, vk::raii::DeviceMemory memory, vk::raii::ImageView imageView, std::string source = "", bool isCubeMap = false, uint32_t texWidth = 0, uint32_t texHeight = 0) {
#if DEBUG == 1
        debugDescriptorSetState("before_texture_allocation");
#endif

        if (textureResources.size() >= MAX_TEXTURES && freeTextureSlots.empty()) {
            throw std::runtime_error("Maximum bindless textures exceeded");
        }
        uint32_t mipLevels = (texWidth > 0 && texHeight > 0)
                                 ? static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1
                                 : 1;
        uint32_t index;
        if (!freeTextureSlots.empty()) {
            index = freeTextureSlots.front();
            freeTextureSlots.pop();
            textureResources[index].imageView = std::move(imageView);
            textureResources[index].image = std::move(image);
            textureResources[index].memory = std::move(memory);
            textureResources[index].source = source;
            textureResources[index].width = texWidth;
            textureResources[index].height = texHeight;
            textureResources[index].mipLevels = mipLevels;
        } else {
            index = textureResources.size();
            TextureResource resource;
            resource.imageView = std::move(imageView);
            resource.image = std::move(image);
            resource.memory = std::move(memory);
            resource.source = source;
            resource.width = texWidth;
            resource.height = texHeight;
            resource.mipLevels = mipLevels;
            textureResources.emplace_back(std::move(resource));
        }
        vk::DescriptorImageInfo imageInfo{.imageView = *textureResources[index].imageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
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

    void updateTexture(uint32_t index, vk::raii::Image image, vk::raii::DeviceMemory memory, vk::raii::ImageView imageView, std::string source = "", bool isCubeMap = false, uint32_t texWidth = 0, uint32_t texHeight = 0) {
        textureResources[index].imageView = std::move(imageView);
        textureResources[index].image = std::move(image);
        textureResources[index].memory = std::move(memory);
        textureResources[index].source = source;
        textureResources[index].width = texWidth;
        textureResources[index].height = texHeight;
        textureResources[index].mipLevels = (texWidth > 0 && texHeight > 0)
                                                ? static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1
                                                : 1;

        vk::DescriptorImageInfo imageInfo{.imageView = *textureResources[index].imageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
        vk::WriteDescriptorSet write{.dstSet = *descriptorSet,
                                     .dstBinding = isCubeMap ? cubemapBindingIndex : textureBindingIndex,
                                     .dstArrayElement = index,
                                     .descriptorCount = 1,
                                     .descriptorType = vk::DescriptorType::eSampledImage,
                                     .pImageInfo = &imageInfo};
        device.getDevice().updateDescriptorSets(write, {});
    }

    uint32_t getTextureMipLevels(uint32_t index) const { return textureResources[index].mipLevels; }

    //ensure no in-flight command buffers are using this texture when calling this
    void freeTexture(uint32_t index) {
        device.getDevice().waitIdle();
        textureResources[index].reset();
        freeTextureSlots.push(index);
    }

    uint32_t allocateSampler(vk::Filter filter, vk::SamplerMipmapMode mipmapMode, vk::SamplerAddressMode addressMode, vk::Bool32 anisotropyEnabled, float maxAnisotropy,
                             vk::Bool32 compareEnable, vk::CompareOp compareOp, vk::BorderColor borderColor) {
        vk::SamplerCreateInfo createInfo{.pNext = nullptr,
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
                                         .minLod = 0.0,
                                         .maxLod = 16.0,
                                         .borderColor = borderColor};
        vk::raii::Sampler sampler(device.getDevice(), createInfo);
        if (samplerResources.size() >= MAX_SAMPLERS && freeSamplerSlots.empty()) {
            throw std::runtime_error("Maximum samplers exceeded");
        }
        uint32_t index;
        vk::Sampler samplerHandle = *sampler;

        if (!freeSamplerSlots.empty()) {
            index = freeSamplerSlots.front();
            freeSamplerSlots.pop();
            samplerResources[index].sampler = std::move(sampler);
        } else {
            index = samplerResources.size();
            SamplerResource resource;
            resource.sampler = std::move(sampler);
            samplerResources.push_back(std::move(resource));
        }
        vk::DescriptorImageInfo imageInfo{.sampler = samplerHandle};
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

    //TODO
    void freeSampler(uint32_t index) {  }

    // Allocates the dedicated shadow comparison sampler at its own descriptor binding.
    // Only one such sampler exists; re-calling this replaces the previous one.
    void allocateShadowCompareSampler(vk::Filter filter, vk::SamplerAddressMode addressMode, vk::CompareOp compareOp, vk::BorderColor borderColor) {
        vk::SamplerCreateInfo createInfo{.pNext = nullptr,
                                         .flags = {},
                                         .magFilter = filter,
                                         .minFilter = filter,
                                         .mipmapMode = vk::SamplerMipmapMode::eNearest,
                                         .addressModeU = addressMode,
                                         .addressModeV = addressMode,
                                         .addressModeW = addressMode,
                                         .anisotropyEnable = VK_FALSE,
                                         .maxAnisotropy = 1.0f,
                                         .compareEnable = VK_TRUE,
                                         .compareOp = compareOp,
                                         .minLod = 0.0,
                                         .maxLod = 0.0,
                                         .borderColor = borderColor};
        shadowCompareSampler = vk::raii::Sampler(device.getDevice(), createInfo);

        vk::DescriptorImageInfo imageInfo{.sampler = **shadowCompareSampler};
        vk::WriteDescriptorSet write{.sType = vk::StructureType::eWriteDescriptorSet,
                                     .dstSet = *descriptorSet,
                                     .dstBinding = shadowCompareSamplerBindingIndex,
                                     .dstArrayElement = 0,
                                     .descriptorCount = 1,
                                     .descriptorType = vk::DescriptorType::eSampler,
                                     .pImageInfo = &imageInfo};
        device.getDevice().updateDescriptorSets(write, {});
    }

    template <typename T> uint32_t createFixedBuffer(uint32_t maxElements = MAX_FIXED_BUFFER, bool usesDynamicOffset = false) {
        vk::DeviceSize bufferSize = maxElements * sizeof(T);

        vk::BufferCreateInfo bufferInfo{.size = bufferSize, .usage = vk::BufferUsageFlagBits::eStorageBuffer, .sharingMode = vk::SharingMode::eExclusive};

        vk::raii::Buffer buffer(device.getDevice(), bufferInfo);

        vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
        uint32_t memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, device);

        vk::MemoryAllocateFlagsInfo allocFlags{ .flags = vk::MemoryAllocateFlagBits::eDeviceAddress };
        vk::MemoryAllocateInfo allocInfo{.pNext = allocFlags, .allocationSize = memRequirements.size, .memoryTypeIndex = memoryTypeIndex};

        vk::raii::DeviceMemory memory(device.getDevice(), allocInfo);
        buffer.bindMemory(*memory, 0);
        void* mappedData = memory.mapMemory(0, bufferSize);

        auto fixedBuffer = std::make_unique<FixedBufferResource<T>>(std::move(buffer), std::move(memory));
        fixedBuffer->address = getBufferAddress(device.getDevice(), fixedBuffer->buffer);
        fixedBuffer->mappedData = mappedData;
        fixedBuffer->bufferSize = bufferSize;
        fixedBuffer->maxSize = maxElements;
        fixedBuffer->perFrame = usesDynamicOffset;

        uint32_t bufferIndex = static_cast<uint32_t>(fixedBuffers.size());
        fixedBuffers.push_back(std::move(fixedBuffer));

        return bufferIndex;
    }

    template <typename T> uint32_t allocateFixedBuffer(uint32_t bufferIndex, const T& data) {
        if (bufferIndex >= fixedBuffers.size()) {
            throw std::runtime_error("Invalid buffer index");
        }
        uint32_t index = fixedBuffers[bufferIndex]->allocateImpl(&data);

        // If this buffer uses dynamic offsets (perFrame), write to all frame sections
        auto* basePtr = fixedBuffers[bufferIndex].get();
        if (basePtr->perFrame) {
            auto* typedPtr = static_cast<FixedBufferResource<T>*>(basePtr);
            for (uint32_t frame = 1; frame < MAX_FRAMES_IN_FLIGHT; frame++) {
                typedPtr->updateTypedWithOffset(index, data, frame);
            }
        }

        return index;
    }

    void freeFixedBuffer(uint32_t bufferIndex, uint32_t index) {
        if (bufferIndex >= fixedBuffers.size()) {
            return;
        }
        auto* basePtr = fixedBuffers[bufferIndex].get();
        if (index < basePtr->allocations.size() && basePtr->allocations[index].inUse) {
            basePtr->allocations[index].inUse = false;
            basePtr->freeSlots.push(index);
        }
    }

    void clearFixedBuffer(uint32_t bufferIndex) {
        if (bufferIndex >= fixedBuffers.size()) {
            return;
        }
        auto* basePtr = fixedBuffers[bufferIndex].get();
        for (auto& allocation : basePtr->allocations) {
            allocation.inUse = false;
        }
        basePtr->freeSlots = {};
        for (uint32_t i = 0; i < basePtr->allocations.size(); i++) {
            basePtr->freeSlots.push(i);
        }
    }

    template <typename T> void updateFixedBuffer(uint32_t bufferIndex, uint32_t index, const T& data) {
        if (bufferIndex >= fixedBuffers.size()) {
            throw std::runtime_error("Invalid buffer index");
        }
        fixedBuffers[bufferIndex]->updateImpl(index, &data);
    }

    template <typename T> void updateFixedBufferWithOffset(uint32_t bufferIndex, uint32_t index, const T& data, uint32_t frameIndex) {
        if (bufferIndex >= fixedBuffers.size()) {
            throw std::runtime_error("Invalid buffer index");
        }
        auto* basePtr = fixedBuffers[bufferIndex].get();
        auto* typedPtr = static_cast<FixedBufferResource<T>*>(basePtr);
        typedPtr->updateTypedWithOffset(index, data, frameIndex);
    }

    void setBufferFrameOffset(uint32_t bufferIndex, uint32_t frameIndex, uint32_t offsetInElements) {
        if (bufferIndex >= fixedBuffers.size()) {
            throw std::runtime_error("Invalid buffer index");
        }
        auto* basePtr = fixedBuffers[bufferIndex].get();
        if (frameIndex >= MAX_FRAMES_IN_FLIGHT) {
            throw std::runtime_error("Invalid frame index");
        }
        // converts element offset to byte offset
        basePtr->frameOffsets[frameIndex] = offsetInElements * basePtr->elementSize;
    }

    uint32_t createVariableBuffer(uint32_t maxSizeBytes = 1024 * 1024, vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress, bool perFrame = false) {
        vk::BufferCreateInfo bufferInfo{.size = maxSizeBytes, .usage = usage, .sharingMode = vk::SharingMode::eExclusive};

        vk::raii::Buffer buffer(device.getDevice(), bufferInfo);
        vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
        uint32_t memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, device);

        vk::MemoryAllocateFlagsInfo allocFlags{.flags = vk::MemoryAllocateFlagBits::eDeviceAddress};
        vk::MemoryAllocateInfo allocInfo{.pNext = allocFlags, .allocationSize = memRequirements.size, .memoryTypeIndex = memoryTypeIndex};

        vk::raii::DeviceMemory memory(device.getDevice(), allocInfo);
        buffer.bindMemory(*memory, 0);
        void* mappedData = memory.mapMemory(0, maxSizeBytes);

        variableBuffers.push_back(VariableBufferResource{getBufferAddress(device.getDevice(),buffer),std::move(buffer), std::move(memory), mappedData, maxSizeBytes, maxSizeBytes});
        return variableBuffers.size() - 1;
    }

    template <typename T> uint32_t allocateVariableBuffer(const std::vector<T>& data, uint32_t bufferIndex = 0) {
        if (bufferIndex >= variableBuffers.size() || !variableBuffers[bufferIndex]) {
            throw std::runtime_error("Invalid variable buffer index");
        }

        auto& buffer = variableBuffers[bufferIndex].value();
        uint32_t elementSize = sizeof(T);
        uint32_t totalSize = static_cast<uint32_t>(data.size()) * elementSize;
        uint32_t count = static_cast<uint32_t>(data.size());

        uint32_t offset = findVariableBufferOffset(bufferIndex, totalSize);
        if (offset + totalSize > buffer.bufferSize) {
            throw std::runtime_error("Variable buffer out of space");
        }

        T* gpuData = reinterpret_cast<T*>(static_cast<char*>(buffer.mappedData) + offset);
        std::memcpy(gpuData, data.data(), totalSize);

        uint32_t allocationIndex = static_cast<uint32_t>(buffer.allocations.size());
        buffer.allocations.push_back(VariableBufferAllocation{.offset = offset, .size = totalSize, .stride = elementSize, .count = count});
        return allocationIndex;
    }

    template <typename T> void updateVariableBuffer(uint32_t bufferIndex, uint32_t allocationIndex, const std::vector<T>& newData) {
        if (bufferIndex >= variableBuffers.size() || !variableBuffers[bufferIndex] || allocationIndex >= variableBuffers[bufferIndex]->allocations.size()) {
            throw std::runtime_error("Invalid variable buffer allocation");
        }
        auto& buffer = variableBuffers[bufferIndex].value();
        auto& allocation = buffer.allocations[allocationIndex];
        uint32_t elementSize = sizeof(T);
        uint32_t newTotalSize = static_cast<uint32_t>(newData.size()) * elementSize;

        if (newTotalSize > allocation.size) {
            throw std::runtime_error("New data too large for existing allocation. Consider reallocating.");
        }

        allocation.count = static_cast<uint32_t>(newData.size());
        T* gpuData = reinterpret_cast<T*>(static_cast<char*>(buffer.mappedData) + allocation.offset);
        std::memcpy(gpuData, newData.data(), newTotalSize);
    }

    void freeVariableBuffer(uint32_t bufferIndex, uint32_t allocationIndex) {
        if (bufferIndex >= variableBuffers.size() || !variableBuffers[bufferIndex] || allocationIndex >= variableBuffers[bufferIndex]->allocations.size()) {
            return;
        }
        variableBuffers[bufferIndex]->allocations.at(allocationIndex).size = 0;
    }

    VariableBufferAllocation getVariableBufferAllocation(uint32_t bufferIndex, uint32_t allocationIndex) const {
        if (bufferIndex >= variableBuffers.size() || !variableBuffers[bufferIndex]) {
            throw std::runtime_error("Invalid variable buffer index");
        }
        return variableBuffers[bufferIndex]->allocations.at(allocationIndex);
    }
    vk::Buffer getVariableBuffer(uint32_t bufferIndex) const {
        if (bufferIndex >= variableBuffers.size() || !variableBuffers[bufferIndex]) {
            throw std::runtime_error("Invalid variable buffer index");
        }
        return *variableBuffers[bufferIndex]->buffer;
    }

    std::vector<FixedBufferAllocation>& getFixedBufferAllocations(uint32_t bufferIndex) const {
        if (bufferIndex >= fixedBuffers.size() || !fixedBuffers[bufferIndex]) {
            throw std::runtime_error("Invalid fixed buffer index");
        }
        return fixedBuffers[bufferIndex]->allocations;
    }

    template <typename T> T* getFixedBufferMappedData(uint32_t bufferIndex) {
        if (bufferIndex >= fixedBuffers.size() || !fixedBuffers[bufferIndex]) {
            return nullptr;
        }
        return static_cast<T*>(fixedBuffers[bufferIndex]->mappedData);
    }

    vk::raii::DescriptorSet& getDescriptorSet() { return *descriptorSet; }
    vk::raii::DescriptorSetLayout& getDescriptorSetLayout() { return *descriptorSetLayout; }
    TextureResource& getTextureResource(uint32_t index) { return textureResources[index]; } // TODO bounds check this
    const std::vector<TextureResource>& getTextureResources() { return textureResources; }
    const std::vector<SamplerResource>& getSamplerResources() { return samplerResources; }
    void debugDescriptorSet(const std::string& context) { debugDescriptorSetState(context); }

    const std::vector<std::optional<VariableBufferResource>>& getVariableBuffers() { return variableBuffers; }
    const std::vector<std::unique_ptr<FixedBufferResourceBase>>& getFixedBuffers() { return fixedBuffers; }

  private:
    Device& device;
    ResourceManager& resourceManager;
    vk::raii::CommandPool* commandPool;

    std::optional<vk::raii::DescriptorSetLayout> descriptorSetLayout;
    std::optional<vk::raii::DescriptorPool> descriptorPool;
    std::optional<vk::raii::DescriptorSet> descriptorSet;

    std::vector<SamplerResource> samplerResources;
    uint32_t samplerBindingIndex;
    std::queue<uint32_t> freeSamplerSlots;
    uint32_t shadowCompareSamplerBindingIndex;
    std::optional<vk::raii::Sampler> shadowCompareSampler;
    std::vector<TextureResource> textureResources;
    uint32_t textureBindingIndex;
    uint32_t cubemapBindingIndex;
    std::queue<uint32_t> freeTextureSlots;
    std::vector<std::optional<VariableBufferResource>> variableBuffers;
    std::vector<std::unique_ptr<FixedBufferResourceBase>> fixedBuffers;

    uint32_t findVariableBufferOffset(uint32_t bufferIndex, uint32_t requestedSize) {
        if (bufferIndex >= variableBuffers.size() || !variableBuffers[bufferIndex]) {
            throw std::runtime_error("Invalid variable buffer index");
        }

        auto& buffer = variableBuffers[bufferIndex].value();
        if (buffer.allocations.empty()) {
            return 0;
        }

        std::vector<std::pair<uint32_t, uint32_t>> usedRanges;
        for (const auto& alloc : buffer.allocations) {
            if (alloc.size > 0) {
                usedRanges.push_back({alloc.offset, alloc.size});
            }
        }

        std::sort(usedRanges.begin(), usedRanges.end());

        uint32_t currentOffset = 0;
        for (const auto& range : usedRanges) {
            if (range.first - currentOffset >= requestedSize) {
                return currentOffset;
            }
            currentOffset = range.first + range.second;
        }

        return currentOffset;
    }

    void debugDescriptorSetState(const std::string& context) {
        return;
        std::cout << "=== Descriptor Set Debug [" << context << "] ===" << std::endl;
        std::cout << "descriptorSet.has_value(): " << descriptorSet.has_value() << std::endl;
        if (descriptorSet.has_value()) {
            VkDescriptorSet handle = **descriptorSet;
            std::cout << "Raw VK handle: " << std::hex << (uintptr_t)handle << std::dec << std::endl;
            std::cout << "Handle is null: " << (handle == VK_NULL_HANDLE ? "YES" : "NO") << std::endl;
        }
        std::cout << "===========================================" << std::endl;
    }
};