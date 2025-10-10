#pragma once
#ifndef VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1 // for raii
#endif
#ifndef VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_NO_CONSTRUCTORS 1 // for structs constructors
#endif

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <stb_image.h>
#include <stdexcept>
#include <unordered_set>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#define GLFW_INCLUDE_VULKAN // REQUIRED only for GLFW CreateWindowSurface.
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#define GLM_FORCE_RADIANS
#define GLM_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#include "constants.hpp"
#include "descriptor_sets.hpp"
#include "devices.hpp"
#include "gizmo.hpp"
#include "pipelines.hpp"
#include "scene_elements.hpp"
#include "swapchain.hpp"
#include "utils.hpp"
// TODO gpu side material data ("uber shader" approach)
const std::vector validationLayers = {"VK_LAYER_KHRONOS_validation"};

class Renderer {
  public:
    void showShadowMap() {
        debugShowShadow++;
        if (debugShowShadow == 4) {
            showShadowMapIndex = 0xFFFFFFFF;
            return;
        }
        if (debugShowShadow >= 5) {
            debugShowShadow = 0;
            return;
        }
        
    }
    uint32_t debugShowShadow = 4;
    uint32_t showShadowMapIndex = 0xFFFFFFFF;
    Camera activeCamera;
    Gizmos* gizmos = nullptr;
    uint32_t selectedNode = MAX_NODES;

    Renderer() : nodes(new std::array<std::optional<Node>, MAX_NODES>()) {}
    ~Renderer() { delete nodes; }

    void initVulkan(uint32_t startWidth, uint32_t startHeight) {
        createInstance();
        setupDebugMessenger();
        createSurface();
        device = std::make_unique<Device>(instance, requiredDeviceExtension, surface);
        msaaSamples = getMaxUsableSampleCount(*device);
        createCommandPool();
        createCommandBuffers();
        resourceManager = std::make_unique<ResourceManager>(*device, commandPool);
        descriptorSet = std::make_unique<DescriptorSet>(*device, *resourceManager, &commandPool);
        swapchain = std::make_unique<Swapchain>(*device, *resourceManager, *descriptorSet, surface, msaaSamples);
        pipelineManager = std::make_unique<PipelineManager>(*device, *swapchain, msaaSamples);

        // initializing camera
        activeCamera = Camera{.position = glm::vec3(1, 1, 1),
                              .target = glm::vec3(0, 0, 0),
                              .fov = 45.0,
                              .aspectRatio = static_cast<float>(startWidth) / static_cast<float>(startHeight),
                              .nearPlane = 0.1,
                              .farPlane = 100.0};
        activeCamera.calculateViewProjectionMatrix();

        vertexBufferIndex = descriptorSet->createVariableBuffer(256 * 1024 * 1024);                                       // 256 mb vertex buffer
        indexBufferIndex = descriptorSet->createVariableBuffer(128 * 1024 * 1024, vk::BufferUsageFlagBits::eIndexBuffer); // index buffer (128 MB)
        modelMatrixBufferIndex = descriptorSet->createFixedBuffer<glm::mat4>();                                           // max 2048 model matrices by default
        lightBufferIndex = descriptorSet->createFixedBuffer<Light>();                                                     // max 2048 lights by default
        gizmos = new Gizmos(MAX_GIZMO_LINES, &*descriptorSet);
        descriptorSet->createDescriptorSet();
        swapchain->create(*window, vSync);
        createSyncObjects();

#if DEBUG == 1
        descriptorSet->debugDescriptorSet("after_createDescriptorSet");
#endif

        // DEFAULTS
        defaultSamplerIndex = descriptorSet->allocateSampler(vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear, vk::SamplerAddressMode::eRepeat, VK_TRUE, 16.0, VK_FALSE,
                                                             vk::CompareOp::eLessOrEqual, vk::BorderColor::eFloatOpaqueBlack);
        depthSamplerIndex = descriptorSet->allocateSampler(vk::Filter::eNearest, vk::SamplerMipmapMode::eNearest, vk::SamplerAddressMode::eClampToEdge, VK_FALSE, 16.0, VK_FALSE,
                                                           vk::CompareOp::eLessOrEqual, vk::BorderColor::eFloatOpaqueBlack);
        shadowSamplerIndex = descriptorSet->allocateSampler(vk::Filter::eLinear,                    // Bilinear filtering for PCF
                                                            vk::SamplerMipmapMode::eNearest,        // No mipmaps
                                                            vk::SamplerAddressMode::eClampToBorder, // Clamp to avoid wrapping
                                                            VK_FALSE,                               // No anisotropy needed
                                                            1.0f,
                                                            VK_FALSE, // No comparison (or VK_TRUE for hardware PCF)
                                                            vk::CompareOp::eLessOrEqual,
                                                            vk::BorderColor::eFloatOpaqueWhite // 1.0 = max depth = not in shadow
        );
        // Default albedo (white)
        std::array<uint8_t, 4> whiteColor = {255, 255, 255, 255};
        auto [albedoImage, albedoMemory, albedoImageView] = resourceManager->createTexture(whiteColor.data(), 1, 1, vk::Format::eR8G8B8A8Srgb);
        uint32_t defaultAlbedoIndex = descriptorSet->allocateTexture(std::move(albedoImage), std::move(albedoMemory), std::move(albedoImageView));

        // Default normal (flat normal = 0.5, 0.5, 1.0 in RGB)
        std::array<uint8_t, 4> normalColor = {128, 128, 255, 255}; // This is (0.5, 0.5, 1.0, 1.0) in normalized values
        auto [normalImage, normalMemory, normalImageView] = resourceManager->createTexture(normalColor.data(), 1, 1, vk::Format::eR8G8B8A8Unorm);
        defaultNormalIndex = descriptorSet->allocateTexture(std::move(normalImage), std::move(normalMemory), std::move(normalImageView));

        // Default roughness (mid-gray = 0.5 roughness)
        std::array<uint8_t, 4> roughnessColor = {128, 128, 128, 255};
        auto [roughnessImage, roughnessMemory, roughnessImageView] = resourceManager->createTexture(roughnessColor.data(), 1, 1, vk::Format::eR8G8B8A8Unorm);
        uint32_t defaultRoughnessIndex = descriptorSet->allocateTexture(std::move(roughnessImage), std::move(roughnessMemory), std::move(roughnessImageView));

        // Default metallic (black = 0.0 metallic)
        std::array<uint8_t, 4> metallicColor = {0, 0, 0, 255};
        auto [metallicImage, metallicMemory, metallicImageView] = resourceManager->createTexture(metallicColor.data(), 1, 1, vk::Format::eR8G8B8A8Unorm);
        uint32_t defaultMetallicIndex = descriptorSet->allocateTexture(std::move(metallicImage), std::move(metallicMemory), std::move(metallicImageView));

        // pipeline(s)
        skyboxPipelineIndex =
            pipelineManager->createPipeline<SkyBoxPushConstants>(PipelineCategory::GEOMETRY, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone, vk::False,
                                                                 vk::False, "shaders/skybox.spv", descriptorSet->getDescriptorSetLayout(), descriptorSet->getDescriptorSet());

        shadowPipelineIndex = pipelineManager->createPipeline<ShadowPushConstants>(PipelineCategory::BEFORE_GEOMETRY, vk::PrimitiveTopology::eTriangleList,
                                                                                   vk::CullModeFlagBits::eNone, vk::True, vk::True, "shaders/shadow_geometry.spv",
                                                                                   descriptorSet->getDescriptorSetLayout(), descriptorSet->getDescriptorSet());

        litPipelineIndex = pipelineManager->createPipeline<PushConstants>(PipelineCategory::GEOMETRY, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eBack, vk::True,
                                                                          vk::True, "shaders/lit.spv", descriptorSet->getDescriptorSetLayout(), descriptorSet->getDescriptorSet());
        gizmoPipelineIndex =
            pipelineManager->createPipeline<LinePushConstants>(PipelineCategory::GEOMETRY, vk::PrimitiveTopology::eLineList, vk::CullModeFlagBits::eNone, vk::False, vk::False,
                                                               "shaders/line.spv", descriptorSet->getDescriptorSetLayout(), descriptorSet->getDescriptorSet());
        depthPipelineIndex =
            pipelineManager->createPipeline<DepthVisPushConstants>(PipelineCategory::AFTER_GEOMETRY, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone, vk::False,
                                                                   vk::False, "shaders/depth_view.spv", descriptorSet->getDescriptorSetLayout(), descriptorSet->getDescriptorSet());

        fallbackLitShader = Shader{.sourceFile = "shaders/lit.spv", .pipelineIndex = litPipelineIndex};
        uint32_t defaultTexMask = 0x000000000;
        //texMask |= (1U << 0);
        //texMask |= (1U << 1);
        //texMask |= (1U << 3);
        Material defaultMaterial = Material{.shaderSource = fallbackLitShader,
                                            .textureMask = defaultTexMask,
                                            .color = glm::vec4(0.5, 0.5, 0.5, 1),
                                            .albedoTextureIndex = defaultAlbedoIndex,
                                            .metallic = 0.0,
                                            .metallicTextureIndex = defaultMetallicIndex,
                                            .roughness = 0.5,
                                            .roughnessTextureIndex = defaultRoughnessIndex,
                                            .normalTextureIndex = defaultNormalIndex};
        fallbackDefaultMaterialIndex = addMaterial(defaultMaterial);

#if DEBUG == 1
        descriptorSet->debugDescriptorSet("after_pipeline_creation");
#endif

        (*nodes)[0] = Node(this, 0, nullptr, glm::vec3(0.0), glm::quat(1.0, 0, 0, 0), glm::vec3(1, 1, 1));
        (*nodes)[0]->name = "root";
        rootNode = &*(*nodes)[0];
        std::cout << sizeof(Vertex) << std::endl;
    }

    void drawFrame() {
        for (auto& [id, light] : lights) {
            if (light.castsShadows == 1) {
                if (light.type == LightType::Directional) {
                    calculateCascadedLightSpaceMatrices(light, activeCamera);
                }
                descriptorSet->updateFixedBuffer<Light>(lightBufferIndex, id, light);
            }
        }

        device->getDevice().waitForFences(*inFlightFences[currentFrame], vk::True, UINT64_MAX);
        auto [result, imageIndex] = swapchain->getSwapChain().acquireNextImage(UINT64_MAX, *presentCompleteSemaphores[currentFrame], nullptr);

        if (result == vk::Result::eErrorOutOfDateKHR) {
            swapchain->recreate(window, vSync);
            int width = 0, height = 0;
            glfwGetFramebufferSize(window, &width, &height);
            activeCamera.aspectRatio = static_cast<float>(width) / static_cast<float>(height);
            activeCamera.calculateViewProjectionMatrix();
            return;
        }
        if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }
        if (imagesInFlight[imageIndex] != VK_NULL_HANDLE) {
            vk::Result waitResult = device->getDevice().waitForFences(imagesInFlight[imageIndex], vk::True, UINT64_MAX);
            if (waitResult != vk::Result::eSuccess) {
                throw std::runtime_error("failed to wait for image fence!");
            }
        }

        imagesInFlight[imageIndex] = *inFlightFences[currentFrame];
        device->getDevice().resetFences(*inFlightFences[currentFrame]);

        commandBuffers[currentFrame].reset();
        recordCommandBuffer(imageIndex);

        vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
        const vk::SubmitInfo submitInfo{.waitSemaphoreCount = 1,
                                        .pWaitSemaphores = &*presentCompleteSemaphores[currentFrame],
                                        .pWaitDstStageMask = &waitDestinationStageMask,
                                        .commandBufferCount = 1,
                                        .pCommandBuffers = &*commandBuffers[currentFrame],
                                        .signalSemaphoreCount = 1,
                                        .pSignalSemaphores = &*renderFinishedSemaphores[imageIndex]};

        device->getGraphicsQueue().submit(submitInfo, inFlightFences[currentFrame]);

        const vk::PresentInfoKHR presentInfoKHR{.waitSemaphoreCount = 1,
                                                .pWaitSemaphores = &*renderFinishedSemaphores[imageIndex],
                                                .swapchainCount = 1,
                                                .pSwapchains = &*swapchain->getSwapChain(),
                                                .pImageIndices = &imageIndex};

        try {
            result = device->getPresentQueue().presentKHR(presentInfoKHR);
        } catch (const vk::OutOfDateKHRError&) {
            result = vk::Result::eErrorOutOfDateKHR;
        }

        if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || framebufferResized) {
            framebufferResized = false;
            swapchain->recreate(window, vSync);

            int width = 0, height = 0;
            glfwGetFramebufferSize(window, &width, &height);
            activeCamera.aspectRatio = static_cast<float>(width) / static_cast<float>(height);
            activeCamera.calculateViewProjectionMatrix();

        } else if (result != vk::Result::eSuccess) {
            throw std::runtime_error("failed to present swap chain image!");
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
        pipelineManager->checkForShaderUpdates(); //TODO enable this
    }

    GLFWwindow* getWindow() { return window; }
    void setWindow(GLFWwindow* pWindow) { window = pWindow; }

    Device& getDevice() { return *device; }
    ResourceManager& getResourceManager() { return *resourceManager; }

    const vk::Instance& getInstance() const { return *instance; }

    Swapchain& getSwapchain() { return *swapchain; }
    void cleanupSwapchain() { swapchain->cleanupSwapChain(); }

    const vk::SampleCountFlagBits& getMsaaSamples() const { return msaaSamples; }

    const uint32_t getGraphicsIndex() const { return graphicsIndex; }

    DescriptorSet& getDescriptorSet() { return *descriptorSet; }
    uint32_t getModelMatrixBufferIndex() { return modelMatrixBufferIndex; }
    uint32_t getLightBufferIndex() { return lightBufferIndex; }
    std::vector<Material>& getMaterials() { return materials; }
    uint32_t addMaterial(Material material) {
        // iterate through materials check if already exists
        for (uint32_t i = 0; i < materials.size(); i++) {
            if (materials[i] == material) {
                return i;
            }
        }
        materials.push_back(material);
        return materials.size() - 1;
    }
    // don't call this directly, should only be called from a node with a valid mesh index
    void addMeshToShader(Node* node, uint32_t submeshIndex, Shader shader, Material material) { shaders[shader][material][node].insert(submeshIndex); }
    void removeMeshFromShader(Node* node, uint32_t subMeshIndex, Shader shader, Material material) { shaders[shader][material][node].erase(subMeshIndex); }
    Shader getFallBackShader() { return fallbackLitShader; }
    uint32_t getFallBackMaterial() { return fallbackDefaultMaterialIndex; }
    std::vector<Mesh>& getMeshes() { return meshes; }
    uint32_t loadMeshFromFile(std::string filePath) {

        // if the mesh already exists in memory
        for (int i = 0; i < meshes.size(); i++) {
            if (meshes[i].sourceFile == filePath) {
                return i;
            }
        }
#if DEBUG == 1
        std::cout << "Loading mesh from " << filePath << std::endl;
#endif
        auto meshData = resourceManager->loadMeshFromFile(filePath);
        Mesh mainMesh{.sourceFile = filePath};

        for (size_t i = 0; i < meshData.subMeshes.size(); i++) {
            auto& vertices = meshData.subMeshes[i];
            auto& indices = meshData.subMeshIndices[i];

            // Allocate vertex buffer
            uint32_t vertexAllocIndex = descriptorSet->allocateVariableBuffer<Vertex>(vertices, vertexBufferIndex);
            VariableBufferAllocation vertexAlloc = descriptorSet->getVariableBufferAllocation(vertexBufferIndex, vertexAllocIndex);

            // Allocate index buffer
            uint32_t indexAllocIndex = descriptorSet->allocateVariableBuffer<uint32_t>(indices, indexBufferIndex);
            VariableBufferAllocation indexAlloc = descriptorSet->getVariableBufferAllocation(indexBufferIndex, indexAllocIndex);

            SubMesh subMesh = {.vertexAllocationIndex = vertexAllocIndex,
                               .vertexOffset = vertexAlloc.offset,
                               .vertexCount = vertexAlloc.count,
                               .vertexStride = vertexAlloc.stride,
                               .indexAllocationIndex = indexAllocIndex,
                               .indexOffset = indexAlloc.offset,
                               .indexCount = indexAlloc.count};

            subMeshes.push_back(subMesh);
            mainMesh.subMeshes.push_back(subMeshes.size() - 1);
        }

        meshes.push_back(mainMesh);
        return meshes.size() - 1;
    }

    uint32_t loadTextureFromFile(std::string filePath, vk::Format format = vk::Format::eR8G8B8A8Srgb) {
        auto [image, memory, view] = resourceManager->loadTextureFromFile(filePath, format);
        return descriptorSet->allocateTexture(std::move(image), std::move(memory), std::move(view));
    }

    uint32_t loadCubemapFromFile(std::string posX, std::string posY, std::string posZ, std::string negX, std::string negY, std::string negZ, uint32_t width = 2048, uint32_t height = 2048) {
        auto [image, memory, view] = resourceManager->loadCubeMapFromFile(posX, negX, posY, negY, posZ, negZ, width, height);
        return descriptorSet->allocateTexture(std::move(image), std::move(memory), std::move(view), true);
    }

    std::map<uint32_t, Light>& getLights() { return lights; }

    Node* getRootNode() { return rootNode; }
    std::array<std::optional<Node>, MAX_NODES>& getNodes() { return *nodes; }
    uint32_t addNode(uint32_t parentIndex = 0, glm::vec3 position = glm::vec3(0.0f), glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f), glm::vec3 scale = glm::vec3(1.0f),
                     bool keepWorldTransform = false) {
        (*nodes)[lastNode + 1].emplace(this, lastNode + 1, &*(*nodes)[parentIndex], position, rotation, scale, keepWorldTransform);
        lastNode++;
        return lastNode;
    }
    void removeNode(uint32_t index) { throw std::runtime_error("remove node not implemented!"); }
    std::vector<uint32_t> rayCastNodes(glm::vec3 origin, glm::vec3 direction) {
        float margin = 0.05f;
        glm::vec3 dir = glm::normalize(direction);
        std::vector<uint32_t> foundNodes;

        for (int i = 1; i <= lastNode; i++) {
            glm::vec3 nodeWorldLoc = glm::vec3((*nodes)[i]->getWorldPosition());
            glm::vec3 toNode = nodeWorldLoc - origin;
            // Project toNode onto the ray direction
            float projectionLength = glm::dot(toNode, dir);
            // Skip nodes behind the ray origin
            if (projectionLength < 0)
                continue;
            // Find closest point on ray to the node
            glm::vec3 closestPointOnRay = origin + dir * projectionLength;
            // Calculate perpendicular distance from node to ray
            float distanceToRay = glm::distance(nodeWorldLoc, closestPointOnRay);
            if (distanceToRay < margin) {
                foundNodes.push_back(i);
            }
        }
        return foundNodes;
    }
    void selectNode(uint32_t nodeIndex) {
        if (nodeIndex <= lastNode) {
            selectedNode = nodeIndex;
        }
    }
    void deSelectNode() { selectedNode = MAX_NODES; }
    uint32_t getNodeCount() { return lastNode + 1; }

    void toggleVsync() {
        vSync = !vSync;
        swapchain->recreate(window, vSync);
    }
    void toggleDepthView() { depthView = !depthView; }
    void setSkyBox(uint32_t skyboxIndex) { this->skyboxIndex = skyboxIndex; }

  private:
    GLFWwindow* window = nullptr;
    bool framebufferResized = true;
    vk::raii::Instance instance = nullptr;
    vk::raii::Context context;
    vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
    vk::raii::SurfaceKHR surface = nullptr;

    std::vector<const char*> requiredDeviceExtension = {VK_KHR_SWAPCHAIN_EXTENSION_NAME,           VK_KHR_SPIRV_1_4_EXTENSION_NAME,
                                                        VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,   VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME,
                                                        VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME};

    vk::raii::CommandPool commandPool = nullptr;
    std::vector<vk::raii::CommandBuffer> commandBuffers;
    uint32_t graphicsIndex = 0;

    std::vector<vk::raii::Semaphore> presentCompleteSemaphores;
    std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
    std::vector<vk::raii::Fence> inFlightFences;
    std::vector<vk::Fence> imagesInFlight;
    uint32_t currentFrame = 0;

    std::unique_ptr<Device> device;
    std::unique_ptr<Swapchain> swapchain;
    std::unique_ptr<PipelineManager> pipelineManager;
    std::unique_ptr<ResourceManager> resourceManager;
    std::unique_ptr<DescriptorSet> descriptorSet;

    std::vector<Material> materials;
    std::map<Shader, std::map<Material, std::map<Node*, std::unordered_set<uint32_t>>>> shaders; // map between Shaders and Nodes + their submeshes to render
    Shader fallbackLitShader;
    uint32_t fallbackDefaultMaterialIndex;
    std::vector<Mesh> meshes;
    std::queue<uint32_t> freeMeshes;
    std::vector<SubMesh> subMeshes;
    std::queue<uint32_t> freeSubMeshes;
    std::map<uint32_t, Light> lights;
    uint32_t vertexBufferIndex;
    uint32_t indexBufferIndex;
    uint32_t modelMatrixBufferIndex;
    uint32_t lightBufferIndex;

    uint32_t skyboxPipelineIndex;
    uint32_t shadowPipelineIndex;
    uint32_t litPipelineIndex;
    uint32_t gizmoPipelineIndex;
    uint32_t depthPipelineIndex;

    uint32_t defaultSamplerIndex;
    uint32_t depthSamplerIndex;
    uint32_t shadowSamplerIndex;
    uint32_t defaultNormalIndex;
    uint32_t skyboxIndex;

    Node* rootNode = nullptr;
    std::array<std::optional<Node>, MAX_NODES>* nodes;
    uint32_t lastNode = 0;

    vk::SampleCountFlagBits msaaSamples;
    bool vSync = true;
    bool depthView = false;

    void createInstance() {
        constexpr vk::ApplicationInfo appInfo{.pApplicationName = "Shader Forge",
                                              .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
                                              .pEngineName = "No Engine",
                                              .engineVersion = VK_MAKE_VERSION(1, 0, 0),
                                              .apiVersion = vk::ApiVersion14};
        // Get the required layers
        std::vector<char const*> requiredLayers;
        requiredLayers.assign(validationLayers.begin(), validationLayers.end());
        // Check if the required layers are supported by the Vulkan implementation.
        auto layerProperties = context.enumerateInstanceLayerProperties();
        if (std::ranges::any_of(requiredLayers, [&layerProperties](auto const& requiredLayer) {
                return std::ranges::none_of(layerProperties, [requiredLayer](auto const& layerProperty) { return strcmp(layerProperty.layerName, requiredLayer) == 0; });
            })) {
            throw std::runtime_error("One or more required layers are not supported!");
        }
        // get the required extensions
        auto requiredExtensions = getRequiredExtensions();
        // Check if the required extensions are supported by the Vulkan implementation.
        auto extensionProperties = context.enumerateInstanceExtensionProperties();
        for (auto const& requiredExtension : requiredExtensions) {
            if (std::ranges::none_of(extensionProperties,
                                     [requiredExtension](auto const& extensionProperty) { return strcmp(extensionProperty.extensionName, requiredExtension) == 0; })) {
                throw std::runtime_error("Required extension not supported: " + std::string(requiredExtension));
            }
        }
        vk::InstanceCreateInfo createInfo{.pApplicationInfo = &appInfo,
                                          .enabledLayerCount = static_cast<uint32_t>(requiredLayers.size()),
                                          .ppEnabledLayerNames = requiredLayers.data(),
                                          .enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size()),
                                          .ppEnabledExtensionNames = requiredExtensions.data()};

        instance = vk::raii::Instance(context, createInfo);
    }

    std::vector<const char*> getRequiredExtensions() {
        uint32_t glfwExtensionCount = 0;
        auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

        return extensions;
    }

    void setupDebugMessenger() {
        vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                                                            vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
        vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags(vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
                                                           vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
        vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT{.messageSeverity = severityFlags, .messageType = messageTypeFlags, .pfnUserCallback = &debugCallback};
        debugMessenger = instance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
    }

    static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity, vk::DebugUtilsMessageTypeFlagsEXT type,
                                                          const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData, void*) {
        std::cerr << "validation layer: type " << to_string(type) << " msg: " << pCallbackData->pMessage << std::endl;
        return vk::False;
    }

    void createSurface() {
        VkSurfaceKHR _surface;
        if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
        surface = vk::raii::SurfaceKHR(instance, _surface);
    }

    void createCommandPool() {
        vk::CommandPoolCreateInfo poolInfo{.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer, .queueFamilyIndex = graphicsIndex};
        commandPool = vk::raii::CommandPool(device->getDevice(), poolInfo);
    }
    void createCommandBuffers() {
        commandBuffers.clear();
        vk::CommandBufferAllocateInfo allocInfo{.commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = MAX_FRAMES_IN_FLIGHT};
        commandBuffers = vk::raii::CommandBuffers(device->getDevice(), allocInfo);
    }

    void createSyncObjects() {
        presentCompleteSemaphores.clear();
        renderFinishedSemaphores.clear();
        inFlightFences.clear();
        imagesInFlight.clear();

        // Separate semaphores for acquisition (per frame) and rendering (per image)
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            presentCompleteSemaphores.emplace_back(device->getDevice(), vk::SemaphoreCreateInfo());
            inFlightFences.emplace_back(device->getDevice(), vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
        }

        // Render finished semaphores per swapchain image
        for (size_t i = 0; i < swapchain->getSwapImageSize(); i++) {
            renderFinishedSemaphores.emplace_back(device->getDevice(), vk::SemaphoreCreateInfo());
        }

        imagesInFlight.resize(swapchain->getSwapImageSize(), VK_NULL_HANDLE);
    }

    void recordCommandBuffer(uint32_t imageIndex) {

        commandBuffers[currentFrame].begin({});

        for (auto& [lightId, light] : lights) {
            if (light.castsShadows == 1) {
                recordShadowPass(commandBuffers[currentFrame], light);
            }
        }

        resourceManager->transitionImageLayout(&commandBuffers[currentFrame], swapchain->getSwapChainImages()[imageIndex], vk::ImageLayout::eUndefined,
                                               vk::ImageLayout::eColorAttachmentOptimal);
        resourceManager->transitionImageLayout(&commandBuffers[currentFrame], swapchain->getColorImage(), vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal);
        resourceManager->transitionImageLayout(&commandBuffers[currentFrame], swapchain->getDepthImage(), vk::ImageLayout::eUndefined,
                                               vk::ImageLayout::eDepthStencilAttachmentOptimal);

        vk::ClearValue clearColor{.color = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f})};
        vk::ClearValue clearDepth{.depthStencil = vk::ClearDepthStencilValue{1.0f, 0}};

        // Color attachment (multisampled) with resolve attachment
        vk::RenderingAttachmentInfo colorAttachment = {.imageView = swapchain->getColorImageView(),
                                                       .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                       .resolveMode = vk::ResolveModeFlagBits::eAverage,
                                                       .resolveImageView = swapchain->getSwapChainImageViews()[imageIndex],
                                                       .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                       .loadOp = vk::AttachmentLoadOp::eClear,
                                                       .storeOp = vk::AttachmentStoreOp::eStore,
                                                       .clearValue = clearColor};

        // depth attachment
        vk::RenderingAttachmentInfo depthAttachmentInfo = {.imageView = swapchain->getDepthImageView(),
                                                           .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                                           .resolveMode = vk::ResolveModeFlagBits::eMin,
                                                           .resolveImageView = swapchain->getDepthResolveImageView(),
                                                           .resolveImageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                                           .loadOp = vk::AttachmentLoadOp::eClear,
                                                           .storeOp = vk::AttachmentStoreOp::eDontCare,
                                                           .clearValue = clearDepth};

        vk::RenderingInfo renderingInfo = {.renderArea = {.offset = {0, 0}, .extent = swapchain->getSwapChainExtent()},
                                           .layerCount = 1,
                                           .colorAttachmentCount = 1,
                                           .pColorAttachments = &colorAttachment,
                                           .pDepthAttachment = &depthAttachmentInfo};

        commandBuffers[currentFrame].beginRendering(renderingInfo);
        commandBuffers[currentFrame].setViewport(
            0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapchain->getSwapChainExtent().width), static_cast<float>(swapchain->getSwapChainExtent().height), 0.0f, 1.0f));
        commandBuffers[currentFrame].setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapchain->getSwapChainExtent()));

        // draw skybox
        auto& currentSkyBoxPipeline = pipelineManager->getGeoPipelines()[skyboxPipelineIndex];
        commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, currentSkyBoxPipeline->pipeline);
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, currentSkyBoxPipeline->layout, 0, {**currentSkyBoxPipeline->descriptorSet}, {});
        SkyBoxPushConstants skyboxConstants = {.skyboxIndex = skyboxIndex, .blur = 0.5, .invViewProjMatrix = glm::inverse(activeCamera.viewProjection)};
        commandBuffers[currentFrame].pushConstants<SkyBoxPushConstants>(*currentSkyBoxPipeline->layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
                                                                        skyboxConstants);
        commandBuffers[currentFrame].draw(3, 1, 0, 0);

        // draw geometry
        vk::Buffer indexBufferHandle = descriptorSet->getVariableBuffer(indexBufferIndex);
        auto& geoPipelines = pipelineManager->getGeoPipelines();
        for (auto [shader, materials] : shaders) {
            auto currentPipeline = &(geoPipelines[shader.pipelineIndex]);
            commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, (*currentPipeline)->pipeline);

            commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, (*currentPipeline)->layout, 0, {*(*currentPipeline)->descriptorSet}, {});

            for (auto [material, node_mesh] : materials) {
                for (auto [node, subMeshIndices] : node_mesh) {

                    if (meshes[node->getMeshIndex()].freed == true) { // skip if the mesh was marked to be freed and free the mesh memory
                        for (uint32_t subMesh : meshes[node->getMeshIndex()].subMeshes) {
                            descriptorSet->freeVariableBuffer(vertexBufferIndex, subMeshes[subMesh].vertexAllocationIndex);
                            descriptorSet->freeVariableBuffer(indexBufferIndex, subMeshes[subMesh].indexAllocationIndex);
                            freeSubMeshes.push(subMesh);
                        }
                        freeMeshes.push(node->getMeshIndex());
                        continue;
                    }
                    for (auto mesh : subMeshIndices) {

                        commandBuffers[currentFrame].bindIndexBuffer(indexBufferHandle, subMeshes[mesh].indexOffset, vk::IndexType::eUint32);
                        PushConstants pushConstants = {.vertexAllocationIndex = subMeshes[mesh].vertexAllocationIndex,      // Index into vertex allocations
                                                       .vertexOffset = static_cast<uint32_t>(subMeshes[mesh].vertexOffset), // Byte offset in vertex buffer
                                                       .vertexStride = subMeshes[mesh].vertexStride,                        // Size of each vertex
                                                       .modelMatrixIndex = node->getModelMatrixIndex(),                     // Index into model matrices
                                                       .albedoTextureIndex = material.albedoTextureIndex,                   // Index into textures
                                                       .roughnessTextureIndex = material.roughnessTextureIndex,
                                                       .metallicTextureIndex = material.metallicTextureIndex,
                                                       .normalTextureIndex = material.normalTextureIndex,
                                                       .environmentMapIndex = material.environmentMapIndex,
                                                       .samplerIndex = defaultSamplerIndex, // Index into samplers
                                                       .lightCount = static_cast<uint32_t>(lights.size()),
                                                       .shadowSamplerIndex = shadowSamplerIndex,
                                                       .cameraPosition = activeCamera.position,
                                                       .textureMask = material.textureMask,
                                                       .viewProjection = activeCamera.viewProjection};

                        commandBuffers[currentFrame].pushConstants<PushConstants>((*currentPipeline)->layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
                                                                                  0, pushConstants);
                        commandBuffers[currentFrame].drawIndexed(subMeshes[mesh].indexCount, 1, 0, 0, 0);
                    }
                }
            }
        }
        // draw gizmos

        auto& currentGizmoPipeline = pipelineManager->getGeoPipelines()[gizmoPipelineIndex];
        commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, currentGizmoPipeline->pipeline);
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, currentGizmoPipeline->layout, 0, {**currentGizmoPipeline->descriptorSet}, {});
        LinePushConstants lineConstants = {.viewProjection = activeCamera.viewProjection};
        commandBuffers[currentFrame].pushConstants<LinePushConstants>(*currentGizmoPipeline->layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
                                                                      lineConstants);
        commandBuffers[currentFrame].draw(gizmos->getVertexCount(), 1, 0, 0);

        // draw GUI
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), *commandBuffers[currentFrame]);
        commandBuffers[currentFrame].endRendering();

        // fullscreen quad
        if(depthView || debugShowShadow < 4){
            vk::RenderingAttachmentInfo swapchainAttachment{.imageView = swapchain->getSwapChainImageViews()[imageIndex],
                                                            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                            .loadOp = vk::AttachmentLoadOp::eLoad,
                                                            .storeOp = vk::AttachmentStoreOp::eStore};
    
            vk::RenderingInfo fullscreenRenderInfo{.renderArea = {{0, 0}, swapchain->getSwapChainExtent()},
                                                   .layerCount = 1,
                                                   .colorAttachmentCount = 1,
                                                   .pColorAttachments = &swapchainAttachment,
                                                   .pDepthAttachment = nullptr};
            commandBuffers[currentFrame].beginRendering(fullscreenRenderInfo);
            
            auto& currentDepthPipeline = pipelineManager->getAfterGeoPipelines()[depthPipelineIndex];
            resourceManager->transitionImageLayout(nullptr, swapchain->getDepthImage(), vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
            commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, currentDepthPipeline->pipeline);
            commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, currentDepthPipeline->layout, 0, {**currentDepthPipeline->descriptorSet}, {});
            DepthVisPushConstants depthConstants = {.depthIndex = swapchain->getDepthResolveIndex(),
                                                    .depthSamplerIndex = depthSamplerIndex,
                                                    .showShadowMap = showShadowMapIndex,
                                                    .shadowMapSamplerIndex = shadowSamplerIndex,
                                                    .nearPlane = activeCamera.nearPlane,
                                                    .farPlane = activeCamera.farPlane,
                                                    .linearize = 1,
                                                    .doDepthBuffering = depthView};
            commandBuffers[currentFrame].pushConstants<DepthVisPushConstants>(*currentDepthPipeline->layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
                                                                              depthConstants);
            commandBuffers[currentFrame].draw(3, 1, 0, 0);
            resourceManager->transitionImageLayout(nullptr, swapchain->getDepthImage(), vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eDepthStencilAttachmentOptimal);
    
            commandBuffers[currentFrame].endRendering();
        }
        resourceManager->transitionImageLayout(&commandBuffers[currentFrame], swapchain->getSwapChainImages()[imageIndex], vk::ImageLayout::eColorAttachmentOptimal,
                                               vk::ImageLayout::ePresentSrcKHR);

        commandBuffers[currentFrame].end();
    }

    void recordShadowPass(vk::raii::CommandBuffer& cmd, Light& light) {

        if (debugShowShadow < 4 && selectedNode != MAX_NODES) {
            if (lights[(*nodes)[selectedNode]->getLightIndex()] == light) {
                showShadowMapIndex = light.cascades[debugShowShadow].shadowMapIndex;
            }
        }
        uint32_t cascadeCount = 1;
        TextureResource* shadowMap = nullptr;
        uint32_t shadowMapResolution = light.shadowResolution;
        if (light.type == LightType::Directional) {
            cascadeCount = light.numCascades;
        } else {
            shadowMap = &descriptorSet->getTextureResource(light.shadowMapIndex);
        }
        for (int i = 0; i < cascadeCount; i++) {

            if (light.type == LightType::Directional) {
                shadowMap = &descriptorSet->getTextureResource(light.cascades[i].shadowMapIndex);
            }
            // Transition shadow map to depth attachment
            resourceManager->transitionImageLayout(&cmd, *shadowMap->image, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal);

            // Begin rendering with depth-only attachment
            vk::RenderingAttachmentInfo depthAttachment{.imageView = *shadowMap->imageView,
                                                        .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                                        .loadOp = vk::AttachmentLoadOp::eClear,
                                                        .storeOp = vk::AttachmentStoreOp::eStore,
                                                        .clearValue = {.depthStencil = {1.0f, 0}}};

            vk::RenderingInfo renderInfo{
                .renderArea = {{0, 0}, {shadowMapResolution, shadowMapResolution}}, .layerCount = 1, .colorAttachmentCount = 0, .pDepthAttachment = &depthAttachment};

            cmd.beginRendering(renderInfo);
            cmd.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(shadowMapResolution), static_cast<float>(shadowMapResolution), 0.0f, 1.0f));
            cmd.setScissor(0, vk::Rect2D({0, 0}, {shadowMapResolution, shadowMapResolution}));

            auto& currentPipeline = pipelineManager->getBeforeGeoPipelines()[shadowPipelineIndex];
            cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, currentPipeline->pipeline);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, currentPipeline->layout, 0, {*currentPipeline->descriptorSet}, {});
            vk::Buffer indexBufferHandle = descriptorSet->getVariableBuffer(indexBufferIndex);

            for (auto [shader, materials] : shaders) {
                for (auto [material, node_mesh] : materials) {
                    for (auto [node, subMeshIndices] : node_mesh) {
                        if (meshes[node->getMeshIndex()].freed == true) {
                            continue;
                        }
                        for (auto mesh : subMeshIndices) {
                            cmd.bindIndexBuffer(indexBufferHandle, subMeshes[mesh].indexOffset, vk::IndexType::eUint32);

                            ShadowPushConstants pushConstants = {.vertexAllocationIndex = subMeshes[mesh].vertexAllocationIndex,
                                                                 .vertexOffset = static_cast<uint32_t>(subMeshes[mesh].vertexOffset),
                                                                 .vertexStride = subMeshes[mesh].vertexStride,
                                                                 .modelMatrixIndex = node->getModelMatrixIndex(),
                                                                 .lightSpaceMatrix = light.lightSpaceMatrix};
                            if (light.type == LightType::Directional) {
                                pushConstants.lightSpaceMatrix = light.cascades[i].lightSpaceMatrix;
                            }
                            cmd.pushConstants<ShadowPushConstants>(*currentPipeline->layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
                                                                   pushConstants);
                            cmd.drawIndexed(subMeshes[mesh].indexCount, 1, 0, 0, 0);
                        }
                    }
                }
            }
            cmd.endRendering();
            resourceManager->transitionImageLayout(&cmd, *shadowMap->image, vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
        }
    }
};
