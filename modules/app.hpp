#pragma once
#ifndef VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1 // for raii
#endif
#ifndef VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_NO_CONSTRUCTORS 1 // for structs constructors
#endif

#ifndef DEBUG
#define DEBUG 0
#endif

#include "include/imgui.h"
#include "include/imgui_impl_glfw.h"
#include "include/imgui_impl_vulkan.h"

#include <chrono>

#include "gui.hpp"
#include "input.hpp"
#include "renderer.hpp"

#define GLM_FORCE_RADIANS
#define GLM_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class App {
  public:
    uint32_t start_width = 800, start_height = 600;

    void run() {
        initWindow();
        InputManager::setRenderer(&renderer);

        renderer.setWindow(window);
        renderer.initVulkan(start_width, start_height);
        renderer.getDescriptorSet().debugDescriptorSet("after_initVulkan");
        initIMGUI(&renderer);

        uint32_t vaseMesh = renderer.loadMeshFromFile("models/vase.OBJ");
        uint32_t meshNode = renderer.addNode(0, glm::vec3(0, 0, 0), glm::quat(1.0, 0, 0, 0), glm::vec3(0.005, 0.005, 0.005));
        renderer.getNodes()[meshNode]->name = "vase_mesh";
        renderer.getNodes()[meshNode]->addMesh(vaseMesh);

        uint32_t texMask = 0x000000000;
        texMask |= (1U << 0);
        texMask |= (1U << 1);
        texMask |= (1U << 3);
        uint32_t cubeMapIndex =
            renderer.loadCubemapFromFile("textures/posx.jpg", "textures/posy.jpg", "textures/posz.jpg", "textures/negx.jpg", "textures/negy.jpg", "textures/negz.jpg");
        Material material = {.shaderSource = renderer.getFallBackShader(),
                             .textureMask = texMask,
                             // 1st bit : hasAlbedo
                             // 2nd bit : hasRoughness
                             // 3rd bit : hasMetallic
                             // 4th bit : hasNormal
                             .color = glm::vec4(1.0, 1.0, 1.0, 1.0),
                             .albedoTextureIndex = renderer.loadTextureFromFile("textures/vase_diffuse.jpg"),
                             .metallic = 0.0,
                             .roughness = 0.8,
                             .roughnessTextureIndex = renderer.loadTextureFromFile("textures/vase_roughness.jpg"),
                             .normalTextureIndex = renderer.loadTextureFromFile("textures/vase_normal.png", vk::Format::eR8G8B8A8Unorm),
                             .environmentMapIndex = cubeMapIndex};

        uint32_t matIndex = renderer.addMaterial(material);
        renderer.getNodes()[meshNode]->addMaterial(0, matIndex);
        renderer.setSkyBox(cubeMapIndex);
        uint32_t lightNodeIndex = renderer.addNode(0, glm::vec3(2, 0, 0), glm::quat(1.0, 0, 0, 0), glm::vec3(1, 1, 1));
        Light light = {.range = 50, .intensity = 15, .color = glm::vec4(1, 1, 1, 1)};
        renderer.getNodes()[lightNodeIndex]->addLight(light);
        mainLoop();
    }

  private:
    Renderer renderer;
    GLFWwindow* window = nullptr;
    bool framebufferResized = false;

    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
        window = glfwCreateWindow(start_width, start_height, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizedCallback);

        glfwSetMouseButtonCallback(window, InputManager::mouse_button_callback);
        glfwSetCursorPosCallback(window, InputManager::cursor_position_callback);
        glfwSetKeyCallback(window, InputManager::key_callback);
    }

    static void framebufferResizedCallback(GLFWwindow* window, int width, int height) {
        auto app = static_cast<App*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    void mainLoop() {
        std::chrono::steady_clock::time_point frame_start = std::chrono::high_resolution_clock::now();
        std::chrono::steady_clock::time_point frame_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> frame_time = frame_end - frame_start;
        // main app loop
        while (!glfwWindowShouldClose(window)) {
            frame_start = std::chrono::high_resolution_clock::now();

            glfwPollEvents();

            drawGui(frame_time);
            renderer.gizmos->clearLineBuffer();
            for (int i = 0; i < renderer.getNodeCount(); i++) {
                renderer.gizmos->drawAxes(renderer.getNodes()[i]->getTransform(), 0.15);
            }
            renderer.drawFrame();

            frame_end = std::chrono::high_resolution_clock::now();
            frame_time = frame_end - frame_start;

            InputManager::tickInputState();
        }
        renderer.getDevice().getDevice().waitIdle();
    }

    void drawGui(std::chrono::duration<double>& frame_time) {
        // start the imgui frame
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGui::Begin("frame time");
        std::string frame_time_str = std::to_string(frame_time.count() * 1000) + " ms";
        ImGui::Text(frame_time_str.c_str());
        ImGui::End();

        ImGui::Begin("node tree");
        traverseNodeTree(renderer.getRootNode(), 0, renderer.selectedNode, &renderer);
        ImGui::End();
        if (renderer.selectedNode != 2048) {
            renderer.getNodes()[renderer.selectedNode]->showInfo();
        }
        if (InputManager::getInstance().contextMenuShown) {
            showActionMenu(0, &renderer, InputManager::getInstance().contextMenuPinX, InputManager::getInstance().contextMenuPinY);
        }
        // render the imgui frame
        ImGui::Render();
    }

    void cleanup() {

        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        renderer.getDevice().getDevice().waitIdle();
        renderer.cleanupSwapchain();

        glfwDestroyWindow(window);
        glfwTerminate();
    }
};
