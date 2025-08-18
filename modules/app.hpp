#pragma once
#ifndef VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1 // for raii
#endif
#ifndef VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_NO_CONSTRUCTORS 1 // for structs constructors
#endif
#include "include/imgui.h"
#include "include/imgui_impl_glfw.h"
#include "include/imgui_impl_vulkan.h"
#include <chrono>
#include <core.hpp>
#include <gui.hpp>
#include <input_callbacks.hpp>
#include <scene_elements.hpp>

#define GLM_FORCE_RADIANS
#define GLM_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class App {

  public:
    uint32_t start_width = 800, start_height = 600;

    void run() {
        initWindow();
        inputManager = &InputManager::getInstance();
        inputManager->setRenderer(&renderer);

        renderer.setWindow(window);
        renderer.initVulkan(start_width, start_height);
        initIMGUI(&renderer);

        uint32_t meshNode = renderer.addNode(&renderer.getRootNode(), "mesh");
        renderer.addMeshComponent(&renderer.getNodes()[meshNode],
                                  renderer.addMeshFromFile("models/vase.obj", "textures/vase_diffuse.jpg", "textures/vase_roughness.jpg", "", "textures/vase_normal.png"));

        renderer.addEnvironmentMap(2048, 2048, "textures/posx.jpg", "textures/negx.jpg", "textures/posy.jpg", "textures/negy.jpg", "textures/posz.jpg",
                                   "textures/negz.jpg");

        renderer.getNodes()[meshNode].updateWorldScale(glm::vec3(0.001));

        uint32_t lightNode0 = renderer.addNode(&renderer.getRootNode(), "light0");
        uint32_t lightNode1 = renderer.addNode(&renderer.getRootNode(), "light1");
        uint32_t lightNode2 = renderer.addNode(&renderer.getRootNode(), "light2");
        renderer.addLightComponent(&renderer.getNodes()[lightNode0], renderer.addPointLight(glm::vec3(1.0), 100, 20));
        renderer.addLightComponent(&renderer.getNodes()[lightNode1], renderer.addPointLight(glm::vec3(1.0), 100, 10));
        renderer.addLightComponent(&renderer.getNodes()[lightNode2], renderer.addPointLight(glm::vec3(1.0,0,0), 100, 20));

        renderer.getNodes()[lightNode0].updateWorldPosition(glm::vec3(0.5, 0.5, 0));
        renderer.getNodes()[lightNode1].updateWorldPosition(glm::vec3(-0.5, -0.5, -0.5));
        renderer.getNodes()[lightNode2].updateWorldPosition(glm::vec3(0.5, 0, 0.5));

        for (auto& node : renderer.getRootNode().children) {
            Gizmo gizmo = {.resourceManager = renderer.getResourceManager()};
            gizmo.addAxes(glm::vec3(0, 0, 0), 0.1);
            node->gizmo = gizmo;
            node->hasGizmo = true;
            node->calculateWorldTransform();
        }
        mainLoop();
        cleanup();
    }

  private:
    GLFWwindow* window = nullptr;
    bool framebufferResized = false;
    Renderer renderer;
    InputManager* inputManager = nullptr;

    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
        window = glfwCreateWindow(start_width, start_height, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizedCallback);

        glfwSetMouseButtonCallback(window, inputManager->mouse_button_callback);
        glfwSetCursorPosCallback(window, inputManager->cursor_position_callback);
        glfwSetKeyCallback(window, inputManager->key_callback);
    }

    static void framebufferResizedCallback(GLFWwindow* window, int width, int height) {
        auto app = static_cast<App*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    void mainLoop() {
        std::chrono::steady_clock::time_point frame_start = std::chrono::high_resolution_clock::now();
        std::chrono::steady_clock::time_point frame_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> frame_time = frame_end - frame_start;

        //main app loop
        while (!glfwWindowShouldClose(window)) {
            frame_start = std::chrono::high_resolution_clock::now();

            glfwPollEvents();
            
            //start the imgui frame
            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            ImGui::Begin("frame time");
            std::string frame_time_str = std::to_string(frame_time.count() * 1000) + " ms";
            ImGui::Text(frame_time_str.c_str());
            ImGui::End();
            
            ImGui::Begin("node tree");
            traverseNodeTree(renderer.getRootNode(),0, &renderer);
            ImGui::End();

            //show info about the selected node
            if (renderer.selectedNode != MAX_NODES) {
                renderer.getNodes()[renderer.selectedNode].showInfo();
            }
            //render the imgui frame
            ImGui::Render();
            //finally draw the frame
            renderer.drawFrame();
            
            frame_end = std::chrono::high_resolution_clock::now();
            frame_time = frame_end - frame_start;
            
            inputManager->tickInputState();
        }
        renderer.getDevices()->getLogicalDevice().waitIdle();
    }

    void cleanup() {

        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        renderer.getDevices()->getLogicalDevice().waitIdle();
        renderer.cleanupSwapChain();

        glfwDestroyWindow(window);
        glfwTerminate();
    }
};
