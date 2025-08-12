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
#include <debug.hpp>
#include <gui.hpp>
#include <input_callbacks.hpp>

class App {

  public:
    uint32_t start_width = 800, start_height = 600;

    void run() {
        initWindow();
        inputManager = &InputManager::getInstance();
        inputManager->setRenderer(&renderer);
        renderer.setWindow(window);
        renderer.initVulkan(start_width, start_height);
        renderer.initializeResourceDefaults();
        initIMGUI(&renderer);

        Node meshNode;
        renderer.addNode(&meshNode,&renderer.getRootNode());
        renderer.addMeshComponent(&meshNode,
            renderer.addMeshFromFile("models/vase.obj", "textures/vase_diffuse.jpg", "textures/vase_roughness.jpg", "", "textures/vase_normal.png"));
        
        meshNode.updateWorldScale(glm::vec3(0.001));
        
        addAxes(glm::vec3(0, 0, 0), 0.1, renderer.getResourceManager());

        Node lightNode0 ;
        renderer.addNode(&lightNode0,&renderer.getRootNode());
        Node lightNode1 ;
        renderer.addNode(&lightNode1,&renderer.getRootNode());
        Node lightNode2 ;
        renderer.addNode(&lightNode2,&renderer.getRootNode());
        renderer.addLightComponent(&lightNode0,renderer.addPointLight(glm::vec3(1.0), 100, 20));
        renderer.addLightComponent(&lightNode1,renderer.addPointLight(glm::vec3(1.0), 100, 10));
        renderer.addLightComponent(&lightNode2,renderer.addPointLight(glm::vec3(1.0), 100, 20));
        
        lightNode0.updateWorldPosition(glm::vec3(0.5,0.5,0));
        lightNode1.updateWorldPosition(glm::vec3(-0.5,-0.5,-0.5));
        lightNode2.updateWorldPosition(glm::vec3(0.5,0,0.5));

        for (auto& node : renderer.getRootNode().children) {
            addAxes(node->transform.position,0.1,renderer.getResourceManager());
            glm::vec3 currentWorldPos;
            glm::quat currentWorldRot;
            glm::vec3 currentWorldScale;
            decomposeMatrix(node->worldTransform, currentWorldPos, currentWorldRot, currentWorldScale);
            addAxes(currentWorldPos,0.1,renderer.getResourceManager());
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

        while (!glfwWindowShouldClose(window)) {
            frame_start = std::chrono::high_resolution_clock::now();

            glfwPollEvents();

            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            ImGui::Begin("frame time");
            std::string frame_time_str = std::to_string(frame_time.count() * 1000) + " ms";
            ImGui::Text(frame_time_str.c_str());
            ImGui::End();
            ImGui::Render();

            renderer.drawFrame();

            inputManager->tickInputState();

            frame_end = std::chrono::high_resolution_clock::now();
            frame_time = frame_end - frame_start;
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
