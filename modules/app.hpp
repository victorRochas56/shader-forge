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

#include "renderer.hpp"
#include "input.hpp"
#include "gui.hpp"

#define GLM_FORCE_RADIANS
#define GLM_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>


class App{
public:
    uint32_t start_width = 800, start_height = 600;

    void run() {
        initWindow();
        inputManager = &InputManager::getInstance();
        inputManager->setRenderer(&renderer);

        renderer.setWindow(window);
        renderer.initVulkan(start_width, start_height);

        initIMGUI(&renderer);

        uint32_t vaseMesh = renderer.loadMeshFromFile("/models/vase.OBJ");
        renderer.getRootNode()->addMesh(vaseMesh);
    }


private:
    Renderer renderer;
    GLFWwindow* window = nullptr;
    bool framebufferResized = false;
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
            /*
            ImGui::Begin("node tree");
            traverseNodeTree(renderer.getRootNode(),0, &renderer);
            ImGui::End();
            
            //show info about the selected node
            if (renderer.selectedNode != MAX_NODES) {
                renderer.getNodes()[renderer.selectedNode].showInfo();
            }*/
            //render the imgui frame
            ImGui::Render();
            //finally draw the frame
            renderer.drawFrame();
            
            frame_end = std::chrono::high_resolution_clock::now();
            frame_time = frame_end - frame_start;
            
            inputManager->tickInputState();
        }
        renderer.getDevice().getDevice().waitIdle();
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
