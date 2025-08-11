#pragma once
#include <core.hpp>
#include <debug.hpp>
#include <input_callbacks.hpp>
#include <gui.hpp>
#include "include/imgui.h"
#include "include/imgui_impl_glfw.h"
#include "include/imgui_impl_vulkan.h"

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
        initIMGUI( &renderer);
        renderer.addMeshFromFile("models/vase.obj", "textures/vase_diffuse.jpg", "textures/vase_roughness.jpg", "", "textures/vase_normal.png", glm::vec3(0.0, 0.0, 0.0), glm::quat(1.0, 0, 0, 0),
                                 glm::vec3(0.001));

        addAxes(glm::vec3(0, 0, 0), 0.1, renderer.getResourceManager());
        renderer.addPointLight(glm::vec3(0, 1, 0), glm::vec3(1.0), 100, 20);
        renderer.addPointLight(glm::vec3(1, 0, 1), glm::vec3(1.0), 100, 10);
        renderer.addPointLight(glm::vec3(-1, -1, -1), glm::vec3(1.0), 100, 20);
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
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            renderer.drawFrame();
            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            ImGui::Render();
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
