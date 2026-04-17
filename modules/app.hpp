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
#include "node_gui.hpp"
#include "input.hpp"
#include "renderer.hpp"
#include "scenes.hpp"
#include "gizmo.hpp"
#include "profiling.hpp"
#include "events.hpp"
#include "material_editor_state.hpp"

#define GLM_FORCE_RADIANS
#define GLM_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class App {
  public:
    uint32_t start_width = 640, start_height = 360;
    glm::vec3 addOffset = glm::vec3(0.01,0,0);
    uint32_t listen = 0;
    uint32_t effect = 0;

    void run() {
        initWindow();
        InputManager::setRenderer(&renderer);
        InputManager::setMaterialEditorState(&materialEditorState);

        renderer.setWindow(window);
        renderer.initVulkan(start_width, start_height);
        renderer.getDescriptorSet().debugDescriptorSet("after_initVulkan");
        initIMGUI(renderer.getDevice(), renderer.getInstance(), renderer.getGraphicsIndex(),
                  renderer.getSwapchain(), window);
        EventSystem::init(renderer.sceneGraph);
        
        //load a default environment map on startup
        uint32_t cubeMapIndex =
            renderer.assetManager.loadCubemapFromFile("textures/sky2/posx.jpg", "textures/sky2/posy.jpg", "textures/sky2/posz.jpg", "textures/sky2/negx.jpg", "textures/sky2/negy.jpg", "textures/sky2/negz.jpg");
        renderer.getMaterials()[renderer.getFallBackMaterial()].environmentMapIndex = cubeMapIndex;
        renderer.setSkyBox(cubeMapIndex);

        printf("SIZE OF NODE : %zu",sizeof(Node));
        //start of render loop
        mainLoop();
    }

  private:
    Renderer renderer;
    GLFWwindow* window = nullptr;
    bool framebufferResized = false;
    SceneManager sceneManager;
    MaterialEditorState materialEditorState;

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
        //basic frame timing
        // main app loop
        while (!glfwWindowShouldClose(window)) {
            Tracer::startTrace("frame time");

            glfwPollEvents();
            InputManager::tickInputState();

            //gizmos are used in "immediate mode" so cleared every frame
            Gizmos::clearLineBuffer();
            Gizmos::clearSDFBuffer();

            drawGui();

            //draw axes visualization for every node
            for (int i = 1; i <= renderer.sceneGraph.getLastNode(); i++) {
                if (!renderer.sceneGraph.isNodeValid(i)) continue;
                Gizmos::drawAxes(renderer.sceneGraph.getNodes()[i].getTransform(), 0.1f);
            }
            EventSystem::pollListeners();
            EventSystem::pollEvents();
            renderer.sceneGraph.syncDirtyNodes();

            //main draw loop
            renderer.drawFrame();

            renderer.activeCamera.updatePrevVPM(); // kinda ulgy to put here .. TODO: find a place for it

            Tracer::endTrace("frame time");
        }
        renderer.getDevice().getDevice().waitIdle();
        cleanup();
        //TODO : improve cleanup, prompt for save & exit?
    }

    void drawGui() {
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("node tree");
        traverseNodeTree(renderer.sceneGraph.getRootNode(), 0, renderer.sceneGraph.selectedNode, renderer.sceneGraph);
        ImGui::End();
        
        showNodeInfo(renderer.sceneGraph.getNodes()[renderer.sceneGraph.selectedNode], renderer);
        if (InputManager::getInstance().contextMenuShown) {
            showActionMenu(0, renderer.getWindow(), renderer.activeCamera, renderer.sceneGraph, InputManager::getInstance().contextMenuPinX, InputManager::getInstance().contextMenuPinY);
        }
        
        showMaterialEditor(materialEditorState, &renderer);
        showImageViewList(&renderer);
        showBufferAllocs(renderer.getDescriptorSet(),renderer.assetManager,renderer.getIndirectCommands());
        showDebugWindow(renderer.culledCount,renderer.cullFovScale);
        showTraces();
        showToggles(renderer.features);
        showScenesMenu(&renderer, &sceneManager);
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
