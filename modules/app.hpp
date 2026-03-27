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

#define GLM_FORCE_RADIANS
#define GLM_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class App {
  public:
    uint32_t start_width = 640, start_height = 360;

    void run() {
        initWindow();
        InputManager::setRenderer(&renderer);
        InputManager::setMaterialEditorState(&materialEditorState);

        renderer.setWindow(window);
        renderer.initVulkan(start_width, start_height);
        renderer.getDescriptorSet().debugDescriptorSet("after_initVulkan");
        initIMGUI(&renderer);
        
        //load a default environment map on startup
        uint32_t cubeMapIndex =
            renderer.assetManager.loadCubemapFromFile("textures/sky2/posx.jpg", "textures/sky2/posy.jpg", "textures/sky2/posz.jpg", "textures/sky2/negx.jpg", "textures/sky2/negy.jpg", "textures/sky2/negz.jpg");
        renderer.getMaterials()[renderer.getFallBackMaterial()].environmentMapIndex = cubeMapIndex;
        renderer.setSkyBox(cubeMapIndex);

        //start of render loop
        mainLoop();
    }

  private:
    Renderer renderer;
    GLFWwindow* window = nullptr;
    bool framebufferResized = false;
    SceneManager sceneManager;
    MaterialEditorState materialEditorState;

    static constexpr int FRAME_HISTORY_COUNT = 128;
    double frameTimeHistory[FRAME_HISTORY_COUNT] = {};
    int frameTimeIndex = 0;
    int frameTimeCount = 0;

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
        std::chrono::steady_clock::time_point frame_start = std::chrono::high_resolution_clock::now();
        std::chrono::steady_clock::time_point frame_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> frame_time = frame_end - frame_start;
        // main app loop
        while (!glfwWindowShouldClose(window)) {
            frame_start = std::chrono::high_resolution_clock::now();

            glfwPollEvents();

            drawGui(frame_time);

            //gizmos are used in "immediate mode" so cleared every frame
            Gizmos::clearLineBuffer();
            //draw axes visualization for every node
            for (int i = 0; i < renderer.sceneGraph.getNodeCount(); i++) {
                Gizmos::drawAxes(renderer.sceneGraph.getNodes()[i]->getTransform(), 0.15);
            }
            //main draw loop
            renderer.drawFrame();

            frame_end = std::chrono::high_resolution_clock::now();
            frame_time = frame_end - frame_start;

            frameTimeHistory[frameTimeIndex] = frame_time.count();
            frameTimeIndex = (frameTimeIndex + 1) % FRAME_HISTORY_COUNT;
            if (frameTimeCount < FRAME_HISTORY_COUNT) frameTimeCount++;

            renderer.activeCamera.updatePrevVPM(); // kinda ulgy to put here .. TODO: find a place for it
            InputManager::tickInputState();
        }
        renderer.getDevice().getDevice().waitIdle();
        cleanup();
        //TODO : improve cleanup, prompt for save & exit?
    }

    void drawGui(std::chrono::duration<double>& frame_time) {

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        
        ImGui::Begin("frame time");
        double avgFrameTime = 0.0;
        for (int i = 0; i < frameTimeCount; i++) avgFrameTime += frameTimeHistory[i];
        if (frameTimeCount > 0) avgFrameTime /= frameTimeCount;
        ImGui::Text("%.2f ms (avg %d frames)", avgFrameTime * 1000.0, frameTimeCount);
        ImGui::End();

        ImGui::Begin("node tree");
        traverseNodeTree(renderer.sceneGraph.getRootNode(), 0, renderer.sceneGraph.selectedNode, &renderer);
        ImGui::End();
        if (renderer.sceneGraph.selectedNode != MAX_NODES) {
            showNodeInfo(*renderer.sceneGraph.getNodes()[renderer.sceneGraph.selectedNode], renderer);
        }
        if (InputManager::getInstance().contextMenuShown) {
            showActionMenu(0, &renderer, InputManager::getInstance().contextMenuPinX, InputManager::getInstance().contextMenuPinY);
        }

        showMaterialEditor(materialEditorState, &renderer);
        
        showImageViewList(&renderer);

        showBufferAllocs(&renderer);
        showDebugWindow(&renderer);

        ImGui::Begin("Toggles");
        ImGui::SliderInt("ImageMip",&renderer.imageVisMipLevel,0,6);
        if(ImGui::Button("Depth Buffer")){
            renderer.imageVisFlags ^= ImageVisFlags::LINEARIZE;
        }
        if(ImGui::Button("SSAO")){
            renderer.toggleSSAO();
        }
        if(renderer.enableSSAO){
            ImGui::SliderFloat("AO Radius",&renderer.ssaoRadius,0.01f,10.0f);
            ImGui::SliderFloat("AO Bias",&renderer.ssaoBias,0.01f,0.1f);
            ImGui::SliderFloat("AO Power",&renderer.ssaoPower,0.01f,5.0f);
        }
        if(ImGui::Button("SSR")){
            renderer.toggleSSR();
        }
        if(renderer.enableSSR){
            ImGui::SliderFloat("Max Distance",&renderer.ssrMaxDistance,1.0f,100.0f);
            ImGui::SliderInt("Max Steps",&renderer.ssrMaxSteps,16,128);
            ImGui::SliderFloat("Thickness",&renderer.ssrThickness,0.01f,5.0f);
            ImGui::SliderFloat("Roughness Threshold",&renderer.ssrRoughnessThreshold,0.0f,1.0f);
            if(ImGui::SliderFloat("Resolution Scale",&renderer.ssrResolutionScale,0.25f,1.0f)){
                renderer.ssrResolutionDirty = true;
            }
        }
        if(ImGui::Button("Show BBOXes")){
            renderer.toggleBBOXes();
        }
        ImGui::End();


        //scenes.hpp
        ImGui::Begin("Scene Manager");
        if(ImGui::Button("Save Scene")){
            sceneManager.saveScene("scene.txt", renderer);
        }
        ImGui::SameLine();
        if(ImGui::Button("Load Scene")){
            sceneManager.loadScene("scene.txt", renderer);
        }
        ImGui::SameLine();
        if(ImGui::Button("Clear Scene")){
            sceneManager.clearScene(renderer);
        }

        ImGui::End();
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
