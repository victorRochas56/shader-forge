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

#include <array>
#include <chrono>
#include <iostream>

#include <nlohmann/json.hpp>

#include "gui.hpp"
#include "node_gui.hpp"
#include "input.hpp"
#include "renderer.hpp"
#include "scene_loader.hpp"
#include "gizmo.hpp"
#include "profiling.hpp"
#include "events.hpp"
#include "material_editor_state.hpp"
#include "manipulators.hpp"

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
        InputManager::init(renderer, scene);
        InputManager::setMaterialEditorState(&materialEditorState);

        gpu.initCore(window);
        renderer.initVulkan(start_width, start_height);
        bindless.descriptorSet->debugDescriptorSet("after_initVulkan");
        initIMGUI(gpu.getDevice(), *gpu.getInstance(), gpu.getGraphicsIndex(),
                  gpu.getSwapchain(), window);
        EventSystem::init(scene.sceneGraph);
        
        //load a default environment map on startup
        uint32_t cubeMapIndex =
            scene.assetManager.loadCubemapFromFile("textures/sky2/posx.jpg", "textures/sky2/posy.jpg", "textures/sky2/posz.jpg", "textures/sky2/negx.jpg", "textures/sky2/negy.jpg", "textures/sky2/negz.jpg");
        scene.getMaterials()[scene.getFallBackMaterial()].environmentMapIndex = cubeMapIndex;
        scene.setSkyBox(cubeMapIndex);

        renderer.buffers.nodeTextureIndex = scene.assetManager.loadTextureFromFile("textures/node.png");

        printf("SIZE OF NODE : %zu",sizeof(Node));

        //start of render loop
        mainLoop();
    }

  private:
    GpuContext gpu;
    BindlessSystem bindless;
    Scene scene;
    Renderer renderer{gpu, bindless, scene};
    GLFWwindow* window = nullptr;
    bool framebufferResized = false;
    SceneLoader sceneLoader;
    MaterialEditorState materialEditorState;

    bool showAllNodes = false;

    std::vector<Line> lines;

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
        std::chrono::steady_clock::time_point frameStart;
        std::chrono::duration<double> deltaTime;

        while (!glfwWindowShouldClose(window)) {
            tracing::startTrace("frame time");
            frameStart = std::chrono::steady_clock::now();

            glfwPollEvents();
            std::chrono::steady_clock::time_point tPoll = std::chrono::steady_clock::now();
            InputManager::tickInputState();

            //gizmos are used in "immediate mode" so cleared every frame
            Gizmos::clearLineBuffer();
            Gizmos::clearSDFBuffer();


            std::chrono::steady_clock::time_point tPreGui = std::chrono::steady_clock::now();
            drawGui();
            std::chrono::steady_clock::time_point tGui = std::chrono::steady_clock::now();

            for(auto& line : lines){
                Gizmos::drawLine(line);
            }

            //draw axes visualization for every node
            Node* currentNode = &scene.sceneGraph.getRootNode();
            if(scene.sceneGraph.selectedNode != 0) {
                currentNode = &scene.sceneGraph.getNode(scene.sceneGraph.selectedNode);
                Manip::handleInput(*currentNode,scene.activeCamera,InputManager::getCurrentState().ndcMousePos,scene.sceneGraph);
            }

            if(InputManager::getCurrentState().keyStates[GLFW_KEY_Q] == GLFW_PRESS && InputManager::getPreviousState().keyStates[GLFW_KEY_Q] != GLFW_PRESS) {
                showAllNodes = !showAllNodes;
            }
            if(InputManager::getCurrentState().keyStates[GLFW_KEY_G] == GLFW_PRESS && InputManager::getPreviousState().keyStates[GLFW_KEY_G] != GLFW_PRESS) {
                renderer.features.showGizmos = !renderer.features.showGizmos;
            }
            //TODO why does the buffer allocs change only for toggling Q not G
            for(auto& bb : scene.billboards){
                bb.second.hidden = !showAllNodes | !renderer.features.showGizmos;
            }
            if(renderer.features.showGizmos){
                scene.getBillboardsMutable()[currentNode->nodeIndex].hidden = false;
                scene.sceneGraph.forEachChild(*currentNode, [this](Node& child){
                    scene.getBillboardsMutable()[child.nodeIndex].hidden = false;
                });
            }

            EventSystem::pollListeners();
            EventSystem::pollEvents();
            scene.sceneGraph.syncDirtyNodes();

            if(scene.sceneGraph.selectedNode != 0) {
                Manip::drawGizmos(*currentNode, scene.activeCamera);
            }

            std::chrono::steady_clock::time_point tPreDraw = std::chrono::steady_clock::now();
            //main draw loop
            renderer.drawFrame();

            scene.activeCamera.updatePrevVPM(); // kinda ulgy to put here .. TODO: find a place for it

            InputManager::endFrame();

            deltaTime = std::chrono::steady_clock::now() - frameStart;
            // Companion to renderer.cpp's [slow frame]: splits a slow loop iteration into its phases.
            static std::chrono::steady_clock::time_point lastLoopPrint{};
            if (deltaTime.count() * 1000.0 > 200.0 && std::chrono::steady_clock::now() - lastLoopPrint > std::chrono::seconds(1)) {
                lastLoopPrint = std::chrono::steady_clock::now();
                auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
                auto tEnd = std::chrono::steady_clock::now();
                std::cout << "[slow loop] total " << ms(frameStart, tEnd)
                          << "ms | pollEvents " << ms(frameStart, tPoll)
                          << "ms | input/gizmoClear " << ms(tPoll, tPreGui)
                          << "ms | gui " << ms(tPreGui, tGui)
                          << "ms | manip/events/sync " << ms(tGui, tPreDraw)
                          << "ms | drawFrame+ " << ms(tPreDraw, tEnd) << "ms" << std::endl;
            }
            gpu.deltaTime = static_cast<float>(deltaTime.count());
            gpu.time += gpu.deltaTime;
            tracing::endTrace("frame time");
        }
        gpu.getDevice().getDevice().waitIdle();
        cleanup();
        //TODO : improve cleanup, prompt for save & exit?
    } 

    void drawGui() {
        // Per-window timing: [slow gui] names the window eating a slow drawGui (see [slow loop]).
        using guiClk = std::chrono::steady_clock;
        std::array<std::pair<const char*, guiClk::time_point>, 14> guiMarks;
        size_t guiMarkCount = 0;
        auto guiMark = [&](const char* name) { guiMarks[guiMarkCount++] = {name, guiClk::now()}; };
        guiMark("start");

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        guiMark("newFrame");

        ImGui::Begin("node tree");
        traverseNodeTree(scene.sceneGraph.getRootNode(), 0, scene.sceneGraph.selectedNode, scene.sceneGraph);
        ImGui::End();
        guiMark("nodeTree");

        showNodeInfo(scene.sceneGraph.getNodes()[scene.sceneGraph.selectedNode], scene, bindless, renderer.buffers);
        if (InputManager::getInstance().contextMenuShown) {
            showActionMenu(0, gpu.getWindow(), scene.activeCamera, scene.sceneGraph, InputManager::getInstance().contextMenuPinX, InputManager::getInstance().contextMenuPinY);
        }
        guiMark("nodeInfo");

        showMaterialEditor(materialEditorState, scene, bindless);
        guiMark("materialEditor");
        showImageViewList(bindless, renderer.features);
        guiMark("imageViewList");
        showBufferAllocs(*bindless.descriptorSet,scene.assetManager,renderer.getIndirectCommands());
        guiMark("bufferAllocs");
        showDebugWindow(renderer.culledCount,renderer.cullFovScale);
        guiMark("debugWindow");
        showTraces();
        guiMark("traces");
        showToggles(renderer.features);
        guiMark("toggles");
        showScenesMenu(scene, bindless, renderer.buffers, sceneLoader);
        guiMark("scenesMenu");
        showMeshList(scene, bindless, renderer.defaultAlbedoIndex);
        guiMark("meshList");
        ImGui::Render();
        guiMark("render");

        auto guiMs = [](guiClk::time_point a, guiClk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        if (guiMs(guiMarks[0].second, guiMarks[guiMarkCount - 1].second) > 200.0) {
            static guiClk::time_point lastGuiPrint{};
            if (guiClk::now() - lastGuiPrint > std::chrono::seconds(1)) {
                lastGuiPrint = guiClk::now();
                std::cout << "[slow gui]";
                for (size_t i = 1; i < guiMarkCount; i++)
                    std::cout << " | " << guiMarks[i].first << " " << guiMs(guiMarks[i - 1].second, guiMarks[i].second) << "ms";
                std::cout << std::endl;
            }
        }
    }

    void cleanup() {

        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        gpu.getDevice().getDevice().waitIdle();
        gpu.getSwapchain().cleanupSwapChain();

        glfwDestroyWindow(window);
        glfwTerminate();
    }
};
