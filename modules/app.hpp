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
#include "GUI.h"

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
        InputManager::setGUI(&gui);

        gpu.initCore(window);
        renderer.initVulkan(start_width, start_height);
        //bindless.descriptorSet->debugDescriptorSet("after_initVulkan");

        gui.init(bindless, gpu);
        gui.loadLayout(GUI_LAYOUT_FILE);
        EventSystem::init(scene.sceneGraph);
        
        //default environment map
        uint32_t cubeMapIndex =
            scene.assetManager.loadCubemapFromFile("textures/sky2/posx.jpg", "textures/sky2/posy.jpg", "textures/sky2/posz.jpg", "textures/sky2/negx.jpg", "textures/sky2/negy.jpg", "textures/sky2/negz.jpg");
        scene.getMaterials()[scene.getFallBackMaterial()].environmentMapIndex = cubeMapIndex;
        scene.setSkyBox(cubeMapIndex);
        // node icon, probably needs a better place to be
        renderer.buffers.nodeTextureIndex = scene.assetManager.loadTextureFromFile("textures/node.png");

        printf("SIZE OF NODE : %zu",sizeof(Node));

        mainLoop();
    }

  private:
    GpuContext          gpu;
    Renderer            renderer{gpu, bindless, scene, gui};
    BindlessSystem      bindless;
    Scene               scene;
    SceneLoader         sceneLoader;
    GUI                 gui;
    MaterialEditorState materialEditorState;
    bool                showAllNodes = false;
    
    GLFWwindow*         window = nullptr;
    bool                framebufferResized = false;

    double              lastFrameMs = 0.0;
    double              lastGuiMs = 0.0;

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
        glfwSetScrollCallback(window, InputManager::scroll_callback);
        glfwSetCharCallback(window, InputManager::char_callback);
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

            auto inputExtent = gpu.getSwapchain().getSwapChainExtent();
            gui.resolveLayout(glm::uvec2(inputExtent.width, inputExtent.height));
            gui.hitTest();
            InputManager::tickInputState();
            // after tickInputState: handlers read this frame's mouseDelta
            gui.runBehaviour();

            //gizmos are rebuilt every frame
            Gizmos::clearLineBuffer();
            Gizmos::clearSDFBuffer();

            std::chrono::steady_clock::time_point tPreGui = std::chrono::steady_clock::now();
            drawGui();
            std::chrono::steady_clock::time_point tGui = std::chrono::steady_clock::now();

            //draw manipulator visualization for every node
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

            auto guiExtent = gpu.getSwapchain().getSwapChainExtent();
            gui.uploadGPU(bindless, gpu.currentFrame, glm::uvec2(guiExtent.width, guiExtent.height));
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
            lastFrameMs = deltaTime.count() * 1000.0;
            lastGuiMs = std::chrono::duration<double, std::milli>(tGui - tPreGui).count();
            gpu.deltaTime = static_cast<float>(deltaTime.count());
            gpu.time += gpu.deltaTime;
            tracing::endTrace("frame time");
        }
        gpu.getDevice().getDevice().waitIdle();
        cleanup();
        //TODO : improve cleanup, prompt for save & exit?
    } 

    void drawGui() {
        // timing for finding windows eating a slow drawGui (in [slow loop])
        using guiClk = std::chrono::steady_clock;
        std::array<std::pair<const char*, guiClk::time_point>, 14> guiMarks;
        size_t guiMarkCount = 0;
        auto guiMark = [&](const char* name) { guiMarks[guiMarkCount++] = {name, guiClk::now()}; };
        guiMark("start");

        gui.beginFrame();
        guiMark("newFrame");

        if (gui.beginWindow("node tree", nullptr, glm::vec2(280, 320), glm::vec2(20, 450))) {
            traverseNodeTree(gui, scene.sceneGraph.getRootNode(), 0, scene.sceneGraph.selectedNode, scene.sceneGraph);
            gui.endWindow();
        }
        guiMark("nodeTree");

        showNodeInfo(gui, scene.sceneGraph.getNodes()[scene.sceneGraph.selectedNode], scene, bindless, renderer.buffers);
        if (InputManager::getInstance().contextMenuShown) {
            showActionMenu(gui, 0, gpu.getWindow(), scene.activeCamera, scene.sceneGraph, InputManager::getInstance().contextMenuPinX, InputManager::getInstance().contextMenuPinY);
        }
        guiMark("nodeInfo");

        showMaterialEditor(gui, materialEditorState, scene, bindless, renderer.features);
        guiMark("materialEditor");
        showImageViewList(gui, bindless, renderer.features);
        guiMark("imageViewList");
        showBufferAllocs(gui, *bindless.descriptorSet,scene.assetManager,renderer.getIndirectCommands());
        guiMark("bufferAllocs");
        if (gui.beginWindow("Debug", nullptr, glm::vec2(320, 150), glm::vec2(-870, 0), GUIAnchor::Top | GUIAnchor::Right, GUIWindowNoSavedSettings | GUIWindowFixed)) {
            gui.textf("Culled : %u", renderer.culledCount);
            gui.sliderFloat("Cull FOV Scale", &renderer.cullFovScale, 0.1f, 1.0f);
            gui.separator();
            gui.textf("frame %.2f ms", lastFrameMs);
            gui.textf("gui   %.2f ms", lastGuiMs);
            gui.textf("quads %u", gui.getQuadCount());
            gui.endWindow();
        }
        guiMark("debugWindow");
        showTraces(gui);
        guiMark("traces");
        showToggles(gui, renderer.features);
        guiMark("toggles");
        showScenesMenu(gui, scene, bindless, renderer.buffers, sceneLoader);
        guiMark("scenesMenu");
        showMeshList(gui, scene, bindless, renderer.defaultAlbedoIndex);
        guiMark("meshList");
        // after every widget call: retires anything this frame stopped drawing
        gui.endFrame();
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
        gui.saveLayout(GUI_LAYOUT_FILE);

        gpu.getDevice().getDevice().waitIdle();
        gpu.getSwapchain().cleanupSwapChain();

        glfwDestroyWindow(window);
        glfwTerminate();
    }
};
