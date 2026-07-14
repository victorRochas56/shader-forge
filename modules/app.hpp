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
            InputManager::tickInputState();

            //gizmos are used in "immediate mode" so cleared every frame
            Gizmos::clearLineBuffer();
            Gizmos::clearSDFBuffer();


            drawGui();

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

            //main draw loop
            renderer.drawFrame();

            scene.activeCamera.updatePrevVPM(); // kinda ulgy to put here .. TODO: find a place for it

            InputManager::endFrame();

            deltaTime = std::chrono::steady_clock::now() - frameStart;
            gpu.deltaTime = static_cast<float>(deltaTime.count());
            gpu.time += gpu.deltaTime;
            tracing::endTrace("frame time");
        }
        gpu.getDevice().getDevice().waitIdle();
        cleanup();
        //TODO : improve cleanup, prompt for save & exit?
    } 

    void drawGui() {
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("node tree");
        traverseNodeTree(scene.sceneGraph.getRootNode(), 0, scene.sceneGraph.selectedNode, scene.sceneGraph);
        ImGui::End();
        
        showNodeInfo(scene.sceneGraph.getNodes()[scene.sceneGraph.selectedNode], scene, bindless, renderer.getLightBufferIndex(), renderer.buffers);
        if (InputManager::getInstance().contextMenuShown) {
            showActionMenu(0, gpu.getWindow(), scene.activeCamera, scene.sceneGraph, InputManager::getInstance().contextMenuPinX, InputManager::getInstance().contextMenuPinY);
        }
        
        showMaterialEditor(materialEditorState, scene, bindless);
        showImageViewList(bindless, renderer.features);
        showBufferAllocs(*bindless.descriptorSet,scene.assetManager,renderer.getIndirectCommands());
        showDebugWindow(renderer.culledCount,renderer.cullFovScale);
        showTraces();
        showToggles(renderer.features);
        showScenesMenu(scene, bindless, renderer.buffers, sceneLoader);
        showMeshList(scene, bindless, renderer.defaultAlbedoIndex);
        ImGui::Render();
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
