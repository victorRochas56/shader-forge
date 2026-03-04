#pragma once
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <unordered_map>
#include "renderer.hpp"
#include "scene_elements.hpp"

/*
reads inputs through GLFW, stores this frame's and last frame's inputs in 2 InputState structs (current & previous)
*/

struct InputState {
    glm::vec2 mousePos = glm::vec2(0);
    glm::vec2 mouseDelta = glm::vec2(0);
    glm::vec2 scroll = glm::vec2(0);
    int mouse_button = 0;
    int mouse_action = 0;
    std::unordered_map<int, int> keyStates; // Track the state of each key
};

class InputManager {
  public:

    bool contextMenuShown = false;
    bool canMove = true;
    float contextMenuPinX = 0;
    float contextMenuPinY = 0;
  
    static InputManager& getInstance() {
        static InputManager instance;
        return instance;
    }

    
    static void tickInputState() { //should be only called once in the main app loop
        InputManager& inst = getInstance();

        //rotating viewport when holding right click
        if (inst.current.mouse_button == 1 && inst.current.mouse_action == 1) {
            inst.renderer->activeCamera.rotatePitch(inst.current.mouseDelta.y * -0.1f);
            inst.renderer->activeCamera.rotateYaw(inst.current.mouseDelta.x * 0.1f);
        }

        glm::vec3 moveVector(0.0f);

        // Check the state of movement keys
        if(inst.canMove){
            if (inst.current.keyStates[GLFW_KEY_W] == GLFW_PRESS || inst.current.keyStates[GLFW_KEY_W] == GLFW_REPEAT) {
                moveVector.z += 0.05f;
            }
            if (inst.current.keyStates[GLFW_KEY_S] == GLFW_PRESS || inst.current.keyStates[GLFW_KEY_S] == GLFW_REPEAT) {
                moveVector.z -= 0.05f;
            }
            if (inst.current.keyStates[GLFW_KEY_A] == GLFW_PRESS || inst.current.keyStates[GLFW_KEY_A] == GLFW_REPEAT) {
                moveVector.x -= 0.05f;
            }
            if (inst.current.keyStates[GLFW_KEY_D] == GLFW_PRESS || inst.current.keyStates[GLFW_KEY_D] == GLFW_REPEAT) {
                moveVector.x += 0.05f;
            }
        }

        // Apply the combined movement
        if (glm::length(moveVector) > 0) {
            inst.renderer->activeCamera.moveCamera(moveVector);
        }

        // raycast on left click for selection
        if (inst.current.mouse_button == 0 && inst.current.mouse_action == 1 && inst.previous.mouse_action != 1) {
            int width = 0, height = 0;
            glfwGetWindowSize(inst.renderer->getWindow(), &width, &height);
            float NDCx = (inst.current.mousePos.x / width) * 2.0 - 1.0;
            float NDCy = (inst.current.mousePos.y / height) * 2.0 - 1.0;
            glm::vec3 origin;
            glm::vec3 direction;
            inst.renderer->activeCamera.rayFromScreenCoords(NDCx, NDCy, &origin, &direction);
            std::vector<uint32_t> hitNodes = inst.renderer->rayCastNodes(origin, direction);
            if (!hitNodes.empty()) {
                inst.renderer->selectNode(hitNodes.front());
            }
        }
        // contextual menu on right click
        if(inst.current.mouse_button == 1 && inst.current.mouse_action == 0 && inst.previous.mouse_action == 1){
            inst.contextMenuShown = !inst.contextMenuShown;
            inst.contextMenuPinX = inst.current.mousePos.x;
            inst.contextMenuPinY = inst.current.mousePos.y;
        }

        if (inst.current.keyStates[GLFW_KEY_ESCAPE] == GLFW_PRESS && inst.previous.keyStates[GLFW_KEY_ESCAPE] != GLFW_PRESS) {
            inst.renderer->deSelectNode();
        }
        if (inst.current.keyStates[GLFW_KEY_V] == GLFW_PRESS && inst.previous.keyStates[GLFW_KEY_V] != GLFW_PRESS) {
            inst.renderer->toggleVsync();
        }
        if (inst.current.keyStates[GLFW_KEY_C] == GLFW_PRESS && inst.previous.keyStates[GLFW_KEY_C] != GLFW_PRESS) {
            inst.renderer->showShadowMap();
        }/*
        if (current.keyStates[GLFW_KEY_R] == GLFW_PRESS && previous.keyStates[GLFW_KEY_R] != GLFW_PRESS) {
            renderer->showMap(SHOW_ROUGHNESS);
        }
        if (current.keyStates[GLFW_KEY_M] == GLFW_PRESS && previous.keyStates[GLFW_KEY_M] != GLFW_PRESS) {
            renderer->showMap(SHOW_METALLIC);
        }
        if (current.keyStates[GLFW_KEY_N] == GLFW_PRESS && previous.keyStates[GLFW_KEY_N] != GLFW_PRESS) {
            renderer->showMap(SHOW_NORMAL);
        }*/
        if (inst.current.keyStates[GLFW_KEY_F] == GLFW_PRESS && inst.previous.keyStates[GLFW_KEY_F] != GLFW_PRESS) {
            inst.renderer->toggleDepthView();
        }
        inst.canMove = true;
        inst.current.mouseDelta = inst.current.mousePos - inst.previous.mousePos;
        inst.previous = inst.current;
    }

    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
        getInstance().current.mouse_button = button;
        getInstance().current.mouse_action = action;
    }

    static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) { getInstance().current.mousePos = glm::vec2(xpos, ypos); }

    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) { getInstance().current.keyStates[key] = action; }

    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {}

    static void setRenderer(Renderer* pRenderer) { getInstance().renderer = pRenderer; }

    InputState& getCurrentState() {return current;}

  private:
    InputState current, previous;
    Renderer* renderer;
};