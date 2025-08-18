#pragma once
#include <GLFW/glfw3.h>
#include <core.hpp>
#include <glm/glm.hpp>
#include <scene_elements.hpp>
#include <unordered_map>

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
    static InputManager& getInstance() {
        static InputManager instance;
        return instance;
    }

    void tickInputState() {
        if (current.mouse_button == 1 && current.mouse_action == 1) {
            renderer->activeCamera.rotatePitch(current.mouseDelta.y * -0.1f);
            renderer->activeCamera.rotateYaw(current.mouseDelta.x * 0.1f);
        }

        glm::vec3 moveVector(0.0f);

        // Check the state of all relevant keys
        if (current.keyStates[GLFW_KEY_W] == GLFW_PRESS || current.keyStates[GLFW_KEY_W] == GLFW_REPEAT) {
            moveVector.z += 0.01f;
        }
        if (current.keyStates[GLFW_KEY_S] == GLFW_PRESS || current.keyStates[GLFW_KEY_S] == GLFW_REPEAT) {
            moveVector.z -= 0.01f;
        }
        if (current.keyStates[GLFW_KEY_A] == GLFW_PRESS || current.keyStates[GLFW_KEY_A] == GLFW_REPEAT) {
            moveVector.x -= 0.01f;
        }
        if (current.keyStates[GLFW_KEY_D] == GLFW_PRESS || current.keyStates[GLFW_KEY_D] == GLFW_REPEAT) {
            moveVector.x += 0.01f;
        }

        // Apply the combined movement
        if (glm::length(moveVector) > 0) {
            renderer->activeCamera.moveCamera(moveVector);
        }

        // raycast on left click
        if (current.mouse_button == 0 && current.mouse_action == 1 && previous.mouse_action != 1) {
            int width = 0, height = 0;
            glfwGetWindowSize(renderer->getWindow(), &width, &height);
            float NDCx = (current.mousePos.x / width) * 2.0 - 1.0;
            float NDCy = (current.mousePos.y / height) * 2.0 - 1.0;
            glm::vec3 origin;
            glm::vec3 direction;
            renderer->activeCamera.rayFromScreenCoords(NDCx, NDCy, &origin, &direction);
            std::vector<uint32_t> hitNodes = renderer->rayCastNodes(origin, direction);
            if (!hitNodes.empty()) {
                renderer->selectNode(hitNodes.front());
            }
        }
        if (current.keyStates[GLFW_KEY_ESCAPE] == GLFW_PRESS && previous.keyStates[GLFW_KEY_ESCAPE] != GLFW_PRESS) {
            if (renderer->showFullPBR() == 1) {
                renderer->deSelectNode();
            }
        }
        if (current.keyStates[GLFW_KEY_V] == GLFW_PRESS && previous.keyStates[GLFW_KEY_V] != GLFW_PRESS) {
            renderer->toggleVsync();
        }
        if (current.keyStates[GLFW_KEY_C] == GLFW_PRESS && previous.keyStates[GLFW_KEY_C] != GLFW_PRESS) {
            renderer->showMap(SHOW_ALBEDO);
        }
        if (current.keyStates[GLFW_KEY_R] == GLFW_PRESS && previous.keyStates[GLFW_KEY_R] != GLFW_PRESS) {
            renderer->showMap(SHOW_ROUGHNESS);
        }
        if (current.keyStates[GLFW_KEY_M] == GLFW_PRESS && previous.keyStates[GLFW_KEY_M] != GLFW_PRESS) {
            renderer->showMap(SHOW_METALLIC);
        }
        if (current.keyStates[GLFW_KEY_N] == GLFW_PRESS && previous.keyStates[GLFW_KEY_N] != GLFW_PRESS) {
            renderer->showMap(SHOW_NORMAL);
        }
        if (current.keyStates[GLFW_KEY_F] == GLFW_PRESS && previous.keyStates[GLFW_KEY_F] != GLFW_PRESS) {
            renderer->toggleDepthBuffering();
        }

        current.mouseDelta = current.mousePos - previous.mousePos;
        previous = current;
    }

    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
        getInstance().current.mouse_button = button;
        getInstance().current.mouse_action = action;
    }

    static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) { getInstance().current.mousePos = glm::vec2(xpos, ypos); }

    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) { getInstance().current.keyStates[key] = action; }

    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {}

    void setRenderer(Renderer* pRenderer) { renderer = pRenderer; }

  private:
    InputState current, previous;
    Renderer* renderer;
};