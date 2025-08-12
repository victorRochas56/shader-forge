#pragma once
#include <core.hpp>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
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

        current.mouseDelta = current.mousePos - previous.mousePos;
        previous = current;
    }

    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
        getInstance().current.mouse_button = button;
        getInstance().current.mouse_action = action;
    }

    static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
        getInstance().current.mousePos = glm::vec2(xpos, ypos);
    }

    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
        getInstance().current.keyStates[key] = action;
    }

    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
        // Handle scroll if needed
    }

    void setRenderer(Renderer* pRenderer) {
        renderer = pRenderer;
    }

private:
    InputState current, previous;
    Renderer* renderer;
};