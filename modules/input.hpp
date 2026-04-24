#pragma once
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <unordered_map>

// Forward declarations
class Renderer;
struct MaterialEditorState;

/*
reads inputs through GLFW, stores this frame's and last frame's inputs in 2 InputState structs (current & previous)
*/

struct InputState {
    glm::vec2 mousePos = glm::vec2(0);
    glm::vec2 ndcMousePos = glm::vec2(0);
    glm::vec2 mouseDelta = glm::vec2(0);
    glm::vec2 normalizedMouseDelta = glm::vec2(0);
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
    bool materialPickMode = false;
    int pickedMaterialIndex = -1; // set when a mesh is clicked in pick mode

    static InputManager& getInstance() {
        static InputManager instance;
        return instance;
    }

    static void tickInputState();

    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
        getInstance().current.mouse_button = button;
        getInstance().current.mouse_action = action;
    }

    static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) { getInstance().current.mousePos = glm::vec2(xpos, ypos); }

    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) { getInstance().current.keyStates[key] = action; }

    static void scrocll_callback(GLFWwindow* window, double xoffset, double yoffset) {}

    static void setRenderer(Renderer* pRenderer) { getInstance().renderer = pRenderer; }
    static void setMaterialEditorState(MaterialEditorState* state) { getInstance().materialEditorState = state; }

    static const InputState& getCurrentState(){
        auto& i = getInstance();
        return i.current;
    }

    static const InputState& getPreviousState(){
        auto& i = getInstance();
        return i.previous;
    }

  private:
    InputState current, previous;
    Renderer* renderer;
    MaterialEditorState* materialEditorState = nullptr;
};