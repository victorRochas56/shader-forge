#pragma once
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Forward declarations
class Renderer;
class Scene;
class GUI;
struct MaterialEditorState;

/*
reads inputs through GLFW, stores this frame's and last frame's inputs in 2 InputState structs (current & previous)
*/

struct InputState {
    glm::vec2 mousePos = glm::vec2(0);
    glm::vec2 ndcMousePos = glm::vec2(0);
    glm::vec2 mouseDelta = glm::vec2(0);
    glm::vec2 normalizedMouseDelta = glm::vec2(0);
    // Wheel travel accumulated over this frame, then zeroed by endFrame.
    glm::vec2 scroll = glm::vec2(0);
    int mouse_button = 0;
    int mouse_action = 0;
    bool mouse_action_consumed = false;
    std::unordered_map<int, int> keyStates; // Track the state of each key
    std::unordered_set<int> consumedKeyStates; // track each key input that's been consumed
    // UTF-32 codepoints typed this frame, in order, cleared by endFrame alongside scroll
    std::vector<uint32_t> charQueue;
};

// A frame that receives more than this many characters is a stuck key or a paste bomb; the queue
// is drained per frame so the cap only has to cover realistic typing.
constexpr size_t MAX_CHARS_PER_FRAME = 64;

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
    static void endFrame();

    // return maxInt on consumed keyState
    static int getKeyState(int keyCode) {
        InputManager& inst = getInstance();
        if(inst.current.consumedKeyStates.contains(keyCode)) {
            return 0xFFFFFFFF;
        }
        return inst.current.keyStates[keyCode];
    }

    static void consumeKBInput(int keyCode); 
    static void consumeMouseInput(); 

    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
        getInstance().current.mouse_button = button;
        getInstance().current.mouse_action = action;
    }

    static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) { getInstance().current.mousePos = glm::vec2(xpos, ypos); }

    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) { getInstance().current.keyStates[key] = action; }

    // Accumulate rather than assign: GLFW can deliver several wheel events between two polls, and
    // a high-resolution trackpad delivers many small ones.
    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
        getInstance().current.scroll += glm::vec2(static_cast<float>(xoffset), static_cast<float>(yoffset));
    }

    static void char_callback(GLFWwindow* window, unsigned int codepoint) {
        InputState& s = getInstance().current;
        if (s.charQueue.size() < MAX_CHARS_PER_FRAME) s.charQueue.push_back(codepoint);
    }

    static void init(Renderer& renderer, Scene& scene) {
        getInstance().renderer = &renderer;
        getInstance().scene = &scene;
    }
    static void setMaterialEditorState(MaterialEditorState* state) { getInstance().materialEditorState = state; }
    // The custom GUI's mouse/keyboard claim gates the scene raycast and camera controls in tickInputState. Set once from App.
    static void setGUI(GUI* gui) { getInstance().gui = gui; }

    static InputState& getCurrentState(){
        auto& i = getInstance();
        return i.current;
    }

    static InputState& getPreviousState(){
        auto& i = getInstance();
        return i.previous;
    }

  private:
    InputState current, previous;
    Renderer* renderer;
    Scene* scene;
    MaterialEditorState* materialEditorState = nullptr;
    GUI* gui = nullptr;
};