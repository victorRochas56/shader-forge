#pragma once
#include <core.hpp>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

struct InputState {
    glm::vec2 mousePos = glm::vec2(0);
    glm::vec2 mouseDelta = glm::vec2(0);
    glm::vec2 scroll = glm::vec2(0);

    int mouse_button = 0;
    int mouse_action = 0;

    int key = 0; //make a vector for multiple simultaneous
    int action = 0;
};

class InputManager {
public:

    static InputManager& getInstance() {
        static InputManager instance;
        return instance;
    }

    void tickInputState(){
        if(current.mouse_button == 1 && current.mouse_action == 1){
            renderer->activeCamera.rotatePitch(current.mouseDelta.y * -0.1f);
            renderer->activeCamera.rotateYaw(current.mouseDelta.x * 0.1f);
        }
        if(current.key == GLFW_KEY_W && (current.action == GLFW_PRESS || current.action == GLFW_REPEAT ) ){
            renderer->activeCamera.moveCamera(glm::vec3(0,0,0.001f));
        }
        if(current.key == GLFW_KEY_S && (current.action == GLFW_PRESS || current.action == GLFW_REPEAT ) ){
            renderer->activeCamera.moveCamera(glm::vec3(0,0,-0.001f));
        }
        if(current.key == GLFW_KEY_A && (current.action == GLFW_PRESS || current.action == GLFW_REPEAT ) ){
            renderer->activeCamera.moveCamera(glm::vec3(-0.001f,0,0));
        }
        if(current.key == GLFW_KEY_D && (current.action == GLFW_PRESS || current.action == GLFW_REPEAT ) ){
            renderer->activeCamera.moveCamera(glm::vec3(0.001f,0,0));
        }

        current.mouseDelta = current.mousePos - previous.mousePos;
        previous = current;
    }

    static void mouse_button_callback(GLFWwindow* window,int button, int action, int mods)
    {
        getInstance().current.mouse_button = button;
        getInstance().current.mouse_action = action;
    }
 
    static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
    {
        getInstance().current.mousePos = glm::vec2(xpos, ypos);
    }

    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
    {
        getInstance().current.key = key;
        getInstance().current.action = action;
    }

    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
    {

    } 

    void setRenderer(Renderer* pRenderer){
        renderer = pRenderer;
    }

private:
    InputState current, previous;
    Renderer* renderer;

};