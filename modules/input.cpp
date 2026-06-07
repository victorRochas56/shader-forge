#pragma once
#include "input.hpp"
#include "renderer.hpp"
#include "raycast.hpp"
#include "imgui.h"
#include "material_editor_state.hpp"

void InputManager::tickInputState() { //should be only called once in the main app loop
    InputManager& inst = getInstance();

    bool canMouse = !ImGui::GetIO().WantCaptureMouseUnlessPopupClose;
    bool canKB = !ImGui::GetIO().WantCaptureKeyboard;

    int w = 0, h = 0;
    glfwGetWindowSize(inst.renderer->gpu.getWindow(), &w, &h);
    inst.current.mouseDelta = inst.current.mousePos - inst.previous.mousePos;
    inst.current.normalizedMouseDelta = glm::vec2(inst.current.mouseDelta.x / w, inst.current.mouseDelta.y / h);
    inst.current.ndcMousePos = glm::vec2((inst.current.mousePos.x / w) * 2.0f - 1.0f, (inst.current.mousePos.y / h) * 2.0f - 1.0f);

    // mouse stuff
    if(canMouse) {
        //rotating viewport when holding right click
        if (inst.current.mouse_button == 1 && inst.current.mouse_action == 1) {
            inst.scene->activeCamera.rotatePitch(inst.current.mouseDelta.y * -0.1f);
            inst.scene->activeCamera.rotateYaw(inst.current.mouseDelta.x * 0.1f);
        }
        // raycast on left click for selection or material picking
        if (inst.current.mouse_button == 0 && inst.current.mouse_action == 1 && inst.previous.mouse_action != 1) {
            glm::vec3 origin;
            glm::vec3 direction;
            inst.scene->activeCamera.rayFromScreenCoords(inst.current.ndcMousePos.x, inst.current.ndcMousePos.y, origin, direction);

            if (inst.materialPickMode) {
                auto hit = Raycast::castMeshes(origin, direction, inst.scene->sceneGraph.getNodes(), inst.scene->sceneGraph.getLastNode(),
                                            inst.scene->assetManager.meshes);
                if (hit.nodeIndex != 0) {
                    auto& node = inst.scene->sceneGraph.getNodes()[hit.nodeIndex];
                    if (node.getMaterialIndex() != 0xFFFFFFFF) {
                        inst.pickedMaterialIndex = static_cast<int>(node.getMaterialIndex());
                    }
                }
                inst.materialPickMode = false;
            } else {
                std::vector<uint32_t> hitNodes = Raycast::castNodes(origin, direction, inst.scene->sceneGraph.getNodes(), inst.scene->sceneGraph.getLastNode());
                if (!hitNodes.empty()) {
                    inst.scene->sceneGraph.selectNode(hitNodes.front());
                }
            }
        }
    }

    glm::vec3 moveVector(0.0f);

    if(canKB) {
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
            inst.scene->activeCamera.moveCamera(moveVector);
        }

        if (inst.current.keyStates[GLFW_KEY_V] == GLFW_PRESS && inst.previous.keyStates[GLFW_KEY_V] != GLFW_PRESS) {
            inst.renderer->toggleVsync();
        }/*
        if (inst.current.keyStates[GLFW_KEY_C] == GLFW_PRESS && inst.previous.keyStates[GLFW_KEY_C] != GLFW_PRESS) {
            inst.renderer->showShadowMap();
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
        if (inst.current.keyStates[GLFW_KEY_F] == GLFW_PRESS && inst.previous.keyStates[GLFW_KEY_F] != GLFW_PRESS) {
            inst.renderer->toggleDepthView();
        }*/
    }
    
    if(canMouse && canKB) {
        // contextual menu on shift + right click
        if (inst.current.mouse_button == 1 && inst.current.mouse_action == 0 && inst.previous.mouse_action == 1 &&
            (inst.current.keyStates[GLFW_KEY_LEFT_SHIFT] == GLFW_REPEAT || inst.current.keyStates[GLFW_KEY_LEFT_SHIFT] == GLFW_PRESS)) {
                
            inst.contextMenuShown = !inst.contextMenuShown;
            inst.contextMenuPinX = inst.current.mousePos.x;
            inst.contextMenuPinY = inst.current.mousePos.y;
        }
    }

    if (inst.current.keyStates[GLFW_KEY_ESCAPE] == GLFW_PRESS && inst.previous.keyStates[GLFW_KEY_ESCAPE] != GLFW_PRESS) {
        ImGui::SetWindowFocus(NULL);
        if (inst.renderer->features.imageVis.imageIndex != 0xFFFFFFFF) {
            inst.renderer->features.imageVis.imageIndex = 0xFFFFFFFF;
        } else if (inst.materialPickMode) {
            inst.materialPickMode = false;
        } else if (inst.materialEditorState != nullptr && inst.materialEditorState->showEditor) {
            inst.materialEditorState->showEditor = false;
        } else if (inst.contextMenuShown) {
            inst.contextMenuShown = false;
        } else {
            inst.scene->sceneGraph.deSelectNode();
        }
    }
        
        
    inst.canMove = true;
}

void InputManager::endFrame() {
    InputManager& inst = getInstance();
    inst.previous = inst.current;
}
