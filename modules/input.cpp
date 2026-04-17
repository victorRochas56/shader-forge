#include "input.hpp"
#include "renderer.hpp"
#include "raycast.hpp"
#include "material_editor_state.hpp"

void InputManager::tickInputState() { //should be only called once in the main app loop
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

    // raycast on left click for selection or material picking
    if (inst.current.mouse_button == 0 && inst.current.mouse_action == 1 && inst.previous.mouse_action != 1) {
        int width = 0, height = 0;
        glfwGetWindowSize(inst.renderer->getWindow(), &width, &height);
        float NDCx = (inst.current.mousePos.x / width) * 2.0 - 1.0;
        float NDCy = (inst.current.mousePos.y / height) * 2.0 - 1.0;
        glm::vec3 origin;
        glm::vec3 direction;
        inst.renderer->activeCamera.rayFromScreenCoords(NDCx, NDCy, &origin, &direction);

        if (inst.materialPickMode) {
            auto hit = Raycast::castMeshes(origin, direction, inst.renderer->sceneGraph.getNodes(), inst.renderer->sceneGraph.getLastNode(),
                                           inst.renderer->assetManager.meshes);
            if (hit.nodeIndex != 0) {
                auto& node = inst.renderer->sceneGraph.getNodes()[hit.nodeIndex];
                if (node.getMaterialIndex() != 0xFFFFFFFF) {
                    inst.pickedMaterialIndex = static_cast<int>(node.getMaterialIndex());
                }
            }
            inst.materialPickMode = false;
        } else {
            std::vector<uint32_t> hitNodes = Raycast::castNodes(origin, direction, inst.renderer->sceneGraph.getNodes(), inst.renderer->sceneGraph.getLastNode());
            if (!hitNodes.empty()) {
                inst.renderer->sceneGraph.selectNode(hitNodes.front());
            }
        }
    }
    // contextual menu on shift + right click
    if (inst.current.mouse_button == 1 && inst.current.mouse_action == 0 && inst.previous.mouse_action == 1 &&
        (inst.current.keyStates[GLFW_KEY_LEFT_SHIFT] == GLFW_REPEAT || inst.current.keyStates[GLFW_KEY_LEFT_SHIFT] == GLFW_PRESS)) {

        inst.contextMenuShown = !inst.contextMenuShown;
        inst.contextMenuPinX = inst.current.mousePos.x;
        inst.contextMenuPinY = inst.current.mousePos.y;
    }

    if (inst.current.keyStates[GLFW_KEY_ESCAPE] == GLFW_PRESS && inst.previous.keyStates[GLFW_KEY_ESCAPE] != GLFW_PRESS) {
        if (inst.renderer->features.imageVis.imageIndex != 0xFFFFFFFF) {
            inst.renderer->features.imageVis.imageIndex = 0xFFFFFFFF;
        } else if (inst.materialPickMode) {
            inst.materialPickMode = false;
        } else if (inst.materialEditorState != nullptr && inst.materialEditorState->showEditor) {
            inst.materialEditorState->showEditor = false;
        } else {
            inst.renderer->sceneGraph.deSelectNode();
        }
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
    inst.canMove = true;
    inst.current.mouseDelta = inst.current.mousePos - inst.previous.mousePos;
    int w;
    int h;
    glfwGetWindowSize(inst.renderer->getWindow(),&w,&h);
    inst.current.normalizedMouseDelta = glm::vec2(inst.current.mouseDelta.x / w, inst.current.mouseDelta.y / h);
    inst.previous = inst.current;
}
