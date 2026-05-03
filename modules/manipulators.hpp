#pragma once
#include "gizmo.hpp"
#include "scene_elements.hpp"
#include "scene_graph.hpp"
#include "utils.hpp"

namespace Manip {

enum ManipAction {
    ROTATE,
    MOVE,
    SCALE
};

bool wasClick = false;
ManipAction prevAction;
int prevAxis = 0;
int hoveredAxis = 0;
ManipAction hoveredAction = ROTATE;
glm::quat baseRotation;
glm::vec3 basePosition;
glm::vec3 dragStartVec;
glm::vec3 ringAxisWorld;
glm::vec3 ringCenterWorld;
glm::vec3 rotLocalAxis;
glm::vec3 moveParentAxis;
float dragStartT = 0.0f;

void handleInput(Node& node, Camera& camera, const glm::vec2& ndcMousePos, SceneGraph& sceneGraph) {

    glm::vec3 origin;
    glm::vec3 direction;
    float cameraDist = glm::distance(camera.position, node.getWorldPosition());
    camera.rayFromScreenCoords(ndcMousePos.x, ndcMousePos.y, origin, direction);
    glm::vec3 worldPos = node.getWorldPosition();

    float rtX = 0, rtY = 0, rtZ = 0;
    float mtX = 0, mtY = 0, mtZ = 0;

    float radius = 0.1f * cameraDist;
    float length = 0.05f * cameraDist;
    float thickness = 0.015f * cameraDist;
    intersectCircle(node.right(), worldPos, radius, thickness, origin, direction, rtX);
    intersectCircle(node.up(), worldPos, radius, thickness, origin, direction, rtY);
    intersectCircle(node.forward(), worldPos, radius, thickness, origin, direction, rtZ);

    intersectCylinder(worldPos, node.right(),   thickness, length, origin, direction, mtX);
    intersectCylinder(worldPos, node.up(),      thickness, length, origin, direction, mtY);
    intersectCylinder(worldPos, node.forward(), thickness, length, origin, direction, mtZ);

    int axis = 0;
    if (wasClick) {
        axis = prevAxis;
        hoveredAxis = prevAxis;
        hoveredAction = prevAction;
    }
    else {
        float bestT = 0.0f;
        int bestAxis = 0;
        ManipAction bestAction = ManipAction::ROTATE;
        auto consider = [&](float t, int ax, ManipAction action) {
            if (t != 0.0f && (bestT == 0.0f || t < bestT)) {
                bestT = t;
                bestAxis = ax;
                bestAction = action;
            }
        };
        consider(rtX, 1, ManipAction::ROTATE);
        consider(rtY, 2, ManipAction::ROTATE);
        consider(rtZ, 3, ManipAction::ROTATE);
        consider(mtX, 1, ManipAction::MOVE);
        consider(mtY, 2, ManipAction::MOVE);
        consider(mtZ, 3, ManipAction::MOVE);

        hoveredAxis = bestAxis;
        hoveredAction = bestAction;
        if (bestAxis != 0) {
            axis = bestAxis;
            prevAction = bestAction;
        }
    }

    const InputState& input = InputManager::getCurrentState();
    if (input.mouse_action == 1 && input.mouse_button == 0) {
        bool justPressed = !wasClick;
        wasClick = true;
        if (justPressed) {
            prevAxis = axis;
            if (prevAxis != 0) {
                switch (prevAxis) {
                case 1: ringAxisWorld = node.right();   rotLocalAxis = glm::vec3(1, 0, 0); break;
                case 2: ringAxisWorld = node.up();      rotLocalAxis = glm::vec3(0, 1, 0); break;
                case 3: ringAxisWorld = node.forward(); rotLocalAxis = glm::vec3(0, 0, 1); break;
                }
                glm::quat parentWorldRot(1.0f, 0.0f, 0.0f, 0.0f);
                glm::vec3 parentWorldScale(1.0f);
                if (node.parentIndex != 0) {
                    Node& parent = sceneGraph.getNode(node.parentIndex);
                    parentWorldRot = parent.getWorldRotation();
                    parentWorldScale = parent.getWorldScale();
                }
                glm::vec3 axisInParent = glm::normalize(glm::inverse(parentWorldRot) * ringAxisWorld);
                moveParentAxis = glm::vec3(axisInParent.x / parentWorldScale.x,
                                           axisInParent.y / parentWorldScale.y,
                                           axisInParent.z / parentWorldScale.z);
                ringCenterWorld = worldPos;
                baseRotation = node.relativeRotation;
                basePosition = node.relativePosition;

                if (prevAction == ManipAction::ROTATE) {
                    float denom = glm::dot(direction, ringAxisWorld);
                    if (glm::abs(denom) > 1e-6f) {
                        float t = glm::dot(ringCenterWorld - origin, ringAxisWorld) / denom;
                        glm::vec3 v = (origin + t * direction) - ringCenterWorld;
                        if (glm::length(v) > 1e-6f) dragStartVec = glm::normalize(v);
                        else prevAxis = 0;
                    } else {
                        prevAxis = 0;
                    }
                } else if (prevAction == ManipAction::MOVE) {
                    float b = glm::dot(ringAxisWorld, direction);
                    float denom = 1.0f - b * b;
                    if (glm::abs(denom) > 1e-6f) {
                        glm::vec3 w0 = ringCenterWorld - origin;
                        float d = glm::dot(ringAxisWorld, w0);
                        float e = glm::dot(direction, w0);
                        dragStartT = (b * e - d) / denom;
                    } else {
                        prevAxis = 0;
                    }
                }
            }
        }
        if (prevAxis != 0) {
            if (prevAction == ManipAction::ROTATE) {
                float denom = glm::dot(direction, ringAxisWorld);
                if (glm::abs(denom) > 1e-6f) {
                    float t = glm::dot(ringCenterWorld - origin, ringAxisWorld) / denom;
                    glm::vec3 v = (origin + t * direction) - ringCenterWorld;
                    if (glm::length(v) > 1e-6f) {
                        glm::vec3 currentVec = glm::normalize(v);
                        float angle = glm::atan(glm::dot(glm::cross(dragStartVec, currentVec), ringAxisWorld),
                                                glm::dot(dragStartVec, currentVec));
                        node.relativeRotation = glm::rotate(baseRotation, angle, rotLocalAxis);
                        node.transformDirty = true;
                    }
                }
            } else if (prevAction == ManipAction::MOVE) {
                float b = glm::dot(ringAxisWorld, direction);
                float denom = 1.0f - b * b;
                if (glm::abs(denom) > 1e-6f) {
                    glm::vec3 w0 = ringCenterWorld - origin;
                    float d = glm::dot(ringAxisWorld, w0);
                    float e = glm::dot(direction, w0);
                    float currentT = (b * e - d) / denom;
                    float delta = currentT - dragStartT;
                    node.relativePosition = basePosition + delta * moveParentAxis;
                    node.transformDirty = true;
                }
            }
        }
    } else {
        wasClick = false;
        prevAxis = 0;
    }
}

void drawGizmos(Node& node, Camera& camera) {
    float cameraDist = glm::distance(camera.position, node.getWorldPosition());
    glm::vec3 worldPos = node.getWorldPosition();

    float radius = 0.1f * cameraDist;
    float length = 0.05f * cameraDist;

    glm::vec4 rxCol(1, 0, 0, 1);
    glm::vec4 ryCol(0, 1, 0, 1);
    glm::vec4 rzCol(0, 0, 1, 1);
    glm::vec4 mxCol(1, 0, 0, 1);
    glm::vec4 myCol(0, 1, 0, 1);
    glm::vec4 mzCol(0, 0, 1, 1);

    if (hoveredAxis != 0) {
        if (hoveredAction == ManipAction::ROTATE) {
            switch (hoveredAxis) {
            case 1: rxCol = {1, 1, 1, 1}; break;
            case 2: ryCol = {1, 1, 1, 1}; break;
            case 3: rzCol = {1, 1, 1, 1}; break;
            }
        } else if (hoveredAction == ManipAction::MOVE) {
            switch (hoveredAxis) {
            case 1: mxCol = {1, 1, 1, 1}; break;
            case 2: myCol = {1, 1, 1, 1}; break;
            case 3: mzCol = {1, 1, 1, 1}; break;
            }
        }
    }

    Gizmos::drawCircle(worldPos, radius, node.right(),   rxCol);
    Gizmos::drawCircle(worldPos, radius, node.up(),      ryCol);
    Gizmos::drawCircle(worldPos, radius, node.forward(), rzCol);

    Gizmos::drawLine(worldPos, worldPos + node.right()   * length, mxCol);
    Gizmos::drawLine(worldPos, worldPos + node.up()      * length, myCol);
    Gizmos::drawLine(worldPos, worldPos + node.forward() * length, mzCol);
}

} // namespace Manip
