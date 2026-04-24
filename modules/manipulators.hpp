#pragma once
#include "gizmo.hpp"
#include "scene_elements.hpp"
#include "utils.hpp"

namespace Manip {

bool wasClick = false;
int prevAxis = 0;
glm::quat baseRotation;
glm::vec3 dragStartVec;
glm::vec3 ringAxisWorld;
glm::vec3 ringCenterWorld;
glm::vec3 ringLocalAxis;

void showManipulator(Node& node, Camera& camera, const glm::vec2& ndcMousePos) {

    glm::vec3 origin;
    glm::vec3 direction;
    float cameraDist = glm::distance(camera.position, node.getWorldPosition());
    camera.rayFromScreenCoords(ndcMousePos.x, ndcMousePos.y, origin, direction);
    glm::vec3 worldPos = node.getWorldPosition();

    float tX = 0;
    glm::vec4 xCol(1, 0, 0, 1);
    float tY = 0;
    glm::vec4 yCol(0, 1, 0, 1);
    float tZ = 0;
    glm::vec4 zCol(0, 0, 1, 1);

    float radius = 0.1f * cameraDist;
    float thickness = 0.025f * cameraDist;
    intersectCircle(node.right(), worldPos, radius, thickness, origin, direction, tX);
    intersectCircle(node.up(), worldPos, radius, thickness, origin, direction, tY);
    intersectCircle(node.forward(), worldPos, radius, thickness, origin, direction, tZ);

    int axis = 0;
    if (wasClick) {
        axis = prevAxis;
        switch (axis) {
        case 1: xCol = {1, 1, 1, 1}; break;
        case 2: yCol = {1, 1, 1, 1}; break;
        case 3: zCol = {1, 1, 1, 1}; break;
        }
    }
    else {
        if (tX != 0 && (tY == 0 || tX < tY) && (tZ == 0 || tX < tZ)) {
            xCol = {1, 1, 1, 1};
            axis = 1;
        }
        if (tY != 0 && (tX == 0 || tY < tX) && (tZ == 0 || tY < tZ)) {
            yCol = {1, 1, 1, 1};
            axis = 2;
        }
        if (tZ != 0 && (tX == 0 || tZ < tX) && (tY == 0 || tZ < tY)) {
            zCol = {1, 1, 1, 1};
            axis = 3;
        }
    }
    Gizmos::drawCircle(node.getWorldPosition(), radius, node.right(), xCol);
    Gizmos::drawCircle(node.getWorldPosition(), radius, node.up(), yCol);
    Gizmos::drawCircle(node.getWorldPosition(), radius, node.forward(), zCol);

    const InputState& input = InputManager::getCurrentState();
    if (input.mouse_action == 1 && input.mouse_button == 0) {
        bool justPressed = !wasClick;
        wasClick = true;
        if (justPressed) {
            prevAxis = axis;
            if (prevAxis != 0) {
                switch (prevAxis) {
                case 1: ringAxisWorld = node.right();   ringLocalAxis = glm::vec3(1, 0, 0); break;
                case 2: ringAxisWorld = node.up();      ringLocalAxis = glm::vec3(0, 1, 0); break;
                case 3: ringAxisWorld = node.forward(); ringLocalAxis = glm::vec3(0, 0, 1); break;
                }
                ringCenterWorld = worldPos;
                baseRotation = node.relativeRotation;
                float denom = glm::dot(direction, ringAxisWorld);
                if (glm::abs(denom) > 1e-6f) {
                    float t = glm::dot(ringCenterWorld - origin, ringAxisWorld) / denom;
                    glm::vec3 v = (origin + t * direction) - ringCenterWorld;
                    if (glm::length(v) > 1e-6f) dragStartVec = glm::normalize(v);
                    else prevAxis = 0;
                } else {
                    prevAxis = 0;
                }
            }
        }
        if (prevAxis != 0) {
            float denom = glm::dot(direction, ringAxisWorld);
            if (glm::abs(denom) > 1e-6f) {
                float t = glm::dot(ringCenterWorld - origin, ringAxisWorld) / denom;
                glm::vec3 v = (origin + t * direction) - ringCenterWorld;
                if (glm::length(v) > 1e-6f) {
                    glm::vec3 currentVec = glm::normalize(v);
                    float angle = glm::atan(glm::dot(glm::cross(dragStartVec, currentVec), ringAxisWorld),
                                            glm::dot(dragStartVec, currentVec));
                    node.relativeRotation = glm::rotate(baseRotation, angle, ringLocalAxis);
                    node.transformDirty = true;
                }
            }
        }
    } else {
        wasClick = false;
        prevAxis = 0;
    }
    return;
}
} // namespace Manip