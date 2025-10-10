#pragma once

#include "descriptor_sets.hpp"
#include "resources.hpp"

#define GLM_FORCE_RADIANS
#define GLM_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

class Gizmos {
  public:
    Gizmos(uint32_t maxLinesCount, DescriptorSet* pDescriptorSet) : descriptorSet(pDescriptorSet) { lineBufferIndex = descriptorSet->createFixedBuffer<LineVertex>(maxLinesCount); }

    uint32_t getVertexCount() { return activeVertexCount; }

    void drawLine(glm::vec3 start, glm::vec3 end, glm::vec4 color) { addLineToBuffer({.startPoint = start, .endPoint = end, .color = color}); }
    void drawLine(Line line) { addLineToBuffer(line); }

    void drawAxes(glm::mat4 transform, float length) {
        glm::vec3 translation;
        glm::vec3 scale;
        glm::quat rotation;
        decomposeTransform(transform, translation, rotation, scale);
        Line lineX = {.startPoint = glm::vec3(0, 0, 0), .endPoint = glm::vec3(length / scale.x, 0, 0), .color = glm::vec4(1, 0, 0, 1)};
        Line lineY = {.startPoint = glm::vec3(0, 0, 0), .endPoint = glm::vec3(0, length / scale.y, 0), .color = glm::vec4(0, 1, 0, 1)};
        Line lineZ = {.startPoint = glm::vec3(0, 0, 0), .endPoint = glm::vec3(0, 0, length / scale.z), .color = glm::vec4(0, 0, 1, 1)};

        lineX.startPoint = glm::vec3(transform * glm::vec4(lineX.startPoint, 1.0f));
        lineX.endPoint = glm::vec3(transform * glm::vec4(lineX.endPoint, 1.0f));
        lineY.startPoint = glm::vec3(transform * glm::vec4(lineY.startPoint, 1.0f));
        lineY.endPoint = glm::vec3(transform * glm::vec4(lineY.endPoint, 1.0f));
        lineZ.startPoint = glm::vec3(transform * glm::vec4(lineZ.startPoint, 1.0f));
        lineZ.endPoint = glm::vec3(transform * glm::vec4(lineZ.endPoint, 1.0f));

        addLineToBuffer(lineX);
        addLineToBuffer(lineY);
        addLineToBuffer(lineZ);
    }
    void drawGrid(glm::vec3 origin, glm::vec3 normal, float spacing) {}

    void drawWireFrame(Mesh mesh) {}
    void drawWireFrame(SubMesh mesh) {}

    void clearLineBuffer() {
        descriptorSet->clearFixedBuffer(lineBufferIndex);
        activeVertexCount = 0;
    }

  private:
    DescriptorSet* descriptorSet = nullptr;
    uint32_t lineBufferIndex;
    uint32_t activeVertexCount = 0;

    void addLineToBuffer(Line line) {
        LineVertex vert1 = {.color = line.color, .position = glm::vec4(line.startPoint, 1.0)};
        LineVertex vert2 = {.color = line.color, .position = glm::vec4(line.endPoint, 1.0)};
        descriptorSet->allocateFixedBuffer<LineVertex>(lineBufferIndex, vert1);
        descriptorSet->allocateFixedBuffer<LineVertex>(lineBufferIndex, vert2);
        activeVertexCount += 2;
    }
};