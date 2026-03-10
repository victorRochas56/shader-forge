#pragma once

#include "descriptor_sets.hpp"
#include "resources.hpp"

#define GLM_FORCE_RADIANS
#define GLM_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

struct PairVec3Hash {
    size_t operator()(const std::pair<glm::vec3,glm::vec3>& p) const {
        size_t h1 = std::hash<glm::vec3>{}(p.first);
        size_t h2 = std::hash<glm::vec3>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

/*
gizmo class allows drawing of colored axes, lines etc
it handles this in "immediate mode", meaning it can be called
to draw gizmos anywhere in the code and it will defer that
to draw all of them in one call during rendering, before clearing itself again

this means no gizmos have persistent state
*/

class Gizmos {
  public:
    Gizmos(const Gizmos&) = delete;
    Gizmos& operator=(const Gizmos&) = delete;

    static void init(uint32_t maxLinesCount, DescriptorSet* pDescriptorSet) {
        auto& g = instance();
        g.descriptorSet = pDescriptorSet;
        g.lineBufferIndex = pDescriptorSet->createFixedBuffer<LineVertex>(maxLinesCount);
    }

    static uint32_t getVertexCount() { return instance().activeVertexCount; }

    static void drawLine(glm::vec3 start, glm::vec3 end, glm::vec4 color, bool noDiscard = false) {
        auto& g = instance();
        if(noDiscard) {
            if(!g.noDiscardLines.contains(std::make_pair(start,end)))
                g.noDiscardLines[std::make_pair(start,end)] = {.startPoint = start, .endPoint = end, .color = color};
        }
        g.addLineToBuffer({.startPoint = start, .endPoint = end, .color = color});
    }

    static void drawLine(Line line, bool noDiscard = false) {
        auto& g = instance();
        if(noDiscard) {
            if(!g.noDiscardLines.contains(std::make_pair(line.startPoint,line.endPoint)))
                g.noDiscardLines[std::make_pair(line.startPoint,line.endPoint)] = line;
        }
        g.addLineToBuffer(line);
    }

    static void drawAxes(glm::mat4 transform, float length) {
        auto& g = instance();
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

        g.addLineToBuffer(lineX);
        g.addLineToBuffer(lineY);
        g.addLineToBuffer(lineZ);
    }

    static void drawGrid(glm::vec3 origin, glm::vec3 normal, float spacing) {}
    static void drawWireFrame(Mesh mesh) {}
    static void drawWireFrame(SubMesh mesh) {}

    static void clearLineBuffer() {
        auto& g = instance();
        g.descriptorSet->clearFixedBuffer(g.lineBufferIndex);
        g.activeVertexCount = 0;
    }

    static auto& getNoDiscardLines() { return instance().noDiscardLines; }

  private:
    Gizmos() = default;

    static Gizmos& instance() {
        static Gizmos g;
        return g;
    }

    std::unordered_map<std::pair<glm::vec3,glm::vec3>,Line,PairVec3Hash> noDiscardLines;
    DescriptorSet* descriptorSet = nullptr;
    uint32_t lineBufferIndex = 0;
    uint32_t activeVertexCount = 0;

    void addLineToBuffer(Line line) {
        LineVertex vert1 = {.color = line.color, .position = glm::vec4(line.startPoint, 1.0)};
        LineVertex vert2 = {.color = line.color, .position = glm::vec4(line.endPoint, 1.0)};
        descriptorSet->allocateFixedBuffer<LineVertex>(lineBufferIndex, vert1);
        descriptorSet->allocateFixedBuffer<LineVertex>(lineBufferIndex, vert2);
        activeVertexCount += 2;
    }
};