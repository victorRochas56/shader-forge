#pragma once

#include "descriptor_sets.hpp"
#include "resources.hpp"
#include "structs.hpp"

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

    static void init(uint32_t maxLinesCount, DescriptorSet* pDescriptorSet, uint32_t sdfBufferIndex) {
        auto& g = instance();
        g.descriptorSet = pDescriptorSet;
        g.lineBufferIndex = pDescriptorSet->createFixedBuffer<LineVertex>(maxLinesCount);
        g.sdfBufferIndex = sdfBufferIndex;
    }

    static uint32_t getVertexCount() { return instance().activeVertexCount; }
    static vk::DeviceAddress getLineBufferAddress() { return instance().descriptorSet->getFixedBuffers()[instance().lineBufferIndex]->address; }

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

    static void drawBox(glm::vec3 min, glm::vec3 max, glm::vec4 color){
        auto& g = instance();
        float spanX = max.x - min.x;
        float spanY = max.y - min.y;
        float spanZ = max.z - min.z;

        Line top0 = {.startPoint = min + glm::vec3(0,spanY,0), .endPoint = min + glm::vec3(spanX,spanY,0), .color = color};
        Line top1 = {.startPoint = min + glm::vec3(spanX,spanY,0), .endPoint = min + glm::vec3(spanX,spanY,spanZ), .color = color};
        Line top2 = {.startPoint = min + glm::vec3(0,spanY,0), .endPoint = min + glm::vec3(0,spanY,spanZ), .color = color};
        Line top3 = {.startPoint = min + glm::vec3(0,spanY,spanZ), .endPoint = min + glm::vec3(spanX,spanY,spanZ), .color = color};

        Line bottom0 = {.startPoint = min, .endPoint = min + glm::vec3(spanX,0,0), .color = color};
        Line bottom1 = {.startPoint = min + glm::vec3(spanX,0,0), .endPoint = min + glm::vec3(spanX,0,spanZ), .color = color};
        Line bottom2 = {.startPoint = min, .endPoint = min + glm::vec3(0,0,spanZ), .color = color};
        Line bottom3 = {.startPoint = min + glm::vec3(0,0,spanZ), .endPoint = min + glm::vec3(spanX,0,spanZ), .color = color};

        Line pillar0 = {.startPoint = min + glm::vec3(spanX,0,0), .endPoint = min + glm::vec3(spanX,spanY,0), .color = color};
        Line pillar1 = {.startPoint = min + glm::vec3(0,0,spanZ), .endPoint = min + glm::vec3(0,spanY,spanZ), .color = color};
        Line pillar2 = {.startPoint = min + glm::vec3(spanX,0,spanZ), .endPoint = min + glm::vec3(spanX,spanY,spanZ), .color = color};
        Line pillar3 = {.startPoint = min + glm::vec3(0,0,0), .endPoint = min + glm::vec3(0,spanY,0), .color = color};
        
        g.addLineToBuffer(top0);
        g.addLineToBuffer(top1);
        g.addLineToBuffer(top2);
        g.addLineToBuffer(top3);
        g.addLineToBuffer(bottom0);
        g.addLineToBuffer(bottom1);
        g.addLineToBuffer(bottom2);
        g.addLineToBuffer(bottom3);
        g.addLineToBuffer(pillar0);
        g.addLineToBuffer(pillar1);
        g.addLineToBuffer(pillar2);
        g.addLineToBuffer(pillar3);
    }

    static void drawFrustum(const glm::mat4& viewProjection, glm::vec4 color) {
        auto& g = instance();
        glm::mat4 inv = glm::inverse(viewProjection);

        // Unproject 8 NDC corners (Vulkan: z in [0,1])
        glm::vec3 corners[8];
        int idx = 0;
        for (int z = 0; z < 2; ++z)
            for (int y = 0; y < 2; ++y)
                for (int x = 0; x < 2; ++x) {
                    glm::vec4 c = inv * glm::vec4(2.f * x - 1.f, 2.f * y - 1.f, static_cast<float>(z), 1.f);
                    corners[idx++] = glm::vec3(c / c.w);
                }

        // Near face edges (z=0: indices 0-3)
        g.addLineToBuffer({.startPoint = corners[0], .endPoint = corners[1], .color = color});
        g.addLineToBuffer({.startPoint = corners[0], .endPoint = corners[2], .color = color});
        g.addLineToBuffer({.startPoint = corners[1], .endPoint = corners[3], .color = color});
        g.addLineToBuffer({.startPoint = corners[2], .endPoint = corners[3], .color = color});

        // Far face edges (z=1: indices 4-7)
        g.addLineToBuffer({.startPoint = corners[4], .endPoint = corners[5], .color = color});
        g.addLineToBuffer({.startPoint = corners[4], .endPoint = corners[6], .color = color});
        g.addLineToBuffer({.startPoint = corners[5], .endPoint = corners[7], .color = color});
        g.addLineToBuffer({.startPoint = corners[6], .endPoint = corners[7], .color = color});

        // Connecting edges (near to far)
        g.addLineToBuffer({.startPoint = corners[0], .endPoint = corners[4], .color = color});
        g.addLineToBuffer({.startPoint = corners[1], .endPoint = corners[5], .color = color});
        g.addLineToBuffer({.startPoint = corners[2], .endPoint = corners[6], .color = color});
        g.addLineToBuffer({.startPoint = corners[3], .endPoint = corners[7], .color = color});
    }

    static void drawSphere(glm::vec3 center, float radius, glm::vec4 color, int segments = 32) {
        auto& g = instance();
        float step = glm::two_pi<float>() / static_cast<float>(segments);
        for (int i = 0; i < segments; i++) {
            float a0 = step * i;
            float a1 = step * (i + 1);
            float c0 = glm::cos(a0), s0 = glm::sin(a0);
            float c1 = glm::cos(a1), s1 = glm::sin(a1);
            // XY ring
            g.addLineToBuffer({.startPoint = center + glm::vec3(c0, s0, 0.0f) * radius,
                                .endPoint   = center + glm::vec3(c1, s1, 0.0f) * radius, .color = color});
            // XZ ring
            g.addLineToBuffer({.startPoint = center + glm::vec3(c0, 0.0f, s0) * radius,
                                .endPoint   = center + glm::vec3(c1, 0.0f, s1) * radius, .color = color});
            // YZ ring
            g.addLineToBuffer({.startPoint = center + glm::vec3(0.0f, c0, s0) * radius,
                                .endPoint   = center + glm::vec3(0.0f, c1, s1) * radius, .color = color});
        }
    }

    static void drawGrid(glm::vec3 origin, glm::vec3 normal, float spacing) {}
    static void drawWireFrame(Mesh mesh) {}

    static void drawSDFSphere(glm::vec3 origin, float radius, glm::vec4 color) {

        auto& g = instance();
        glm::mat4 worldTransform = makeTransform(origin);
        g.descriptorSet->allocateFixedBuffer(g.sdfBufferIndex, SDF {.worldTransform = worldTransform,
                                                                    .invWorldTransform = glm::inverse(worldTransform),
                                                                    .color = color,
                                                                    .type = SDFType::SPHERE,
                                                                    .radius = radius,
                                                                    .height = 0});
    }

    static void drawSDFCylinder(glm::vec3 origin, glm::vec3 axis,float radius, float height, glm::vec4 color) {

        auto& g = instance();
        glm::vec3 from = glm::vec3(0, 1, 0);
        glm::quat rotation;
        float d = glm::dot(from, axis);
        if (d > 0.9999f) {
            rotation = glm::quat(1, 0, 0, 0);
        } else if (d < -0.9999f) {
            rotation = glm::angleAxis(glm::pi<float>(), glm::vec3(1, 0, 0));
        } else {
            glm::vec3 cross = glm::cross(from, axis);
            rotation = glm::normalize(glm::quat(1.0f + d, cross.x, cross.y, cross.z));
        }
        glm::mat4 worldTransform = makeTransform(origin,rotation);
        g.descriptorSet->allocateFixedBuffer(g.sdfBufferIndex, SDF {.worldTransform = worldTransform,
                                                                    .invWorldTransform = glm::inverse(worldTransform),
                                                                    .color = color,
                                                                    .type = SDFType::CYLINDER,
                                                                    .radius = radius,
                                                                    .height = height});
    }

    static void drawSDFPyramid(glm::vec3 origin, glm::vec3 axis, float height, glm::vec4 color) {
        auto& g = instance();
        glm::vec3 from = glm::vec3(0, 1, 0);
        glm::quat rotation;
        float d = glm::dot(from, axis);
        if (d > 0.9999f) {
            rotation = glm::quat(1, 0, 0, 0);
        } else if (d < -0.9999f) {
            rotation = glm::angleAxis(glm::pi<float>(), glm::vec3(1, 0, 0));
        } else {
            glm::vec3 cross = glm::cross(from, axis);
            rotation = glm::normalize(glm::quat(1.0f + d, cross.x, cross.y, cross.z));
        }
        glm::mat4 worldTransform = makeTransform(origin,rotation);
        g.descriptorSet->allocateFixedBuffer(g.sdfBufferIndex, SDF {.worldTransform = worldTransform,
                                                                    .invWorldTransform = glm::inverse(worldTransform),
                                                                    .color = color,
                                                                    .type = SDFType::PYRAMID,
                                                                    .radius = 0,
                                                                    .height = height});
    }

    static void drawSDFCone(glm::vec3 origin, glm::quat rotation, float radius, float height, glm::vec4 color) {
        auto& g = instance();
        glm::mat4 worldTransform = makeTransform(origin, rotation);
        g.descriptorSet->allocateFixedBuffer(g.sdfBufferIndex, SDF {.worldTransform = worldTransform,
                                                                    .invWorldTransform = glm::inverse(worldTransform),
                                                                    .color = color,
                                                                    .type = SDFType::CONE,
                                                                    .radius = radius,
                                                                    .height = height});
    }

    static void drawSDFArrow(glm::vec3 origin, glm::vec3 axis, float radius, float length, glm::vec4 color) {
        auto& g = instance();
        glm::vec3 from = glm::vec3(0, 1, 0);
        glm::quat rotation;
        float d = glm::dot(from, axis);
        if (d > 0.9999f) {
            rotation = glm::quat(1, 0, 0, 0);
        } else if (d < -0.9999f) {
            rotation = glm::angleAxis(glm::pi<float>(), glm::vec3(1, 0, 0));
        } else {
            glm::vec3 cross = glm::cross(from, axis);
            rotation = glm::normalize(glm::quat(1.0f + d, cross.x, cross.y, cross.z));
        }
        drawSDFCone(origin + axis * 0.5f * length, rotation, radius * 1.5f, length * 0.25f, color);
        drawSDFCylinder(origin, axis, radius, length * 0.5f, color);
    }

    static void clearLineBuffer() {
        auto& g = instance();
        g.descriptorSet->clearFixedBuffer(g.lineBufferIndex);
        g.activeVertexCount = 0;
    }

    static void clearSDFBuffer() {
        auto& g = instance();
        g.descriptorSet->clearFixedBuffer(g.sdfBufferIndex);
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
    uint32_t lineBufferIndex = 0xFFFFFFFF;
    uint32_t activeVertexCount = 0;

    uint32_t sdfBufferIndex = 0xFFFFFFFF;

    void addLineToBuffer(Line line) {
        LineVertex vert1 = {.color = line.color, .position = glm::vec4(line.startPoint, 1.0)};
        LineVertex vert2 = {.color = line.color, .position = glm::vec4(line.endPoint, 1.0)};
        descriptorSet->allocateFixedBuffer<LineVertex>(lineBufferIndex, vert1);
        descriptorSet->allocateFixedBuffer<LineVertex>(lineBufferIndex, vert2);
        activeVertexCount += 2;
    }
};