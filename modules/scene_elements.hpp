#pragma once
#ifndef VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1 // for raii
#endif
#ifndef VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_NO_CONSTRUCTORS 1 // for structs constructors
#endif

#include <string>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include "constants.hpp"
#include "descriptor_sets.hpp"
#include "structs.hpp"

#define GLM_FORCE_RADIANS
#define GLM_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

class Renderer; // forward declaration for free functions below
class SceneGraph;

class Node { // need to be able to hide nodes from tree for internal logic
  public:
    Node(const Node& node) = default;

    Node(uint32_t arrayIndex, bool internal = true, uint32_t parentIndex = 0, glm::vec3 position = glm::vec3(0.0f), glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
         glm::vec3 scale = glm::vec3(1.0f), std::string nodeName = "empty");
    glm::vec3 getWorldPosition() { return glm::vec3(worldTransform[3]); }
    glm::vec3 getRelativePosition() { return relativePosition; }
    glm::mat4 getTransform() { return worldTransform; }
    glm::vec3 getWorldScale() {
        glm::vec3 scale;
        scale.x = glm::length(glm::vec3(worldTransform[0]));
        scale.y = glm::length(glm::vec3(worldTransform[1]));
        scale.z = glm::length(glm::vec3(worldTransform[2]));
        return scale;
    }
    glm::vec3 getRelativeScale() { return relativeScale; }

    glm::quat getWorldRotation() {
        glm::mat3 rotationMatrix;
        glm::vec3 scale = getWorldScale();
        rotationMatrix[0] = glm::vec3(worldTransform[0]) / scale.x;
        rotationMatrix[1] = glm::vec3(worldTransform[1]) / scale.y;
        rotationMatrix[2] = glm::vec3(worldTransform[2]) / scale.z;
        // Convert to quaternion
        glm::quat rotation = glm::quat_cast(rotationMatrix);
        return rotation;
    }
    glm::quat getRelativeRotation() { return relativeRotation; }

    glm::vec3 right()   { return  glm::normalize(glm::vec3(worldTransform[0])); }
    glm::vec3 left()    { return -glm::normalize(glm::vec3(worldTransform[0])); }
    glm::vec3 up()      { return  glm::normalize(glm::vec3(worldTransform[1])); }
    glm::vec3 down()    { return -glm::normalize(glm::vec3(worldTransform[1])); }
    glm::vec3 forward() { return  glm::normalize(glm::vec3(worldTransform[2])); }
    glm::vec3 back()    { return -glm::normalize(glm::vec3(worldTransform[2])); }
    glm::vec3 getWorldRotationEuler() { return glm::eulerAngles(getWorldRotation()); }
    glm::vec3 getRelativeRotationEuler() { return relativeRotationEuler; }

    uint32_t getIndex() { return nodeIndex; }
    uint32_t getModelMatrixIndex() { return modelMatrixIndices[0]; } 

    uint32_t getMeshIndex() { return meshIndex; }
    uint32_t getLightIndex() { return lightIndex; }
    uint32_t getMaterialIndex() { return materialIndex; }

    // Bounding box getters for frustum culling
    glm::vec3 getBoundingBoxMin() const { return boundingBoxMin; }
    glm::vec3 getBoundingBoxMax() const { return boundingBoxMax; }
    bool isBoundingBoxValid() const { return boundingBoxValid; }

    void toggleWireframe() { showWireframe = !showWireframe; }

    std::string name = "empty";

    uint32_t nodeIndex;
    // Tree structure (indices into SceneGraph::nodes, 0 = none/invalid)
    uint32_t parentIndex = 0;
    uint32_t firstChild = 0;
    uint32_t nextSibling = 0;

    glm::vec3 relativePosition;
    glm::vec3 relativeScale;
    glm::quat relativeRotation;
    glm::vec3 relativeRotationEuler;
    glm::vec3 worldRotationEuler;
    glm::mat4 worldTransform; // aka model matrix
    std::array<uint32_t, MAX_FRAMES_IN_FLIGHT> modelMatrixIndices;
    glm::mat4 localTransform; // relative to parent

    uint32_t meshIndex = MAX_MESHES;
    uint32_t materialIndex = 0xFFFFFFFF;
    uint32_t lightIndex = MAX_LIGHTS;
    // Volumes are streamed and keyed by node index (like billboards), so there is no per-node
    // volume slot to store — presence is `scene.volumes.contains(nodeIndex)`.
    uint32_t particleIndex = 0xFFFFFFFF;

    // Bounding box for frustum culling (in world space)
    glm::vec3 boundingBoxMin = glm::vec3(0.0f);
    glm::vec3 boundingBoxMax = glm::vec3(0.0f);
    bool boundingBoxValid = false;
    bool internal = false;
    bool isSelected = false;
    bool alive = true;
    bool transformDirty = false;
    // Countdown mirroring Light::gpuDirtyFrames: set to MAX_FRAMES_IN_FLIGHT when the world
    // transform changes so the model-matrix write fans out one frame-in-flight slice per frame
    // (each written post-fence) instead of stomping every slice at once. Driven by
    // SceneGraph::uploadDirtyTransforms.
    uint32_t gpuDirtyFrames = 0;
    bool showWireframe = false;
};

struct Billboard {
    uint32_t textureIndex;
    uint32_t nodeIndex;
    bool hidden = false;
    float clipThreshold = 0.5f;
    bool screenSpaceSize = false;
    float size = 0.1f;
    bool depthTest = false;
    
    GPUBillboard toGPU(glm::vec3& position) {
        GPUBillboard out;
        out.position = position;
        out.clipThreshold = clipThreshold;
        out.textureIndex = textureIndex;
        out.screenSpace = screenSpaceSize ? 1 : 0;
        out.size = size;
        out.alphaBlend = false;
        return out;
    }
};

struct Cascade {
    glm::mat4 lightSpaceMatrix;
    uint32_t shadowAtlasTile;
    glm::vec4 shadowAtlasUVRange = glm::vec4(0);
    float splitDistance = 0.0f;
    float texelSize = 0.0f;
    float worldTexelSize = 0.0f;
};

struct PointShadowFace {
    glm::mat4 lightSpaceMatrix;
    uint32_t shadowAtlasTile;
    glm::vec4 shadowAtlasUVRange = glm::vec4(0);
};

struct Light {
    LightType type = LightType::Point;
    uint32_t modelMatrixIndex = 0;
    uint32_t nodeIndex = 0;
    float range = 10.0f;
    float intensity = 1.0f;
    uint32_t shadowResolution = DEFAULT_SHADOW_RESOLUTION;
    glm::vec4 color = glm::vec4(0, 0, 0, 1);
    glm::mat4 lightSpaceMatrix;
    glm::vec3 direction = glm::vec3(1, 0, 0);
    int castsShadows = 0;
    int showCascades = 0;
    uint32_t numCascades = 3;
    std::array<Cascade, 3> cascades;
    std::array<PointShadowFace,6> cubeMapIndices;
    bool shadowDirty = true;
    // Countdown for fanning out a GPULight write across every frame-in-flight slice of the
    // per-frame light buffer. Set to MAX_FRAMES_IN_FLIGHT whenever any field feeding
    // Light::toGPU changes (position, direction, range, matrices, color, flags, etc.).
    // The per-frame renderer loop writes the current frame's slice and decrements.
    uint32_t gpuDirtyFrames = 0;
    // Point-light only: node indices whose world AABB currently overlaps this light's sphere.
    // Maintained exclusively by LightInfluence — do not mutate elsewhere.
    std::unordered_set<uint32_t> influencedNodes;

    GPULight toGPU(glm::vec3 lightPos, glm::vec3 lightDir) const {
        GPULight gpu;
        gpu.type = type;
        gpu.position = lightPos;
        gpu.direction = lightDir;
        gpu.range = range;
        gpu.intensity = intensity;
        gpu.color = color;
        gpu.castsShadows = castsShadows;
        gpu.showCascades = showCascades;
        gpu.numCascades = numCascades;
        gpu.shadowResolution = shadowResolution;
        for (uint32_t i = 0; i < 3; i++) {
            gpu.cascades[i].lightSpaceMatrix = cascades[i].lightSpaceMatrix;
            gpu.cascades[i].shadowAtlasRange = cascades[i].shadowAtlasUVRange;
            gpu.cascades[i].splitDistance = cascades[i].splitDistance;
            gpu.cascades[i].texelSize = cascades[i].texelSize;
            gpu.cascades[i].worldTexelSize = cascades[i].worldTexelSize;
        }
        for (uint32_t i = 0; i < 6; i++) {
            gpu.pointFaces[i].lightSpaceMatrix = cubeMapIndices[i].lightSpaceMatrix;
            gpu.pointFaces[i].shadowAtlasRange = cubeMapIndices[i].shadowAtlasUVRange;
        }
        return gpu;
    }

    bool operator==(const Light& other) const {
        return type == other.type && modelMatrixIndex == other.modelMatrixIndex && range == other.range && intensity == other.intensity && shadowResolution == other.shadowResolution && 
               color == other.color && lightSpaceMatrix == other.lightSpaceMatrix && direction == other.direction && castsShadows == other.castsShadows;
    }
};

struct ParticleEmitter {
    uint32_t nodeIndex;
    uint32_t textureIndex = 0xFFFFFFFF;
    //offsets from the node it's attached to
    glm::vec3 positionOffset = glm::vec3(0);
    glm::quat rotationOffset = glm::quat(0,0,0,1);

    //=====Emission Params=====//
    // particles / second
    float emissionRate = 5.0f;
    glm::vec2 lifeTime = glm::vec2(1.0f,1.0f); // implicitly the particle cap of this emitter is (lifetime * emissionRate)
    // this is the half angle of the spread
    float spreadAngle = 30.0f;
    float speedMin = 0.0f;
    float speedMax = 1.0f;
    glm::vec2 angularVelocityRandom = glm::vec2(0,0);
    float drag = 0.01f;
    glm::vec2 sizeRandom = glm::vec2(1.0f, 1.0f); // per-particle size range (min, max)

    //=====Rendering Behavior=====//
    bool initialized = false;
    bool animated = false;
    uint8_t numFrames = 0;
    bool lit = false;
    bool volumetric = false;
    bool volumetricSphere = false;  // volumetric: inject a view-independent sphere (fly-through) vs a textured billboard
    bool softParticle = false;      // fade out where particles intersect scene geometry
    float softRadius = 1.0f;        // depth-fade distance (view-space units)
    float sphereRoundness = 1.0f;   // lit: spherical-impostor bulge, 0 flat card .. 1 sphere (>1 exaggerates)
    float opacity = 1.0f;           // global transparency multiplier for the whole emitter
    glm::vec2 densityRange = glm::vec2(0,1.0f);
    float volumePhase = 0.5f;
    glm::vec2 emissiveRange = glm::vec2(1.0f,1.0f);             
    //=====Pool residency (assigned by Scene::addEmitter)=====//
    // Contiguous sub-range this emitter owns in the shared particle pool, run as a ring buffer.
    uint32_t particleOffset = 0;    // base index into the pool, in Particle units
    uint32_t particleCapacity = 0;  // ring size; 0 until the emitter is registered

    // Compute dispatches process the pool in workgroups of this size
    static constexpr uint32_t PARTICLE_WORKGROUP = 64;

    // Steady-state live count is emissionRate * maxLifetime; because the ring holds at least that
    // many, the head can never lap a still-alive particle. 
    // +1 to cover the spawn-in-the-same-frame-as-expiry edge, then round up to keep dispatches aligned.
    uint32_t capacity() const {
        float maxLife = glm::max(lifeTime.x, lifeTime.y);
        uint32_t raw = static_cast<uint32_t>(maxLife * emissionRate) + 1u + PARTICLE_WORKGROUP;
        return ((raw + PARTICLE_WORKGROUP - 1u) / PARTICLE_WORKGROUP) * PARTICLE_WORKGROUP;
    }

    GPUParticleEmitter toGPU(const glm::vec3& worldPos, const glm::quat& worldRot) const {
        GPUParticleEmitter gpu{};
        gpu.position          = worldPos;
        gpu.emissionRate      = emissionRate;
        gpu.spawnRotation     = glm::vec4(worldRot.x, worldRot.y, worldRot.z, worldRot.w);
        gpu.speedMin          = speedMin;
        gpu.lifeTimeMin       = lifeTime.x;
        gpu.speedMax          = speedMax;
        gpu.lifeTimeMax       = lifeTime.y;
        gpu.angularVelocityRandom = angularVelocityRandom;
        gpu.spreadAngle       = glm::radians(spreadAngle);
        gpu.drag              = drag;
        gpu.sizeRandom        = sizeRandom;
        gpu.softRadius        = softRadius;
        gpu.densityRange      = densityRange;
        gpu.phase             = volumePhase;
        gpu.emissiveRange     = emissiveRange;
        gpu.particleOffset    = particleOffset;
        gpu.particleCapacity  = particleCapacity;
        gpu.textureIndex      = textureIndex;
        gpu.numFrames         = numFrames;
        gpu.flags             = (animated ? EMITTER_FLAG_ANIMATED : 0u) |
                                (lit ? EMITTER_FLAG_LIT : 0u) |
                                (volumetric ? EMITTER_FLAG_VOLUMETRIC : 0u) |
                                (volumetricSphere ? EMITTER_FLAG_VOLUME_SPHERE : 0u) |
                                (softParticle ? EMITTER_FLAG_SOFT : 0u);
        return gpu;
    }
};

struct Camera {
    glm::vec3 position;
    glm::vec3 target;
    float fov;
    float aspectRatio;
    float nearPlane;
    float farPlane;
    glm::mat4 viewProjection;
    glm::mat4 prevViewProjection;
    glm::mat4 viewMatrix;
    glm::mat4 projectionMatrix;
    float yaw = -90.0f;
    float pitch = 0.0f;

    void rayFromScreenCoords(float x, float y, glm::vec3& outOrigin, glm::vec3& outDirection) {

        glm::vec4 nearClip = glm::vec4(x, y, -1.0f, 1.0f);
        glm::vec4 farClip = glm::vec4(x, y, 1.0f, 1.0f);

        // view space
        glm::vec4 nearView = glm::inverse(projectionMatrix) * nearClip;
        glm::vec4 farView = glm::inverse(projectionMatrix) * farClip;

        // Perspective divide
        nearView /= nearView.w;
        farView /= farView.w;

        // Transform to world space
        glm::vec4 nearWorld = glm::inverse(viewMatrix) * nearView;
        glm::vec4 farWorld = glm::inverse(viewMatrix) * farView;

        // Ray from near to far
        outOrigin = glm::vec3(nearWorld);
        outDirection = glm::normalize(glm::vec3(farWorld - nearWorld));
    }

    void rotateYaw(float deltaYaw = 0.0f) {
        yaw += deltaYaw;
        if (yaw > 360.0f)
            yaw -= 360.0f;
        if (yaw < -360.0f)
            yaw += 360.0f;
        updateTarget();
    }

    void rotatePitch(float deltaPitch = 0.0f) {
        pitch += deltaPitch;
        // prevents gimbal lock
        if (pitch > 89.0f)
            pitch = 89.0f;
        if (pitch < -89.0f)
            pitch = -89.0f;
        updateTarget();
    }

    void moveCamera(glm::vec3 deltaPosition) {
        glm::vec3 forward = glm::normalize(target - position);
        position += forward * deltaPosition.z;
        position += glm::cross(forward, glm::vec3(0.0, 1.0, 0.0)) * deltaPosition.x;
        updateTarget();
    }

    // based on current yaw and pitch
    void updateTarget() {
        glm::vec3 direction;
        direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        direction.y = sin(glm::radians(pitch));
        direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        direction = glm::normalize(direction);
        target = position + direction;
        calculateViewProjectionMatrix();
    }

    void lookAt(const glm::vec3& targetPos) {
        // Calculates yaw and pitch
        target = targetPos;
        glm::vec3 direction = glm::normalize(target - position);
        yaw = glm::degrees(atan2(direction.z, direction.x));
        pitch = glm::degrees(asin(direction.y));
        calculateViewProjectionMatrix();
    }

    void calculateViewProjectionMatrix() {
        glm::vec3 upVector = glm::vec3(0.0f, 1.0f, 0.0f); // World up direction
        // Create view matrix using lookAt
        viewMatrix = glm::lookAt(position, target, upVector);
        // Reverse-Z: swap near/far so near maps to NDC 1 and far to 0 (better depth precision).
        // Use the explicit _ZO variant so the [0,1] clip convention holds regardless of
        // GLM_DEPTH_ZERO_TO_ONE include ordering in this translation unit.
        projectionMatrix = glm::perspectiveRH_ZO(glm::radians(fov), aspectRatio, farPlane, nearPlane);
        projectionMatrix[1][1] *= -1.0f; // Flip Y axis for vulkan
        viewProjection = projectionMatrix * viewMatrix;
    }

    void updatePrevVPM() {
        prevViewProjection = viewProjection;
    }
    
    std::vector<glm::vec4> getFrustumCorners(Camera& camera) {
        glm::mat4 invViewProj = glm::inverse(camera.viewProjection);

        std::vector<glm::vec4> corners;
        corners.reserve(8);

        // NDC corners of the frustum
        for (int x = 0; x < 2; ++x) {
            for (int y = 0; y < 2; ++y) {
                for (int z = 0; z < 2; ++z) {
                    glm::vec4 corner = invViewProj * glm::vec4(2.0f * x - 1.0f, 2.0f * y - 1.0f, static_cast<float>(z), 1.0f);
                    corners.push_back(corner / corner.w);
                }
            }
        }

        return corners;
    }

    glm::vec3 getLookDir() {
        return glm::normalize(target - position);
    }
};


