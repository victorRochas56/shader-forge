#include "node_ops.hpp"
#include "renderer.hpp"
#include "transform_system.hpp"

namespace NodeOps {

void assignMesh(Node& node, uint32_t meshIndex, Renderer& renderer) {
    if (node.meshIndex < renderer.assetManager.meshes.size()) {
        renderer.assetManager.meshes[meshIndex].refCount--;
        if (renderer.assetManager.meshes[meshIndex].refCount <= 0) {
            renderer.assetManager.meshes[meshIndex].freed = true;
        }
    }
    node.meshIndex = meshIndex;
    renderer.assetManager.meshes[meshIndex].refCount++;

    // Calculate world-space bounding box from mesh local-space bounding box
    const auto& mesh = renderer.assetManager.meshes[meshIndex];
    glm::vec3 corners[8] = {
        glm::vec3(mesh.boundingBoxMin.x, mesh.boundingBoxMin.y, mesh.boundingBoxMin.z), glm::vec3(mesh.boundingBoxMax.x, mesh.boundingBoxMin.y, mesh.boundingBoxMin.z),
        glm::vec3(mesh.boundingBoxMin.x, mesh.boundingBoxMax.y, mesh.boundingBoxMin.z), glm::vec3(mesh.boundingBoxMax.x, mesh.boundingBoxMax.y, mesh.boundingBoxMin.z),
        glm::vec3(mesh.boundingBoxMin.x, mesh.boundingBoxMin.y, mesh.boundingBoxMax.z), glm::vec3(mesh.boundingBoxMax.x, mesh.boundingBoxMin.y, mesh.boundingBoxMax.z),
        glm::vec3(mesh.boundingBoxMin.x, mesh.boundingBoxMax.y, mesh.boundingBoxMax.z), glm::vec3(mesh.boundingBoxMax.x, mesh.boundingBoxMax.y, mesh.boundingBoxMax.z)};

    node.boundingBoxMin = glm::vec3(std::numeric_limits<float>::max());
    node.boundingBoxMax = glm::vec3(std::numeric_limits<float>::lowest());

    for (const auto& corner : corners) {
        glm::vec3 worldCorner = glm::vec3(node.worldTransform * glm::vec4(corner, 1.0f));
        node.boundingBoxMin = glm::min(node.boundingBoxMin, worldCorner);
        node.boundingBoxMax = glm::max(node.boundingBoxMax, worldCorner);
    }
    node.boundingBoxValid = true;
}

void assignMaterial(Node& node, uint32_t submeshIndex, uint32_t materialIndex, Renderer& renderer) {
    if (submeshIndex >= renderer.assetManager.meshes[node.meshIndex].subMeshes.size()) {
        throw std::runtime_error("tried adding material to invalid material slot on node mesh!");
    } else {
        renderer.addMeshToShader(&node, renderer.assetManager.meshes[node.meshIndex].subMeshes[submeshIndex], renderer.getMaterials()[materialIndex].shaderSource,
                                  renderer.getMaterials()[materialIndex]);
        node.materialIndices.push_back(materialIndex);
        node.materialIndexCount++;
    }
}

uint32_t allocateShadowMap(Light& light, const std::string& name, Renderer& renderer) {
    ResourceManager& resourceManager = renderer.getResourceManager();
    vk::raii::Image image = nullptr;
    vk::raii::DeviceMemory memory = nullptr;
    resourceManager.createImage(light.shadowResolution, light.shadowResolution, 1, vk::SampleCountFlagBits::e1, vk::Format::eD32Sfloat,
                                vk::ImageTiling::eOptimal,
                                vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
                                vk::MemoryPropertyFlagBits::eDeviceLocal, image, memory, 1);
    vk::raii::ImageView imageView = resourceManager.createImageView(image, vk::Format::eD32Sfloat, vk::ImageAspectFlagBits::eDepth);
    resourceManager.transitionImageLayout(nullptr, image, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal);
    resourceManager.transitionImageLayout(nullptr, image, vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
    return renderer.getDescriptorSet().allocateTexture(std::move(image), std::move(memory), std::move(imageView), name);
}

void enableLightShadows(Light& light, const std::string& nodeName, Renderer& renderer) {
    light.castsShadows = 1;
    light.shadowDirty = true;
    switch (light.type) {
    case LightType::Directional:
        for (int i = 0; i < light.numCascades; i++) {
            light.cascades[i].shadowMapIndex = allocateShadowMap(light, "internal/" + nodeName + "/csm_" + std::to_string(i), renderer);
        }
        break;
    case LightType::Point:
        for (int i = 0; i < 6; i++) {
            light.cubeMapIndices[i].shadowMapIndex = allocateShadowMap(light, "internal/" + nodeName + "/point_" + std::to_string(i), renderer);
        }
        break;
    default:
        break;
    }
}

void disableLightShadows(Light& light, Renderer& renderer) {
    light.castsShadows = 0;
    switch (light.type) {
    case LightType::Directional:
        for (int i = 0; i < light.numCascades; i++) {
            renderer.getDescriptorSet().freeTexture(light.cascades[i].shadowMapIndex);
        }
        break;
    case LightType::Point:
        for (int i = 0; i < 6; i++) {
            renderer.getDescriptorSet().freeTexture(light.cubeMapIndices[i].shadowMapIndex);
        }
        break;
    default:
        break;
    }
}

void assignLight(Node& node, Light light, Renderer& renderer) {
    light.nodeIndex = node.nodeIndex;
    light.modelMatrixIndex = node.modelMatrixIndices[0];

    if (light.castsShadows == 1) {
        light.castsShadows = 0; // reset so enableLightShadows sets it
        enableLightShadows(light, node.name, renderer);
    }

    node.lightIndex = renderer.getDescriptorSet().allocateFixedBuffer<GPULight>(renderer.getLightBufferIndex(), light.toGPU());
    renderer.addLight(node.lightIndex, light);
    std::cout << "added light at index : " << node.lightIndex << std::endl;
    TransformSystem::updateAll(node, renderer.sceneGraph.getNodes(), renderer.getDescriptorSet(), renderer.getModelMatrixBufferIndex(),
                               renderer.assetManager.meshes, renderer.getLightsMutable());
}


glm::mat4 calculateLightSpaceMatrix(Light& light, Camera& camera) {

    glm::mat4 lightProjection = glm::ortho(-light.range, light.range, -light.range, light.range, camera.nearPlane, camera.farPlane);
    glm::vec3 lightPos = camera.position - light.direction * 0.5f * camera.farPlane;
    glm::mat4 lightView = glm::lookAt(lightPos, lightPos + light.direction, glm::vec3(0.0f, 1.0f, 0.0f));
    return lightProjection * lightView;
}

void calculatePointLightFaceMatrices(Light& light, const glm::vec3& lightPos) {
    float nearPlane = 0.1f;
    float farPlane = light.range;
    glm::mat4 projection = glm::perspective(glm::radians(90.0f), 1.0f, nearPlane, farPlane);

    // 6 cubemap faces: +X, -X, +Y, -Y, +Z, -Z
    const glm::vec3 directions[6] = {
        { 1.0f,  0.0f,  0.0f},  // +X
        {-1.0f,  0.0f,  0.0f},  // -X
        { 0.0f,  1.0f,  0.0f},  // +Y
        { 0.0f, -1.0f,  0.0f},  // -Y
        { 0.0f,  0.0f,  1.0f},  // +Z
        { 0.0f,  0.0f, -1.0f},  // -Z
    };
    const glm::vec3 ups[6] = {
        { 0.0f, -1.0f,  0.0f},  // +X
        { 0.0f, -1.0f,  0.0f},  // -X
        { 0.0f,  0.0f,  1.0f},  // +Y
        { 0.0f,  0.0f, -1.0f},  // -Y
        { 0.0f, -1.0f,  0.0f},  // +Z
        { 0.0f, -1.0f,  0.0f},  // -Z
    };

    for (int i = 0; i < 6; i++) {
        glm::mat4 view = glm::lookAt(lightPos, lightPos + directions[i], ups[i]);
        light.cubeMapIndices[i].lightSpaceMatrix = projection * view;
    }
}

void calculateCascadedLightSpaceMatrices(Light& light, Camera& camera, Renderer* renderer) {
    glm::vec3 lightDir = glm::normalize(light.direction);

    // Stable up vector that avoids degeneracy when light is near-vertical
    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
    if (glm::abs(glm::dot(lightDir, up)) > 0.99f) {
        up = glm::vec3(1.0f, 0.0f, 0.0f);
    }
    // Unproject the full camera frustum to world space (Vulkan NDC: z in [0,1])
    // z-outermost so indices 0-3 = near plane, 4-7 = far plane
    glm::mat4 invCamVP = glm::inverse(camera.viewProjection);
    glm::vec3 fullCorners[8];
    int idx = 0;
    for (int z = 0; z < 2; ++z)
        for (int y = 0; y < 2; ++y)
            for (int x = 0; x < 2; ++x) {
                glm::vec4 c = invCamVP * glm::vec4(2.f * x - 1.f, 2.f * y - 1.f, static_cast<float>(z), 1.f);
                fullCorners[idx++] = glm::vec3(c / c.w);
            }

    float lastSplitDist = 0.0f;

    for (uint32_t i = 0; i < light.numCascades; i++) {

        // Cascade splits are in world-space units
        float splitDist;
        if (light.cascades[i].splitDistance > 0.0f) {
            splitDist = (light.cascades[i].splitDistance - camera.nearPlane) / (camera.farPlane - camera.nearPlane);
            splitDist = glm::clamp(splitDist, lastSplitDist + 0.001f, 1.0f);
        } else {
            splitDist = static_cast<float>(i + 1) / static_cast<float>(light.numCascades);
            light.cascades[i].splitDistance = camera.nearPlane + splitDist * (camera.farPlane - camera.nearPlane);
        }

        // Slice the full frustum into this cascade's sub-frustum
        glm::vec3 corners[8];
        for (int j = 0; j < 4; j++) {
            glm::vec3 ray = fullCorners[j + 4] - fullCorners[j];
            corners[j]     = fullCorners[j] + ray * lastSplitDist;
            corners[j + 4] = fullCorners[j] + ray * splitDist;
        }

        // Sub-frustum center
        glm::vec3 center(0.0f);
        for (const auto& c : corners) center += c;
        center /= 8.0f;

        // Build light view matrix looking at the frustum center
        float zPullBack = 500.0f;
        glm::mat4 lightView = glm::lookAt(
            center - lightDir * zPullBack,
            center,
            up
        );

        // Compute tight AABB in light space from the frustum corners
        glm::vec3 lsMin(std::numeric_limits<float>::max());
        glm::vec3 lsMax(std::numeric_limits<float>::lowest());
        for (const auto& c : corners) {
            glm::vec3 ls = glm::vec3(lightView * glm::vec4(c, 1.0f));
            lsMin = glm::min(lsMin, ls);
            lsMax = glm::max(lsMax, ls);
        }

        float extentX = lsMax.x - lsMin.x;
        float extentY = lsMax.y - lsMin.y;
        float maxExtent = glm::max(extentX, extentY);

        // Expand AABB for cascade overlap
        float overlapMargin = maxExtent * 0.1f;
        lsMin.x -= overlapMargin;
        lsMin.y -= overlapMargin;
        lsMax.x += overlapMargin;
        lsMax.y += overlapMargin;

        // Near=0.1 captures shadow casters between the light eye and the frustum.
        // Far extends just past the farthest frustum corner in light space.
        float orthoNear = 0.1f;
        float orthoFar  = -lsMin.z + 10.0f;

        glm::mat4 lightProj = glm::ortho(
            lsMin.x, lsMax.x,
            lsMin.y, lsMax.y,
            orthoNear, orthoFar
        );

        // Snap the shadow matrix translation to texel boundaries so the
        // shadow map stays locked to a fixed world-space grid as the camera moves.
        glm::mat4 shadowMatrix = lightProj * lightView;
        float halfRes = static_cast<float>(light.shadowResolution) * 0.5f;
        glm::vec4 shadowOrigin = shadowMatrix * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        shadowOrigin *= halfRes;
        glm::vec4 rounded = glm::round(shadowOrigin);
        glm::vec4 roundOffset = rounded - shadowOrigin;
        roundOffset /= halfRes;
        lightProj[3][0] += roundOffset.x;
        lightProj[3][1] += roundOffset.y;

        float finalExtent = glm::max(lsMax.x - lsMin.x, lsMax.y - lsMin.y);
        light.cascades[i].lightSpaceMatrix = lightProj * lightView;
        light.cascades[i].texelSize = 1.0f / static_cast<float>(light.shadowResolution);
        light.cascades[i].worldTexelSize = finalExtent / static_cast<float>(light.shadowResolution);

        lastSplitDist = splitDist;
    }
}

}
