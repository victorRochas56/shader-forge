#pragma once
#include "scene_elements.hpp"

class Renderer;

namespace NodeOps {

    void assignMesh(Node& node, uint32_t meshIndex, Renderer& renderer);
    void assignMaterial(Node& node, uint32_t submeshIndex, uint32_t materialIndex, Renderer& renderer);
    void assignLight(Node& node, Light light, Renderer& renderer);
    void enableLightShadows(Light& light, const std::string& nodeName, Renderer& renderer);
    void disableLightShadows(Light& light, Renderer& renderer);

    glm::mat4 calculateLightSpaceMatrix(Light& light, Camera& camera);
    void calculatePointLightFaceMatrices(Light& light, const glm::vec3& lightPos);
    void calculateCascadedLightSpaceMatrices(Light& light, Camera& camera, Renderer* renderer);
}
