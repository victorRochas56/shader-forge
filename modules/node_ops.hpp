#pragma once
#include "scene_elements.hpp"

class Renderer;

namespace NodeOps {

    void assignMesh(Node& node, uint32_t meshIndex, Renderer& renderer);
    void assignMaterial(Node& node, uint32_t materialIndex, Renderer& renderer);
    void assignLight(Node& node, Light light, Renderer& renderer);
    void assignVolume(Node& node, Volume volume, Renderer& renderer);
    void enableLightShadows(Light& light, const std::string& nodeName, Renderer& renderer);
    void disableLightShadows(Light& light, Renderer& renderer);
}
