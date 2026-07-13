#pragma once
#include "scene_elements.hpp"

class Renderer;
class Scene;
class BindlessSystem;

namespace NodeOps {

    void assignMesh(Node& node, uint32_t meshIndex, Scene& scene);
    void assignMaterial(Node& node, uint32_t materialIndex, Scene& scene);
    void assignLight(Node& node, Light light, Scene& scene, BindlessSystem& bindless, uint32_t lightBufferIndex);
    void assignVolume(Node& node, Volume volume, Scene& scene);
    void assignBillboard(Node& node, Billboard billboard, Scene& scene);
    void assignEmitter(Node& node, ParticleEmitter emitter, Scene& scene, BindlessSystem& bindless, uint32_t particleBufferIndex);
    void enableLightShadows(Light& light, const std::string& nodeName, Scene& scene);
    void disableLightShadows(Light& light, Scene& scene);
}
