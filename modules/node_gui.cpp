#include "node_gui.hpp"
#include "gui.hpp"
#include "input.hpp"
#include "node_ops.hpp"
#include "scene.hpp"
#include "gizmo.hpp"
#include "manipulators.hpp"

static std::unordered_map<uint32_t, NodeGuiState> guiStates;

NodeGuiState& getNodeGuiState(uint32_t nodeIndex) { return guiStates[nodeIndex]; }

void showNodeInfo(GUI& gui, Node& node, Scene& scene, BindlessSystem& bindless, RenderBuffers& buffers) {
    if (scene.sceneGraph.selectedNode == 0)
        return;

    auto& state = getNodeGuiState(node.nodeIndex);
   
    showTransformModifiers(gui);
    if (!gui.beginWindow("selected node", nullptr, glm::vec2(360, 620), glm::vec2(950, 20))) return;
    gui.textf("%s", node.name.c_str());

    showNodeMeshInfo(gui, node, scene);
    if (state.changingMaterials) {
        showNodeMaterialDialog(gui, node, scene);
    }

    if (gui.button("Duplicate")) {
        // Step the copy clear along X by the mesh's own width, so a big model doesn't land inside
        // itself. Nodes with no geometry (lights, emitters) fall back to a fixed nudge.
        float step = 0.5f;
        if (node.meshIndex < scene.assetManager.meshes.size()) {
            const Mesh& mesh = scene.assetManager.meshes[node.meshIndex];
            step = (mesh.boundingBoxMax.x - mesh.boundingBoxMin.x) * node.relativeScale.x;
        }
        if (step < 1e-3f) step = 0.5f;

        uint32_t copyIndex = scene.sceneGraph.duplicateNode(node, node.relativePosition + glm::vec3(step, 0.0f, 0.0f));
        // Selecting the copy makes a second click step clear of the first rather than stacking
        // another copy on the same spot.
        scene.sceneGraph.deSelectNode();
        scene.sceneGraph.selectNode(copyIndex);
        gui.endWindow();
        return; // node reference and selection both just changed
    }

    showNodeLightInfo(gui, node, scene, bindless, buffers.lightBufferIndex);
    showNodeVolumeInfo(gui, node, scene);
    showNodeEmitterInfo(gui, node, scene, bindless, buffers);
    showNodeTransformInfo(gui, node, scene);
    node.transformDirty = true;

    gui.separator();
    if (gui.button("Delete Node")) {
        uint32_t idx = node.nodeIndex;
        scene.sceneGraph.deSelectNode();
        scene.sceneGraph.removeNode(idx);
        gui.endWindow();
        return;
    }

    gui.endWindow();
}

void showNodeMeshInfo(GUI& gui, Node& node, Scene& scene) {
    auto& state = getNodeGuiState(node.nodeIndex);

    if (!state.changingMesh) {
        if (node.meshIndex < scene.assetManager.meshes.size()) {
            const Mesh& mesh = scene.assetManager.meshes[node.meshIndex];
            if (gui.button(mesh.sourceFile.c_str())) {
                state.changingMesh = true;
                state.textBuffer[0] = '\0';
            }

            // LOD override. Auto reports what the screen-size heuristic currently wants, so the
            // preview doubles as a readout of what this mesh is drawing at.
            std::string preview = node.lodOverride == Node::LOD_AUTO
                                ? "Auto (LOD " + std::to_string(mesh.currentLOD) + ")"
                                : "LOD " + std::to_string(std::min<uint32_t>(node.lodOverride, static_cast<uint32_t>(mesh.LODs.size()) - 1));
            gui.setNextItemWidth(160);
            if (gui.beginCombo("LOD", preview)) {
                // Re-sorting the render list is what keeps nodes on the same level batched together.
                if (gui.comboItem("Auto", node.lodOverride == Node::LOD_AUTO)) {
                    node.lodOverride = Node::LOD_AUTO;
                    scene.renderListDirty = true;
                }
                for (uint32_t i = 0; i < mesh.LODs.size(); i++) {
                    std::string label = "LOD " + std::to_string(i) + "  (" + std::to_string(mesh.LODs[i] / 3) + " tris)";
                    if (gui.comboItem(label, node.lodOverride == i)) {
                        node.lodOverride = i;
                        scene.renderListDirty = true;
                    }
                }
                gui.endCombo();
            }
        } else {
            if (gui.button("Add Mesh")) {
                state.changingMesh = true;
                state.textBuffer[0] = '\0';
            }
        } 
        if ( gui.button("Show Wireframe") ) {
            node.toggleWireframe();
        }
    } else {
        InputManager::getInstance().canMove = false;
        gui.setNextItemWidth(160);
        gui.inputText("mesh source", state.textBuffer, sizeof(state.textBuffer));
        browseButton(gui, "mesh", state.textBuffer, sizeof(state.textBuffer));
        gui.setNextItemWidth(160);
        gui.inputFloat("import scale", &state.importScale);
        gui.checkbox("Keep material assignments", &state.keepMaterialAssignments);

        if (gui.button("Confirm")) {
            uint32_t thisNodeIndex = node.nodeIndex; // capture before any vector reallocation

            // remove old mesh from render queue
            if (node.meshIndex < scene.assetManager.meshes.size() && node.materialIndex != 0xFFFFFFFF) {
                scene.removeMeshFromShader(node.nodeIndex, scene.getMaterials()[node.materialIndex].shaderSource,
                                                     scene.getMaterials()[node.materialIndex]);
            }

            auto loadResult = scene.assetManager.loadMeshFromFile(std::string(state.textBuffer), state.importScale);
            auto& meshIndices = loadResult.meshIndices;

            // Build material mapping: source material ID -> renderer material index
            // When keepMaterialAssignments is on, create a dummy material per source material
            std::map<int, uint32_t> matIdToRendererIdx;
            if (state.keepMaterialAssignments) {
                Material baseMat = scene.getMaterials()[scene.getFallBackMaterial()];
                for (auto& [srcMatId, name] : loadResult.materialNames) {
                    Material mat = baseMat;
                    mat.name = name;
                    uint32_t matIdx = scene.addMaterial(mat);
                    matIdToRendererIdx[srcMatId] = matIdx;
                }
            }

            if (meshIndices.size() == 1) {
                // Single mesh — assign directly to this node
                auto& n = scene.sceneGraph.getNodes()[thisNodeIndex];
                NodeOps::assignMesh(n, meshIndices[0], scene);
                uint32_t matIdx = scene.getFallBackMaterial();
                if (state.keepMaterialAssignments && !loadResult.materialIds.empty()) {
                    int srcMatId = loadResult.materialIds[0];
                    if (matIdToRendererIdx.count(srcMatId))
                        matIdx = matIdToRendererIdx[srcMatId];
                }
                NodeOps::assignMaterial(n, matIdx, scene);
            } else if (meshIndices.size() > 1) {
                for (size_t i = 0; i < meshIndices.size(); i++) {
                    glm::vec3 pos;
                    glm::quat rot;
                    glm::vec3 scale;
                    uint32_t childIdx = scene.sceneGraph.addNode(false, thisNodeIndex);
                    // re-fetch after addNode since nodes vector may have reallocated
                    Node& childNode = scene.sceneGraph.getNodes()[childIdx];
                    childNode.name = scene.sceneGraph.makeUniqueNodeName(scene.assetManager.getMeshes()[meshIndices[i]].name);
                    decomposeTransform(loadResult.transforms[i], pos, rot, scale);
                    childNode.relativePosition = pos;
                    childNode.relativeRotation = rot;
                    childNode.transformDirty = true;
                    NodeOps::assignMesh(childNode, meshIndices[i], scene);

                    uint32_t matIdx = scene.getFallBackMaterial();
                    if (state.keepMaterialAssignments) {
                        matIdx = matIdToRendererIdx[loadResult.materialIds[i]];
                    }
                    NodeOps::assignMaterial(childNode, matIdx, scene);
                }
            }
#if DEBUG == 1
            std::cout << "loaded " << meshIndices.size() << " mesh(es)" << std::endl;
#endif
            state.keepMaterialAssignments = false;
            state.changingMesh = false;
            // node reference is now potentially invalid — return early
            return;
        }
        gui.sameLine();
        if (gui.button("Cancel")) {
            state.changingMesh = false;
        }
    }
    if (node.meshIndex < scene.assetManager.meshes.size()) {
        if (gui.button("Assign Materials")) {
            state.changingMaterials = !state.changingMaterials;
        }
    }

    if(node.showWireframe) {
        Mesh& mesh = scene.assetManager.meshes[node.meshIndex];
        // wireframe of the LOD this node actually draws: its override, else what auto last picked
        uint32_t maxLod = mesh.LODs.empty() ? 0u : static_cast<uint32_t>(mesh.LODs.size()) - 1;
        uint32_t lod = node.lodOverride == Node::LOD_AUTO ? mesh.currentLOD : std::min(node.lodOverride, maxLod);
        uint32_t first = mesh.lodIndexStart(lod);
        uint32_t last  = first + mesh.lodIndexCount(lod);
        for(uint32_t i = first; i < last; i+=3) {
            auto& t = node.worldTransform;
            Gizmos::drawLine({t*glm::vec4(mesh.cpuPositions[mesh.cpuIndices[i]],1.0f),t*glm::vec4(mesh.cpuPositions[mesh.cpuIndices[i+1]],1.0f), {2.0,2.0,0.0,1.0}});
            Gizmos::drawLine({t*glm::vec4(mesh.cpuPositions[mesh.cpuIndices[i+1]],1.0f),t*glm::vec4(mesh.cpuPositions[mesh.cpuIndices[i+2]],1.0f), {2.0,2.0,0.0,1.0}});
            Gizmos::drawLine({t*glm::vec4(mesh.cpuPositions[mesh.cpuIndices[i]],1.0f),t*glm::vec4(mesh.cpuPositions[mesh.cpuIndices[i+2]],1.0f), {2.0,2.0,0.0,1.0}});
        }
    }
}

void showNodeMaterialDialog(GUI& gui, Node& node, Scene& scene) {
    auto& state = getNodeGuiState(node.nodeIndex);

    if (state.materialList.size() != scene.getMaterials().size()) {
        state.materialList.clear();
        for (int i = 0; i < scene.getMaterials().size(); i++) {
            const auto& material = scene.getMaterials()[i];
            std::string textOption = material.name.empty() ? ("Material " + std::to_string(i)) : material.name;
            state.materialList.push_back(textOption);
        }
    }

    if (!gui.beginWindow("Change Material", nullptr, glm::vec2(300, 200), glm::vec2(620, 300))) return;

    uint32_t currentMatIdx = node.materialIndex != 0xFFFFFFFF ? node.materialIndex : 0;
    std::string currentMatName = state.materialList[currentMatIdx];

    gui.setNextItemWidth(200);
    if (gui.beginCombo("Material", currentMatName)) {
        for (int n = 0; n < state.materialList.size(); n++) {
            bool is_selected = (currentMatIdx == static_cast<uint32_t>(n));
            if (gui.comboItem(state.materialList[n].c_str(), is_selected)) {
                if (static_cast<uint32_t>(n) != currentMatIdx) {
                    NodeOps::assignMaterial(node, n, scene);
                }
            }
            if (is_selected) {
            }
        }
        gui.endCombo();
    }

    gui.endWindow();
}

void showNodeLightInfo(GUI& gui, Node& node, Scene& scene, BindlessSystem& bindless, uint32_t lightBufferIndex) {
    auto& state = getNodeGuiState(node.nodeIndex);

    if (node.lightIndex != MAX_LIGHTS && node.lightIndex < scene.getLights().size()) {
        auto& light = scene.getLight(node.lightIndex);
        const char* lightTypeNames[] = {"Point", "Directional", "Spot", "Area"};
        if (gui.beginCombo("Light Type", lightTypeNames[static_cast<int>(light.type)])) {
            for (int i = 0; i < static_cast<int>(LightType::COUNT); i++) {
                bool isSelected = (static_cast<int>(light.type) == i);
                if (gui.comboItem(lightTypeNames[i], isSelected)) {
                    light.type = static_cast<LightType>(i);
                }
                if (isSelected) {
                }
            }
            gui.endCombo();
        }
        if (light.type != LightType::Directional) {
            if (gui.dragFloat("range", &light.range)) {
                // Range change alters which nodes fall inside the sphere —
                // flag the light's node so syncDirtyNodes rebuilds the set.
                node.transformDirty = true;
            }
        }

        for(uint32_t nodeIdx : light.influencedNodes) {
            Gizmos::drawLine(Line{.startPoint = scene.sceneGraph.getNode(nodeIdx).getWorldPosition(),.endPoint = node.getWorldPosition(),.color = glm::vec4(1,1,0,1)});
        }

        gui.dragFloat("intensity", &light.intensity);
        float colR = light.color.r;
        float colG = light.color.g;
        float colB = light.color.b;
        float color[3] = {colR, colG, colB};
        gui.colorPicker3("color", color);
        light.color = glm::vec4(color[0], color[1], color[2], 1);

        if (light.type == LightType::Directional) {

            gui.dragInt("show cascades", &light.showCascades, 1, 0, 1);

            for (int i = 0; i < light.numCascades; i++) {
                std::string cascadeLabel = "cascade ";
                cascadeLabel += std::to_string(i);
                gui.dragFloat(cascadeLabel.c_str(), &light.cascades[i].splitDistance);
            }
        }

        if (light.type == LightType::Directional || light.type == LightType::Point) {
            state.lightShadow = light.castsShadows;
            if (gui.checkbox("Enable Shadows", &state.lightShadow)) {
                if (state.lightShadow) {
                    if(light.type == LightType::Directional)
                        light.shadowResolution = DEFAULT_CSM_SHADOW_RESOLUTION;
                    NodeOps::enableLightShadows(light, node.name, scene);
                } else {
                    NodeOps::disableLightShadows(light, scene);
                }
                // Toggling shadows changes whether this light participates
                // in the influence set — rebuild on next sync.
                node.transformDirty = true;
            }
        }
        // Flag for GPU fanout — the per-frame renderer loop writes each frame-in-flight slice.
        light.gpuDirtyFrames = MAX_FRAMES_IN_FLIGHT;
        

        Gizmos::drawSphere(node.getWorldPosition(),light.range, glm::vec4(1,1,0,1));

        if ( gui.button("Remove Light")) {
            scene.removeLight(bindless, lightBufferIndex, node.lightIndex);
            node.lightIndex = MAX_LIGHTS;
        }

    } else if (gui.button("Add Light")) {
        Light light = {.type = LightType::Point, .range = 10, .intensity = 1, .color = glm::vec4(1, 1, 1, 1)};
        NodeOps::assignLight(node, light, scene, bindless, lightBufferIndex);
    }

}

void showNodeVolumeInfo(GUI& gui, Node& node, Scene& scene) {

    if(!scene.volumes.contains(node.nodeIndex)) {
        if(gui.button("Add Volume")){
            Volume vol;
            NodeOps::assignVolume(node, vol, scene);
        }
    }
    else {
        // Edits land in the CPU-side Volume; VolumetricsPass streams it to the GPU next frame,
        // so no explicit re-upload (or transformDirty) is needed here.
        gui.sliderFloat("radius", &scene.volumes[node.nodeIndex].radius,0.0f,20.0f);
        gui.sliderFloat("density", &scene.volumes[node.nodeIndex].density,0.0f,1.0f);
        gui.sliderFloat("phase", &scene.volumes[node.nodeIndex].phase,0.0f,1.0f);
    }
}

void showNodeEmitterInfo(GUI& gui, Node& node, Scene& scene, BindlessSystem& bindless, const RenderBuffers& buffers) {
    if(node.particleIndex == 0xFFFFFFFF) {
        if(gui.button("Add Emitter")){
            ParticleEmitter emitter;
            NodeOps::assignEmitter(node, emitter, scene, bindless, buffers);
        }
    }
    else {
        if ( gui.button("Remove Emitter")) {
            scene.removeEmitter(bindless, buffers, node.particleIndex);
            node.particleIndex = 0xFFFFFFFF;
            return; // emitter gone; don't touch it below
        }

        ParticleEmitter& emitter = scene.particleEmitters.at(node.particleIndex);

        // These two feed capacity() -> the reserved pool range. Re-reserve when the drag ends
        // (not every frame it's held) so the ring never under-/over-sizes vs the sim.
        gui.dragFloat("Emission Rate", &emitter.emissionRate, 0.5f, 0.0f, 1000.0f);
        if (gui.isItemDeactivatedAfterEdit()) scene.resizeEmitterPool(bindless, buffers, node.particleIndex);
        gui.dragFloat2("Lifetime (min/max)", &emitter.lifeTime.x, 0.05f, 0.0f, 60.0f);
        if (gui.isItemDeactivatedAfterEdit()) scene.resizeEmitterPool(bindless, buffers, node.particleIndex);

        // The rest only flow through toGPU (re-uploaded each frame in ParticlePass::record),
        // so editing the CPU struct is enough — no buffer work.
        gui.sliderFloat("Spread Angle", &emitter.spreadAngle, 0.0f, 180.0f);
        gui.dragFloat("Speed Min", &emitter.speedMin, 0.05f, 0.0f, 1000.0f);
        gui.dragFloat("Speed Max", &emitter.speedMax, 0.05f, 0.0f, 1000.0f);
        gui.dragFloat2("Angular Vel (min/max)", &emitter.angularVelocityRandom.x, 0.05f);
        gui.sliderFloat("Drag", &emitter.drag, 0.0f, 5.0f);
        gui.dragFloat2("Size (min/max)", &emitter.sizeRandom.x, 0.01f, 0.0f, 100.0f);
        // Global transparency multiplier applied to every particle's alpha in the draw shader.
        gui.sliderFloat("Opacity", &emitter.opacity, 0.0f, 1.0f);

        // Lit billboards sample scene lighting in the draw shader.
        gui.checkbox("Lit", &emitter.lit);
        // Spherical-impostor bulge for lit shading: 0 = flat card, 1 = sphere, >1 exaggerates.
        if (emitter.lit)
            gui.sliderFloat("Roundness", &emitter.sphereRoundness, 0.0f, 2.0f);

        // Volumetric: particles inject density into the froxel grid (VolumetricsPass pass B).
        // Density Range is the per-particle density and only matters when this is on.
        gui.checkbox("Volumetric", &emitter.volumetric);
        if (emitter.volumetric) {
            gui.dragFloat2("Density Range", &emitter.densityRange.x, 0.01f, 0.0f, 1.0f);
            gui.dragFloat("Phase", &emitter.volumePhase, 0.01f, 0.0f, 1.0f);
            // Sphere: view-independent radial density (correct when you're inside / fly through it).
            // Off = textured billboard (samples the sprite's alpha, but only reads right from outside).
            gui.checkbox("Sphere (fly-through)", &emitter.volumetricSphere);
        }
        // Soft particles: depth-fade where the billboard meets scene geometry.
        gui.checkbox("Soft Particles", &emitter.softParticle);
        if (emitter.softParticle)
            gui.sliderFloat("Soft Radius", &emitter.softRadius, 0.01f, 10.0f);

        // Texture picker (like the material editor): seed the buffer from the current texture once,
        // then Browse/type a path and Apply to (re)load it.
        auto& state = getNodeGuiState(node.nodeIndex);
        if (!state.emitterTexInit) {
            std::string cur = scene.assetManager.getTexturePathFromIndex(emitter.textureIndex);
            strncpy(state.emitterTexBuffer, cur.c_str(), sizeof(state.emitterTexBuffer) - 1);
            state.emitterTexBuffer[sizeof(state.emitterTexBuffer) - 1] = '\0';
            state.emitterTexInit = true;
        }
        gui.setNextItemWidth(200);
        gui.inputText("Texture", state.emitterTexBuffer, sizeof(state.emitterTexBuffer));
        browseButton(gui, "emitterTex", state.emitterTexBuffer, sizeof(state.emitterTexBuffer));        
        if (gui.button("Apply Texture##emitter") && strlen(state.emitterTexBuffer) > 0) {
            try { emitter.textureIndex = scene.assetManager.loadTextureFromFile(state.emitterTexBuffer); }
            catch (...) {}
        }
        gui.dragFloat2("Emissive Range", &emitter.emissiveRange.x, 0.01f, 0.0f, 10.0f);

    }
}

void showNodeTransformInfo(GUI& gui, Node& node, Scene& scene) {

    //Gizmos::drawSDFArrow(node.getWorldPosition(),node.right(),0.01f,0.25f,glm::vec4(1,0,0,1));
    //Gizmos::drawSDFArrow(node.getWorldPosition(),node.up(),0.01f,0.25f,glm::vec4(0,1,0,1));
    //Gizmos::drawSDFArrow(node.getWorldPosition(),node.forward(),0.01f,0.25f,glm::vec4(0,0,1,1));

    gui.textf("Parent: %s", scene.sceneGraph.getNode(node.parentIndex).name.c_str());

    const char* currentParentName = scene.sceneGraph.getNode(node.parentIndex).name.c_str();
    if (gui.beginCombo("Change Parent", currentParentName)) {
        auto& nodes = scene.sceneGraph.getNodes();
        for (uint32_t i = SceneGraph::ROOT_INDEX; i <= scene.sceneGraph.getLastNode(); i++) {
            if (!scene.sceneGraph.isNodeValid(i)) continue;
            if (i == node.nodeIndex || i == node.parentIndex) continue;
            // Skip non user-facing nodes
            if(nodes[i].internal) continue;
            // Skip descendants to prevent cycles
            bool isDescendant = false;
            for (uint32_t p = nodes[i].parentIndex; p != 0; p = nodes[p].parentIndex) {
                if (p == node.nodeIndex) { isDescendant = true; break; }
            }
            if (isDescendant) continue;

            if (gui.comboItem(nodes[i].name.c_str(), false)) {
                scene.sceneGraph.reparent(node.nodeIndex, i, true);
            }
        }
        gui.endCombo();
    }

    gui.textf("position: ");
    gui.setNextItemWidth(80);
    gui.dragFloat("Pos X", &node.relativePosition.x, 0.1);
    gui.setNextItemWidth(80);
    gui.dragFloat("Pos Y", &node.relativePosition.y, 0.1);
    gui.setNextItemWidth(80);
    gui.dragFloat("Pos Z", &node.relativePosition.z, 0.1);

    gui.textf("rotation: ");

    // absolute world Euler angles
    gui.textf("Euler");
    glm::vec3 worldEulerDegrees = glm::degrees(node.worldRotationEuler);
    bool absoluteRotationChanged = false;
    gui.sameLine();
    gui.textf("Delta");
    static glm::vec3 worldRotationDelta = glm::vec3(0);
    bool deltaRotationChanged = false;

    gui.setNextItemWidth(80);
    absoluteRotationChanged |= gui.dragFloat("X##abs", &worldEulerDegrees.x, 0.5f, -180.0f, 180.0f, "%.1f°");
    gui.sameLine();
    gui.setNextItemWidth(80);
    deltaRotationChanged |= gui.dragFloat("X##delta", &worldRotationDelta.x, 0.5f, 0.0f, 0.0f, "%.1f°");

    gui.setNextItemWidth(80);
    absoluteRotationChanged |= gui.dragFloat("Y##abs", &worldEulerDegrees.y, 0.5f, -180.0f, 180.0f, "%.1f°");
    gui.sameLine();
    gui.setNextItemWidth(80);
    deltaRotationChanged |= gui.dragFloat("Y##delta", &worldRotationDelta.y, 0.5f, 0.0f, 0.0f, "%.1f°");

    gui.setNextItemWidth(80);
    absoluteRotationChanged |= gui.dragFloat("Z##abs", &worldEulerDegrees.z, 0.5f, -180.0f, 180.0f, "%.1f°");
    gui.sameLine();
    gui.setNextItemWidth(80);
    deltaRotationChanged |= gui.dragFloat("Z##delta", &worldRotationDelta.z, 0.5f, 0.0f, 0.0f, "%.1f°");

    if (absoluteRotationChanged) {
        node.worldRotationEuler = glm::radians(worldEulerDegrees);

        glm::quat qX = glm::angleAxis(node.worldRotationEuler.x, glm::vec3(1, 0, 0));
        glm::quat qY = glm::angleAxis(node.worldRotationEuler.y, glm::vec3(0, 1, 0));
        glm::quat qZ = glm::angleAxis(node.worldRotationEuler.z, glm::vec3(0, 0, 1));
        glm::quat desiredWorldRotation = qZ * qY * qX;

        if (node.parentIndex != 0) {
            auto& nodes = scene.sceneGraph.getNodes();
            glm::quat parentWorldRotation = nodes[node.parentIndex].getWorldRotation();
            node.relativeRotation = glm::inverse(parentWorldRotation) * desiredWorldRotation;
        } else {
            node.relativeRotation = desiredWorldRotation;
        }
    }

    if (deltaRotationChanged) {
        glm::quat currentWorld = node.getWorldRotation();
        glm::quat newWorld = currentWorld;

        if (worldRotationDelta.x != 0.0f) {
            glm::quat deltaRot = glm::angleAxis(glm::radians(worldRotationDelta.x), glm::vec3(1, 0, 0));
            newWorld = deltaRot * newWorld;
            worldRotationDelta.x = 0.0f;
        }
        if (worldRotationDelta.y != 0.0f) {
            glm::quat deltaRot = glm::angleAxis(glm::radians(worldRotationDelta.y), glm::vec3(0, 1, 0));
            newWorld = deltaRot * newWorld;
            worldRotationDelta.y = 0.0f;
        }
        if (worldRotationDelta.z != 0.0f) {
            glm::quat deltaRot = glm::angleAxis(glm::radians(worldRotationDelta.z), glm::vec3(0, 0, 1));
            newWorld = deltaRot * newWorld;
            worldRotationDelta.z = 0.0f;
        }

        if (node.parentIndex != 0) {
            auto& nodes = scene.sceneGraph.getNodes();
            node.relativeRotation = glm::inverse(nodes[node.parentIndex].getWorldRotation()) * newWorld;
        } else {
            node.relativeRotation = newWorld;
        }
    }

    gui.textf("scale: ");
    gui.setNextItemWidth(80);
    gui.dragFloat("Scale X", &node.relativeScale.x, 0.1 * node.relativeScale.x);
    gui.setNextItemWidth(80);
    gui.dragFloat("Scale Y", &node.relativeScale.y, 0.1 * node.relativeScale.y);
    gui.setNextItemWidth(80);
    gui.dragFloat("Scale Z", &node.relativeScale.z, 0.1 * node.relativeScale.z);
}

void showTransformModifiers(GUI& gui) {
    GUIStyle& style = gui.getStyle();
    // widen to fit whichever increment slider is showing — 8px of window padding per side. With
    // both off it only has to hold the checkbox: its box, the gap, and the caption.
    bool anySnap = Manip::doSnap || Manip::doSnapAngle;
    float width = anySnap ? style.itemWidth + 16.0f : gui.measureText("Angle Snap").x + 48.0f;
    gui.setNextWindowSize(glm::vec2(width, 200));
    if (gui.beginWindow("Transform", nullptr,glm::vec2(50,200) , glm::vec2(0, 0),
                        GUIAnchor::Right | GUIAnchor::Center, GUIWindowFixed | GUIWindowNoSavedSettings)) {

        gui.checkbox("Move Snap", &Manip::doSnap);
        if( Manip::doSnap) {
            gui.sliderFloat("Increment",&Manip::snapIncrement,0.0f,5.0f,"%.1f",false);
        }
        gui.separator();
        gui.checkbox("Angle Snap", &Manip::doSnapAngle);
        if (Manip::doSnapAngle) {
            gui.sliderFloat("Degrees", &Manip::snapAngleIncrement, 0.0f, 90.0f, "%.1f", false);
        }
        gui.endWindow();
    }
}
