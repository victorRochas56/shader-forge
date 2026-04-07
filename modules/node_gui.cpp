#include "node_gui.hpp"
#include "gui.hpp"
#include "input.hpp"
#include "node_ops.hpp"
#include "renderer.hpp"
#include "transform_system.hpp"

static std::unordered_map<uint32_t, NodeGuiState> guiStates;

NodeGuiState& getNodeGuiState(uint32_t nodeIndex) { return guiStates[nodeIndex]; }

void showNodeInfo(Node& node, Renderer& renderer) {
    auto& state = getNodeGuiState(node.nodeIndex);

    ImGui::Begin("selected node");
    ImGui::Text(node.name.c_str());

    showNodeMeshInfo(node, renderer);
    if (state.changingMaterials) {
        showNodeMaterialDialog(node, renderer);
    }
    showNodeLightInfo(node, renderer);
    showNodeTransformInfo(node, renderer);
    TransformSystem::updateAll(node, renderer.sceneGraph.getNodes(), renderer.getDescriptorSet(), renderer.getModelMatrixBufferIndex(), renderer.assetManager.meshes,
                               renderer.getLightsMutable());
    ImGui::End();
}

void showNodeMeshInfo(Node& node, Renderer& renderer) {
    auto& state = getNodeGuiState(node.nodeIndex);

    if (!state.changingMesh) {
        if (node.meshIndex < renderer.assetManager.meshes.size()) {
            if (ImGui::Button(renderer.assetManager.meshes[node.meshIndex].sourceFile.c_str())) {
                state.changingMesh = true;
                state.textBuffer[0] = '\0';
            }
        } else {
            if (ImGui::Button("Add Mesh")) {
                state.changingMesh = true;
                state.textBuffer[0] = '\0';
            }
        }
    } else {
        InputManager::getInstance().canMove = false;
        ImGui::SetNextItemWidth(160);
        ImGui::InputText("mesh source", state.textBuffer, sizeof(state.textBuffer));
        browseButton("mesh", state.textBuffer, sizeof(state.textBuffer));

        if (ImGui::Button("Confirm")) {
            // remove old mesh from render queue for shader
            auto& matIndices = node.getMaterialIndices();
            for (int i = 0; i < node.materialIndexCount; i++) {
                int index = matIndices[i];
                for (int subMeshIdx = 0; subMeshIdx < renderer.assetManager.meshes[node.meshIndex].subMeshes.size(); subMeshIdx++) {
                    renderer.removeMeshFromShader(&node, renderer.assetManager.meshes[node.meshIndex].subMeshes[subMeshIdx],
                                                  renderer.getMaterials()[index].shaderSource, renderer.getMaterials()[index]);
                }
            }
            // load new mesh
            node.meshIndex = renderer.assetManager.loadMeshFromFile(std::string(state.textBuffer));
            state.materialList.clear();
            state.selectedMaterials.clear();
            node.materialIndexCount = 0;
            for (int i = 0; i < renderer.assetManager.meshes[node.meshIndex].subMeshes.size(); i++) {
                NodeOps::assignMaterial(node, i, renderer.getFallBackMaterial(), renderer);
            }
#if DEBUG == 1
            std::cout << "mesh index " << node.meshIndex << std::endl;
#endif
            state.changingMesh = false;
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel")) {
            state.changingMesh = false;
        }
    }
    if (node.meshIndex < renderer.assetManager.meshes.size()) {
        if (ImGui::Button("Assign Materials")) {
            state.changingMaterials = !state.changingMaterials;
        }
    }
}

void showNodeMaterialDialog(Node& node, Renderer& renderer) {
    auto& state = getNodeGuiState(node.nodeIndex);

    if (state.materialList.size() != renderer.getMaterials().size()) {
        state.materialList.clear();
        state.selectedMaterials.clear();

        for (int i = 0; i < renderer.getMaterials().size(); i++) {
            const auto& material = renderer.getMaterials()[i];
            std::string textOption = material.name.empty() ? ("Material " + std::to_string(i)) : material.name;
            state.materialList.push_back(textOption);
        }

        for (int n = 0; n < renderer.assetManager.meshes[node.meshIndex].subMeshes.size(); n++) {
            state.selectedMaterials.push_back(node.materialIndices[n]);
        }
    }

    ImGui::Begin("Change Material");

    // Group submeshes by their original material ID from the OBJ file
    std::map<int, std::vector<uint32_t>> originalMatIdToSubmeshes;
    const auto& mesh = renderer.assetManager.meshes[node.meshIndex];

    for (int i = 0; i < mesh.subMeshes.size(); i++) {
        int originalMatId = (i < mesh.originalMaterialIds.size()) ? mesh.originalMaterialIds[i] : -1;
        originalMatIdToSubmeshes[originalMatId].push_back(i);
    }

    // Show one material slot per unique original material
    for (const auto& [originalMatId, submeshIndices] : originalMatIdToSubmeshes) {
        // Use the first submesh's current material as the representative
        uint32_t firstSubmeshIdx = submeshIndices[0];
        uint32_t currentMatIdx = node.materialIndices[firstSubmeshIdx];

        // Get the original material name from the OBJ file
        std::string originalMatName = (firstSubmeshIdx < mesh.originalMaterialNames.size()) ? mesh.originalMaterialNames[firstSubmeshIdx]
                                                                                            : ("Material_" + std::to_string(originalMatId));

        std::string displayText = originalMatName;
        if (submeshIndices.size() > 1) {
            displayText += " (" + std::to_string(submeshIndices.size()) + " submeshes)";
        }

        ImGui::SetNextItemWidth(200);

        if (ImGui::BeginCombo(displayText.c_str(), state.materialList[currentMatIdx].c_str())) {
            for (int n = 0; n < state.materialList.size(); n++) {
                bool is_selected = (currentMatIdx == n);
                if (ImGui::Selectable(state.materialList[n].c_str(), is_selected)) {
                    // Check if selection changed
                    if (n != currentMatIdx) {
                        // Update all submeshes that share this original material ID
                        for (uint32_t subMeshIdx : submeshIndices) {
                            renderer.removeMeshFromShader(&node, mesh.subMeshes[subMeshIdx],
                                                          renderer.getMaterials()[node.materialIndices[subMeshIdx]].shaderSource,
                                                          renderer.getMaterials()[node.materialIndices[subMeshIdx]]);
                            node.materialIndices[subMeshIdx] = n;
                            renderer.addMeshToShader(&node, mesh.subMeshes[subMeshIdx], renderer.getMaterials()[n].shaderSource,
                                                     renderer.getMaterials()[n]);
                            state.selectedMaterials[subMeshIdx] = n;
                        }
                    }
                }
                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
    }
    ImGui::End();
}

void showNodeLightInfo(Node& node, Renderer& renderer) {
    auto& state = getNodeGuiState(node.nodeIndex);

    if (node.lightIndex != MAX_LIGHTS && node.lightIndex < renderer.getLights().size()) {
        auto& light = renderer.getLight(node.lightIndex);
        const char* lightTypeNames[] = {"Point", "Directional", "Spot", "Area"};
        if (ImGui::BeginCombo("Light Type", lightTypeNames[static_cast<int>(light.type)])) {
            for (int i = 0; i < static_cast<int>(LightType::COUNT); i++) {
                bool isSelected = (static_cast<int>(light.type) == i);
                if (ImGui::Selectable(lightTypeNames[i], isSelected)) {
                    light.type = static_cast<LightType>(i);
                }
                if (isSelected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
        if (light.type != LightType::Directional) {
            ImGui::DragFloat("range", &light.range);
        }
        ImGui::DragFloat("intensity", &light.intensity);
        float colR = light.color.r;
        float colG = light.color.g;
        float colB = light.color.b;
        float color[3] = {colR, colG, colB};
        ImGui::ColorPicker3("color", color);
        light.color = glm::vec4(color[0], color[1], color[2], 1);

        if (light.type == LightType::Directional) {

            ImGui::DragInt("show cascades", &light.showCascades, 1, 0, 1);

            for (int i = 0; i < light.numCascades; i++) {
                std::string cascadeLabel = "cascade ";
                cascadeLabel += std::to_string(i);
                ImGui::DragFloat(cascadeLabel.c_str(), &light.cascades[i].splitDistance);
            }
        }

        if (light.type == LightType::Directional || light.type == LightType::Point) {
            state.lightShadow = light.castsShadows;
            if (ImGui::Checkbox("Enable Shadows", &state.lightShadow)) {
                if (state.lightShadow) {
                    NodeOps::enableLightShadows(light, node.name, renderer);
                } else {
                    NodeOps::disableLightShadows(light, renderer);
                }
            }
        }
        // Update light in all frame sections of the buffer
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            renderer.getDescriptorSet().updateFixedBufferWithOffset<GPULight>(renderer.getLightBufferIndex(), node.lightIndex, light.toGPU(), i);
        }
        
        Gizmos::drawSphere(node.getWorldPosition(),light.range, glm::vec4(1,1,0,1));


    } else if (ImGui::Button("Add Light")) {
        Light light = {.type = LightType::Point, .range = 10, .intensity = 1, .color = glm::vec4(1, 1, 1, 1)};
        NodeOps::assignLight(node, light, renderer);
    }

}

void showNodeTransformInfo(Node& node, Renderer& renderer) {

    ImGui::Text("Parent: %s", renderer.sceneGraph.getNode(node.parentIndex).name.c_str());

    const char* currentParentName = renderer.sceneGraph.getNode(node.parentIndex).name.c_str();
    if (ImGui::BeginCombo("Change Parent", currentParentName)) {
        auto& nodes = renderer.sceneGraph.getNodes();
        for (uint32_t i = SceneGraph::ROOT_INDEX; i <= renderer.sceneGraph.getLastNode(); i++) {
            if (i == node.nodeIndex || i == node.parentIndex) continue;
            // Skip descendants to prevent cycles
            bool isDescendant = false;
            for (uint32_t p = nodes[i].parentIndex; p != 0; p = nodes[p].parentIndex) {
                if (p == node.nodeIndex) { isDescendant = true; break; }
            }
            if (isDescendant) continue;

            if (ImGui::Selectable(nodes[i].name.c_str(), false)) {
                renderer.sceneGraph.reparent(node.nodeIndex, i, true);
            }
        }
        ImGui::EndCombo();
    }

    ImGui::Text("position: ");
    ImGui::SetNextItemWidth(80);
    ImGui::DragFloat("Pos X", &node.relativePosition.x, 0.1);
    ImGui::SetNextItemWidth(80);
    ImGui::DragFloat("Pos Y", &node.relativePosition.y, 0.1);
    ImGui::SetNextItemWidth(80);
    ImGui::DragFloat("Pos Z", &node.relativePosition.z, 0.1);

    ImGui::Text("rotation: ");

    // absolute world Euler angles
    ImGui::Text("Euler");
    glm::vec3 worldEulerDegrees = glm::degrees(node.worldRotationEuler);
    bool absoluteRotationChanged = false;
    ImGui::SameLine();
    ImGui::Text("Delta");
    static glm::vec3 worldRotationDelta = glm::vec3(0);
    bool deltaRotationChanged = false;

    ImGui::SetNextItemWidth(80);
    absoluteRotationChanged |= ImGui::DragFloat("X##abs", &worldEulerDegrees.x, 0.5f, -180.0f, 180.0f, "%.1f°");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80);
    deltaRotationChanged |= ImGui::DragFloat("X##delta", &worldRotationDelta.x, 0.5f, 0.0f, 0.0f, "%.1f°");

    ImGui::SetNextItemWidth(80);
    absoluteRotationChanged |= ImGui::DragFloat("Y##abs", &worldEulerDegrees.y, 0.5f, -180.0f, 180.0f, "%.1f°");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80);
    deltaRotationChanged |= ImGui::DragFloat("Y##delta", &worldRotationDelta.y, 0.5f, 0.0f, 0.0f, "%.1f°");

    ImGui::SetNextItemWidth(80);
    absoluteRotationChanged |= ImGui::DragFloat("Z##abs", &worldEulerDegrees.z, 0.5f, -180.0f, 180.0f, "%.1f°");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80);
    deltaRotationChanged |= ImGui::DragFloat("Z##delta", &worldRotationDelta.z, 0.5f, 0.0f, 0.0f, "%.1f°");

    if (absoluteRotationChanged) {
        node.worldRotationEuler = glm::radians(worldEulerDegrees);

        glm::quat qX = glm::angleAxis(node.worldRotationEuler.x, glm::vec3(1, 0, 0));
        glm::quat qY = glm::angleAxis(node.worldRotationEuler.y, glm::vec3(0, 1, 0));
        glm::quat qZ = glm::angleAxis(node.worldRotationEuler.z, glm::vec3(0, 0, 1));
        glm::quat desiredWorldRotation = qZ * qY * qX;

        if (node.parentIndex != 0) {
            auto& nodes = renderer.sceneGraph.getNodes();
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
            auto& nodes = renderer.sceneGraph.getNodes();
            node.relativeRotation = glm::inverse(nodes[node.parentIndex].getWorldRotation()) * newWorld;
        } else {
            node.relativeRotation = newWorld;
        }
    }

    ImGui::Text("scale: ");
    ImGui::SetNextItemWidth(80);
    ImGui::DragFloat("Scale X", &node.relativeScale.x, 0.1 * node.relativeScale.x);
    ImGui::SetNextItemWidth(80);
    ImGui::DragFloat("Scale Y", &node.relativeScale.y, 0.1 * node.relativeScale.y);
    ImGui::SetNextItemWidth(80);
    ImGui::DragFloat("Scale Z", &node.relativeScale.z, 0.1 * node.relativeScale.z);
}
