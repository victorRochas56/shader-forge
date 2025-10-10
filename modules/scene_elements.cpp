#include "scene_elements.hpp"
#include "input.hpp"
#include "renderer.hpp"

Node::Node(Renderer* pRenderer, uint32_t arrayIndex, Node* parent, glm::vec3 position, glm::quat rotation, glm::vec3 scale, bool keepWorldTransform) {

    renderer = pRenderer;
    resourceManager = &renderer->getResourceManager();
    nodeIndex = arrayIndex;

    relativePosition = position;
    relativeRotation = rotation;
    relativeScale = scale;
    relativeRotationEuler = glm::eulerAngles(rotation);
    worldRotationEuler = glm::eulerAngles(rotation);

    if (parent == nullptr && renderer->getRootNode() != nullptr) { // conv to scene->
        parent = renderer->getRootNode();                          // conv to scene->
    }
    // Handle keepWorldTransform logic
    if (keepWorldTransform && parent != nullptr) {
        glm::mat4 desiredWorldTransform = makeTransform(position, rotation, scale);
        // Calculate relative transform
        glm::mat4 parentInverse = glm::inverse(parent->worldTransform);
        glm::mat4 relativeTransformMatrix = parentInverse * desiredWorldTransform;
        // Extract relative components
        relativePosition = glm::vec3(relativeTransformMatrix[3]);
        relativeScale.x = glm::length(glm::vec3(relativeTransformMatrix[0]));
        relativeScale.y = glm::length(glm::vec3(relativeTransformMatrix[1]));
        relativeScale.z = glm::length(glm::vec3(relativeTransformMatrix[2]));
        glm::mat3 rotMatrix;
        rotMatrix[0] = glm::vec3(relativeTransformMatrix[0]) / relativeScale.x;
        rotMatrix[1] = glm::vec3(relativeTransformMatrix[1]) / relativeScale.y;
        rotMatrix[2] = glm::vec3(relativeTransformMatrix[2]) / relativeScale.z;
        relativeRotation = glm::quat_cast(rotMatrix);
        relativeRotationEuler = glm::eulerAngles(relativeRotation);
        worldRotationEuler = glm::eulerAngles(parent->getWorldRotation() * relativeRotation);
    }
    modelMatrixIndex = renderer->getDescriptorSet().allocateFixedBuffer(renderer->getModelMatrixBufferIndex(), glm::mat4(1.0f));

    if (parent != nullptr) {
        parent->addChild(this);
    } else {
        update();
    }
    std::cout << "Created node " << arrayIndex << " with modelMatrixIndex=" << modelMatrixIndex << ", scale=(" << scale.x << "," << scale.y << "," << scale.z << ")" << std::endl;
}

void Node::update() {
    localTransform = makeTransform(relativePosition, relativeRotation, relativeScale);
    if (parent != nullptr) {
        worldTransform = parent->worldTransform * localTransform;
        glm::quat worldRotation = parent->getWorldRotation() * relativeRotation;
        worldRotationEuler = glm::eulerAngles(worldRotation);
    } else {
        worldTransform = localTransform;
        worldRotationEuler = glm::eulerAngles(relativeRotation);
    }

    renderer->getDescriptorSet().updateFixedBuffer(renderer->getModelMatrixBufferIndex(), modelMatrixIndex, worldTransform);
    renderer->getLights()[lightIndex].direction = glm::normalize(worldTransform[0]);
    for (Node* child : children) {
        child->update();
    }
}

void Node::addMesh(uint32_t meshIndex) {
    if (this->meshIndex < renderer->getMeshes().size()) {
        renderer->getMeshes()[meshIndex].refCount--;
        if (renderer->getMeshes()[meshIndex].refCount <= 0) {
            renderer->getMeshes()[meshIndex].freed = true;
        }
    }
    this->meshIndex = meshIndex;
    renderer->getMeshes()[meshIndex].refCount++;
}

void Node::addMaterial(uint32_t index, uint32_t materialIndex) {
    if (index >= renderer->getMeshes()[meshIndex].subMeshes.size() || index < 0) {
        throw std::runtime_error("tried adding material to invalid material slot on node mesh!");
    } else {
        renderer->addMeshToShader(this, renderer->getMeshes()[meshIndex].subMeshes[index], renderer->getMaterials()[materialIndex].shaderSource,
                                  renderer->getMaterials()[materialIndex]);
        materialIndices.push_back(materialIndex);
    }
}

void Node::addLight(Light light) {
    light.modelMatrixIndex = modelMatrixIndex;
    lightIndex = renderer->getDescriptorSet().allocateFixedBuffer<Light>(renderer->getLightBufferIndex(), light);
    renderer->getLights()[lightIndex] = light; // conv to scene
    std::cout << "added light at index : " << lightIndex << std::endl;
}

void Node::showMaterialDialog() {
    if (materialList.size() != renderer->getMaterials().size()) {
        materialList.clear();
        selectedMaterials.clear();

        for (int i = 0; i < renderer->getMaterials().size(); i++) {
            std::string textOption = "Material " + std::to_string(i);
            materialList.push_back(textOption);
        }

        for (int n = 0; n < renderer->getMeshes()[meshIndex].subMeshes.size(); n++) {
            selectedMaterials.push_back(materialIndices[n]);
        }
    }
    ImGui::Begin("Change Material");
    for (int i = 0; i < renderer->getMeshes()[meshIndex].subMeshes.size(); i++) {
        std::string displayText = "material slot " + std::to_string(i);
        ImGui::SetNextItemWidth(100);

        if (ImGui::BeginCombo(displayText.c_str(), materialList[selectedMaterials[i]].c_str())) {
            for (int n = 0; n < materialList.size(); n++) {
                bool is_selected = (selectedMaterials[i] == n);
                if (ImGui::Selectable(materialList[n].c_str(), is_selected)) {
                    // Check if selection changed when item is clicked
                    if (n != selectedMaterials[i]) {
                        renderer->removeMeshFromShader(this, renderer->getMeshes()[meshIndex].subMeshes[i], renderer->getMaterials()[materialIndices[i]].shaderSource,
                                                       renderer->getMaterials()[materialIndices[i]]);
                        materialIndices[i] = n;
                        renderer->addMeshToShader(this, renderer->getMeshes()[meshIndex].subMeshes[i], renderer->getMaterials()[n].shaderSource, renderer->getMaterials()[n]);
                    }
                    selectedMaterials[i] = n;
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

void Node::showLightInfo() {

    if (lightIndex != MAX_LIGHTS && lightIndex < renderer->getLights().size()) {
        auto& light = renderer->getLights()[lightIndex];
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

            if (ImGui::Checkbox("Enable Shadows", &lightShadow)) {
                light.castsShadows = lightShadow ? 1 : 0;
                if (light.castsShadows == 1) {

                    for (int i = 0; i < light.numCascades; i++) {

                        vk::raii::Image shadowMapImage = nullptr;
                        vk::raii::DeviceMemory shadowMapMemory = nullptr;
                        resourceManager->createImage(light.shadowResolution, light.shadowResolution, 1, vk::SampleCountFlagBits::e1, vk::Format::eD32Sfloat,
                                                     vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
                                                     vk::MemoryPropertyFlagBits::eDeviceLocal, shadowMapImage, shadowMapMemory, 1);
                        vk::raii::ImageView shadowMapImageView = resourceManager->createImageView(shadowMapImage, vk::Format::eD32Sfloat, vk::ImageAspectFlagBits::eDepth);

                        light.cascades[i].shadowMapIndex =
                            renderer->getDescriptorSet().allocateTexture(std::move(shadowMapImage), std::move(shadowMapMemory), std::move(shadowMapImageView));
                    }
                } else {
                    for(int i = 0; i < light.numCascades; i++) {
                        renderer->getDescriptorSet().freeTexture(light.cascades[i].shadowMapIndex);
                    }
                }
            }
        }
        renderer->getDescriptorSet().updateFixedBuffer(renderer->getLightBufferIndex(), lightIndex, light);
    } else if (ImGui::Button("Add Light")) {
        Light light = {.type = LightType::Point, .range = 10, .intensity = 1, .color = glm::vec4(1, 1, 1, 1)};
        addLight(light);
    }
}

void Node::showTransformInfo() {
    ImGui::Text("position: ");
    ImGui::SetNextItemWidth(80);
    ImGui::DragFloat("Pos X", &relativePosition.x, 0.1);
    ImGui::SetNextItemWidth(80);
    ImGui::DragFloat("Pos Y", &relativePosition.y, 0.1);
    ImGui::SetNextItemWidth(80);
    ImGui::DragFloat("Pos Z", &relativePosition.z, 0.1);

    ImGui::Text("rotation: ");

    // absolute world Euler angles
    ImGui::Text("Euler");
    glm::vec3 worldEulerDegrees = glm::degrees(worldRotationEuler);
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
        worldRotationEuler = glm::radians(worldEulerDegrees);

        glm::quat qX = glm::angleAxis(worldRotationEuler.x, glm::vec3(1, 0, 0));
        glm::quat qY = glm::angleAxis(worldRotationEuler.y, glm::vec3(0, 1, 0));
        glm::quat qZ = glm::angleAxis(worldRotationEuler.z, glm::vec3(0, 0, 1));
        glm::quat desiredWorldRotation = qZ * qY * qX;

        if (parent != nullptr) {
            glm::quat parentWorldRotation = parent->getWorldRotation();
            relativeRotation = glm::inverse(parentWorldRotation) * desiredWorldRotation;
        } else {
            relativeRotation = desiredWorldRotation;
        }
    }

    if (deltaRotationChanged) {
        glm::quat currentWorld = getWorldRotation();
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

        if (parent != nullptr) {
            relativeRotation = glm::inverse(parent->getWorldRotation()) * newWorld;
        } else {
            relativeRotation = newWorld;
        }
    }

    ImGui::Text("scale: ");
    ImGui::SetNextItemWidth(80);
    ImGui::DragFloat("Scale X", &relativeScale.x, 0.1 * relativeScale.x);
    ImGui::SetNextItemWidth(80);
    ImGui::DragFloat("Scale Y", &relativeScale.y, 0.1 * relativeScale.y);
    ImGui::SetNextItemWidth(80);
    ImGui::DragFloat("Scale Z", &relativeScale.z, 0.1 * relativeScale.z);
}

void Node::showMeshInfo() {
    if (!changingMesh) {
        if (meshIndex < renderer->getMeshes().size()) {
            if (ImGui::Button(renderer->getMeshes()[meshIndex].sourceFile.c_str())) {
                changingMesh = true;
                textBuffer[0] = '\0';
            }
        } else {
            if (ImGui::Button("Add Mesh")) {
                changingMesh = true;
                textBuffer[0] = '\0';
            }
        }
    } else {
        InputManager::getInstance().canMove = false;
        ImGui::SetNextItemWidth(160);
        ImGui::InputText("mesh source", textBuffer, sizeof(textBuffer));

        if (ImGui::Button("Confirm")) {
            // remove old mesh from render queue for shader
            for (uint32_t index : materialIndices) {
                for (int i = 0; i < renderer->getMeshes()[meshIndex].subMeshes.size(); i++) {
                    renderer->removeMeshFromShader(this, renderer->getMeshes()[meshIndex].subMeshes[i], renderer->getMaterials()[index].shaderSource,
                                                   renderer->getMaterials()[index]);
                }
            }
            // load new mesh
            meshIndex = renderer->loadMeshFromFile(std::string(textBuffer));
            materialList.clear();
            selectedMaterials.clear();
            materialIndices.clear();
            for (int i = 0; i < renderer->getMeshes()[meshIndex].subMeshes.size(); i++) {
                addMaterial(i, renderer->getFallBackMaterial());
            }
#if DEBUG == 1
            std::cout << "mesh index " << meshIndex << std::endl;
#endif
            changingMesh = false;
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel")) {
            changingMesh = false;
        }
    }
    if (meshIndex < renderer->getMeshes().size()) {
        if (ImGui::Button("Assign Materials")) {
            changingMaterials = !changingMaterials;
        }
    }
}

glm::mat4 calculateLightSpaceMatrix(Light& light, Camera& camera) {

    glm::mat4 lightProjection = glm::ortho(-light.range, light.range, -light.range, light.range, camera.nearPlane, camera.farPlane);
    glm::vec3 lightPos = camera.position - light.direction * 0.5f * camera.farPlane;
    glm::mat4 lightView = glm::lookAt(lightPos, lightPos + light.direction, glm::vec3(0.0f, 1.0f, 0.0f));
    return lightProjection * lightView;
}

void calculateCascadedLightSpaceMatrices(Light& light, Camera& camera) {
    std::vector<float> cascadeSplits = {2.0f, 4.0f, 8.0f, 16.0f, 32.0f};

    std::vector<glm::mat4> cascadeMatrices;

    for (int i = 0; i < 4; i++) {

        Camera cascadeCamera = camera;
        cascadeCamera.nearPlane =  -50.0f;
        cascadeCamera.farPlane = 100.0f;

        Light cascadeLight = light;
        cascadeLight.range = 4 * cascadeSplits[i];
        
        glm::mat4 cascadeMatrix = calculateLightSpaceMatrix(cascadeLight, cascadeCamera);
        light.cascades[i].lightSpaceMatrix = cascadeMatrix;
        light.cascades[i].texelSize = cascadeSplits[i]/light.shadowResolution;
        light.cascades[i].splitDistance = cascadeSplits[i];
    }
}