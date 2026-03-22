#pragma once
#include "gui.hpp"
#include "scene_elements.hpp"
#include "input.hpp"
#include "renderer.hpp"

#define GLM_DEPTH_ZERO_TO_ONE

/*
all node related logic and code that interacts with renderer
*/

Node::Node(Renderer* pRenderer, uint32_t arrayIndex, Node* parent, glm::vec3 position, glm::quat rotation, glm::vec3 scale, bool keepWorldTransform) {

    renderer = pRenderer;
    resourceManager = &renderer->getResourceManager();
    nodeIndex = arrayIndex;

    relativePosition = position;
    relativeRotation = rotation;
    relativeScale = scale;
    relativeRotationEuler = glm::eulerAngles(rotation);
    worldRotationEuler = glm::eulerAngles(rotation);

    if (parent == nullptr && renderer->sceneGraph.getRootNode() != nullptr) { // conv to scene->
        parent = renderer->sceneGraph.getRootNode();                          // conv to scene->
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
    // Allocate once - same index used across all frames (offset handles frame separation)
    uint32_t singleIndex = renderer->getDescriptorSet().allocateFixedBuffer(renderer->getModelMatrixBufferIndex(), glm::mat4(1.0f));
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        modelMatrixIndices[i] = singleIndex;
    }
    if (parent != nullptr) {
        parent->addChild(this);
    } else {
        update();
    }
    std::cout << "Created node " << arrayIndex << ", position=("<<position.x<<","<<position.y<<","<<position.z<<") " << ", scale=(" << scale.x << "," << scale.y << "," << scale.z << ")" << std::endl;
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

    // Update model matrix for all frames using dynamic offset approach
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        renderer->getDescriptorSet().updateFixedBufferWithOffset(renderer->getModelMatrixBufferIndex(), modelMatrixIndices[i], worldTransform, i);
    }
    if (lightIndex != MAX_LIGHTS && renderer->getLights().contains(lightIndex)) {
        renderer->getLight(lightIndex).direction = glm::normalize(worldTransform[0]);
    }

    // Update world-space bounding box if node has a mesh
    if (meshIndex < renderer->assetManager.meshes.size() && boundingBoxValid) {
        const auto& mesh = renderer->assetManager.meshes[meshIndex];
        glm::vec3 corners[8] = {
            glm::vec3(mesh.boundingBoxMin.x, mesh.boundingBoxMin.y, mesh.boundingBoxMin.z), glm::vec3(mesh.boundingBoxMax.x, mesh.boundingBoxMin.y, mesh.boundingBoxMin.z),
            glm::vec3(mesh.boundingBoxMin.x, mesh.boundingBoxMax.y, mesh.boundingBoxMin.z), glm::vec3(mesh.boundingBoxMax.x, mesh.boundingBoxMax.y, mesh.boundingBoxMin.z),
            glm::vec3(mesh.boundingBoxMin.x, mesh.boundingBoxMin.y, mesh.boundingBoxMax.z), glm::vec3(mesh.boundingBoxMax.x, mesh.boundingBoxMin.y, mesh.boundingBoxMax.z),
            glm::vec3(mesh.boundingBoxMin.x, mesh.boundingBoxMax.y, mesh.boundingBoxMax.z), glm::vec3(mesh.boundingBoxMax.x, mesh.boundingBoxMax.y, mesh.boundingBoxMax.z)};

        boundingBoxMin = glm::vec3(std::numeric_limits<float>::max());
        boundingBoxMax = glm::vec3(std::numeric_limits<float>::lowest());

        for (const auto& corner : corners) {
            glm::vec3 worldCorner = glm::vec3(worldTransform * glm::vec4(corner, 1.0f));
            boundingBoxMin = glm::min(boundingBoxMin, worldCorner);
            boundingBoxMax = glm::max(boundingBoxMax, worldCorner);
        }
    }

    for (Node* child : children) {
        child->update();
    }
}

void Node::addMesh(uint32_t meshIndex) {
    if (this->meshIndex < renderer->assetManager.meshes.size()) {
        renderer->assetManager.meshes[meshIndex].refCount--;
        if (renderer->assetManager.meshes[meshIndex].refCount <= 0) {
            renderer->assetManager.meshes[meshIndex].freed = true;
        }
    }
    this->meshIndex = meshIndex;
    renderer->assetManager.meshes[meshIndex].refCount++;

    // Calculate world-space bounding box from mesh local-space bounding box
    const auto& mesh = renderer->assetManager.meshes[meshIndex];
    // Transform all 8 corners of the AABB and find new min/max
    glm::vec3 corners[8] = {
        glm::vec3(mesh.boundingBoxMin.x, mesh.boundingBoxMin.y, mesh.boundingBoxMin.z), glm::vec3(mesh.boundingBoxMax.x, mesh.boundingBoxMin.y, mesh.boundingBoxMin.z),
        glm::vec3(mesh.boundingBoxMin.x, mesh.boundingBoxMax.y, mesh.boundingBoxMin.z), glm::vec3(mesh.boundingBoxMax.x, mesh.boundingBoxMax.y, mesh.boundingBoxMin.z),
        glm::vec3(mesh.boundingBoxMin.x, mesh.boundingBoxMin.y, mesh.boundingBoxMax.z), glm::vec3(mesh.boundingBoxMax.x, mesh.boundingBoxMin.y, mesh.boundingBoxMax.z),
        glm::vec3(mesh.boundingBoxMin.x, mesh.boundingBoxMax.y, mesh.boundingBoxMax.z), glm::vec3(mesh.boundingBoxMax.x, mesh.boundingBoxMax.y, mesh.boundingBoxMax.z)};

    boundingBoxMin = glm::vec3(std::numeric_limits<float>::max());
    boundingBoxMax = glm::vec3(std::numeric_limits<float>::lowest());

    for (const auto& corner : corners) {
        glm::vec3 worldCorner = glm::vec3(worldTransform * glm::vec4(corner, 1.0f));
        boundingBoxMin = glm::min(boundingBoxMin, worldCorner);
        boundingBoxMax = glm::max(boundingBoxMax, worldCorner);
    }
    boundingBoxValid = true;
}

void Node::addMaterial(uint32_t index, uint32_t materialIndex) {
    if (index >= renderer->assetManager.meshes[meshIndex].subMeshes.size() || index < 0) {
        throw std::runtime_error("tried adding material to invalid material slot on node mesh!");
    } else {
        renderer->addMeshToShader(this, renderer->assetManager.meshes[meshIndex].subMeshes[index], renderer->getMaterials()[materialIndex].shaderSource,
                                  renderer->getMaterials()[materialIndex]);
        materialIndices.push_back(materialIndex);
    }
}

void Node::addLight(Light light) {
    light.nodeIndex = nodeIndex;
    light.modelMatrixIndex = modelMatrixIndices[0];  // Store this node's matrix index for light position/direction

    // shadow mapping (PCF - Percentage Closer Filtering)
    if (light.castsShadows == 1) {
        for (int i = 0; i < light.numCascades; i++) {

            vk::raii::Image shadowMapImage = nullptr;
            vk::raii::DeviceMemory shadowMapMemory = nullptr;
            // PCF uses R32F format to store raw depth values
            // Allows for efficient depth comparison and filtering
            resourceManager->createImage(light.shadowResolution, light.shadowResolution, 1, vk::SampleCountFlagBits::e1, vk::Format::eR32Sfloat, vk::ImageTiling::eOptimal,
                                         vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, shadowMapImage,
                                         shadowMapMemory, 1);
            vk::raii::ImageView shadowMapImageView = resourceManager->createImageView(shadowMapImage, vk::Format::eR32Sfloat, vk::ImageAspectFlagBits::eColor);

            // Initial layout transition from undefined to shader read optimal
            resourceManager->transitionImageLayout(nullptr, shadowMapImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal);

            light.cascades[i].shadowMapIndex = renderer->getDescriptorSet().allocateTexture(std::move(shadowMapImage), std::move(shadowMapMemory), std::move(shadowMapImageView),"internal/"+name+"/csm_"+std::to_string(i));
        }
    }
    // Allocate once in the single buffer that spans all frames
    // We only need to allocate once since all frames use the same buffer with offsets
    lightIndex = renderer->getDescriptorSet().allocateFixedBuffer<Light>(renderer->getLightBufferIndex(), light);
    renderer->addLight(lightIndex, light); // conv to scene
    std::cout << "added light at index : " << lightIndex << std::endl;
    update();
}

void Node::showMaterialDialog() {
    if (materialList.size() != renderer->getMaterials().size()) {
        materialList.clear();
        selectedMaterials.clear();

        for (int i = 0; i < renderer->getMaterials().size(); i++) {
            const auto& material = renderer->getMaterials()[i];
            std::string textOption = material.name.empty() ? ("Material " + std::to_string(i)) : material.name;
            materialList.push_back(textOption);
        }

        for (int n = 0; n < renderer->assetManager.meshes[meshIndex].subMeshes.size(); n++) {
            selectedMaterials.push_back(materialIndices[n]);
        }
    }

    ImGui::Begin("Change Material");

    // Group submeshes by their original material ID from the OBJ file
    std::map<int, std::vector<uint32_t>> originalMatIdToSubmeshes;
    const auto& mesh = renderer->assetManager.meshes[meshIndex];

    for (int i = 0; i < mesh.subMeshes.size(); i++) {
        int originalMatId = (i < mesh.originalMaterialIds.size()) ? mesh.originalMaterialIds[i] : -1;
        originalMatIdToSubmeshes[originalMatId].push_back(i);
    }

    // Show one material slot per unique original material
    for (const auto& [originalMatId, submeshIndices] : originalMatIdToSubmeshes) {
        // Use the first submesh's current material as the representative
        uint32_t firstSubmeshIdx = submeshIndices[0];
        uint32_t currentMatIdx = materialIndices[firstSubmeshIdx];

        // Get the original material name from the OBJ file
        std::string originalMatName =
            (firstSubmeshIdx < mesh.originalMaterialNames.size()) ? mesh.originalMaterialNames[firstSubmeshIdx] : ("Material_" + std::to_string(originalMatId));

        std::string displayText = originalMatName;
        if (submeshIndices.size() > 1) {
            displayText += " (" + std::to_string(submeshIndices.size()) + " submeshes)";
        }

        ImGui::SetNextItemWidth(200);

        if (ImGui::BeginCombo(displayText.c_str(), materialList[currentMatIdx].c_str())) {
            for (int n = 0; n < materialList.size(); n++) {
                bool is_selected = (currentMatIdx == n);
                if (ImGui::Selectable(materialList[n].c_str(), is_selected)) {
                    // Check if selection changed
                    if (n != currentMatIdx) {
                        // Update all submeshes that share this original material ID
                        for (uint32_t subMeshIdx : submeshIndices) {
                            renderer->removeMeshFromShader(this, mesh.subMeshes[subMeshIdx], renderer->getMaterials()[materialIndices[subMeshIdx]].shaderSource,
                                                           renderer->getMaterials()[materialIndices[subMeshIdx]]);
                            materialIndices[subMeshIdx] = n;
                            renderer->addMeshToShader(this, mesh.subMeshes[subMeshIdx], renderer->getMaterials()[n].shaderSource, renderer->getMaterials()[n]);
                            selectedMaterials[subMeshIdx] = n;
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

void Node::showLightInfo() {

    if (lightIndex != MAX_LIGHTS && lightIndex < renderer->getLights().size()) {
        auto& light = renderer->getLight(lightIndex);
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

            ImGui::DragInt("show cascades", &light.showCascades,1,0,1);

            for(int i = 0; i< light.numCascades;i++){
                std::string cascadeLabel = "cascade ";
                cascadeLabel+= std::to_string(i);
                ImGui::DragFloat(cascadeLabel.c_str(), &light.cascades[i].splitDistance);
            }
            lightShadow = light.castsShadows;
            if (ImGui::Checkbox("Enable Shadows", &lightShadow)) {
                light.castsShadows = lightShadow ? 1 : 0;
                if (light.castsShadows == 1) {

                    for (int i = 0; i < light.numCascades; i++) {

                        vk::raii::Image shadowMapImage = nullptr;
                        vk::raii::DeviceMemory shadowMapMemory = nullptr;
                        // PCF uses R32F format to store raw depth values
                        resourceManager->createImage(light.shadowResolution, light.shadowResolution, 1, vk::SampleCountFlagBits::e1, vk::Format::eR32Sfloat,
                                                     vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
                                                     vk::MemoryPropertyFlagBits::eDeviceLocal, shadowMapImage, shadowMapMemory, 1);
                        vk::raii::ImageView shadowMapImageView = resourceManager->createImageView(shadowMapImage, vk::Format::eR32Sfloat, vk::ImageAspectFlagBits::eColor);

                        // Initial layout transition from undefined to shader read optimal
                        resourceManager->transitionImageLayout(nullptr, shadowMapImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal);

                        light.cascades[i].shadowMapIndex =
                            renderer->getDescriptorSet().allocateTexture(std::move(shadowMapImage), std::move(shadowMapMemory), std::move(shadowMapImageView),"internal/"+name+"/csm_"+std::to_string(i));
                    }
                } else {
                    for (int i = 0; i < light.numCascades; i++) {
                        renderer->getDescriptorSet().freeTexture(light.cascades[i].shadowMapIndex);
                    }
                }
            }
        }
        // Update light in all frame sections of the buffer
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            renderer->getDescriptorSet().updateFixedBufferWithOffset(renderer->getLightBufferIndex(), lightIndex, light, i);
        }
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
        if (meshIndex < renderer->assetManager.meshes.size()) {
            if (ImGui::Button(renderer->assetManager.meshes[meshIndex].sourceFile.c_str())) {
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
        browseButton("mesh",textBuffer,sizeof(textBuffer));

        if (ImGui::Button("Confirm")) {
            // remove old mesh from render queue for shader
            for (uint32_t index : materialIndices) {
                for (int i = 0; i < renderer->assetManager.meshes[meshIndex].subMeshes.size(); i++) {
                    renderer->removeMeshFromShader(this, renderer->assetManager.meshes[meshIndex].subMeshes[i], renderer->getMaterials()[index].shaderSource,
                                                   renderer->getMaterials()[index]);
                }
            }
            // load new mesh
            meshIndex = renderer->assetManager.loadMeshFromFile(std::string(textBuffer));
            materialList.clear();
            selectedMaterials.clear();
            materialIndices.clear();
            for (int i = 0; i < renderer->assetManager.meshes[meshIndex].subMeshes.size(); i++) {
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
    if (meshIndex < renderer->assetManager.meshes.size()) {
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

        light.cascades[i].lightSpaceMatrix = lightProj * lightView;
        light.cascades[i].texelSize = 1.0f / static_cast<float>(light.shadowResolution);
        light.cascades[i].worldTexelSize = maxExtent / static_cast<float>(light.shadowResolution);

        lastSplitDist = splitDist;
    }
}

std::vector<glm::vec4> getCameraFrustumCorners(Camera& camera) {
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

// Extracts the 6 frustum planes from a light space matrix
std::array<Plane, 6> extractFrustumPlanes(const glm::mat4& lightSpaceMatrix) {
    std::array<Plane, 6> planes;

    // Left plane
    planes[0].normal = glm::vec3(lightSpaceMatrix[0][3] + lightSpaceMatrix[0][0], lightSpaceMatrix[1][3] + lightSpaceMatrix[1][0], lightSpaceMatrix[2][3] + lightSpaceMatrix[2][0]);
    planes[0].distance = lightSpaceMatrix[3][3] + lightSpaceMatrix[3][0];

    // Right plane
    planes[1].normal = glm::vec3(lightSpaceMatrix[0][3] - lightSpaceMatrix[0][0], lightSpaceMatrix[1][3] - lightSpaceMatrix[1][0], lightSpaceMatrix[2][3] - lightSpaceMatrix[2][0]);
    planes[1].distance = lightSpaceMatrix[3][3] - lightSpaceMatrix[3][0];

    // Bottom plane
    planes[2].normal = glm::vec3(lightSpaceMatrix[0][3] + lightSpaceMatrix[0][1], lightSpaceMatrix[1][3] + lightSpaceMatrix[1][1], lightSpaceMatrix[2][3] + lightSpaceMatrix[2][1]);
    planes[2].distance = lightSpaceMatrix[3][3] + lightSpaceMatrix[3][1];

    // Top plane
    planes[3].normal = glm::vec3(lightSpaceMatrix[0][3] - lightSpaceMatrix[0][1], lightSpaceMatrix[1][3] - lightSpaceMatrix[1][1], lightSpaceMatrix[2][3] - lightSpaceMatrix[2][1]);
    planes[3].distance = lightSpaceMatrix[3][3] - lightSpaceMatrix[3][1];

    // Near plane (Vulkan/GLM_DEPTH_ZERO_TO_ONE: depth range [0,1], near at z_ndc=0, so just row2)
    planes[4].normal = glm::vec3(lightSpaceMatrix[0][2], lightSpaceMatrix[1][2], lightSpaceMatrix[2][2]);
    planes[4].distance = lightSpaceMatrix[3][2];

    // Far plane
    planes[5].normal = glm::vec3(lightSpaceMatrix[0][3] - lightSpaceMatrix[0][2], lightSpaceMatrix[1][3] - lightSpaceMatrix[1][2], lightSpaceMatrix[2][3] - lightSpaceMatrix[2][2]);
    planes[5].distance = lightSpaceMatrix[3][3] - lightSpaceMatrix[3][2];

    // Normalize all planes
    for (auto& plane : planes) {
        float length = glm::length(plane.normal);
        plane.normal /= length;
        plane.distance /= length;
    }

    return planes;
}

bool isAABBInFrustum(const glm::vec3& aabbMin, const glm::vec3& aabbMax, const std::array<Plane, 6>& planes, float epsilon) {
    // Test the AABB against each plane
    for (const auto& plane : planes) {
        glm::vec3 positiveVertex;
        positiveVertex.x = (plane.normal.x >= 0.0f) ? aabbMax.x : aabbMin.x;
        positiveVertex.y = (plane.normal.y >= 0.0f) ? aabbMax.y : aabbMin.y;
        positiveVertex.z = (plane.normal.z >= 0.0f) ? aabbMax.z : aabbMin.z;

        // Negative epsilon makes the frustum "bigger" (more conservative culling)
        if (glm::dot(plane.normal, positiveVertex) + plane.distance < -epsilon) {
            return false;
        }
    }

    return true;
}