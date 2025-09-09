#include "scene_elements.hpp"
#include "renderer.hpp"

Node::Node(Renderer* pRenderer, uint32_t arrayIndex, Node* parent, glm::vec3 position, glm::quat rotation, glm::vec3 scale, bool keepWorldTransform) : renderer(pRenderer) {

    nodeIndex = arrayIndex;

    relativePosition = position;
    relativeRotation = rotation;
    relativeScale = scale;
    relativeRotationEuler = glm::eulerAngles(rotation);

    if (parent == nullptr) {
        parent = renderer->getRootNode();
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
    }

    parent->addChild(this); // Don't use keepRelativeTransform since we handled it above
    modelMatrixIndex = renderer->getDescriptorSet().allocateFixedBuffer(renderer->getModelMatrixBufferIndex(), glm::mat4(1.0f));
    update();
}

void Node::update() {
    localTransform = makeTransform(relativePosition, relativeRotation, relativeScale);
    if (parent != nullptr) {
        worldTransform = parent->worldTransform * localTransform;
    } else {
        worldTransform = localTransform;
    }

    renderer->getDescriptorSet().updateFixedBuffer(renderer->getModelMatrixBufferIndex(), modelMatrixIndex, worldTransform);

    for (Node* child : children) {
        child->update();
    }
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
