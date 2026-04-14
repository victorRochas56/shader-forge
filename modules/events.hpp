#pragma once
#include <unordered_map>
#include "scene_graph.hpp"

struct NodeProperty {
    size_t offset;
    ValueType type;
};

inline const std::unordered_map<std::string, NodeProperty>& getNodePropertyMap() {
    static const std::unordered_map<std::string, NodeProperty> map = {
        {"relativePosition",      {offsetof(Node, relativePosition),      ValueType::VEC3}},
        {"relativeScale",         {offsetof(Node, relativeScale),         ValueType::VEC3}},
        {"relativeRotation",      {offsetof(Node, relativeRotation),      ValueType::QUAT}},
        {"relativeRotationEuler", {offsetof(Node, relativeRotationEuler), ValueType::VEC3}},
        {"worldRotationEuler",    {offsetof(Node, worldRotationEuler),    ValueType::VEC3}},
        {"boundingBoxMin",        {offsetof(Node, boundingBoxMin),        ValueType::VEC3}},
        {"boundingBoxMax",        {offsetof(Node, boundingBoxMax),        ValueType::VEC3}},
        {"nodeIndex",             {offsetof(Node, nodeIndex),             ValueType::UINT32}},
        {"parentIndex",           {offsetof(Node, parentIndex),           ValueType::UINT32}},
        {"firstChild",            {offsetof(Node, firstChild),            ValueType::UINT32}},
        {"nextSibling",           {offsetof(Node, nextSibling),           ValueType::UINT32}},
        {"meshIndex",             {offsetof(Node, meshIndex),             ValueType::UINT32}},
        {"materialIndex",         {offsetof(Node, materialIndex),         ValueType::UINT32}},
        {"lightIndex",            {offsetof(Node, lightIndex),            ValueType::UINT32}},
    };
    return map;
}

enum class ListenerBehaviour {
    CHANGED,
    EQUALS,
    LESS,
    MORE,
};

enum class ValueType {
    FLOAT,
    VEC3,
    QUAT,
    UINT32,
    BOOL,
    INT,
};

struct Listener {
    ListenerBehaviour behaviour;
    ValueType type;
    uint32_t nodeIdx = 0;
    bool triggered = false;
    bool latch = false;       // once triggered, stay triggered permanently?
    bool dead = false;        // no longer pointing to something valid
    size_t offset;
    void* param;
    void* prevValue;

    void markDead() {
        if(dead) return;
        dead = true;
        if(prevValue) { free(prevValue); prevValue = nullptr; }
    }

    template <typename T>
    void listen(SceneGraph& sceneGraph) {
        if(dead) return;
        if (!sceneGraph.isNodeValid(nodeIdx)) {
            markDead();
            return;
        }
        T* value = reinterpret_cast<T*>(reinterpret_cast<char*>(&sceneGraph.getNode(nodeIdx)) + offset);
        if(latch && triggered)
            return;
        else{ triggered = false; }
        switch (behaviour)
        {
        case ListenerBehaviour::CHANGED:
            if(*static_cast<T*>(prevValue) != *value) {
                *static_cast<T*>(prevValue) = *value;
                triggered = true;
            }
            break;
        case ListenerBehaviour::EQUALS:
            if(*value == *static_cast<T*>(param))
                triggered = true;
            break;
        case ListenerBehaviour::LESS:
            if constexpr (std::is_arithmetic_v<T>) {
                if(*value < *static_cast<T*>(param))
                    triggered = true;
            }
            break;
        case ListenerBehaviour::MORE:
            if constexpr (std::is_arithmetic_v<T>) {
                if(*value > *static_cast<T*>(param))
                    triggered = true;
            }
            break;

        default:
            break;
        }
    };
};

enum class EffectorBehaviour {
    SET,
    ADD,
    SUB,
    MULT,
};

struct Effector {
    EffectorBehaviour behaviour;
    ValueType type;
    uint32_t nodeIdx;
    bool active = false;     // has this effector been triggered?
    bool singleUse = true;   // fire once then deactivate, or re-arm for next trigger?
    bool dead = false;        // no longer pointing to something valid
    size_t offset;
    void* param;

    template <typename T>
    void effect(SceneGraph& sceneGraph) {
        if(dead) return;
        if (!sceneGraph.isNodeValid(nodeIdx)) {
            dead = true;
            return;
        }
        T* targetValue = (char*)&(sceneGraph.getNode(nodeIdx)) + offset;
        switch (behaviour)
        {
        case EffectorBehaviour::SET:
            *static_cast<T*>(targetValue) = *static_cast<T*>(param);
            break;
        case EffectorBehaviour::ADD:
            *static_cast<T*>(targetValue) += *static_cast<T*>(param);
            break;
        case EffectorBehaviour::SUB:
            *static_cast<T*>(targetValue) -= *static_cast<T*>(param);
            break;
        case EffectorBehaviour::MULT:
            *static_cast<T*>(targetValue) *= *static_cast<T*>(param);
            break;
        
        default:
            break;
        }
    }
};

enum class EventTriggerBehaviour {
    FIRST,
    ALL,
    LESS,
    MORE,
};

constexpr uint32_t MAX_LISTENERS = 16;
constexpr uint32_t MAX_EFFECTORS = 16;

struct Event {
    EventTriggerBehaviour behaviour;
    uint32_t numListeners = 0;
    uint32_t numEffectors = 0;
    std::array<uint32_t,MAX_LISTENERS> listenerIdxs;
    std::array<uint32_t,MAX_EFFECTORS> effectorIdxs;
    uint32_t threshold = 0;
};

class EventSystem {

  public:
    EventSystem(const EventSystem&) = delete;
    EventSystem& operator=(const EventSystem&) = delete;

    static void init(SceneGraph& sceneGraph) {
        auto& e = instance();
        e.sceneGraph = &sceneGraph;
        // index 0 is reserved as sentinel for dead/invalid
        e.listeners.emplace_back();
        e.effectors.emplace_back();
        e.events.emplace_back();
    }

    static uint32_t addListener(uint32_t nodeIdx, const std::string& valueName, ListenerBehaviour behaviour, void* behaviourParam = nullptr) {
        auto& e = instance();
        if(nodeIdx == 0) return 0;
        auto& propMap = getNodePropertyMap();
        auto it = propMap.find(valueName);
        if(it == propMap.end()) return 0;

        const auto& prop = it->second;
        uint32_t idx = static_cast<uint32_t>(e.listeners.size());
        Listener listener{};
        listener.behaviour = behaviour;
        listener.type = prop.type;
        listener.nodeIdx = nodeIdx;
        listener.offset = prop.offset;
        listener.param = behaviourParam;
        listener.prevValue = malloc(sizeOfValueType(prop.type));
        // initialize prevValue from current node state
        const char* src = reinterpret_cast<const char*>(&e.sceneGraph->getNode(nodeIdx)) + prop.offset;
        memcpy(listener.prevValue, src, sizeOfValueType(prop.type));
        e.listeners.push_back(listener);
        return idx;
    }

    static uint32_t addEffector(uint32_t nodeIdx, const std::string& valueName, EffectorBehaviour behaviour, void* effectorParam) {
        auto& e = instance();
        if(nodeIdx == 0) return 0;
        auto& propMap = getNodePropertyMap();
        auto it = propMap.find(valueName);
        if(it == propMap.end()) return 0;

        const auto& prop = it->second;
        uint32_t idx = static_cast<uint32_t>(e.effectors.size());
        Effector effector{};
        effector.behaviour = behaviour;
        effector.type = prop.type;
        effector.nodeIdx = nodeIdx;
        effector.offset = prop.offset;
        effector.param = effectorParam;
        e.effectors.push_back(effector);
        return idx;
    }

    static uint32_t addEvent(EventTriggerBehaviour behaviour, const std::vector<uint32_t>& listenerIdxs, const std::vector<uint32_t>& effectorIdxs, uint32_t threshold = 0) {
        auto& e = instance();
        uint32_t idx = static_cast<uint32_t>(e.events.size());
        Event event{};
        event.behaviour = behaviour;
        event.threshold = threshold;
        event.numListeners = std::min(static_cast<uint32_t>(listenerIdxs.size()), MAX_LISTENERS);
        event.numEffectors = std::min(static_cast<uint32_t>(effectorIdxs.size()), MAX_EFFECTORS);
        for(uint32_t i = 0; i < event.numListeners; i++)
            event.listenerIdxs[i] = listenerIdxs[i];
        for(uint32_t i = 0; i < event.numEffectors; i++)
            event.effectorIdxs[i] = effectorIdxs[i];
        e.events.push_back(event);
        return idx;
    }

    static size_t sizeOfValueType(ValueType type) {
        switch(type) {
            case ValueType::FLOAT:  return sizeof(float);
            case ValueType::VEC3:   return sizeof(glm::vec3);
            case ValueType::QUAT:   return sizeof(glm::quat);
            case ValueType::UINT32: return sizeof(uint32_t);
            case ValueType::BOOL:   return sizeof(bool);
            case ValueType::INT:    return sizeof(int);
            default: return 0;
        }
    }

    static void pollListeners() {
        auto& e = instance();
        for(Listener& listener : e.listeners){
            switch (listener.type)
            {
            case ValueType::FLOAT:
                listener.listen<float>(*e.sceneGraph);
                break;
            case ValueType::VEC3:
                listener.listen<glm::vec3>(*e.sceneGraph);
                break;
            case ValueType::QUAT:
                listener.listen<glm::quat>(*e.sceneGraph);
                break;
            case ValueType::UINT32:
                listener.listen<uint32_t>(*e.sceneGraph);
                break;
            case ValueType::BOOL:
                listener.listen<bool>(*e.sceneGraph);
                break;
            case ValueType::INT:
                listener.listen<int>(*e.sceneGraph);
                break;
            default:
                break;
            }
        }
    }

    static void pollEvents() {
        auto& e = instance();
        for(Event& event : e.events) {

            bool triggered = false;

            // check listeners — compact out dead/invalid (index 0) entries
            uint32_t triggeredListeners = 0;
            uint32_t lWriteIdx = 0;
            for(uint32_t i = 0; i < event.numListeners; i++) {
                uint32_t listenerIdx = event.listenerIdxs[i];
                if(listenerIdx == 0) continue;
                if(e.listeners[listenerIdx].dead) continue;
                if(e.listeners[listenerIdx].triggered)
                    triggeredListeners++;
                event.listenerIdxs[lWriteIdx++] = listenerIdx;
            }
            event.numListeners = lWriteIdx;
            switch (event.behaviour)
            {
            case EventTriggerBehaviour::FIRST:
                if(triggeredListeners > 0) 
                    triggered = true;
                break;
            case EventTriggerBehaviour::ALL:
                if(triggeredListeners == event.numListeners)
                    triggered = true;
                break;
            case EventTriggerBehaviour::LESS:
                if(triggeredListeners < event.threshold)
                    triggered = true;
                break;
            case EventTriggerBehaviour::MORE:
                if(triggeredListeners > event.threshold)
                    triggered = true;
                break;
            default:
                break;
            }
            // compact out dead/invalid (index 0) effectors
            uint32_t eWriteIdx = 0;
            for(uint32_t i = 0; i < event.numEffectors; i++) {
                uint32_t effectorIdx = event.effectorIdxs[i];
                if(effectorIdx == 0) continue;
                if(e.effectors[effectorIdx].dead) continue;
                event.effectorIdxs[eWriteIdx++] = effectorIdx;
            }
            event.numEffectors = eWriteIdx;

            if(triggered){
                for(uint32_t i = 0; i < event.numEffectors; i++){
                    auto& effector = e.effectors[event.effectorIdxs[i]];
                    if(effector.active) continue; // already fired, waiting to re-arm
                    effector.active = true;
                    switch (effector.type)
                    {
                    case ValueType::FLOAT:
                        effector.effect<float>(*e.sceneGraph);
                        break;
                    case ValueType::VEC3:
                        effector.effect<glm::vec3>(*e.sceneGraph);
                        break;
                    case ValueType::QUAT:
                        effector.effect<glm::quat>(*e.sceneGraph);
                        break;
                    case ValueType::UINT32:
                        effector.effect<uint32_t>(*e.sceneGraph);
                        break;
                    case ValueType::BOOL:
                        effector.effect<bool>(*e.sceneGraph);
                        break;
                    case ValueType::INT:
                        effector.effect<int>(*e.sceneGraph);
                        break;
                    default:
                        break;
                    }
                    if(effector.singleUse)
                        effector.dead = true;
                }
            } else {
                // event not triggered — re-arm non-single-use effectors
                for(uint32_t i = 0; i < event.numEffectors; i++){
                    e.effectors[event.effectorIdxs[i]].active = false;
                }
            }
        }
    }

  private:
    EventSystem() = default;
    
    static EventSystem& instance() {
        static EventSystem e;
        return e;
    }

    SceneGraph* sceneGraph = nullptr;

    std::vector<Listener> listeners;
    std::vector<Effector> effectors;
    std::vector<Event> events;
};