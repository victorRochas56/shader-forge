#include "scene_graph.hpp"

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
    bool continuous = false; // signal once?
    bool dead = false;       // no longer pointing to something valid
    size_t offset;
    void* param;
    void* prevValue;

    template <typename T>
    bool listen(SceneGraph& sceneGraph) {
        if(dead) return;
        if (!sceneGraph.isNodeValid(nodeIdx)) {
            dead = true;
            return false;
        }
        value = (char*)&(sceneGraph.getNode(nodeIdx)) + offset;
        switch (behaviour)
        {
        case ListenerBehaviour::CHANGED:
            if(*static_cast<T*>(prevValue) != *value) {
                *static_cast<T*>(prevValue) = *value;
                return true;
            }
            break;
        case ListenerBehaviour::EQUALS:
            if(*value == *static_cast<T*>(param))
                return true;
            break;
        case ListenerBehaviour::LESS:
            if(*value < *static_cast<T*>(param))
                return true;
            break;
        case ListenerBehaviour::MORE:
            if(*value > *static_cast<T*>(param))
                    return true;
            break;

        default:
            return false;
            break;
        }

        return false;
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
    bool active = false;     // even triggered effector?
    bool continuous = false; // trigger continuously while active?
    bool dead = false;       // no longer pointing to something valid
    size_t offset;
    void* param;

    template <typename T>
    void effect(SceneGraph& sceneGraph) {
        if(dead) return;
        if (!sceneGraph.isNodeValid(nodeIdx)) {
            dead = true;
            return false;
        }
        targetValue = (char*)&(sceneGraph.getNode(nodeIdx)) + offset;
        switch (behaviour)
        {
        case EffectorBehaviour::SET:
            *static_cast<T*>(targetValue) = *param;
            break;
        case EffectorBehaviour::ADD:
            *static_cast<T*>(targetValue) += *param;
            break;
        case EffectorBehaviour::SUB:
            *static_cast<T*>(targetValue) -= *param;
            break;
        case EffectorBehaviour::SUB:
            *static_cast<T*>(targetValue) *= *param;
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

struct Event {
    EventTriggerBehaviour behaviour;
};
// map / dictionary of param name -> offset size & type
class EventSystem {

  public:
    void addEvent(std::vector<uint32_t> listenerIdxs, std::vector<uint32_t> effectorIdxs) {}

    void addListener(uint32_t nodeIdx, std::string valueName, ListenerBehaviour behaviour, void* behaviourParam) {}

    void addEffector(uint32_t nodeIdx, std::string valueName, EffectorBehaviour behaviour, void* effectorParam) {}

    void pollListeners() {}

    void pollEvents() {
        // triggers effectors.
        // handles dead listeners & effectors
        // handle continuous or dead etc
    }

  private:
    std::vector<Listener> listeners;
    std::vector<Effector> effectors;
    std::vector<Event> events;
};