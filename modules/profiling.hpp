#pragma once

#include <unordered_map>
#include <vector>
#include <chrono>

struct Trace {
    std::string name;
    uint32_t scope = 0;
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;
    std::chrono::duration<double> duration;
    std::chrono::duration<double> averageDuration{0};
    int sampleIndex = 0;
    double runningTotal = 0.0;

    void updateDuration(int samples){
        duration = end - start;
        runningTotal += duration.count();
        sampleIndex++;
        if(sampleIndex >= samples){
            averageDuration = std::chrono::duration<double>(runningTotal / samples);
            runningTotal = 0.0;
            sampleIndex = 0;
        }
    }
};

namespace tracing {

inline int samples = 60;
inline std::unordered_map<std::string,Trace> traces;
inline std::vector<std::string> order;
inline uint32_t scope = 0;
inline uint32_t maxScope = 0;

inline void startTrace(std::string name) {
    if(traces.contains(name)){
        traces[name].start = std::chrono::high_resolution_clock::now();
    }
    else {
        traces[name] = {.name = name, .scope = scope, .start = std::chrono::high_resolution_clock::now()};
        order.push_back(name);
    }
    scope++;
    if(scope > maxScope) maxScope = scope;
}

inline void endTrace(std::string name) {
    if(!traces.contains(name)) {
        std::printf("! trying to end non existent trace : %s\n ", name.c_str());
        return;
    }
    traces[name].end = std::chrono::high_resolution_clock::now();
    traces[name].updateDuration(samples);
    // Never wrap: an unbalanced end (e.g. after an early-return skipped its start) would push scope
    // to ~4 billion, and every zone registered after that bakes the garbage in as its indent depth.
    if (scope > 0) scope--;
}

inline auto& getTraces() { return traces; }

inline const std::vector<std::string>& getOrder() { return order; }

inline Trace& getTrace(std::string name) { 
    if(traces.contains(name)){
        return traces[name]; 
    }
    throw std::runtime_error("trace does not exist!");
}

inline void setSamples(int s) { samples = s; }
inline int getSamples() { return samples; }

}