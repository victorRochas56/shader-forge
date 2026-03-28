#pragma once

#include <unordered_map>
#include <chrono>

struct Trace {
    std::string name;
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

class Tracer {

public:
    static void startTrace(std::string name) {
        if(instance().traces.contains(name)){
            instance().traces[name].start = std::chrono::high_resolution_clock::now();
        }
        else {
            instance().traces[name] = {.name = name, .start = std::chrono::high_resolution_clock::now()};
        }
    }

    static void endTrace(std::string name) {
        if(!instance().traces.contains(name)) {
            std::printf("! trying to end non existent trace : ", name);
            return;
        }
        instance().traces[name].end = std::chrono::high_resolution_clock::now();
        instance().traces[name].updateDuration(instance().samples);
    }

    static auto& getTraces() { return instance().traces; }

    static Trace& getTrace(std::string name) { 
        if(instance().traces.contains(name)){
            return instance().traces[name]; 
        }
        throw std::runtime_error("trace does not exist!");
    }

    static void setSamples(int s) { instance().samples = s; }
    static int getSamples() { return instance().samples; }

private:
    Tracer() = default;

    int samples = 60;
    std::unordered_map<std::string,Trace> traces;

    static Tracer& instance() {
        static Tracer t;
        return t;
    }

};