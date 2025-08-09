#pragma once
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1 // for raii
#define VULKAN_HPP_NO_CONSTRUCTORS 1         // for structs constructors
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#define GLFW_INCLUDE_VULKAN // REQUIRED only for GLFW CreateWindowSurface.
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#endif

// my modules
#include <bindless_resources.hpp>
#include <devices.hpp>
#include <load_resources.hpp>
#include <pipeline.hpp>
#include <structs.hpp>
#include <swapchain.hpp>
#include <utils.hpp>
///

std::vector<Line> lines;

void addLine(glm::vec3 startPoint, glm::vec3 endPoint, glm::vec4 color, BindlessResourceManager* resourceManager){

    std::vector<Point> lineData{ {.position = startPoint, .color = color}, {.position = endPoint, .color = color}};
    auto vertexInfo = resourceManager->allocateVertexBuffer(lineData.data(),lineData.size() * sizeof(Point),2,sizeof(Point));
    
    Line line{
        .allocIndex = vertexInfo.allocationIndex,
        .offset = static_cast<uint32_t>(vertexInfo.offset),
        .stride = sizeof(glm::vec3)
    };

    lines.push_back(line);
};