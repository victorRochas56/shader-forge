#pragma once

// Vulkan defines (must come before vulkan includes)
#ifndef VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#endif
#ifndef VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_NO_CONSTRUCTORS 1
#endif

// Vulkan C++ headers (largest compile-time cost)
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

// GLM (used almost everywhere)
#define GLM_FORCE_RADIANS
#define GLM_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>
#include <glm/gtc/quaternion.hpp>

// STL headers used broadly
#include <algorithm>
#include <array>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
