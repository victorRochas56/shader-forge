# Vulkan Renderer - Shader Forge

A **real-time 3D graphics renderer** built with **Vulkan** and modern C++. This project was made as a way for me to learn advanced graphics programming, the Vulkan API, and high-performance rendering techniques, as well as learning to use modern c++20 features. 

## Features

### Core Rendering

- **Bindless Resources** - gpu driven rendering with bindless resources
- **Physically-Based Rendering (PBR)** - Metallic/roughness workflow with proper material properties
- **Cascaded Shadow Maps (CSM)** - Multi-cascade directional light shadows with PCF filtering
- **Environment Mapping** - Cubemap-based lighting with blur support
- **Dynamic Rendering** - Modern VK_KHR_dynamic_rendering (no render passes)
- **Post-Processing** - bloom, blur, ssao (WIP)

### Scene Management

- **Node-Based Hierarchy** - Parent-child scene graph with transformations
- **Component System** - Per-node components: Transform, Mesh, Material, Light
- **Frustum Culling** - CPU-side AABB culling for performance optimization
- **Scene Serialization** - Save and load scene state

### UI

- **IMGUI** - project uses imgui for all ui features 

### Editor

- **Real-Time Property Editing** - user can edit transform, mesh, material, and light properties
- **Node Tree Visualization** - Hierarchical scene structure
- **Gizmo System** - Colored axes for visual transform feedback
- **Frame Statistics** - Real-time performance monitoring

### Input & Controls

```
WASD             - Camera movement
Right Click      - Camera rotation
Left Click       - Object selection (ray casting)
V                - Toggle V-Sync
D                - Toggle depth visualization
C                - visualize CSM cascades
```

## Dependencies

- **Vulkan SDK** - Graphics API
- **GLFW3** - Window management and input
- **GLM** - Linear algebra/mathematics
- **ImGui** - UI framework
- **tinyobjloader** - OBJ model loading
- **stb_image** - Image file loading
- **Slang** - Shader compiler (SPIR-V generation)
- **KTX** - Compressed texture format support

## Building

### Prerequisites

1. Install [Vulkan SDK](https://vulkan.lunarg.com/sdk/home)
2. Install CMake 3.20+
3. Install VCPKG for dependency management (optional, but recommended)

### Build Steps

builds with CMAKE, the build system will:
- Compile Slang shaders to SPIR-V
- Copy shaders, textures, and models to output directory
- Link all Vulkan dependencies

## Resources

### Vulkan Learning
- [Vulkan Tutorial](https://vulkan-tutorial.com/)
- [Khronos Vulkan Documentation](https://registry.khronos.org/vulkan/)
- [AMD Vulkan Samples](https://gpuopen.com/learn/vulkan-samples/)

### Graphics Programming
- [LearnOpenGL - PBR](https://learnopengl.com/PBR/Theory)
- [Real-Time Rendering](https://www.realtimerendering.com/) (Book)
- [Physically Based Rendering](https://pbr-book.org/) (Free online book)

### C++20
- [cppreference.com](https://en.cppreference.com/)
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)

This is a personal learning project.

---

