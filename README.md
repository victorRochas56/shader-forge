# Vulkan Renderer - Shader Forge

A modern, feature-rich **real-time 3D graphics renderer** built with **Vulkan** and modern C++. This project serves as an educational platform for learning advanced graphics programming, the Vulkan API, and high-performance rendering techniques.

## Overview

This is a comprehensive renderer featuring a node-based scene hierarchy, physically-based rendering (PBR), cascaded shadow maps, and an interactive ImGui-based editor. Perfect for learning C++20, Vulkan architecture, and modern graphics techniques.

## Features

### Core Rendering

- **Physically-Based Rendering (PBR)** - Metallic/roughness workflow with proper material properties
- **Cascaded Shadow Maps (CSM)** - Multi-cascade directional light shadows with PCF filtering
- **Environment Mapping** - Cubemap-based lighting with blur support
- **Normal Mapping** - Surface detail via normal maps
- **MSAA Anti-Aliasing** - Configurable multi-sample anti-aliasing
- **Dynamic Rendering** - Modern VK_KHR_dynamic_rendering (no render passes)
- **Post-Processing** - Gaussian blur for effects and bloom

### Scene Management

- **Node-Based Hierarchy** - Parent-child scene graph with transformations
- **Component System** - Per-node components: Transform, Mesh, Material, Light
- **Frustum Culling** - CPU-side AABB culling for performance optimization
- **Ray Casting** - Interactive object selection
- **Scene Serialization** - Save and load scene state

### Lighting System

- **Multiple Light Types** - Point, directional, spot, and area lights
- **Shadow Casting** - Per-light shadow map generation
- **Cascaded Shadow Maps** - 4 cascades per directional light with configurable overlap
- **Light Debugging** - Visualize shadow maps by cascade
- **Soft Shadows** - PCF (Percentage Closer Filtering) for realistic shadow edges

### Material System

- **Bindless Textures** - Up to 2048 textures accessible without rebinding
- **Material Database** - Deduplication and efficient material management
- **Texture Masks** - Conditional texture binding via bitfield
- **PBR Properties** - Albedo, roughness, metallic, normal, and environment maps

### Interactive Editor

- **ImGui Integration** - Comprehensive property inspector and scene editor
- **Real-Time Property Editing** - Transform, mesh, material, and light properties
- **Node Tree Visualization** - Hierarchical scene structure
- **Gizmo System** - Colored axes for visual transform feedback
- **Frame Statistics** - Real-time performance monitoring
- **Context Menus** - Quick actions for scene objects

### Input & Controls

```
WASD             - Camera movement
Right Click      - Camera rotation
Left Click       - Object selection (ray casting)
V                - Toggle V-Sync
D                - Toggle depth visualization
Scroll           - Camera zoom (via zoom multiplier)
Delete           - Delete selected node
```

## Project Structure

```
.
├── main.cpp                    # Application entry point
├── CMakeLists.txt              # Build configuration
├── modules/
│   ├── app.hpp                 # Main application, UI loop, input
│   ├── renderer.hpp            # Core rendering engine
│   ├── scene_elements.hpp      # Node, Camera, Light structures
│   ├── structs.hpp             # Push constants, vertex layouts
│   ├── devices.hpp             # Vulkan device initialization
│   ├── descriptor_sets.hpp     # Texture/buffer descriptor management
│   ├── pipelines.hpp           # Graphics pipeline creation
│   ├── resources.hpp           # Model/texture loading
│   ├── gizmo.hpp               # Debug visualization
│   ├── input.hpp               # Input handling
│   ├── gui.hpp                 # ImGui integration
│   ├── scenes.hpp              # Scene save/load
│   ├── constants.hpp           # Configuration constants
│   ├── utils.hpp               # Utility functions
│   └── swapchain.hpp           # Swapchain management
├── shaders/
│   ├── lit.slang               # Main PBR lighting shader
│   ├── shadow_geometry.slang   # Shadow map generation
│   ├── skybox.slang            # Cubemap rendering
│   ├── blur.slang              # Post-process blur
│   ├── line.slang              # Debug line rendering
│   └── depth_view.slang        # Depth visualization
├── models/
│   └── sponza.obj              # Sample 3D model (Sponza architecture)
└── textures/
    ├── sky1/, sky2/            # Cubemap environment maps
    └── sponza/                 # Material textures
```

## System Requirements

- **Windows 10/11** with Vulkan-capable GPU
- **Vulkan 1.4** compatible graphics card (NVIDIA, AMD, Intel Arc)
- **CMake 3.20** or higher
- **C++20** compatible compiler (MSVC, GCC, Clang)

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

```bash
# Clone the repository
git clone <repo-url>
cd repo

# Create build directory
mkdir build
cd build

# Configure and build
cmake ..
cmake --build . --config Release
```

The build system will:
- Compile Slang shaders to SPIR-V
- Copy shaders, textures, and models to output directory
- Link all Vulkan dependencies

## Learning Outcomes

This project is designed to teach:

### C++ Concepts
- Modern C++20 features (concepts, ranges, spaceship operator)
- RAII for resource management
- Template metaprogramming for type-safe graphics code
- Performance optimization techniques

### Vulkan Graphics API
- Device selection and queue management
- Command buffer recording and submission
- Pipeline creation and state management
- Descriptor sets and bindless textures
- Synchronization (semaphores, fences)
- Dynamic rendering extensions
- Indirect drawing commands

### Graphics Programming
- Physically-Based Rendering (PBR) theory and implementation
- Shadow mapping techniques (cascaded shadow maps)
- Normal mapping and parallax mapping
- Cubemap environment mapping
- Post-processing effects
- Frustum culling for performance
- Indirect drawing for batch rendering

### Software Architecture
- Component-based entity systems
- Scene graph design
- Resource management and pooling
- Hot-reload infrastructure for shaders
- RAII patterns in graphics code

## Configuration

Key constants in [modules/constants.hpp](modules/constants.hpp):

```cpp
MAX_FRAMES_IN_FLIGHT = 2           // Frame buffering
MAX_MESHES = 2048                  // Maximum meshes in scene
MAX_LIGHTS = 2048                  // Maximum lights
MAX_NODES = 2048                   // Maximum scene nodes
MAX_TEXTURES = 2048                // Bindless texture limit
DEFAULT_SHADOW_RESOLUTION = 2048   // Shadow map resolution
```

## Scene File Format

Scenes are saved as text files with a simple format:

```
Node NodeName
  Transform x y z rx ry rz sx sy sz
  Mesh mesh_name
  Material material_name
  Light light_type intensity ...
  Children
    # Child nodes
```

## Performance Tips

- **Frustum Culling** - Reduces draw calls for hidden objects
- **Shadow Cascade Optimization** - Adjust cascade splits for your scene
- **Texture Compression** - Use KTX2 format for faster loading
- **Indirect Drawing** - Shadow pass uses indirect draw for efficiency
- **MSAA Tuning** - Balance quality vs. performance with sample count

## Future Improvements

- Material buffer system (to replace push constant limits)
- GPU-driven rendering pipeline
- Mesh shader support (VK_KHR_mesh_shader)
- Temporal anti-aliasing (TAA)
- Screen-space reflections (SSR)
- Compute shader support for post-processing
- Path tracing for offline rendering

## License

This project is provided as an educational resource. Modify and use freely for learning purposes.

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

## Contributing

This is a personal learning project, but improvements and bug fixes are welcome!

---

Built with passion for learning modern graphics programming. Happy rendering! 🎨
