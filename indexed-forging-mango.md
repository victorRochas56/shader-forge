# SSAO Implementation Plan

## Context

The renderer uses forward rendering with PBR lighting (lit.slang). AO is currently hardcoded to `1.0` in the lighting calculation. We want screen-space ambient occlusion computed from the depth buffer and applied to the scene.

**Approach:** Compute SSAO as a post-process after geometry, then apply it as a fullscreen multiply on the color buffer. This is an approximation (it darkens direct lighting too, not just ambient), but it's the standard approach for forward renderers and looks good in practice.

## Files to Create
- `shaders/ssao.slang` — SSAO computation shader
- `shaders/ssao_apply.slang` — Composite pass (multiply AO onto color)

## Files to Modify
- `modules/renderer.hpp` — Add SSAO pass to render loop, create resources
- `modules/structs.hpp` — Add `SSAOPushConstants` and `SSAOApplyPushConstants` structs
- `CMakeLists.txt` — New .slang files are auto-discovered via glob, no change needed

## Implementation Steps

### Step 1: Define push constant structs (`structs.hpp`)

```cpp
struct SSAOPushConstants {
    glm::mat4 projection;       // 64 bytes — for reconstructing view-space positions
    glm::mat4 invProjection;    // 64 bytes
    // 128 bytes total (max)
};

// Separate struct for the apply pass
struct SSAOApplyPushConstants {
    uint32_t ssaoTextureIndex;
    uint32_t samplerIndex;
    uint32_t padding[2];
};
```

Note: depth index & sampler index can be packed into the SSAO struct by reducing to a single matrix and reconstructing the other, or by using a UBO. But since each pipeline has its own push constant layout, we have the full 128 bytes per pipeline.

Revised SSAO push constants to fit everything:
```cpp
struct SSAOPushConstants {
    glm::mat4 invProjection;    // 64 bytes
    uint32_t depthIndex;        // 4
    uint32_t depthSamplerIndex; // 4
    uint32_t noiseIndex;        // 4
    uint32_t noiseSamplerIndex; // 4
    glm::uvec2 resolution;     // 8
    float radius;               // 4
    float bias;                 // 4
    float power;                // 4
    uint32_t kernelSize;        // 4
    // Total: 100 bytes — fits in 128
};
```

### Step 2: Create SSAO shader (`shaders/ssao.slang`)

Fullscreen pass that:
1. Samples depth buffer at current fragment
2. Reconstructs view-space position from depth + inverse projection
3. Reconstructs view-space normal from depth partial derivatives (cross product of ddx/ddy of position)
4. For N samples in a hemisphere oriented along the normal:
   - Offset sample position by random kernel vector (rotated by noise texture)
   - Project back to screen space
   - Sample depth at that screen position
   - Compare: if sampled depth is closer than sample, it's occluded
5. Output occlusion factor (0 = fully occluded, 1 = no occlusion)

Key details:
- Use 16-32 sample kernel (hemisphere, cosine-weighted)
- 4x4 noise texture (random rotation vectors) tiled across screen to vary kernel rotation
- Hardcode kernel samples as constants in the shader (avoids needing a UBO)
- Output to R8_UNORM single-channel texture

### Step 3: Create SSAO apply/composite shader (`shaders/ssao_apply.slang`)

Fullscreen pass that:
1. Samples the existing color from the swapchain (as a texture)
2. Samples the blurred SSAO texture
3. Outputs `color * ao`

Actually — simpler approach: render the SSAO result as a multiplicative blend over the color attachment. Use blend mode `srcColor * dstColor` (modulate blend). This avoids needing to read the color as a texture.

Pipeline blend state: `srcColorBlendFactor = eZero, dstColorBlendFactor = eSrcColor` → `finalColor = dstColor * srcColor`. The fragment shader outputs the AO value in all RGB channels.

This eliminates the need for `ssao_apply.slang` entirely — just use blend mode on the SSAO output pass.

**Revised: Only need `shaders/ssao.slang`, applied with multiplicative blending.**

### Step 4: Create resources in renderer (`renderer.hpp`)

During initialization (`initVulkan` or similar):

1. **SSAO render target** — R8_UNORM texture at screen resolution (or half-res for performance)
   - Create via `resourceManager->createImage(...)` with `eColorAttachment | eSampled` usage
   - Allocate in descriptor set for shader access

2. **SSAO blur target** — Same format, for blur pass (reuse existing blur infrastructure)

3. **4x4 noise texture** — R16G16_SFLOAT, 4x4 pixels of random unit vectors in tangent space
   - Generate CPU-side, upload once
   - Allocate in descriptor set

4. **Noise sampler** — Repeat address mode, nearest filter

5. **SSAO pipeline** — Post-process pipeline with `SSAOPushConstants`, no blending, outputs to SSAO texture

6. **SSAO apply pipeline** — Post-process pipeline with multiplicative blending, renders to swapchain color attachment. Fragment shader just outputs AO value; blend mode multiplies it onto existing color.

### Step 5: Integrate into render loop (`renderer.hpp::recordCommandBuffer`)

Insert after the main geometry `endRendering()` (line ~897) and before the depth view post-process:

```
[existing] endRendering() — geometry pass complete

// --- SSAO ---
if (enableSSAO) {
    // 1. Transition depth to shader-readable
    transitionImageLayout(depth, DEPTH_ATTACHMENT → SHADER_READ_ONLY)

    // 2. Transition SSAO target to color attachment
    transitionImageLayout(ssaoImage, SHADER_READ_ONLY → COLOR_ATTACHMENT)

    // 3. Render SSAO to ssaoImage
    beginRendering(ssaoRenderInfo → ssaoImage)
    bindPipeline(ssaoPipeline)
    pushConstants(SSAOPushConstants{...})
    draw(3, 1, 0, 0)  // fullscreen triangle
    endRendering()

    // 4. Transition SSAO to shader-readable for blur
    transitionImageLayout(ssaoImage, COLOR_ATTACHMENT → SHADER_READ_ONLY)

    // 5. Blur the SSAO texture (reuse existing blur pass)
    // Horizontal blur: ssaoImage → tempBlurImage
    // Vertical blur: tempBlurImage → ssaoImage

    // 6. Apply SSAO to color buffer via multiplicative blend
    transitionImageLayout(ssaoImage, ... → SHADER_READ_ONLY)
    beginRendering(swapchainRenderInfo with LOAD, multiplicative blend pipeline)
    bindPipeline(ssaoApplyPipeline)  // has multiplicative blend state
    pushConstants(SSAOApplyPushConstants{ssaoTextureIndex, samplerIndex})
    draw(3, 1, 0, 0)
    endRendering()

    // 7. Transition depth back
    transitionImageLayout(depth, SHADER_READ_ONLY → DEPTH_ATTACHMENT)
}

[existing] depth view post-process...
```

### Step 6: Add multiplicative blend pipeline variant

In `pipelines.hpp`, the POSTPROCESS category currently uses default blend (no blending or standard alpha). Add support for multiplicative blend:
- `srcColorBlendFactor = eDstColor`
- `dstColorBlendFactor = eZero`
- `colorBlendOp = eAdd`
- Result: `finalColor = srcColor * dstColor` (AO value multiplied onto existing color)

This may require a new pipeline creation parameter or a separate pipeline category.

### Step 7: Handle swapchain resize

When the swapchain is recreated (window resize), the SSAO and blur textures need to be recreated at the new resolution. Add cleanup/recreation logic alongside existing swapchain recreation code.

## Verification

1. Build shaders: rebuild to compile new .slang files
2. Run the app — SSAO should darken corners, creases, and contact areas
3. Toggle `enableSSAO` to compare with/without
4. Tune `radius` (0.3-1.0), `bias` (0.01-0.05), `power` (1.0-3.0), `kernelSize` (16-64)
5. Check validation layers for zero new warnings
