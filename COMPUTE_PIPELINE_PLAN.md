# Compute Pipeline Scaffolding — Implementation Plan

Goal: add a reusable compute pipeline abstraction + dispatch + barrier to the
renderer, with a trivial SSBO-writing example shader to build on.

## Key architectural fact

`PipelineManager` is deeply graphics-specific: `createPipelineInternal` always
builds vertex+fragment stages (`pipelines.hpp:191-193`), a rasterizer,
color-blend state, depth-stencil, and a category switch over render targets. A
compute pipeline has none of that — it is just `{shader module + pipeline
layout}` plus a `vkCmdDispatch`. **Do not shoehorn it into the category switch.**
Add a small parallel abstraction instead.

Three hard pieces already work:

| Need              | Status                                                                            |
| ----------------- | --------------------------------------------------------------------------------- |
| Compute queue     | Graphics queue supports compute (spec-guaranteed). Use it — no new queue needed.  |
| Slang→SPIR-V      | `compileSlangToSpv` (`utils.hpp:236`) is stage-agnostic; `-fvk-use-entrypoint-name` handles any entry point. |
| Descriptor binding| Bindless set has storage buffers (bindings 4+). Compute can read/write SSBOs now. |

## Dispatch on the graphics queue (no separate compute queue)

The device exposes one unified graphics+present family (`devices.hpp:108-119`),
and graphics queues are required by spec to support `VK_QUEUE_COMPUTE_BIT`.
Recording `vkCmdDispatch` into the existing per-frame command buffer means zero
cross-queue semaphores — a pipeline barrier orders compute against later draws.
Only pursue an async-compute queue later if a measured overlap win justifies the
synchronization cost.

## File-by-file changes

### 1. `modules/pipelines.hpp` — add a compute pipeline type

Add a `ComputePipeline<T>` struct alongside `Pipeline<T>` (`pipelines.hpp:58`).
Do **not** inherit `PipelineBase` — that base is full of graphics-only fields
(topology, cullMode, colorAttachmentFormat). Keep it standalone:

```cpp
template <typename T> struct ComputePipeline {
    vk::raii::Pipeline pipeline = nullptr;
    vk::raii::PipelineLayout layout = nullptr;
    vk::raii::DescriptorSetLayout* descriptorSetLayout = nullptr;
    vk::raii::DescriptorSet* descriptorSet = nullptr;
    std::string shaderPath;
    T pushConstantData;

    void pushConstants(vk::raii::CommandBuffer& cmd) {
        cmd.pushConstants<T>(layout, vk::ShaderStageFlagBits::eCompute, 0, pushConstantData);
    }
};
```

For hot-reload, the type-erased `std::unique_ptr<PipelineBase>` vectors won't fit
a non-`PipelineBase` type. Two clean choices:
- **Simplest:** a separate `std::vector<std::unique_ptr<ComputePipelineBase>>`
  with its own tiny abstract base (`virtual recreate()`, `virtual ~`), mirroring
  the existing `PipelineBase`/`Pipeline<T>` split.
- Skip hot-reload for v1 and add it once dispatch works.

### 2. `modules/pipelines.hpp` — add `createComputePipeline<T>()`

A standalone method (not part of the graphics `createPipelineInternal` switch):

```cpp
ensureSpvUpToDate(shaderPath); // build .spv from .slang like the graphics path (pipelines.hpp:189)
auto module = createShaderModule(readFile(shaderPath));
vk::PipelineShaderStageCreateInfo stage{
    .stage = vk::ShaderStageFlagBits::eCompute, .module = module, .pName = "computeMain"};
vk::PushConstantRange pc{vk::ShaderStageFlagBits::eCompute, 0, sizeof(T)};
vk::PipelineLayoutCreateInfo layoutInfo{.setLayoutCount = 1, .pSetLayouts = &*setLayout,
    .pushConstantRangeCount = 1, .pPushConstantRanges = &pc};
layout = vk::raii::PipelineLayout(device.getDevice(), layoutInfo);
vk::ComputePipelineCreateInfo info{.stage = stage, .layout = *layout};
pipeline = vk::raii::Pipeline(device.getDevice(), nullptr, info);
```

Return a `uint32_t` index into the compute vector, matching the convention of
`createPipeline` (`pipelines.hpp:77`).

### 3. Hot-reload wiring (optional, if doing it now)

In `checkForShaderUpdates` (`pipelines.hpp:87-109`), the `shaderPathToIndices`
map drives recompilation. Register compute shaders there too with a category tag
(add a `COMPUTE` enum value or a parallel map) so `recreatePipelineAtIndex`
routes to the compute vector. Keep the `waitIdle()` before recreation
(`pipelines.hpp:104`) — same device-lost hazard applies.

### 4. `shaders/compute_example.slang` — example shader

Add a `[shader("compute")]` entry named `computeMain` with `[numthreads(64,1,1)]`
(or 8x8 for image work). Have it write one of the existing storage buffers so
output is verifiable without a storage-image binding. Compiles through the same
`compileSlangToSpv` path → `compute_example.spv`.

### 5. `modules/renderer.cpp` — dispatch in the render loop

**Creation** — alongside the other `createPipeline` calls (near
`renderer.cpp:184`), call `createComputePipeline<YourPushConstants>(...)` passing
`bindless.descriptorSet->getDescriptorSetLayout()` and `getDescriptorSet()`.

**Dispatch** — inside `recordCommandBuffer` (`renderer.cpp:656-711`), early
(before the geometry pass that consumes the result):

```cpp
cmd.bindPipeline(vk::PipelineBindPoint::eCompute, computePipeline.pipeline);
cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, computePipeline.layout, 0,
                       *computePipeline.descriptorSet, nullptr);
computePipeline.pushConstants(cmd);
cmd.dispatch(groupsX, groupsY, 1);  // groupsX = ceil(workItems / threadsPerGroup)
// barrier: eComputeShader/eShaderWrite -> eVertexShader|eFragmentShader/eShaderRead
cmd.pipelineBarrier2(...);
```

No queue-submission changes — it rides the existing graphics submit
(`renderer.cpp:367-376`).

## Gotchas specific to this codebase

1. **Push-constant stage flags.** The graphics path hardcodes
   `eVertex | eFragment` (`pipelines.hpp:64`, `:255`). Compute layouts and
   `pushConstants` calls must use `eCompute`, or expect validation errors / silent
   garbage.
2. **The barrier is the whole correctness story.** Same queue means no
   semaphores, but without the `eComputeShader → eVertex/eFragmentShader` memory
   barrier the draw may read stale data. Easy to forget, hard to debug.
3. **Storage images need a new binding.** The bindless set is `eSampledImage`
   only (`descriptor_sets.hpp:208-253`). This scaffolding targets SSBOs and
   sidesteps that. A post-process effect that writes an image will need an
   `eStorageImage` binding — a follow-up, not part of v1.

## Suggested order

Start with steps 1, 2, 4, 5 (skip hot-reload) and an SSBO-writing example — the
smallest loop that proves the path end to end. Add hot-reload (step 3) and
storage images afterward.
