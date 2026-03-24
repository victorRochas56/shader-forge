# Plan: Enable Buffer Device Address (BDA) in the Renderer

## Context

The renderer uses bindless descriptor indexing to access storage buffers in shaders (bindings 3-8 in `shaders/modules/common.slang`). Buffer Device Address replaces these descriptor entries with raw 64-bit GPU pointers, passed via push constants. This simplifies the binding model — no more descriptor slots for buffers, no more `elementOffset` indirection. The extension is already listed in required extensions but never enabled.

Push constants will grow beyond 128 bytes (~144 for the lit pass), but **256 bytes is universally supported on desktop GPUs** (NVIDIA, AMD, Intel all report 256). We should add a startup assert on `maxPushConstantsSize >= 256`.

---

## Step 1: Enable BDA feature on the device
**File:** `modules/devices.hpp` ~line 135-150

- Add `bufferDeviceAddress = vk::True` to the existing `PhysicalDeviceVulkan12Features` chain.
- Add a startup check: `assert(deviceProperties.limits.maxPushConstantsSize >= 256)`.

---

## Step 2: Add BDA flags to buffer/memory creation
**Files:** `modules/descriptor_sets.hpp`, `modules/resources.hpp`

For **all** storage buffers (vertex, index, model matrices, lights, draw data, indirect):

- Add `vk::BufferUsageFlagBits::eShaderDeviceAddress` to buffer creation usage flags.
- Chain `vk::MemoryAllocateFlagsInfo{ .flags = vk::MemoryAllocateFlagBits::eDeviceAddress }` into `vk::MemoryAllocateInfo` via `.pNext`.

Affected functions:
- `createVariableBuffer()` (~line 519 in descriptor_sets.hpp)
- `createFixedBuffer<T>()` (~line 419 in descriptor_sets.hpp)
- `createIndirectDrawBuffer()` (~line 42 in resources.hpp)

---

## Step 3: Add address retrieval helper + store addresses
**File:** `modules/descriptor_sets.hpp` or `modules/utils.hpp`

```cpp
vk::DeviceAddress getBufferAddress(const vk::raii::Device& device, const vk::raii::Buffer& buffer) {
    return device.getBufferAddress(vk::BufferDeviceAddressInfo{.buffer = *buffer});
}
```

Store the `VkDeviceAddress` for each buffer in the existing buffer structs (e.g. `VariableBuffer`, `FixedBuffer`). Compute once at creation time.

---

## Step 4: Restructure push constants
**Files:** `modules/structs.hpp`, `shaders/modules/common.slang`

### LitPushConstants (main pass) — 128 → 144 bytes

**Before (128 bytes):**
```
samplerIndex, lightCount, shadowSamplerIndex, elementOffsetModel,
elementOffsetLight, elementOffsetLit, padding0, padding1,          // 32
cameraPos + pad, cameraForward + pad,                               // 32
viewProjection                                                      // 64
```

**After (144 bytes):**
```cpp
struct LitPushConstants {
    uint64_t vertexBufferAddress;      // 8  — replaces binding 3
    uint64_t modelMatricesAddress;     // 8  — replaces binding 4 (pre-offset to frame)
    uint64_t lightsAddress;            // 8  — replaces binding 5 (pre-offset to frame)
    uint64_t litDrawDataAddress;       // 8  — replaces binding 8 (pre-offset to frame)
    uint32_t samplerIndex;             // 4
    uint32_t lightCount;               // 4
    uint32_t shadowSamplerIndex;       // 4
    uint32_t padding;                  // 4
    glm::vec3 cameraPosition;          // 12
    uint32_t padding2;                 // 4
    glm::vec3 cameraForward;           // 12
    uint32_t padding3;                 // 4
    glm::mat4 viewProjection;          // 64
};                                     // Total: 144 bytes
```

Key change: `elementOffsetModel/Light/Lit` are gone. Instead, the CPU computes `baseAddress + frameOffset * sizeof(T)` and passes the pre-offset address. Shaders index directly from the pointer.

### ShadowPushConstants — 80 → 88 bytes

```cpp
struct ShadowPushConstants {
    uint64_t vertexBufferAddress;      // 8
    uint64_t modelMatricesAddress;     // 8  (pre-offset)
    uint64_t shadowDrawDataAddress;    // 8  (pre-offset)
    glm::mat4 lightSpaceMatrix;        // 64
};                                     // Total: 88 bytes
```

### Other push constant structs
`LinePushConstants`, `SkyBoxPushConstants`, etc. don't access storage buffers — no changes needed. Line rendering uses its own vertex buffer via binding 6/7 which can be migrated later.

---

## Step 5: Update shaders to use BDA pointers
**Files:** `shaders/modules/common.slang` and all `.slang` files

In Slang, BDA is accessed via pointer types. Replace descriptor array access:

```slang
// Before:
[[vk::binding(3,0)]] ByteAddressBuffer vertexBuffer[];
// ...
vertexBuffer[0].Load(byteOffset);

// After: use uint64_t address from push constants, cast to pointer
// Slang supports buffer pointers via extensions or raw pointer casts
```

For `loadVertex()`: pass the vertex buffer address and use it directly instead of `vertexBuffer[0].Load()`.

For `modelMatrices`, `lights`, `litDrawData`: replace `modelMatrices[0][offset + index]` with pointer-based access. The address is already pre-offset so it's just `ptr[index]`.

**Slang BDA approach:** Use `vk::RawBufferLoad<T>(address + byteOffset)` or define pointer structs. The exact syntax depends on Slang's BDA support — may need `[vk::ext_buffer_reference]` attribute.

---

## Step 6: Remove old descriptor bindings (cleanup)
**File:** `modules/descriptor_sets.hpp`

Once all storage buffers use BDA, remove bindings 3-8 from the descriptor set layout. The descriptor set shrinks to just:
- Binding 0: Texture2D array
- Binding 1: TextureCube array
- Binding 2: Sampler array

---

## Migration Order (incremental)

1. **Steps 1-3**: Enable BDA, add flags, get addresses. Zero behavior change — purely additive. Verify no validation errors.
2. **Step 4-5 for vertex buffer only**: Migrate binding 3 (`ByteAddressBuffer vertexBuffer[]`) to BDA. This is the simplest since it's already doing raw byte loads. Keep other bindings as-is.
3. **Step 4-5 for remaining buffers**: Migrate model matrices, lights, and draw data one at a time.
4. **Step 6**: Remove unused descriptor bindings.

## Verification

- After each phase: renderer produces identical visual output
- Run with `VK_LAYER_KHRONOS_validation` — it catches missing BDA flags, alignment issues, etc.
- Assert `maxPushConstantsSize >= 256` at device creation
- Check Slang compiler output (SPIR-V) includes `PhysicalStorageBuffer` capability
