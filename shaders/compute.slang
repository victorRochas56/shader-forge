// Option B: compute writes an offscreen storage image; a later graphics pass samples it to the swapchain.
// Images can't be reached via buffer device address — they go through a descriptor (here binding 4).

struct PushConstants {
    uint outputImageIndex; // index into storageImages[]
    uint width;
    uint height;
    float time;
}

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

// Compute-writable images. Must be in vk::ImageLayout::eGeneral while written.
[[vk::binding(4,0)]]
RWTexture2D<float4> storageImages[];

[NumThreads(8,8,1)]
[shader("compute")]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.width || tid.y >= pc.height) return;

    float2 uv = float2(tid.xy) / float2(pc.width, pc.height);
    // simple animated test pattern
    float3 col = float3(uv, 0.5 + 0.5 * sin(pc.time));
    storageImages[pc.outputImageIndex][int2(tid.xy)] = float4(col, 0.2);
}
