// Per-draw data stored in a storage buffer
struct ShadowDrawData {
    uint vertexAllocationIndex;
    uint vertexOffset;
    uint vertexStride;
    uint modelMatrixIndex;
};

// Push constants with BDA pointers (pre-offset to current frame)
struct ShadowPushConstants {
    float*          vertexBuffer;
    float4x4*       modelMatrices;
    ShadowDrawData* shadowDrawData;
    uint64_t        _pad;              // align lightSpaceMatrix to 16
    float4x4 lightSpaceMatrix;
};

[[vk::push_constant]]
ConstantBuffer<ShadowPushConstants> pc;

struct Vertex
{
    float3 position;
    float3 normal;
    float2 texCoord;
    float3 tangent;
};

struct VertexOutput
{
    float4 position : SV_Position;
};

/*
TODO : position buffer
The biggest issue here is almost certainly cache efficiency from your interleaved vertex layout, not the load itself.
Your Vertex struct is position(12) + normal(12) + texCoord(8) + tangent(12) = 44 bytes stride. So when the shadow shader loads just position, each 128-byte cache line carries only ~3 useful positions and ~92 bytes of normals/texcoords/tangents that get fetched and thrown away. That's a ~10× hit on effective bandwidth for the depth pass.
Use a separate position-only vertex stream for shadow rendering. This is the standard approach in modern engines (Doom, UE5, etc. all do this). Maintain two buffers per mesh: a tightly-packed float3 positions[] for depth/shadow passes, and the full interleaved buffer for the main pass. The shadow VS now loads from a 12-byte stride, cache lines are saturated with useful data, and the long scoreboard largely goes away.
hlslstruct ShadowDrawData {
    uint vertexAllocationIndex;
    uint positionBufferOffset;   // offset into position-only stream
    uint modelMatrixIndex;
    uint _pad;
};

// In VS:
float3* positions = (float3*)(pc.positionBuffer + drawData.positionBufferOffset);
float3 pos = positions[vertexID];
The stride field also goes away because position-only is always 12 bytes.
Second win: precompute the combined matrix CPU-side. Right now you do:
load modelMatrix  →  mul(model, pos)  →  mul(lightSpace, worldPos)
Store lightSpaceMatrix * modelMatrix per draw in a buffer indexed by drawID. Then the VS becomes a single load and a single matrix multiply:
hlslfloat4x4 mvp = pc.shadowMVPMatrices[drawID];
float3 pos = positions[vertexID];
output.position = mul(mvp, float4(pos, 1.0));
You also drop lightSpaceMatrix from push constants entirely. One fewer matrix load, one fewer mul, and the matrix fetch can issue in parallel with the position fetch — they're independent now.
Third thing worth checking: scalarization of the drawData load. drawID is wave-uniform, so pc.shadowDrawData[drawID] should compile to a scalar load (SMEM on AMD, uniform path on NVIDIA) — one fetch per wave instead of 32–64 redundant ones. Check the disassembly (RGA on AMD, or Nsight's SASS view on NVIDIA). If you see vector loads for those fields, the compiler isn't recognizing the uniformity and you may need to wrap with WaveReadLaneFirst() or similar to force it. Usually it works automatically with SV_DrawIndex since it's specified as wave-uniform, but it's worth verifying since you're already in the profiler.
A few smaller notes:
The _pad in push constants is fine but you can drop it once you remove lightSpaceMatrix (push it into the per-draw matrix instead).
Once you have a position-only stream, consider whether you actually need a per-vertex stride at all in shadowDrawData — if every shadow caster uses the same packed float3 format, that field becomes implicit and drawData shrinks to 8–12 bytes, fitting more entries per cache line.
If you're rendering a lot of small shadow casters, the next bottleneck after this will likely be VS launch overhead and triangle setup rather than memory. At that point, mesh shaders or meshlet-based culling start paying off, but that's a much bigger restructure.


*/

Vertex loadVertex(float* buffer, uint vertexIndex, uint stride, uint offset)
{
    Vertex vert;
    uint floatOffset = (offset + vertexIndex * stride) / 4;
    float3* v = (float3*)(buffer + floatOffset);

    vert.position = v[0];

    return vert;
}

[shader("vertex")]
VertexOutput vertMain(uint vertexID : SV_VertexID, uint drawID : SV_DrawIndex)
{
    ShadowDrawData drawData = pc.shadowDrawData[drawID];

    // Issue matrix load first so its latency overlaps the vertex fetch.
    float4x4 modelMatrix = pc.modelMatrices[drawData.modelMatrixIndex];
    Vertex vertex = loadVertex(pc.vertexBuffer, vertexID, drawData.vertexStride, drawData.vertexOffset);

    float4 worldPos = mul(modelMatrix, float4(vertex.position, 1.0));

    VertexOutput output;
    output.position = mul(pc.lightSpaceMatrix, worldPos);
    return output;
}

// Depth-only pass — hardware writes depth automatically via SV_Position.z
[shader("fragment")]
void fragMain(VertexOutput input)
{
}
