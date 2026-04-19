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
