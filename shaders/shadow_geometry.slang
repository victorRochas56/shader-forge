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
    float depth : DEPTH;
};

Vertex loadVertex(float* buffer, uint vertexIndex, uint stride, uint offset)
{
    Vertex vert;
    uint floatOffset = (offset + vertexIndex * stride) / 4;
    float* v = buffer + floatOffset;

    vert.position = float3(v[0], v[1], v[2]);

    return vert;
}

[shader("vertex")]
VertexOutput vertMain(uint vertexID : SV_VertexID, uint drawID : SV_DrawIndex)
{
    // Fetch per-draw data using gl_DrawID (SV_DrawIndex in SLANG)
    ShadowDrawData drawData = pc.shadowDrawData[drawID];

    Vertex vertex = loadVertex(pc.vertexBuffer, vertexID, drawData.vertexStride, drawData.vertexOffset);

    float4x4 modelMatrix = pc.modelMatrices[drawData.modelMatrixIndex];
    float4 worldPos = mul(modelMatrix, float4(vertex.position, 1.0));

    VertexOutput output;
    output.position = mul(pc.lightSpaceMatrix, worldPos);
    // Store NDC depth (after perspective divide) to match sampling in lit.slang
    output.depth = output.position.z / output.position.w;
    return output;
}

struct FragmentOutput
{
    float depth : SV_Target0;
};

[shader("fragment")]
FragmentOutput fragMain(VertexOutput input)
{
    FragmentOutput output;
    output.depth = input.depth;

    return output;
}
