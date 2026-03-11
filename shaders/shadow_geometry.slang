// Per-draw data stored in a storage buffer
struct ShadowDrawData {
    uint vertexAllocationIndex;
    uint vertexOffset;
    uint vertexStride;
    uint modelMatrixIndex;
};

// Push constants now only contain light space matrix (same for all draws in a cascade)
struct ShadowPushConstants {
    float4x4 lightSpaceMatrix;
    uint elementOffsetModel;  // Element offset for this frame's model matrices
    uint elementOffsetShadow; // Element offset for this frame's shadow draw data
};

[[vk::push_constant]]
ConstantBuffer<ShadowPushConstants> pc;

[[vk::binding(3,0)]]
ByteAddressBuffer vertexBuffer[];

[[vk::binding(4, 0)]]
StructuredBuffer<float4x4> modelMatrices[];

[[vk::binding(6, 0)]]
StructuredBuffer<ShadowDrawData> shadowDrawData[];

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

Vertex loadVertex(uint vertexIndex, uint stride, uint offset)
{
    Vertex vert;
    uint byteOffset = offset + vertexIndex * stride;
   
    uint posX_uint = vertexBuffer[0].Load(byteOffset);
    uint posY_uint = vertexBuffer[0].Load(byteOffset + 4);
    uint posZ_uint = vertexBuffer[0].Load(byteOffset + 8);
   
    vert.position = float3(asfloat(posX_uint), asfloat(posY_uint), asfloat(posZ_uint));
   
   
    return vert;
}

[shader("vertex")]
VertexOutput vertMain(uint vertexID : SV_VertexID, uint drawID : SV_DrawIndex)
{
    // Fetch per-draw data using gl_DrawID (SV_DrawIndex in SLANG)
    ShadowDrawData drawData = shadowDrawData[0][pc.elementOffsetShadow + drawID];

    Vertex vertex = loadVertex(vertexID, drawData.vertexStride, drawData.vertexOffset);

    float4x4 modelMatrix = modelMatrices[0][pc.elementOffsetModel + drawData.modelMatrixIndex];
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