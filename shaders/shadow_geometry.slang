struct ShadowPushConstants {
    uint vertexAllocationIndex;
    uint vertexOffset;
    uint vertexStride;
    uint modelMatrixIndex;
    float4x4 lightSpaceMatrix;
};

[[vk::push_constant]]
ConstantBuffer<ShadowPushConstants> pc;

[[vk::binding(3,0)]]
ByteAddressBuffer vertexBuffer[];

[[vk::binding(4, 0)]]
StructuredBuffer<float4x4> modelMatrices[];

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
VertexOutput vertMain(uint vertexID : SV_VertexID)
{
    Vertex vertex = loadVertex(vertexID, pc.vertexStride, pc.vertexOffset);
   
    float4x4 modelMatrix = modelMatrices[0][pc.modelMatrixIndex];
    float4 worldPos = mul(modelMatrix, float4(vertex.position, 1.0));
    
    VertexOutput output;
    output.position = mul(pc.lightSpaceMatrix, worldPos);  
    return output;
}

[shader("fragment")]
void fragMain()
{
 //alpha testing would go here (TODO?)
}