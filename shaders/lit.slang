struct PushConstants {
    int vertexAllocationIndex; // Index into vertex allocations
    int vertexOffset;          // Byte offset in vertex buffer
    int vertexStride;          // Size of each vertex (e.g., sizeof(Vertex))
    int modelMatrixIndex;      // Index into model matrices
    int albedoTextureIndex;    // Index into textures
    int roughnessTextureIndex;
    int metallicTextureIndex;
    int normalTextureIndex;
    int samplerIndex; // Index into samplers
    int padding1;
    int padding2;
    int padding3;
    float4x4 viewProjection;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[[vk::binding(0,0)]]
Texture2D textures[];

[[vk::binding(1,0)]]
TextureCube cubemaps[];

[[vk::binding(2,0)]]
SamplerState samplers[];

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
    float3 worldPos : WORLD_POS;
    float3 normal : NORMAL;
    float2 texCoord : TEXCOORD0;
    float3 tangent : TANGENT; 
    float3 bitangent : BITANGENT; 
};

Vertex loadVertex( uint vertexIndex, uint stride, uint offset)
{
    Vertex vert;

    uint byteOffset = offset + vertexIndex * stride;
    vert.position = float3(asfloat(vertexBuffer[0].Load(byteOffset)),
                        asfloat(vertexBuffer[0].Load(byteOffset + 4)),
                        asfloat(vertexBuffer[0].Load(byteOffset + 8)));

    vert.normal = float3(asfloat(vertexBuffer[0].Load(byteOffset + 12)),
                        asfloat(vertexBuffer[0].Load(byteOffset + 16)),
                        asfloat(vertexBuffer[0].Load(byteOffset + 20)));

    vert.texCoord = float2(asfloat(vertexBuffer[0].Load(byteOffset + 24)),
                        asfloat(vertexBuffer[0].Load(byteOffset + 28)));
    
    vert.tangent = float3(asfloat(vertexBuffer[0].Load(byteOffset + 32)),
                        asfloat(vertexBuffer[0].Load(byteOffset + 36)),
                        asfloat(vertexBuffer[0].Load(byteOffset + 40)));
    
    return vert;
}

[shader("vertex")]
VertexOutput vertMain(uint vertexID : SV_VertexID)
{
    // load vertex data from bindless buffer
    Vertex vertex = loadVertex(
        vertexID,
        pc.vertexStride,
        pc.vertexOffset
    );
   
    float4x4 modelMatrix = modelMatrices[0][pc.modelMatrixIndex];

    VertexOutput output;

    float4 worldPos = mul(modelMatrix, float4(vertex.position, 1.0));
    output.worldPos = worldPos.xyz;
    output.position = mul(pc.viewProjection,worldPos);
    output.normal = mul((float3x3)modelMatrix, vertex.normal);
    output.texCoord = vertex.texCoord;
    output.tangent = normalize(mul((float3x3)modelMatrix, vertex.tangent));;
    output.bitangent = normalize(cross(output.normal, output.tangent));
    return output;
}

struct FragmentOutput
{
    float4 color : SV_Target0;
};

[shader("fragment")]
[earlydepthstencil]
FragmentOutput fragMain(VertexOutput input)
{
    FragmentOutput output;

    output.color = float4(1,1,1,1);
    return output;
}