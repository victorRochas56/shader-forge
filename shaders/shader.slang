// Push constants
struct PushConstants
{
    uint vertexBufferIndex;
    uint vertexOffset;
    uint vertexStride;
    uint modelMatrixIndex;
    uint textureIndex;
    uint samplerIndex;
    uint padding[2];            // 8 bytes - align to 16 bytes
    float4x4 viewProjection;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]]
Texture2D bindlessTextures[];

[[vk::binding(1, 0)]]
SamplerState bindlessSamplers[];

[[vk::binding(2, 0)]]
ByteAddressBuffer bindlessVertexBuffers[];

[[vk::binding(3, 0)]]
StructuredBuffer<float4x4> bindlessModelMatrices[];

struct Vertex
{
    float3 position;
    float3 normal;
    float2 texCoord;
};

struct VertexOutput
{
    float4 position : SV_Position;
    float3 worldPos : WORLD_POS;
    float3 normal : NORMAL;
    float2 texCoord : TEXCOORD0;
};

Vertex loadVertex(uint bufferIndex, uint vertexIndex, uint stride, uint offset)
{
    Vertex vert;

    uint byteOffset = offset + vertexIndex * stride;
    vert.position = float3(asfloat(bindlessVertexBuffers[bufferIndex].Load(byteOffset)),
                        asfloat(bindlessVertexBuffers[bufferIndex].Load(byteOffset + 4)),
                        asfloat(bindlessVertexBuffers[bufferIndex].Load(byteOffset + 8)));

    vert.normal = float3(asfloat(bindlessVertexBuffers[bufferIndex].Load(byteOffset + 12)),
                        asfloat(bindlessVertexBuffers[bufferIndex].Load(byteOffset + 16)),
                        asfloat(bindlessVertexBuffers[bufferIndex].Load(byteOffset + 20)));

    vert.texCoord = float2(asfloat(bindlessVertexBuffers[bufferIndex].Load(byteOffset + 24)),
                        asfloat(bindlessVertexBuffers[bufferIndex].Load(byteOffset + 28)));
    
    return vert;
}

[shader("vertex")]
VertexOutput vertMain(uint vertexID : SV_VertexID)
{
    // load vertex data from bindless buffer
    Vertex vertex = loadVertex(
        pc.vertexBufferIndex,
        vertexID,
        pc.vertexStride,
        pc.vertexOffset
    );
   
    float4x4 modelMatrix = bindlessModelMatrices[0][pc.modelMatrixIndex];

    VertexOutput output;

    float4 worldPos = mul(modelMatrix, float4(vertex.position, 1.0));
    output.worldPos = worldPos.xyz;
    output.position = mul(pc.viewProjection,worldPos);
    output.normal = mul((float3x3)modelMatrix, vertex.normal);
    output.texCoord = vertex.texCoord;
   
    return output;
}

struct FragmentOutput
{
    float4 color : SV_Target0;
};

[shader("fragment")]
FragmentOutput fragMain(VertexOutput input)
{
    FragmentOutput output;
   
    float4 albedo = bindlessTextures[pc.textureIndex].Sample(bindlessSamplers[pc.samplerIndex], input.texCoord);

    float3 normal = normalize(input.normal);
    float3 lightDir = normalize(float3(1.0, 1.0, 1.0));
    float ndotl = max(dot(normal, lightDir), 0.0);
   
    output.color = albedo * ndotl ;
    return output;
}