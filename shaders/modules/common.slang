// Per-draw data stored in a GPU buffer, indexed by SV_DrawIndex
struct LitDrawData {
    uint vertexAllocationIndex;
    uint vertexOffset;
    uint vertexStride;
    uint modelMatrixIndex;
    uint albedoTextureIndex;
    uint roughnessTextureIndex;
    uint metallicTextureIndex;
    uint normalTextureIndex;
    uint environmentMapIndex;
    uint materialFlags;
    float metallic;
    float roughness;
    float alphaCutoff;
    uint padding;
};

// Frame-level push constants (same for all draws in a drawIndexedIndirect call)
struct LitPushConstants {
    uint samplerIndex;
    uint lightCount;
    uint shadowSamplerIndex;
    uint elementOffsetModel;
    uint elementOffsetLight;
    uint elementOffsetLit;
    uint padding0;
    uint padding1;
    float3 cameraPos;
    uint padding2;
    float3 cameraForward;
    uint padding3;
    float4x4 viewProjection;
};

struct Cascade {
    float4x4 lightSpaceMatrix;
    uint shadowMapIndex;
    float splitDistance;
    float texelSize;
    float worldTexelSize;
};

struct Light {
    uint type;
    uint modelMatrixIndex;
    uint padding;
    uint padding1;
    float range;
    float intensity;
    uint shadowMapIndex;
    uint shadowResolution;
    float4 color;
    float4x4 lightSpaceMatrix;
    float3 direction;
    int castsShadows;
    uint showCascades;
    uint padding3;
    uint padding4;
    uint numCascades;
    Cascade cascades[4];
}

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
    nointerpolation uint drawID : DRAW_ID;
};

// Shared descriptor bindings
[[vk::push_constant]]
ConstantBuffer<LitPushConstants> pc;

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

[[vk::binding(5,0)]]
StructuredBuffer<Light> lights[];

[[vk::binding(8,0)]]
StructuredBuffer<LitDrawData> litDrawData[];


float3 reconstructViewPos(float2 uv, float depth, float4x4 invProjection) {
    float4 clip = float4(uv * 2.0 - 1.0, depth, 1.0);
    float4 view = mul(invProjection, clip);
    return view.xyz / view.w;
}

float3 calculateViewSpaceNormal(uint depthIndex,uint depthSamplerIndex, float4x4 invProjection, float2 texelSize, float2 uv, float3 fragPos) {
    float3 posRight = reconstructViewPos(uv + float2(texelSize.x, 0.0),
        textures[depthIndex].Sample(samplers[depthSamplerIndex], uv + float2(texelSize.x, 0.0)).r, invProjection);
    float3 posLeft = reconstructViewPos(uv - float2(texelSize.x, 0.0),
        textures[depthIndex].Sample(samplers[depthSamplerIndex], uv - float2(texelSize.x, 0.0)).r, invProjection);
    float3 posUp = reconstructViewPos(uv + float2(0.0, texelSize.y),
        textures[depthIndex].Sample(samplers[depthSamplerIndex], uv + float2(0.0, texelSize.y)).r, invProjection);
    float3 posDown = reconstructViewPos(uv - float2(0.0, texelSize.y),
        textures[depthIndex].Sample(samplers[depthSamplerIndex], uv - float2(0.0, texelSize.y)).r, invProjection);

    // Use the smaller delta to avoid artifacts at edges
    float3 dx = (abs(posRight.z - fragPos.z) < abs(fragPos.z - posLeft.z)) ? (posRight - fragPos) : (fragPos - posLeft);
    float3 dy = (abs(posUp.z - fragPos.z) < abs(fragPos.z - posDown.z)) ? (posUp - fragPos) : (fragPos - posDown);
    return normalize(cross(dx, dy));
}

Vertex loadVertex(uint vertexIndex, uint stride, uint offset)
{
    Vertex vert;

    uint byteOffset = offset + vertexIndex * stride;

    uint posX_uint = vertexBuffer[0].Load(byteOffset);
    uint posY_uint = vertexBuffer[0].Load(byteOffset + 4);
    uint posZ_uint = vertexBuffer[0].Load(byteOffset + 8);

    vert.position = float3(asfloat(posX_uint), asfloat(posY_uint), asfloat(posZ_uint));

    uint normX_uint = vertexBuffer[0].Load(byteOffset + 12);
    uint normY_uint = vertexBuffer[0].Load(byteOffset + 16);
    uint normZ_uint = vertexBuffer[0].Load(byteOffset + 20);

    vert.normal = float3(asfloat(normX_uint), asfloat(normY_uint), asfloat(normZ_uint));

    uint tangX_uint = vertexBuffer[0].Load(byteOffset + 24);
    uint tangY_uint = vertexBuffer[0].Load(byteOffset + 28);
    uint tangZ_uint = vertexBuffer[0].Load(byteOffset + 32);

    vert.tangent = float3(asfloat(tangX_uint), asfloat(tangY_uint), asfloat(tangZ_uint));

    // Skip materialIndex at offset 36-39

    uint texU_uint = vertexBuffer[0].Load(byteOffset + 40);
    uint texV_uint = vertexBuffer[0].Load(byteOffset + 44);

    vert.texCoord = float2(asfloat(texU_uint), asfloat(texV_uint));

    return vert;
}

float3 getPosFromMM(int index){
    float3 ret;
    ret.x = modelMatrices[0][pc.elementOffsetModel + index][0].w;
    ret.y = modelMatrices[0][pc.elementOffsetModel + index][1].w;
    ret.z = modelMatrices[0][pc.elementOffsetModel + index][2].w;
    return ret;
}

VertexOutput litVertMain(uint vertexID, uint drawID)
{
    LitDrawData draw = litDrawData[0][pc.elementOffsetLit + drawID];

    Vertex vertex = loadVertex(vertexID, draw.vertexStride, draw.vertexOffset);

    float4x4 modelMatrix = modelMatrices[0][pc.elementOffsetModel + draw.modelMatrixIndex];
    VertexOutput output;
    float4 worldPos = mul(modelMatrix, float4(vertex.position, 1.0));
    output.worldPos = worldPos.xyz;
    output.position = mul(pc.viewProjection, worldPos);
    output.normal = mul((float3x3)modelMatrix, vertex.normal);
    output.texCoord = vertex.texCoord;
    output.tangent = normalize(mul((float3x3)modelMatrix, vertex.tangent));
    output.bitangent = normalize(cross(output.normal, output.tangent));
    output.drawID = drawID;
    return output;
}
