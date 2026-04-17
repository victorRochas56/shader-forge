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
// Pointer fields map to 64-bit PhysicalStorageBuffer addresses (binary-compatible with uint64_t on CPU)
struct LitPushConstants {
    float*        vertexBuffer;
    float4x4*     modelMatrices;
    Light*        lights;
    LitDrawData*  litDrawData;
    LitPass*      litPassData;
};

struct LitPass {
    uint samplerIndex;
    uint lightCount;
    uint shadowSamplerIndex;
    uint shadowAtlasIndex;
    float3 cameraPos;
    uint padding1;
    float3 cameraForward;
    uint padding2;
    float4x4 viewProjection;
    float4x4 prevViewProjection;
};

struct Cascade {
    float4x4 lightSpaceMatrix;
    float4 shadowAtlasRange;
    float splitDistance;
    float texelSize;
    float worldTexelSize;
};

struct PointFace {
    float4x4 lightSpaceMatrix;
    float4 shadowAtlasRange;
};

struct Light {
    uint type;
    uint modelMatrixIndex;
    float range;
    float intensity;
    float4 color;
    int castsShadows;
    int showCascades;
    uint numCascades;
    uint shadowResolution;
    Cascade cascades[3];
    PointFace pointFaces[6];
};

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

Vertex loadVertex(float* buffer, uint vertexIndex, uint stride, uint offset)
{
    Vertex vert;

    // stride is in bytes; float* arithmetic is in units of 4 bytes
    uint floatOffset = (offset + vertexIndex * stride) / 4;
    float* v = buffer + floatOffset;

    vert.position = float3(v[0], v[1], v[2]);
    vert.normal   = float3(v[3], v[4], v[5]);
    vert.tangent  = float3(v[6], v[7], v[8]);
    // v[9] = materialIndex — skip
    vert.texCoord = float2(v[10], v[11]);

    return vert;
}

float3 getPosFromMM(int index){
    float4x4 mat = pc.modelMatrices[index];
    return float3(mat[0].w, mat[1].w, mat[2].w);
}

VertexOutput litVertMain(uint vertexID, uint drawID)
{
    LitDrawData draw = pc.litDrawData[drawID];

    Vertex vertex = loadVertex(pc.vertexBuffer, vertexID, draw.vertexStride, draw.vertexOffset);

    float4x4 modelMatrix = pc.modelMatrices[draw.modelMatrixIndex];
    VertexOutput output;
    float4 worldPos = mul(modelMatrix, float4(vertex.position, 1.0));
    output.worldPos = worldPos.xyz;
    output.position = mul(pc.litPassData.viewProjection, worldPos);
    output.normal = mul((float3x3)modelMatrix, vertex.normal);
    output.texCoord = vertex.texCoord;
    output.tangent = normalize(mul((float3x3)modelMatrix, vertex.tangent));
    output.bitangent = normalize(cross(output.normal, output.tangent));
    output.drawID = drawID;
    return output;
}
