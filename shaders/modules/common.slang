// Per-instance data stored in a GPU buffer, indexed by SV_InstanceID + firstInstance
struct LitInstanceData {
    uint modelMatrixIndex;
    uint albedoTextureIndex;
    uint roughnessTextureIndex;
    uint metallicTextureIndex;
    uint normalTextureIndex;
    uint environmentMapIndex;
    float maxEnvMips;
    uint materialFlags;
    float metallic;
    float roughness;
    float alphaCutoff;
    uint padding;
};

struct LitMeshDrawData {
    uint vertexAllocationOffset;
    uint vertexOffset;
    uint vertexStride;
    uint firstInstance; // base offset into LitInstanceData[] for this draw's instances
};

// Frame-level push constants (same for all draws in a drawIndexedIndirect call)
// Pointer fields map to 64-bit PhysicalStorageBuffer addresses (binary-compatible with uint64_t on CPU)
struct LitPushConstants {
    float*              vertexBuffer;
    float4x4*           modelMatrices;
    Light*              lights;
    LitInstanceData*    litInstanceData;
    LitMeshDrawData*    litMeshDrawData;
    LitPass*            litPassData;
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
    float _padding; // pads stride to 96 to match C++ sizeof(GPUCascade) under scalarBlockLayout
};

struct PointFace {
    float4x4 lightSpaceMatrix;
    float4 shadowAtlasRange;
};

enum LightType {
    Point,
    Directional,
    Spot,
    Area,
};

struct Light {
    LightType type;
    float3 position;
    float3 direction;
    float range;
    float intensity;
    uint padding[3];
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
    nointerpolation uint instanceID : INSTANCE_ID;
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
SamplerComparisonState shadowCmpSampler;



float calculateDirectionalShadow(float3 fragWorldPos, float3 cameraPos, float3 cameraForward, float3 geometricNormal, float3 L, Light* light, uint shadowAtlasIdx)
{
    float viewDepth = dot(fragWorldPos - cameraPos, cameraForward);

    int cascadeIndex = int(light.numCascades) - 1;
    for (int i = 0; i < int(light.numCascades); i++) {
        if (viewDepth < light.cascades[i].splitDistance) {
            cascadeIndex = i;
            break;
        }
    }
    if( light.showCascades>0){
        return cascadeIndex;
    }

    // Walk outward from the split-selected cascade; if the sample lands outside
    // a tile's bounds, try the next-larger cascade rather than returning lit.
    // This prevents block-shaped unshadowed regions on jittered volume samples
    // near cascade XY edges.
    for (int c = cascadeIndex; c < int(light.numCascades); c++) {
        Cascade cascade = light.cascades[c];

        float NdotL = abs(dot(geometricNormal, normalize(L + float3(1e-8))));
        float normalOffset = cascade.worldTexelSize * (1.0 - NdotL) * 2.0;
        float3 biasedPos = fragWorldPos + geometricNormal * normalOffset;

        float4 fragPosLightSpace = mul(cascade.lightSpaceMatrix, float4(biasedPos, 1.0));
        float3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
        projCoords.x = projCoords.x * 0.5 + 0.5;
        projCoords.y = projCoords.y * 0.5 + 0.5;

        float texelSize = cascade.texelSize;
        if (projCoords.x < texelSize || projCoords.x > 1.0 - texelSize ||
            projCoords.y < texelSize || projCoords.y > 1.0 - texelSize ||
            projCoords.z < 0.0 || projCoords.z > 1.0) {
            continue;
        }

        float currentDepth = projCoords.z - 0.0001f;

        float2 rangeMin  = cascade.shadowAtlasRange.xy;
        float2 rangeSpan = cascade.shadowAtlasRange.zw - cascade.shadowAtlasRange.xy;

        float2 atlasUV = rangeMin + projCoords.xy * rangeSpan;
        float lit = textures[shadowAtlasIdx].SampleCmpLevelZero(
            shadowCmpSampler,
            atlasUV,
            currentDepth
        );
        return 1.0 - lit;
    }
    return 0.0;
}

float calculatePointShadow(float3 fragWorldPos, float3 geometricNormal, float3 L, Light* light, uint shadowAtlasIdx)
{
    float3 fragToLight = fragWorldPos - light.position;
    float3 absDir = abs(fragToLight);

    int faceIndex;
    if (absDir.x >= absDir.y && absDir.x >= absDir.z) {
        faceIndex = fragToLight.x > 0.0 ? 0 : 1;
    } else if (absDir.y >= absDir.x && absDir.y >= absDir.z) {
        faceIndex = fragToLight.y > 0.0 ? 2 : 3;
    } else {
        faceIndex = fragToLight.z > 0.0 ? 4 : 5;
    }

    PointFace face = light.pointFaces[faceIndex];

    float NdotL = saturate(abs(dot(geometricNormal, L)));
    float normalOffset = (1.0 - NdotL) * light.range / float(light.shadowResolution) * 3.0;
    float3 biasedPos = fragWorldPos + geometricNormal * normalOffset;

    float4 fragPosLightSpace = mul(face.lightSpaceMatrix, float4(biasedPos, 1.0));
    float3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords.x = projCoords.x * 0.5 + 0.5;
    projCoords.y = projCoords.y * 0.5 + 0.5;

    float texelSize = 1.0 / float(light.shadowResolution);
    if (projCoords.x < texelSize || projCoords.x > 1.0 - texelSize ||
        projCoords.y < texelSize || projCoords.y > 1.0 - texelSize ||
        projCoords.z < 0.0 || projCoords.z > 1.0) {
        return 0.0;
    }

    float currentDepth = projCoords.z - 0.0001f;

    float2 rangeMin  = face.shadowAtlasRange.xy;
    float2 rangeSpan = face.shadowAtlasRange.zw - face.shadowAtlasRange.xy;

    float2 atlasUV = rangeMin + projCoords.xy * rangeSpan;
    float lit = textures[shadowAtlasIdx].SampleCmpLevelZero(
        shadowCmpSampler,
        atlasUV,
        currentDepth
    );
    return 1.0 - lit;
}

float calculateShadow(float3 fragWorldPos, float3 cameraPos, float3 cameraForward, float3 geometricNormal, float3 L, Light* light, uint shadowAtlasIdx)
{
    if (light.type == LightType::Directional) {
        return calculateDirectionalShadow(fragWorldPos, cameraPos, cameraForward, geometricNormal, L, light, shadowAtlasIdx);
    } else if (light.type == LightType::Point) {
        return calculatePointShadow(fragWorldPos, geometricNormal, L, light, shadowAtlasIdx);
    }
    return 0.0;
}

float3 reconstructPixelRayDir(float2 uv, float4x4 view, float proj00, float proj11) {
    float2 ndc = uv * 2.0 - 1.0;
    float3 viewRay = float3(ndc.x / proj00, ndc.y / proj11, -1.0);
    return normalize(mul(viewRay, (float3x3)view));
}

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

VertexOutput litVertMain(uint vertexID, uint instanceID, uint drawID)
{
    LitMeshDrawData mesh = pc.litMeshDrawData[drawID];
    uint absInstance = mesh.firstInstance + instanceID;
    LitInstanceData draw = pc.litInstanceData[absInstance];

    Vertex vertex = loadVertex(pc.vertexBuffer, vertexID, mesh.vertexStride, mesh.vertexOffset);

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
    output.instanceID = absInstance;
    return output;
}

uint pcgHash(uint x) {
    uint state = x * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float rand(inout uint seed) {
    seed = pcgHash(seed);
    return float(seed) * (1.0 / 4294967296.0);
}