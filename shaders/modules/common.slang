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
    uint packedColor; // material base colour, RGBA8 linear (was padding)
};

// Unpacks LitInstanceData.packedColor. Mirrors packColorRGBA8() in structs.hpp; linear, not sRGB,
// so it multiplies straight onto a decoded albedo sample.
float4 unpackColorRGBA8(uint c) {
    return float4((c >> 24) & 0xFFu, (c >> 16) & 0xFFu, (c >> 8) & 0xFFu, c & 0xFFu) * (1.0 / 255.0);
}

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
    // Added to SV_DrawIndex for per-draw lookups. SV_DrawIndex restarts at 0 for each indirect call,
    // so the lit pass sets this to the pipeline range's first command index (0 for the prepass).
    uint                drawIDOffset;
    float               time;
    // Slot into litFrame[] (binding 5) holding this pass's LitFrameUniforms.
    uint                frameUniformsIndex;
    uint                _pad0;
};

struct LitPass {
    uint samplerIndex;
    uint lightCount;
    uint shadowSamplerIndex;
    uint shadowAtlasIndex;
    float3 cameraPos;
    uint voxelTextureIndex;
    float3 cameraForward;
    uint voxelSamplerIndex;
    float4x4 viewProjection;
    float4x4 prevViewProjection;
    float4x4 voxelViewProjection; // world -> voxel grid clip
    uint voxelResolution;
    float voxelWorldExtent;
    // VXGI cone-trace tuning (GUI sliders)
    uint giHemisphereRays; // side cones at 60° (0..5); the normal cone always runs
    uint giMaxSteps;       // taps per cone
    uint giFetchBatch;     // taps issued per batch
    uint giMode;           // 0 = per-pixel cone trace, 1 = ambient-cube lookup
    uint giIrradianceIndices[6]; // per-face irradiance volumes, VOXEL_FACE_DIRS order
    float giStrength;
    float giSkyIntensity;  // cone-miss sky radiance scale (skyboxIntensity * vxgi.skyStrength)
};

#define MAX_UBO_LIGHTS 256

// Hot per-light data mirrored into the lit frame UBO (GPULightHot in structs.hpp).
// vec4-only members so std140 layout == the C++ struct byte-for-byte.
struct LightHot {
    float4 positionRange;   // xyz world position, w range
    float4 direction;       // xyz normalized direction, w unused
    float4 colorIntensity;  // rgb color, w intensity (<= 0 -> disabled sentinel slot)
    uint4  typeFlags;       // x LightType, y castsShadows, z numCascades, w unused
    float4 cascadeSplits;   // CSM split distances (xyz)
};

// Per-frame uniform block for the lit pass (GPULitFrameUniforms in structs.hpp). Hot pass/light
// data reads go through the constant cache here instead of per-fragment BDA loads; the fat
// Light/LitPass BDA buffers stay authoritative for every other pass.
struct LitFrameUniforms {
    float4x4 voxelViewProjection; // world -> voxel grid clip
    float4 cameraPosExtent;       // xyz cameraPos, w voxelWorldExtent
    float4 cameraForwardGI;       // xyz cameraForward, w giStrength
    uint4  indicesA;              // x samplerIndex, y lightCount (<= MAX_UBO_LIGHTS), z shadowSamplerIndex, w shadowAtlasIndex
    uint4  indicesB;              // x voxelTextureIndex, y voxelSamplerIndex, z voxelResolution, w giMode
    uint4  giParams;              // x giHemisphereRays, y giMaxSteps, z giFetchBatch, w unused
    uint4  giIrradianceA;         // irradiance volume indices 0-3 (VOXEL_FACE_DIRS order)
    uint4  giIrradianceB;         // xy irradiance indices 4-5, zw unused
    float4 giFloats;              // x giSkyIntensity, yzw unused
    LightHot lights[MAX_UBO_LIGHTS];
};

struct Cascade {
    float4x4 lightSpaceMatrix;
    float4 shadowAtlasRange;
    float splitDistance;
    float texelSize;
    float worldTexelSize;
    float _padding; // pads stride to 96 to match C++ sizeof(GPUCascade) under scalarBlockLayout
};

struct ShadowMap {
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
    ShadowMap shadowMaps[6];
};

struct Vertex
{
    float3 position;
    float3 normal;
    float2 texCoord;
    float3 tangent;
};

struct Particle {
    float3 position;
    float age;
    float3 velocity;
    float lifespan;
    float rotation;
    float angularVelocity;
    float size;
    uint seed; // also drives animation index
    uint emitterIdx;
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
    // Per-instance material data forwarded from the VS so the FS never loads LitInstanceData —
    // that per-fragment instance -> texture-index -> descriptor -> sample chain was the lit
    // pass's dominant stall. Attribute reads don't hit the memory scoreboard.
    nointerpolation uint4  texIndices : TEX_INDICES; // albedo, roughness, metallic, normal
    nointerpolation uint3  matInfo    : MAT_INFO;    // x environmentMapIndex, y materialFlags, z packed base colour
    nointerpolation float3 matParams  : MAT_PARAMS;  // x metallic, y roughness, z alphaCutoff
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

[[vk::binding(5,0)]]
ConstantBuffer<LitFrameUniforms> litFrame[];



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
        // Member reads through the pointer, not a 96-byte struct copy.
        float NdotL = abs(dot(geometricNormal, normalize(L + float3(1e-8))));
        float normalOffset = light.cascades[c].worldTexelSize * (1.0 - NdotL) * 2.0;
        float3 biasedPos = fragWorldPos + geometricNormal * normalOffset;

        float4 fragPosLightSpace = mul(light.cascades[c].lightSpaceMatrix, float4(biasedPos, 1.0));
        float3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
        projCoords.x = projCoords.x * 0.5 + 0.5;
        projCoords.y = projCoords.y * 0.5 + 0.5;

        float texelSize = light.cascades[c].texelSize;
        if (projCoords.x < texelSize || projCoords.x > 1.0 - texelSize ||
            projCoords.y < texelSize || projCoords.y > 1.0 - texelSize ||
            projCoords.z < 0.0 || projCoords.z > 1.0) {
            continue;
        }

        float currentDepth = projCoords.z - 0.0001f;

        float4 atlasRange = light.cascades[c].shadowAtlasRange;
        float2 rangeMin  = atlasRange.xy;
        float2 rangeSpan = atlasRange.zw - atlasRange.xy;

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

    float NdotL = saturate(abs(dot(geometricNormal, L)));
    float normalOffset = (1.0 - NdotL) * light.range / float(light.shadowResolution);
    float3 biasedPos = fragWorldPos + geometricNormal * normalOffset;

    float4 fragPosLightSpace = mul(light.shadowMaps[faceIndex].lightSpaceMatrix, float4(biasedPos, 1.0));
    float3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords.x = projCoords.x * 0.5 + 0.5;
    projCoords.y = projCoords.y * 0.5 + 0.5;

    float texelSize = 1.0 / float(light.shadowResolution);
    if (projCoords.z < 0.0 || projCoords.z > 1.0) {
        return 0.0;
    }
    projCoords.xy = clamp(projCoords.xy, 0.5 * texelSize, 1.0 - 0.5 * texelSize);

    float currentDepth = projCoords.z - 0.0001f;

    float4 atlasRange = light.shadowMaps[faceIndex].shadowAtlasRange;
    float2 rangeMin  = atlasRange.xy;
    float2 rangeSpan = atlasRange.zw - atlasRange.xy;

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

// Lit-pass shadow path: cascade selection comes from UBO-provided splits instead of the
// stride-96 walk over light.cascades[i].splitDistance, and there is no showCascades branch,
// so the compiler can sink the compare-fetch wait past independent BRDF math at the call site.
float calculateDirectionalShadowHot(float3 fragWorldPos, float3 cameraPos, float3 cameraForward,
                                    float3 geometricNormal, float3 L, Light* light, uint shadowAtlasIdx,
                                    float3 cascadeSplits, uint numCascades)
{
    float viewDepth = dot(fragWorldPos - cameraPos, cameraForward);

    int cascadeIndex = int(numCascades) - 1;
    for (int i = 0; i < int(numCascades); i++) {
        if (viewDepth < cascadeSplits[i]) {
            cascadeIndex = i;
            break;
        }
    }

    // Same outward walk as calculateDirectionalShadow.
    for (int c = cascadeIndex; c < int(numCascades); c++) {
        float NdotL = abs(dot(geometricNormal, normalize(L + float3(1e-8))));
        float normalOffset = light.cascades[c].worldTexelSize * (1.0 - NdotL) * 2.0;
        float3 biasedPos = fragWorldPos + geometricNormal * normalOffset;

        float4 fragPosLightSpace = mul(light.cascades[c].lightSpaceMatrix, float4(biasedPos, 1.0));
        float3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
        projCoords.x = projCoords.x * 0.5 + 0.5;
        projCoords.y = projCoords.y * 0.5 + 0.5;

        float texelSize = light.cascades[c].texelSize;
        if (projCoords.x < texelSize || projCoords.x > 1.0 - texelSize ||
            projCoords.y < texelSize || projCoords.y > 1.0 - texelSize ||
            projCoords.z < 0.0 || projCoords.z > 1.0) {
            continue;
        }

        float currentDepth = projCoords.z - 0.0001f;

        float4 atlasRange = light.cascades[c].shadowAtlasRange;
        float2 rangeMin  = atlasRange.xy;
        float2 rangeSpan = atlasRange.zw - atlasRange.xy;

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

float calculateShadowHot(float3 fragWorldPos, float3 cameraPos, float3 cameraForward, float3 geometricNormal,
                         float3 L, uint lightType, Light* light, uint shadowAtlasIdx,
                         float3 cascadeSplits, uint numCascades)
{
    if (lightType == 1) { // Directional
        return calculateDirectionalShadowHot(fragWorldPos, cameraPos, cameraForward, geometricNormal, L, light, shadowAtlasIdx, cascadeSplits, numCascades);
    } else if (lightType == 0) { // Point
        return calculatePointShadow(fragWorldPos, geometricNormal, L, light, shadowAtlasIdx);
    }
    return 0.0;
}

// Shadow lookup for voxel radiance injection. Directional lights sample the dedicated VXGI
// tile (shadowMaps[0], fit around the voxel volume) instead of the camera-driven cascades,
// so injected radiance doesn't shift with the view. Point lights use the regular cube faces.
float calculateVXGIShadow(float3 fragWorldPos, float3 geometricNormal, float3 L, Light* light, uint shadowAtlasIdx)
{
    if (light.type == LightType::Point) {
        return calculatePointShadow(fragWorldPos, geometricNormal, L, light, shadowAtlasIdx);
    }

    // ~one voxel of normal offset against acne; the tile has no per-cascade texel metadata.
    float NdotL = saturate(abs(dot(geometricNormal, L)));
    float3 biasedPos = fragWorldPos + geometricNormal * (1.0 - NdotL) * 0.2;

    float4 fragPosLightSpace = mul(light.shadowMaps[0].lightSpaceMatrix, float4(biasedPos, 1.0));
    float3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords.x = projCoords.x * 0.5 + 0.5;
    projCoords.y = projCoords.y * 0.5 + 0.5;

    if (projCoords.z < 0.0 || projCoords.z > 1.0) {
        return 0.0;
    }
    // Half-texel clamp keeps the lookup inside the tile (VXGI_DIRECTIONAL_SHADOW_RESOLUTION).
    float texelSize = 1.0 / 256.0;
    projCoords.xy = clamp(projCoords.xy, 0.5 * texelSize, 1.0 - 0.5 * texelSize);

    float currentDepth = projCoords.z - 0.0001f;

    float4 atlasRange = light.shadowMaps[0].shadowAtlasRange;
    float2 rangeMin  = atlasRange.xy;
    float2 rangeSpan = atlasRange.zw - atlasRange.xy;

    float2 atlasUV = rangeMin + projCoords.xy * rangeSpan;
    float lit = textures[shadowAtlasIdx].SampleCmpLevelZero(
        shadowCmpSampler,
        atlasUV,
        currentDepth
    );
    return 1.0 - lit;
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
    LitMeshDrawData mesh = pc.litMeshDrawData[drawID + pc.drawIDOffset];
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
    output.texIndices = uint4(draw.albedoTextureIndex, draw.roughnessTextureIndex, draw.metallicTextureIndex, draw.normalTextureIndex);
    output.matInfo = uint3(draw.environmentMapIndex, draw.materialFlags, draw.packedColor);
    output.matParams = float3(draw.metallic, draw.roughness, draw.alphaCutoff);
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

// 0: cubic
// 1: quintic
#define INTERPOLANT 0
float hash(int2 p)  // this hash is not production ready, please
{                         // replace this by something better
    // 2D -> 1D
    int n = p.x*3 + p.y*113;

    // 1D hash by Hugo Elias
	n = (n << 13) ^ n;
    n = n * (n * n * 15731 + 789221) + 1376312589;
    return -1.0+2.0*float( n & 0x0fffffff)/float(0x0fffffff);
}

float mix(float x, float y, float a) {
    return (x*(1.0f - a) + y * a);
}

float valueNoise(float2 p)
{
    int2 i = int2(floor( p ));
    float2 f = fract( p );
	
    #if INTERPOLANT==1
    // quintic interpolant
    float2 u = f*f*f*(f*(f*6.0-15.0)+10.0);
    #else
    // cubic interpolant
    float2 u = f*f*(3.0-2.0*f);
    #endif    

    return mix( mix( hash( i + int2(0,0) ),
                     hash( i + int2(1,0) ), u.x),
                mix( hash( i + int2(0,1) ),
                     hash( i + int2(1,1) ), u.x), u.y);
}

// -----------------------------------------------