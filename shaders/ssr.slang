import modules.common;
import modules.pbr;

struct SSRPushConstants {
    SSRPassData* passData;
};

struct SSRPassData {
    float4x4 invViewProj;
    float4x4 viewProj;
    float3 cameraPos;
    uint depthIndex;
    uint depthSamplerIndex;
    uint colorIndex;
    uint colorSamplerIndex;
    uint roughnessMetalIndex;
    uint roughnessMetalSamplerIndex;
    uint normalIndex;
    uint normalSamplerIndex;
    uint _pad_resolution;
    uint2 resolution;
    float maxDistance;
    uint hiZIndex;
    uint hiZMipLevels;
    float thickness;
    float roughnessThreshold;
    int maxSteps;
    uint frameIndex;
};

[[vk::push_constant]] 
ConstantBuffer<SSRPushConstants> SSRpc;

struct VSOutput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

[shader("vertex")]
VSOutput vertMain(uint vertexID : SV_VertexID) {
    VSOutput output;
    float2 uv = float2((vertexID << 1) & 2, vertexID & 2);
    output.texCoord = uv;
    output.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    return output;
}

// Project world position to screen-space UV (.xy) + projected depth (.z)
float3 worldToScreen(float3 worldPos) {
    float4 clip = mul(SSRpc.passData.viewProj, float4(worldPos, 1.0));
    clip.xyz /= clip.w;
    float2 uv = clip.xy * 0.5 + 0.5;
    return float3(uv, clip.z);
}

struct RayTraceOutput {
    bool Hit;
    float2 UV;
};

RayTraceOutput RayTrace(float3 worldPos, float3 reflDir, float jitter) {
    RayTraceOutput output;
    output.Hit = false;
    output.UV = float2(0, 0);

    SamplerState depthSampler = samplers[SSRpc.passData.depthSamplerIndex];
    float2 res = float2(SSRpc.passData.resolution);
    int maxLevel = max(0, int(SSRpc.passData.hiZMipLevels) - 1);

    // Project ray origin and endpoint to screen space (UV + depth)
    float3 ssOrigin = worldToScreen(worldPos);
    float3 ssEnd = worldToScreen(worldPos + reflDir * SSRpc.passData.maxDistance);
    float3 ssDir = ssEnd - ssOrigin;

    // Skip degenerate rays (< 1 pixel of screen movement)
    float pixelLen = length(ssDir.xy * res);
    if (pixelLen < 1.0)
        return output;

    // Parametric t per pixel (for epsilon calculations)
    float tPerPixel = 1.0 / pixelLen;
    // Minimum advancement per step to prevent stalling on cell boundaries
    float minStep = tPerPixel * 1.0;

    // Start slightly offset to avoid self-intersection, jittered to break grid artifacts
    float t = tPerPixel * (2.0 + jitter);
    int level = min(max(maxLevel - 2, 3), maxLevel);
    // Track consecutive empty cells to accelerate mip ascent
    int emptyStreak = 0;

    for (int i = 0; i < SSRpc.passData.maxSteps; i++) {
        float3 pos = ssOrigin + ssDir * t;

        // Bounds check
        if (any(pos.xy < 0.0) || any(pos.xy > 1.0) || pos.z < 0.0 || pos.z > 1.0 || t >= 1.0)
            break;

        // Cell size at current mip level
        float2 mipRes = max(floor(res * exp2(-float(level))), float2(1, 1));

        // Next cell boundary in the ray's direction
        float2 pixelCoord = pos.xy * mipRes;
        float2 nextBound;
        nextBound.x = ssDir.x > 0 ? floor(pixelCoord.x) + 1.0 : ceil(pixelCoord.x) - 1.0;
        nextBound.y = ssDir.y > 0 ? floor(pixelCoord.y) + 1.0 : ceil(pixelCoord.y) - 1.0;
        float2 boundaryUV = nextBound / mipRes;

        // Parametric distance to each cell boundary
        float2 tBound;
        tBound.x = abs(ssDir.x) > 1e-8 ? (boundaryUV.x - pos.x) / ssDir.x : 1e10;
        tBound.y = abs(ssDir.y) > 1e-8 ? (boundaryUV.y - pos.y) / ssDir.y : 1e10;
        float tCell = max(min(max(0.0, tBound.x), max(0.0, tBound.y)), minStep);

        // Sample HiZ min-depth at current position and mip
        float minDepth = textures[SSRpc.passData.hiZIndex].SampleLevel(depthSampler, pos.xy, level).r;

        // Conservative test: max ray depth within this cell vs min surface depth
        float exitDepth = pos.z + ssDir.z * tCell;
        float rayMaxZ = max(pos.z, exitDepth);

        if (rayMaxZ > minDepth) {
            // Ray potentially intersects geometry in this cell
            emptyStreak = 0;
            if (level == 0) {
                // At finest level - verify with thickness test
                if (pos.z > minDepth) {
                    float3 surfaceWorldPos = reconstructViewPos(pos.xy, minDepth, SSRpc.passData.invViewProj);
                    float3 rayWorldPos = reconstructViewPos(pos.xy, pos.z, SSRpc.passData.invViewProj);
                    if (length(rayWorldPos - surfaceWorldPos) < SSRpc.passData.thickness) {
                        output.Hit = true;
                        output.UV = pos.xy;
                        return output;
                    }
                }
                // Failed thickness test - skip past this surface and resume coarse traversal
                t += tCell;
                level = min(2, maxLevel);
            } else {
                // Descend to finer mip for more precision
                level--;
            }
        } else {
            // Ray is in front of everything in this cell - skip it
            t += tCell;
            emptyStreak++;
            // Ascend mip aggressively: jump up by 2 after consecutive empty cells
            if (emptyStreak >= 2 && level + 2 <= maxLevel) {
                level += 2;
                emptyStreak = 0;
            } else if (level < maxLevel) {
                level++;
            }
        }
    }

    return output;
}
#define M1 1597334677U     //1719413*929
#define M2 3812015801U     //140473*2467*11

float2 hash2D( uint2 q, uint seed )
{
    q *= uint2(M1, M2); 
    
    uint n = ((q.x ^ q.y) ^ seed) * M1;
    
    return float2(float(n) * (1.0/float(0xffffffffU)),float(n+1) * (1.0/float(0xffffffffU)));
}


[shader("fragment")]
float4 fragMain(VSOutput input) : SV_Target {
    float2 uv = input.texCoord;

    float roughness = textures[SSRpc.passData.roughnessMetalIndex].SampleLevel(samplers[SSRpc.passData.roughnessMetalSamplerIndex], uv,0).r;

    if (roughness > SSRpc.passData.roughnessThreshold)
        return float4(0, 0, 0, 0);

    float initDepth = textures[SSRpc.passData.depthIndex].SampleLevel(samplers[SSRpc.passData.depthSamplerIndex], uv, 0).r;
    float3 viewPos = reconstructViewPos(uv, initDepth, SSRpc.passData.invViewProj);

    float3 normalData = textures[SSRpc.passData.normalIndex].Sample(samplers[SSRpc.passData.normalSamplerIndex], uv).xyz;
    float3 normal = normalize(2.0f * normalData - 1.0f);

    float3 viewDir = normalize(viewPos - SSRpc.passData.cameraPos);

    // For near-mirror surfaces, skip importance sampling and use perfect reflection
    if (roughness < 0.01) {
        float2 mirrorXi = hash2D(uint2(uv * float2(SSRpc.passData.resolution)), SSRpc.passData.frameIndex);
        float3 reflectDir = normalize(reflect(viewDir, normal));
        RayTraceOutput ray = RayTrace(viewPos, reflectDir, mirrorXi.x);
        if (ray.Hit) {
            float3 color = textures[SSRpc.passData.colorIndex].Sample(samplers[SSRpc.passData.colorSamplerIndex], ray.UV).rgb;
            return float4(color, 1.0);
        }
        return float4(0, 0, 0, 0);
    }

    // Importance sample GGX: multiple samples per pixel, varied across frames
    float3 totalColor = float3(0, 0, 0);
    float totalWeight = 0.0;
    float3x3 TBN = buildTBN(normal);

    int sampleCount = 1;
    for (uint s = 0; s < sampleCount; s++) {
        float2 xi = hash2D(uint2(uv * float2(SSRpc.passData.resolution)), SSRpc.passData.frameIndex * sampleCount + s);

        float3 H = normalize(mul(sampleGGX(xi, roughness), TBN));
        float3 L = normalize(reflect(viewDir, H));

        float NdotL = dot(normal, L);
        if (NdotL <= 0.0)
            continue;

        RayTraceOutput ray = RayTrace(viewPos, L, xi.x);
        if (ray.Hit) {
            float3 color = textures[SSRpc.passData.colorIndex].Sample(samplers[SSRpc.passData.colorSamplerIndex], ray.UV).rgb;
            totalColor += color * NdotL;
            totalWeight += NdotL;
        }
    }

    if (totalWeight > 0.0) {
        return float4(totalColor / totalWeight, 1.0 - roughness);
    }

    return float4(0, 0, 0, 0);
}
