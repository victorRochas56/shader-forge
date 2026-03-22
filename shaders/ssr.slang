import modules.common;

struct SSRPushConstants {
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
    uint2 resolution;
    float maxDistance;
    uint hiZIndex;
    uint hiZMipLevels;
    float thickness;
    float roughnessThreshold;
    int maxSteps;
};

[[vk::push_constant]] SSRPushConstants SSRpc;

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
    float4 clip = mul(SSRpc.viewProj, float4(worldPos, 1.0));
    clip.xyz /= clip.w;
    float2 uv = clip.xy * 0.5 + 0.5;
    return float3(uv, clip.z);
}

struct RayTraceOutput {
    bool Hit;
    float2 UV;
};

RayTraceOutput RayTrace(float3 worldPos, float3 reflDir) {
    RayTraceOutput output;
    output.Hit = false;
    output.UV = float2(0, 0);

    SamplerState depthSampler = samplers[SSRpc.depthSamplerIndex];
    float2 res = float2(SSRpc.resolution);
    int maxLevel = max(0, int(SSRpc.hiZMipLevels) - 1);

    // Project ray origin and endpoint to screen space (UV + depth)
    float3 ssOrigin = worldToScreen(worldPos);
    float3 ssEnd = worldToScreen(worldPos + reflDir * SSRpc.maxDistance);
    float3 ssDir = ssEnd - ssOrigin;

    // Skip degenerate rays (< 1 pixel of screen movement)
    float pixelLen = length(ssDir.xy * res);
    if (pixelLen < 1.0)
        return output;

    // Parametric t per pixel (for epsilon calculations)
    float tPerPixel = 1.0 / pixelLen;

    // Start slightly offset to avoid self-intersection
    float t = tPerPixel * 2.0;
    int level = min(3, maxLevel);

    for (int i = 0; i < SSRpc.maxSteps; i++) {
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
        float tCell = min(max(0.0, tBound.x), max(0.0, tBound.y));

        // Sample HiZ min-depth at current position and mip
        float minDepth = textures[SSRpc.hiZIndex].SampleLevel(depthSampler, pos.xy, level).r;

        // Conservative test: max ray depth within this cell vs min surface depth
        float exitDepth = pos.z + ssDir.z * tCell;
        float rayMaxZ = max(pos.z, exitDepth);

        if (rayMaxZ > minDepth) {
            // Ray potentially intersects geometry in this cell
            if (level == 0) {
                // At finest level - verify with thickness test
                if (pos.z > minDepth) {
                    float3 surfaceWorldPos = reconstructViewPos(pos.xy, minDepth, SSRpc.invViewProj);
                    float3 rayWorldPos = reconstructViewPos(pos.xy, pos.z, SSRpc.invViewProj);
                    if (length(rayWorldPos - surfaceWorldPos) < SSRpc.thickness) {
                        output.Hit = true;
                        output.UV = pos.xy;
                        return output;
                    }
                }
                // Failed thickness test or ray not yet behind surface - advance
                t += tCell + tPerPixel * 0.5;
            } else {
                // Descend to finer mip for more precision
                level--;
            }
        } else {
            // Ray is in front of everything in this cell - skip it
            t += tCell + tPerPixel * 0.5;
            // Ascend mip for faster traversal
            if (level < maxLevel) level++;
        }
    }

    return output;
}



[shader("fragment")]
float4 fragMain(VSOutput input) : SV_Target {
    float2 uv = input.texCoord;

    // Early out for rough surfaces
    float roughness = textures[SSRpc.roughnessMetalIndex].Sample(samplers[SSRpc.roughnessMetalSamplerIndex], uv).r;
    if (roughness > SSRpc.roughnessThreshold)
        return float4(0, 0, 0, 0);

    float initDepth = textures[SSRpc.depthIndex].SampleLevel(samplers[SSRpc.depthSamplerIndex], uv, 0).r;
    float3 worldPos = reconstructViewPos(uv, initDepth, SSRpc.invViewProj);

    float3 normalData = textures[SSRpc.normalIndex].Sample(samplers[SSRpc.normalSamplerIndex], uv).xyz;
    float3 normal = normalize(2.0f * normalData - 1.0f);

    float3 viewDir = normalize(worldPos - SSRpc.cameraPos);
    float3 reflectDir = normalize(reflect(viewDir, normal));

    RayTraceOutput ray = RayTrace(worldPos, reflectDir);

    if (ray.Hit) {
        float3 color = textures[SSRpc.colorIndex].Sample(samplers[SSRpc.colorSamplerIndex], ray.UV).rgb;
        float fade = 1.0 - (roughness / SSRpc.roughnessThreshold);
        return float4(color, fade);
    }

    return float4(0, 0, 0, 0);
}
