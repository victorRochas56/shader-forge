import modules.common;

struct SSRPushConstants {
    float4x4 invProjection;
    float4x4 viewMatrix;
    float4x4 projection;
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
    float normalMipLevel;
    int maxSteps;
    int padding0;
    int padding1;
    int padding2;
};

[[vk::push_constant]] SSRPushConstants SSRpc;

struct VSOutput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

// Project view-space position to UV coordinates
float2 viewToUV(float3 viewPos) {
    float2 uv;
    uv.x = (viewPos.x / (-viewPos.z) * (1.0 / SSRpc.invProjection[0][0])) * 0.5 + 0.5;
    uv.y = (viewPos.y / (-viewPos.z) * (1.0 / SSRpc.invProjection[1][1])) * 0.5 + 0.5;
    return uv;
}

// Linearize depth to view-space Z
float linearizeDepth(float depth) {
    return SSRpc.invProjection[2][3] / (SSRpc.invProjection[3][2] * depth + SSRpc.invProjection[3][3]);
}

// Convert view-space Z back to raw depth value (inverse of linearizeDepth)
float viewZToDepth(float viewZ) {
    return (SSRpc.invProjection[2][3] / viewZ - SSRpc.invProjection[3][3]) / SSRpc.invProjection[3][2];
}

[shader("vertex")]
VSOutput vertMain(uint vertexID : SV_VertexID) {
    VSOutput output;
    float2 uv = float2((vertexID << 1) & 2, vertexID & 2);
    output.texCoord = uv;
    output.texCoord.y = 1.0 - output.texCoord.y;
    output.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    return output;
}

// Hi-Z ray march: traces a ray through the hierarchical depth buffer.
// Starts at a coarse mip and refines down to mip 0 when a potential hit is found.
// Returns: hit UV in .xy, parametric t in .z, hit flag in .w (1.0 = hit)
float4 traceHiZ(float3 startView, float3 endView)
{
    float4x4 projection = SSRpc.projection;
    
    float2 texSize = float2(SSRpc.resolution);

    // Project endpoints to UV
    float4 clipStart = mul(projection, float4(startView, 1.0));
    float4 clipEnd   = mul(projection, float4(endView,   1.0));

    float2 startUV = clipStart.xy / clipStart.w * 0.5 + 0.5;
    float2 endUV   = clipEnd.xy   / clipEnd.w   * 0.5 + 0.5;

    float2 rayDir = endUV - startUV;

    int maxMip = min(int(SSRpc.hiZMipLevels) - 1, 5);
    int mip = min(2, maxMip);

    float minStep = 1.0 / max(abs(rayDir.x * texSize.x), abs(rayDir.y * texSize.y));
    float t = minStep * 2.0;

    for (int i = 0; i < SSRpc.maxSteps; i++)
    {
        float2 uv = startUV + rayDir * t;

        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || t > 1.0)
            break;

        // Perspective-correct interpolation of view-space position
        float viewZ = (startView.z * endView.z) / lerp(endView.z, startView.z, t);
        float3 rayPos = lerp(startView, endView, t);
        rayPos.z = viewZ;

        // Sample Hi-Z depth
        float hiZDepth = textures[SSRpc.hiZIndex]
            .SampleLevel(samplers[SSRpc.depthSamplerIndex], uv, float(mip)).r;

        // Reconstruct view-space position at sampled depth
        float3 samplePos = reconstructViewPos(uv, hiZDepth, SSRpc.invProjection);

        float depthDiff = samplePos.z - rayPos.z;

        if (depthDiff > 0.0 && hiZDepth < 1.0)
        {
            // Potential hit
            if (mip == 0)
            {
                if (depthDiff < SSRpc.thickness)
                {
                    return float4(uv, t, 1.0);
                }

                // Too far behind surface → advance
                t += minStep;
                mip = min(2, maxMip);
            }
            else
            {
                mip--;
            }
        }
        else
        {
            // Advance to next cell boundary
            float cellSize = float(1 << mip);
            float2 cellCount = texSize / cellSize;
            float2 cellCoord = uv * cellCount;

            float2 nextBoundary;
            nextBoundary.x = rayDir.x > 0 ? floor(cellCoord.x) + 1.0 : ceil(cellCoord.x) - 1.0;
            nextBoundary.y = rayDir.y > 0 ? floor(cellCoord.y) + 1.0 : ceil(cellCoord.y) - 1.0;

            float2 boundaryUV = nextBoundary / cellCount;

            float tX = abs(rayDir.x) > 1e-6 ? (boundaryUV.x - startUV.x) / rayDir.x : 1e30;
            float tY = abs(rayDir.y) > 1e-6 ? (boundaryUV.y - startUV.y) / rayDir.y : 1e30;

            float tNext = min(tX, tY);

            t = max(tNext + minStep * 0.1, t + minStep);
            mip = min(mip + 1, maxMip);
        }
    }

    return float4(0, 0, 0, 0);
}

[shader("fragment")]
float4 fragMain(VSOutput input) : SV_Target {
    float2 texSize = float2(SSRpc.resolution);
    float2 uv      = input.texCoord;

    // Early-out for high roughness surfaces
    float2 roughnessMetal = textures[SSRpc.roughnessMetalIndex].Sample(
        samplers[SSRpc.roughnessMetalSamplerIndex], uv).rg;
    float roughness = roughnessMetal.r;
    float metallic  = roughnessMetal.g;
    if (roughness > SSRpc.roughnessThreshold) return float4(0, 0, 0, 0);

    // Reconstruct view-space position, sample smooth normal from G-buffer
    float depth = textures[SSRpc.depthIndex].Sample(samplers[SSRpc.depthSamplerIndex], uv).r;
    if (depth >= 1.0) return float4(0, 0, 0, 0);

    float3 positionFrom = reconstructViewPos(uv, depth, SSRpc.invProjection);
    float3 worldNormal = textures[SSRpc.normalIndex].SampleLevel(samplers[SSRpc.normalSamplerIndex], uv, SSRpc.normalMipLevel).rgb * 2.0 - 1.0;
    float3 normal = normalize(mul((float3x3)SSRpc.viewMatrix, worldNormal));

    // Reflection vector in view space
    float3 unitPositionFrom = normalize(positionFrom);
    float3 pivot = normalize(reflect(unitPositionFrom, normal));

    // Ray endpoints in view space
    float3 startView = positionFrom;
    float3 endView   = positionFrom + pivot * SSRpc.maxDistance;

    // Clip ray endpoint to stay in front of camera (avoids mirrored UV projection)
    if (endView.z > -0.01) {
        float clipT = (-0.01 - startView.z) / (endView.z - startView.z);
        endView = startView + (endView - startView) * clipT;
    }

    // Project to UV space
    float2 startUV = viewToUV(startView);
    float2 endUV   = viewToUV(endView);

    // Hi-Z trace
    float4 traceResult = traceHiZ(startView, endView);
    float2 finalUV = traceResult.xy;
    float  search1 = traceResult.z;
    bool   hit     = traceResult.w > 0.5;

    if (!hit) return float4(0, 0, 0, 0);

    // Compute depth difference at hit for fade
    float hitDepth = textures[SSRpc.depthIndex].Sample(samplers[SSRpc.depthSamplerIndex], finalUV).r;
    float hitViewZ = linearizeDepth(hitDepth);
    float viewZ = (startView.z * endView.z) / lerp(endView.z, startView.z, search1);
    float depthDiff = hitViewZ - viewZ;

    // Fresnel-based reflectivity
    float3 viewDir = -unitPositionFrom;
    float NdotV = max(dot(normal, viewDir), 0.0);
    float F0 = lerp(0.04, 1.0, metallic);
    float fresnel = F0 + (1.0 - F0) * pow(1.0 - NdotV, 5.0);
    float reflectivity = fresnel * (1.0 - roughness);

    // Visibility with multiple fade factors
    float visibility = reflectivity
        * (1.0 - max(dot(-unitPositionFrom, pivot), 0.0))    // fade reflections pointing toward camera
        * (1.0 - clamp(depthDiff / SSRpc.thickness, 0.0, 1.0)) // fade at depth boundary
        * (1.0 - clamp(search1, 0.0, 1.0))                   // distance fade along ray
        * (finalUV.x < 0.0 || finalUV.x > 1.0 ? 0.0 : 1.0)  // screen edge cull
        * (finalUV.y < 0.0 || finalUV.y > 1.0 ? 0.0 : 1.0);
    visibility = clamp(visibility, 0.0, 1.0);

    // Sample scene color at reflection hit point
    float3 reflectedColor = textures[SSRpc.colorIndex].Sample(
        samplers[SSRpc.colorSamplerIndex], finalUV
    ).rgb;

    return float4(reflectedColor * visibility, visibility);
}
