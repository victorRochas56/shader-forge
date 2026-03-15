import modules.common;

struct SSRPushConstants {
    float4x4 invProjection;
    float4x4 viewMatrix;
    uint depthIndex;
    uint depthSamplerIndex;
    uint colorIndex;
    uint colorSamplerIndex;
    uint roughnessMetalIndex;
    uint roughnessMetalSamplerIndex;
    uint normalIndex;
    uint normalSamplerIndex;
    uint2 resolution;
    float maxDistance;       // max ray travel distance in view space
    float marchResolution;  // 0-1, fraction of pixels to sample in coarse pass
    int refinementSteps;    // binary search iterations after coarse hit
    float thickness;        // depth tolerance for hit detection
    float roughnessThreshold; // skip SSR for fragments above this roughness
    float normalMipLevel;     // mip level for normal pre-filtering
};

[[vk::push_constant]] SSRPushConstants SSRpc;

struct VSOutput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

// Project view-space position to UV coordinates
// Derives projection parameters from inverse projection matrix (same approach as SSAO)
float2 viewToUV(float3 viewPos) {
    float2 uv;
    uv.x = (viewPos.x / (-viewPos.z) * (1.0 / SSRpc.invProjection[0][0])) * 0.5 + 0.5;
    uv.y = (viewPos.y / (-viewPos.z) * (1.0 / SSRpc.invProjection[1][1])) * 0.5 + 0.5;
    return uv;
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

[shader("fragment")]
float4 fragMain(VSOutput input) : SV_Target {
    float maxDistance    = SSRpc.maxDistance;
    float rayResolution = SSRpc.marchResolution;
    int   steps         = SSRpc.refinementSteps;
    float thickness     = SSRpc.thickness;

    float2 texSize   = float2(SSRpc.resolution);
    float2 texelSize = 1.0 / texSize;
    float2 uv        = input.texCoord;

    // Early-out for high roughness surfaces (SSR contributes very little)
    float2 roughnessMetal = textures[SSRpc.roughnessMetalIndex].Sample(
        samplers[SSRpc.roughnessMetalSamplerIndex], uv).rg;
    float roughness = roughnessMetal.r;
    float metallic  = roughnessMetal.g;
    if (roughness > SSRpc.roughnessThreshold) return float4(0, 0, 0, 0);

    // Reconstruct view-space position from depth, sample smooth normal from G-buffer
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
    float3 endView   = positionFrom + pivot * maxDistance;

    // Project ray endpoints to screen-space pixel coordinates
    float2 startFrag = viewToUV(startView) * texSize;
    float2 endFrag   = viewToUV(endView)   * texSize;

    // Determine march direction along dominant screen axis
    float2 frag   = startFrag;
    float deltaX  = endFrag.x - startFrag.x;
    float deltaY  = endFrag.y - startFrag.y;
    float useX    = abs(deltaX) >= abs(deltaY) ? 1.0 : 0.0;
    float delta   = lerp(abs(deltaY), abs(deltaX), useX) * clamp(rayResolution, 0.0, 1.0);
    float2 increment = float2(deltaX, deltaY) / max(delta, 0.001);

    float search0  = 0.0;
    float search1  = 0.0;
    int   hit0     = 0;
    int   hit1     = 0;
    float viewDistance;
    float depthDiff = 0.0;

    // ---- Coarse pass: march along the ray in screen space ----
    for (int i = 0; i < int(delta); ++i) {
        frag += increment;
        float2 sampleUV = frag / texSize;

        if (sampleUV.x < 0.0 || sampleUV.x > 1.0 ||
            sampleUV.y < 0.0 || sampleUV.y > 1.0) break;

        float sampleDepth = textures[SSRpc.depthIndex].Sample(
            samplers[SSRpc.depthSamplerIndex], sampleUV).r;
        if (sampleDepth >= 1.0) continue;

        float3 positionTo = reconstructViewPos(sampleUV, sampleDepth, SSRpc.invProjection);

        search1 = lerp(
            (frag.y - startFrag.y) / deltaY,
            (frag.x - startFrag.x) / deltaX,
            useX
        );
        search1 = clamp(search1, 0.0, 1.0);

        // Perspective-correct interpolation of view-space Z
        viewDistance = (startView.z * endView.z) / lerp(endView.z, startView.z, search1);
        depthDiff = positionTo.z - viewDistance;

        if (depthDiff > 0.0 && depthDiff < thickness) {
            hit0 = 1;
            break;
        } else {
            search0 = search1;
        }
    }

    // ---- Refinement pass: binary search around coarse hit ----
    search1 = search0 + ((search1 - search0) / 2.0);
    int refinementSteps = steps * hit0;
    float2 finalUV = frag / texSize;

    for (int j = 0; j < refinementSteps; ++j) {
        frag    = lerp(startFrag, endFrag, search1);
        finalUV = frag / texSize;

        float sampleDepth = textures[SSRpc.depthIndex].Sample(
            samplers[SSRpc.depthSamplerIndex], finalUV).r;
        float3 positionTo = reconstructViewPos(finalUV, sampleDepth, SSRpc.invProjection);

        viewDistance = (startView.z * endView.z) / lerp(endView.z, startView.z, search1);
        depthDiff = positionTo.z - viewDistance;

        if (depthDiff > 0.0 && depthDiff < thickness) {
            hit1 = 1;
            search1 = search0 + ((search1 - search0) / 2.0);
        } else {
            float temp = search1;
            search1 = search1 + ((search1 - search0) / 2.0);
            search0 = temp;
        }
    }

    // Fresnel-based reflectivity: even dielectrics reflect strongly at grazing angles
    float3 viewDir = -unitPositionFrom;
    float NdotV = max(dot(normal, viewDir), 0.0);
    float F0 = lerp(0.04, 1.0, metallic);
    float fresnel = F0 + (1.0 - F0) * pow(1.0 - NdotV, 5.0);
    float reflectivity = fresnel * (1.0 - roughness);

    // Visibility with multiple fade factors
    float visibility = float(hit1)
        * reflectivity
        * (1.0 - max(dot(-unitPositionFrom, pivot), 0.0))    // fade reflections pointing toward camera
        * (1.0 - clamp(depthDiff / thickness, 0.0, 1.0))     // fade at depth boundary
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
