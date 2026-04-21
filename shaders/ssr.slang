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

// Advance ray to a given depth: since d = v/v.z, d.z == 1, so t == depth
float3 intersectDepthPlane(float3 o, float3 d, float t) {
    return o + d * t;
}

// Get the cell (tile) index for a given position at a given mip level size
float2 getCell(float2 ray, float2 cellCount) {
    return floor(ray * cellCount);
}

// Find the intersection of the ray with the next cell boundary
float3 intersectCellBoundary(float3 o, float3 d, float2 cellIndex, float2 cellCount, float2 crossStep, float2 crossOffset) {
    float2 cellSize = 1.0 / cellCount;
    float2 nextBoundary = (cellIndex + crossStep) * cellSize + crossOffset;
    float2 t = (nextBoundary - o.xy) / d.xy;
    float tMin = min(t.x, t.y);
    return o + d * tMin;
}

// Get the cell count at a given mip level
float2 getCellCount(float level, float rootLevel) {
    return float2(SSRpc.passData.resolution) / exp2(level);
}

// Sample the minimum depth from the Hi-Z buffer at a given mip level
float getMinimumDepthPlane(float2 ray, float level, float rootLevel) {
    return textures[SSRpc.passData.hiZIndex].SampleLevel(samplers[SSRpc.passData.depthSamplerIndex], ray, level).r;
}

// Check if the ray crossed a cell boundary
bool crossedCellBoundary(float2 oldCellIdx, float2 newCellIdx) {
    return any(oldCellIdx != newCellIdx);
}

/*
*   p is screen space point
*   v is screen space ray direction
*/
#define HIZ_START_LEVEL 2
#define HIZ_STOP_LEVEL 2

float3 RayTrace(float3 p, float3 v) {
    const float rootLevel = SSRpc.passData.hiZMipLevels - 1.0f;

    float level = HIZ_START_LEVEL;

    float iterations = 0.0f;

    float2 crossStep;
    float2 crossOffset;

    crossStep.x = (v.x >= 0)? 1.f : -1.f;
    crossStep.y = (v.y >= 0)? 1.f : -1.f;
    crossOffset.xy = crossStep.xy * 0.025;
    crossStep.xy = saturate(crossStep.xy);

    float3 ray = p.xyz;

    float3 d = v.xyz /= v.z;

    float3 o = intersectDepthPlane(p, d, -p.z);

    float2 hiZSize = float2(SSRpc.passData.resolution);
    float2 rayCell = getCell(ray.xy, hiZSize);
    ray = intersectCellBoundary(o, d, rayCell, hiZSize, crossStep.xy, crossOffset.xy);

    [loop]
    while( level >= HIZ_STOP_LEVEL && iterations < SSRpc.passData.maxSteps){
        
        float minZ = getMinimumDepthPlane(ray.xy,level,rootLevel);

        const float2 cellCount = getCellCount(level,rootLevel);
        const float2 oldCellIdx = getCell(ray.xy, cellCount);

        float3 tmpRay = intersectDepthPlane(o,d,max(ray.z,minZ));

        const float2 newCellIdx = getCell(tmpRay.xy,cellCount);

        [branch]
        if(crossedCellBoundary(oldCellIdx,newCellIdx)){
            tmpRay = intersectCellBoundary(o,d,oldCellIdx,cellCount.xy,crossStep.xy,crossOffset.xy);
            level = min(rootLevel, level + 2.0f);
        }
        ray.xyz = tmpRay.xyz;

        --level;

        ++iterations;
    }

    return ray;
}


#define M1 1597334677U     //1719413*929
#define M2 3812015801U     //140473*2467*11
// Convert roughness [0,1] to Blinn-Phong specular power
float roughnessToSpecularPower(float roughness) {
    float a = max(roughness, 0.001);
    return 2.0 / (a * a) - 2.0;
}

// Convert specular power to cone half-angle (radians)
float specularPowerToConeAngle(float specularPower) {
    // Based on the relationship between specular lobe and cone angle
    // At low specular power (rough) -> wide cone, high specular power (smooth) -> narrow cone
    if (specularPower >= exp2(1024.0))
        return 0.0;
    const float xi = 0.244;
    return acos(pow(xi, 1.0 / (specularPower + 1.0)));
}

// Opposite side length of an isoceles triangle given the adjacent (leg) length and half-angle
float isoscelesTriangleOpposite(float adjacentLength, float coneTheta)
{
	return 2.0f * tan(coneTheta) * adjacentLength;
}

// Incircle radius of an isoceles triangle given adjacent (leg) and opposite side lengths
float isoscelesTriangleInRadius(float a, float h)
{
	float a2 = a * a;
	float fh2 = 4.0f * h * h;
	return (a * (sqrt(a2 + fh2) - a)) / (4.0f * h);
}

// Next adjacent length after stepping inward by the incircle diameter
float isoscelesTriangleNextAdjacent(float adjacentLength, float inCircleSize) {
    return adjacentLength - (inCircleSize * 2.0);
}

// Sample the color mip chain weighted for cone tracing accumulation
float4 coneSampleWeightedColor(float2 samplePos, float mipChannel, float gloss) {
    float mip = clamp(mipChannel, 0.0, float(SSRpc.passData.hiZMipLevels - 1));
    float3 color = textures[SSRpc.passData.colorIndex].SampleLevel(
        samplers[SSRpc.passData.colorSamplerIndex], samplePos, mip).rgb;
    // Weight by gloss — smooth surfaces get sharper (higher weight per sample),
    // rough surfaces spread weight across more samples
    float weight = gloss;
    return float4(color * weight, weight);
}

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

    if (initDepth >= 1.0)
        return float4(0, 0, 0, 0);

    // Reconstruct world position
    float3 worldPos = reconstructViewPos(uv, initDepth, SSRpc.passData.invViewProj);

    // View direction and world-space normal
    float3 viewDir = normalize(worldPos - SSRpc.passData.cameraPos);
    float3 N = textures[SSRpc.passData.normalIndex].SampleLevel(samplers[SSRpc.passData.normalSamplerIndex], uv, 0).rgb;
    N = normalize(N * 2.0 - 1.0);

    // World-space reflection direction
    float3 reflectDir = reflect(viewDir, N);

    if (dot(reflectDir, viewDir) < 0.0)
        return float4(0, 0, 0, 0);

    // P: screen-space origin
    float3 p = float3(uv, initDepth);

    // P': project reflected point into screen space
    float3 pPrime = worldToScreen(worldPos + reflectDir * 10.0f);

    // Screen-space ray direction for tracing
    float3 v = pPrime - p;

    // Reject rays going toward the camera in screen-space depth
    if (v.z < 0.0)
        return float4(0, 0, 0, 0);

    float3 ray = RayTrace(p,v);

    float reflDepth = ray.z;
    float4 realDepth = textures[SSRpc.passData.hiZIndex].SampleLevel(samplers[SSRpc.passData.depthSamplerIndex],ray.xy,0);

    // Screen-edge fade: smoothly fade out reflections near screen borders
    float2 edgeFactor = smoothstep(0.0, 0.1, ray.xy) * (1.0 - smoothstep(0.9, 1.0, ray.xy));
    float fadeOnBorder = min(edgeFactor.x, edgeFactor.y);

    if(fadeOnBorder <= 0.0)
        return float4(0,0,0,0);

    if(abs(reflDepth - realDepth.r) > SSRpc.passData.thickness)
        return float4(0,0,0,0);

    float gloss = 1.0 - roughness;
    float specularPower = roughnessToSpecularPower(roughness);
    float coneTheta = specularPowerToConeAngle(specularPower);

    // Cone trace setup
    float2 screenPos = uv;
    float adjacentLength = length(ray.xy - uv);

    // Guard against degenerate rays (near-zero screen-space travel)
    if(adjacentLength < 1e-5)
        return float4(textures[SSRpc.passData.colorIndex].SampleLevel(
            samplers[SSRpc.passData.colorSamplerIndex], ray.xy, 0).rgb, fadeOnBorder);

    float2 adjacentUnit = (ray.xy - uv) / adjacentLength;
    float4 totalColor = float4(0, 0, 0, 0);
    float glossMult = gloss;
    float maxMipLevel = float(SSRpc.passData.hiZMipLevels - 1);

    float remainingAlpha = 1.0f;
    for(int i = 0; i < 14; i++) {

        float oppositeLength = isoscelesTriangleOpposite(adjacentLength, coneTheta);

        float inCircleSize = isoscelesTriangleInRadius(oppositeLength, adjacentLength);

        float2 samplePos = screenPos.xy + adjacentUnit * (adjacentLength - inCircleSize);

        float mipChannel = clamp(log2(inCircleSize * max(float(SSRpc.passData.resolution.x), float(SSRpc.passData.resolution.y))), 0.0, maxMipLevel);

        float4 newColor = coneSampleWeightedColor(samplePos, mipChannel, glossMult);
        remainingAlpha -= newColor.a;
        if(remainingAlpha < 0.0) {
            newColor.rgb *= (1.0 - abs(remainingAlpha));
        }
        totalColor += newColor;
        if(totalColor.a >= 1.0f) {
            break;
        }

        adjacentLength = isoscelesTriangleNextAdjacent(adjacentLength, inCircleSize);
        if(adjacentLength <= 0.0)
            break;
        glossMult *= gloss;
    }

    float3 reflColor = totalColor.rgb / max(totalColor.a, 0.001);

    // Fade on perpendicular rays (grazing angle reflections are stronger)
    float fadeOnPerpendicular = saturate(dot(reflectDir, viewDir) * 4.0);

    // Fade on roughness (quadratic falloff for more aggressive attenuation on rough surfaces)
    float fadeOnRoughness = gloss * gloss;

    float totalFade = fadeOnBorder * fadeOnPerpendicular * fadeOnRoughness;

    return float4(reflColor, totalFade);
}
