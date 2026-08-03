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
    uint hiZStopLevel;
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

// Project world position to screen-space UV (.xy) + projected reverse-Z depth (.z).
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

// Advance the ray by a depth delta: d.z == 1, so the step lands exactly on the target plane.
// Relative (from current position, not a depth-0 origin) to keep intermediates small when
// d.xy blows up on grazing rays.
float3 intersectDepthPlane(float3 o, float3 d, float t) {
    return o + d * t;
}

// Get the cell (tile) index for a given position at a given mip level size
float2 getCell(float2 ray, float2 cellCount) {
    return floor(ray * cellCount);
}

// Find the intersection of the ray with the next cell boundary.
// crossOffset is a fraction of a cell (signed by travel direction) so the nudge past
// the boundary scales with the level — escapes coarse cells, stays sub-pixel at fine ones.
float3 intersectCellBoundary(float3 o, float3 d, float2 cellIndex, float2 cellCount, float2 crossStep, float2 crossOffset) {
    float2 cellSize = 1.0 / cellCount;
    float2 nextBoundary = (cellIndex + crossStep + crossOffset) * cellSize;
    float2 t = (nextBoundary - o.xy) / d.xy;
    // Axis-aligned rays: ignore the parallel axis (division gave ±inf)
    t.x = d.x != 0.0 ? t.x : -1e30;
    t.y = d.y != 0.0 ? t.y : -1e30;
    // Reverse-Z: the ray marches with DECREASING depth (t), so the first boundary crossed is the
    // largest t below the current position -> max, not min.
    float tMin = max(t.x, t.y);
    return o + d * tMin;
}

// Get the cell count at a given mip level, from a hoisted base resolution.
// Mip dimensions floor at each level (w >> level), so floor here to match
// the actual Hi-Z texel grid instead of assuming exact power-of-two division.
float2 getCellCount(float level, float2 baseRes) {
    // clamp at 1 like mip dims do, so cellSize never divides by zero
    return max(floor(baseRes / exp2(level)), 1.0);
}

// Fetch the nearest depth plane for a Hi-Z cell at a given mip level.
// Reverse-Z: Hi-Z stores raw reverse depth max-reduced, so nearest occluder = maximum.
// Load (texel fetch): no sampler, and the cell tested is exactly the cell computed —
// UV sampling can round into a neighboring texel at cell boundaries.
float getNearestDepthPlane(Texture2D hiZ, float2 cellIdx, float level) {
    return hiZ.Load(int3(int2(cellIdx), int(level))).r;
}

// Check if the ray crossed a cell boundary
bool crossedCellBoundary(float2 oldCellIdx, float2 newCellIdx) {
    return any(oldCellIdx != newCellIdx);
}

/*
*   p is screen space point
*   v is screen space ray direction
*/
// Start level: where the walk begins — lower = cheaper nearby hits, more ascent for far ones.
// Max level: ceiling the walk can climb back to; bounds the largest skip (2^level px).
// Coarser than ~6 the max-reduced plane covers so much screen the ray rarely gets to advance.
#define HIZ_START_LEVEL 2
#define HIZ_MAX_LEVEL 6
// Stop level is derived from the SSR resolution scale (CPU side) so one Hi-Z cell at the
// stop level maps to ~one output pixel — refining finer than that is wasted work.

float3 RayTrace(float3 p, float3 v, float jitter) {
    // Hoist loop-invariant buffer-pointer reads into locals
    const int maxSteps = SSRpc.passData.maxSteps;
    const float stopLevel = float(SSRpc.passData.hiZStopLevel);
    Texture2D hiZ = textures[SSRpc.passData.hiZIndex];
    const float2 baseRes = float2(SSRpc.passData.resolution);

    // Depth-parameterized direction: d.z == 1, so stepping d by a depth delta lands on that
    // plane. Reverse-Z: depth DECREASES along the ray; clamp v.z so d stays finite on
    // depth-parallel (grazing) rays.
    float3 d = v / min(v.z, -1e-7);

    // Cell crossing direction follows v, not d: v.z < 0 flips d.xy backwards
    float2 crossStep = float2(v.x >= 0.0 ? 1.0 : 0.0, v.y >= 0.0 ? 1.0 : 0.0);
    float2 crossOffset = (crossStep * 2.0 - 1.0) / 128.0;

    // Cross out of the origin's mip 0 texel (half-texel offset) to avoid immediate
    // self-intersection, instead of the old fixed 1% jump that skipped nearby geometry
    float2 mip0CellCount = getCellCount(0.0, baseRes);
    float3 ray = intersectCellBoundary(p, d, getCell(p.xy, mip0CellCount), mip0CellCount, crossStep, crossOffset * 64.0);

    // Per-pixel/frame phase jitter along the ray, up to one start-level cell. Movement is
    // along the ray line so converged hits are unaffected; only where step-starved rays
    // stall decorrelates, letting temporal accumulation average the blocks out.
    float baseT = 1.0 / max(abs(v.x) * baseRes.x, abs(v.y) * baseRes.y);
    ray += v * (baseT * exp2(HIZ_START_LEVEL) * jitter);

    float level = HIZ_START_LEVEL;
    for(int i = 0; i < maxSteps && level >= stopLevel; i++) {
        // Ray left the screen: no hit possible, stop fetching (border fade rejects the UV)
        if(any(ray.xy < 0.0) || any(ray.xy >= 1.0))
            break;

        float2 cellCount = getCellCount(level, baseRes);
        float2 oldCellIdx = getCell(ray.xy, cellCount);

        float planeZ = getNearestDepthPlane(hiZ, oldCellIdx, level);

        // In front of the cell's nearest plane (reverse-Z: larger = nearer): drop onto it.
        // At or behind it: stay put and refine a level down.
        float3 tmpRay = intersectDepthPlane(ray, d, min(0.0, planeZ - ray.z));

        float2 newCellIdx = getCell(tmpRay.xy, cellCount);
        if(crossedCellBoundary(oldCellIdx, newCellIdx)) {
            // Plane lies beyond the cell: clamp to the cell boundary and go coarser
            tmpRay = intersectCellBoundary(ray, d, oldCellIdx, cellCount, crossStep, crossOffset);
            level = min(level + 2.0, HIZ_MAX_LEVEL + 1.0);
        }

        level -= 1.0;
        ray = tmpRay;
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

    if (initDepth <= 0.0) // reverse-Z: skybox/far = 0
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

    // P: screen-space origin (native reverse-Z depth)
    float3 p = float3(uv, initDepth);

    // P': project reflected point into screen space
    float3 pPrime = worldToScreen(worldPos + reflectDir * 10.0f);

    // Screen-space ray direction for tracing
    float3 v = pPrime - p;

    // Reject rays going toward the camera in screen-space depth (reverse-Z: nearer = larger z)
    if (v.z > 0.0)
        return float4(0, 0, 0, 0);

    float jitter = hash2D(uint2(input.position.xy), SSRpc.passData.frameIndex).x;
    float3 ray = RayTrace(p, v, jitter);
    //return float4(ray,1.0);
    float reflDepth = ray.z;

    float4 realDepth = textures[SSRpc.passData.hiZIndex].SampleLevel(samplers[SSRpc.passData.depthSamplerIndex],ray.xy,0);

    // Screen-edge fade: smoothly fade out reflections near screen borders
    float2 edgeFactor = smoothstep(0.0, 0.1, ray.xy) * (1.0 - smoothstep(0.9, 1.0, ray.xy));
    float fadeOnBorder = min(edgeFactor.x, edgeFactor.y);

    if(fadeOnBorder <= 0.0)
        return float4(0,0,0,0);

    if(abs(reflDepth - realDepth.r) > SSRpc.passData.thickness)
        return float4(0,0,0,0);

    float gloss = 1.0 - roughness / SSRpc.passData.roughnessThreshold;
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
    float minFade = 0.8;
    float maxFade = 0.0;
    gloss = gloss * (minFade-maxFade) + maxFade;
    float fadeOnRoughness = gloss * gloss;

    float totalFade = fadeOnBorder * fadeOnPerpendicular * fadeOnRoughness;

    return float4(reflColor, totalFade);
}
