import modules.common;
import modules.pbr;

// GPU-built ray list for the tiled trace path: classifyMain compacts trace-worthy pixels and
// builds the dispatch-indirect args, traceMain runs one thread per surviving ray. Trace warps are
// therefore fully occupied instead of dragging rejected/early-out lanes through the whole march.
struct SSRRayHeader {
    uint groupsX; // dispatch-indirect args; y/z are filled with 1 CPU-side
    uint groupsY;
    uint groupsZ;
    uint rayCount;
};

#define SSR_TRACE_GROUP 64

struct SSRPushConstants {
    SSRPassData* passData;
    SSRRayHeader* rayHeader;
    uint* rayPixels; // rayHeader + 16 bytes, same buffer
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
    // Tiled/tunable trace additions (mirrored in structs.hpp).
    uint hiZStartLevel;    // where the walk begins — lower = cheaper nearby hits, more ascent for far ones
    uint hiZMaxLevel;      // ascend ceiling; bounds the largest skip (2^level px)
    uint outputImageIndex; // storage slot traceMain writes (ssr_current)
    uint2 ssrResolution;   // trace target size in pixels
    uint rayCapacity;      // rayPixels element capacity
    uint _pad2;
};

[[vk::push_constant]]
ConstantBuffer<SSRPushConstants> SSRpc;

// 2D alias of the storage-image binding (voxel.slang aliases the same binding as 3D).
[[vk::binding(4,0)]]
RWTexture2D<float4> storageImages2D[];

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

// Cell count at a given mip level. Integer shifts matching the actual mip dims (w >> level,
// floored at 1) — the old floor(baseRes / exp2(level)) burned an exp2+floor+max SFU chain
// on every march step for the same value.
float2 getCellCount(int level, uint2 baseRes) {
    return float2(max(baseRes >> uint(level), uint2(1, 1)));
}

// Fetch the nearest depth plane for a Hi-Z cell at a given mip level.
// Reverse-Z: Hi-Z stores raw reverse depth max-reduced, so nearest occluder = maximum.
// Load (texel fetch): no sampler, and the cell tested is exactly the cell computed —
// UV sampling can round into a neighboring texel at cell boundaries.
float getNearestDepthPlane(Texture2D hiZ, float2 cellIdx, int level) {
    return hiZ.Load(int3(int2(cellIdx), level)).r;
}

// Check if the ray crossed a cell boundary
bool crossedCellBoundary(float2 oldCellIdx, float2 newCellIdx) {
    return any(oldCellIdx != newCellIdx);
}

/*
*   p is screen space point
*   v is screen space ray direction
*/
// Start/max levels are runtime-tunable (SSRSettings). Stop level is derived from the SSR
// resolution scale (CPU side) so one Hi-Z cell at the stop level maps to ~one output pixel —
// refining finer than that is wasted work.
float3 RayTrace(float3 p, float3 v, float jitter) {
    // Hoist loop-invariant buffer-pointer reads into locals
    const int maxSteps = SSRpc.passData.maxSteps;
    const int stopLevel = int(SSRpc.passData.hiZStopLevel);
    const int startLevel = int(SSRpc.passData.hiZStartLevel);
    const int maxLevel = int(SSRpc.passData.hiZMaxLevel);
    Texture2D hiZ = textures[SSRpc.passData.hiZIndex];
    const uint2 baseRes = SSRpc.passData.resolution;

    // Depth-parameterized direction: d.z == 1, so stepping d by a depth delta lands on that
    // plane. Reverse-Z: depth DECREASES along the ray; clamp v.z so d stays finite on
    // depth-parallel (grazing) rays.
    float3 d = v / min(v.z, -1e-7);

    // Cell crossing direction follows v, not d: v.z < 0 flips d.xy backwards
    float2 crossStep = float2(v.x >= 0.0 ? 1.0 : 0.0, v.y >= 0.0 ? 1.0 : 0.0);
    float2 crossOffset = (crossStep * 2.0 - 1.0) / 128.0;

    // Cross out of the origin's mip 0 texel (half-texel offset) to avoid immediate
    // self-intersection, instead of the old fixed 1% jump that skipped nearby geometry
    float2 mip0CellCount = getCellCount(0, baseRes);
    float3 ray = intersectCellBoundary(p, d, getCell(p.xy, mip0CellCount), mip0CellCount, crossStep, crossOffset * 64.0);

    // Per-pixel/frame phase jitter along the ray, up to one start-level cell. Movement is
    // along the ray line so converged hits are unaffected; only where step-starved rays
    // stall decorrelates, letting temporal accumulation average the blocks out.
    float baseT = 1.0 / max(abs(v.x) * float(baseRes.x), abs(v.y) * float(baseRes.y));
    ray += v * (baseT * float(1u << uint(startLevel)) * jitter);

    int level = startLevel;
    for (int i = 0; i < maxSteps && level >= stopLevel; i++) {
        // Ray left the screen: no hit possible, stop fetching (border fade rejects the UV)
        if (any(ray.xy < 0.0) || any(ray.xy >= 1.0))
            break;

        float2 cellCount = getCellCount(level, baseRes);
        float2 oldCellIdx = getCell(ray.xy, cellCount);

        float planeZ = getNearestDepthPlane(hiZ, oldCellIdx, level);

        // In front of the cell's nearest plane (reverse-Z: larger = nearer): drop onto it.
        // At or behind it: stay put and refine a level down.
        float3 tmpRay = intersectDepthPlane(ray, d, min(0.0, planeZ - ray.z));

        float2 newCellIdx = getCell(tmpRay.xy, cellCount);
        if (crossedCellBoundary(oldCellIdx, newCellIdx)) {
            // Plane lies beyond the cell: clamp to the cell boundary and go coarser
            tmpRay = intersectCellBoundary(ray, d, oldCellIdx, cellCount, crossStep, crossOffset);
            level = min(level + 2, maxLevel + 1);
        }

        level -= 1;
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

// Cone-trace resolve along the reflection ray, closed form. The old loop rebuilt the
// isosceles-triangle chain per iteration, so each of the 14 fetches waited on the previous one
// (the pass's second-largest stall). But the incircle radius is proportional to the adjacent
// length:
//   r_n = kr * L_n,   L_{n+1} = (1 - 2*kr) * L_n,   kr = tan(θ) * (sec(θ) - tan(θ))
// so every tap position, mip and weight is known up front — the fetches issue in batches with
// no data dependency between them, and the alpha budget is replicated exactly in plain ALU.
#define CONE_TAPS 14
#define CONE_BATCH 7

float4 coneResolve(float2 uv, float2 hitUV, float adjacentLength, float gloss, float coneTheta, float maxMipLevel) {
    float2 adjacentUnit = (hitUV - uv) / adjacentLength;

    float t = tan(coneTheta);
    float kr = t * (sqrt(t * t + 1.0) - t); // incircle radius / adjacent length
    float k = max(1.0 - 2.0 * kr, 1e-4);    // adjacent shrink ratio per tap
    float log2k = log2(k);
    float maxRes = float(max(SSRpc.passData.resolution.x, SSRpc.passData.resolution.y));
    float mip0 = log2(max(kr * adjacentLength * maxRes, 1e-20));

    Texture2D colorTex = textures[SSRpc.passData.colorIndex];
    SamplerState colorSmp = samplers[SSRpc.passData.colorSamplerIndex];

    float3 rgb = float3(0.0);
    float totalAlpha = 0.0;
    float cum = 0.0; // accumulated weight (the old totalColor.a)
    float L = adjacentLength;
    float w = gloss;

    [ForceUnroll]
    for (int b = 0; b < CONE_TAPS / CONE_BATCH; b++) {
        float2 pos[CONE_BATCH];
        float mip[CONE_BATCH];
        float wRGB[CONE_BATCH];
        float wA[CONE_BATCH];
        [ForceUnroll]
        for (int j = 0; j < CONE_BATCH; j++) {
            int n = b * CONE_BATCH + j;
            pos[j] = uv + adjacentUnit * (L * (1.0 - kr));
            mip[j] = clamp(mip0 + float(n) * log2k, 0.0, maxMipLevel);
            // Weight/alpha budget matching the old loop exactly: full weight while the budget
            // lasts, the crossing tap's color scaled by the overshoot, nothing after saturation.
            float over = cum + w - 1.0;
            wRGB[j] = (cum < 1.0) ? w * ((over > 0.0) ? (1.0 - over) : 1.0) : 0.0;
            wA[j]   = (cum < 1.0) ? w : 0.0;
            cum += wA[j];
            L *= k;
            w *= gloss;
        }
        float3 s[CONE_BATCH];
        [ForceUnroll]
        for (int j = 0; j < CONE_BATCH; j++)
            s[j] = colorTex.SampleLevel(colorSmp, pos[j], mip[j]).rgb;
        [ForceUnroll]
        for (int j = 0; j < CONE_BATCH; j++) {
            rgb += s[j] * wRGB[j];
            totalAlpha += wA[j];
        }
    }
    return float4(rgb, totalAlpha);
}

float2 hash2D( uint2 q, uint seed )
{
    q *= uint2(M1, M2);

    uint n = ((q.x ^ q.y) ^ seed) * M1;

    return float2(float(n) * (1.0/float(0xffffffffU)),float(n+1) * (1.0/float(0xffffffffU)));
}

// Cheap pre-trace rejections, shared by the fragment path and classifyMain.
bool ssrShouldTrace(float2 uv) {
    float roughness = textures[SSRpc.passData.roughnessMetalIndex].SampleLevel(samplers[SSRpc.passData.roughnessMetalSamplerIndex], uv, 0).r;
    if (roughness > SSRpc.passData.roughnessThreshold)
        return false;

    float initDepth = textures[SSRpc.passData.depthIndex].SampleLevel(samplers[SSRpc.passData.depthSamplerIndex], uv, 0).r;
    if (initDepth <= 0.0) // reverse-Z: skybox/far = 0
        return false;

    float3 worldPos = reconstructViewPos(uv, initDepth, SSRpc.passData.invViewProj);
    float3 viewDir = normalize(worldPos - SSRpc.passData.cameraPos);
    float3 N = textures[SSRpc.passData.normalIndex].SampleLevel(samplers[SSRpc.passData.normalSamplerIndex], uv, 0).rgb;
    N = normalize(N * 2.0 - 1.0);
    float3 reflectDir = reflect(viewDir, N);

    if (dot(reflectDir, viewDir) < 0.0)
        return false;

    float3 pPrime = worldToScreen(worldPos + reflectDir * 10.0f);
    float3 v = pPrime - float3(uv, initDepth);

    // Reject rays going toward the camera in screen-space depth (reverse-Z: nearer = larger z)
    if (v.z > 0.0)
        return false;

    return true;
}

// Full per-pixel SSR: cheap rejects, Hi-Z march, cone resolve. pixelCoord drives the temporal jitter.
float4 ssrPixel(float2 uv, uint2 pixelCoord) {
    float roughness = textures[SSRpc.passData.roughnessMetalIndex].SampleLevel(samplers[SSRpc.passData.roughnessMetalSamplerIndex], uv, 0).r;
    float initDepth = textures[SSRpc.passData.depthIndex].SampleLevel(samplers[SSRpc.passData.depthSamplerIndex], uv, 0).r;

    // Warp-uniform early exit before any trace setup; per-lane exits below still apply. (On the
    // tiled path this is moot — classification never hands such pixels to traceMain.)
    bool cheapReject = (roughness > SSRpc.passData.roughnessThreshold) || (initDepth <= 0.0);
    if (WaveActiveAllTrue(cheapReject))
        return float4(0, 0, 0, 0);
    if (cheapReject)
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

    float jitter = hash2D(pixelCoord, SSRpc.passData.frameIndex).x;
    float3 ray = RayTrace(p, v, jitter);
    float reflDepth = ray.z;

    float4 realDepth = textures[SSRpc.passData.hiZIndex].SampleLevel(samplers[SSRpc.passData.depthSamplerIndex], ray.xy, 0);

    // Screen-edge fade: smoothly fade out reflections near screen borders
    float2 edgeFactor = smoothstep(0.0, 0.1, ray.xy) * (1.0 - smoothstep(0.9, 1.0, ray.xy));
    float fadeOnBorder = min(edgeFactor.x, edgeFactor.y);

    if (fadeOnBorder <= 0.0)
        return float4(0, 0, 0, 0);

    if (abs(reflDepth - realDepth.r) > SSRpc.passData.thickness)
        return float4(0, 0, 0, 0);

    float gloss = 1.0 - roughness / SSRpc.passData.roughnessThreshold;
    float specularPower = roughnessToSpecularPower(roughness);
    float coneTheta = specularPowerToConeAngle(specularPower);

    float adjacentLength = length(ray.xy - uv);

    // Guard against degenerate rays (near-zero screen-space travel)
    if (adjacentLength < 1e-5)
        return float4(textures[SSRpc.passData.colorIndex].SampleLevel(
            samplers[SSRpc.passData.colorSamplerIndex], ray.xy, 0).rgb, fadeOnBorder);

    float maxMipLevel = float(SSRpc.passData.hiZMipLevels - 1);
    float4 totalColor = coneResolve(uv, ray.xy, adjacentLength, gloss, coneTheta, maxMipLevel);

    float3 reflColor = totalColor.rgb / max(totalColor.a, 0.001);

    // Fade on perpendicular rays (grazing angle reflections are stronger)
    float fadeOnPerpendicular = saturate(dot(reflectDir, viewDir) * 4.0);

    // Fade on roughness (quadratic falloff for more aggressive attenuation on rough surfaces)
    float minFade = 0.8;
    float maxFade = 0.0;
    gloss = gloss * (minFade - maxFade) + maxFade;
    float fadeOnRoughness = gloss * gloss;

    float totalFade = fadeOnBorder * fadeOnPerpendicular * fadeOnRoughness;

    return float4(reflColor, totalFade);
}

// Fullscreen fallback path (features.ssr.tiledTrace == false).
[shader("fragment")]
float4 fragMain(VSOutput input) : SV_Target {
    return ssrPixel(input.texCoord, uint2(input.position.xy));
}

// Tiled path A: compact trace-worthy pixels into the ray list and zero the rest of the output.
// Wave-aggregated append — one atomic per wave instead of one per pixel; the indirect group
// count is maintained with an InterlockedMax so no finalize dispatch is needed.
[shader("compute")]
[NumThreads(8, 8, 1)]
void classifyMain(uint3 tid : SV_DispatchThreadID) {
    uint2 res = SSRpc.passData.ssrResolution;
    bool inBounds = all(tid.xy < res);

    bool trace = false;
    if (inBounds) {
        float2 uv = (float2(tid.xy) + 0.5) / float2(res);
        trace = ssrShouldTrace(uv);
        if (!trace)
            storageImages2D[SSRpc.passData.outputImageIndex][tid.xy] = float4(0.0);
    }

    uint idxInWave = WavePrefixCountBits(trace);
    uint waveTotal = WaveActiveCountBits(trace);
    uint waveBase = 0;
    if (WaveIsFirstLane() && waveTotal > 0) {
        InterlockedAdd(SSRpc.rayHeader.rayCount, waveTotal, waveBase);
        uint end = min(waveBase + waveTotal, SSRpc.passData.rayCapacity);
        InterlockedMax(SSRpc.rayHeader.groupsX, (end + SSR_TRACE_GROUP - 1) / SSR_TRACE_GROUP);
    }
    waveBase = WaveReadLaneFirst(waveBase);

    if (trace) {
        uint idx = waveBase + idxInWave;
        if (idx < SSRpc.passData.rayCapacity)
            SSRpc.rayPixels[idx] = tid.x | (tid.y << 16);
    }
}

// Tiled path B: one thread per compacted ray, dispatched indirect from the header.
[shader("compute")]
[NumThreads(SSR_TRACE_GROUP, 1, 1)]
void traceMain(uint3 tid : SV_DispatchThreadID) {
    uint rayCount = min(SSRpc.rayHeader.rayCount, SSRpc.passData.rayCapacity);
    if (tid.x >= rayCount)
        return;

    uint packed = SSRpc.rayPixels[tid.x];
    uint2 pixel = uint2(packed & 0xFFFFu, packed >> 16);
    float2 uv = (float2(pixel) + 0.5) / float2(SSRpc.passData.ssrResolution);

    storageImages2D[SSRpc.passData.outputImageIndex][pixel] = ssrPixel(uv, pixel);
}
