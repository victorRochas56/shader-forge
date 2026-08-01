// Voxel grid visualizer. Fullscreen ray-march of the RGBA8 volume the voxelization pass builds,
// alpha-blended over the HDR composite so it reads as an overlay on the real scene.
//
// The grid is oriented along the camera's look direction at voxelization time, so it is an OBB in
// world space, not an AABB — marching it in world space would need the full transform per step.
// Instead everything is done in grid space: the CPU folds voxelVPM * invCameraViewProjection into
// camNdcToGrid, so one matrix multiply takes a camera-NDC point straight into the grid's cube, and
// the march is a straight line through [0,1]^3.
//
// The far end of the ray is the scene surface (reconstructed from the depth buffer), which gives
// depth occlusion for free — voxels behind geometry don't draw over it.

import modules.voxel;

// Keep in sync with VoxelDebugMode in structs.hpp and the GUI combo.
static const uint VOXDBG_RADIANCE  = 0; // stored radiance, alpha-composited
static const uint VOXDBG_OCCUPANCY = 1; // coverage only, as grey
static const uint VOXDBG_ALBEDO    = 2; // radiance normalized to its own max — shows structure in dark voxels
static const uint VOXDBG_DEPTH     = 3; // distance to the first hit, as a heat ramp

struct VoxelDebugPushConstants {
    float4x4 camNdcToGrid;  // camera NDC (+ depth) -> voxel grid clip space
    float3   cameraPosGrid; // camera position in grid UVW
    uint     mipLevel;
    uint     volumeTexIndex;
    uint     samplerIndex;
    uint     depthTexIndex;
    uint     depthSamplerIndex;
    uint     resolution;    // mip-0 grid side
    uint     mode;
    uint     maxSteps;
    float    alphaScale;    // opacity multiplier — raise to make sparse grids readable
}

[[vk::push_constant]] ConstantBuffer<VoxelDebugPushConstants> vdpc;

[[vk::binding(0, 0)]] Texture2D textures[];
[[vk::binding(0, 0)]] Texture3D<float4> textures3D[];
[[vk::binding(2, 0)]] SamplerState samplers[];

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

float3 heat(float t) {
    t = saturate(t);
    return saturate(float3(1.5 - abs(4.0 * t - 3.0), 1.5 - abs(4.0 * t - 2.0), 1.5 - abs(4.0 * t - 1.0)));
}

// Grid clip -> UVW. xy is NDC so it needs the half-range remap; z is already [0,1] under orthoRH_ZO,
// matching how the raster pass derived its slice index from SV_Position.z.
float3 gridClipToUVW(float4 clip) {
    float3 ndc = clip.xyz / clip.w;
    return float3(ndc.xy * 0.5 + 0.5, ndc.z);
}

// Slab test against the unit cube. Returns false if the ray misses it entirely.
bool intersectUnitCube(float3 origin, float3 dir, out float tNear, out float tFar) {
    float3 invDir = 1.0 / (abs(dir) < 1e-8 ? float3(1e-8) * sign(dir + 1e-20) : dir);
    float3 t0 = (float3(0.0) - origin) * invDir;
    float3 t1 = (float3(1.0) - origin) * invDir;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    tNear = max(max(tMin.x, tMin.y), tMin.z);
    tFar  = min(min(tMax.x, tMax.y), tMax.z);
    return tFar > max(tNear, 0.0);
}

[shader("fragment")]
float4 fragMain(VSOutput input) : SV_Target {
    float2 ndc = input.texCoord * 2.0 - 1.0;
    float sceneDepth = textures[vdpc.depthTexIndex].SampleLevel(samplers[vdpc.depthSamplerIndex], input.texCoord, 0).r;

    // The scene surface, in grid space. Everything between the camera and it is what we march.
    float4 surfaceClip = mul(vdpc.camNdcToGrid, float4(ndc, sceneDepth, 1.0));
    if (abs(surfaceClip.w) < 1e-6) return float4(0, 0, 0, 0); // degenerate (surface on the eye plane)
    float3 surfaceGrid = gridClipToUVW(surfaceClip);

    float3 origin = vdpc.cameraPosGrid;
    float3 toSurface = surfaceGrid - origin;
    float rayLength = length(toSurface);
    if (rayLength < 1e-6) return float4(0, 0, 0, 0);
    float3 dir = toSurface / rayLength;

    float tNear, tFar;
    if (!intersectUnitCube(origin, dir, tNear, tFar)) return float4(0, 0, 0, 0);
    tNear = max(tNear, 0.0);
    tFar  = min(tFar, rayLength); // stop at the scene surface
    if (tFar <= tNear) return float4(0, 0, 0, 0);

    // One step per voxel at the selected mip, capped so a grazing ray can't run away.
    uint mipRes = max(1u, vdpc.resolution >> vdpc.mipLevel);
    float stepSize = 1.0 / float(mipRes);
    uint steps = min(vdpc.maxSteps, (uint)ceil((tFar - tNear) / stepSize) + 1u);

    SamplerState samp = samplers[vdpc.samplerIndex];
    float3 accum = float3(0);
    float  transmittance = 1.0;
    float  firstHitT = -1.0;
    float  peak = 0.0;

    for (uint i = 0; i < steps; i++) {
        float t = tNear + (float(i) + 0.5) * stepSize;
        if (t >= tFar) break;

        float3 uvw = origin + dir * t;
        float4 s = textures3D[vdpc.volumeTexIndex].SampleLevel(samp, uvw, float(vdpc.mipLevel));
        s.rgb = voxelToneMapInv(s.rgb); // back to linear HDR — the overlay blends into the HDR composite

        float alpha = saturate(s.a * vdpc.alphaScale);
        if (alpha <= 0.001) continue;
        if (firstHitT < 0.0) firstHitT = t;
        peak = max(peak, max(s.r, max(s.g, s.b)));

        float3 sample = (vdpc.mode == VOXDBG_OCCUPANCY) ? float3(s.a) : s.rgb;
        // Front-to-back "over".
        accum += transmittance * alpha * sample;
        transmittance *= (1.0 - alpha);
        if (transmittance < 0.003) break;
    }

    float coverage = 1.0 - transmittance;
    if (coverage <= 0.001) return float4(0, 0, 0, 0);

    if (vdpc.mode == VOXDBG_DEPTH) {
        // Normalized against the cube diagonal so the ramp spans the whole grid.
        float d = (firstHitT < 0.0) ? 1.0 : saturate(firstHitT / 1.7320508);
        return float4(heat(d), coverage);
    }
    if (vdpc.mode == VOXDBG_ALBEDO) {
        // Renormalize so dim voxels stay visible; structure matters more than absolute level here.
        return float4(accum / max(peak, 1e-4), coverage);
    }
    return float4(accum, coverage);
}
