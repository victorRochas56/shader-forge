// Shared encoding for the voxel scatter buffers (voxelization.slang -> voxel_resolve.slang).
//
// One uint per voxel per channel. The top byte is a luminance key, so InterlockedMax orders samples
// by brightness rather than by whichever bit pattern happens to be numerically larger — the result is
// the brightest contributor to that voxel, independent of draw order. Zero means empty; that makes a
// pure-black surface indistinguishable from no surface, which is the price of not carrying a separate
// occupancy buffer.
module voxel;

[[vk::binding(0, 0)]] Texture3D<float4> textures3D[];
[[vk::binding(2, 0)]] SamplerState samplers[];

// Radiance is HDR, so it goes through Reinhard on the way in and the inverse on the way out. Values
// well above ~10 compress hard — raise the knee if the scene has very bright emissives.
public float3 voxelToneMap(float3 v)    { return v / (1.0 + v); }
public float3 voxelToneMapInv(float3 v) { return v / max(1.0 - v, 1e-4); }

public uint packVoxel(float3 color) {
    float3 c = saturate(color);
    uint luma = (uint)(saturate(dot(c, float3(0.2126, 0.7152, 0.0722))) * 255.0 + 0.5);
    uint3 rgb = (uint3)(c * 255.0 + 0.5);
    // Keep the key non-zero for anything that isn't pure black, so occupancy survives dim surfaces.
    if (luma == 0 && any(rgb > 0)) luma = 1;
    return (luma << 24) | (rgb.r << 16) | (rgb.g << 8) | rgb.b;
}

public float3 unpackVoxel(uint packed) {
    return float3((packed >> 16) & 0xFF, (packed >> 8) & 0xFF, packed & 0xFF) * (1.0 / 255.0);
}

public bool voxelOccupied(uint packed) { return packed != 0; }

public uint voxelIndex(uint3 coord, uint resolution) {
    return coord.x + resolution * (coord.y + resolution * coord.z);
}

// World -> volume UVW via the grid's ortho VPM. xy is NDC so it needs the half-range remap; z is
// already [0,1] under orthoRH_ZO. Same mapping as gridClipToUVW in voxel_debug.slang.
public float3 worldToGridUVW(float3 worldPos, float4x4 worldToGridClip) {
    float4 clip = mul(worldToGridClip, float4(worldPos, 1.0));
    float3 ndc = clip.xyz / clip.w;
    return float3(ndc.xy * 0.5 + 0.5, ndc.z);
}

#define HEMISPHERE_RAYS 5
#define MAX_STEPS 16

// Crassin's cone fan: one along the normal plus HEMISPHERE_RAYS at 60°, ~60° aperture each.
// Tangent-space, z = normal. Weights are cosine-lobe-ish and sum to 1.
static const float3 CONE_DIRS[HEMISPHERE_RAYS + 1] = {
    float3( 0.0,       0.0,      1.0),
    float3( 0.866025,  0.0,      0.5),
    float3( 0.267617,  0.823639, 0.5),
    float3(-0.700629,  0.509037, 0.5),
    float3(-0.700629, -0.509037, 0.5),
    float3( 0.267617, -0.823639, 0.5),
};
static const float CONE_WEIGHTS[HEMISPHERE_RAYS + 1] = { 0.25, 0.15, 0.15, 0.15, 0.15, 0.15 };
static const float CONE_TAN = 0.577; // tan(30°) half-angle

// Front-to-back march along one cone. The volume holds (radiance, coverage), colour not
// premultiplied, so each tap composites as a * rgb. Mip follows the cone diameter — that growing
// trilinear footprint is what stands in for integrating over the cone's solid angle.
float3 traceCone(Texture3D<float4> volume, SamplerState smp, float4x4 worldToGridClip,
                 float3 origin, float3 dir, float voxelSize, float maxDist, float maxMip) {
    float4 acc = float4(0.0);
    float dist = voxelSize;
    for (int i = 0; i < MAX_STEPS && dist < maxDist && acc.a < 0.99; i++) {
        float diameter = max(voxelSize, 2.0 * CONE_TAN * dist);
        float mip = clamp(log2(diameter / voxelSize), 0.0, maxMip);
        float3 uvw = worldToGridUVW(origin + dir * dist, worldToGridClip);
        if (any(uvw < 0.0) || any(uvw > 1.0)) break; // off the grid; nothing but border black past here
        float4 s = volume.SampleLevel(smp, uvw, mip);
        acc.rgb += (1.0 - acc.a) * s.a * s.rgb;
        acc.a   += (1.0 - acc.a) * s.a;
        dist += diameter * 0.5; // half-diameter steps against banding
    }
    return acc.rgb;
}

public float3 coneTraceVXGI(uint voxel3DTextureIndex, uint voxelSamplerIndex, float4x4 worldToGridClip,
                            uint resolution, float worldExtent, float3 worldPos, float3 N) {
    Texture3D<float4> volume = textures3D[voxel3DTextureIndex];
    SamplerState smp = samplers[voxelSamplerIndex];
    float voxelSize = worldExtent / resolution;
    float maxMip = log2(float(resolution));

    N = normalize(N);
    float3 up = abs(N.y) < 0.99 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    float3 T = normalize(cross(up, N));
    float3 B = cross(N, T);

    // Start clear of the surface's own voxel — sampling it reads back direct lighting and flickers.
    float3 origin = worldPos + N * (2.0 * voxelSize);

    float3 gi = float3(0.0);
    [unroll]
    for (uint i = 0; i < HEMISPHERE_RAYS + 1; i++) {
        float3 dir = normalize(T * CONE_DIRS[i].x + B * CONE_DIRS[i].y + N * CONE_DIRS[i].z);
        gi += CONE_WEIGHTS[i] * traceCone(volume, smp, worldToGridClip, origin, dir, voxelSize, worldExtent * 0.5, maxMip);
    }
    return gi;
}
