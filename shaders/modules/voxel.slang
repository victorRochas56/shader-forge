// Shared encoding for the voxel scatter buffers (voxelization.slang -> voxel_resolve.slang).
//
// One uint per voxel per channel. The top byte is a luminance key, so InterlockedMax orders samples
// by brightness rather than by whichever bit pattern happens to be numerically larger — the result is
// the brightest contributor to that voxel, independent of draw order. Zero means empty; that makes a
// pure-black surface indistinguishable from no surface, which is the price of not carrying a separate
// occupancy buffer.
module voxel;

[[vk::binding(0, 0)]] Texture3D<float4> textures3D[];
// Alias of common.slang's cubemaps — same binding, module-local name to avoid a collision in
// shaders that import both modules.
[[vk::binding(1, 0)]] TextureCube voxelCubemaps[];
[[vk::binding(2, 0)]] SamplerState samplers[];

// Radiance is HDR, so it goes through Reinhard on the way in and the inverse on the way out. The
// volume is RGBA8 unorm, so it stores the compressed domain too — readers decode per tap. Values
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

// Ambient-cube face order: +X, -X, +Y, -Y, +Z, -Z. Shared by the gather pass (trace axes) and the
// lit-shader lookup (face picked by normal sign).
public static const float3 VOXEL_FACE_DIRS[6] = {
    float3( 1, 0, 0), float3(-1, 0, 0),
    float3( 0, 1, 0), float3( 0,-1, 0),
    float3( 0, 0, 1), float3( 0, 0,-1),
};

public float voxelLoadCoverage(uint textureIndex, uint3 coord, uint mip) {
    return textures3D[textureIndex].Load(int4(coord, mip)).a;
}

// Voxel center in world space. Grid clip is orthographic, so w == 1 and the inverse is exact.
public float3 voxelCenterWorld(uint3 coord, uint resolution, float4x4 gridClipToWorld) {
    float3 uvw = (float3(coord) + 0.5) / float(resolution);
    return mul(gridClipToWorld, float4(uvw.xy * 2.0 - 1.0, uvw.z, 1.0)).xyz;
}

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

// Compile-time caps for the runtime-tunable trace params (LitPass gi* fields).
#define MAX_HEMISPHERE_RAYS 5
#define MAX_FETCH_BATCH 8

// Crassin's cone fan: one along the normal plus up to MAX_HEMISPHERE_RAYS at 60°, ~60° aperture each.
// Tangent-space, z = normal. Weights are cosine-lobe-ish; renormalized at trace time by however many
// cones actually run.
static const float3 CONE_DIRS[MAX_HEMISPHERE_RAYS + 1] = {
    float3( 0.0,       0.0,      1.0),
    float3( 0.866025,  0.0,      0.5),
    float3( 0.267617,  0.823639, 0.5),
    float3(-0.700629,  0.509037, 0.5),
    float3(-0.700629, -0.509037, 0.5),
    float3( 0.267617, -0.823639, 0.5),
};
static const float CONE_WEIGHTS[MAX_HEMISPHERE_RAYS + 1] = { 0.25, 0.15, 0.15, 0.15, 0.15, 0.15 };
static const float CONE_TAN = 0.577; // tan(30°) half-angle

// Front-to-back march along one cone. The volume holds (Reinhard-compressed radiance, coverage),
// colour not premultiplied, so each tap decodes then composites as a * rgb. Mip follows the cone
// diameter — that growing trilinear footprint is what stands in for integrating over the cone's
// solid angle.
//
// Fetches go out in batches of fetchBatch: tap positions depend only on dist, never on sampled
// data, so batching keeps several texture requests in flight instead of one serialized round-trip
// per step (long-scoreboard stalls). A batch may overshoot the opacity break or the grid by up to
// fetchBatch-1 taps; the sampler's transparent-black border makes those composite as no-ops.
// Runtime bounds mean no [unroll]; the fetch loop still has no data dependency, so requests overlap.
float3 traceCone(Texture3D<float4> volume, SamplerState smp, float4x4 worldToGridClip,
                 float3 origin, float3 dir, float voxelSize, float maxDist, float maxMip,
                 uint maxSteps, uint fetchBatch, uint envMapIndex, float skyIntensity) {
    // Grid clip is orthographic (w == 1), so uvw is linear in dist — one affine ray, no per-step matrix mul.
    float3 uvwOrigin = worldToGridUVW(origin, worldToGridClip);
    float4 clipDir = mul(worldToGridClip, float4(dir, 0.0));
    float3 uvwDir = float3(clipDir.xy * 0.5, clipDir.z);

    // Clamp the march at the grid exit (slab test) instead of bounds-checking every step.
    float3 tFar = (step(0.0, uvwDir) - uvwOrigin) / uvwDir;
    maxDist = min(maxDist, min(min(tFar.x, tFar.y), tFar.z));

    float4 acc = float4(0.0);
    float dist = voxelSize;
    for (uint i = 0; i < maxSteps / fetchBatch && dist < maxDist && acc.a < 0.99; i++) {
        float4 s[MAX_FETCH_BATCH];
        float d = dist;
        for (uint j = 0; j < fetchBatch; j++) {
            float diameter = max(voxelSize, 2.0 * CONE_TAN * d);
            float mip = clamp(log2(diameter / voxelSize), 0.0, maxMip);
            s[j] = volume.SampleLevel(smp, uvwOrigin + uvwDir * d, mip);
            d += diameter; // half-diameter steps against banding
        }
        for (uint j = 0; j < fetchBatch; j++) {
            acc.rgb += (1.0 - acc.a) * s[j].a * voxelToneMapInv(s[j].rgb);
            acc.a   += (1.0 - acc.a) * s[j].a;
        }
        dist = d;
    }
    // Whatever transmittance survives the march sees sky: sample the env map along the cone at a
    // coarse mip standing in for the ~60° aperture. Occluded cones (acc.a ~ 1) get nothing, so this
    // is soft sky visibility for free. Assumes anything past maxDist is open sky.
    if (skyIntensity > 0.0) {
        uint w, h, mips;
        voxelCubemaps[envMapIndex].GetDimensions(0, w, h, mips);
        acc.rgb += (1.0 - acc.a) * skyIntensity * voxelCubemaps[envMapIndex].SampleLevel(smp, dir, float(mips) - 2.0).rgb;
    }
    return acc.rgb;
}

public float3 coneTraceVXGI(uint voxel3DTextureIndex, uint voxelSamplerIndex, float4x4 worldToGridClip,
                            uint resolution, float worldExtent, float3 worldPos, float3 N,
                            uint hemisphereRays, uint maxSteps, uint fetchBatch,
                            uint envMapIndex, float skyIntensity) {
    Texture3D<float4> volume = textures3D[voxel3DTextureIndex];
    SamplerState smp = samplers[voxelSamplerIndex];
    float voxelSize = worldExtent / resolution;
    float maxMip = log2(float(resolution));
    uint rays  = min(hemisphereRays, uint(MAX_HEMISPHERE_RAYS));
    uint batch = min(max(fetchBatch, 1u), uint(MAX_FETCH_BATCH));

    N = normalize(N);
    float3 up = abs(N.y) < 0.99 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    float3 T = normalize(cross(up, N));
    float3 B = cross(N, T);

    // Start clear of the surface's own voxel — sampling it reads back direct lighting and flickers.
    float3 origin = worldPos + N * (2.0 * voxelSize);

    // Renormalize by the weights actually used so fewer cones doesn't dim the GI.
    float3 gi = float3(0.0);
    float totalWeight = 0.0;
    for (uint i = 0; i < rays + 1; i++) {
        float3 dir = normalize(T * CONE_DIRS[i].x + B * CONE_DIRS[i].y + N * CONE_DIRS[i].z);
        gi += CONE_WEIGHTS[i] * traceCone(volume, smp, worldToGridClip, origin, dir, voxelSize, worldExtent * 0.5, maxMip, maxSteps, batch, envMapIndex, skyIntensity);
        totalWeight += CONE_WEIGHTS[i];
    }
    return gi / totalWeight;
}

// Irradiance faces store (Reinhard(irradiance) * valid, valid) — premultiplied so hardware trilinear
// against empty black neighbours renormalizes on the divide instead of dimming. The 0.25 floor makes
// mostly-empty neighbourhoods fade out rather than amplify unorm8 quantization noise.
float3 decodeIrradiance(float4 s) {
    return voxelToneMapInv(s.rgb / max(s.a, 0.25));
}

// Ambient-cube GI lookup: three trilinear taps from the sign-matching face volumes, blended by N².
// Sampled at the surface position, no normal offset: an offset along an axis-aligned N lands a full
// voxel away and zeroes the valid slab's trilinear weight (black walls/floors). The surface's own
// voxel is the right source — face selection by normal sign already separates the two sides of thin
// geometry sharing a voxel.
public float3 sampleAmbientCubeGI(uint faceIndices[6], uint samplerIndex, float4x4 worldToGridClip,
                                  uint resolution, float worldExtent, float3 worldPos, float3 N) {
    SamplerState smp = samplers[samplerIndex];
    N = normalize(N);
    float3 uvw = worldToGridUVW(worldPos, worldToGridClip);
    float3 w = N * N; // face weights, sum to 1
    uint fx = N.x < 0.0 ? 1 : 0;
    uint fy = N.y < 0.0 ? 3 : 2;
    uint fz = N.z < 0.0 ? 5 : 4;
    return w.x * decodeIrradiance(textures3D[faceIndices[fx]].SampleLevel(smp, uvw, 0))
         + w.y * decodeIrradiance(textures3D[faceIndices[fy]].SampleLevel(smp, uvw, 0))
         + w.z * decodeIrradiance(textures3D[faceIndices[fz]].SampleLevel(smp, uvw, 0));
}
