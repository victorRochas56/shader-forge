// Froxel volumetrics — Pass E: composite (+ debug overlay).
// Per pixel: linearize scene depth -> view Z -> fractional slice, trilinearly sample integratedVol,
// and output (accumInScatter.rgb, transmittance). The POSTPROCESS_VOLUMETRIC blend then applies
//   out.rgb = inScatter + sceneColor * transmittance.
// When debugMode != 0, the shader instead returns (debugColor, 0) so the blend overwrites the scene
// with a visualization of a grid volume.
import modules.froxel;   // viewZToSlice
import modules.rng;      // pcgHash, rand

// Keep in sync with the GUI combo in gui.cpp.
static const uint DBG_OFF          = 0;
static const uint DBG_EXTINCTION   = 1;  // mediaVol.a, max along the ray
static const uint DBG_SCATTER      = 2;  // scatterVol.rgb, max luminance along the ray (shows shadows)
static const uint DBG_INSCATTER    = 3;  // integratedVol.rgb at the surface slice
static const uint DBG_TRANSMITTANCE = 4; // integratedVol.a at the surface slice
static const uint DBG_SLICE        = 5;  // surface slice index as a heatmap
static const uint DBG_PHASE        = 6;  // per-froxel HG g at the surface slice
static const uint DBG_SHADOW       = 7;  // raw (1 - shadow) written by pass C, at the surface slice

struct VolumetricApplyPushConstants {
    uint integratedTexIndex;
    uint samplerIndex;
    uint depthTexIndex;
    uint depthSamplerIndex;
    uint3 dims;
    float nearZ;
    float3 cameraPos;
    float gridFar;
    float3 cameraForward;
    uint debugMode;
    float4x4 invViewProjection;
    uint mediaTexIndex;
    uint scatterTexIndex;
    uint mediaPhaseTexIndex;
    uint frame;
}

[[vk::push_constant]] VolumetricApplyPushConstants pc;

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

// Simple blue->green->red ramp for scalar debug views.
float3 heat(float t) {
    t = saturate(t);
    return saturate(float3(1.5 - abs(4.0 * t - 3.0), 1.5 - abs(4.0 * t - 2.0), 1.5 - abs(4.0 * t - 1.0)));
}

[shader("fragment")]
float4 fragMain(VSOutput input) : SV_Target {
    // Reconstruct the view-space depth of the scene surface at this pixel (reverse-Z).
    float sceneDepth = textures[pc.depthTexIndex].SampleLevel(samplers[pc.depthSamplerIndex], input.texCoord, 0).r;
    float2 ndc = input.texCoord * 2.0 - 1.0;
    float4 worldH = mul(pc.invViewProjection, float4(ndc, sceneDepth, 1.0));
    float3 world = worldH.xyz / worldH.w;
    float viewZ = dot(world - pc.cameraPos, pc.cameraForward);

    // Pixels past gridFar clamp to the last (fully integrated) slice; before near, the first.
    viewZ = clamp(viewZ, pc.nearZ, pc.gridFar);
    float slice = viewZToSlice(viewZ, pc.dims.z, pc.nearZ, pc.gridFar);
    float w = (slice + 0.5) / float(pc.dims.z); // fractional slice -> normalized texel-centered depth
    // integratedVol[z] holds the accumulation at slab z's FAR edge (the integrate pass adds the whole
    // slab before writing), so reading it for a surface at `slice` must shift one texel NEARER — else
    // it returns a full slab of fog from behind the surface (fog peeking through thin geometry).
    // (mediaVol/scatterVol are slab-centered, so those still sample at w below.)
    float wInt = (slice - 0.5) / float(pc.dims.z);

    if (pc.debugMode != DBG_OFF) {
        SamplerState samp = samplers[pc.samplerIndex];
        float3 dbg = float3(0);
        if (pc.debugMode == DBG_INSCATTER) {
            dbg = textures3D[pc.integratedTexIndex].SampleLevel(samp, float3(input.texCoord, wInt), 0).rgb;
        } else if (pc.debugMode == DBG_TRANSMITTANCE) {
            dbg = textures3D[pc.integratedTexIndex].SampleLevel(samp, float3(input.texCoord, wInt), 0).aaa;
        } else if (pc.debugMode == DBG_SLICE) {
            dbg = heat(w);
        } else if (pc.debugMode == DBG_PHASE) {
            float4 m = textures3D[pc.mediaTexIndex].SampleLevel(samp, float3(input.texCoord, w), 0);
            float ph = textures3D[pc.mediaPhaseTexIndex].SampleLevel(samp, float3(input.texCoord, w), 0).r;
            float g = ph / max(m.a, 1e-5);
            dbg = heat(g * 0.5 + 0.5); // -1..1 -> 0..1
        } else {
            // Column modes (EXTINCTION / SCATTER / SHADOW): walk z up to the surface slice and take the
            // strongest sample along the ray. SHADOW reads scatterVol too (pass C wrote 1-shadow there),
            // so white = at least one lit froxel along this ray (a sunbeam), black = whole ray shadowed.
            uint zEnd = uint(clamp(slice, 0.0, float(pc.dims.z)));
            float3 acc = float3(0);
            for (uint z = 0; z < zEnd; z++) {
                float wz = (float(z) + 0.5) / float(pc.dims.z);
                if (pc.debugMode == DBG_EXTINCTION) {
                    float e = textures3D[pc.mediaTexIndex].SampleLevel(samp, float3(input.texCoord, wz), 0).a;
                    acc = max(acc, float3(e));
                } else { // DBG_SCATTER or DBG_SHADOW (both read scatterVol.rgb)
                    float3 s = textures3D[pc.scatterTexIndex].SampleLevel(samp, float3(input.texCoord, wz), 0).rgb;
                    acc = max(acc, s);
                }
            }
            dbg = (pc.debugMode == DBG_EXTINCTION) ? heat(acc.r) : acc;
        }
        return float4(dbg, 0.0); // a=0 -> overwrites the scene (blend: src + dst*a)
    }

    // Per-pixel depth dither. The froxel passes jitter too (froxelZJitter, one-sided toward camera),
    // but that's seeded per column, so it can't go finer than one froxel (8x8 px at the default dims)
    // — this breaks up the z-slice banding at pixel granularity. +-0.5 slice, in normalized w units.
    uint seed = pcgHash(uint(input.position.x) | (uint(input.position.y) << 16));
    float jitter = (rand(seed) - 0.5) / float(pc.dims.z);
    // defaultSampler is eRepeat, so clamp to the first/last texel centre: wInt+jitter straying outside
    // [0,1] would wrap to the opposite end of the grid. Cap at the surface texel (upper) too: positive
    // jitter must not sample integratedVol past the surface, bleeding occluded froxels through geometry.
    float halfTexel = 0.5 / float(pc.dims.z);
    float upper = clamp(wInt, halfTexel, 1.0 - halfTexel);   // valid, never inverts vs the lower bound
    float wj = clamp(wInt + jitter, halfTexel, upper);
    float4 v = textures3D[pc.integratedTexIndex].SampleLevel(samplers[pc.samplerIndex], float3(input.texCoord, wj), 0);
    // v = (accumulated in-scatter, transmittance). rgb added, a scales the scene (see blend state).
    return float4(v.rgb, v.a);
}
