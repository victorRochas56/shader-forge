import modules.common;
import modules.pbr;

struct ParticleDrawPushConstants {
    float4x4  viewProjection;
    Particle*   particlesBDA;       // pool base address
    uint  particleOffset;     // this emitter's base index into the pool
    uint  particleCount;      // instances to draw (emitter ring capacity)
    uint  textureIndex;
    uint  numFrames;          // atlas frames, 0 = static
    uint  samplerIndex;
    uint  depthTextureIndex;
    uint  depthSamplerIndex;
    uint  flags;              // EMITTER_FLAG_*
    uint2 resolution;
    float softRadius;         // EMITTER_FLAG_SOFT depth-fade distance (view-space units)
    float nearPlane;          // for linearizing the sampled scene depth
    float farPlane;
};

// Mirror of the EMITTER_FLAG_* bits in structs.hpp.
static const uint EMITTER_FLAG_SOFT = 1u << 3;

// Shared descriptor bindings
[[vk::push_constant]]
ConstantBuffer<ParticleDrawPushConstants> particlePc;

struct PSInput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
    nointerpolation uint instanceID : TEXCOORD1;
    float  coverage : TEXCOORD2;   // sub-pixel alpha fade (1.0 = full size)
    float  viewDepth : TEXCOORD3;  // clip.w of the center = linear view-space depth (soft particles)
};

[shader("vertex")]
PSInput vertMain(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    static const float2 CORNERS[6] = {
        float2(-0.5, -0.5), float2( 0.5, -0.5), float2( 0.5,  0.5),
        float2(-0.5, -0.5), float2( 0.5,  0.5), float2(-0.5,  0.5),
    };
    static const float2 UVS[6] = {
        float2(0, 0), float2(1, 0), float2(1, 1),
        float2(0, 0), float2(1, 1), float2(0, 1),
    };

    Particle p = particlePc.particlesBDA[particlePc.particleOffset + instanceID];

    PSInput o = {};
    // Dead slot: collapse all 3 verts of the triangle to one clip point -> zero area -> nothing
    // rasterized. Every vertex of this instance reads the same particle, so they coincide.
    if (p.age < 0.0) {
        o.position = float4(0, 0, 0, 1);
        return o;
    }

    // Project the center first; clip.w drives both the billboard scale and the pixel-size test.
    float4 clip = mul(particlePc.viewProjection, float4(p.position, 1.0));

    // Rotate the corner in the billboard plane, then scale by the particle's size.
    float2 corner = CORNERS[vertexID] * p.size;
    float s = sin(p.rotation);
    float c = cos(p.rotation);
    corner = float2(corner.x * c - corner.y * s, corner.x * s + corner.y * c);
    // Aspect-correct x so the quad stays square on screen (offset is applied in clip space).
    corner.x *= float(particlePc.resolution.y) / float(particlePc.resolution.x);

    // Sub-pixel handling: the quad's on-screen half-extent in pixels. Offset is in clip space with
    // no *w, so NDC half-extent = (0.5*size)/clip.w; *0.5*resolution.y converts NDC->pixels (y axis
    // has no aspect skew). Below ~1px a particle would shimmer, so instead of shrinking it further we
    // clamp the quad up to the floor and pay back the extra area in alpha (coverage) -> smooth fade
    // out with distance, no popping, roughly energy-preserving for additive/soft particles.
    float halfPixels = (0.5 * p.size / clip.w) * 0.5 * float(particlePc.resolution.y);
    const float MIN_PIXELS = 1.0;
    float coverage = 1.0;
    if (halfPixels < MIN_PIXELS) {
        float grow = MIN_PIXELS / max(halfPixels, 1e-4);
        corner   *= grow;
        coverage  = 1.0 / (grow * grow);
    }

    // Offset in clip space so the quad always faces the camera (shrinks with distance).
    clip.xy += corner;

    o.position   = clip;
    o.texCoord   = UVS[vertexID];
    o.instanceID = instanceID;   // re-fetch the particle in fragMain (atlas frame, age fade)
    o.coverage   = coverage;
    o.viewDepth  = clip.w;       // center's view-space depth (clip.w unaffected by the xy offset)
    return o;
}

[shader("fragment")]
float4 fragMain(PSInput input) : SV_Target
{
    Particle p = particlePc.particlesBDA[particlePc.particleOffset + input.instanceID];

    float2 uv = input.texCoord;
    // Animated atlas: advance the frame across the particle's lifetime. Assumes a horizontal strip
    // of numFrames cells (change the remap if the atlas is a grid).
    if (particlePc.numFrames > 0) {
        float ageFrac = p.age / max(p.lifespan, 1e-4);
        uint  frame   = min(uint(ageFrac * float(particlePc.numFrames)), particlePc.numFrames - 1);
        uv.x = (uv.x + float(frame)) / float(particlePc.numFrames);
    }

    float4 outCol = textures[particlePc.textureIndex].Sample(samplers[particlePc.samplerIndex], uv);

    // Depth test (always on): sample the resolved scene depth at this pixel. Reverse-Z means a
    // larger stored value is nearer, so a particle whose fragment depth is smaller sits behind the
    // scene surface -> fully occluded, discard.
    uint depthW, depthH;
    textures[particlePc.depthTextureIndex].GetDimensions(depthW, depthH);
    float2 screenUV = input.position.xy / float2(depthW, depthH);
    float sceneDepth = textures[particlePc.depthTextureIndex].Sample(samplers[particlePc.depthSamplerIndex], screenUV).r;
    if (input.position.z < sceneDepth) discard;

    // Fade alpha out over the last 20% of life so particles don't vanish abruptly.
    float ageFrac = p.age / max(p.lifespan, 1e-4);
    float ageFade = 1.0 - smoothstep(0.8, 1.0, ageFrac);

    // Soft particles: fade as the billboard approaches the scene surface in front of it. Linearize
    // the sampled reverse-Z depth to view-space distance and compare with the particle's view depth.
    float softFade = 1.0;
    if ((particlePc.flags & EMITTER_FLAG_SOFT) != 0u && particlePc.softRadius > 0.0) {
        float n = particlePc.nearPlane, f = particlePc.farPlane;
        float sceneLinear = n * f / (n + sceneDepth * (f - n));
        softFade = saturate((sceneLinear - input.viewDepth) / particlePc.softRadius);
    }

    outCol.a *= input.coverage * ageFade * softFade;
    return outCol;
}