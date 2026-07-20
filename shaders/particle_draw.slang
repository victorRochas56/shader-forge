import modules.common;
import modules.pbr;
import modules.particles;

struct ParticleDrawPushConstants {
    float4x4  viewProjection; 

    Particle*   particlesBDA;
    ParticleEmitter* emittersBDA;
    
    Light*      lightsBDA;
    uint  lightCount;
    uint  samplerIndex;       

    uint  depthTextureIndex;  
    uint  depthSamplerIndex;  
    float2 invResolution;         

    float3 cameraPos;         // world-space, for lit particles
    float farPlane;
    float3 cameraForward;     // normalized look dir, for cascade selection
    float nearPlane;
    
    uint  shadowAtlasIndex;
    uint  emitterIndex;
    float sphereRoundness;    // lit spherical-impostor bulge: 0 flat .. 1 sphere (>1 exaggerates)
    float opacity;            // global transparency multiplier for the whole emitter
};

// Mirror of the EMITTER_FLAG_* bits in structs.hpp.
static const uint EMITTER_FLAG_SOFT = 1u << 3;
static const uint EMITTER_FLAG_VOLUMETRIC  = 1u << 2;
static const uint EMITTER_FLAG_LIT  = 1u << 1;

// Shared descriptor bindings
[[vk::push_constant]]
ConstantBuffer<ParticleDrawPushConstants> particlePc;

struct PSInput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
    nointerpolation uint instanceID : TEXCOORD1;
    float  coverage : TEXCOORD2;   // sub-pixel alpha fade (1.0 = full size)
    float  viewDepth : TEXCOORD3;  // clip.w of the center = linear view-space depth (soft particles)
    float3 worldPos : TEXCOORD4;   // per-corner world-space position (lighting, shadows)
};

[shader("vertex")]
PSInput vertMain(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    ParticleEmitter* emitter = &particlePc.emittersBDA[particlePc.emitterIndex];

    static const float2 CORNERS[6] = {
        float2(-0.5, -0.5), float2( 0.5, -0.5), float2( 0.5,  0.5),
        float2(-0.5, -0.5), float2( 0.5,  0.5), float2(-0.5,  0.5),
    };
    static const float2 UVS[6] = {
        float2(0, 0), float2(1, 0), float2(1, 1),
        float2(0, 0), float2(1, 1), float2(0, 1),
    };

    Particle p = particlePc.particlesBDA[emitter.particleOffset + instanceID];

    PSInput o = {};
    // Dead slot: collapse all 3 verts of the triangle to one clip point -> zero area -> nothing
    // rasterized. Every vertex of this instance reads the same particle, so they coincide.
    if (p.age < 0.0) {
        o.position = float4(0, 0, 0, 1);
        return o;
    }

    // Camera-facing (spherical) billboard basis, derived from the camera's forward direction.
    float3 fwd     = particlePc.cameraForward;
    float3 worldUp = abs(fwd.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 right   = normalize(cross(worldUp, fwd));
    float3 up      = cross(fwd, right);

    // Project the center: gives the view depth and the NDC anchor for the pixel-size test.
    float4 centerClip = mul(particlePc.viewProjection, float4(p.position, 1.0));
    float2 ndcCenter  = centerClip.xy / centerClip.w;

    // Rotate the corner in the billboard plane, then scale by the particle's size.
    float2 corner = CORNERS[vertexID] * p.size;
    float s = sin(p.rotation);
    float c = cos(p.rotation);
    corner = float2(corner.x * c - corner.y * s, corner.x * s + corner.y * c);

    // Sub-pixel handling: measure the quad's real on-screen half-extent by projecting a reference
    // offset (0.5*size along the billboard up axis) and taking the NDC->pixel delta -> exact under
    // any projection. Below ~1px a particle would shimmer, so instead of shrinking it further we
    // clamp the quad up to the floor and pay back the extra area in alpha (coverage) -> smooth fade
    // out with distance, no popping, roughly energy-preserving for additive/soft particles.
    float4 refClip    = mul(particlePc.viewProjection, float4(p.position + up * (0.5 * p.size), 1.0));
    float2 refPix     = (refClip.xy / refClip.w - ndcCenter) * 0.5 * 1.0f/particlePc.invResolution;
    float  halfPixels = length(refPix);
    const float MIN_PIXELS = 1.0;
    float coverage = 1.0;
    if (halfPixels < MIN_PIXELS) {
        float grow = MIN_PIXELS / max(halfPixels, 1e-4);
        corner   *= grow;
        coverage  = 1.0 / (grow * grow);
    }

    // Build the corner in world space so the fragment gets a true per-corner world position.
    float3 worldPos = p.position + right * corner.x + up * corner.y;
    float4 clip     = mul(particlePc.viewProjection, float4(worldPos, 1.0));

    o.position   = clip;
    o.worldPos   = worldPos;
    o.texCoord   = UVS[vertexID];
    o.instanceID = instanceID;   // re-fetch the particle in fragMain (atlas frame, age fade)
    o.coverage   = coverage;
    o.viewDepth  = centerClip.w; // center's view-space depth (independent of the corner offset)
    return o;
}

[shader("fragment")]
float4 fragMain(PSInput input) : SV_Target
{
    // fully transparent shortcut (can be the case for volumetric only particles)
    if(particlePc.opacity == 0) discard;

    // Depth test (always on): sample the resolved scene depth at this pixel. Reverse-Z means a
    // larger stored value is nearer, so a particle whose fragment depth is smaller sits behind the
    // scene surface -> fully occluded, discard.
    float2 screenUV = input.position.xy * particlePc.invResolution;
    float sceneDepth = textures[particlePc.depthTextureIndex].Sample(samplers[particlePc.depthSamplerIndex], screenUV).r;


    ParticleEmitter* emitter = &particlePc.emittersBDA[particlePc.emitterIndex];
    Particle p = particlePc.particlesBDA[emitter.particleOffset + input.instanceID];

    float2 uv = input.texCoord;
    // Animated atlas: advance the frame across the particle's lifetime. Assumes a horizontal strip
    // of numFrames cells (change the remap if the atlas is a grid).
    if (emitter.numFrames > 0) {
        float ageFrac = p.age / max(p.lifespan, 1e-4);
        uint  frame   = min(uint(ageFrac * float(emitter.numFrames)), emitter.numFrames - 1);
        uv.x = (uv.x + float(frame)) / float(emitter.numFrames);
    }

    // discard here to avoid stalling scoreboard for on screen particles, most should be on screen anyways if reach here (otherwise clipped or cpu culled)
    if (input.position.z < sceneDepth) discard;

    float4 outCol = float4(2,2,2,0.2);;
    if(emitter.textureIndex != 0xFFFFFFFF) {
        outCol = textures[emitter.textureIndex].Sample(samplers[particlePc.samplerIndex], uv);
    }

    // --- Lit particles ------------------------------------------------------
    // A billboard has no real surface, so we light the whole quad at the
    // particle center (p.position) and fake a hemispherical normal that bulges
    // toward the camera from the quad UV -> the sprite reads like a lit sphere.
    outCol *= lerp(emitter.emissiveRange[0], emitter.emissiveRange[1], outCol.toFloat());

    if((emitter.flags & EMITTER_FLAG_LIT) != 0u) {
        float3 worldPos = input.worldPos;  // per-corner world position
        float3 V        = normalize(particlePc.cameraPos - worldPos);
        const float roughness = 0.8;       // particles aren't PBR surfaces; soft response
        const float metallic  = 0.0;
        // Spherical-impostor normal: treat the sprite as the front hemisphere of a sphere. Build the
        // billboard tangent frame from the SAME basis the vertex shader used (cameraForward, not V --
        // V points toward the camera so cross(up0,V) yields -right and mirrors the shading). Rotate
        // the centered UV by p.rotation so the hemisphere lines up with the drawn, rotated quad;
        // without this each particle is mis-oriented by its own rotation -> dark side points a random
        // way per particle.
        float3 fwd   = particlePc.cameraForward;
        float3 up0   = abs(fwd.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
        float3 right = normalize(cross(up0, fwd));
        float3 bUp   = cross(fwd, right);
        float  sr    = sin(p.rotation);
        float  cr    = cos(p.rotation);
        float2 d0    = input.texCoord * 2.0 - 1.0;                       // centered UV, [-1,1]
        float2 d     = float2(d0.x * cr - d0.y * sr, d0.x * sr + d0.y * cr); // -> rotated quad frame
        float  z     = sqrt(saturate(1.0 - dot(d0, d0)));               // 1 at center, 0 at the rim
        // Roundness blends the flat billboard normal (-fwd) toward the full spherical bulge:
        // 0 = flat card (uniform facing normal, soft flat lighting), 1 = full sphere, >1 exaggerates.
        float3 sphereN = normalize(d.x * right + d.y * bUp - z * fwd);  // bulges toward camera (-fwd)
        float3 N       = normalize(lerp(-fwd, sphereN, particlePc.sphereRoundness));

        // Volumetric self-shadow setup: extinction from the emitter's density range (same age fade as
        // the froxel sphere injection), sphere radius = billboard half-extent (matches billboardScale=1).
        bool  volumetric = (emitter.flags & EMITTER_FLAG_VOLUMETRIC) != 0u;
        float sphereR    = 0.5 * p.size;
        float volDensity = 0.0;
        if (volumetric) {
            float af      = p.age / max(p.lifespan, 1e-4);
            float ageFade = 1.0 - smoothstep(0.8, 1.0, af);
            volDensity    = lerp(emitter.densityRange[0], emitter.densityRange[1], ageFade);
        }

        float3 albedo   = outCol.rgb;      // particle texture is the base color

        float3 Lo = float3(0.0);
        for(int i = 0; i < particlePc.lightCount; i++) {
            Light* currentLight = &particlePc.lightsBDA[i];

            // Sparse slot skip: removed lights leave a zeroed sentinel in the buffer.
            if(currentLight.intensity <= 0.0) continue;

            if(currentLight.type != 1) {
                float dist = length(currentLight.position - worldPos);
                if(dist > currentLight.range) continue;
            }

            float3 L;
            float3 radiance;
            if(currentLight.type == 1) {
                L = -normalize(currentLight.direction);
                radiance = currentLight.color.rgb * currentLight.intensity;
            }
            else {
                float3 toLight = currentLight.position - worldPos;
                float  dist    = length(toLight);
                L = toLight / max(dist, 1e-4);
                float attenuation = 1.0 / (dist * dist);
                radiance = currentLight.color.rgb * currentLight.intensity * attenuation;
            }

            // Volumetric particles scatter rather than reflect: light wraps around the sphere and is
            // absorbed crossing it. For the parabolic sphere the optical depth of the light ray to this
            // surface point reduces to a cube of (N.L) -- closed form, no ray march -- so the shadow
            // side darkens smoothly and the shadow is "extruded" through the volume. N is the
            // roundness-blended normal, so sphereRoundness still shapes the spherical effect here.
            // Wrapped diffuse, no specular.

            if (volumetric) {
                float shadowV = 0.0;
                if(currentLight.castsShadows == 1)
                    shadowV = calculateShadow(worldPos, particlePc.cameraPos, particlePc.cameraForward,
                                              N, L, currentLight, particlePc.shadowAtlasIndex);

                float3 N1 = lerp(N,L,0.33-0.33*emitter.densityRange[1]);
                float sndl1  = (0.5 * particlePc.sphereRoundness) * dot(N1, L) + 1.0 - pow(particlePc.sphereRoundness,0.2);
                float sndl = max(sndl1,0.0);
                float depth = max(-sndl, 0.0);                  // light-ray penetration, shadow side only
                float tau   = (4.0 / 3.0) * sphereR * volDensity * depth * depth * depth;
                float trans = exp(-tau);                        // 1 on the lit side -> 0 deep in shadow
                float wrap  = sndl * max(0.95,0.95*emitter.densityRange[1]) + max(0.05,0.05*emitter.densityRange[1]);// soft wrap, no hard terminator
                Lo += (1.0 - shadowV) * albedo * radiance * (wrap * trans);
                continue;
            }

            float3 H = normalize(V + L);

            float NdotL_raw = dot(N, L);
            if(NdotL_raw <= 0.0) continue;

            float NdotV = max(dot(N, V), 0.001);
            float NdotL = max(NdotL_raw, 0.001);
            float HdotV = max(dot(H, V), 0.001);

            float3 F0 = lerp(float3(0.04), albedo, metallic);

            float NDF  = DistributionGGX(N, H, roughness);
            float G    = GeometrySmith(N, V, L, roughness);
            float3 F   = fresnelSchlick(HdotV, F0);

            float3 kS = F;
            float3 kD = (float3(1.0) - kS) * (1.0 - metallic);

            float3 specular = (NDF * G * F) / max(4.0 * NdotV * NdotL, 0.001);
            specular = clamp(specular, float3(0.0), float3(100.0));

            // Shadow: reuse the impostor normal for the normal-offset bias. Small
            // billboards, so the bias direction matters little.
            float shadow = 0.0;
            if(currentLight.castsShadows == 1) {
                shadow = calculateShadow(worldPos, particlePc.cameraPos, particlePc.cameraForward,
                                         N, L, currentLight, particlePc.shadowAtlasIndex);
            }

            Lo += (1.0 - shadow) * (kD * albedo / PI + specular) * radiance * NdotL;
        }

        // Small ambient so the shadowed side of the billboard isn't pure black.
        outCol.rgb = Lo + albedo * 0.3;
    }

    // Fade alpha out over the last 20% of life so particles don't vanish abruptly.
    float ageFrac = p.age / max(p.lifespan, 1e-4);
    float ageFade = 1.0 - smoothstep(0.8, 1.0, ageFrac);

    // Soft particles: fade as the billboard approaches the scene surface in front of it. Linearize
    // the sampled reverse-Z depth to view-space distance and compare with the particle's view depth.
    float softFade = 1.0;
    if ((emitter.flags & EMITTER_FLAG_SOFT) != 0u && emitter.softRadius > 0.0) {
        float n = particlePc.nearPlane, f = particlePc.farPlane;
        float sceneLinear = n * f / (n + sceneDepth * (f - n));
        softFade = saturate((sceneLinear - input.viewDepth) / emitter.softRadius);
    }

    outCol.a *= particlePc.opacity  * input.coverage * ageFade * softFade;
    return outCol;
}