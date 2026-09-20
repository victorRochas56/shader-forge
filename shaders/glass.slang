import modules.common;
import modules.pbr;

[shader("vertex")]
VertexOutput vertMain(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID, uint drawID : SV_DrawIndex)
{
    return litVertMain(vertexID, instanceID, drawID);
}

struct FragmentOutput
{
    float4 color : SV_Target0;
    float4 roughness_metal : SV_Target1;
    float4 worldNormal : SV_Target2;
};

#define UF litFrame[pc.frameUniformsIndex]

// Common window/bottle glass. Drives both the refraction bend and F0.
static const float GLASS_IOR = 1.5;

// Dielectric glass: albedo tints, its alpha is the opacity, roughness blurs both the reflection
// and the transmission. The lit pass has no blending and the prepass has already written this
// surface's depth, so "seeing through" is the environment cubemap sampled along the refracted
// ray, not the framebuffer — opacity crossfades between that and normal opaque shading.
[shader("fragment")]
FragmentOutput fragMain(VertexOutput input)
{
    FragmentOutput output;
    float2 uv = input.texCoord;
    uint flags = input.matInfo.y;
    bool hasRoughness = (flags & (1u << 2)) != 0;
    bool hasNormal    = (flags & (1u << 4)) != 0;

    SamplerState smp = samplers[UF.indicesA.x];

    // Issued back-to-back so their latencies overlap.
    float4 albedo    = textures[input.texIndices.x].Sample(smp, uv);
    float4 roughTex  = textures[input.texIndices.y].Sample(smp, uv);
    float3 normalTex = textures[input.texIndices.w].Sample(smp, uv).rgb;

    float3 N_world = normalize(input.normal);
    float3 N;
    if (hasNormal) {
        float3 normalMap = normalTex * 2.0 - 1.0;
        normalMap.y *= -1.0;
        normalMap = normalize(normalMap);
        float3 T = normalize(input.tangent);
        T = normalize(T - dot(T, N_world) * N_world);
        float3 B = normalize(cross(N_world, T));
        N = normalize(mul(normalMap, float3x3(T, B, N_world)));
    } else {
        N = N_world;
    }

    // Base colour factor tints the albedo map; its alpha is the opacity. No alpha clip here —
    // on glass alpha means transparency, not a cutout.
    albedo *= unpackColorRGBA8(input.matInfo.z);
    float opacity = saturate(albedo.a);

    float roughness = hasRoughness ? roughTex.r : input.matParams.y;

    float3 cameraPos = UF.cameraPosExtent.xyz;
    float3 cameraForward = UF.cameraForwardGI.xyz;
    float3 V = normalize(cameraPos - input.worldPos);
    float NdotV = max(dot(N, V), 0.001);
    uint shadowAtlasIdx = UF.indicesA.w;

    // Dielectric only — no metallic path, F0 comes from the IOR (0.04 at 1.5).
    float f0 = (GLASS_IOR - 1.0) / (GLASS_IOR + 1.0);
    float3 F0 = float3(f0 * f0);

    TextureCube envMap = cubemaps[input.matInfo.x];
    uint envWidth, envHeight, envMips;
    envMap.GetDimensions(0, envWidth, envHeight, envMips);
    float maxEnvMip = float(envMips - 1);

    // Single interface: the ray bends on entry and is never bent back on exit, so thin panes are
    // right and solid shapes just read as slightly displaced. Total internal reflection returns a
    // zero vector; fall back to the mirror direction there.
    float3 refractDir = refract(-V, N, 1.0 / GLASS_IOR);
    if (dot(refractDir, refractDir) < 1e-6) refractDir = reflect(-V, N);
    float3 transmitted = envMap.SampleLevel(smp, refractDir, roughness * maxEnvMip).rgb * albedo.rgb;

    float3 specularIBL = calculateSpecularIBL(N, V, roughness, F0, envMap, smp); // Fresnel folded in
    float3 diffuseIBL  = calculateDiffuseIBL(N, albedo.rgb, envMap, maxEnvMip, smp);

    float3 Lo = float3(0.0);

    for (uint i = 0; i < UF.indicesA.y; i++) {
        float4 colorIntensity = UF.lights[i].colorIntensity;
        // Sparse slot skip: removed lights leave a zeroed sentinel.
        if (colorIntensity.w <= 0.0) continue;

        uint4 typeFlags = UF.lights[i].typeFlags; // x type, y castsShadows, z numCascades, w cascade base
        float3 L;
        float3 radiance;
        if (typeFlags.x == 1) { // directional
            L = -normalize(UF.lights[i].direction.xyz);
            radiance = colorIntensity.rgb * colorIntensity.w;
        } else {
            float4 positionRange = UF.lights[i].positionRange;
            float3 toLight = positionRange.xyz - input.worldPos;
            float dist2 = dot(toLight, toLight);
            if (dist2 > positionRange.w * positionRange.w) continue;
            L = toLight / sqrt(dist2);
            radiance = colorIntensity.rgb * (colorIntensity.w / dist2);
        }

        float NdotL_raw = dot(N, L);
        if (NdotL_raw <= 0.0) continue;

        // Shadow tap first so its latency hides behind the BRDF math.
        float shadow = 0.0;
        if (typeFlags.y == 1) {
            Light* currentLight = &pc.lights[i];
            shadow = calculateShadowHot(input.worldPos, cameraPos, cameraForward, N_world, L,
                                        typeFlags.x, currentLight, shadowAtlasIdx,
                                        UF.lights[i].cascadeSplits.xyz, typeFlags.z, typeFlags.w);
        }

        float3 H = normalize(V + L);
        float NdotL = max(NdotL_raw, 0.001);
        float HdotV = max(dot(H, V), 0.001);

        float NDF = DistributionGGX(N, H, roughness);
        float G = GeometrySmith(N, V, L, roughness);
        float3 F = fresnelSchlick(HdotV, F0);

        // Specular reflects regardless of opacity; only the diffuse lobe needs a body to scatter in.
        float3 kD = (float3(1.0) - F) * opacity;

        float3 specular = NDF * G * F / max(4.0 * NdotV * NdotL, 0.001);
        specular = clamp(specular, float3(0.0), float3(100.0));

        Lo += (1.0 - shadow) * (kD * albedo.rgb / PI + specular) * radiance * NdotL;
    }

    float3 kS = fresnelSchlickRoughness(NdotV, F0, roughness);
    float3 kD = float3(1.0) - kS;

    // Whatever isn't reflected splits between diffuse scattering and transmission by opacity.
    float3 color = Lo + specularIBL + kD * lerp(transmitted, diffuseIBL, opacity);

    output.color = float4(color, 1.0);
    output.roughness_metal = float4(roughness, 0.0, 0.0, 1.0);
    output.worldNormal = float4(N * 0.5 + 0.5, 1.0);
    return output;
}
