import modules.common;
import modules.pbr;
import modules.voxel;

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

// Hot frame data for this pass (constant-cache reads instead of per-fragment BDA loads).
#define UF litFrame[pc.frameUniformsIndex]

// GI_MODE: 0 = per-pixel cone trace, 1 = ambient-cube lookup. Baked per entry point so each
// pipeline variant only compiles its own GI path — carrying both pushed live registers to ~124
// and collapsed occupancy for every fragment, even when the cone path never ran.
// DEBUG_CASCADES compiles the cascade-visualization path; only the fallback fragMain keeps it.
FragmentOutput litFragImpl<let GI_MODE : int, let DEBUG_CASCADES : bool>(VertexOutput input)
{
    FragmentOutput output;
    float2 uv = input.texCoord;
    uint flags = input.matInfo.y;
    bool hasRoughness = (flags & (1u << 2)) != 0;
    bool hasMetallic  = (flags & (1u << 3)) != 0;
    bool hasNormal    = (flags & (1u << 4)) != 0;
    bool alphaClip    = (flags & (1u << 5)) != 0;

    SamplerState smp = samplers[UF.indicesA.x];

    // Ambient-cube taps first: they only need worldPos + geometric normal, so the three volume
    // fetches stay in flight through the material fetches, TBN math and the whole light loop.
    AmbientCubeTaps giTaps;
    giTaps.s0 = float4(0.0);
    giTaps.s1 = float4(0.0);
    giTaps.s2 = float4(0.0);
    giTaps.w = float3(0.0);
    if (GI_MODE == 1) {
        uint faceIndices[6] = { UF.giIrradianceA.x, UF.giIrradianceA.y, UF.giIrradianceA.z,
                                UF.giIrradianceA.w, UF.giIrradianceB.x, UF.giIrradianceB.y };
        giTaps = sampleAmbientCubeTaps(faceIndices, UF.indicesB.y, UF.voxelViewProjection, input.worldPos, input.normal);
    }

    // All four material fetches issued back-to-back with no control flow between them, so their
    // latencies overlap instead of serializing. Absent maps point at 1x1 defaults (renderer side),
    // making every index safe to sample.
    float4 albedo    = textures[input.texIndices.x].Sample(smp, uv);
    float4 roughTex  = textures[input.texIndices.y].Sample(smp, uv);
    float4 metalTex  = textures[input.texIndices.z].Sample(smp, uv);
    float3 normalTex = textures[input.texIndices.w].Sample(smp, uv).rgb;

    // Base colour factor. Tints the albedo map, and is the whole colour when there is no map —
    // absent albedo resolves to the 1x1 white default, so the multiply is the only thing there.
    // Before the alpha test so the colour's alpha clips with it.
    albedo *= unpackColorRGBA8(input.matInfo.z);

    if (alphaClip && albedo.a < input.matParams.z)
        discard;

    float roughness = hasRoughness ? roughTex.r : input.matParams.y;
    float metallic  = hasMetallic  ? metalTex.r : input.matParams.x;

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

    float3 cameraPos = UF.cameraPosExtent.xyz;
    float3 cameraForward = UF.cameraForwardGI.xyz;
    float3 V = normalize(cameraPos - input.worldPos);
    float3 F0 = lerp(float3(0.04), albedo.rgb, metallic);
    float NdotV = max(dot(N, V), 0.001);
    uint shadowAtlasIdx = UF.indicesA.w;

    float3 Lo = float3(0.0);

    for (uint i = 0; i < UF.indicesA.y; i++) {
        float4 colorIntensity = UF.lights[i].colorIntensity;
        // Sparse slot skip: removed lights leave a zeroed sentinel.
        if (colorIntensity.w <= 0.0) continue;

        uint4 typeFlags = UF.lights[i].typeFlags; // x type, y castsShadows, z numCascades
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

        // Shadow tap issued before the BRDF math so the compare-fetch latency hides behind it.
        float shadow = 0.0;
        if (typeFlags.y == 1) {
            Light* currentLight = &pc.lights[i];
            if (DEBUG_CASCADES) {
                if (currentLight.showCascades > 0) {
                    shadow = calculateShadow(input.worldPos, cameraPos, cameraForward, N_world, L, currentLight, shadowAtlasIdx);
                    if (shadow == 0) output.color = float4(1, 0, 0, 1);
                    if (shadow == 1) output.color = float4(0, 1, 0, 1);
                    if (shadow == 2) output.color = float4(0, 0, 1, 1);
                    if (shadow == 3) output.color = float4(1, 1, 0, 1);
                    return output;
                }
            }
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

        float3 kD = (float3(1.0) - F) * (1.0 - metallic);

        float3 specular = NDF * G * F / max(4.0 * NdotV * NdotL, 0.001);
        specular = clamp(specular, float3(0.0), float3(100.0));

        Lo += (1.0 - shadow) * (kD * albedo.rgb / 3.14159265 + specular) * radiance * NdotL;
    }

    float3 GI;
    if (GI_MODE == 1) {
        GI = resolveAmbientCubeGI(giTaps);
    } else {
        GI = coneTraceVXGI(UF.indicesB.x, UF.indicesB.y, UF.voxelViewProjection,
                           UF.indicesB.z, UF.cameraPosExtent.w, input.worldPos, input.normal,
                           UF.giParams.x, UF.giParams.y, UF.giParams.z,
                           input.matInfo.x, UF.giFloats.x);
    }
    GI *= UF.cameraForwardGI.w;

    float3 kS = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);
    float3 kD = (float3(1.0) - kS) * (1.0 - metallic);

    float3 color = Lo + GI * kD * albedo.rgb;

    output.color = float4(color, 1.0);
    output.roughness_metal = float4(roughness, metallic, 0.0, 1.0);
    output.worldNormal = float4(N * 0.5 + 0.5, 1.0);
    return output;
}

// Fallback/debug: runtime giMode dispatch + cascade visualization. Bound for material thumbnails
// and whenever a light has showCascades on; the main pass binds the specialized entries below.
[shader("fragment")]
FragmentOutput fragMain(VertexOutput input)
{
    if (UF.indicesB.w == 1) return litFragImpl<1, true>(input);
    return litFragImpl<0, true>(input);
}

[shader("fragment")]
FragmentOutput fragMainGICone(VertexOutput input)
{
    return litFragImpl<0, false>(input);
}

[shader("fragment")]
FragmentOutput fragMainGICube(VertexOutput input)
{
    return litFragImpl<1, false>(input);
}
