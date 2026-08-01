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

[shader("fragment")]
FragmentOutput fragMain(VertexOutput input)
{
    FragmentOutput output;

    LitInstanceData inst = pc.litInstanceData[input.instanceID];

    bool flipNormal  = (inst.materialFlags & (1u << 0)) != 0;
    bool hasAlbedo   = (inst.materialFlags & (1u << 1)) != 0;
    bool hasRoughness= (inst.materialFlags & (1u << 2)) != 0;
    bool hasMetallic = (inst.materialFlags & (1u << 3)) != 0;
    bool hasNormal   = (inst.materialFlags & (1u << 4)) != 0;
    bool alphaClip   = (inst.materialFlags & (1u << 5)) != 0;

    float3 N;
    float3 normalMap;
    if(hasNormal) {
        normalMap = textures[inst.normalTextureIndex].Sample(samplers[pc.litPassData.samplerIndex], input.texCoord).rgb;
    }

    float4 albedo = float4(1.0, 1.0, 1.0, 1.0);
    if(hasAlbedo) {
        albedo = textures[inst.albedoTextureIndex].Sample(samplers[pc.litPassData.samplerIndex], input.texCoord);
    }
    if (alphaClip && albedo.a < inst.alphaCutoff)
        discard;


    float4 roughness = float4(inst.roughness, inst.roughness, inst.roughness, 1.0);
    if(hasRoughness) {
        roughness = textures[inst.roughnessTextureIndex].Sample(samplers[pc.litPassData.samplerIndex], input.texCoord);
    }

    float4 metallic = float4(inst.metallic, inst.metallic, inst.metallic, 1.0);
    if(hasMetallic){
        metallic = textures[inst.metallicTextureIndex].Sample(samplers[pc.litPassData.samplerIndex], input.texCoord);
    }


    if (hasNormal) {
        normalMap = normalMap * 2.0 - 1.0;
        float normalIntensity = 1.0;
        normalMap.xy *= normalIntensity;
        normalMap.y *= -1.0;
        normalMap = normalize(normalMap);

        float3 T = normalize(input.tangent);
        float3 N_world = normalize(input.normal);
        T = normalize(T - dot(T, N_world) * N_world);
        float3 B = normalize(cross(N_world, T));
        float3x3 TBN = float3x3(T, B, N_world);

        N = normalize(mul(normalMap, TBN));
    } else {
        N = normalize(input.normal);
    }

    //do as early as possible to not wait for scoreboard stall
    TextureCube envMap = cubemaps[inst.environmentMapIndex];
    SamplerState envSampler = samplers[pc.litPassData.samplerIndex];
    //float3 specularIBL = calculateSpecularIBL(N, V, roughness.r, F0, envMap, envSampler);
    float3 diffuseIBL = calculateDiffuseIBL(N, albedo.rgb, envMap, inst.maxEnvMips, envSampler);

    float3 V = normalize(pc.litPassData.cameraPos.xyz - input.worldPos);
    float3 F0 = float3(0.04);
    F0 = lerp(F0, albedo.rgb, metallic.r);

    float3 Lo = float3(0.0);

    for(int i = 0; i < pc.litPassData.lightCount; i++){
        Light* currentLight = &pc.lights[i];

        // Sparse slot skip: removed lights leave a zeroed sentinel in the buffer.
        if(currentLight.intensity <= 0.0) continue;

        if(currentLight.type != 1) {
            float distance = length(currentLight.position - input.worldPos);
            if(distance > currentLight.range){
                continue;
            }
        }

        float3 L;
        float attenuation;
        float3 radiance;

        if(currentLight.type == 1) {
            float3 lightDir = currentLight.direction;
            lightDir = normalize(lightDir);
            L = -lightDir;
            attenuation = 1.0;
            radiance = currentLight.color.rgb * currentLight.intensity;
        }
        else {
            float3 lightPos = currentLight.position;
            float distance = length(lightPos - input.worldPos);

            L = normalize(lightPos - input.worldPos);
            attenuation = 1.0 / (distance * distance);
            radiance = currentLight.color.rgb * currentLight.intensity * attenuation;
        }

        float3 H = normalize(V + L);

        float NdotL_raw = dot(N, L);
        if(NdotL_raw <= 0.0) {
            continue;
        }

        float NdotV = max(dot(N, V), 0.001);
        float NdotL = max(NdotL_raw, 0.001);
        float NdotH = max(dot(N, H), 0.001);
        float HdotV = max(dot(H, V), 0.001);

        float NDF = DistributionGGX(N, H, roughness.r);
        float G = GeometrySmith(N, V, L, roughness.r);
        float3 F = fresnelSchlick(HdotV, F0);

        float3 kS = F;
        float3 kD = float3(1.0) - kS;
        kD *= 1.0 - metallic.r;

        float3 numerator = NDF * G * F;
        float denominator = max(4.0 * NdotV * NdotL, 0.001);
        float3 specular = numerator / denominator;
        specular = max(specular, float3(0.0, 0.0, 0.0));
        specular = min(specular, float3(100.0, 100.0, 100.0));

        float shadow = 0.0;
        if(currentLight.castsShadows == 1) {
            float3 geometricNormal = normalize(input.normal);
            shadow = calculateShadow(input.worldPos, pc.litPassData.cameraPos, pc.litPassData.cameraForward, geometricNormal, L, currentLight, pc.litPassData.shadowAtlasIndex);
            if( currentLight.showCascades>0){
                if(shadow == 0){
                    output.color = float4(1,0,0,1);
                }
                if(shadow == 1){
                    output.color = float4(0,1,0,1);
                }
                if(shadow == 2){
                    output.color = float4(0,0,1,1);
                }
                if(shadow == 3){
                    output.color = float4(1,1,0,1);
                }
                return output;
            }
        }

        Lo += (1.0 - shadow) * (kD * albedo.rgb / 3.14159265 + specular) * radiance * NdotL;
    }

    float3 GI = coneTraceVXGI(pc.litPassData.voxelTextureIndex, pc.litPassData.voxelSamplerIndex,
                              pc.litPassData.voxelViewProjection, pc.litPassData.voxelResolution,
                              pc.litPassData.voxelWorldExtent, input.worldPos, input.normal,
                              pc.litPassData.giHemisphereRays, pc.litPassData.giMaxSteps,
                              pc.litPassData.giFetchBatch);

    float3 kS = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness.r);
    float3 kD = float3(1.0) - kS;
    kD *= 1.0 - metallic.r;

    float ao = 1.0;
    float3 ambient = 0.0; //(kD * diffuseIBL) * ao;
    float3 color = ambient + Lo + GI * kD * albedo.rgb;

    output.color = float4(color, 1.0);
    output.roughness_metal = float4(roughness.r, metallic.r, 0.0, 1.0);
    output.worldNormal = float4(N * 0.5 + 0.5, 1.0);
    return output;
}

