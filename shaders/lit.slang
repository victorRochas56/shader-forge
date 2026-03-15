import modules.common;

[shader("vertex")]
VertexOutput vertMain(uint vertexID : SV_VertexID, uint drawID : SV_DrawIndex)
{
    return litVertMain(vertexID, drawID);
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

    LitDrawData draw = litDrawData[0][pc.elementOffsetLit + input.drawID];

    bool flipNormal  = (draw.materialFlags & (1u << 0)) != 0;
    bool hasAlbedo   = (draw.materialFlags & (1u << 1)) != 0;
    bool hasRoughness= (draw.materialFlags & (1u << 2)) != 0;
    bool hasMetallic = (draw.materialFlags & (1u << 3)) != 0;
    bool hasNormal   = (draw.materialFlags & (1u << 4)) != 0;
    bool alphaClip   = (draw.materialFlags & (1u << 5)) != 0;

    float4 albedo = hasAlbedo ?
        textures[draw.albedoTextureIndex].Sample(samplers[pc.samplerIndex], input.texCoord) :
        float4(1.0, 1.0, 1.0, 1.0);

    if (alphaClip && albedo.a < draw.alphaCutoff)
        discard;

    float4 roughness = hasRoughness ?
        textures[draw.roughnessTextureIndex].Sample(samplers[pc.samplerIndex], input.texCoord) :
        float4(draw.roughness, draw.roughness, draw.roughness, 1.0);

    float4 metallic = hasMetallic ?
        textures[draw.metallicTextureIndex].Sample(samplers[pc.samplerIndex], input.texCoord) :
        float4(draw.metallic, draw.metallic, draw.metallic, 1.0);

    float3 N;
    if (hasNormal) {
        float3 normalMap = textures[draw.normalTextureIndex].SampleLevel(samplers[pc.samplerIndex], input.texCoord, 0).rgb;
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

    float3 V = normalize(pc.cameraPos.xyz - input.worldPos);
    float3 F0 = float3(0.04);
    F0 = lerp(F0, albedo.rgb, metallic.r);

    float3 Lo = float3(0.0);

    for(int i = 0; i < pc.lightCount; i++){
        Light currentLight = lights[0][pc.elementOffsetLight + i];

        if(currentLight.type != 1) {
            float distance = length(getPosFromMM(currentLight.modelMatrixIndex) - input.worldPos);
            if(distance > currentLight.range){
                continue;
            }
        }

        float3 L;
        float attenuation;
        float3 radiance;

        if(currentLight.type == 1) {
            float3 lightDir = float3(
                modelMatrices[0][pc.elementOffsetModel + currentLight.modelMatrixIndex][0].x,
                modelMatrices[0][pc.elementOffsetModel + currentLight.modelMatrixIndex][1].x,
                modelMatrices[0][pc.elementOffsetModel + currentLight.modelMatrixIndex][2].x
            );
            lightDir = normalize(lightDir);
            L = -lightDir;
            attenuation = 1.0;
            radiance = currentLight.color.rgb * currentLight.intensity;
        }
        else {
            float3 lightPos = getPosFromMM(currentLight.modelMatrixIndex);
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
        specular = min(specular, float3(10.0, 10.0, 10.0));

        float shadow = 0.0;
        if(currentLight.castsShadows == 1) {
            float3 geometricNormal = normalize(input.normal);
            shadow = calculateShadow(input.worldPos, pc.cameraPos, pc.cameraForward, geometricNormal, L, currentLight);
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

    TextureCube envMap = cubemaps[draw.environmentMapIndex];
    SamplerState envSampler = samplers[pc.samplerIndex];
    float3 specularIBL = calculateSpecularIBL(N, V, roughness.r, F0, envMap, envSampler);
    float3 diffuseIBL = calculateDiffuseIBL(N, albedo.rgb, envMap, envSampler);

    float3 kS = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness.r);
    float3 kD = float3(1.0) - kS;
    kD *= 1.0 - metallic.r;

    float ao = 1.0;
    float3 ambient = (kD * diffuseIBL + kS * specularIBL) * ao;
    float3 color = ambient + Lo;

    output.color = float4(color, 1.0);
    output.roughness_metal = float4(roughness.r, metallic.r, 0.0, 1.0);
    output.worldNormal = float4(N * 0.5 + 0.5, 1.0);
    return output;
}



float3 fresnelSchlick(float cosTheta, float3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float3 fresnelSchlickRoughness(float cosTheta, float3 F0, float roughness)
{
    return F0 + (max(float3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float DistributionGGX(float3 N, float3 H, float roughness)
{
    float a      = roughness*roughness;
    float a2     = a*a;
    float NdotH  = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float num   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = 3.14159265 * denom * denom;

    return num / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float num   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return num / denom;
}

float GeometrySmith(float3 N, float3 V, float3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2  = GeometrySchlickGGX(NdotV, roughness);
    float ggx1  = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

float3 calculateSpecularIBL(float3 N, float3 V, float roughness, float3 F0, TextureCube envMap, SamplerState envSampler)
{
    float3 R = reflect(-V, N);

    uint width, height, mipLevels;
    envMap.GetDimensions(0, width, height, mipLevels);
    float maxMipLevel = float(mipLevels - 1);

    float mipLevel = roughness * maxMipLevel;

    float3 prefilteredColor = envMap.SampleLevel(envSampler, R, mipLevel).rgb;

    float NdotV = max(dot(N, V), 0.0);
    float3 F = fresnelSchlickRoughness(NdotV, F0, roughness);

    return prefilteredColor * F;
}

float3 calculateDiffuseIBL(float3 N, float3 albedo, TextureCube envMap, SamplerState envSampler)
{
    uint width, height, mipLevels;
    envMap.GetDimensions(0, width, height, mipLevels);
    float maxMipLevel = float(mipLevels - 1);
    float3 irradiance = envMap.SampleLevel(envSampler, N, maxMipLevel).rgb;
    return irradiance * albedo / 3.14159265;
}

float calculateShadow(float3 fragWorldPos, float3 cameraPos, float3 cameraForward, float3 geometricNormal, float3 L, Light light)
{
    float viewDepth = dot(fragWorldPos - cameraPos, cameraForward);

    int cascadeIndex = int(light.numCascades) - 1;
    for (int i = 0; i < int(light.numCascades); i++) {
        if (viewDepth < light.cascades[i].splitDistance) {
            cascadeIndex = i;
            break;
        }
    }
    if( light.showCascades>0){
        return cascadeIndex;
    }

    Cascade cascade = light.cascades[cascadeIndex];

    float4 fragPosLightSpace = mul(cascade.lightSpaceMatrix, float4(fragWorldPos, 1.0));
    float3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords.x = projCoords.x * 0.5 + 0.5;
    projCoords.y = projCoords.y * 0.5 + 0.5;

    if (projCoords.x < 0.0 || projCoords.x > 1.0 ||
        projCoords.y < 0.0 || projCoords.y > 1.0 ||
        projCoords.z < 0.0 || projCoords.z > 1.0) {
        return 0.0;
    }

    float currentDepth = projCoords.z;

    float baselineBias = max(0.001, cascade.texelSize * 0.5);
    float normalBias = abs(dot(geometricNormal, normalize(L))) * cascade.texelSize;
    float bias = baselineBias + normalBias * 2.0;
    currentDepth -= bias;

    float shadow = 0.0;
    float texelSize = cascade.texelSize;
    int samplesPerSide = 3;
    int totalSamples = samplesPerSide * samplesPerSide;

    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            float2 sampleCoords = projCoords.xy + float2(x, y) * texelSize;

            float shadowMapDepth = textures[cascade.shadowMapIndex].Sample(
                samplers[pc.shadowSamplerIndex],
                sampleCoords
            ).r;

            if (currentDepth < shadowMapDepth) {
                shadow += 1.0;
            }
        }
    }

    shadow /= float(totalSamples);
    return 1.0 - shadow;
}
