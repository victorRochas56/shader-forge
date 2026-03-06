struct PushConstants {
    uint vertexAllocationIndex;
    uint vertexOffset;
    uint vertexStride;
    uint modelMatrixIndex;

    uint albedoTextureIndex;
    uint roughnessTextureIndex;
    uint metallicTextureIndex;
    uint normalTextureIndex;

    uint environmentMapIndex;
    uint samplerIndex;
    uint lightCount;
    uint shadowSamplerIndex;

    float3 cameraPos;
    uint materialFlags;

    uint elementOffsetModel;  // Element offset for this frame's model matrices
    uint elementOffsetLight;  // Element offset for this frame's lights
    float metallic;
    float roughness;

    float4x4 viewProjection;
};

struct Cascade {
    float4x4 lightSpaceMatrix;
    uint shadowMapIndex;
    float splitDistance;
    float texelSize;
    float worldTexelSize;
};

struct Light {
    uint type;
    uint modelMatrixIndex;
    uint padding;
    uint padding1;
    float range;
    float intensity;
    uint shadowMapIndex;
    uint shadowResolution;
    float4 color;
    float4x4 lightSpaceMatrix;
    float3 direction;
    int castsShadows;
    uint showCascades;
    uint padding3;
    uint padding4;
    uint numCascades;
    Cascade cascades[4];
}

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[[vk::binding(0,0)]]
Texture2D textures[];

[[vk::binding(1,0)]]
TextureCube cubemaps[];

[[vk::binding(2,0)]]
SamplerState samplers[];

[[vk::binding(3,0)]]
ByteAddressBuffer vertexBuffer[];

[[vk::binding(4, 0)]]
StructuredBuffer<float4x4> modelMatrices[];

[[vk::binding(5,0)]]
StructuredBuffer<Light> lights[];

struct Vertex
{
    float3 position;
    float3 normal;
    float2 texCoord;
    float3 tangent;
};

struct VertexOutput
{
    float4 position : SV_Position;
    float3 worldPos : WORLD_POS;
    float3 normal : NORMAL;
    float2 texCoord : TEXCOORD0;
    float3 tangent : TANGENT; 
    float3 bitangent : BITANGENT; 
};

Vertex loadVertex(uint vertexIndex, uint stride, uint offset)
{
    Vertex vert;
    
    uint byteOffset = offset + vertexIndex * stride;
    
    //position 
    uint posX_uint = vertexBuffer[0].Load(byteOffset);
    uint posY_uint = vertexBuffer[0].Load(byteOffset + 4);
    uint posZ_uint = vertexBuffer[0].Load(byteOffset + 8);
    
    vert.position = float3(asfloat(posX_uint), asfloat(posY_uint), asfloat(posZ_uint));
    
    //normal 
    uint normX_uint = vertexBuffer[0].Load(byteOffset + 12);
    uint normY_uint = vertexBuffer[0].Load(byteOffset + 16);  
    uint normZ_uint = vertexBuffer[0].Load(byteOffset + 20);
    
    vert.normal = float3(asfloat(normX_uint), asfloat(normY_uint), asfloat(normZ_uint));
    
    //tangent
    uint tangX_uint = vertexBuffer[0].Load(byteOffset + 24);
    uint tangY_uint = vertexBuffer[0].Load(byteOffset + 28);
    uint tangZ_uint = vertexBuffer[0].Load(byteOffset + 32);
    
    vert.tangent = float3(asfloat(tangX_uint), asfloat(tangY_uint), asfloat(tangZ_uint));
    
    // Skip materialIndex at offset 36-39
    
    //UV 
    uint texU_uint = vertexBuffer[0].Load(byteOffset + 40);
    uint texV_uint = vertexBuffer[0].Load(byteOffset + 44);
    
    vert.texCoord = float2(asfloat(texU_uint), asfloat(texV_uint));
    
    return vert;
}

float3 getPosFromMM(int index){
    float3 ret;
    ret.x = modelMatrices[0][pc.elementOffsetModel + index][0].w;
    ret.y = modelMatrices[0][pc.elementOffsetModel + index][1].w;
    ret.z = modelMatrices[0][pc.elementOffsetModel + index][2].w;
    return ret;
} 


[shader("vertex")]
VertexOutput vertMain(uint vertexID : SV_VertexID)
{
    Vertex vertex = loadVertex(vertexID, pc.vertexStride, pc.vertexOffset);
    
    float4x4 modelMatrix = modelMatrices[0][pc.elementOffsetModel + pc.modelMatrixIndex];
    VertexOutput output;
    float4 worldPos = mul(modelMatrix, float4(vertex.position, 1.0));
    output.worldPos = worldPos.xyz;
    output.position = mul(pc.viewProjection, worldPos);
    output.normal = mul((float3x3)modelMatrix, vertex.normal);
    output.texCoord = vertex.texCoord;
    output.tangent = normalize(mul((float3x3)modelMatrix, vertex.tangent));
    output.bitangent = normalize(cross(output.normal, output.tangent));
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
    //TODO replace with irradiance map
    uint width, height, mipLevels;
    envMap.GetDimensions(0, width, height, mipLevels);
    float maxMipLevel = float(mipLevels - 1);
    float3 irradiance = envMap.SampleLevel(envSampler, N, maxMipLevel).rgb; // highest mip of cubemap for ambient lighting
    return irradiance * albedo / 3.14159265;
}

float calculateShadow(float3 fragWorldPos, float3 cameraPos, float3 geometricNormal, float3 L, Light light)
{
    float viewDepth = length(cameraPos - fragWorldPos);

    // Select the appropriate cascade based on view distance
    // We want the last cascade whose split distance is greater than viewDepth
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

    // Check if fragment is outside cascade bounds - return no shadow (fully lit)
    if (projCoords.x < 0.0 || projCoords.x > 1.0 ||
        projCoords.y < 0.0 || projCoords.y > 1.0 ||
        projCoords.z < 0.0 || projCoords.z > 1.0) {
        return 0.0; // 0.0 = no shadow = fully lit
    }

    float currentDepth = projCoords.z;

    // Add a small bias to reduce shadow acne
    // The bias is scaled by the texel size to be resolution-independent
    // Normal-based bias: surfaces at grazing angles need more bias to prevent self-shadowing
    float baselineBias = max(0.001, cascade.texelSize * 0.5);
    float normalBias = abs(dot(geometricNormal, normalize(L))) * cascade.texelSize;
    float bias = baselineBias + normalBias * 2.0;
    currentDepth -= bias;

    // ============================================================================
    // PCF (Percentage Closer Filtering) Implementation
    // Samples a 3x3 kernel around the fragment position and performs depth comparison
    // ============================================================================

    float shadow = 0.0;
    float texelSize = cascade.texelSize;
    int samplesPerSide = 3;
    int totalSamples = samplesPerSide * samplesPerSide;

    // 3x3 PCF kernel
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            float2 sampleCoords = projCoords.xy + float2(x, y) * texelSize;

            // Sample the depth from the shadow map
            float shadowMapDepth = textures[cascade.shadowMapIndex].Sample(
                samplers[pc.shadowSamplerIndex],
                sampleCoords
            ).r;

            // Compare: if current depth > shadow map depth, this sample is in shadow
            // Count samples that are IN shadow
            if (currentDepth < shadowMapDepth) {
                shadow += 1.0;
            }
        }
    }

    // Average the results to get shadow factor (0.0 = fully lit, 1.0 = fully shadowed)
    shadow /= float(totalSamples);

    // Return inverted shadow value: 1.0 = fully lit, 0.0 = fully shadowed
    // This matches the expected behavior in the lighting calculation
    return 1.0 - shadow;
}

struct FragmentOutput
{
    float4 color : SV_Target0;
};
[shader("fragment")]
[earlydepthstencil]
FragmentOutput fragMain(VertexOutput input)
{
    FragmentOutput output;
    
    // texture mask checks
    bool flipNormal = (pc.materialFlags & (1u <<0 )) != 0;
    bool hasAlbedo = (pc.materialFlags & (1u << 1)) != 0;
    bool hasRoughness = (pc.materialFlags & (1u << 2)) != 0;
    bool hasMetallic = (pc.materialFlags & (1u << 3)) != 0;
    bool hasNormal = (pc.materialFlags & (1u << 4)) != 0;
    
    float4 albedo = hasAlbedo ? 
        textures[pc.albedoTextureIndex].Sample(samplers[pc.samplerIndex], input.texCoord) : 
        float4(1.0, 1.0, 1.0, 1.0);
    
    float4 roughness = hasRoughness ?
        textures[pc.roughnessTextureIndex].Sample(samplers[pc.samplerIndex], input.texCoord) :
        float4(pc.roughness, pc.roughness, pc.roughness, 1.0);

    float4 metallic = hasMetallic ?
        textures[pc.metallicTextureIndex].Sample(samplers[pc.samplerIndex], input.texCoord) :
        float4(pc.metallic, pc.metallic, pc.metallic, 1.0);
    
    float3 N;
    if (hasNormal) {
        float3 normalMap = textures[pc.normalTextureIndex].SampleLevel(samplers[pc.samplerIndex], input.texCoord, 0).rgb;
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
            shadow = calculateShadow(input.worldPos, pc.cameraPos, geometricNormal, L, currentLight);
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
    
    // Image-Based Lighting (IBL)
    TextureCube envMap = cubemaps[pc.environmentMapIndex];
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
    return output;
}