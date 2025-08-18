// Push constants
struct PushConstants
{
    uint vertexBufferIndex;
    uint vertexOffset;
    uint vertexStride;
    uint modelMatrixIndex;
    uint albedoTextureIndex;
    uint roughnessTextureIndex;
    uint metallicTextureIndex;
    uint normalTextureIndex;
    uint samplerIndex;
    uint lightCount;
    uint environmentMapIndex;
    uint debugMaps;
    float4x4 viewProjection;
    float4 cameraPos;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]]
Texture2D bindlessTextures[];

[[vk::binding(1, 0)]]
SamplerState bindlessSamplers[];

[[vk::binding(2, 0)]]
ByteAddressBuffer bindlessVertexBuffers[];

[[vk::binding(3, 0)]]
StructuredBuffer<float4x4> bindlessModelMatrices[];

[[vk::binding(4,0)]]
StructuredBuffer<PointLight> lights[];

[[vk::binding(5,0)]]
TextureCube cubemaps[];

struct PointLight{
    float4 position;
    float4 color;
    float range;
    float intensity;
    uint padding1;
    uint padding2;
};

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

Vertex loadVertex(uint bufferIndex, uint vertexIndex, uint stride, uint offset)
{
    Vertex vert;

    uint byteOffset = offset + vertexIndex * stride;
    vert.position = float3(asfloat(bindlessVertexBuffers[bufferIndex].Load(byteOffset)),
                        asfloat(bindlessVertexBuffers[bufferIndex].Load(byteOffset + 4)),
                        asfloat(bindlessVertexBuffers[bufferIndex].Load(byteOffset + 8)));

    vert.normal = float3(asfloat(bindlessVertexBuffers[bufferIndex].Load(byteOffset + 12)),
                        asfloat(bindlessVertexBuffers[bufferIndex].Load(byteOffset + 16)),
                        asfloat(bindlessVertexBuffers[bufferIndex].Load(byteOffset + 20)));

    vert.texCoord = float2(asfloat(bindlessVertexBuffers[bufferIndex].Load(byteOffset + 24)),
                        asfloat(bindlessVertexBuffers[bufferIndex].Load(byteOffset + 28)));
    
    vert.tangent = float3(asfloat(bindlessVertexBuffers[bufferIndex].Load(byteOffset + 32)),
                        asfloat(bindlessVertexBuffers[bufferIndex].Load(byteOffset + 36)),
                        asfloat(bindlessVertexBuffers[bufferIndex].Load(byteOffset + 40)));
    
    return vert;
}

[shader("vertex")]
VertexOutput vertMain(uint vertexID : SV_VertexID)
{
    // load vertex data from bindless buffer
    Vertex vertex = loadVertex(
        pc.vertexBufferIndex,
        vertexID,
        pc.vertexStride,
        pc.vertexOffset
    );
   
    float4x4 modelMatrix = bindlessModelMatrices[0][pc.modelMatrixIndex];

    VertexOutput output;

    float4 worldPos = mul(modelMatrix, float4(vertex.position, 1.0));
    output.worldPos = worldPos.xyz;
    output.position = mul(pc.viewProjection,worldPos);
    output.normal = mul((float3x3)modelMatrix, vertex.normal);
    output.texCoord = vertex.texCoord;
    output.tangent = normalize(mul((float3x3)modelMatrix, vertex.tangent));;
    output.bitangent = normalize(cross(output.normal, output.tangent));
    return output;
}

float3 fresnelSchlick(float cosTheta, float3 F0, float roughness)
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

// Calculate specular IBL contribution  
float3 calculateSpecularIBL(float3 N, float3 V, float roughness, float3 F0, TextureCube envMap, SamplerState envSampler)
{
    // Calculate reflection vector
    float3 R = reflect(-V, N);
    
    // Sample environment map with roughness-based mip level
    // For proper IBL, you'd use a prefiltered environment map
    float mipLevel = roughness * 7.0; // Assuming 8 mip levels (0-7)
    float3 prefilteredColor = envMap.SampleLevel(envSampler, R, mipLevel).rgb;
    
    // Simplified BRDF approximation (normally you'd use a BRDF LUT)
    float NdotV = max(dot(N, V), 0.0);
    float3 F = fresnelSchlick(NdotV, F0, roughness);
    
    // Simple approximation for the BRDF integration
    float2 envBRDF;
    envBRDF.x = lerp(0.0, 1.0, 1.0 - roughness);
    envBRDF.y = lerp(0.0, 1.0, NdotV);
    
    return prefilteredColor * (F * envBRDF.x + envBRDF.y);
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
    float4 metallic= bindlessTextures[pc.metallicTextureIndex].Sample(bindlessSamplers[pc.samplerIndex], input.texCoord);
    float4 roughness = bindlessTextures[pc.roughnessTextureIndex].Sample(bindlessSamplers[pc.samplerIndex], input.texCoord);
    float4 albedo = bindlessTextures[pc.albedoTextureIndex].Sample(bindlessSamplers[pc.samplerIndex], input.texCoord);
    float3 normalMap = bindlessTextures[pc.normalTextureIndex].Sample(bindlessSamplers[pc.samplerIndex], input.texCoord).rgb;
    normalMap = normalMap * 2.0 - 1.0;
    float normalIntensity = 1.0;
    normalMap.xy *= normalIntensity; 
    normalMap = normalize(normalMap);

    switch(pc.debugMaps){
        case 1 : {
            output.color  = albedo;
            return output;
        }
        case 2 : {
            output.color = roughness;
            return output;
        }
        case 3 : {
            output.color = metallic;
            return output;
        }
        case 4 : {
            output.color = float4(normalMap,1.0);
            return output;
        }
    }
    // Create TBN matrix
    float3 T = normalize(input.tangent);
    float3 B = normalize(input.bitangent);
    float3 N_world = normalize(input.normal);
    // Ensure orthogonality
    T = normalize(T - dot(T, N_world) * N_world);
    B = cross(N_world, T);
    float3x3 TBN = float3x3(T, B, N_world);
    // Transform normal to world space
    float3 N = normalize(mul(normalMap, TBN));

    float3 V = normalize(pc.cameraPos.xyz - input.worldPos);
    float3 F0 = float3(0.04); 
    F0 = lerp(F0, albedo.rgb, metallic.r);
    output.color = float4(0.0,0.0,0.0,1.0);
   
    float3 Lo = float3(0.0);

    for(int i = 0; i<pc.lightCount; i++){
        float distance    = length(lights[0][i].position.xyz - input.worldPos);
        if(distance > lights[0][i].range){
            continue;
        }
        float3 L = normalize(lights[0][i].position.xyz - input.worldPos);
        float3 H = normalize(V + L);
        float attenuation = 1.0 / (distance * distance);
        float3 radiance     = lights[0][i].color.rgb * lights[0][i].intensity *attenuation;        
        
        // cook-torrance brdf
        float NDF = DistributionGGX(N, H, roughness.r);        
        float G   = GeometrySmith(N, V, L, roughness.r);      
        float3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0, roughness.r);       
        
        float3 kS = F;
        float3 kD = float3(1.0) - kS;
        kD *= 1.0 - metallic.r;	  
        
        float3 numerator    = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
        float3 specular     = numerator / denominator;  
            
        // add to outgoing radiance Lo
        float NdotL = max(dot(N, L), 0.0);                
        Lo += (1.0* albedo.rgb / 3.14159265 + specular) * radiance * NdotL;
    }
    
    // Image-Based Lighting (IBL)
    TextureCube envMap = cubemaps[pc.environmentMapIndex];
    SamplerState envSampler = bindlessSamplers[pc.samplerIndex];
        
    // Calculate specular IBL
    float3 specularIBL = calculateSpecularIBL(N, V, roughness.r, F0, envMap, envSampler);
    
    // Combine direct lighting with IBL
    float3 kS = fresnelSchlick(max(dot(N, V), 0.0), F0, roughness.r);
    
    // Final IBL contribution
    float ao = 1.0; // You might want to add an AO texture
    float3 color = kS * specularIBL + Lo;
    output.color = float4(color, 1.0);

    return output;
}