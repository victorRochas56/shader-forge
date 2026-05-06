static const float PI = 3.14159265;
static const float invPI = 0.31830988618;

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
    denom = PI * denom * denom;

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

float3 calculateDiffuseIBL(float3 N, float3 albedo, TextureCube envMap, float maxMip, SamplerState envSampler)
{
    float3 irradiance = envMap.SampleLevel(envSampler, N, maxMip).rgb;
    return irradiance * albedo * invPI;
}

// Importance sampling: draw a half-vector from the GGX distribution
// xi is a uniform random 2D sample in [0,1)^2
float3 sampleGGX(float2 xi, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float phi = 2.0 * PI * xi.x;
    float cosTheta = sqrt((1.0 - xi.y) / (1.0 + (a2 - 1.0) * xi.y));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    return float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

// Build an orthonormal basis (TBN) from a normal vector
float3x3 buildTBN(float3 N)
{
    float3 up = abs(N.z) < 0.999 ? float3(0, 0, 1) : float3(1, 0, 0);
    float3 T = normalize(cross(up, N));
    float3 B = cross(N, T);
    return float3x3(T, B, N);
}
