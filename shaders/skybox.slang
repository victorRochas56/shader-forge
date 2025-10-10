struct SkyBoxConstants
{
    uint skyboxIndex;
    float blur;
    uint padding2;
    uint padding3;
    float4x4 invViewProjMatrix;

};

struct Light {
    int type;
    int modelMatrixIndex;
    int padding;
    int padding1;
    float4 color;
    float range;
    float intensity;
}

[[vk::push_constant]]
ConstantBuffer<SkyBoxConstants> pc;

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


struct ScreenVertex
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
ScreenVertex vertMain(uint vertexID : SV_VertexID)
{
    ScreenVertex output;
    
    // Generate fullscreen triangle
    output.uv = float2((vertexID << 1) & 2, vertexID & 2);
    output.position = float4(output.uv * 2.0 - 1.0, 1.0, 1.0);
    //output.uv.y = 1.0 - output.uv.y; no need to flip Y
    
    return output;
}

[shader("fragment")]
float4 fragMain(ScreenVertex input) : SV_Target
{
    // Convert UV to NDC (clip space)
    float2 ndc = input.uv * 2.0 - 1.0;
    
    // Create points at near and far plane
    float4 nearPoint = float4(ndc, 0.0, 1.0); // Near plane (z=0 in NDC)
    float4 farPoint = float4(ndc, 1.0, 1.0);  // Far plane (z=1 in NDC)
    
    // Transform to world space
    float4 nearWorld = mul(pc.invViewProjMatrix, nearPoint);
    float4 farWorld = mul(pc.invViewProjMatrix, farPoint);
    
    // Perform perspective divide
    nearWorld.xyz /= nearWorld.w;
    farWorld.xyz /= farWorld.w;
    
    // Create direction vector from camera to far plane
    float3 rayDirection = normalize(farWorld.xyz - nearWorld.xyz);
    
    // Sample cubemap
    uint width, height, mipLevels;
    cubemaps[pc.skyboxIndex].GetDimensions(0, width, height, mipLevels);
    float maxMipLevel = float(mipLevels - 1);
    return cubemaps[pc.skyboxIndex].SampleLevel(samplers[0], rayDirection,maxMipLevel);
}