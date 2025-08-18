struct SkyBoxConstants
{
    uint skyboxIndex;
    uint padding1;
    uint padding2;
    uint padding3;
    float4x4 invViewProjMatrix;

};

[[vk::push_constant]]
ConstantBuffer<SkyBoxConstants> skyboxConsts;

[[vk::binding(0, 0)]]
Texture2D bindlessTextures[];

[[vk::binding(1, 0)]]
SamplerState bindlessSamplers[];

[[vk::binding(2, 0)]]
ByteAddressBuffer bindlessVertexBuffers[];

[[vk::binding(3, 0)]]
StructuredBuffer<float4x4> bindlessModelMatrices[];

[[vk::binding(5, 0)]]
TextureCube cubemaps[];

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
    float4 nearWorld = mul(skyboxConsts.invViewProjMatrix, nearPoint);
    float4 farWorld = mul(skyboxConsts.invViewProjMatrix, farPoint);
    
    // Perform perspective divide
    nearWorld.xyz /= nearWorld.w;
    farWorld.xyz /= farWorld.w;
    
    // Create direction vector from camera to far plane
    float3 rayDirection = normalize(farWorld.xyz - nearWorld.xyz);
    
    // Sample cubemap
    return cubemaps[skyboxConsts.skyboxIndex].Sample(bindlessSamplers[0], rayDirection);
}