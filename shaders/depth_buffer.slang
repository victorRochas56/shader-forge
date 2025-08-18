struct PushConstants
{
    float nearPlane;
    float farPlane;
    int linearize;
    int padding;
};

[[vk::push_constant]]
PushConstants pushConstants;

[[vk::binding(0,0)]]
Texture2D depthTexture[];

[[vk::binding(1,0)]]
SamplerState depthSampler[];

struct VertexOutput
{
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

[shader("vertex")]
VertexOutput vertMain(uint vertexID : SV_VertexID)
{
    VertexOutput output;
    
    // Generate fullscreen triangle
    output.texCoord = float2((vertexID << 1) & 2, vertexID & 2);
    output.texCoord.y = 1.0 - output.texCoord.y;
    output.position = float4(output.texCoord * 2.0 - 1.0, 0.0, 1.0);
    
    return output;
}

[shader("fragment")]
float4 fragMain(VertexOutput input) : SV_Target0
{
    float depth = depthTexture[0].Sample(depthSampler[0], input.texCoord).r;
    
    if (pushConstants.linearize != 0)
    {
        // Linearize depth for better visualization
        float linearDepth = (2.0 * pushConstants.nearPlane) / 
                           (pushConstants.farPlane + pushConstants.nearPlane - 
                            depth * (pushConstants.farPlane - pushConstants.nearPlane));
        depth = linearDepth;
    }
    
    return float4(depth, depth, depth, 1.0);
}