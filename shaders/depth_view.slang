struct PushConstants
{
    uint depthIndex;
    uint depthSamplerIndex;
    uint showShadowMap;
    uint shadowMapSamplerIndex;
    float nearPlane;
    float farPlane;
    uint linearize;
    uint depthBuffer;
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

struct VertexOutput
{
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

[shader("vertex")]
VertexOutput vertMain(uint vertexID : SV_VertexID)
{
    VertexOutput output;
    
    output.texCoord = float2((vertexID << 1) & 2, vertexID & 2);
    output.texCoord.y = 1.0 - output.texCoord.y;
    output.position = float4(output.texCoord * 2.0 - 1.0, 0.0, 1.0);
    
    return output;
}

[shader("fragment")]
float4 fragMain(VertexOutput input) : SV_Target0
{
    if(pc.depthBuffer == 0 && pc.showShadowMap == 0xFFFFFFFF) { 
        discard;
    }
    
    if(pc.showShadowMap != 0xFFFFFFFF){
        float depth = textures[pc.showShadowMap].Sample(samplers[pc.shadowMapSamplerIndex], input.texCoord).r;
        return float4(depth, depth, depth, 1.0);
    }
    else{
        float depth = textures[pc.depthIndex].Sample(samplers[pc.depthSamplerIndex], input.texCoord).r;
       
        if (pc.linearize != 0)
        {
            // Linearize depth for better visualization
            float linearDepth = (2.0 * pc.nearPlane) /
                               (pc.farPlane + pc.nearPlane -
                                depth * (pc.farPlane - pc.nearPlane));
            depth = linearDepth;
        }
        depth = pow(depth, 0.5);
        return float4(depth, depth, depth, 1.0);
    }
}