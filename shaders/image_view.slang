struct PushConstants
{
    uint imageIndex;
    uint samplerIndex;
    uint flags;
    float nearPlane;
    float farPlane;
    float imageAspect;
    float screenAspect;
    uint mipLevel;
};

struct Light {
    uint type;
    uint modelMatrixIndex;
    float range;
    float intensity;
    float4 color;
}

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[[vk::binding(0,0)]]
Texture2D textures[];

[[vk::binding(1,0)]]
TextureCube cubemaps[];

[[vk::binding(2,0)]]
SamplerState samplers[];

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
    if(pc.imageIndex == 0xFFFFFFFF) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }

    // Remap screen UVs to image UVs preserving aspect ratio
    float2 uv = input.texCoord;
    if(pc.flags & 1<<1) {
        uv.y = 1 - uv.y;
    }
    float ratio = pc.imageAspect / pc.screenAspect;
    if (ratio > 1.0) {
        // image wider than screen -> letterbox (bars top/bottom)
        uv.y = (uv.y - 0.5) * ratio + 0.5;
    } else {
        // image taller than screen -> pillarbox (bars left/right)
        uv.x = (uv.x - 0.5) / ratio + 0.5;
    }

    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }

    float4 color = textures[pc.imageIndex].SampleLevel(samplers[pc.samplerIndex], uv, pc.mipLevel);

    if(pc.flags & 1<<2) { // LINEARIZE_DEPTH
        float z = color.r;
        float linearDepth = pc.nearPlane * pc.farPlane / (pc.farPlane + z * (pc.nearPlane - pc.farPlane));
        float normalized = (linearDepth - pc.nearPlane) / (pc.farPlane - pc.nearPlane);
        return float4(linearDepth,linearDepth,linearDepth, 1.0);
    }
    if(pc.flags & 1) { //BW
        return float4(color.r,color.r,color.r, 1.0);
    }

    return color;
}