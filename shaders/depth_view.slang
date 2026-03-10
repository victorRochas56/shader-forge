struct PushConstants
{
    uint imageIndex;
    uint samplerIndex;
    uint flags;
    float nearPlane;
    float farPlane;
    float imageAspect;
    float screenAspect;
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
    if(pc.imageIndex == 0xFFFFFFFF) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }

    // Remap screen UVs to image UVs preserving aspect ratio
    float2 uv = input.texCoord;
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

    float4 color = textures[pc.imageIndex].Sample(samplers[pc.samplerIndex], uv);

    if(pc.flags & 1) { // LINEARIZE_DEPTH
        float depth = color.r;
        float linearDepth = (2.0 * pc.nearPlane) /
                            (pc.farPlane + pc.nearPlane -
                            depth * (pc.farPlane - pc.nearPlane));
        depth = pow(linearDepth, 0.5);
        return float4(depth, depth, depth, 1.0);
    }

    return color;
}