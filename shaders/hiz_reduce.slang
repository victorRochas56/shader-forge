import modules.common;

struct HiZPushConstants {
    uint inputTextureIndex;
    uint samplerIndex;
    uint inputMipLevel;
    uint reduceMode; // 0 = copy from depth, 1 = 2x2 min reduction
    uint2 inputResolution;
    uint padding0;
    uint padding1;
};

[[vk::push_constant]] HiZPushConstants hizPC;

struct VSOutput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

[shader("vertex")]
VSOutput vertMain(uint vertexID : SV_VertexID) {
    VSOutput output;
    float2 uv = float2((vertexID << 1) & 2, vertexID & 2);
    output.texCoord = uv;
    output.texCoord.y = 1.0 - output.texCoord.y;
    output.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    return output;
}

[shader("fragment")]
float4 fragMain(VSOutput input) : SV_Target {
    float2 uv = input.texCoord;

    if (hizPC.reduceMode == 0) {
        // Copy mode: sample depth resolve texture directly
        float depth = textures[hizPC.inputTextureIndex].Sample(
            samplers[hizPC.samplerIndex], uv).r;
        return float4(depth, 0, 0, 1);
    } else {
        // 2x2 min reduction: sample 4 texel centers from previous mip
        float2 texelSize = 1.0 / float2(hizPC.inputResolution);
        float2 offset = texelSize * 0.5;

        float d0 = textures[hizPC.inputTextureIndex].SampleLevel(
            samplers[hizPC.samplerIndex], uv + float2(-offset.x, -offset.y), hizPC.inputMipLevel).r;
        float d1 = textures[hizPC.inputTextureIndex].SampleLevel(
            samplers[hizPC.samplerIndex], uv + float2( offset.x, -offset.y), hizPC.inputMipLevel).r;
        float d2 = textures[hizPC.inputTextureIndex].SampleLevel(
            samplers[hizPC.samplerIndex], uv + float2(-offset.x,  offset.y), hizPC.inputMipLevel).r;
        float d3 = textures[hizPC.inputTextureIndex].SampleLevel(
            samplers[hizPC.samplerIndex], uv + float2( offset.x,  offset.y), hizPC.inputMipLevel).r;

        float minDepth = min(min(d0, d1), min(d2, d3));
        return float4(minDepth, 0, 0, 1);
    }
}