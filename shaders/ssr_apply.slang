struct SSRApplyPushConstants {
    uint ssrTextureIndex;
    uint samplerIndex;
    uint sceneColorIndex;
    uint sceneSamplerIndex;
};

[[vk::push_constant]] SSRApplyPushConstants pc;

[[vk::binding(0, 0)]] Texture2D textures[];
[[vk::binding(2, 0)]] SamplerState samplers[];

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
    float4 ssr = textures[pc.ssrTextureIndex].Sample(samplers[pc.samplerIndex], input.texCoord);
    float3 sceneColor = textures[pc.sceneColorIndex].Sample(samplers[pc.sceneSamplerIndex], float2(input.texCoord.x, 1.0 - input.texCoord.y)).rgb;
    return float4(sceneColor + ssr.rgb, 1.0);
}
