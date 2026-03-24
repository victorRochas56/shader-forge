struct SSRApplyPushConstants {
    uint samplerIndex;
    uint sceneColorIndex;
    uint sceneSamplerIndex;
    uint[8] ssrTextureIndices;
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
    output.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    return output;
}

[shader("fragment")]
float4 fragMain(VSOutput input) : SV_Target {
    float numFrames = 0;
    float4 ssr = float4(0);

    for(int i = 0; i<8 ; i++){
        if(pc.ssrTextureIndices[i] != 0xFFFFFFFF){
            ssr += textures[pc.ssrTextureIndices[i]].Sample(samplers[pc.samplerIndex], input.texCoord);
            numFrames++;
        }
    }
    ssr = ssr / numFrames;
    float3 sceneColor = textures[pc.sceneColorIndex].Sample(samplers[pc.sceneSamplerIndex], input.texCoord).rgb;
    float3 result = lerp(sceneColor, ssr.rgb, ssr.a);
    return float4(result, 1.0);
}
