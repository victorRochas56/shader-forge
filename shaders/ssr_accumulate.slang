struct SSRAccumulatePushConstants {
    uint currentSSRIndex;
    uint historySSRIndex;
    uint motionVectorIndex;
    uint samplerIndex;
    float temporalBlend;
    uint historyValid;
    uint padding[2];
};

[[vk::push_constant]] SSRAccumulatePushConstants pc;
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
    float2 uv = input.texCoord;

    float4 current = textures[pc.currentSSRIndex].Sample(samplers[pc.samplerIndex], uv);

    // No valid history: use current frame directly
    if (pc.historyValid == 0)
        return current;

    // Reproject using motion vectors (NDC space: current - prev, scale by 0.5 for UV)
    float2 motionNDC = textures[pc.motionVectorIndex].Sample(samplers[pc.samplerIndex], uv).rg;
    float2 historyUV = uv - motionNDC * 0.5;

    // Out-of-bounds: fully disoccluded
    bool disoccluded = any(historyUV < 0.0) || any(historyUV > 1.0);

    float4 history = textures[pc.historySSRIndex].Sample(samplers[pc.samplerIndex], historyUV);

    // Neighborhood clamping (3x3) to reject stale history and prevent ghosting
    uint w, h;
    textures[pc.currentSSRIndex].GetDimensions(w, h);
    float2 texelSize = 1.0 / float2(w, h);

    float4 minColor = current;
    float4 maxColor = current;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            if (x == 0 && y == 0) continue;
            float4 neighbor = textures[pc.currentSSRIndex].Sample(
                samplers[pc.samplerIndex], uv + float2(x, y) * texelSize);
            minColor = min(minColor, neighbor);
            maxColor = max(maxColor, neighbor);
        }
    }
    history = clamp(history, minColor, maxColor);

    float blend = disoccluded ? 1.0 : pc.temporalBlend;
    return lerp(history, current, blend);
}
