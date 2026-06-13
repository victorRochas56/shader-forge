struct ExposureAdaptPushConstants {
    uint currentLumIndex;  // mipped log-luminance target
    uint currentLumMip;    // smallest mip = avg log-luminance
    uint prevAdaptedIndex; // previous adapted luminance (1x1)
    uint samplerIndex;
    float dt;
    float speed;
    uint initialized;
    uint padding;
};

[[vk::push_constant]] ExposureAdaptPushConstants pc;

[[vk::binding(0, 0)]] Texture2D textures[];
[[vk::binding(2, 0)]] SamplerState samplers[];

struct VSOutput {
    float4 position : SV_Position;
};

[shader("vertex")]
VSOutput vertMain(uint vertexID : SV_VertexID) {
    VSOutput output;
    float2 uv = float2((vertexID << 1) & 2, vertexID & 2);
    output.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    return output;
}

// Exponential eye-adaptation toward the metered luminance (frame-rate independent).
[shader("fragment")]
float fragMain(VSOutput input) : SV_Target {
    float avgLog2 = textures[pc.currentLumIndex].SampleLevel(samplers[pc.samplerIndex], float2(0.5, 0.5), (float)pc.currentLumMip).r;
    float current = exp2(avgLog2);

    if (pc.initialized == 0)
        return current;

    float prev = textures[pc.prevAdaptedIndex].SampleLevel(samplers[pc.samplerIndex], float2(0.5, 0.5), 0).r;
    float t = 1.0 - exp(-pc.dt * pc.speed);
    return lerp(prev, current, t);
}
