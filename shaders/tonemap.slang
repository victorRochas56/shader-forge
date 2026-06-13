struct TonemapPushConstants {
    uint hdrTextureIndex;
    uint samplerIndex;
    float exposure;        // manual exposure factor (used when autoExposure == 0)
    uint op;               // 0 = Reinhard, 1 = ACES, 2 = none (clamp)
    uint autoExposure;     // 0/1
    uint lumTextureIndex;  // metering source (mipped)
    uint lumMipLevel;      // smallest mip ≈ scene average
    float exposureComp;
    float minEV;
    float maxEV;
};

[[vk::push_constant]] TonemapPushConstants pc;

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
    // No Y-flip: this samples the composite 1:1, replacing the old colorRe5solve->swapchain copy.
    output.texCoord = uv;
    output.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    return output;
}

// Narkowicz 2015 ACES approximation.
float3 acesFilm(float3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

[shader("fragment")]
float4 fragMain(VSOutput input) : SV_Target {
    float3 hdr = textures[pc.hdrTextureIndex].Sample(samplers[pc.samplerIndex], input.texCoord).rgb;

    // Single exposure application for the whole frame, then the tonemap curve.
    float exposure = pc.exposure;
    if (pc.autoExposure != 0) {
        // Adapted scene luminance (geometric mean, temporally smoothed) lives in the 1x1 texture's R channel.
        float L = max(textures[pc.lumTextureIndex].SampleLevel(samplers[pc.samplerIndex], float2(0.5, 0.5), (float)pc.lumMipLevel).r, 1e-4);
        // Saturation-based metering: EV100 = log2(L * S/K), S=100, K=12.5.
        float ev = log2(L * 8.0) - pc.exposureComp;
        ev = clamp(ev, pc.minEV, pc.maxEV);
        exposure = 1.0 / (1.2 * exp2(ev));
    }
    hdr *= exposure;

    float3 mapped;
    if (pc.op == 0)
        mapped = hdr / (hdr + 1.0);          // Reinhard
    else if (pc.op == 1)
        mapped = acesFilm(hdr);              // ACES
    else
        mapped = saturate(hdr);              // none

    // Output linear; the sRGB swapchain format applies the OETF on write.
    return float4(mapped, 1.0);
}
