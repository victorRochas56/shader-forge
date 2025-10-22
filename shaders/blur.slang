// Generic blur shader for post-processing
// Performs separable Gaussian blur (horizontal or vertical)

struct BlurPushConstants {
    uint inputTextureIndex;
    uint samplerIndex;
    int isHorizontal;
    float blurRadius;
    uint2 resolution;
    uint padding1;
    uint padding2;
};

[[vk::push_constant]] BlurPushConstants pc;

[[vk::binding(0, 0)]] Texture2D textures[];
[[vk::binding(2, 0)]] SamplerState samplers[];

struct VSOutput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

[shader("vertex")]
VSOutput vertMain(uint vertexID : SV_VertexID) {
    VSOutput output;
    // Fullscreen triangle trick
    float2 uv = float2((vertexID << 1) & 2, vertexID & 2);
    output.texCoord = uv;
    output.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    return output;
}

[shader("fragment")]
float4 fragMain(VSOutput input) : SV_Target {
    float2 texelSize = 1.0 / float2(pc.resolution);
    float4 result = float4(0.0);
    float totalWeight = 0.0;

    // Gaussian blur weights for 5-tap kernel
    // These weights sum to approximately 1.0
    const int kernelRadius = 4;
    const float weights[5] = { 0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216 };

    if (pc.isHorizontal != 0) {
        // Horizontal blur
        for (int i = -kernelRadius; i <= kernelRadius; i++) {
            float2 offset = float2(float(i) * texelSize.x * pc.blurRadius, 0.0);
            int weightIdx = abs(i);
            if (weightIdx < 5) {
                float4 sample = textures[pc.inputTextureIndex].Sample(
                    samplers[pc.samplerIndex],
                    input.texCoord + offset
                );
                result += sample * weights[weightIdx];
                totalWeight += weights[weightIdx];
            }
        }
    } else {
        // Vertical blur
        for (int i = -kernelRadius; i <= kernelRadius; i++) {
            float2 offset = float2(0.0, float(i) * texelSize.y * pc.blurRadius);
            int weightIdx = abs(i);
            if (weightIdx < 5) {
                float4 sample = textures[pc.inputTextureIndex].Sample(
                    samplers[pc.samplerIndex],
                    input.texCoord + offset
                );
                result += sample * weights[weightIdx];
                totalWeight += weights[weightIdx];
            }
        }
    }

    // Normalize by total weight to maintain brightness
    return result / totalWeight;
}
