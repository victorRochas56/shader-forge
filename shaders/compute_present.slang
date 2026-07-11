// Fullscreen blit of the compute output onto the swapchain (Option B present stage).

struct PushConstants {
    uint textureIndex; // sampled view of the compute output image
    uint samplerIndex;
}

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[[vk::binding(0,0)]]
Texture2D textures[];

[[vk::binding(2,0)]]
SamplerState samplers[];

struct VertexOutput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

[shader("vertex")]
VertexOutput vertMain(uint vertexID : SV_VertexID) {
    VertexOutput output;
    output.texCoord = float2((vertexID << 1) & 2, vertexID & 2);
    output.position = float4(output.texCoord * 2.0 - 1.0, 0.0, 1.0);
    return output;
}

[shader("fragment")]
float4 fragMain(VertexOutput input) : SV_Target0 {
    return textures[pc.textureIndex].SampleLevel(samplers[pc.samplerIndex], input.texCoord, 0);
}
