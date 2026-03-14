import modules.common;

struct SSRPushConstants {
    float4x4 invProjection;
    uint depthIndex;
    uint depthSamplerIndex;
    uint noiseIndex;
    uint noiseSamplerIndex;
    uint2 resolution;
    float radius;
    float bias;
    float power;
    uint kernelSize;
};

[[vk::push_constant]] SSRPushConstants SSRpc;

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
    float2 texelSize = 1.0 / float2(SSRpc.resolution);
    float2 uv = input.texCoord;

    float depth = textures[SSRpc.depthIndex].Sample(samplers[SSRpc.depthSamplerIndex],uv ).r;
    float3 fragPos = reconstructViewPos(uv,depth,SSRpc.invProjection);
    float3 normal = calculateViewSpaceNormal(SSRpc.depthIndex,SSRpc.depthSamplerIndex,SSRpc.invProjection,texelSize,uv,fragPos);
    return float4(normal * 0.5 + 0.5, 1);
}