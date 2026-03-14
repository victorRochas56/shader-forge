import modules.common;

struct SSAOPushConstants {
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

[[vk::push_constant]] SSAOPushConstants SSAOpc;

struct VSOutput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

// Hemisphere sample kernel — 32 samples, cosine-weighted distribution
// Pre-computed with positions inside a unit hemisphere, scaled so
// more samples cluster near the origin (importance sampling)
static const float3 KERNEL[32] = {
    float3( 0.0340,  0.0482,  0.0285),
    float3(-0.0577,  0.0198,  0.0540),
    float3( 0.0100, -0.0278,  0.0660),
    float3( 0.0714,  0.0568,  0.0390),
    float3(-0.0441, -0.0870,  0.0170),
    float3(-0.0109,  0.0197,  0.1020),
    float3( 0.1040, -0.0352,  0.0620),
    float3(-0.0870,  0.0780,  0.0870),
    float3( 0.0380,  0.1310, -0.0060),
    float3(-0.1400, -0.0210,  0.0720),
    float3( 0.0640, -0.0100,  0.1680),
    float3( 0.1290,  0.1260,  0.0290),
    float3(-0.0430, -0.1720,  0.0960),
    float3( 0.1920,  0.0500,  0.1100),
    float3(-0.1620,  0.1450,  0.0540),
    float3( 0.0300, -0.0850,  0.2340),
    float3( 0.2150,  0.0120, -0.1430),
    float3(-0.1070,  0.2480,  0.0580),
    float3( 0.1780, -0.2120,  0.0420),
    float3(-0.2700,  0.0290,  0.1520),
    float3( 0.0600,  0.2890,  0.1370),
    float3( 0.2930, -0.1100,  0.1380),
    float3(-0.2110, -0.2410,  0.0920),
    float3( 0.1250,  0.1400,  0.3120),
    float3(-0.3460,  0.1560,  0.0710),
    float3( 0.2700, -0.0420,  0.3030),
    float3(-0.0620,  0.3690,  0.1890),
    float3( 0.3760,  0.1930,  0.0870),
    float3(-0.3100, -0.2700,  0.1830),
    float3( 0.1370, -0.3620,  0.2640),
    float3( 0.4210,  0.0650,  0.2520),
    float3(-0.2400,  0.3290,  0.2940),
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
    float2 texelSize = 1.0 / float2(SSAOpc.resolution);

    // Sample depth and reconstruct view-space position
    float depth = textures[SSAOpc.depthIndex].Sample(samplers[SSAOpc.depthSamplerIndex], uv).r;
    if (depth >= 1.0) return float4(1.0); // skip skybox

    float3 fragPos = reconstructViewPos(uv,depth,SSAOpc.invProjection);
    float3 normal = calculateViewSpaceNormal(SSAOpc.depthIndex,SSAOpc.depthSamplerIndex,SSAOpc.invProjection,texelSize,uv,fragPos);

    // Sample noise texture (tiled across screen)
    float2 noiseUV = uv * float2(SSAOpc.resolution) / 4.0;
    float3 randomVec = textures[SSAOpc.noiseIndex].Sample(samplers[SSAOpc.noiseSamplerIndex], noiseUV).rgb * 2.0 - 1.0;
    randomVec.z = 0.0;
    randomVec = normalize(randomVec);

    // Build TBN matrix to orient hemisphere along surface normal
    float3 tangent = normalize(randomVec - normal * dot(randomVec, normal));
    float3 bitangent = cross(normal, tangent);
    float3x3 TBN = float3x3(tangent, bitangent, normal);

    // Offset fragment position along normal to prevent self-occlusion on flat surfaces
    float3 biasedFragPos = fragPos - normal * SSAOpc.bias;

    // Accumulate occlusion
    float occlusion = 0.0;
    uint sampleCount = min(SSAOpc.kernelSize, 32u);

    for (uint i = 0; i < sampleCount; i++) {
        // Orient sample in view space
        float3 sampleOffset = mul(KERNEL[i], TBN);
        float3 samplePos = biasedFragPos + sampleOffset * SSAOpc.radius;

        // Project sample to screen space
        // Reconstruct proj from invProj: proj[0][0] = 1/invProj[0][0], proj[1][1] = 1/invProj[1][1]
        float2 sampleUV;
        sampleUV.x = (samplePos.x / (-samplePos.z) * (1.0 / SSAOpc.invProjection[0][0])) * 0.5 + 0.5;
        sampleUV.y = (samplePos.y / (-samplePos.z) * (1.0 / SSAOpc.invProjection[1][1])) * 0.5 + 0.5;

        // Sample depth at projected position
        float sampleDepth = textures[SSAOpc.depthIndex].Sample(samplers[SSAOpc.depthSamplerIndex], sampleUV).r;
        float3 actualPos = reconstructViewPos(sampleUV, sampleDepth,SSAOpc.invProjection);

        // Range check: only occlude if the depth difference is within radius
        float rangeCheck = smoothstep(0.0, 1.0, SSAOpc.radius / abs(fragPos.z - actualPos.z));
        occlusion += (actualPos.z >= samplePos.z ? 1.0 : 0.0) * rangeCheck;
    }

    float ao = 1.0 - (occlusion / float(sampleCount));
    ao = pow(saturate(ao), SSAOpc.power);
    return float4(ao, ao, ao, 1.0);
}