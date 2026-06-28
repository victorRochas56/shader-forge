import modules.common;

struct BillboardPushConstants {
    float4x4 viewProj;
    GPUBillboard* billboardBufferAddress;
    uint32_t billboardCount;
    uint32_t samplerIndex;
    uint2 resolution;
    uint32_t depthTextureIndex;
    uint32_t depthSamplerIndex;
    uint32_t depthTest;
    uint32_t _pad;
};

struct GPUBillboard {
    float3 position;
    uint screenSpace;
    float size;
    uint textureIndex;
    bool alphaBlend;
    float clipThreshold;
};
[[vk::push_constant]] BillboardPushConstants BBpc;

struct VSOutput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
    nointerpolation uint instanceID : TEXCOORD1;
};

[shader("vertex")]
VSOutput vertMain(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID) {
    // Two triangles: (0,1,2) and (0,2,3) over a [-0.5, 0.5] quad.
    static const float2 CORNERS[6] = {
        float2(-0.5, -0.5), float2( 0.5, -0.5), float2( 0.5,  0.5),
        float2(-0.5, -0.5), float2( 0.5,  0.5), float2(-0.5,  0.5),
    };
    static const float2 UVS[6] = {
        float2(0, 0), float2(1, 0), float2(1, 1),
        float2(0, 0), float2(1, 1), float2(0, 1),
    };

    GPUBillboard bb = BBpc.billboardBufferAddress[instanceID];
    float aspect = float(BBpc.resolution.y) / float(BBpc.resolution.x);
    float2 corner = CORNERS[vertexID] * bb.size;
    corner.x *= aspect;

    float4 clip = mul(BBpc.viewProj, float4(bb.position, 1.0));

    // Offset in clip space so the quad always faces the camera.
    // screenSpace == 1: multiply by clip.w so the NDC offset is constant after
    // perspective divide -> constant on-screen size regardless of depth.
    // screenSpace == 0: leave in clip space -> divides by w -> shrinks with distance (world-ish).
    float wScale = bb.screenSpace != 0 ? clip.w : 1.0;
    clip.xy += corner * wScale;
    VSOutput output;
    output.position   = clip;
    output.texCoord   = UVS[vertexID];
    output.instanceID = instanceID;
    return output;
}

[shader("fragment")]
float4 fragMain(VSOutput input) : SV_Target {
    if (BBpc.depthTest != 0) {
        uint depthW, depthH;
        textures[BBpc.depthTextureIndex].GetDimensions(depthW, depthH);
        float2 screenUV = input.position.xy / float2(depthW, depthH);
        float sceneDepth = textures[BBpc.depthTextureIndex].Sample(samplers[BBpc.depthSamplerIndex], screenUV).r;
        if (input.position.z < sceneDepth) discard; // reverse-Z: behind scene = smaller z
    }
    return textures[BBpc.billboardBufferAddress[input.instanceID].textureIndex].Sample(samplers[BBpc.samplerIndex],input.texCoord);
}