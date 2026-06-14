import modules.common;

struct HiZPushConstants {
    uint inputTextureIndex;
    uint samplerIndex;
    uint inputMipLevel;
    uint reduceMode; // 0 = copy from depth, 1 = 2x2 min reduction
    uint2 inputResolution;
    uint padding0;
    uint padding1;
};

[[vk::push_constant]] HiZPushConstants hizPC;

struct VSOutput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

[shader("vertex")]
VSOutput vertMain(uint vertexID : SV_VertexID) {
    VSOutput output;
    float2 uv = float2((vertexID << 1) & 2, vertexID & 2);
    output.texCoord = uv;
    output.texCoord.y = output.texCoord.y;
    output.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    return output;
}

[shader("fragment")]
float4 fragMain(VSOutput input) : SV_Target {
    float2 uv = input.texCoord;

    if (hizPC.reduceMode == 0) {
        // Copy mode: sample depth resolve texture directly
        float depth = textures[hizPC.inputTextureIndex].Sample(
            samplers[hizPC.samplerIndex], uv).r;
        return float4(depth, 0, 0, 1);
    } else {
        // 2x2 min reduction via integer Load (exact texels, no UV drift on odd-sized mips).
        int2 inSize = int2(hizPC.inputResolution);
        int mip = int(hizPC.inputMipLevel);
        int2 base = int2(input.position.xy) * 2;   // this output texel maps to a 2x2 input block

        float minDepth = textures[hizPC.inputTextureIndex].Load(int3(min(base + int2(0,0), inSize - 1), mip)).r;
        minDepth = min(minDepth, textures[hizPC.inputTextureIndex].Load(int3(min(base + int2(1,0), inSize - 1), mip)).r);
        minDepth = min(minDepth, textures[hizPC.inputTextureIndex].Load(int3(min(base + int2(0,1), inSize - 1), mip)).r);
        minDepth = min(minDepth, textures[hizPC.inputTextureIndex].Load(int3(min(base + int2(1,1), inSize - 1), mip)).r);

        // Odd input dimensions drop a column/row from the 2x2 grid; fold those texels back
        // in so the reduced min stays conservative (otherwise the trace skips them -> stripes).
        bool oddX = (inSize.x & 1) != 0;
        bool oddY = (inSize.y & 1) != 0;
        if (oddX) {
            minDepth = min(minDepth, textures[hizPC.inputTextureIndex].Load(int3(min(base + int2(2,0), inSize - 1), mip)).r);
            minDepth = min(minDepth, textures[hizPC.inputTextureIndex].Load(int3(min(base + int2(2,1), inSize - 1), mip)).r);
        }
        if (oddY) {
            minDepth = min(minDepth, textures[hizPC.inputTextureIndex].Load(int3(min(base + int2(0,2), inSize - 1), mip)).r);
            minDepth = min(minDepth, textures[hizPC.inputTextureIndex].Load(int3(min(base + int2(1,2), inSize - 1), mip)).r);
        }
        if (oddX && oddY) {
            minDepth = min(minDepth, textures[hizPC.inputTextureIndex].Load(int3(min(base + int2(2,2), inSize - 1), mip)).r);
        }
        return float4(minDepth, 0, 0, 1);
    }
}