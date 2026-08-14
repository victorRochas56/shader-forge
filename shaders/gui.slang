import modules.common;

struct GPUGuiQuad {
    float2 minPx;
    float2 maxPx;
    float2 uvMin;
    float2 uvMax;
    float4 color;
    uint32_t textureIndex;
    uint32_t samplerIndex;
    uint32_t _pad1;
    uint32_t _pad2;
};

struct GUIPushConstants {
    GPUGuiQuad* quadBufferAddress;
    uint2 resolution;
    uint32_t quadCount;
    uint32_t _pad0;
};
[[vk::push_constant]] GUIPushConstants GUIpc;

struct VSOutput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
    nointerpolation uint instanceID : TEXCOORD1;
};

[shader("vertex")]
VSOutput vertMain(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID) {
    // Two triangles over a [0,1] unit quad; CORNERS doubles as the uv lerp factor.
    static const float2 CORNERS[6] = {
        float2(0, 0), float2(1, 0), float2(1, 1),
        float2(0, 0), float2(1, 1), float2(0, 1),
    };

    GPUGuiQuad q = GUIpc.quadBufferAddress[instanceID];
    float2 t = CORNERS[vertexID];
    float2 posPx = lerp(q.minPx, q.maxPx, t);

    // px (top-left origin, +Y down) -> NDC. Vulkan NDC is also +Y down, so no flip.
    float2 ndc = posPx / float2(GUIpc.resolution) * 2.0 - 1.0;

    VSOutput output;
    output.position   = float4(ndc, 0.0, 1.0);
    output.texCoord   = lerp(q.uvMin, q.uvMax, t);
    output.instanceID = instanceID;
    return output;
}

[shader("fragment")]
float4 fragMain(VSOutput input) : SV_Target {
    GPUGuiQuad q = GUIpc.quadBufferAddress[input.instanceID];
    return q.color * textures[q.textureIndex].Sample(samplers[q.samplerIndex], input.texCoord);
}
