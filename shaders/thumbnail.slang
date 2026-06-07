// Mesh preview thumbnail: draw one mesh in local space with a fixed key light.
struct ThumbnailPush {
    float4x4 mvp;
    float*   vertexBuffer;
    uint     vertexStride; // bytes
    uint     vertexOffset; // bytes
};

[[vk::push_constant]] ConstantBuffer<ThumbnailPush> pc;

struct VSOut {
    float4 position : SV_Position;
    float3 normal   : NORMAL;
};

[shader("vertex")]
VSOut vertMain(uint vertexID : SV_VertexID)
{
    // Vertex layout: pos(3), normal(3), ... — matches loadVertex in modules/common.slang.
    uint floatOffset = (pc.vertexOffset + vertexID * pc.vertexStride) / 4;
    float* v = pc.vertexBuffer + floatOffset;

    VSOut o;
    o.position = mul(pc.mvp, float4(v[0], v[1], v[2], 1.0));
    o.normal   = float3(v[3], v[4], v[5]);
    return o;
}

[shader("fragment")]
float4 fragMain(VSOut input) : SV_Target0
{
    float3 N = normalize(input.normal);
    float3 L = normalize(float3(0.4, 0.8, 0.6));
    float  ndl = saturate(dot(N, L)) * 0.8 + 0.2; // lambert + ambient floor
    return float4(float3(0.82) * ndl, 1.0);
}
