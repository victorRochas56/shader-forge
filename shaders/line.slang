// Push constants
struct Line
{
    uint allocIndex;
    uint offset;
    uint stride;
    uint padding;
    float4x4 viewProjection;
};

[[vk::push_constant]]
ConstantBuffer<Line> lineConsts;
/*
[[vk::binding(0, 0)]]
Texture2D bindlessTextures[];

[[vk::binding(1, 0)]]
SamplerState bindlessSamplers[];
*/
[[vk::binding(2, 0)]]
ByteAddressBuffer bindlessVertexBuffers[];

[[vk::binding(3, 0)]]
StructuredBuffer<float4x4> bindlessModelMatrices[];

struct Point
{
    float3 position;
    float4 color;
};

Point loadPoint(uint bufferIndex, uint vertexIndex, uint stride, uint offset){
    Point point;

    uint byteOffset = offset + vertexIndex * stride;
    point.position = float3(asfloat(bindlessVertexBuffers[bufferIndex].Load(byteOffset)),
                        asfloat(bindlessVertexBuffers[bufferIndex].Load(byteOffset + 4)),
                        asfloat(bindlessVertexBuffers[bufferIndex].Load(byteOffset + 8)));

    point.color = float4(asfloat(bindlessVertexBuffers[bufferIndex].Load(byteOffset+ 12)),
                        asfloat(bindlessVertexBuffers[bufferIndex].Load(byteOffset + 16)),
                        asfloat(bindlessVertexBuffers[bufferIndex].Load(byteOffset + 20)),
                        asfloat(bindlessVertexBuffers[bufferIndex].Load(byteOffset + 24)));
    return point;
}

struct VertexOutput
{
    float4 position : SV_Position;
    float4 color : COLOR;
    float3 worldPos : WORLD_POS;
};

[shader("vertex")]
VertexOutput vertMain(uint vertexID : SV_VertexID)
{
    // load vertex data from bindless buffer
    Point point = loadPoint(
        lineConsts.allocIndex,
        vertexID,
        lineConsts.stride,
        lineConsts.offset
    );
    VertexOutput output;

    float4 position = float4(point.position,1.0);
    output.position = mul(lineConsts.viewProjection,position);
    output.color = point.color;
    return output;
}

struct FragmentOutput
{
    float4 color : SV_Target0;
};

[shader("fragment")]
FragmentOutput fragMain(VertexOutput input)
{
    FragmentOutput output;
    output.color = input.color;
    return output;
}