// Push constants
struct Line
{
    float4x4 viewProjection;
    uint allocIndex;
    uint offset;
    uint stride;
    uint padding;
};

[[vk::push_constant]]
ConstantBuffer<Line> lineConsts;

[[vk::binding(0, 0)]]
Texture2D bindlessTextures[];

[[vk::binding(1, 0)]]
SamplerState bindlessSamplers[];

[[vk::binding(2, 0)]]
ByteAddressBuffer bindlessVertexBuffers[];

[[vk::binding(3, 0)]]
StructuredBuffer<float4x4> bindlessModelMatrices[];

struct Point
{
    float4 position;
    float4 color;
};

Point loadPoint(uint bufferIndex, uint vertexIndex, uint stride, uint offset){
    Point point;

    uint byteOffset = offset + vertexIndex * stride;
    point.position = float4(asfloat(bindlessVertexBuffers[bufferIndex].Load(byteOffset)),
                        asfloat(bindlessVertexBuffers[bufferIndex].Load(byteOffset + 4)),
                        asfloat(bindlessVertexBuffers[bufferIndex].Load(byteOffset + 8)),
                        asfloat(bindlessVertexBuffers[bufferIndex].Load(byteOffset + 12)));

    point.color = float4(asfloat(bindlessVertexBuffers[bufferIndex].Load(byteOffset+ 16)),
                        asfloat(bindlessVertexBuffers[bufferIndex].Load(byteOffset + 20)),
                        asfloat(bindlessVertexBuffers[bufferIndex].Load(byteOffset + 24)),
                        asfloat(bindlessVertexBuffers[bufferIndex].Load(byteOffset + 28)));
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
        0,
        vertexID,
        lineConsts.stride,
        lineConsts.offset
    );
    VertexOutput output;

    output.position = mul(lineConsts.viewProjection,point.position);
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