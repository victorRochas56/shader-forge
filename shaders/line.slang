struct LinePushConstants {
    float4x4 viewProjection;
};

struct LineVertex{
    float4 color;
    float4 position;
};


[[vk::push_constant]]
ConstantBuffer<LinePushConstants> pc;

[[vk::binding(0,0)]]
Texture2D textures[];

[[vk::binding(1,0)]]
TextureCube cubemaps[];

[[vk::binding(2,0)]]
SamplerState samplers[];

[[vk::binding(3,0)]]
ByteAddressBuffer vertexBuffer[];

[[vk::binding(4, 0)]]
StructuredBuffer<float4x4> modelMatrices[];

[[vk::binding(7,0)]]
StructuredBuffer<LineVertex> lineVerts;

struct VertexOutput {
    float4 position : SV_Position;
    float4 color : COLOR0;
};

[shader("vertex")]
VertexOutput vertMain(uint vertexID : SV_VertexID)
{
    VertexOutput output;
    
    // Get the line vertex data from the buffer
    LineVertex lineVert = lineVerts[vertexID];
    
    // Transform position to clip space
    output.position = mul(pc.viewProjection, lineVert.position);
    
    // Pass through color
    output.color = lineVert.color;
    
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