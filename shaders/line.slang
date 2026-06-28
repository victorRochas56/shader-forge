struct LineVertex{
    float4 color;
    float4 position;
};

struct LinePushConstants {
    LineVertex* lineVerts;
    uint depthTextureIndex;
    uint depthSamplerIndex;
    float4x4 viewProjection;
};

[[vk::push_constant]]
ConstantBuffer<LinePushConstants> pc;

[[vk::binding(0,0)]]
Texture2D textures[];

[[vk::binding(1,0)]]
TextureCube cubemaps[];

[[vk::binding(2,0)]]
SamplerState samplers[];

struct VertexOutput {
    float4 position : SV_Position;
    float4 color : COLOR0;
};

[shader("vertex")]
VertexOutput vertMain(uint vertexID : SV_VertexID)
{
    VertexOutput output;

    LineVertex lineVert = pc.lineVerts[vertexID];

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

    uint depthW, depthH;
    textures[pc.depthTextureIndex].GetDimensions(depthW, depthH);
    float2 screenUV = input.position.xy / float2(depthW, depthH);
    float sceneDepth = textures[pc.depthTextureIndex].Sample(samplers[pc.depthSamplerIndex], screenUV).r;
    float lineDepth = input.position.z;

    float occluded = lineDepth < sceneDepth ? 1.0 : 0.0; // reverse-Z: behind scene = smaller z
    output.color = float4(input.color.rgb * lerp(1.0, 0.25, occluded), input.color.a);

    return output;
}
