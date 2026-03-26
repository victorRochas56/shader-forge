import modules.common;

struct VertOutput {
    Vertex vert;
    uint drawID;
}

[shader("vertex")]
VertOutput vertMain(uint vertexID : SV_VertexID, uint drawID : SV_DrawIndex)
{
    VertOutput out;
    LitDrawData draw = pc.litDrawData[drawID];
    out.vert = loadVertex(pc.vertexBuffer, vertexID, draw.vertexStride, draw.vertexOffset);
    out.drawID = drawID;
    return out;
}

struct FragmentOutput 
{
    float4 motionVecs : SV_Target0;
}

[shader("fragment")]
float4 fragMain(VertOutput input)
{
    LitDrawData draw = pc.litDrawData[input.drawID];

    // Alpha clipping
    if (draw.materialFlags & (1u << 5)) {
        float alpha = textures[draw.albedoTextureIndex].Sample(samplers[pc.samplerIndex], input.vert.texCoord).a;
        if (alpha < draw.alphaCutoff)
            discard;
    }

    float4 out = float4(0,0,0,0);
    return out;
}