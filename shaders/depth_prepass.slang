import modules.common;

[shader("vertex")]
VertexOutput vertMain(uint vertexID : SV_VertexID, uint drawID : SV_DrawIndex)
{
    return litVertMain(vertexID, drawID);
}

[shader("fragment")]
void fragMain(VertexOutput input)
{
    LitDrawData draw = pc.litDrawData[input.drawID];

    // Alpha clipping
    if (draw.materialFlags & (1u << 5)) {
        float alpha = textures[draw.albedoTextureIndex].Sample(samplers[pc.samplerIndex], input.texCoord).a;
        if (alpha < draw.alphaCutoff)
            discard;
    }
}