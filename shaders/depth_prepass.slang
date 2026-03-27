import modules.common;

struct VertOutput {
    float4 position    : SV_Position;
    float4 currentClip : CURRENT_CLIP;
    float4 prevClip    : PREV_CLIP;
    float2 texCoord    : TEXCOORD0;
    nointerpolation uint drawID : DRAW_ID;
}

[shader("vertex")]
VertOutput vertMain(uint vertexID : SV_VertexID, uint drawID : SV_DrawIndex)
{
    VertOutput out;
    LitDrawData draw = pc.litDrawData[drawID];
    Vertex vert = loadVertex(pc.vertexBuffer, vertexID, draw.vertexStride, draw.vertexOffset);

    float4x4 modelMatrix = pc.modelMatrices[draw.modelMatrixIndex];
    float4 worldPos = mul(modelMatrix, float4(vert.position, 1.0));

    out.position    = mul(pc.litPassData.viewProjection, worldPos);
    out.currentClip = out.position;
    out.prevClip    = mul(pc.litPassData.prevViewProjection, worldPos);
    out.texCoord    = vert.texCoord;
    out.drawID      = drawID;
    return out;
}

struct FragmentOutput
{
    float2 motionVecs : SV_Target0;
}

[shader("fragment")]
FragmentOutput fragMain(VertOutput input)
{
    LitDrawData draw = pc.litDrawData[input.drawID];

    // Alpha clipping
    if (draw.materialFlags & (1u << 5)) {
        float alpha = textures[draw.albedoTextureIndex].Sample(samplers[pc.litPassData.samplerIndex], input.texCoord).a;
        if (alpha < draw.alphaCutoff)
            discard;
    }

    float2 currentNDC = input.currentClip.xy / input.currentClip.w;
    float2 prevNDC    = input.prevClip.xy    / input.prevClip.w;
    float2 motionVec  = currentNDC - prevNDC;

    FragmentOutput out;
    out.motionVecs = motionVec;
    return out;
}
