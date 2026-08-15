import modules.common;

struct VertOutput {
    float4 position    : SV_Position;
    float4 currentClip : CURRENT_CLIP;
    float4 prevClip    : PREV_CLIP;
    float2 texCoord    : TEXCOORD0;
    nointerpolation uint instanceID : INSTANCE_ID;
}

[shader("vertex")]
VertOutput vertMain(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID, uint drawID : SV_DrawIndex)
{
    VertOutput out;
    LitMeshDrawData mesh = pc.litMeshDrawData[drawID];
    uint absInstance = mesh.firstInstance + instanceID;
    LitInstanceData inst = pc.litInstanceData[absInstance];
    Vertex vert = loadVertex(pc.vertexBuffer, vertexID, mesh.vertexStride, mesh.vertexOffset);

    float4x4 modelMatrix = pc.modelMatrices[inst.modelMatrixIndex];
    float4 worldPos = mul(modelMatrix, float4(vert.position, 1.0));

    out.position    = mul(pc.litPassData.viewProjection, worldPos);
    out.currentClip = out.position;
    out.prevClip    = mul(pc.litPassData.prevViewProjection, worldPos);
    out.texCoord    = vert.texCoord;
    out.instanceID  = absInstance;
    return out;
}

struct FragmentOutput
{
    float2 motionVecs : SV_Target0;
}

[shader("fragment")]
FragmentOutput fragMain(VertOutput input)
{
    LitInstanceData inst = pc.litInstanceData[input.instanceID];

    // Alpha clipping
    if (inst.materialFlags & (1u << 5)) {
        // Same base-colour alpha the lit pass applies — clipping differently here punches holes in
        // the depth buffer that the lit pass then fills.
        float alpha = textures[inst.albedoTextureIndex].Sample(samplers[pc.litPassData.samplerIndex], input.texCoord).a
                    * unpackColorRGBA8(inst.packedColor).a;
        if (alpha < inst.alphaCutoff)
            discard;
    }

    float2 currentNDC = input.currentClip.xy / input.currentClip.w;
    float2 prevNDC    = input.prevClip.xy    / input.prevClip.w;
    float2 motionVec  = currentNDC - prevNDC;

    FragmentOutput out;
    out.motionVecs = motionVec;
    return out;
}
