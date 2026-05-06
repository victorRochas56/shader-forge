// Per-draw data — `MMxLSM` is `lightSpaceMatrix * modelMatrix` baked CPU-side per face/cascade.
// shadowDrawData is pre-offset to the current frame slot AND the current face block.
struct ShadowDrawData {
    uint     positionBufferOffset;   // byte offset into position-only stream
    uint     positionBufferStride;   // typically 12 (sizeof(float3))
    uint     _pad0;
    uint     _pad1;
    float4x4 MMxLSM;
};

struct ShadowPushConstants {
    float*          positionBuffer;
    ShadowDrawData* shadowDrawData;
};

[[vk::push_constant]]
ConstantBuffer<ShadowPushConstants> pc;

[shader("vertex")]
float4 vertMain(uint vertexID : SV_VertexID, uint drawID : SV_DrawIndex) : SV_Position
{
    ShadowDrawData drawData = pc.shadowDrawData[drawID];

    uint floatOffset = (drawData.positionBufferOffset + vertexID * drawData.positionBufferStride) / 4;
    float3 pos = ((float3*)(pc.positionBuffer + floatOffset))[0];

    return mul(drawData.MMxLSM, float4(pos, 1.0));
}

// Depth-only pass — hardware writes depth automatically via SV_Position.z
[shader("fragment")]
void fragMain(float4 position : SV_Position)
{
}
