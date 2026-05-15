// Per-draw mesh data — vertex-fetch offsets for the mesh group, indexed by SV_DrawIndex.
struct ShadowMeshDrawData {
    uint positionBufferOffset;   // byte offset into position-only stream
    uint positionBufferStride;   // typically 12 (sizeof(float3))
    uint firstInstance;          // base offset into ShadowInstanceData[] for this draw's instances
    uint _pad1;
};

// Per-instance data — pre-baked `lightSpaceMatrix * worldTransform` for one instance/face.
// Pointer is pre-offset to the current face's instance block.
struct ShadowInstanceData {
    float4x4 MMxLSM;
};

struct ShadowPushConstants {
    float*               positionBuffer;
    ShadowInstanceData*  shadowInstanceData;
    ShadowMeshDrawData*  shadowMeshDrawData;
};

[[vk::push_constant]]
ConstantBuffer<ShadowPushConstants> pc;

[shader("vertex")]
float4 vertMain(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID, uint drawID : SV_DrawIndex) : SV_Position
{
    ShadowMeshDrawData mesh = pc.shadowMeshDrawData[drawID];
    uint absInstance = mesh.firstInstance + instanceID;
    ShadowInstanceData inst = pc.shadowInstanceData[absInstance];

    uint floatOffset = (mesh.positionBufferOffset + vertexID * mesh.positionBufferStride) / 4;
    float3 pos = ((float3*)(pc.positionBuffer + floatOffset))[0];

    return mul(inst.MMxLSM, float4(pos, 1.0));
}

// Depth-only pass — hardware writes depth automatically via SV_Position.z
[shader("fragment")]
void fragMain(float4 position : SV_Position)
{
}
