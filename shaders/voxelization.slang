// Single-pass slicemap voxelization. Each fragment sets one bit of an R32_UINT target; the fixed
// -function OR in PipelineCategory::VOXELIZATION merges overlapping triangles without atomics.
import modules.common;

struct VoxelizationPushConstants {
    float4x4 vpm;              // orthographic voxel-grid view-projection
    float4x4 model;            // node world transform
    float*   vertexBuffer;     // shared vertex buffer (BDA), float-addressed like loadVertex expects
    uint     vertexStride;     // bytes
    uint     vertexOffset;     // bytes
    uint     meshIndex;
    uint     padding;
};

[[vk::push_constant]]
ConstantBuffer<VoxelizationPushConstants> voxelPc;

// Position is all the slicemap needs — no interpolants beyond SV_Position.
struct VoxelVSOut {
    float4 position : SV_Position;
};

[shader("vertex")]
VoxelVSOut vertMain(uint vertexID : SV_VertexID)
{
    Vertex vertex = loadVertex(voxelPc.vertexBuffer, vertexID, voxelPc.vertexStride, voxelPc.vertexOffset);

    VoxelVSOut output;
    output.position = mul(voxelPc.vpm, mul(voxelPc.model, float4(vertex.position, 1.0)));
    return output;
}

[shader("fragment")]
uint fragMain(VoxelVSOut input) : SV_Target0
{
    uint slice = min((uint)(input.position.z * 64.0), 63u);
    return 1u << slice;
}
