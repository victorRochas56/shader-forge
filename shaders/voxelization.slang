// Attachment-less voxelization. The pass renders with no colour or depth target; the fragment shader
// scatters into two uint[res^3] BDA buffers with InterlockedMax. Max over the luma-keyed packing in
// modules.voxel is order-independent, so overlapping triangles converge on the brightest contributor
// without atomics-per-channel or a dependence on draw order. voxel_resolve.slang unpacks the result.
import modules.common;
import modules.voxel;

struct VoxelizationPushConstants {
    float4x4 vpm;              // orthographic voxel-grid view-projection
    float4x4 model;            // node world transform
    float*   vertexBuffer;     // shared vertex buffer (BDA), float-addressed like loadVertex expects
    uint*    voxelAlbedo;      // uint[res^3]
    uint*    voxelRadiance;    // uint[res^3]
    uint     vertexStride;     // bytes
    uint     vertexOffset;     // bytes
    uint     albedoTextureIndex;
    uint     samplerIndex;
    uint     voxelResolution;
    uint     padding;
};

[[vk::push_constant]]
ConstantBuffer<VoxelizationPushConstants> voxelPc;

struct VoxelVSOut {
    float4 position : SV_Position;
    float3 worldPos : WORLD_POS;
    float3 normal   : NORMAL;
    float2 texCoord : TEXCOORD0;
};

[shader("vertex")]
VoxelVSOut vertMain(uint vertexID : SV_VertexID)
{
    Vertex vertex = loadVertex(voxelPc.vertexBuffer, vertexID, voxelPc.vertexStride, voxelPc.vertexOffset);

    float4 world = mul(voxelPc.model, float4(vertex.position, 1.0));

    VoxelVSOut output;
    output.position = mul(voxelPc.vpm, world);
    output.worldPos = world.xyz;
    output.normal   = normalize(mul((float3x3)voxelPc.model, vertex.normal));
    output.texCoord = vertex.texCoord;
    return output;
}

[shader("fragment")]
void fragMain(VoxelVSOut input)
{
    // SV_Position is already in grid space: xy in [0, res) pixels, z in [0, 1] under orthoRH_ZO.
    uint res = voxelPc.voxelResolution;
    uint3 coord = uint3(uint2(input.position.xy), (uint)(input.position.z * (float)res));
    if (any(coord >= res)) return;
    uint idx = voxelIndex(coord, res);

    // Fixed coarse mip: a voxel (~0.2m at 128^3 over 25m) is far bigger than a texel, so the
    // prefiltered average is both cheaper and closer to the right answer than mip 0.
    float3 albedo = textures[voxelPc.albedoTextureIndex].SampleLevel(samplers[voxelPc.samplerIndex], input.texCoord, 2).rgb;
    // TODO: radiance from the light loop, the way lit.slang does. Until then a normal-derived
    // brightness stands in, so the resolve's radiance*albedo still shows surface colour.
    float3 radiance = float3(0.5) * (0.5 + 0.5 * input.normal.y);

    InterlockedMax(voxelPc.voxelAlbedo[idx],   packVoxel(albedo));
    InterlockedMax(voxelPc.voxelRadiance[idx], packVoxel(voxelToneMap(radiance)));
}
