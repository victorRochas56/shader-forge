// Attachment-less voxelization. The pass renders with no colour or depth target; the fragment shader
// scatters into two uint[res^3] BDA buffers with InterlockedMax. Max over the luma-keyed packing in
// modules.voxel is order-independent, so overlapping triangles converge on the brightest contributor
// without atomics-per-channel or a dependence on draw order. voxel_resolve.slang unpacks the result.
//
// The geometry shader does dominant-axis selection: each triangle is rasterized along the grid axis
// its face normal is most aligned with, so triangles edge-on to any one sweep direction still cover
// fragments. The FS ignores SV_Position and derives the voxel coord from the interpolated unswizzled
// grid position, so no per-axis unswizzle is needed.
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
    float3 gridPos  : GRID_POS; // vpm clip space: xy in [-1,1], z in [0,1] (ortho, w = 1)
    float3 normal   : NORMAL;
    float2 texCoord : TEXCOORD0;
};

struct VoxelGSOut {
    float4 position : SV_Position; // swizzled: dominant axis mapped to depth
    float3 gridPos  : GRID_POS;    // original grid clip, drives the voxel coord in the FS
    float3 normal   : NORMAL;
    float2 texCoord : TEXCOORD0;
};

[shader("vertex")]
VoxelVSOut vertMain(uint vertexID : SV_VertexID)
{
    Vertex vertex = loadVertex(voxelPc.vertexBuffer, vertexID, voxelPc.vertexStride, voxelPc.vertexOffset);

    float4 world = mul(voxelPc.model, float4(vertex.position, 1.0));

    VoxelVSOut output;
    output.gridPos  = mul(voxelPc.vpm, world).xyz;
    output.normal   = normalize(mul((float3x3)voxelPc.model, vertex.normal));
    output.texCoord = vertex.texCoord;
    return output;
}

// Symmetric [-1,1]^3 grid coords: uniform scale on all axes, so cross products pick the right axis.
float3 gridSym(float3 gp) { return float3(gp.xy, gp.z * 2.0 - 1.0); }

[shader("geometry")]
[maxvertexcount(3)]
void geomMain(triangle VoxelVSOut input[3], inout TriangleStream<VoxelGSOut> output)
{
    float3 s0 = gridSym(input[0].gridPos);
    float3 s1 = gridSym(input[1].gridPos);
    float3 s2 = gridSym(input[2].gridPos);

    // Dominant axis of the face normal. Ties resolve to z, which matches the old fixed sweep.
    float3 n = abs(cross(s1 - s0, s2 - s0));
    uint axis = (n.x > n.y && n.x > n.z) ? 0 : (n.y > n.z) ? 1 : 2;

    [unroll]
    for (int i = 0; i < 3; i++) {
        float3 s = gridSym(input[i].gridPos);
        // Rotate the dominant axis into depth; xy becomes the raster plane. Clipping the swizzled
        // position still bounds every component of the original, so the FS sees in-grid coords only.
        float3 sw = (axis == 0) ? s.zyx : (axis == 1) ? s.xzy : s.xyz;

        VoxelGSOut v;
        v.position = float4(sw.xy, sw.z * 0.5 + 0.5, 1.0);
        v.gridPos  = input[i].gridPos;
        v.normal   = input[i].normal;
        v.texCoord = input[i].texCoord;
        output.Append(v);
    }
    output.RestartStrip();
}

[shader("fragment")]
void fragMain(VoxelGSOut input)
{
    uint res = voxelPc.voxelResolution;
    // Same mapping the old SV_Position path produced: xy NDC -> [0,res), z already [0,1].
    float3 uvw = float3(input.gridPos.xy * 0.5 + 0.5, input.gridPos.z);
    int3 icoord = (int3)floor(uvw * (float)res);
    if (any(icoord < 0) || any(icoord >= (int)res)) return;
    uint idx = voxelIndex((uint3)icoord, res);

    // Fixed coarse mip: a voxel (~0.2m at 128^3 over 25m) is far bigger than a texel, so the
    // prefiltered average is both cheaper and closer to the right answer than mip 0.
    float3 albedo = textures[voxelPc.albedoTextureIndex].SampleLevel(samplers[voxelPc.samplerIndex], input.texCoord, 2).rgb;
    // TODO: radiance from the light loop, the way lit.slang does. Until then a normal-derived
    // brightness stands in, so the resolve's radiance*albedo still shows surface colour.
    float3 radiance = float3(0.5) * (0.5 + 0.5 * input.normal.y);

    InterlockedMax(voxelPc.voxelAlbedo[idx],   packVoxel(albedo));
    InterlockedMax(voxelPc.voxelRadiance[idx], packVoxel(voxelToneMap(radiance)));
}
