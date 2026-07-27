// Voxel debug cubes. extractMain compacts occupied voxels of one mip into an instance list and bumps
// the instanceCount of an indirect draw; vertMain/fragMain then draw one shaded cube per instance,
// depth-tested against the scene. All GPU-driven — the CPU never learns how many voxels are occupied.
//
// One push-constant struct serves both stages (two [[vk::push_constant]] blocks in one module would
// collide); each entry just ignores the fields it doesn't use.

struct CubeInstance {
    uint coord; // 10|10|10 packed voxel coord — caps the grid at 1024^3
    uint color; // RGBA8, tonemapped
};

struct DrawArgs { // VkDrawIndirectCommand
    uint vertexCount;
    uint instanceCount;
    uint firstVertex;
    uint firstInstance;
};

struct VoxelCubePushConstants {
    float4x4      gridToClip;  // cameraVP * voxelCamInvVPM: grid clip space -> camera clip space
    CubeInstance* instances;
    DrawArgs*     drawArgs;
    uint          volumeTexIndex;
    uint          mipLevel;
    uint          mipRes;      // grid side at mipLevel
    float         threshold;   // min coverage (volume alpha) for a voxel to get a cube
}

[[vk::push_constant]] ConstantBuffer<VoxelCubePushConstants> cpc;

[[vk::binding(0, 0)]] Texture3D<float4> textures3D[];

[NumThreads(4, 4, 4)]
[shader("compute")]
void extractMain(uint3 id : SV_DispatchThreadID) {
    if (any(id >= cpc.mipRes)) return;

    float4 s = textures3D[cpc.volumeTexIndex].Load(int4(id, cpc.mipLevel));
    if (s.a < cpc.threshold) return;

    // Thread count is exactly mipRes^3 <= the buffer's mip-0 capacity, so slot can't overflow.
    uint slot;
    InterlockedAdd(cpc.drawArgs[0].instanceCount, 1, slot);

    float3 c = saturate(s.rgb / (1.0 + s.rgb)); // Reinhard — the volume is HDR radiance
    cpc.instances[slot].coord = id.x | (id.y << 10) | (id.z << 20);
    cpc.instances[slot].color = (uint)(c.r * 255.0 + 0.5) | ((uint)(c.g * 255.0 + 0.5) << 8) | ((uint)(c.b * 255.0 + 0.5) << 16);
}

// Cube face table: normal + one tangent per face; the bitangent is their cross product. Two triangles
// per face as a 6-vertex strip-in-place, so SV_VertexID alone addresses the whole cube (36 verts).
static const float3 FACE_N[6] = { float3( 1,0,0), float3(-1,0,0), float3(0, 1,0), float3(0,-1,0), float3(0,0, 1), float3(0,0,-1) };
static const float3 FACE_U[6] = { float3(0,1,0),  float3(0,0,1),  float3(0,0,1),  float3(1,0,0),  float3(1,0,0),  float3(0,1,0)  };
static const float2 QUAD[6]   = { float2(-1,-1), float2(1,-1), float2(-1,1), float2(-1,1), float2(1,-1), float2(1,1) };

struct CubeVSOut {
    float4 position : SV_Position;
    nointerpolation float3 color  : COLOR0;
    nointerpolation float3 normal : NORMAL;
};

[shader("vertex")]
CubeVSOut vertMain(uint vid : SV_VertexID, uint iid : SV_InstanceID) {
    CubeInstance inst = cpc.instances[iid];
    uint3 coord = uint3(inst.coord & 0x3FF, (inst.coord >> 10) & 0x3FF, (inst.coord >> 20) & 0x3FF);

    uint face = vid / 6;
    float3 n = FACE_N[face];
    float3 u = FACE_U[face];
    float3 v = cross(n, u);
    float2 q = QUAD[vid % 6];
    float3 corner = (n + q.x * u + q.y * v) * 0.5; // centered unit cube corner

    // Voxel center in grid UVW, corner offset slightly shrunk so adjacent cubes read as separate.
    float3 uvw = (float3(coord) + 0.5) / float(cpc.mipRes) + corner/ float(cpc.mipRes);

    CubeVSOut o;
    o.position = mul(cpc.gridToClip, float4(uvw.xy * 2.0 - 1.0, uvw.z, 1.0));
    o.color = float3(inst.color & 0xFF, (inst.color >> 8) & 0xFF, (inst.color >> 16) & 0xFF) * (1.0 / 255.0);
    o.normal = n;
    return o;
}

[shader("fragment")]
float4 fragMain(CubeVSOut input) : SV_Target {
    return float4(input.color, 1.0);
}
