// Voxel volume resolve. resolveMain unpacks the two atomic scatter buffers into mip 0 of the RGBA16F
// volume; downsampleMain folds one level into the next. Both write through single-mip storage slots,
// so the read and write subresources are always disjoint and the whole chain can stay in eGeneral.
//
// The fold is opacity-weighted, not a box filter: a plain average of a 2x2x2 neighbourhood pulls
// empty voxels into occupied ones and the chain fades to black as you climb it.
import modules.voxel;

struct VoxelResolvePushConstants {
    uint*  voxelAlbedo;
    uint*  voxelRadiance;
    uint   dstStorageIndex;
    uint   srcStorageIndex;
    uint   dstResolution;
    uint   voxelResolution;
}

[[vk::push_constant]] ConstantBuffer<VoxelResolvePushConstants> vpc;

[[vk::binding(4, 0)]]
RWTexture3D<float4> storageImages3D[];

// (radiance.rgb, coverage). Coverage is binary at mip 0 and becomes a partial-occupancy fraction as
// the chain folds — that fraction is what the cone tracer integrates as opacity.
[NumThreads(4, 4, 4)]
[shader("compute")]
void resolveMain(uint3 id : SV_DispatchThreadID) {
    if (any(id >= vpc.voxelResolution)) return;

    uint idx = voxelIndex(id, vpc.voxelResolution);
    uint packedAlbedo   = vpc.voxelAlbedo[idx];
    uint packedRadiance = vpc.voxelRadiance[idx];

    if (!voxelOccupied(packedAlbedo)) {
        storageImages3D[vpc.dstStorageIndex][int3(id)] = float4(0, 0, 0, 0);
        return;
    }

    float3 albedo   = unpackVoxel(packedAlbedo);
    float3 radiance = voxelToneMapInv(unpackVoxel(packedRadiance));
    // Albedo modulates the stored radiance so a single bounce off this voxel is already tinted.
    storageImages3D[vpc.dstStorageIndex][int3(id)] = float4(radiance * albedo, 1.0);
}

[NumThreads(4, 4, 4)]
[shader("compute")]
void downsampleMain(uint3 id : SV_DispatchThreadID) {
    if (any(id >= vpc.dstResolution)) return;

    uint3 base = id * 2;
    float3 weightedColor = float3(0);
    float  coverage = 0.0;

    [unroll]
    for (uint i = 0; i < 8; i++) {
        uint3 offset = uint3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        float4 s = storageImages3D[vpc.srcStorageIndex][int3(base + offset)];
        weightedColor += s.rgb * s.a;
        coverage += s.a;
    }

    // Colour is the coverage-weighted mean of the occupied children; alpha is the mean occupancy over
    // all eight, so a half-filled parent reads as half opaque.
    float3 color = coverage > 0.0 ? weightedColor / coverage : float3(0);
    storageImages3D[vpc.dstStorageIndex][int3(id)] = float4(color, coverage * 0.125);
}
