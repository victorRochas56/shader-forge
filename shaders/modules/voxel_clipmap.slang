// Voxel clipmap layout shared by the GI passes (GPUVoxelClipmap in structs.hpp).
//
// Level l is a cube of extent E_0 * 2^l metres at `resolution` voxels a side, snapped independently
// around the camera: voxel size doubles per level, so each level covers twice the radius of the one
// inside it for the same voxel count. Grid axes are world axes.
//
// Storage is one 3D texture per volume with the levels stacked along Z. Level l owns the slab
// z in [l*res, (l+1)*res) at mip 0, and the slab boundaries stay 2x2x2-aligned through every mip
// because res is a power of two. Trilinear reads clamp z by half a texel so a tap never blends into
// the neighbouring level's slab; x and y still see the sampler's transparent border.
//
// Which level a point reads from is decided against the camera, not the snapped cube: a level's
// centre snaps to CLIP_SNAP_VOXELS of its voxels, so its cube jumps by that step on a recentre, and
// any choice or blend measured from the cube edge jumps with it. Half the extent minus the snap step
// is a radius around the camera that is inside the cube whatever the snap phase, and it slides
// continuously with the camera.
module voxel_clipmap;

// Slabs allocated per volume; == MAX_VOXEL_CLIP_LEVELS in constants.hpp.
public static const uint MAX_CLIP_LEVELS = 6;
// Level centres snap to this many of their own voxels; == VoxelizationPass::SNAP_VOXELS.
public static const float CLIP_SNAP_VOXELS = 8.0;

public struct VoxelClipmap {
    public float4 levels[MAX_CLIP_LEVELS]; // xyz centre, w extent (m)
    public uint   levelCount;              // active levels this frame
    public uint   resolution;              // radiance voxels a side, per level
    public uint   irradianceResolution;    // irradiance voxels a side, per level
    public uint   mipLevels;               // mips per level in the radiance volume
    public float4 anchor;                  // xyz camera position the levels were snapped around; w unused
};

public float clipVoxelSize(VoxelClipmap cm, uint level) { return cm.levels[level].w / float(cm.resolution); }

// World -> level-local UVW: [0,1]^3 inside the level's cube.
public float3 clipLevelUVW(VoxelClipmap cm, uint level, float3 worldPos) {
    float4 L = cm.levels[level];
    return (worldPos - L.xyz) / L.w + 0.5;
}

// Radius around the camera anchor (Chebyshev) inside which a level holds a point whatever its snap
// phase: half the extent minus the snap step.
public float clipLevelRadius(VoxelClipmap cm, uint level) {
    return cm.levels[level].w * (0.5 - CLIP_SNAP_VOXELS / float(cm.resolution));
}

// Chebyshev distance from the camera anchor.
public float clipAnchorDistance(VoxelClipmap cm, float3 p) {
    float3 d = abs(p - cm.anchor.xyz);
    return max(d.x, max(d.y, d.z));
}

// Finest level whose anchored radius holds p with a margin of marginFrac * extent, or the outermost
// level when none does (the caller's clamp and border handle what falls outside it).
public uint clipFinestLevel(VoxelClipmap cm, float3 p, float marginFrac) {
    uint last = cm.levelCount - 1;
    uint level = 0;
    float dist = clipAnchorDistance(cm, p);
    while (level < last && dist + marginFrac * cm.levels[level].w > clipLevelRadius(cm, level)) level++;
    return level;
}

// Level-local UVW -> stacked texture UVW. halfTexelZ is half a texel of the sampled mip in
// level-local units (0.5 * 2^mip / res): the clamp keeps the trilinear footprint inside the slab.
public float3 clipStackedUVW(float3 uvw, uint level, float halfTexelZ) {
    float z = clamp(uvw.z, halfTexelZ, 1.0 - halfTexelZ);
    return float3(uvw.xy, (z + float(level)) / float(MAX_CLIP_LEVELS));
}

// Integer texel coordinate into the stacked volume for a level-local coord on a grid `res` a side.
public uint3 clipStackedCoord(uint3 coord, uint level, uint res) {
    return uint3(coord.xy, coord.z + level * res);
}

// World-space centre of level-local voxel `coord` on a grid `res` a side.
public float3 clipVoxelCenterWorld(VoxelClipmap cm, uint level, uint res, uint3 coord) {
    float4 L = cm.levels[level];
    return L.xyz + ((float3(coord) + 0.5) / float(res) - 0.5) * L.w;
}
