// Shared encoding for the voxel scatter buffers (voxelization.slang -> voxel_resolve.slang).
//
// One uint per voxel per channel. The top byte is a luminance key, so InterlockedMax orders samples
// by brightness rather than by whichever bit pattern happens to be numerically larger — the result is
// the brightest contributor to that voxel, independent of draw order. Zero means empty; that makes a
// pure-black surface indistinguishable from no surface, which is the price of not carrying a separate
// occupancy buffer.
module voxel;

// Radiance is HDR, so it goes through Reinhard on the way in and the inverse on the way out. Values
// well above ~10 compress hard — raise the knee if the scene has very bright emissives.
public float3 voxelToneMap(float3 v)    { return v / (1.0 + v); }
public float3 voxelToneMapInv(float3 v) { return v / max(1.0 - v, 1e-4); }

public uint packVoxel(float3 color) {
    float3 c = saturate(color);
    uint luma = (uint)(saturate(dot(c, float3(0.2126, 0.7152, 0.0722))) * 255.0 + 0.5);
    uint3 rgb = (uint3)(c * 255.0 + 0.5);
    // Keep the key non-zero for anything that isn't pure black, so occupancy survives dim surfaces.
    if (luma == 0 && any(rgb > 0)) luma = 1;
    return (luma << 24) | (rgb.r << 16) | (rgb.g << 8) | rgb.b;
}

public float3 unpackVoxel(uint packed) {
    return float3((packed >> 16) & 0xFF, (packed >> 8) & 0xFF, packed & 0xFF) * (1.0 / 255.0);
}

public bool voxelOccupied(uint packed) { return packed != 0; }

public uint voxelIndex(uint3 coord, uint resolution) {
    return coord.x + resolution * (coord.y + resolution * coord.z);
}
