// Shared froxel-grid mapping. Imported by every froxel pass + the composite.
// The grid is view-aligned: x/y map to screen tiles, z is distance along the view ray,
// distributed exponentially between the near plane and gridFar so slices bunch near the camera.
import rng;   // sibling module: no "modules." prefix, imports resolve relative to this file's dir

static const float FROXEL_PI = 3.14159265;

// Henyey-Greenstein phase function (moved from volumetrics.slang; reused by the light pass).
float phaseHG(float g, float cosTheta) {
    float denom = 1.0 + g * g - 2.0 * g * cosTheta;
    return (1.0 / (4.0 * FROXEL_PI)) * (1.0 - g * g) / (denom * sqrt(max(denom, 1e-6)));
}

// slice (0..D, fractional ok) -> view-space depth (distance along camera forward).
float sliceToViewZ(float slice, uint D, float nearZ, float gridFar) {
    return nearZ * pow(gridFar / nearZ, slice / float(D));
}

// inverse of sliceToViewZ: view-space depth -> fractional slice.
float viewZToSlice(float viewZ, uint D, float nearZ, float gridFar) {
    return float(D) * log(viewZ / nearZ) / log(gridFar / nearZ);
}

// Per-column depth jitter in slice units, to break up the banding from quantizing media into slab centers.
// One-sided toward the camera ([-1, 0], never positive): a positive jitter lets an in-front froxel
// sample density from behind the scene surface, which then composites in front of the occluding
// geometry (bleed-through). Full slice of width is kept, just shifted to the near side.
float froxelZJitter(uint2 coordXY, uint dt = 0) {
    uint seed = pcgHash(coordXY.x | (coordXY.y << 16) /*| dt*/);
    return rand(seed) - 1.0;  // [-1, 0] slice, toward camera only
}

// Froxel cell center -> world position. Rebuilds the view ray for the cell's screen UV
// (reverse-Z: near=1, far=0, matching volumetrics.slang:112-116) then walks it out to the
// slab-center view depth, offset by zJitter slices. camForward must be the normalized camera
// look direction.
float3 froxelToWorld(uint3 coord, uint3 dims, float nearZ, float gridFar, float3 camPos, float3 camForward, float4x4 invViewProj, float zJitter) {
    float2 uv  = (float2(coord.xy) + 0.5) / float2(dims.xy);
    float2 ndc = uv * 2.0 - 1.0;

    float4 nearW = mul(invViewProj, float4(ndc, 1.0, 1.0));
    float4 farW  = mul(invViewProj, float4(ndc, 0.0, 1.0));
    nearW.xyz /= nearW.w;
    farW.xyz  /= farW.w;
    float3 rayDir = normalize(farW.xyz - nearW.xyz);

    // vz is depth measured along camForward; convert to a distance along the pixel ray.
    float vz = sliceToViewZ(float(coord.z) + 0.5 + zJitter, dims.z, nearZ, gridFar);
    float t  = vz / max(dot(rayDir, camForward), 1e-4);
    return camPos + rayDir * t;
}
