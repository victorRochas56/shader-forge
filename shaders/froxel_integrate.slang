// Froxel volumetrics — Pass D: integration (1 thread per x,y column).
// Marches z front-to-back through scatterVol, accumulating in-scattered light and transmittance,
// and writes per-slice (accumInScatter.rgb, transmittance) into integratedVol. This is what the
// composite (Pass E) samples.
import modules.froxel;   // sliceToViewZ

struct FroxelIntegratePushConstants {
    uint3 dims;
    uint scatterVolIndex;
    uint integratedVolIndex;
    float nearZ;
    float gridFar;
}

[[vk::push_constant]] ConstantBuffer<FroxelIntegratePushConstants> ipc;

[[vk::binding(4, 0)]]
RWTexture3D<float4> storageImages3D[];

[NumThreads(8, 8, 1)]
[shader("compute")]
void integrateMain(uint3 id : SV_DispatchThreadID) {
    if (id.x >= ipc.dims.x || id.y >= ipc.dims.y) return;

    float3 accum = float3(0);
    float  transmittance = 1.0;

    float zNear = sliceToViewZ(0.0, ipc.dims.z, ipc.nearZ, ipc.gridFar);
    for (uint z = 0; z < ipc.dims.z; z++) {
        uint3 c = uint3(id.x, id.y, z);
        float4 s = storageImages3D[ipc.scatterVolIndex][int3(c)]; // (inScatter.rgb, extinction)

        float zFar = sliceToViewZ(float(z + 1), ipc.dims.z, ipc.nearZ, ipc.gridFar);
        float thickness = zFar - zNear;
        zNear = zFar;

        float t = exp(-s.a * thickness);
        // Energy-conserving analytic integration of constant in-scatter over the slab.
        float3 sIntegrated = (s.rgb - s.rgb * t) / max(s.a, 1e-5);
        accum += transmittance * sIntegrated;
        transmittance *= t;

        storageImages3D[ipc.integratedVolIndex][int3(c)] = float4(accum, transmittance);
    }
}
