// Froxel volumetrics — Pass C: light scattering (1 thread/froxel).
// Reads mediaVol (scattering.rgb, extinction), runs the light loop once per froxel (the same
// shadow + phase math the per-pixel march used), and writes lit (inScatter, extinction) to scatterVol.
import modules.common;   // Light, LightType, calculateShadow, samplers/textures/shadowCmpSampler
import modules.froxel;   // froxelToWorld, phaseHG

// Per-froxel HG asymmetry g is the density-weighted average of each Volume's `phase` (injected by
// pass A) and each volumetric particle's emitter `phase` (added by pass B gather), normalized here.
// 0 = isotropic; >0 = forward scattering (god rays).

struct FroxelLightPushConstants {
    Light* lights;
    uint lightCount;
    uint shadowAtlasIndex;
    uint3 dims;
    uint mediaVolIndex;
    uint scatterVolIndex;
    float nearZ;
    float gridFar;
    uint mediaPhaseIndex;
    float3 cameraPos;
    uint debugMode;
    float3 cameraForward;
    uint frame;
    float4x4 invViewProjection;
}

static const uint DBG_SHADOW = 7; // matches volumetrics_apply.slang / the GUI combo

[[vk::push_constant]] ConstantBuffer<FroxelLightPushConstants> lpc;

[[vk::binding(4, 0)]]
RWTexture3D<float4> storageImages3D[];
[[vk::binding(4, 0)]]
RWTexture3D<float> storageImagesR[];

[NumThreads(4, 4, 4)]
[shader("compute")]
void lightScatterMain(uint3 id : SV_DispatchThreadID) {
    if (any(id >= lpc.dims)) return;

    // Shadow probe: write (1 - shadow) of the first shadow-casting light at EVERY froxel, ignoring
    // media/phase/integration. The composite samples scatterVol directly in this mode. white = lit,
    // black = shadowed. Lets us see exactly what calculateShadow returns for froxel (air) positions.
    if (lpc.debugMode == DBG_SHADOW) {
        float3 wpos = froxelToWorld(id, lpc.dims, lpc.nearZ, lpc.gridFar, lpc.cameraPos, lpc.cameraForward, lpc.invViewProjection, froxelZJitter(id.xy,lpc.frame));
        float shadow = 0.0;
        for (uint li = 0; li < lpc.lightCount; li++) {
            Light* light = &lpc.lights[li];
            if (light.intensity <= 0.0 || light.castsShadows != 1) continue;
            shadow = calculateShadow(wpos, lpc.cameraPos, lpc.cameraForward, float3(0), float3(0), light, lpc.shadowAtlasIndex);
            break;
        }
        storageImages3D[lpc.scatterVolIndex][int3(id)] = float4(float3(1.0 - shadow), 1.0);
        return;
    }

    float4 media = storageImages3D[lpc.mediaVolIndex][int3(id)]; // (scattering.rgb, extinction)

    float3 inscatter = float3(0);
    // Only bother lighting froxels that actually hold media.
    if (media.a > 0.0) {
        // Density-weighted phase from volumes (pass A) and particles (pass B), normalized by total density.
        float g = storageImagesR[lpc.mediaPhaseIndex][int3(id)] / max(media.a, 1e-5);

        float3 wpos   = froxelToWorld(id, lpc.dims, lpc.nearZ, lpc.gridFar, lpc.cameraPos, lpc.cameraForward, lpc.invViewProjection, froxelZJitter(id.xy,lpc.frame));
        float3 rayDir = normalize(wpos - lpc.cameraPos); // camera -> froxel (for the phase cosine)

        for (uint li = 0; li < lpc.lightCount; li++) {
            Light* light = &lpc.lights[li];
            if (light.intensity <= 0.0) continue;

            float3 toLight = -light.direction;
            if (light.type == LightType::Point) {
                toLight = light.position - wpos;
                if (length(toLight) > light.range) continue;
            }
            float3 lightDir = normalize(toLight);

            float shadow = 0.0;
            if (light.castsShadows == 1) {
                shadow = calculateShadow(wpos, lpc.cameraPos, lpc.cameraForward, float3(0), float3(0), light, lpc.shadowAtlasIndex);
            }

            float cosTheta = dot(rayDir, lightDir);
            inscatter += (1.0 - shadow) * light.color.rgb * light.intensity * phaseHG(g, cosTheta);
        }
    }

    // Modulate radiance by the froxel's scattering albedo; carry extinction through for integration.
    storageImages3D[lpc.scatterVolIndex][int3(id)] = float4(inscatter * media.rgb, media.a);
}
