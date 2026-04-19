import modules.common;

struct VolumetricPushConstants {
    Light* lights;            // offset 0
    uint lightCount;          // offset 8
    uint shadowAtlasIndex;    // offset 12
    float3 cameraPos;         // offset 16
    uint depthTextureIndex;   // offset 28
    float3 cameraDir;         // offset 32
    uint depthSamplerIndex;   // offset 44
    uint2 screenSize;         // offset 48
    uint2 _pad3;              // offset 56
    float4x4 view;            // offset 64
    float4x4 projection;      // offset 128
    float4x4 invViewProjection; // offset 192
}

[[vk::push_constant]] ConstantBuffer<VolumetricPushConstants> volPc;

struct VSOutput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

[shader("vertex")]
VSOutput vertMain(uint vertexID : SV_VertexID) {
    VSOutput output;
    float2 uv = float2((vertexID << 1) & 2, vertexID & 2);
    output.texCoord = uv;
    output.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    return output;
}

bool rayIntersectSphere(float3 rayStart, float3 rayDir, float3 center, float radius, out float3 intersectPointA, out float3 intersectPointB) {

    //ray to sphere
    float3 CL = center - rayStart;
    //ray dir
    float3 CP = rayDir;
    // parametric distance to projected center along CP
    float tCenter = dot(CL, CP) / dot(CP, CP);
    // perpendicular vector from ray to center
    float3 perp = CL - tCenter * CP;

    float r2 = radius * radius;
    float perp2 = dot(perp, perp);

    bool inside = dot(CL, CL) < r2;

    if (!inside && tCenter < 0.0) {
        intersectPointA = float3(0, 0, 0);
        intersectPointB = float3(0, 0, 0);
        return false;
    }
    if (perp2 >= r2) {
        intersectPointA = float3(0, 0, 0);
        intersectPointB = float3(0, 0, 0);
        return false;
    }

    // half-chord length in units of t (parametric along CP)
    float tHalfChord = sqrt(r2 - perp2) / sqrt(dot(CP, CP));
    // near: entry point if outside, rayStart if inside; far: exit point
    float tA = inside ? 0.0 : (tCenter - tHalfChord);
    float tB = tCenter + tHalfChord;
    intersectPointA = rayStart + tA * CP;
    intersectPointB = rayStart + tB * CP;
    return true;
}


struct FragmentOutput {
    float4 color : SV_Target0;
};

#define PI 3.1415
#define NUMSTEPS 16
#define SIGMA_A 0.5
#define SIGMA_S 0.5
#define DENSITY 1

// Sample the point-light shadow atlas at a world-space position. Returns 0.0
// when lit (or outside the shadow frustum) and 1.0 when occluded.
float samplePointShadowAt(float3 worldPos, Light* light, uint shadowAtlasIdx) {
    float3 fragToLight = worldPos - light.position;
    float3 absDir = abs(fragToLight);

    int faceIndex;
    if (absDir.x >= absDir.y && absDir.x >= absDir.z) {
        faceIndex = fragToLight.x > 0.0 ? 0 : 1;
    } else if (absDir.y >= absDir.x && absDir.y >= absDir.z) {
        faceIndex = fragToLight.y > 0.0 ? 2 : 3;
    } else {
        faceIndex = fragToLight.z > 0.0 ? 4 : 5;
    }

    PointFace face = light.pointFaces[faceIndex];
    float4 lsp = mul(face.lightSpaceMatrix, float4(worldPos, 1.0));
    float3 proj = lsp.xyz / lsp.w;
    proj.xy = proj.xy * 0.5 + 0.5;

    if (proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0 ||
        proj.z < 0.0 || proj.z > 1.0) {
        return 0.0;
    }

    float currentDepth = proj.z - 0.0005f;
    float2 rangeMin  = face.shadowAtlasRange.xy;
    float2 rangeSpan = face.shadowAtlasRange.zw - face.shadowAtlasRange.xy;
    float2 atlasUV = rangeMin + proj.xy * rangeSpan;

    float lit = textures[shadowAtlasIdx].SampleCmpLevelZero(
        shadowCmpSampler, atlasUV, currentDepth);
    return 1.0 - lit;
}

// the Henyey-Greenstein phase function
float phase(float g, float cos_theta)
{
    float denom = 1 + g * g - 2 * g * cos_theta;
    return 1 / (4 * PI) * (1 - g * g) / (denom * sqrt(denom));
}

[shader("fragment")]
FragmentOutput fragMain(VSOutput input) {
    FragmentOutput out;

    uint2 pix = uint2(input.position.xy);
    uint seed = pix.x * 1973u + pix.y * 9277u + 1u;

    float3 totalInScatter = float3(0);
    float  totalTransmittance = 1.0f;

    float proj00 = volPc.projection[0][0];
    float proj11 = volPc.projection[1][1];

    float3 pixelRayDir = reconstructPixelRayDir(input.texCoord, volPc.view, proj00, proj11);

    // Reconstruct world-space distance to the opaque surface so volumetrics
    // never scatter light from behind solid geometry.
    float2 ndc = input.texCoord * 2.0 - 1.0;
    float sceneDepth = textures[volPc.depthTextureIndex].Sample(samplers[volPc.depthSamplerIndex], input.texCoord).r;
    float4 surfaceWorld = mul(volPc.invViewProjection, float4(ndc, sceneDepth, 1.0));
    surfaceWorld.xyz /= surfaceWorld.w;
    float surfaceDist = dot(surfaceWorld.xyz - volPc.cameraPos, pixelRayDir);

    for (uint i = 0; i < volPc.lightCount; i++) {
        if (volPc.lights[i].type != 0) continue;
        Light* light = &volPc.lights[i];

        float3 hitPointA;
        float3 hitPointB;
        if (!rayIntersectSphere(volPc.cameraPos, pixelRayDir, light.position, light.range, hitPointA, hitPointB))
            continue;

        // Clip the march segment against scene depth.
        float tA = dot(hitPointA - volPc.cameraPos, pixelRayDir);
        float tB = dot(hitPointB - volPc.cameraPos, pixelRayDir);
        if (tA >= surfaceDist) continue;
        if (tB > surfaceDist) {
            tB = surfaceDist;
            hitPointB = volPc.cameraPos + tB * pixelRayDir;
        }

        float stepDist = length(hitPointB - hitPointA) / NUMSTEPS;
        float transparency = 1.0f;
        float3 result = float3(0);

        float g = 0.8; // assymetry factor of the phase function
        for (int n = 0; n <= NUMSTEPS; n++) {
            float3 sample_pos = hitPointA + (n + rand(seed)) * stepDist * pixelRayDir;
            float3 toLight = light.position - sample_pos;
            float  lightDist = length(toLight);
            float falloff = 1.0 / max(lightDist * lightDist, 1e-4);

            // Smooth density falloff: full density at the light, zero at light.range.
            float densityFalloff = saturate(1.0 - lightDist / light.range);
            densityFalloff *= densityFalloff;
            float density = DENSITY * densityFalloff;

            float sample_attenuation = exp(-stepDist * density * (SIGMA_A + SIGMA_S));
            transparency *= sample_attenuation;

            float cos_theta = dot(pixelRayDir, -normalize(toLight));

            float light_attenuation = exp(-lightDist * density * (SIGMA_A + SIGMA_S));

            float shadow = 0.0;
            if (light.castsShadows == 1) {
                shadow = samplePointShadowAt(sample_pos, light, volPc.shadowAtlasIndex);
            }

            result += (1.0 - shadow) * transparency * light.color.rgb * phase(g, cos_theta) * light.intensity * light_attenuation * falloff * stepDist;
        }

        // in-scattered light already premultiplied by its own transmittance,
        // but still needs attenuation by media in front of this sphere.
        totalInScatter += totalTransmittance * result;
        totalTransmittance *= transparency;
    }

    out.color = float4(totalInScatter, 1.0 - totalTransmittance);
    return out;
}