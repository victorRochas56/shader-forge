import modules.common;

enum SDFType : uint {
    SPHERE,
    CYLINDER,
    CONE,
};

struct SDF {
    float4x4 worldTransform;
    float4x4 invWorldTransform;
    float4 color;
    uint type;
    float radius;
    float height;
    float padding;
};

struct SDFPushConstants {
    SDF*      drawData;
    uint      sdfCount;
    uint      depthTextureIndex;
    uint      depthSamplerIndex;
    uint      padding;
    float3    cameraPos;
    float     padding1;
    float4x4  invViewProjection;
};

[[vk::push_constant]] ConstantBuffer<SDFPushConstants> sdfPC;

struct VSOutput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

[shader("vertex")]
VSOutput vertMain(uint vertexID : SV_VertexID) {
    VSOutput output;
    float2 uv = float2((vertexID << 1) & 2, vertexID & 2);
    output.texCoord = uv;
    output.texCoord.y = 1.0 - output.texCoord.y;
    output.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    return output;
}

// ---- SDF distance functions (operate in local space) ----

float sdSphere(float3 p, float radius) {
    return length(p) - radius;
}

float sdCylinder(float3 p, float radius, float height) {
    float2 d = float2(length(p.xz) - radius, abs(p.y) - height * 0.5);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

float sdCone(float3 p, float radius, float height) {
    float2 q = float2(length(p.xz), p.y);
    float2 tip = float2(0.0, height);
    float2 base = float2(radius, 0.0);
    float2 e = base - tip;
    float2 w = q - tip;
    float t = clamp(dot(w, e) / dot(e, e), 0.0, 1.0);
    float2 closest = tip + e * t;
    float d = length(w - (closest - tip));
    float sign = (q.x * e.y - q.y * e.x > 0.0) ? 1.0 : -1.0;
    // cap the base
    float baseDist = q.y; // distance below base plane
    float baseRadial = length(p.xz) - radius;
    float capDist = max(-baseDist, baseRadial);
    return (baseDist < 0.0) ? max(capDist, -d) : sign * d;
}

float evalSDF(float3 worldPos, SDF shape) {
    // Transform world position into the shape's local space
    float3 localPos = mul(shape.invWorldTransform, float4(worldPos, 1.0)).xyz;

    switch (shape.type) {
    case SDFType.SPHERE:
        return sdSphere(localPos, shape.radius);
    case SDFType.CYLINDER:
        return sdCylinder(localPos, shape.radius, shape.height);
    case SDFType.CONE:
        return sdCone(localPos, shape.radius, shape.height);
    default:
        return 1e10;
    }
}

// Evaluate the closest distance across all SDF shapes, return index of nearest
float sceneSDF(float3 worldPos, out uint hitIndex) {
    float minDist = 1e10;
    hitIndex = 0;
    for (uint i = 0; i < sdfPC.sdfCount; i++) {
        float d = evalSDF(worldPos, sdfPC.drawData[i]);
        if (d < minDist) {
            minDist = d;
            hitIndex = i;
        }
    }
    return minDist;
}

// Gradient-based normal at hit point
float3 calcNormal(float3 p, uint hitIndex) {
    float eps = 0.001;
    SDF shape = sdfPC.drawData[hitIndex];
    return normalize(float3(
        evalSDF(p + float3(eps, 0, 0), shape) - evalSDF(p - float3(eps, 0, 0), shape),
        evalSDF(p + float3(0, eps, 0), shape) - evalSDF(p - float3(0, eps, 0), shape),
        evalSDF(p + float3(0, 0, eps), shape) - evalSDF(p - float3(0, 0, eps), shape)
    ));
}

struct FragmentOutput {
    float4 color : SV_Target0;
};

[shader("fragment")]
FragmentOutput fragMain(VSOutput input) {
    FragmentOutput output;

    // Reconstruct world-space ray from screen UV
    float2 ndc = input.texCoord * 2.0 - 1.0;
    float4 nearWorld = mul(sdfPC.invViewProjection, float4(ndc, 0.0, 1.0));
    float4 farWorld  = mul(sdfPC.invViewProjection, float4(ndc, 1.0, 1.0));
    nearWorld.xyz /= nearWorld.w;
    farWorld.xyz  /= farWorld.w;

    float3 rayOrigin = sdfPC.cameraPos;
    float3 rayDir = normalize(farWorld.xyz - nearWorld.xyz);

    // Read scene depth to stop marching behind geometry
    uint depthW, depthH;
    textures[sdfPC.depthTextureIndex].GetDimensions(depthW, depthH);
    float2 depthUV = input.position.xy / float2(depthW, depthH);
    float sceneDepth = textures[sdfPC.depthTextureIndex].Sample(samplers[sdfPC.depthSamplerIndex], depthUV).r;
    float4 depthWorld = mul(sdfPC.invViewProjection, float4(ndc, sceneDepth, 1.0));
    depthWorld.xyz /= depthWorld.w;
    float maxDist = length(depthWorld.xyz - rayOrigin);

    // Raymarch
    float t = 0.0;
    uint hitIndex = 0;
    const int MAX_STEPS = 128;
    const float HIT_THRESHOLD = 0.001;

    for (int step = 0; step < MAX_STEPS; step++) {
        float3 p = rayOrigin + rayDir * t;
        float d = sceneSDF(p, hitIndex);

        if (d < HIT_THRESHOLD) {
            float3 normal = calcNormal(p, hitIndex);
            float3 lightDir = normalize(float3(1.0, 1.0, -1.0));
            float diffuse = max(dot(normal, lightDir), 0.0) * 0.8 + 0.2;
            output.color = float4(sdfPC.drawData[hitIndex].color.rgb * diffuse, sdfPC.drawData[hitIndex].color.a);
            return output;
        }

        t += d;
        if (t > maxDist) break;
    }

    discard;
    output.color = float4(0, 0, 0, 0);
    return output;
}
