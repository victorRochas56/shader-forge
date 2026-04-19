import modules.common;

enum SDFType : uint {
    SPHERE,
    CYLINDER,
    CONE,
    PYRAMID
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

//need ConstantBuffer because it contain BDAs  
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
    output.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    return output;
}

// ---- SDF distance functions (operate in local space) ----

float sdSphere(float3 p, float radius) {
    return length(p) - radius;
}

float sdCylinder(float3 p, float radius, float height)
{
  float2 d = abs(float2(length(p.xz),p.y)) - float2(radius,height);
  return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float sdCone(float3 p, float radius, float height) {
    float2 q = float2(length(p.xz), p.y);

    // Cone: base at y=0 with given radius, tip at y=height
    float2 tip = float2(0.0, height);
    float2 baseEdge = float2(radius, 0.0);
    float2 e = baseEdge - tip;

    // Distance to cone surface (line segment from tip to base edge in 2D)
    float2 w = q - tip;
    float t = clamp(dot(w, e) / dot(e, e), 0.0, 1.0);
    float surfaceDist = length(q - (tip + e * t));

    // Distance to base cap (line segment from origin to base edge along y=0)
    float2 closestOnBase = float2(clamp(q.x, 0.0, radius), 0.0);
    float baseDist = length(q - closestOnBase);

    float d = min(surfaceDist, baseDist);

    // Inside test: above base, below tip, and within the cone's taper
    bool inside = (q.y >= 0.0) && (q.y <= height) && (q.x <= radius * (height - q.y) / height);

    return inside ? -d : d;
}

float sdPyramid( float3 p, float h )
{
  float m2 = h*h + 0.25;
    
  p.xz = abs(p.xz);
  p.xz = (p.z>p.x) ? p.zx : p.xz;
  p.xz -= 0.5;

  float3 q = float3( p.z, h*p.y - 0.5*p.x, h*p.x + 0.5*p.y);
  float s = max(-q.x,0.0);
  float t = clamp( (q.y-0.5*p.z)/(m2+0.25), 0.0, 1.0 );
  float a = m2*(q.x+s)*(q.x+s) + q.y*q.y;
  float b = m2*(q.x+0.5*t)*(q.x+0.5*t) + (q.y-m2*t)*(q.y-m2*t);
    
  float d2 = min(q.y,-q.x*m2-q.y*0.5) > 0.0 ? 0.0 : min(a,b);
  return sqrt( (d2+q.z*q.z)/m2 ) * sign(max(q.z,-p.y));
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
    case SDFType.PYRAMID:
        return sdPyramid(localPos, shape.height);
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
    float surfaceDist = length(depthWorld.xyz - rayOrigin);
    float maxDist = 50.0f;
    // Raymarch
    float t = 0.0;
    uint hitIndex = 0;
    const int MAX_STEPS = 64;
    const float HIT_THRESHOLD = 0.001;

    float alphaMult = 1.0f;
    for (int step = 0; step < MAX_STEPS; step++) {
        float3 p = rayOrigin + rayDir * t;
        float d = sceneSDF(p, hitIndex);

        if (t > surfaceDist)
            alphaMult = 0.2f;
        if (d < HIT_THRESHOLD) {
            float3 normal = calcNormal(p, hitIndex);
            float3 lightDir = normalize(float3(1.0, 1.0, -1.0));
            float diffuse = max(dot(normal, lightDir), 0.0) * 0.8 + 0.2;

            SDF hitShape = sdfPC.drawData[hitIndex];
            float3 baseColor = hitShape.color.rgb;

            output.color = float4(baseColor * diffuse, hitShape.color.a * alphaMult);
            return output;
        }

        t += d;
        if (t > maxDist) break;
    }

    output.color = float4(0, 0, 0, 0);
    return output;
}
