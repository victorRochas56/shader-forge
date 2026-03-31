import modules.common;
import modules.pbr;

struct SSRPushConstants {
    SSRPassData* passData;
};

struct SSRPassData {
    float4x4 invViewProj;
    float4x4 viewProj;
    float3 cameraPos;
    uint depthIndex;
    uint depthSamplerIndex;
    uint colorIndex;
    uint colorSamplerIndex;
    uint roughnessMetalIndex;
    uint roughnessMetalSamplerIndex;
    uint normalIndex;
    uint normalSamplerIndex;
    uint _pad_resolution;
    uint2 resolution;
    float maxDistance;
    uint hiZIndex;
    uint hiZMipLevels;
    float thickness;
    float roughnessThreshold;
    int maxSteps;
    uint frameIndex;
};

[[vk::push_constant]] 
ConstantBuffer<SSRPushConstants> SSRpc;

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

// Project world position to screen-space UV (.xy) + projected depth (.z)
float3 worldToScreen(float3 worldPos) {
    float4 clip = mul(SSRpc.passData.viewProj, float4(worldPos, 1.0));
    clip.xyz /= clip.w;
    float2 uv = clip.xy * 0.5 + 0.5;
    return float3(uv, clip.z);
}

struct RayTraceOutput {
    bool Hit;
    float2 UV;
};
/*
float3 RayTrace(float3 p, float3 v) {

    const float rootLevel = SSRpc.passData.hiZMipLevels - 1.0f; //convert to 0 based indexing

    float level = 3; //starting hiZ level
    float2 crossEpsilon = float2(0.001,0.001);

    //get cell cross direction and small offset to enter the next cell when crossing
    float2 crossStep, crossOffset;
    crossStep.x = (v.x >= 0) ? 1.f : -1.f;
    crossStep.y = (v.y >= 0) ? 1.f : -1.f;
    crossOffset.xy = crossStep.xy * crossEpsilon.xy;
    crossStep.xy = saturate(crossStep.xy);

    //set current ray to the original screen coordinate and depth
    // max depth
    float3 ray = p.xyz;

    //scale the vector such that z is 1.0f
    float3 d = v.xyz /= v.z;

    //set starting point to the point where z = 0.0f
    // min depth
    float3 o = intersectDepthPlane(p.xy,d.xy,-p.z);

    // cross to next cell so that we don't self intersect immediately
    float2 rayCell = getCell(ray.xy, hiZSize.xy);
    ray = intersectCellBoundary(o.xy,d.xy,rayCell.xy,hiZSize.xy,crossStep.xy,crossOffset.xy);

    int iterations = 0;
    // stop level is 2, too high can lead to artifacts
    [loop]
    while(level >= 2 && iterations < SSRpc.passData.maxSteps){
        // get the minimum depth plane in which the current ray resides
        float minZ = getMinimumDepthPlane(ray.xy, level, rootLevel);

        //get the cell number of our current ray
        const float2 cellCount = getCellCount(level, rootLevel);
        const float2 oldCellIdx = getCell(ray.xy, cellCount);

        //intersect only if ray depth is below the minimum depth plane
        float3 tmpRay = intersectDepthPlane(o.xy,d.xy,max(ray.z,minZ));

        //get the new cell number as well
        const float2 newCellIdx = getCell(tmpRay.xy, cellCount);

        //if the new cell number is different from the old cell number, we crossed a cell
        [branch]
        if(crossedCellBoundary(oldCellIdx,newCellIdx)){
            //so intersect the boundary of that cell instead and go up a level
            //for taking a larger step next loop
            tmpRay = intersectCellBoundary(o,d,oldCellIdx,cellCount.xy,crossStep.xy, crossOffset.xy);
            //clamp to a highest level to avoid too much stepping around
            level = min(SSRpc.passData.hiZMipLevels - 1.0f, level + 2.0f);
        }
        ray.xyz = tmpRay.xyz;

        //go down a level in the hiZ
        --level;

        ++iterations;
        //end while
    } 
    return ray;
}*/


#define M1 1597334677U     //1719413*929
#define M2 3812015801U     //140473*2467*11
float2 hash2D( uint2 q, uint seed )
{
    q *= uint2(M1, M2); 
    
    uint n = ((q.x ^ q.y) ^ seed) * M1;
    
    return float2(float(n) * (1.0/float(0xffffffffU)),float(n+1) * (1.0/float(0xffffffffU)));
}


[shader("fragment")]
float4 fragMain(VSOutput input) : SV_Target {
    return float4(0, 0, 0, 0);
    float2 uv = input.texCoord;

    float roughness = textures[SSRpc.passData.roughnessMetalIndex].SampleLevel(samplers[SSRpc.passData.roughnessMetalSamplerIndex], uv,0).r;

    if (roughness > SSRpc.passData.roughnessThreshold)
        return float4(0, 0, 0, 0);

    float initDepth = textures[SSRpc.passData.depthIndex].SampleLevel(samplers[SSRpc.passData.depthSamplerIndex], uv, 0).r;
    float3 viewPos = reconstructViewPos(uv, initDepth, SSRpc.passData.invViewProj);

    float3 normalData = textures[SSRpc.passData.normalIndex].Sample(samplers[SSRpc.passData.normalSamplerIndex], uv).xyz;
    float3 normal = normalize(2.0f * normalData - 1.0f);

    float3 viewDir = normalize(viewPos - SSRpc.passData.cameraPos);

    float3 reflectDir = normalize(reflect(viewDir, normal));
    //float3 ray = RayTrace(viewPos, reflectDir);

}
