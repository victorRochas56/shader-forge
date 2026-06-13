import modules.common;
import modules.pbr;

[shader("vertex")]
VertexOutput vertMain(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID, uint drawID : SV_DrawIndex)
{
    return litVertMain(vertexID, instanceID, drawID);
}

struct FragmentOutput
{
    float4 color : SV_Target0;
    float4 roughness_metal : SV_Target1;
    float4 worldNormal : SV_Target2;
};

float valueNoiseFract(float2 coord, float scale)
{
    float2 pos = coord * scale;
	float f = 0.0;
    float2x2 m = float2x2( 1.6,  1.2, -1.2,  1.6 );
    f  = 0.5000*valueNoise( pos ); pos = mul(m,pos);
    f += 0.2500*valueNoise( pos ); pos = mul(m,pos);
    f += 0.1250*valueNoise( pos ); pos = mul(m,pos);
    f += 0.0625*valueNoise( pos ); pos = mul(m,pos);
	f = 0.5 + 0.5*f;
	return f;
}

#define DRAG_MULT 0.2 // changes how much waves pull on the water
#define CAMERA_HEIGHT 1.5 // how high the camera should be
#define ITERATIONS_RAYMARCH 12 // waves iterations of raymarching
#define ITERATIONS_NORMAL 36 // waves iterations when calculating normals

// Calculates wave value and its derivative, 
// for the wave direction, position in space, wave frequency and time
float2 wavedx(float2 position, float2 direction, float frequency, float timeshift) {
  float x = dot(direction, position) * frequency + timeshift;
  float wave = exp(sin(x) - 1.0);
  float dx = wave * cos(x);
  return float2(wave, -dx);
}

// Calculates waves by summing octaves of various waves with various parameters
float getwaves(float2 position, int iterations) {
  float wavePhaseShift = length(position) * 0.1; // this is to avoid every octave having exactly the same phase everywhere
  float iter = 0.0; // this will help generating well distributed wave directions
  float frequency = 2.0; // frequency of the wave, this will change every iteration
  float timeMultiplier = 1.0; // time multiplier for the wave, this will change every iteration
  float weight = 1.0;// weight in final sum for the wave, this will change every iteration
  float sumOfValues = 0.0; // will store final sum of values
  float sumOfWeights = 0.0; // will store final sum of weights
  for(int i=0; i < iterations; i++) {
    // generate some wave direction that looks kind of random
    float2 p = float2(sin(iter), cos(iter));
    
    // calculate wave data
    float2 res = wavedx(position, p, frequency, pc.time * timeMultiplier + wavePhaseShift);

    // shift position around according to wave drag and derivative of the wave
    position += p * res.y * weight * DRAG_MULT;

    // add the results to sums
    sumOfValues += res.x * weight;
    sumOfWeights += weight;

    // modify next octave ;
    weight = mix(weight, 0.0, 0.2);
    frequency *= 1.18;
    timeMultiplier *= 1.07;

    // add some kind of random value to make next wave look random too
    iter += 1232.399963;
  }
  // calculate and return
  return sumOfValues / sumOfWeights;
}

[shader("fragment")]
FragmentOutput fragMain(VertexOutput input)
{
    FragmentOutput output;

    LitInstanceData inst = pc.litInstanceData[input.instanceID];
    LitPass* pass = pc.litPassData;

    float4 color = float4(0,0,0,1);
    float roughness = 0.01f;

    //float3 N = normalize(input.normal);
    float amplitude = 0.05f;
    float3 pos = input.worldPos;
    // finite-difference gradient of the height field for the normal
    float e = 0.01;
    float h  = getwaves(pos.xz,                  ITERATIONS_NORMAL) * amplitude;
    float hx = getwaves(pos.xz + float2(e, 0.0), ITERATIONS_NORMAL) * amplitude;
    float hz = getwaves(pos.xz + float2(0.0, e), ITERATIONS_NORMAL) * amplitude;

    //output.color = float4(h,h,h,1);
    //return output;
    float3 N = normalize(float3(-(hx - h) / e, 1.0, -(hz - h) / e));
    
    //do as early as possible to not wait for scoreboard stall
    TextureCube envMap = cubemaps[inst.environmentMapIndex];
    SamplerState envSampler = samplers[pc.litPassData.samplerIndex];
    float3 diffuseIBL = calculateDiffuseIBL(N, color.rgb, envMap, inst.maxEnvMips, envSampler);

    float3 V = normalize(pc.litPassData.cameraPos.xyz - input.worldPos);
    float3 F0 = float3(0.04);
    F0 = lerp(F0, color.rgb, 0);

    // mirror-like env reflection; Fresnel already folded in
    float3 specularIBL = calculateSpecularIBL(N, V, roughness.r, F0, envMap, envSampler);

    float3 Lo = float3(0.0);

    for(int i = 0; i < pc.litPassData.lightCount; i++){
        Light* currentLight = &pc.lights[i];

        // Sparse slot skip: removed lights leave a zeroed sentinel in the buffer.
        if(currentLight.intensity <= 0.0) continue;

        if(currentLight.type != 1) {
            float distance = length(currentLight.position - input.worldPos);
            if(distance > currentLight.range){
                continue;
            }
        }

        float3 L;
        float attenuation;
        float3 radiance;

        if(currentLight.type == 1) {
            float3 lightDir = currentLight.direction;
            lightDir = normalize(lightDir);
            L = -lightDir;
            attenuation = 1.0;
            radiance = currentLight.color.rgb * currentLight.intensity;
        }
        else {
            float3 lightPos = currentLight.position;
            float distance = length(lightPos - input.worldPos);

            L = normalize(lightPos - input.worldPos);
            attenuation = 1.0 / (distance * distance);
            radiance = currentLight.color.rgb * currentLight.intensity * attenuation;
        }

        float3 H = normalize(V + L);

        float NdotL_raw = dot(N, L);
        if(NdotL_raw <= 0.0) {
            continue;
        }

        float NdotV = max(dot(N, V), 0.001);
        float NdotL = max(NdotL_raw, 0.001);
        float NdotH = max(dot(N, H), 0.001);
        float HdotV = max(dot(H, V), 0.001);

        float NDF = DistributionGGX(N, H, roughness.r);
        float G = GeometrySmith(N, V, L, roughness.r);
        float3 F = fresnelSchlick(HdotV, F0);

        float3 kS = F;
        float3 kD = float3(1.0) - kS;

        float3 numerator = NDF * G * F;
        float denominator = max(4.0 * NdotV * NdotL, 0.001);
        float3 specular = numerator / denominator;
        specular = max(specular, float3(0.0, 0.0, 0.0));
        specular = min(specular, float3(10.0, 10.0, 10.0));

        float shadow = 0.0;
        if(currentLight.castsShadows == 1) {
            float3 geometricNormal = normalize(input.normal);
            shadow = calculateShadow(input.worldPos, pc.litPassData.cameraPos, pc.litPassData.cameraForward, geometricNormal, L, currentLight, pc.litPassData.shadowAtlasIndex);
            if( currentLight.showCascades>0){
                if(shadow == 0){
                    output.color = float4(1,0,0,1);
                }
                if(shadow == 1){
                    output.color = float4(0,1,0,1);
                }
                if(shadow == 2){
                    output.color = float4(0,0,1,1);
                }
                if(shadow == 3){
                    output.color = float4(1,1,0,1);
                }
                return output;
            }
        }

        Lo += (1.0 - shadow) * (kD * color.rgb / 3.14159265 + specular) * radiance * NdotL;
    }

    float3 kS = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness.r);
    float3 kD = float3(1.0) - kS; 

    float ao = 1.0;
    float3 ambient = (kD * diffuseIBL + specularIBL) * ao;
    float3 outColor = ambient + Lo;

    output.color = float4(outColor, 1.0);
    output.roughness_metal = float4(roughness, 0.0, 0.0, 1.0);
    output.worldNormal = float4(N * 0.5 + 0.5, 1.0);
    return output;
}

