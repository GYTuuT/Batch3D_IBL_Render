#version 450 core
out vec4 FragColor;

in VS_OUT {
    vec3 FragPos;
    vec3 Normal;
    vec2 TexCoords;
    vec3 Color;
    vec3 Material; // metallic, roughness, ao
    vec4 SSSParams; // strength, distortion, power, scale
    vec3 SSSColor;
} fs_in;

// Lighting
uniform vec3 lightPosition;
uniform vec3 lightColor;
uniform vec3 viewPosition; // Camera position

// Material properties (already in fs_in.Material)
// uniform float metallic;
// uniform float roughness;
// uniform float ao; // Ambient Occlusion

// SSS global enable
uniform bool enableSSS;

// Texture
layout(binding = 1) uniform sampler2D diffuseTexture;
uniform bool useTexture;

// IBL Textures (if integrating with IBL)
layout(binding = 3) uniform samplerCube environmentMap;
layout(binding = 4) uniform sampler2D brdfLUT;
uniform bool useIBL; // To switch between basic lighting and IBL

const float PI = 3.14159265359;

// ACES Filmic Tone Mapping (approximation)
// Source: http://chilliant.blogspot.com/2012/08/filmic-tonemapping-operators.html (John Hable)
// and https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
// Args:
//   color: Input HDR color (vec3)
// Returns:
//   Tone-mapped LDR color (vec3)
vec3 ACESFilmicToneMapping(vec3 color) {
    color *= 0.6; // Exposure adjustment (can be a uniform)
    color = (color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14);
    return color;
}

// Gamma Correction
// Args:
//   color: Input linear color (vec3)
// Returns:
//   Gamma-corrected color (vec3)
vec3 gammaCorrect(vec3 color) {
    return pow(color, vec3(1.0/2.2));
}

// Fresnel Schlick approximation
// Args:
//   cosTheta: Cosine of the angle between the view direction and the half-vector (float)
//   F0: Base reflectivity at normal incidence (vec3)
// Returns:
//   Fresnel reflectance (vec3)
vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Normal Distribution Function (NDF) - Trowbridge-Reitz GGX
// Args:
//   N: Surface normal (vec3)
//   H: Half-vector (vec3)
//   roughness: Surface roughness (float, [0,1])
// Returns:
//   GGX NDF value (float)
float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / max(denom, 0.0000001);
}

// Geometry Function (Schlick-GGX approximation for Smith)
// Args:
//   NdotV: Dot product of normal and view/light vector (float)
//   roughness: Surface roughness (float, [0,1])
// Returns:
//   Geometry term for one direction (float)
float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}

// Geometry Function (Smith's method using Schlick-GGX)
// Args:
//   N: Surface normal (vec3)
//   V: View vector (vec3)
//   L: Light vector (vec3)
//   roughness: Surface roughness (float, [0,1])
// Returns:
//   Combined geometry term (float)
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

// Subsurface Scattering approximation
// Args:
//   lightDir: Direction of light (vec3)
//   viewDir: Direction of view (vec3)
//   normal: Surface normal (vec3)
//   sssColor: Base color for SSS effect (vec3)
//   sssParams: SSS parameters [strength, distortion, power, scale] (vec4)
// Returns:
//   Subsurface scattering contribution (vec3)
vec3 subsurfaceScattering(vec3 lightDir, vec3 viewDir, vec3 normal, vec3 sssColor, vec4 sssParams)
{
    // Full SSS approximation with distortion, power, and scale
    float strength = sssParams.x;
    float distortion = sssParams.y;
    float power = sssParams.z;
    float scale = sssParams.w;

    // Calculate scattered light direction
    vec3 scatteredLightDir = lightDir + normal * distortion;
    // Angle between view direction and scattered light
    float scatteredDot = pow(clamp(dot(viewDir, -scatteredLightDir), 0.0, 1.0), power) * scale;

    return sssColor * scatteredDot * strength;
}

void main()
{
    // Extract material properties and compute full PBR + SSS
    // Extract material properties from fragment shader input
    float metallic = fs_in.Material.x;
    float roughness = fs_in.Material.y;
    float ao = fs_in.Material.z; // Ambient Occlusion factor

    // Base color calculation, potentially mixed with texture
    vec3 albedo = fs_in.Color;
    if (useTexture) {
        vec4 texColor = texture(diffuseTexture, fs_in.TexCoords);
        albedo = mix(albedo, texColor.rgb, texColor.a);
    }

    // Calculate PBR material properties
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    // Lighting vectors
    vec3 N = normalize(fs_in.Normal);
    vec3 V = normalize(viewPosition - fs_in.FragPos);
    vec3 L = normalize(lightPosition - fs_in.FragPos);
    vec3 H = normalize(V + L);

    // Calculate distance attenuation for point light
    float distance = length(lightPosition - fs_in.FragPos);
    float attenuation = 1.0 / (distance * distance);
    vec3 radiance = lightColor * attenuation;

    // Cook-Torrance BRDF terms for direct lighting
    float NDF = DistributionGGX(N, H, roughness);
    float G   = GeometrySmith(N, V, L, roughness);
    vec3  F   = fresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 kS = F;
    vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);

    vec3 numerator = NDF * G * F;
    float denom = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
    vec3 specular = numerator / denom;

    float NdotL = max(dot(N, L), 0.0);
    vec3 Lo = (kD * albedo / PI + specular) * radiance * NdotL;

    // Subsurface scattering contribution
    vec3 sssContribution = vec3(0.0);
    if (enableSSS && fs_in.SSSParams.x > 0.0) {
        sssContribution = subsurfaceScattering(L, V, N, fs_in.SSSColor, fs_in.SSSParams);
    }

    // IBL contribution
    vec3 ambient = vec3(0.03) * albedo * ao;
    if (useIBL) {
        vec3 R = reflect(-V, N);
        vec3 irradiance = texture(environmentMap, N).rgb;
        vec3 diffuseIBL = irradiance * albedo;
        const float MAX_REFLECTION_LOD = 7.0;
        vec3 prefilteredColor = textureLod(environmentMap, R, roughness * MAX_REFLECTION_LOD).rgb;
        vec2 brdf  = texture(brdfLUT, vec2(max(dot(N, V), 0.0), roughness)).rg;
        vec3 specularIBL  = prefilteredColor * (F * brdf.x + brdf.y);
        ambient = (kD * diffuseIBL + specularIBL) * ao;
    }

    // Compose final color: Ambient + Direct + SSS
    vec3 finalColor = ambient + Lo + sssContribution;
    // Apply tone mapping and gamma correction
    finalColor = ACESFilmicToneMapping(finalColor);
    finalColor = gammaCorrect(finalColor);
    FragColor = vec4(finalColor, 1.0);
    return;
    // End of main
}
