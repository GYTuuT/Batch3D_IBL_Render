#version 450 core

in VS_OUT {
    vec3 worldPos;      // World space position
    vec3 viewPos;       // View space position
    vec3 normal;        // World space normal
    vec3 color;         // Vertex color
    vec2 texCoord;      // Texture coordinates
    vec3 material;      // Material properties [metallic, roughness, ao]
    vec4 instanceData;  // Instance data
    float distance;     // Distance from camera
} fs_in;

out vec4 FragColor;

layout(binding = 3) uniform samplerCube environmentMap;    // Environment cubemap (texture unit 3)
layout(binding = 5) uniform samplerCube irradianceMap;    // Irradiance map (texture unit 5)
layout(binding = 6) uniform samplerCube prefilterMap;     // Prefiltered environment map (texture unit 6)
uniform vec3 viewPos;                  // Camera position in world space
uniform float ambientIntensity;        // Global ambient intensity
uniform float specularIntensity;       // Global specular intensity
uniform float baseRoughness;           // Base roughness offset
layout(binding = 4) uniform sampler2D brdfLUT; // BRDF Lookup Table (texture unit 4)

uniform bool useTexture;              // Enable texture sampling
layout(binding = 1) uniform sampler2D diffuseTexture;     // Diffuse texture (texture unit 1)
uniform vec3 materialDiffuse;         // Global material diffuse multiplier
uniform vec3 materialSpecular;        // Global material specular multiplier
uniform float materialAlpha;          // Global material alpha

uniform bool enableInstancing;        // Enable instancing effects
uniform bool enableDistanceFade;      // Enable distance-based fading
uniform float fadeDistance;           // Distance at which fading starts

const float PI = 3.14159265359;
const float MAX_REFLECTION_LOD = 4.0;  // Maximum mip level for prefilter map (0-4 for 5 levels)
const vec3 F0_DIELECTRIC = vec3(0.04); // F0 for non-metallic materials

// ACES Filmic Tone Mapping (approximation)
// Args:
//   color: Input HDR color (vec3)
// Returns:
//   Tone-mapped LDR color (vec3)
vec3 ACESFilmicToneMapping(vec3 color) {
    color *= 0.6; // Exposure adjustment
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


vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    /**
     * Schlick's approximation for Fresnel reflectance.
     *
     * Args:
     *   cosTheta: Cosine of angle between view and half vector (float)
     *   F0: Base reflectance at normal incidence (vec3)
     * Returns:
     *   Fresnel reflectance factor (vec3)
     */
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    /**
     * Fresnel with roughness compensation for environment mapping.
     *
     * Args:
     *   cosTheta: Cosine of angle between view and normal (float)
     *   F0: Base reflectance at normal incidence (vec3)
     *   roughness: Surface roughness (float, [0,1])
     * Returns:
     *   Roughness-compensated Fresnel factor (vec3)
     */
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float distributionGGX(vec3 N, vec3 H, float roughness) {
    /**
     * GGX/Trowbridge-Reitz normal distribution function.
     *
     * Args:
     *   N: Surface normal (vec3)
     *   H: Half vector between view and light (vec3)
     *   roughness: Surface roughness (float, [0,1])
     * Returns:
     *   Normal distribution term (float)
     */
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return num / denom;
}

float geometrySchlickGGX(float NdotV, float roughness) {
    /**
     * Schlick-GGX geometry function for a single direction.
     *
     * Args:
     *   NdotV: Dot product of normal and view/light direction (float)
     *   roughness: Surface roughness (float, [0,1])
     * Returns:
     *   Geometry masking/shadowing term (float)
     */
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;

    float num = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return num / denom;
}

float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    /**
     * Smith geometry function combining masking and shadowing.
     *
     * Args:
     *   N: Surface normal (vec3)
     *   V: View direction (vec3)
     *   L: Light direction (vec3)
     *   roughness: Surface roughness (float, [0,1])
     * Returns:
     *   Combined geometry term (float)
     */
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = geometrySchlickGGX(NdotV, roughness);
    float ggx1 = geometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}


vec3 sampleEnvironmentDiffuse(vec3 normal) {
    /**
     * Sample irradiance map for diffuse (Lambertian) lighting.
     * Uses the dedicated irradiance map that has been pre-convolved.
     *
     * Args:
     *   normal: Surface normal (vec3)
     * Returns:
     *   Diffuse irradiance from pre-convolved irradiance map (vec3)
     */
    return texture(irradianceMap, normal).rgb;
}

vec3 sampleEnvironmentSpecular(vec3 reflectDir, float roughness) {
    /**
     * Sample prefiltered environment map for specular reflection.
     * Uses the dedicated prefilter map with roughness-based mip levels.
     *
     * Args:
     *   reflectDir: Reflection vector (vec3)
     *   roughness: Surface roughness (float, [0,1])
     * Returns:
     *   Specular color from prefiltered environment map (vec3)
     */
    float mipLevel = roughness * MAX_REFLECTION_LOD;
    return textureLod(prefilterMap, reflectDir, mipLevel).rgb;
}


void main() {
        vec3 N = normalize(fs_in.normal);

    // Handle degenerate normals
    if (length(fs_in.normal) < 0.1) {
        FragColor = vec4(fs_in.color, materialAlpha);
        return;
    }

    vec3 V = normalize(viewPos - fs_in.worldPos);
    vec3 R = reflect(-V, N);

        // Extract and validate material properties
    float metallic = clamp(fs_in.material.x, 0.0, 1.0);
    float roughness = clamp(fs_in.material.y + baseRoughness, 0.01, 1.0); // Prevent zero roughness
    float ao = clamp(fs_in.material.z, 0.0, 1.0);

        vec3 baseColor = fs_in.color;

    // Apply texture if enabled - ensure proper texture unit usage
    if (useTexture) {
        vec4 texColor = texture(diffuseTexture, fs_in.texCoord);
        baseColor = mix(baseColor, texColor.rgb, texColor.a);
    }

    // Apply global material diffuse multiplier
    baseColor *= materialDiffuse;

        // Calculate F0 (surface reflection at normal incidence)
    vec3 F0 = mix(F0_DIELECTRIC, baseColor, metallic);

    // Calculate Fresnel for environment lighting (using NdotV for BRDF LUT lookup)
    // This F is used for direct lighting, for IBL, F0 is used with BRDF LUT.
    // vec3 F = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);

    // Energy conservation for diffuse part
    vec3 kS_direct = fresnelSchlick(max(dot(N,V),0.0), F0); // For direct light, if any
    vec3 kD = vec3(1.0) - kS_direct;       // Diffuse contribution factor
    kD *= (1.0 - metallic);                // Metallics have no diffuse reflection

    // Sample IBL using pre-computed irradiance and prefilter maps for better performance
    vec3 irradiance = sampleEnvironmentDiffuse(N); // Sample from dedicated irradiance map
    vec3 diffuseIBL = irradiance * baseColor;

    vec3 prefilteredColor = sampleEnvironmentSpecular(R, roughness); // Sample from prefiltered environment map
    vec2 brdfSample = texture(brdfLUT, vec2(max(dot(N, V), 0.0), roughness)).rg;
    vec3 specularIBL = prefilteredColor * (F0 * brdfSample.x + brdfSample.y);

        // Modulate with AO
    // Diffuse part scaled by ambientIntensity, Specular part by specularIntensity and materialSpecular
    vec3 ambientLighting = (kD * diffuseIBL * ambientIntensity + specularIBL * specularIntensity * materialSpecular) * ao;

    vec3 finalColor = ambientLighting; // This is the IBL contribution

        if (enableInstancing) {
        // Apply instance-based color modulation
        float instanceFactor = clamp(fs_in.instanceData.w, 0.5, 2.0);
        finalColor = mix(finalColor, finalColor * instanceFactor, 0.2);
    }

        float finalAlpha = materialAlpha;

    if (enableDistanceFade) {
        float fadeFactor = clamp(1.0 - (fs_in.distance - fadeDistance) / fadeDistance, 0.0, 1.0);
        finalAlpha *= fadeFactor;
    }

    // Apply ACES Filmic Tone Mapping
    finalColor = ACESFilmicToneMapping(finalColor);
    // Apply Gamma correction
    finalColor = gammaCorrect(finalColor);

    // Ensure alpha is valid
    finalAlpha = clamp(finalAlpha, 0.0, 1.0);

    // Output final fragment color
    FragColor = vec4(finalColor, finalAlpha);
}
