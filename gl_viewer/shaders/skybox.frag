#version 450 core

in vec3 TexCoords;  // 3D texture coordinates from vertex shader

out vec4 FragColor;

layout(binding = 3) uniform samplerCube environmentMap;  // Environment cubemap texture

const float EXPOSURE = 1.0;           // HDR exposure adjustment
const float GAMMA = 2.2;              // Gamma correction value
const float SATURATION_BOOST = 1.1;   // Slight saturation enhancement

// ACES Filmic Tone Mapping (approximation)
vec3 ACESFilmicToneMapping(vec3 color) {
    // Skybox might need different exposure adjustment compared to scene objects
    // color *= 1.0; // Example: Skybox exposure, adjust as needed
    // Or use the same as other shaders for consistency if EXPOSURE constant is removed/adjusted
    color = (color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14);
    return color;
}

vec3 tonemap_reinhard(vec3 color) {
    /**
     * Simple Reinhard tone mapping for HDR to LDR conversion.
     *
     * Args:
     *   color: HDR color value
     *
     * Returns:
     *   Tone mapped LDR color
     */
    return color / (color + vec3(1.0));
}

vec3 apply_gamma_correction(vec3 color) {
    /**
     * Apply gamma correction for proper display.
     *
     * Args:
     *   color: Linear color value
     *
     * Returns:
     *   Gamma corrected color
     */
    return pow(color, vec3(1.0 / GAMMA));
}

vec3 enhance_saturation(vec3 color, float saturation) {
    /**
     * Enhance color saturation for more vibrant skybox.
     *
     * Args:
     *   color: Input color
     *   saturation: Saturation multiplier
     *
     * Returns:
     *   Saturation enhanced color
     */
    float luminance = dot(color, vec3(0.299, 0.587, 0.114));
    return mix(vec3(luminance), color, saturation);
}


void main() {
        // Sample the environment cubemap using the interpolated texture coordinates
    vec3 envColor = texture(environmentMap, TexCoords).rgb;

        // Apply exposure adjustment (can be part of ACES or separate)
    envColor *= EXPOSURE; // Keep this if ACES exposure is different or for fine-tuning skybox

    // Enhance saturation slightly for more vibrant appearance
    envColor = enhance_saturation(envColor, SATURATION_BOOST);

    // Apply ACES Filmic Tone Mapping
    envColor = ACESFilmicToneMapping(envColor);

    // Apply gamma correction for proper display
    envColor = apply_gamma_correction(envColor); // apply_gamma_correction uses GAMMA constant

        // Output final skybox color with full opacity
    FragColor = vec4(envColor, 1.0);
}
