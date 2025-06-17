#version 450 core

in vec3 WorldPos;  // World position from vertex shader

out vec4 FragColor;

uniform sampler2D equirectangularMap;  // Input equirectangular HDR texture

const float PI = 3.14159265359;
const float TWO_PI = 6.28318530718;
const float HALF_PI = 1.57079632679;


vec2 sampleSphericalMap(vec3 direction) {
    /**
     * Convert 3D direction vector to 2D equirectangular coordinates.
     *
     * This function maps a 3D direction vector to UV coordinates for
     * sampling an equirectangular (spherical) texture.
     *
     * Args:
     *   direction: Normalized 3D direction vector
     *
     * Returns:
     *   2D UV coordinates [0,1] for equirectangular sampling
     */

    // Normalize the direction vector to ensure correct calculations
    vec3 normalizedDir = normalize(direction);

    // Calculate spherical coordinates
    // atan2 gives azimuth angle [-π, π], we need [0, 2π]
    float azimuth = atan(normalizedDir.z, normalizedDir.x);
    if (azimuth < 0.0) {
        azimuth += TWO_PI;
    }

    // asin gives elevation angle [-π/2, π/2], we need [0, π]
    float elevation = asin(normalizedDir.y);

    // Convert to UV coordinates [0, 1]
    vec2 uv;
    uv.x = azimuth / TWO_PI;           // Horizontal: 0 to 1
    uv.y = (elevation + HALF_PI) / PI;  // Vertical: 0 to 1

    return uv;
}

vec3 sampleEquirectangular(vec3 direction) {
    /**
     * Sample equirectangular texture in the given direction.
     *
     * Args:
     *   direction: 3D sampling direction
     *
     * Returns:
     *   RGB color from equirectangular map
     */
    vec2 uv = sampleSphericalMap(direction);

    // Clamp UV coordinates to prevent sampling artifacts at edges
    uv = clamp(uv, 0.001, 0.999);

    return texture(equirectangularMap, uv).rgb;
}


void main() {
        // Use the world position as the sampling direction
    // This works because the cube vertices are positioned at unit distance
    vec3 samplingDirection = normalize(WorldPos);

        // Sample the equirectangular map in the calculated direction
    vec3 color = sampleEquirectangular(samplingDirection);

        // Check for invalid values and handle them gracefully
    if (any(isnan(color)) || any(isinf(color))) {
        color = vec3(0.0);  // Set to black for invalid values
    }

    // Clamp extremely bright values to prevent overflow
    color = min(color, vec3(100.0));

        // Output the sampled color for this cubemap face
    FragColor = vec4(color, 1.0);
}
