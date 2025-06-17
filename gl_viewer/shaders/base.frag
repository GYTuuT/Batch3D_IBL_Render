#version 450 core

in VS_OUT {
    vec3 worldPos;      // World space position
    vec3 viewPos;       // View space position
    vec3 normal;        // World space normal
    vec3 color;         // Vertex color
    vec2 texCoord;      // Texture coordinates
    float distance;     // Distance from camera
    float pointSize;    // Final point size
    float lineWidth;    // Final line width
    flat int shapeType; // Shape type
    vec4 instanceData;  // Instance data
} fs_in;

out vec4 FragColor;

uniform vec3 lightPosition;    // Light position in world space
uniform vec3 lightColor;       // Light color
uniform vec3 viewPosition;     // Camera position in world space
uniform float ambientStrength; // Ambient light strength [0,1]
uniform float specularStrength;// Specular reflection strength [0,1]
uniform float shininess;       // Specular shininess exponent

uniform bool useTexture;           // Enable texture sampling
uniform sampler2D diffuseTexture;  // Diffuse texture
uniform vec3 materialDiffuse;      // Material diffuse color
uniform vec3 materialSpecular;     // Material specular color
uniform float materialAlpha;       // Material transparency

uniform bool enableFog;            // Enable fog effect
uniform vec3 fogColor;             // Fog color
uniform float fogNear;             // Fog start distance
uniform float fogFar;              // Fog end distance

uniform bool enableWireframe;      // Enable wireframe effect
uniform vec3 wireframeColor;       // Wireframe line color

uniform bool enableDistanceFade;   // Enable distance-based fading
uniform float fadeDistance;        // Distance at which fading starts

uniform bool enableCustomShapes;   // Enable custom point shapes
uniform sampler2D shapeTexture;    // Custom shape texture
uniform bool enableInstancing;     // Enable instancing effects

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

// Calculates Phong lighting.
// Args:
//   normal: Surface normal (vec3)
//   lightDir: Direction from surface point to light (vec3)
//   viewDir: Direction from surface point to camera (vec3)
//   diffuseColor: Base diffuse color of the surface (vec3)
// Returns:
//   Calculated Phong lighting color (vec3)
vec3 calculatePhongLighting(vec3 normal, vec3 lightDir, vec3 viewDir, vec3 diffuseColor) {
    // Normalize all vectors
    vec3 N = normalize(normal);
    vec3 L = normalize(lightDir);
    vec3 V = normalize(viewDir);

    // Ambient component
    vec3 ambient = ambientStrength * lightColor;

    // Diffuse component (Lambertian reflection)
    float NdotL = max(dot(N, L), 0.0);
    vec3 diffuse = NdotL * lightColor * diffuseColor;

    // Specular component (Phong reflection model)
    vec3 R = reflect(-L, N);
    float RdotV = max(dot(R, V), 0.0);
    float specularFactor = pow(RdotV, shininess);
    vec3 specular = specularStrength * specularFactor * lightColor * materialSpecular;

    return ambient + diffuse + specular;
}

// Calculates fog factor based on distance.
// Args:
//   distance: Distance from camera to fragment (float)
// Returns:
//   Fog factor (float, [0,1], 0 = full fog, 1 = no fog)
float calculateFogFactor(float distance) {
    // Linear fog calculation
    return clamp((fogFar - distance) / (fogFar - fogNear), 0.0, 1.0);
}

// All shape functions return alpha value [0,1] for antialiased rendering
// Args:
//   coord: Centered 2D coordinate within the point sprite (vec2, typically [-1,1])
//   radius/size: Size parameter for the shape (float)
// Returns:
//   Alpha value for the shape at the given coordinate (float, [0,1])

float drawCircle(vec2 coord, float radius) {
    float dist = length(coord);
    return 1.0 - smoothstep(radius - 0.05, radius, dist);
}

float drawSquare(vec2 coord, float size) {
    vec2 d = abs(coord) - vec2(size);
    float dist = max(d.x, d.y);
    return 1.0 - smoothstep(-0.02, 0.02, dist);
}

float drawDiamond(vec2 coord, float size) {
    float dist = abs(coord.x) + abs(coord.y) - size;
    return 1.0 - smoothstep(-0.02, 0.02, dist);
}

float drawStar(vec2 coord, float size) {
    // Convert to polar coordinates
    float angle = atan(coord.y, coord.x);
    float radius = length(coord);

    // Create 5-pointed star shape
    float starRadius = size * (0.7 + 0.3 * cos(5.0 * angle));
    return 1.0 - smoothstep(starRadius - 0.05, starRadius + 0.05, radius);
}

float drawTriangle(vec2 coord, float size) {
    // Offset center for better visual balance
    coord.y += size * 0.2;

    // Calculate signed distance to triangle
    float d = max(
        abs(coord.x) * 0.866 + coord.y * 0.5,  // Two angled sides
        -coord.y                                // Bottom side
    ) - size * 0.5;

    return 1.0 - smoothstep(-0.02, 0.02, d);
}

float drawCross(vec2 coord, float size) {
    // Create cross from two rectangles
    vec2 horizontal = abs(coord) - vec2(size, size * 0.25);
    vec2 vertical = abs(coord) - vec2(size * 0.25, size);

    float crossH = max(horizontal.x, horizontal.y);
    float crossV = max(vertical.x, vertical.y);
    float dist = min(crossH, crossV);

    return 1.0 - smoothstep(-0.02, 0.02, dist);
}

// Draws a custom shape sampled from shapeTexture.
// Args:
//   coord: Centered 2D coordinate (vec2, [-1,1])
// Returns:
//   Alpha value from texture (float, [0,1])
float drawCustomShape(vec2 coord) {
    // Sample custom shape from texture
    vec2 uv = coord * 0.5 + 0.5;  // Convert [-1,1] to [0,1]
    uv = clamp(uv, 0.0, 1.0);     // Ensure valid texture coordinates
    return texture(shapeTexture, uv).r;
}

void main() {
        vec3 baseColor = fs_in.color;

    // Apply texture if enabled
    if (useTexture) {
        vec4 texColor = texture(diffuseTexture, fs_in.texCoord);
        baseColor = mix(baseColor, texColor.rgb, texColor.a);
    }

    // Apply material diffuse color
    baseColor *= materialDiffuse;

        vec3 finalColor = baseColor;

    // Apply Phong lighting if normal is valid
    if (length(fs_in.normal) > 0.1) { // Apply lighting only if normal is valid
        vec3 lightDir = lightPosition - fs_in.worldPos;
        vec3 viewDir = viewPosition - fs_in.worldPos;

        finalColor = calculatePhongLighting(fs_in.normal, lightDir, viewDir, baseColor);
    }

        if (enableWireframe) {
        // Create a grid pattern based on texture coordinates for a simple wireframe look.
        // fwidth provides an estimate of the rate of change, useful for antialiasing procedural lines.
        vec2 grid = abs(fract(fs_in.texCoord * 10.0) - 0.5) / fwidth(fs_in.texCoord * 10.0);
        float line = min(grid.x, grid.y); // Distance to the nearest grid line
        float wireframeFactor = 1.0 - clamp(line, 0.0, 1.0); // Inverted and clamped to make lines appear

        // Blend wireframe color with surface color. 0.5 factor for semi-transparency.
        finalColor = mix(finalColor, wireframeColor, wireframeFactor * 0.5);
    }

        float alpha = materialAlpha; // Base alpha from material properties
    float shapeAlpha = 1.0;      // Alpha contribution from point shape rendering (if applicable)

        if (enableCustomShapes) {
        // Convert gl_PointCoord [0,1] to centered coordinates [-1,1]
        vec2 pointCoord = gl_PointCoord * 2.0 - 1.0;

        // Apply shape-specific rendering
        switch (fs_in.shapeType) {
            case 0: // Circle
                shapeAlpha = drawCircle(pointCoord, 0.8);
                break;
            case 1: // Square
                shapeAlpha = drawSquare(pointCoord, 0.7);
                break;
            case 2: // Diamond
                shapeAlpha = drawDiamond(pointCoord, 0.8);
                break;
            case 3: // Star
                shapeAlpha = drawStar(pointCoord, 0.6);
                break;
            case 4: // Triangle
                shapeAlpha = drawTriangle(pointCoord, 0.8);
                break;
            case 5: // Cross
                shapeAlpha = drawCross(pointCoord, 0.7);
                break;
            case 6: // Custom texture shape
                shapeAlpha = drawCustomShape(pointCoord);
                break;
            default:
                shapeAlpha = drawCircle(pointCoord, 0.8);
                break;
        }

        // Add antialiasing for all shapes
        float edgeDistance = length(pointCoord);
        float antiAlias = 1.0 - smoothstep(0.8, 1.0, edgeDistance);
        shapeAlpha *= antiAlias;
    }

        if (enableInstancing) {
        // Apply instance-based color modulation
        float instanceFactor = clamp(fs_in.instanceData.w, 0.5, 2.0);
        finalColor = mix(finalColor, finalColor * instanceFactor, 0.3);
    }

        if (enableDistanceFade) {
        float fadeFactor = clamp(1.0 - (fs_in.distance - fadeDistance) / fadeDistance, 0.0, 1.0);
        alpha *= fadeFactor;
    }

        if (enableFog) {
        float fogFactor = calculateFogFactor(fs_in.distance);
        finalColor = mix(fogColor, finalColor, fogFactor);
    }

    // Apply ACES Filmic Tone Mapping
    finalColor = ACESFilmicToneMapping(finalColor);
    // Apply Gamma correction
    finalColor = gammaCorrect(finalColor);

    // Combine material alpha with shape alpha
    float finalAlpha = alpha * shapeAlpha;

    // Ensure alpha is valid
    finalAlpha = clamp(finalAlpha, 0.0, 1.0);

    // Output final fragment color
    FragColor = vec4(finalColor, finalAlpha);
}
