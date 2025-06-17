#version 450 core

layout (location = 0) in vec3 aPos;           // Vertex position
layout (location = 1) in vec3 aColor;         // Vertex color
layout (location = 2) in vec3 aNormal;        // Vertex normal
layout (location = 3) in vec2 aTexCoord;      // Texture coordinates
layout (location = 4) in float aPointSize;    // Per-vertex point size
layout (location = 5) in float aLineWidth;    // Per-vertex line width
layout (location = 6) in int aShapeType;      // Shape type index
layout (location = 7) in vec4 aInstanceData;  // Instancing data [offset.xyz, scale]

out VS_OUT {
    vec3 worldPos;      // World space position
    vec3 viewPos;       // View space position
    vec3 normal;        // World space normal
    vec3 color;         // Vertex color
    vec2 texCoord;      // Texture coordinates
    float distance;     // Distance from camera
    float pointSize;    // Final point size
    float lineWidth;    // Final line width
    flat int shapeType; // Shape type (flat for no interpolation)
    vec4 instanceData;  // Instance data
} vs_out;

uniform mat4 model;        // Model matrix
uniform mat4 view;         // View matrix
uniform mat4 projection;   // Projection matrix
uniform mat3 normalMatrix; // Normal transformation matrix
uniform vec3 viewPosition; // Camera position in world space

uniform float globalPointSize;  // Global point size multiplier
uniform float globalLineWidth;  // Global line width multiplier
uniform bool enableInstancing;  // Enable instancing support

const float BASE_POINT_SIZE = 50.0;      // Base size for distance calculation
const float MIN_POINT_SIZE = 1.0;        // Minimum point size
const float MAX_POINT_SIZE = 100.0;      // Maximum point size
const float SHAPE_SIZE_MULTIPLIERS[7] = float[](
    1.0,   // Circle (0)
    1.2,   // Square (1)
    0.8,   // Diamond (2)
    1.1,   // Star (3)
    1.0,   // Triangle (4)
    1.3,   // Cross (5)
    1.0    // Custom (6)
);

void main() {
        vec3 position = aPos;

    // Apply instancing transformations if enabled
    if (enableInstancing) {
        // Apply instance offset and scale
        position = position * aInstanceData.w + aInstanceData.xyz;
    }

    // Transform to world space
    vec4 worldPosition = model * vec4(position, 1.0);
    vs_out.worldPos = worldPosition.xyz;

    // Transform to view space
    vs_out.viewPos = (view * worldPosition).xyz;

    // Final clip space position
    gl_Position = projection * view * worldPosition;

        // Transform normal to world space (handles non-uniform scaling)
    vs_out.normal = normalMatrix * aNormal;

    // Pass through vertex attributes
    vs_out.color = aColor;
    vs_out.texCoord = aTexCoord;
    vs_out.instanceData = aInstanceData;
    vs_out.shapeType = aShapeType;

        // Calculate distance from camera for size/fade effects
    vs_out.distance = length(viewPosition - vs_out.worldPos);

    // Calculate final point and line sizes
    vs_out.pointSize = aPointSize * globalPointSize;
    vs_out.lineWidth = aLineWidth * globalLineWidth;

        // Calculate distance-based size factor
    float distanceFactor = BASE_POINT_SIZE / max(vs_out.distance, 1.0);

    // Apply shape-specific size multiplier
    float shapeMultiplier = 1.0;
    if (aShapeType >= 0 && aShapeType < 7) {
        shapeMultiplier = SHAPE_SIZE_MULTIPLIERS[aShapeType];
    }

    // Calculate final OpenGL point size with clamping
    float finalPointSize = distanceFactor * vs_out.pointSize * shapeMultiplier;
    gl_PointSize = clamp(finalPointSize, MIN_POINT_SIZE, MAX_POINT_SIZE);
}
