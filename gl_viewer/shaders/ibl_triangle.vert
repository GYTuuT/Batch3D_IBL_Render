#version 450 core

// Must match the enhanced buffer layout exactly
layout (location = 0) in vec3 aPos;           // Vertex position
layout (location = 1) in vec3 aColor;         // Vertex color
layout (location = 2) in vec3 aNormal;        // Vertex normal
layout (location = 3) in vec2 aTexCoord;      // Texture coordinates
layout (location = 4) in float aPointSize;    // Per-vertex point size (unused for triangles)
layout (location = 5) in float aLineWidth;    // Per-vertex line width (unused for triangles)
layout (location = 6) in int aShapeType;      // Shape type index (unused for triangles)
layout (location = 7) in vec4 aInstanceData;  // Instancing data [offset.xyz, scale]
layout (location = 8) in vec3 aMaterial;      // Material properties [metallic, roughness, ao]

out VS_OUT {
    vec3 worldPos;      // World space position
    vec3 viewPos;       // View space position
    vec3 normal;        // World space normal
    vec3 color;         // Vertex color
    vec2 texCoord;      // Texture coordinates
    vec3 material;      // Material properties [metallic, roughness, ao]
    vec4 instanceData;  // Instance data for potential effects
    float distance;     // Distance from camera
} vs_out;

uniform mat4 model;        // Model matrix
uniform mat4 view;         // View matrix
uniform mat4 projection;   // Projection matrix
uniform vec3 viewPosition; // Camera position in world space

uniform bool enableInstancing;  // Enable instancing support

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

        // Transform normal to world space (handles non-uniform scaling properly)
    mat3 normalMatrix = mat3(transpose(inverse(model)));
    vs_out.normal = normalize(normalMatrix * aNormal);

        vs_out.color = aColor;
    vs_out.texCoord = aTexCoord;
    vs_out.material = aMaterial;
    vs_out.instanceData = aInstanceData;

        vs_out.distance = length(viewPosition - vs_out.worldPos);
}
