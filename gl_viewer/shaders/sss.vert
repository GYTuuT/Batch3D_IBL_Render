#version 450 core

// Vertex attributes - matching the enhanced layout from GLwidget
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec3 aNormal;
layout (location = 3) in vec2 aTexCoord;
layout (location = 4) in float aPointSize;     // Not used directly but keeps layout consistent
layout (location = 5) in float aLineWidth;     // Not used directly but keeps layout consistent
layout (location = 6) in int aShapeType;       // Not used directly but keeps layout consistent
layout (location = 7) in vec4 aInstanceData;   // Not used directly but keeps layout consistent
layout (location = 8) in vec3 aMaterial;       // metallic, roughness, ao
layout (location = 9) in vec4 aSSSParams;      // strength, distortion, power, scale
layout (location = 10) in vec3 aSSSColor;

out VS_OUT {
    vec3 FragPos;
    vec3 Normal;
    vec2 TexCoords;
    vec3 Color;
    vec3 Material; // metallic, roughness, ao
    vec4 SSSParams; // strength, distortion, power, scale
    vec3 SSSColor;
} vs_out;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat3 normalMatrix; // For transforming normals

void main()
{
    vs_out.FragPos = vec3(model * vec4(aPos, 1.0));
    vs_out.Normal = normalize(normalMatrix * aNormal);
    vs_out.TexCoords = aTexCoord;
    vs_out.Color = aColor;
    vs_out.Material = aMaterial;
    vs_out.SSSParams = aSSSParams;
    vs_out.SSSColor = aSSSColor;

    gl_Position = projection * view * vec4(vs_out.FragPos, 1.0);
}
