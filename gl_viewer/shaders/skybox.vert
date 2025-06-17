#version 450 core

layout (location = 0) in vec3 aPos;  // Vertex position (cube vertices)

out vec3 TexCoords;  // 3D texture coordinates for cubemap sampling

uniform mat4 projection;  // Projection matrix
uniform mat4 view;        // View matrix (without translation for infinite distance effect)

void main() {
        // Use position as texture coordinates directly
    TexCoords = aPos;

    // Remove translation from view matrix for skybox
    mat4 rotView = mat4(mat3(view));
    vec4 pos = projection * rotView * vec4(aPos, 1.0);

    // This ensures skybox is always behind all other geometry
    gl_Position = pos.xyww;
}
