#version 450 core

layout (location = 0) in vec3 aPos;  // Vertex position (cube vertices)

out vec3 WorldPos;  // World position for cubemap sampling

uniform mat4 projection;  // Projection matrix (90-degree FOV)
uniform mat4 view;        // View matrix for each cubemap face

void main() {
        // Pass world position to fragment shader for sampling calculations
    WorldPos = aPos;

    // Transform vertex to clip space
    gl_Position = projection * view * vec4(WorldPos, 1.0);
}
