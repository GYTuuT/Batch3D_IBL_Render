#version 450 core

// Quad vertices (NDC)
layout (location = 0) in vec2 aPos;
// Texture coordinates for sampling (passed to fragment shader)
layout (location = 1) in vec2 aTexCoords;

out vec2 TexCoords;

void main() {
    TexCoords = aTexCoords;
    // Render a full-screen quad by outputting NDC directly
    gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
}
