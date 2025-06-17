#version 450 core

uniform int sampleCount; // Number of samples for BRDF integration

out vec2 FragColor; // Output is two channels (scale and bias for Fresnel)
in vec2 TexCoords;  // Input NdotV and roughness

const float PI = 3.14159265359;

// GGX/Smith Microfacet BRDF functions
// Normal Distribution Function
// Args:
//   N: Surface normal (vec3)
//   H: Half-vector (vec3)
//   roughness: Surface roughness (float, [0,1])
// Returns:
//   GGX NDF value (float)
float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / max(denom, 0.0000001); // Prevent division by zero
}

// Geometry Function (Schlick-GGX)
// Args:
//   NdotV: Dot product of normal and view/light vector (float)
//   roughness: Surface roughness (float, [0,1])
// Returns:
//   Geometry term for one direction (float)
float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0; // For direct lighting

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}

// Geometry Function (Smith)
// Args:
//   N: Surface normal (vec3)
//   V: View vector (vec3)
//   L: Light vector (vec3)
//   roughness: Surface roughness (float, [0,1])
// Returns:
//   Combined geometry term (float)
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

// Hammersley sequence for quasi-Monte Carlo integration
// Args:
//   i: Sample index (uint)
//   N: Total number of samples (uint)
// Returns:
//   2D Hammersley point (vec2)
vec2 Hammersley(uint i, uint N) {
    uint bits = (i << 16u) | (i >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    float rdi = float(bits) * 2.3283064365386963e-10; // / 0x100000000
    return vec2(float(i) / float(N), rdi);
}

// Importance sampling for GGX distribution
// Args:
//   Xi: 2D random number (vec2, [0,1]^2)
//   N: Surface normal (vec3)
//   roughness: Surface roughness (float, [0,1])
// Returns:
//   Importance-sampled half-vector H (vec3)
vec3 ImportanceSampleGGX(vec2 Xi, vec3 N, float roughness) {
    float a = roughness * roughness;

    float phi = 2.0 * PI * Xi.x;
    float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a*a - 1.0) * Xi.y));
    float sinTheta = sqrt(1.0 - cosTheta*cosTheta);

    // Spherical to cartesian coordinates
    vec3 H;
    H.x = cos(phi) * sinTheta;
    H.y = sin(phi) * sinTheta;
    H.z = cosTheta;

    // Tangent-space to world-space
    vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, N));
    vec3 bitangent = cross(N, tangent);

    vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
    return normalize(sampleVec);
}

// BRDF integration for environment lighting (pre-calculates scale and bias for Fresnel term)
// Args:
//   NdotV: Cosine of the angle between the normal and the view vector (float)
//   roughness: Surface roughness (float, [0,1])
// Returns:
//   Integrated BRDF values (scale, bias) for Fresnel (vec2)
vec2 IntegrateBRDF(float NdotV, float roughness) {
    vec3 V;
    V.x = sqrt(1.0 - NdotV*NdotV);
    V.y = 0.0;
    V.z = NdotV;

    float A = 0.0;
    float B = 0.0;

    vec3 N = vec3(0.0, 0.0, 1.0);

    uint SAMPLE_COUNT = uint(sampleCount);
    for(uint i = 0u; i < SAMPLE_COUNT; ++i) {
        vec2 Xi = Hammersley(i, SAMPLE_COUNT);
        vec3 H = ImportanceSampleGGX(Xi, N, roughness);
        vec3 L = normalize(2.0 * dot(V, H) * H - V);

        float NdotL = max(L.z, 0.0);
        float NdotH = max(H.z, 0.0);
        float VdotH = max(dot(V, H), 0.0);

        if(NdotL > 0.0) {
            float G = GeometrySmith(N, V, L, roughness);
            float G_Vis = (G * VdotH) / (NdotH * NdotV);
            float Fc = pow(1.0 - VdotH, 5.0);

            A += (1.0 - Fc) * G_Vis;
            B += Fc * G_Vis;
        }
    }
    A /= float(SAMPLE_COUNT);
    B /= float(SAMPLE_COUNT);
    return vec2(A, B);
}

void main() {
    vec2 integratedBRDF = IntegrateBRDF(TexCoords.x, TexCoords.y);
    FragColor = integratedBRDF;
}
