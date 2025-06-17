"""
Shader utilities for loading, compiling, and managing shader programs.
"""

import os
from typing import Dict, List
from dataclasses import dataclass, field
from OpenGL.GL import GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, glDeleteShader, glDeleteProgram, glGetUniformLocation
from OpenGL.GL import shaders as glshaders

@dataclass
class ShaderProgram:
    """
    Represents a compiled and linked OpenGL shader program.

    Attributes:
        program_id: The OpenGL ID for the shader program.
        locations: A dictionary mapping uniform names (str) to their OpenGL locations (int).
                   A location of -1 means the uniform was not found or not active.
    """
    program_id: int
    locations: Dict[str, int] = field(default_factory=dict)

_shader_program_cache: Dict[str, ShaderProgram] = {}

# Predefined lists of uniform names for each shader type
# These should match the uniforms actually used in the shaders
SHADER_UNIFORM_LISTS = {
    "base": [
        "model", "view", "projection", "normalMatrix", "lightPosition", "lightColor",
        "viewPosition", "ambientStrength", "specularStrength", "shininess",
        "useTexture", "materialDiffuse", "materialSpecular", "materialAlpha",
        "enableFog", "fogColor", "fogNear", "fogFar", "enableWireframe", "wireframeColor",
        "enableDistanceFade", "fadeDistance", "enableCustomShapes", "enableInstancing",
        "globalPointSize", "globalLineWidth", "diffuseTexture", "shapeTexture", "enableSSS"
    ],
    "skybox": ["view", "projection", "environmentMap"],
    "ibl_triangle": [
        "model", "view", "projection", "normalMatrix", "viewPos", "environmentMap", "brdfLUT",
        "irradianceMap", "prefilterMap", # Added irradianceMap and prefilterMap
        "ambientIntensity", "specularIntensity", "baseRoughness", "useTexture",
        "diffuseTexture", "materialDiffuse", "materialSpecular", "materialAlpha",
        "enableInstancing", "enableDistanceFade", "fadeDistance"
    ],
    "equirect_to_cubemap": ["projection", "view", "equirectangularMap"],
    "irradiance_convolution": ["projection", "view", "environmentMap"], # Added
    "prefilter_env_map": ["projection", "view", "environmentMap", "roughness"], # Added
    "brdf_lut": ["sampleCount"], # Add uniform for BRDF sample count
    "sss": [
        "model", "view", "projection", "normalMatrix", "lightPosition", "lightColor",
        "viewPosition", "enableSSS", "useTexture", "diffuseTexture", "environmentMap",
        "brdfLUT", "useIBL"
        # Material, SSSParams, SSSColor are attributes, not uniforms here
    ]
}


def load_shader_file(filename: str) -> str:
    """
    Load shader source code from a .glsl file located in the 'shaders' subdirectory.

    Args:
        filename: The name of the shader file (e.g., "base.vert").

    Returns:
        The shader source code as a string.

    Raises:
        FileNotFoundError: If the shader file does not exist.
        IOError: If there is an error reading the file.
    """
    shader_dir = os.path.join(os.path.dirname(__file__), 'shaders')
    shader_path = os.path.join(shader_dir, filename)
    if not os.path.exists(shader_path):
        raise FileNotFoundError(f"Shader file not found at {shader_path}")
    try:
        with open(shader_path, 'r', encoding='utf-8') as f:
            return f.read()
    except IOError as e:
        raise IOError(f"Failed to read shader file {shader_path}: {e}")


def create_shader_program(vs_source: str, fs_source: str, uniform_names: List[str]) -> ShaderProgram:
    """
    Create, compile, and link a shader program from vertex and fragment shader sources.
    Retrieves and stores locations for specified uniform names.

    Args:
        vs_source: The source code for the vertex shader.
        fs_source: The source code for the fragment shader.
        uniform_names: A list of uniform names (str) whose locations should be retrieved.

    Returns:
        A ShaderProgram object containing the program ID and uniform locations.

    Raises:
        RuntimeError: If shader compilation or linking fails.
                      The error message from PyOpenGL is included.
    """
    vs = fs = program_id = None
    try:
        vs = glshaders.compileShader(vs_source, GL_VERTEX_SHADER)
        fs = glshaders.compileShader(fs_source, GL_FRAGMENT_SHADER)
        program_id = glshaders.compileProgram(vs, fs)
        glDeleteShader(vs)
        glDeleteShader(fs)

        locations = {}
        if program_id is not None: # Ensure program_id is valid before getting locations
            for name in uniform_names:
                loc = glGetUniformLocation(program_id, name)
                # if loc == -1:
                #     print(f"Warning: Uniform '{name}' not found in shader program.") # Optional warning
                locations[name] = loc

        return ShaderProgram(program_id=program_id, locations=locations)
    except Exception as e:
        if vs:
            glDeleteShader(vs)
        if fs:
            glDeleteShader(fs)
        if program_id: # Check if program_id was assigned before trying to delete
            glDeleteProgram(program_id)
        # Shader compilation/linking error suppressed; exception will be raised
        raise RuntimeError(f"Failed to create shader program: {e}")


def get_shader_program(key: str) -> ShaderProgram:
    """
    Get a shader program by a predefined key, utilizing a cache.
    If not cached, it loads, compiles, and links the shader program.

    Args:
        key: A string key identifying the shader program (e.g., "base", "skybox").
             This key must be defined in SHADER_UNIFORM_LISTS and shader_file_map.

    Returns:
        A ShaderProgram object.

    Raises:
        ValueError: If the key is unknown or not properly configured.
        RuntimeError: If loading, compilation, or linking fails.
    """
    global _shader_program_cache
    if key in _shader_program_cache:
        return _shader_program_cache[key]
    if key not in SHADER_UNIFORM_LISTS:
        raise ValueError(f"Unknown shader key: {key}. Uniform list not defined.")

    shader_file_map = {
        "base": ("base.vert", "base.frag"),
        "skybox": ("skybox.vert", "skybox.frag"),
        "ibl_triangle": ("ibl_triangle.vert", "ibl_triangle.frag"),
        "equirect_to_cubemap": ("equirect_to_cubemap.vert", "equirect_to_cubemap.frag"),
        "irradiance_convolution": ("irradiance_convolution.vert", "irradiance_convolution.frag"), # Added
        "prefilter_env_map": ("prefilter_env_map.vert", "prefilter_env_map.frag"), # Added
        "brdf_lut": ("brdf.vert", "brdf.frag"), # Assuming brdf.vert exists or is simple pass-through
        "sss": ("sss.vert", "sss.frag")
    }
    if key not in shader_file_map:
        raise ValueError(f"Unknown shader key: {key}. File mapping not defined.")

    vs_filename, fs_filename = shader_file_map[key]
    uniform_names = SHADER_UNIFORM_LISTS[key]
    try:
        vs_source = load_shader_file(vs_filename)
        fs_source = load_shader_file(fs_filename)
        shader_prog_obj = create_shader_program(vs_source, fs_source, uniform_names)
        _shader_program_cache[key] = shader_prog_obj
        return shader_prog_obj
    except (FileNotFoundError, IOError) as e:
        raise RuntimeError(f"Failed to load shader files for '{key}' ({vs_filename}, {fs_filename}): {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to create shader program for '{key}': {e}")

def clear_shader_cache() -> None:
    """
    Clear the shader program cache and delete the associated OpenGL shader programs
    from the GPU. This is crucial for resource management, especially when the OpenGL
    context is being destroyed or if shaders need to be recompiled and reloaded dynamically.
    Failure to delete programs can lead to GPU memory leaks.
    """
    global _shader_program_cache
    # Iterate over a copy of values if modification during iteration is a concern,
    # but here we are just deleting based on stored IDs.
    for shader_prog_obj in _shader_program_cache.values():
        try:
            if shader_prog_obj.program_id is not None: # Ensure program_id is valid before attempting deletion
                glDeleteProgram(shader_prog_obj.program_id)
        except Exception as e: # pylint: disable=broad-except
            # Warning suppressed: Failed to delete shader program ID on GPU (ignored)
            pass
    _shader_program_cache.clear()
    # Shader cache cleared; GPU programs deletion attempted

