"""
Shader controllers and Image-Based Lighting (IBL) renderer.
"""

import array
import ctypes
import os
from typing import Dict, List, Optional, Tuple, Union

import Imath
import numpy as np
import OpenEXR
import pyrr
from OpenGL.GL import *  # type: ignore[import-untyped, unused-ignore]
from OpenGL.GLU import gluErrorString

from .shader_utils import get_shader_program, clear_shader_cache, ShaderProgram # Import ShaderProgram
from .config import RenderingDefaults


class BaseShaderController:
    """
    Controls enhanced shader parameters and visual effects.
    Manages uniforms for lighting, materials, fog, wireframe, etc.
    """

    def __init__(self,
                 light_position: Optional[Union[List[float], np.ndarray]] = None, # (3,)
                 light_color: Optional[Union[List[float], np.ndarray]] = None, # (3,)
                 ambient_strength: Optional[float] = None,
                 specular_strength: Optional[float] = None,
                 shininess: Optional[float] = None,
                 material_diffuse: Optional[Union[List[float], np.ndarray]] = None, # (3,)
                 material_specular: Optional[Union[List[float], np.ndarray]] = None, # (3,)
                 material_alpha: Optional[float] = None,
                 enable_fog: Optional[bool] = None,
                 fog_color: Optional[Union[List[float], np.ndarray]] = None, # (3,)
                 fog_near: Optional[float] = None,
                 fog_far: Optional[float] = None,
                 enable_wireframe: Optional[bool] = None,
                 wireframe_color: Optional[Union[List[float], np.ndarray]] = None, # (3,)
                 enable_distance_fade: Optional[bool] = None,
                 fade_distance: Optional[float] = None,
                 enable_custom_shapes: Optional[bool] = None,
                 enable_instancing: Optional[bool] = None,
                 global_point_size: Optional[float] = None,
                 global_line_width: Optional[float] = None,
                 use_texture: Optional[bool] = None,
                 enable_sss: Optional[bool] = None
                 ):
        """
        Initialize shader controller with default or provided parameters.

        Args:
            light_position: World space light position, e.g., [5.0, 10.0, 5.0].
            light_color: Light RGB color, e.g., [1.0, 1.0, 1.0].
            ambient_strength: Ambient light intensity [0,1].
            specular_strength: Specular reflection intensity [0,1].
            shininess: Specular shininess exponent.
            material_diffuse: Default diffuse material color.
            material_specular: Default specular material color.
            material_alpha: Default material alpha [0,1].
            enable_fog: True to enable fog.
            fog_color: Fog RGB color.
            fog_near: Fog start distance.
            fog_far: Fog end distance.
            enable_wireframe: True to enable wireframe overlay.
            wireframe_color: Wireframe line RGB color.
            enable_distance_fade: True to enable distance-based alpha fading.
            fade_distance: Distance for 50% fade.
            enable_custom_shapes: True to enable custom point shapes.
            enable_instancing: True to enable instancing effects (shader-dependent).
            global_point_size: Default point size.
            global_line_width: Default line width.
            use_texture: True to enable texture mapping globally.
            enable_sss: True to enable Subsurface Scattering effect.
        """
        # Lighting parameters
        self.light_position = np.array(light_position, dtype=np.float32) if light_position is not None else RenderingDefaults.DEFAULT_LIGHT_POSITION.copy()
        self.light_color = np.array(light_color, dtype=np.float32) if light_color is not None else RenderingDefaults.DEFAULT_LIGHT_COLOR.copy()
        self.ambient_strength = ambient_strength if ambient_strength is not None else RenderingDefaults.DEFAULT_AMBIENT_STRENGTH
        self.specular_strength = specular_strength if specular_strength is not None else RenderingDefaults.DEFAULT_SPECULAR_STRENGTH
        self.shininess = shininess if shininess is not None else RenderingDefaults.DEFAULT_SHININESS

        # Material parameters
        self.material_diffuse = np.array(material_diffuse, dtype=np.float32) if material_diffuse is not None else np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.material_specular = np.array(material_specular, dtype=np.float32) if material_specular is not None else np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.material_alpha = material_alpha if material_alpha is not None else 1.0

        # Fog effect parameters
        self.enable_fog = enable_fog if enable_fog is not None else False
        self.fog_color = np.array(fog_color, dtype=np.float32) if fog_color is not None else RenderingDefaults.DEFAULT_FOG_COLOR.copy()
        self.fog_near = fog_near if fog_near is not None else RenderingDefaults.DEFAULT_FOG_NEAR
        self.fog_far = fog_far if fog_far is not None else RenderingDefaults.DEFAULT_FOG_FAR

        # Wireframe effect parameters
        self.enable_wireframe = enable_wireframe if enable_wireframe is not None else False
        self.wireframe_color = np.array(wireframe_color, dtype=np.float32) if wireframe_color is not None else RenderingDefaults.DEFAULT_WIREFRAME_COLOR.copy()

        # Distance-based effects
        self.enable_distance_fade = enable_distance_fade if enable_distance_fade is not None else False
        self.fade_distance = fade_distance if fade_distance is not None else RenderingDefaults.DEFAULT_FADE_DISTANCE

        # Enhanced rendering features
        self.enable_custom_shapes = enable_custom_shapes if enable_custom_shapes is not None else True
        self.enable_instancing = enable_instancing if enable_instancing is not None else False
        self.global_point_size = global_point_size if global_point_size is not None else RenderingDefaults.DEFAULT_POINT_SIZE
        self.global_line_width = global_line_width if global_line_width is not None else RenderingDefaults.DEFAULT_LINE_WIDTH

        # Texture support
        self.use_texture = use_texture if use_texture is not None else False
        self.diffuse_texture_id: Optional[int] = None
        self.shape_texture_id: Optional[int] = None

        # Subsurface Scattering parameters
        self.enable_sss = enable_sss if enable_sss is not None else False

        # Internal cache for texture bindings
        self._active_texture_unit: Optional[int] = -1  # Uninitialized state
        self._bound_textures: Dict[int, Dict[int, Optional[int]]] = {} # unit -> {target: texture_id}
        # Reference to IBL renderer for binding IBL textures
        self.ibl_controller: Optional[IBLRenderer] = None

    def set_lighting_params(self, position: Optional[np.ndarray] = None, # (3,)
                           color: Optional[np.ndarray] = None, # (3,)
                           ambient: Optional[float] = None,
                           specular: Optional[float] = None,
                           shininess: Optional[float] = None) -> None:
        """
        Update lighting parameters.

        Args:
            position: Light position in world space (NumPy array, shape (3,)).
            color: Light color (RGB NumPy array, shape (3,)).
            ambient: Ambient light strength [0,1].
            specular: Specular reflection strength [0,1].
            shininess: Specular shininess exponent.
        """
        if position is not None:
            self.light_position = np.array(position, dtype=np.float32)
        if color is not None:
            self.light_color = np.array(color, dtype=np.float32)
        if ambient is not None:
            self.ambient_strength = max(0.0, ambient)
        if specular is not None:
            self.specular_strength = max(0.0, specular)
        if shininess is not None:
            self.shininess = max(1.0, shininess)

    def set_material_params(self, diffuse: Optional[np.ndarray] = None, # (3,)
                           specular: Optional[np.ndarray] = None, # (3,)
                           alpha: Optional[float] = None) -> None:
        """
        Update global material parameters.

        Args:
            diffuse: Diffuse material color (RGB NumPy array, shape (3,)).
            specular: Specular material color (RGB NumPy array, shape (3,)).
            alpha: Material transparency [0,1].
        """
        if diffuse is not None:
            self.material_diffuse = np.array(diffuse, dtype=np.float32)
        if specular is not None:
            self.material_specular = np.array(specular, dtype=np.float32)
        if alpha is not None:
            self.material_alpha = np.clip(alpha, 0.0, 1.0)

    def set_fog_params(self, enabled: bool, color: Optional[np.ndarray] = None, # (3,)
                      near: Optional[float] = None, far: Optional[float] = None) -> None:
        """
        Update fog effect parameters.

        Args:
            enabled: Enable/disable fog effect.
            color: Fog color (RGB NumPy array, shape (3,)).
            near: Fog start distance.
            far: Fog end distance.
        """
        self.enable_fog = enabled
        if color is not None:
            self.fog_color = np.array(color, dtype=np.float32)
        if near is not None:
            self.fog_near = max(0.0, near)
        if far is not None:
            self.fog_far = max(self.fog_near + 0.1, far)

    def set_wireframe_params(self, enabled: bool, color: Optional[np.ndarray] = None) -> None: # (3,)
        """
        Update wireframe effect parameters.

        Args:
            enabled: Enable/disable wireframe effect.
            color: Wireframe color (RGB NumPy array, shape (3,)).
        """
        self.enable_wireframe = enabled
        if color is not None:
            self.wireframe_color = np.array(color, dtype=np.float32)

    def set_sss_enabled(self, enabled: bool) -> None:
        """
        Enable or disable Subsurface Scattering effect.

        Args:
            enabled: True to enable SSS, False to disable.
        """
        self.enable_sss = enabled

    def apply_uniforms(self, shader_prog: ShaderProgram, view_position: np.ndarray) -> bool: # view_position (3,)
        """
        Apply all shader uniforms to the given program using cached locations.

        Args:
            shader_prog: ShaderProgram object containing program ID and cached locations.
            view_position: Camera position in world space (NumPy array, shape (3)).

        Returns:
            True if successful, False if errors occurred (e.g., shader_prog.program_id is None).
        """
        if shader_prog.program_id is None:
            return False
        try:
            # Lighting uniforms
            self._set_uniform_3fv(shader_prog.locations.get("lightPosition", -1), self.light_position)
            self._set_uniform_3fv(shader_prog.locations.get("lightColor", -1), self.light_color)
            self._set_uniform_3fv(shader_prog.locations.get("viewPosition", -1), view_position)
            self._set_uniform_1f(shader_prog.locations.get("ambientStrength", -1), self.ambient_strength)
            self._set_uniform_1f(shader_prog.locations.get("specularStrength", -1), self.specular_strength)
            self._set_uniform_1f(shader_prog.locations.get("shininess", -1), self.shininess)

            # Material uniforms
            self._set_uniform_1i(shader_prog.locations.get("useTexture", -1), self.use_texture)
            self._set_uniform_3fv(shader_prog.locations.get("materialDiffuse", -1), self.material_diffuse)
            self._set_uniform_3fv(shader_prog.locations.get("materialSpecular", -1), self.material_specular)
            self._set_uniform_1f(shader_prog.locations.get("materialAlpha", -1), self.material_alpha)

            # Effect uniforms
            self._set_uniform_1i(shader_prog.locations.get("enableFog", -1), self.enable_fog)
            self._set_uniform_3fv(shader_prog.locations.get("fogColor", -1), self.fog_color)
            self._set_uniform_1f(shader_prog.locations.get("fogNear", -1), self.fog_near)
            self._set_uniform_1f(shader_prog.locations.get("fogFar", -1), self.fog_far)

            self._set_uniform_1i(shader_prog.locations.get("enableWireframe", -1), self.enable_wireframe)
            self._set_uniform_3fv(shader_prog.locations.get("wireframeColor", -1), self.wireframe_color)

            self._set_uniform_1i(shader_prog.locations.get("enableDistanceFade", -1), self.enable_distance_fade)
            self._set_uniform_1f(shader_prog.locations.get("fadeDistance", -1), self.fade_distance)

            # Enhanced features uniforms
            self._set_uniform_1i(shader_prog.locations.get("enableCustomShapes", -1), self.enable_custom_shapes)
            self._set_uniform_1i(shader_prog.locations.get("enableInstancing", -1), self.enable_instancing)
            self._set_uniform_1f(shader_prog.locations.get("globalPointSize", -1), self.global_point_size)
            self._set_uniform_1f(shader_prog.locations.get("globalLineWidth", -1), self.global_line_width)

            # SSS uniform
            self._set_uniform_1i(shader_prog.locations.get("enableSSS", -1), self.enable_sss)

            # Texture binding
            self._bind_textures(shader_prog) # Pass ShaderProgram to _bind_textures

            return True

        except Exception as e:
            return False

    def _set_uniform_1f(self, location: int, value: float) -> None:
        """
        Safely set a float uniform if its location is valid.

        Args:
            location: Uniform location ID.
            value: Float value to set.
        """
        if location != -1:
            glUniform1f(location, value)

    def _set_uniform_1i(self, location: int, value: Union[bool, int]) -> None:
        """
        Safely set an integer/boolean uniform if its location is valid.

        Args:
            location: Uniform location ID.
            value: Integer or boolean value to set.
        """
        if location != -1:
            glUniform1i(location, int(value))

    def _set_uniform_3fv(self, location: int, value: np.ndarray) -> None: # value (3,)
        """
        Safely set a vec3 uniform if its location is valid.

        Args:
            location: Uniform location ID.
            value: NumPy array (shape (3,)) for the vec3 uniform.
        """
        if location != -1:
            glUniform3fv(location, 1, value)

    def _activate_texture_unit(self, unit: int) -> None:
        """Activates a texture unit if it's not already active."""
        if self._active_texture_unit != unit:
            glActiveTexture(int(GL_TEXTURE0) + unit)
            self._active_texture_unit = unit

    def _bind_texture_on_current_unit(self, target: int, texture_id: Optional[int]) -> None:
        """Binds a texture to the currently active unit if not already bound."""
        # Assumes _activate_texture_unit was called before this.
        if self._active_texture_unit is None:
            return
        current_unit_bindings = self._bound_textures.get(self._active_texture_unit, {})
        if current_unit_bindings.get(target) != texture_id:
            glBindTexture(target, texture_id if texture_id is not None else 0)
            if self._active_texture_unit not in self._bound_textures:
                self._bound_textures[self._active_texture_unit] = {}
            self._bound_textures[self._active_texture_unit][target] = texture_id

    def _bind_textures(self, shader_prog: ShaderProgram) -> None:
        """
        Bind diffuse and shape textures to appropriate texture units using cached locations.

        Args:
            shader_prog: ShaderProgram object with cached uniform locations.
        """
        # Diffuse texture binding
        if self.use_texture and self.diffuse_texture_id is not None:
            self._activate_texture_unit(RenderingDefaults.TEXTURE_UNIT_DIFFUSE)
            self._bind_texture_on_current_unit(GL_TEXTURE_2D, self.diffuse_texture_id)
            self._set_uniform_1i(shader_prog.locations.get("diffuseTexture", -1), RenderingDefaults.TEXTURE_UNIT_DIFFUSE)

        # Shape texture binding
        if self.shape_texture_id is not None:
            self._activate_texture_unit(RenderingDefaults.TEXTURE_UNIT_SHAPE)
            self._bind_texture_on_current_unit(GL_TEXTURE_2D, self.shape_texture_id)
            self._set_uniform_1i(shader_prog.locations.get("shapeTexture", -1), RenderingDefaults.TEXTURE_UNIT_SHAPE)

        # Bind IBL textures if available and uniforms present
        if self.ibl_controller is not None:
            # BRDF LUT
            loc = shader_prog.locations.get("brdfLUT", -1)
            if loc != -1 and self.ibl_controller.brdf_lut_texture is not None:
                self._activate_texture_unit(RenderingDefaults.TEXTURE_UNIT_BRDF_LUT)
                self._bind_texture_on_current_unit(GL_TEXTURE_2D, self.ibl_controller.brdf_lut_texture)
                self._set_uniform_1i(loc, RenderingDefaults.TEXTURE_UNIT_BRDF_LUT)
            # Irradiance map
            loc = shader_prog.locations.get("irradianceMap", -1)
            if loc != -1 and self.ibl_controller.irradiance_map_texture is not None:
                self._activate_texture_unit(RenderingDefaults.TEXTURE_UNIT_IRRADIANCE_MAP)
                self._bind_texture_on_current_unit(GL_TEXTURE_CUBE_MAP, self.ibl_controller.irradiance_map_texture)
                self._set_uniform_1i(loc, RenderingDefaults.TEXTURE_UNIT_IRRADIANCE_MAP)
            # Prefilter map
            loc = shader_prog.locations.get("prefilterMap", -1)
            if loc != -1 and self.ibl_controller.prefilter_map_texture is not None:
                self._activate_texture_unit(RenderingDefaults.TEXTURE_UNIT_PREFILTER_MAP)
                self._bind_texture_on_current_unit(GL_TEXTURE_CUBE_MAP, self.ibl_controller.prefilter_map_texture)
                self._set_uniform_1i(loc, RenderingDefaults.TEXTURE_UNIT_PREFILTER_MAP)
            # Environment map for IBL in SSS or other shaders
            loc = shader_prog.locations.get("environmentMap", -1)
            if loc != -1 and self.ibl_controller.env_cubemap_texture is not None:
                self._activate_texture_unit(RenderingDefaults.TEXTURE_UNIT_ENVIRONMENT)
                self._bind_texture_on_current_unit(GL_TEXTURE_CUBE_MAP, self.ibl_controller.env_cubemap_texture)
                self._set_uniform_1i(loc, RenderingDefaults.TEXTURE_UNIT_ENVIRONMENT)

    def reset_texture_binding_cache(self) -> None:
        """Resets the internal cache for texture bindings."""
        self._active_texture_unit = -1
        self._bound_textures.clear()


class IBLRenderer:
    """
    Handles Image-Based Lighting (IBL) rendering setup and resources.
    Manages HDR environment maps, cubemap generation, BRDF LUT, and skybox.
    """

    def __init__(self, exr_filepath: Optional[str],
                 ambient_intensity: Optional[float] = None,
                 specular_intensity: Optional[float] = None,
                 base_roughness: Optional[float] = None,
                 cubemap_size: Optional[int] = None,
                 brdf_lut_size: Optional[int] = None
                 ):
        """
        Initialize IBL renderer.

        Args:
            exr_filepath: Path to HDR environment map file (.exr), or None to disable IBL.
            ambient_intensity: Default ambient intensity for IBL [0, inf).
            specular_intensity: Default specular intensity for IBL [0, inf).
            base_roughness: Default base roughness for IBL materials [0,1].
            cubemap_size: Resolution for each face of the generated environment cubemap (e.g., 512).
            brdf_lut_size: Resolution for the BRDF lookup texture (e.g., 512).
        """
        self.enabled = exr_filepath is not None
        self.exr_filepath = exr_filepath

        # OpenGL texture resources
        self.equirectangular_map_tex: Optional[int] = None
        self.env_cubemap_texture: Optional[int] = None
        self.brdf_lut_texture: Optional[int] = None
        self.irradiance_map_texture: Optional[int] = None
        self.prefilter_map_texture: Optional[int] = None

        # Geometry for rendering (cube for skybox, quad for LUTs)
        self.cube_vao: Optional[int] = None
        self.cube_vbo: Optional[int] = None

        # Quad geometry for BRDF LUT generation
        self.quad_vao: Optional[int] = None
        self.quad_vbo: Optional[int] = None

        # Framebuffer for cubemap generation
        self.capture_fbo: Optional[int] = None
        self.capture_rbo: Optional[int] = None

        # IBL rendering parameters
        self.ambient_intensity = ambient_intensity if ambient_intensity is not None else RenderingDefaults.DEFAULT_IBL_AMBIENT_INTENSITY
        self.specular_intensity = specular_intensity if specular_intensity is not None else RenderingDefaults.DEFAULT_IBL_SPECULAR_INTENSITY
        self.base_roughness = base_roughness if base_roughness is not None else RenderingDefaults.DEFAULT_IBL_BASE_ROUGHNESS

        # Cubemap generation settings
        self.cubemap_size = cubemap_size if cubemap_size is not None else RenderingDefaults.DEFAULT_IBL_CUBEMAP_SIZE
        self.cubemap_mip_levels = int(np.log2(self.cubemap_size)) + 1 if self.cubemap_size > 0 else 9
        self.brdf_lut_size = brdf_lut_size if brdf_lut_size is not None else RenderingDefaults.DEFAULT_IBL_BRDF_LUT_SIZE

        # Matrices for cubemap capture
        self.capture_projection = pyrr.matrix44.create_perspective_projection_matrix(
            RenderingDefaults.IBL_CUBEMAP_FOV_DEGREES,
            RenderingDefaults.IBL_CUBEMAP_ASPECT_RATIO,
            RenderingDefaults.IBL_CUBEMAP_NEAR_PLANE,
            RenderingDefaults.IBL_CUBEMAP_FAR_PLANE
        )
        self.capture_views = [
            pyrr.matrix44.create_look_at(pyrr.Vector3([0,0,0]), pyrr.Vector3([ 1,0,0]), pyrr.Vector3([0,-1,0])),
            pyrr.matrix44.create_look_at(pyrr.Vector3([0,0,0]), pyrr.Vector3([-1,0,0]), pyrr.Vector3([0,-1,0])),
            pyrr.matrix44.create_look_at(pyrr.Vector3([0,0,0]), pyrr.Vector3([0, 1,0]), pyrr.Vector3([0,0, 1])),
            pyrr.matrix44.create_look_at(pyrr.Vector3([0,0,0]), pyrr.Vector3([0,-1,0]), pyrr.Vector3([0,0,-1])),
            pyrr.matrix44.create_look_at(pyrr.Vector3([0,0,0]), pyrr.Vector3([0,0, 1]), pyrr.Vector3([0,-1,0])),
            pyrr.matrix44.create_look_at(pyrr.Vector3([0,0,0]), pyrr.Vector3([0,0,-1]), pyrr.Vector3([0,-1,0]))
        ]

    def check_gl_error(self, tag: str = "") -> bool:
        """
        Check for OpenGL errors and log them.

        Args:
            tag: A descriptive tag for the context of the error check.

        Returns:
            True if no OpenGL error occurred, False otherwise.
        """
        err = glGetError()
        if err != GL_NO_ERROR:
            err_str = gluErrorString(err)
            if isinstance(err_str, bytes):
                err_str = err_str.decode('utf-8', 'ignore')
            return False
        return True

    def setup_cube_geometry(self) -> bool:
        """
        Setup cube geometry (VAO/VBO) for skybox rendering and cubemap generation.

        Returns:
            True if successful, False on error.
        """
        try:
            # Cube vertices for skybox (36 vertices for 12 triangles)
            vertices = np.array([
                # Front face
                -1.0,  1.0, -1.0, -1.0, -1.0, -1.0,  1.0, -1.0, -1.0,
                 1.0, -1.0, -1.0,  1.0,  1.0, -1.0, -1.0,  1.0, -1.0,
                # Back face
                -1.0, -1.0,  1.0, -1.0, -1.0, -1.0, -1.0,  1.0, -1.0,
                -1.0,  1.0, -1.0, -1.0,  1.0,  1.0, -1.0, -1.0,  1.0,
                # Right face
                 1.0, -1.0, -1.0,  1.0, -1.0,  1.0,  1.0,  1.0,  1.0,
                 1.0,  1.0,  1.0,  1.0,  1.0, -1.0,  1.0, -1.0, -1.0,
                # Left face
                -1.0, -1.0,  1.0, -1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
                 1.0,  1.0,  1.0,  1.0, -1.0,  1.0, -1.0, -1.0,  1.0,
                # Top face
                -1.0,  1.0, -1.0,  1.0,  1.0, -1.0,  1.0,  1.0,  1.0,
                 1.0,  1.0,  1.0, -1.0,  1.0,  1.0, -1.0,  1.0, -1.0,
                # Bottom face
                -1.0, -1.0, -1.0, -1.0, -1.0,  1.0,  1.0, -1.0, -1.0,
                 1.0, -1.0, -1.0, -1.0, -1.0,  1.0,  1.0, -1.0,  1.0
            ], dtype=np.float32)

            # Create and setup VAO/VBO
            self.cube_vao = glGenVertexArrays(1)
            self.cube_vbo = glGenBuffers(1)

            glBindVertexArray(self.cube_vao)
            glBindBuffer(GL_ARRAY_BUFFER, self.cube_vbo)
            glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

            # Setup vertex attributes
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                                3 * sizeof(GLfloat), ctypes.c_void_p(0))

            glBindVertexArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

            return True

        except Exception as e:
            return False

    def setup_quad_geometry(self) -> bool:
        """
        Setup quad geometry (VAO/VBO) for rendering the BRDF LUT.

        Returns:
            True if successful, False on error.
        """
        try:
            # Quad vertices: 2 triangles covering the screen
            # Positions (NDC), TexCoords
            vertices = np.array([
                # Pos        TexCoords
                -1.0,  1.0,  0.0, 1.0,
                -1.0, -1.0,  0.0, 0.0,
                 1.0,  1.0,  1.0, 1.0,
                 1.0, -1.0,  1.0, 0.0,
            ], dtype=np.float32)

            self.quad_vao = glGenVertexArrays(1)
            self.quad_vbo = glGenBuffers(1)

            glBindVertexArray(self.quad_vao)
            glBindBuffer(GL_ARRAY_BUFFER, self.quad_vbo)
            glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

            # Position attribute
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), ctypes.c_void_p(0))
            # TexCoord attribute
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), ctypes.c_void_p(2 * sizeof(GLfloat)))

            glBindVertexArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            return True
        except Exception as e:
            return False

    @staticmethod
    def load_exr_image(filepath: str) -> Optional[np.ndarray]: # Returns (H, W, 3)
        """
        Load an EXR image file and return as a NumPy array (H, W, 3).
        Handles lineOrder attribute for correct image orientation.

        Args:
            filepath: Path to the .exr file.

        Returns:
            HDR image as a NumPy array (float32, shape (H,W,3)) or None on error.
        """
        try:
            # Open EXR file
            exr_file = OpenEXR.InputFile(filepath)
            header = exr_file.header()

            # Get image dimensions
            dw = header['dataWindow']
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1

            # Get available channels
            channels = header['channels']
            channel_names = list(channels.keys())
            # print(f"Available channels: {channel_names}") # Less verbose

            # Read RGB channels with fallback strategy
            pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)

            if all(ch in channel_names for ch in ['R', 'G', 'B']):
                # Standard RGB channels
                r_channel_str = exr_file.channel('R', pixel_type)
                g_channel_str = exr_file.channel('G', pixel_type)
                b_channel_str = exr_file.channel('B', pixel_type)
            elif 'Y' in channel_names and len(channel_names) == 1:
                single_channel_str = exr_file.channel('Y', pixel_type)
                r_channel_str = g_channel_str = b_channel_str = single_channel_str
            elif len(channel_names) >= 3:
                r_channel_str = exr_file.channel(channel_names[0], pixel_type)
                g_channel_str = exr_file.channel(channel_names[1], pixel_type)
                b_channel_str = exr_file.channel(channel_names[2], pixel_type)
            elif len(channel_names) >= 1:
                single_channel_str = exr_file.channel(channel_names[0], pixel_type)
                r_channel_str = g_channel_str = b_channel_str = single_channel_str
            else:
                exr_file.close()
                return None


            # Convert to numpy arrays
            r_data = np.array(array.array('f', r_channel_str), dtype=np.float32)
            g_data = np.array(array.array('f', g_channel_str), dtype=np.float32)
            b_data = np.array(array.array('f', b_channel_str), dtype=np.float32)

            # Reshape and stack
            r_array = r_data.reshape(height, width)
            g_array = g_data.reshape(height, width)
            b_array = b_data.reshape(height, width)

            exr_image = np.stack([r_array, g_array, b_array], axis=-1)            # Check lineOrder attribute for vertical flip
            TARGET_INCREASING_Y_INT_VAL = 0
            current_line_order_int_val = TARGET_INCREASING_Y_INT_VAL

            raw_line_order_attr = header.get('lineOrder')

            if raw_line_order_attr is not None:
                if isinstance(raw_line_order_attr, int):
                    current_line_order_int_val = raw_line_order_attr
                elif hasattr(raw_line_order_attr, 'v'):
                    current_line_order_int_val = raw_line_order_attr.v
                else:
                    pass

            # Flip if the determined integer value corresponds to INCREASING_Y
            if current_line_order_int_val == TARGET_INCREASING_Y_INT_VAL:
                exr_image = np.flipud(exr_image)

            exr_file.close()

            # Validate and clamp values
            if np.any(np.isnan(exr_image)) or np.any(np.isinf(exr_image)):
                exr_image = np.nan_to_num(exr_image, nan=0.0, posinf=10.0, neginf=0.0)

            return exr_image

        except Exception as e:
            return None

    def create_environment_cubemap(self, equirect_to_cubemap_shader_prog: ShaderProgram) -> bool:
        """
        Create an environment cubemap from an equirectangular HDR image.

        Args:
            equirect_to_cubemap_shader_prog: ShaderProgram object for the conversion shader.

        Returns:
            True if successful, False on error (e.g., file not found, shader error).
        """
        if not self.enabled or not self.exr_filepath or equirect_to_cubemap_shader_prog.program_id is None:
            return False

        try:
            # Load HDR image
            exr_image = self.load_exr_image(self.exr_filepath)
            if exr_image is None:
                return False

            # Create equirectangular texture
            self.equirectangular_map_tex = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.equirectangular_map_tex)

            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F,
                        exr_image.shape[1], exr_image.shape[0],
                        0, GL_RGB, GL_FLOAT, exr_image)

            # Set texture parameters
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

            # Setup framebuffer for cubemap generation
            if not self._setup_cubemap_framebuffer():
                return False            # Generate cubemap
            if not self._generate_cubemap_faces(equirect_to_cubemap_shader_prog):
                return False

            # Clean up temporary framebuffer and renderbuffer to avoid GPU resource leak
            if hasattr(self, 'capture_fbo') and self.capture_fbo is not None:
                glDeleteFramebuffers(1, [self.capture_fbo])
                self.capture_fbo = None
            if hasattr(self, 'capture_rbo') and self.capture_rbo is not None:
                glDeleteRenderbuffers(1, [self.capture_rbo])
                self.capture_rbo = None
            return True

        except Exception as e:
            return False

    def create_irradiance_map(self, convolution_shader: 'ShaderProgram') -> bool:
        """
        Create an irradiance map by convolving the environment cubemap.        Args:
            convolution_shader: The shader program for irradiance convolution.

        Returns:
            True if successful, False otherwise.
        """
        if self.env_cubemap_texture is None or convolution_shader.program_id is None:
            return False

        capture_fbo = glGenFramebuffers(1)
        capture_rbo = glGenRenderbuffers(1)

        glBindFramebuffer(GL_FRAMEBUFFER, capture_fbo)
        glBindRenderbuffer(GL_RENDERBUFFER, capture_rbo)
        # Use dynamic cubemap size for irradiance map resolution
        size = self.cubemap_size
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, size, size)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, capture_rbo)

        self.irradiance_map_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.irradiance_map_texture)
        for i in range(6):
            glTexImage2D(int(GL_TEXTURE_CUBE_MAP_POSITIVE_X) + i, 0, GL_RGB16F, size, size, 0, GL_RGB, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glViewport(0, 0, size, size)
        glUseProgram(convolution_shader.program_id)
        glUniform1i(convolution_shader.locations.get("environmentMap", -1), RenderingDefaults.TEXTURE_UNIT_ENVIRONMENT)
        glActiveTexture(int(GL_TEXTURE0) + RenderingDefaults.TEXTURE_UNIT_ENVIRONMENT)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.env_cubemap_texture)
        glUniformMatrix4fv(convolution_shader.locations.get("projection", -1), 1, GL_FALSE, self.capture_projection)

        for i in range(6):
            glUniformMatrix4fv(convolution_shader.locations.get("view", -1), 1, GL_FALSE, self.capture_views[i])
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, int(GL_TEXTURE_CUBE_MAP_POSITIVE_X) + i, self.irradiance_map_texture, 0)
            glClear(int(GL_COLOR_BUFFER_BIT) | int(GL_DEPTH_BUFFER_BIT))
            if self.cube_vao is not None:
                glBindVertexArray(self.cube_vao)
                glDrawArrays(GL_TRIANGLES, 0, 36)
                glBindVertexArray(0)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        # Clean up temporary framebuffer and renderbuffer
        glDeleteFramebuffers(1, [capture_fbo])
        glDeleteRenderbuffers(1, [capture_rbo])
        return self.check_gl_error("Irradiance map generation")

    def create_prefilter_map(self, prefilter_shader: 'ShaderProgram') -> bool:
        """
        Create a prefiltered environment map for specular IBL.

        Args:
            prefilter_shader: The shader program for prefiltering.        Returns:
            True if successful, False otherwise.
        """
        if self.env_cubemap_texture is None or prefilter_shader.program_id is None:
            return False

        self.prefilter_map_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.prefilter_map_texture)
        # Use dynamic cubemap size for prefilter base resolution
        base_size = self.cubemap_size
        for i in range(6):
            glTexImage2D(int(GL_TEXTURE_CUBE_MAP_POSITIVE_X) + i, 0, GL_RGB16F, base_size, base_size, 0, GL_RGB, GL_FLOAT, None)

        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR) # Mipmaps for roughness
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glGenerateMipmap(GL_TEXTURE_CUBE_MAP) # Generate mipmaps

        capture_fbo = glGenFramebuffers(1)
        capture_rbo = glGenRenderbuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, capture_fbo)
        # Attach depth renderbuffer for completeness
        glBindRenderbuffer(GL_RENDERBUFFER, capture_rbo)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, capture_rbo)

        glUseProgram(prefilter_shader.program_id)
        glUniform1i(prefilter_shader.locations.get("environmentMap", -1), RenderingDefaults.TEXTURE_UNIT_ENVIRONMENT)
        glActiveTexture(int(GL_TEXTURE0) + RenderingDefaults.TEXTURE_UNIT_ENVIRONMENT)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.env_cubemap_texture)
        glUniformMatrix4fv(prefilter_shader.locations.get("projection",-1), 1, GL_FALSE, self.capture_projection)

        # Dynamically determine number of mip levels
        max_mip_levels = self.cubemap_mip_levels
        for mip in range(max_mip_levels):
            mip_width = int(base_size * pow(0.5, mip))
            mip_height = int(base_size * pow(0.5, mip))
            glBindRenderbuffer(GL_RENDERBUFFER, capture_rbo)
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, mip_width, mip_height)
            glViewport(0, 0, mip_width, mip_height)

            roughness = float(mip) / float(max_mip_levels - 1)
            glUniform1f(prefilter_shader.locations.get("roughness", -1), roughness)
            for i in range(6):
                glUniformMatrix4fv(prefilter_shader.locations.get("view", -1), 1, GL_FALSE, self.capture_views[i])
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, int(GL_TEXTURE_CUBE_MAP_POSITIVE_X) + i, self.prefilter_map_texture, mip)
                glClear(int(GL_COLOR_BUFFER_BIT) | int(GL_DEPTH_BUFFER_BIT))
                if self.cube_vao is not None:
                    glBindVertexArray(self.cube_vao)
                    glDrawArrays(GL_TRIANGLES, 0, 36)
                    glBindVertexArray(0)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glDeleteFramebuffers(1, [capture_fbo])
        glDeleteRenderbuffers(1, [capture_rbo])
        return self.check_gl_error("Prefilter map generation")

    def generate_brdf_lut(self, brdf_shader: 'ShaderProgram') -> bool:
        """
        Generate the BRDF Lookup Table (LUT) texture.
        The LUT stores pre-calculated scale and bias for the Fresnel term
        of the specular IBL integration, based on NdotV and roughness.

        Args:
            brdf_shader_prog: ShaderProgram object for the BRDF LUT generation shader.

        Returns:
            True if successful, False on error.
        """
        if not self.enabled or brdf_shader.program_id is None:
            return False
        try:
            self.brdf_lut_texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.brdf_lut_texture)
            # Use RG16F for two float components (scale and bias for Fresnel term)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RG16F, self.brdf_lut_size, self.brdf_lut_size, 0, GL_RG, GL_FLOAT, None)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

            # Setup framebuffer: reuse or create capture_fbo
            if self.capture_fbo is None:
                if not self._setup_cubemap_framebuffer(): # This sets up self.capture_fbo and self.capture_rbo
                    return False

            glBindFramebuffer(GL_FRAMEBUFFER, self.capture_fbo)
            # Attach BRDF LUT texture as color attachment
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.brdf_lut_texture, 0)
            # Ensure depth renderbuffer is attached for framebuffer completeness
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.capture_rbo)

            if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
                # Attempt to detach depth if it's causing incompleteness, though usually not the issue here.
                # glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, 0)
                # if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
                raise RuntimeError(f"BRDF LUT Framebuffer is not complete! Status: {glCheckFramebufferStatus(GL_FRAMEBUFFER)}")

            original_viewport = glGetIntegerv(GL_VIEWPORT) # Save current viewport
            glViewport(0, 0, self.brdf_lut_size, self.brdf_lut_size) # Set viewport to LUT size
            glUseProgram(brdf_shader.program_id)
            # Pass sampleCount uniform for BRDF integration
            loc = brdf_shader.locations.get('sampleCount', -1)
            if loc != -1:
                glUniform1i(loc, RenderingDefaults.DEFAULT_IBL_BRDF_SAMPLE_COUNT)
            glClear(GL_COLOR_BUFFER_BIT) # Clear color buffer (depth clear not strictly needed for a full-screen quad)

            if self.quad_vao is None: # Ensure quad VAO is set up
                if not self.setup_quad_geometry():
                    glBindFramebuffer(GL_FRAMEBUFFER, 0) # Restore default FBO
                    glViewport(original_viewport[0], original_viewport[1], original_viewport[2], original_viewport[3]) # Restore viewport
                    if self.brdf_lut_texture is not None: glDeleteTextures(1, [self.brdf_lut_texture]); self.brdf_lut_texture = None
                    return False

            glBindVertexArray(self.quad_vao)
            glDrawArrays(GL_TRIANGLE_STRIP, 0, RenderingDefaults.IBL_TRIANGLE_STRIP_VERTEX_COUNT) # Draw fullscreen quad to render LUT

            glBindFramebuffer(GL_FRAMEBUFFER, 0) # Unbind FBO, reverting to default framebuffer
            glViewport(original_viewport[0], original_viewport[1], original_viewport[2], original_viewport[3]) # Restore original viewport

            return True

        except Exception as e:
            # Clean up partially created texture if any error occurs
            if self.brdf_lut_texture is not None:
                glDeleteTextures(1, [self.brdf_lut_texture])
                self.brdf_lut_texture = None
            # Ensure FBO is unbound on error too
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            return False

    def _setup_cubemap_framebuffer(self) -> bool:
        """
        Setup framebuffer and renderbuffer for offscreen cubemap generation.

        Returns:
            True if successful, False on error.
        """
        try:
            self.capture_fbo = glGenFramebuffers(1)
            self.capture_rbo = glGenRenderbuffers(1)

            glBindFramebuffer(GL_FRAMEBUFFER, self.capture_fbo)
            glBindRenderbuffer(GL_RENDERBUFFER, self.capture_rbo)

            # Setup depth buffer
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24,
                                self.cubemap_size, self.cubemap_size)
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                    GL_RENDERBUFFER, self.capture_rbo)

            return True

        except Exception as e:
            return False

    def _generate_cubemap_faces(self, equirect_to_cubemap_shader_prog: ShaderProgram) -> bool:
        """
        Internal: Generate all 6 faces of the environment cubemap.

        Args:
            equirect_to_cubemap_shader_prog: ShaderProgram for equirectangular to cubemap conversion.

        Returns:
            True if successful, False on error.
        """
        if equirect_to_cubemap_shader_prog.program_id is None:
            return False
        try:
            # Create cubemap texture
            self.env_cubemap_texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_CUBE_MAP, self.env_cubemap_texture)

            # Allocate storage for all faces
            for i in range(RenderingDefaults.IBL_CUBEMAP_FACES_COUNT):
                glTexImage2D(int(GL_TEXTURE_CUBE_MAP_POSITIVE_X) + i, 0, GL_RGB16F,
                            self.cubemap_size, self.cubemap_size, 0, GL_RGB, GL_FLOAT, None)

            # Set texture parameters
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)            # Setup shader uniforms using precomputed capture_projection
            glUseProgram(equirect_to_cubemap_shader_prog.program_id)
            glUniform1i(equirect_to_cubemap_shader_prog.locations.get("equirectangularMap", -1), RenderingDefaults.TEXTURE_UNIT_ENVIRONMENT)
            glUniformMatrix4fv(equirect_to_cubemap_shader_prog.locations.get("projection", -1),
                              1, GL_FALSE, self.capture_projection)

            # Bind equirectangular texture
            glActiveTexture(int(GL_TEXTURE0) + RenderingDefaults.TEXTURE_UNIT_ENVIRONMENT)
            glBindTexture(GL_TEXTURE_2D, self.equirectangular_map_tex) # Bind the loaded EXR texture

            # Set viewport to cubemap size
            glViewport(0, 0, self.cubemap_size, self.cubemap_size)

            # Render to each face of the cubemap
            if self.cube_vao is None: # Ensure cube VAO is ready
                if not self.setup_cube_geometry():
                    glBindFramebuffer(GL_FRAMEBUFFER, 0)
                    return False

            glBindVertexArray(self.cube_vao)
            for i in range(RenderingDefaults.IBL_CUBEMAP_FACES_COUNT):
                glUniformMatrix4fv(equirect_to_cubemap_shader_prog.locations.get("view", -1),
                                 1, GL_FALSE, self.capture_views[i]) # Use pre-defined capture_views
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                     int(GL_TEXTURE_CUBE_MAP_POSITIVE_X) + i, self.env_cubemap_texture, 0)
                glClear(int(GL_COLOR_BUFFER_BIT) | int(GL_DEPTH_BUFFER_BIT))
                glDrawArrays(GL_TRIANGLES, 0, 36) # Draw the cube

            glBindVertexArray(0)
            glBindFramebuffer(GL_FRAMEBUFFER, 0)            # Generate mipmaps for the cubemap
            glBindTexture(GL_TEXTURE_CUBE_MAP, self.env_cubemap_texture)
            glGenerateMipmap(GL_TEXTURE_CUBE_MAP)

            return True

        except Exception as e:
            return False

    def cleanup_environment_maps(self) -> None:
        """Cleans up only the environment-specific textures (HDR, cubemap, irradiance, prefilter)."""
        print("Cleaning up IBL environment maps...")
        try:
            if self.equirectangular_map_tex is not None:
                glDeleteTextures(1, [self.equirectangular_map_tex])
                self.equirectangular_map_tex = None
                print("  - Equirectangular map texture deleted.")

            if self.env_cubemap_texture is not None:
                glDeleteTextures(1, [self.env_cubemap_texture])
                self.env_cubemap_texture = None
                print("  - Environment cubemap texture deleted.")

            if self.irradiance_map_texture is not None:
                glDeleteTextures(1, [self.irradiance_map_texture])
                self.irradiance_map_texture = None
                print("  - Irradiance map texture deleted.")

            if self.prefilter_map_texture is not None:
                glDeleteTextures(1, [self.prefilter_map_texture])
                self.prefilter_map_texture = None
                print("  - Prefilter map texture deleted.")

            # Note: BRDF LUT, cube/quad VAO/VBOs, and capture FBO/RBO are NOT cleaned here.
        except Exception as e:
            print(f"Error during IBL environment map cleanup: {e}")
            # Log error, but don't necessarily halt everything.

    def cleanup(self) -> None:
        """Clean up all OpenGL resources (textures, buffers, framebuffers) used by IBL."""
        try:
            if self.equirectangular_map_tex is not None:
                glDeleteTextures(1, [self.equirectangular_map_tex])
                self.equirectangular_map_tex = None

            if self.env_cubemap_texture is not None:
                glDeleteTextures(1, [self.env_cubemap_texture])
                self.env_cubemap_texture = None

            if self.brdf_lut_texture is not None:
                glDeleteTextures(1, [self.brdf_lut_texture])
                self.brdf_lut_texture = None

            if self.irradiance_map_texture is not None:
                glDeleteTextures(1, [self.irradiance_map_texture])
                self.irradiance_map_texture = None

            if self.prefilter_map_texture is not None:
                glDeleteTextures(1, [self.prefilter_map_texture])
                self.prefilter_map_texture = None

            if self.cube_vao is not None:
                glDeleteVertexArrays(1, [self.cube_vao])
                self.cube_vao = None

            if self.cube_vbo is not None:
                glDeleteBuffers(1, [self.cube_vbo])
                self.cube_vbo = None

            if self.quad_vao is not None:
                glDeleteVertexArrays(1, [self.quad_vao])
                self.quad_vao = None

            if self.quad_vbo is not None:
                glDeleteBuffers(1, [self.quad_vbo])
                self.quad_vbo = None

            if self.capture_fbo is not None:
                glDeleteFramebuffers(1, [self.capture_fbo])
                self.capture_fbo = None

            if self.capture_rbo is not None:
                glDeleteRenderbuffers(1, [self.capture_rbo])
                self.capture_rbo = None

            # Clear shader program cache to release GPU programs
            clear_shader_cache()

        except Exception as e:
            pass

