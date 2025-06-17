"""
Scene Renderer for the OpenGL viewer.
"""

import ctypes
import warnings
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np
import pyrr
from OpenGL.GL import *  # type: ignore[import-untyped, unused-ignore]
from OpenGL.GLU import gluErrorString

from .config import RenderingDefaults

if TYPE_CHECKING:
    from .cameras import CameraController
    from .geometries import GeometryBuffer
    from .controller import BaseShaderController, IBLRenderer
    from .shader_utils import ShaderProgram # Import ShaderProgram

from .shader_utils import get_shader_program


class SceneRenderer:
    """
    Manages shader loading and executes the core rendering logic for the scene.
    This includes drawing points, lines, triangles, and the skybox,
    with support for different rendering paths like IBL and SSS.
    """
    def __init__(self, camera: 'CameraController', ibl_controller: 'IBLRenderer',
                 base_shader_controller: 'BaseShaderController',
                 points_buffer: 'GeometryBuffer', lines_buffer: 'GeometryBuffer',
                 triangles_buffer: 'GeometryBuffer',
                 draw_objects_individually: bool = False):
        """
        Initialize the SceneRenderer.

        Args:
            camera: The CameraController instance for view and projection matrices.
            ibl_controller: The IBLRenderer instance for image-based lighting.
            base_shader_controller: The BaseShaderController for common shader uniforms.
            points_buffer: GeometryBuffer for point data.
            lines_buffer: GeometryBuffer for line data.
            triangles_buffer: GeometryBuffer for triangle data.
            draw_objects_individually: If True, draw objects within each buffer one by one,
                                       allowing for potential per-object culling.
                                       If False, draw all objects of a type in a single batch.
        """
        self.camera = camera
        self.ibl_controller = ibl_controller
        self.base_shader_controller = base_shader_controller
        self.points_buffer = points_buffer
        self.lines_buffer = lines_buffer
        self.triangles_buffer = triangles_buffer
        self.draw_objects_individually = draw_objects_individually # Store the flag

        self.base_shader: Optional['ShaderProgram'] = None
        self.equirect_to_cubemap_shader: Optional['ShaderProgram'] = None
        self.skybox_shader: Optional['ShaderProgram'] = None
        self.ibl_triangle_shader: Optional['ShaderProgram'] = None
        self.brdf_lut_shader: Optional['ShaderProgram'] = None
        self.sss_shader: Optional['ShaderProgram'] = None
        self.irradiance_convolution_shader: Optional['ShaderProgram'] = None # Added
        self.prefilter_env_map_shader: Optional['ShaderProgram'] = None # Added

        self._shaders_initialized = False

        # Cache for GL state
        self._current_program_id: Optional[int] = 0 # 0 is default unbound program
        self._current_vao_id: Optional[int] = 0     # 0 is default unbound VAO

        # Cache for texture bindings managed by SceneRenderer
        self._renderer_active_texture_unit: Optional[int] = -1
        self._renderer_bound_textures: Dict[int, Dict[int, Optional[int]]] = {} # unit -> {target: texture_id}


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
            print(f"OpenGL Error ({tag}) in SceneRenderer: Code {err} - {err_str}")
            return False
        return True

    def initialize_shaders(self) -> None:
        """
        Load and compile all shader programs required by the renderer.
        Populates shader attributes like self.base_shader, self.skybox_shader, etc.

        Raises:
            RuntimeError: If any shader fails to load, compile, or link,
                          or if an OpenGL error occurs during initialization.
        """
        try:
            # Use get_shader_program from shader_utils for each shader
            self.base_shader = get_shader_program('base')
            self.sss_shader = get_shader_program('sss')
            self.equirect_to_cubemap_shader = get_shader_program('equirect_to_cubemap')
            self.skybox_shader = get_shader_program('skybox')
            self.ibl_triangle_shader = get_shader_program('ibl_triangle')
            self.brdf_lut_shader = get_shader_program('brdf_lut')
            self.irradiance_convolution_shader = get_shader_program('irradiance_convolution') # Added
            self.prefilter_env_map_shader = get_shader_program('prefilter_env_map') # Added

            if not self.check_gl_error("Shader creation in SceneRenderer"):
                raise RuntimeError("OpenGL error after shader creation.")
            self._shaders_initialized = True
        except Exception as e:
            self._shaders_initialized = False
            # Re-raise to propagate the error
            raise

    def _bind_program(self, program_id: Optional[int]) -> None:
        """Binds a shader program if it's not already bound."""
        pid = program_id if program_id is not None else 0
        if self._current_program_id != pid:
            glUseProgram(pid)
            self._current_program_id = pid

    def _bind_vao(self, vao_id: Optional[int]) -> None:
        """Binds a VAO if it's not already bound."""
        vid = vao_id if vao_id is not None else 0
        if self._current_vao_id != vid:
            glBindVertexArray(vid)
            self._current_vao_id = vid

    def _activate_renderer_texture_unit(self, unit: int) -> None:
        """Activates a texture unit for renderer-managed textures if not already active."""
        if self._renderer_active_texture_unit != unit:
            glActiveTexture(int(GL_TEXTURE0) + unit)
            self._renderer_active_texture_unit = unit

    def _bind_renderer_texture_on_current_unit(self, target: int, texture_id: Optional[int]) -> None:
        """Binds a renderer-managed texture to the currently active unit if not already bound."""
        if self._renderer_active_texture_unit is None:
            return
        current_unit_bindings = self._renderer_bound_textures.get(self._renderer_active_texture_unit, {})
        if current_unit_bindings.get(target) != texture_id:
            glBindTexture(target, texture_id if texture_id is not None else 0)
            if self._renderer_active_texture_unit not in self._renderer_bound_textures:
                self._renderer_bound_textures[self._renderer_active_texture_unit] = {}
            self._renderer_bound_textures[self._renderer_active_texture_unit][target] = texture_id

    def _renderer_activate_and_bind_texture(self, unit: int, target: int, texture_id: Optional[int]) -> None:
        """Activates texture unit and binds texture using renderer's cache."""
        self._activate_renderer_texture_unit(unit)
        self._bind_renderer_texture_on_current_unit(target, texture_id)

    def _draw_geometry_type(self, buffer: 'GeometryBuffer', primitive_type: int) -> bool:
        """
        Draw a specific type of geometry (points, lines, triangles) from a GeometryBuffer.

        Args:
            buffer: The GeometryBuffer containing the vertex data and VAO.
            primitive_type: The OpenGL primitive type (e.g., GL_POINTS, GL_LINES, GL_TRIANGLES).

        Returns:
            True if drawing was successful or if there was no data to draw, False on OpenGL error.
        """
        # Check if data is None or empty, or if VAO is not set
        if buffer.current_elements_count == 0 or buffer.vao is None: # Use current_elements_count for active elements
            # If buffer.data is None but current_elements_count > 0, that's an inconsistent state.
            # However, current_elements_count == 0 should cover cases where data might be None after full clear.
            return True  # Not an error if no data to draw or VAO not ready

        try:
            self._bind_vao(buffer.vao) # Use cached VAO binding
            if self.draw_objects_individually:
                # Draw each object separately using glMultiDrawArrays for batching API calls
                first_indices = []
                counts = []
                for key, (start_vertex, vertex_count) in buffer.object_ranges.items():
                    if vertex_count > 0:
                        first_indices.append(start_vertex)
                        counts.append(vertex_count)

                if counts: # If there's anything to draw
                    # Convert Python lists to ctypes arrays for glMultiDrawArrays
                    # PyOpenGL's glMultiDrawArrays can often take Python sequences directly,
                    # but using ctypes arrays is more explicit and robust.
                    first_arr = (GLint * len(first_indices))(*first_indices)
                    count_arr = (GLsizei * len(counts))(*counts)
                    glMultiDrawArrays(primitive_type, first_arr, count_arr, len(counts))
            else:
                # Draw all objects in one go. Buffer must be compact for this.
                if buffer.has_fragmentation():
                    # print(f"Defragmenting buffer for non-individual draw (type: {primitive_type})...")
                    buffer.defragment()
                    # Defragmentation changed CPU data and set dirty = True.
                    # We need to update GPU buffer *now* for this draw call to be correct.
                    buffer.update_gpu_buffer() # This will upload the newly compacted data and clear dirty flag.

                vertex_count = buffer.get_vertex_count() # Total vertices in active objects
                if vertex_count > 0:
                    # Assumes buffer is now compact from 0 to vertex_count
                    glDrawArrays(primitive_type, 0, vertex_count)

            # Unbind VAO after drawing this geometry type
            # glBindVertexArray(0) # Moved to draw_scene finally block for overall unbind
            # self._bind_vao(0) # Unbinding individual VAOs here might be too frequent if drawing multiple types with same shader
            return self.check_gl_error(f"Drawing geometry type {primitive_type}")
        except Exception as e:
            print(f"Error drawing geometry: {e}")
            return False

    def _set_ibl_uniforms(self, shader_prog: 'ShaderProgram') -> None:
        """
        Set IBL-specific uniforms for a shader program using cached locations.
        This includes environment maps, BRDF LUT, camera position, and material properties.

        Args:
            shader_prog: The ShaderProgram (typically self.ibl_triangle_shader) to set uniforms for.
        """
        glUniform3fv(shader_prog.locations.get("viewPos", -1), 1, self.camera.eye)

        # Environment Map
        self._renderer_activate_and_bind_texture(
            RenderingDefaults.TEXTURE_UNIT_ENVIRONMENT,
            GL_TEXTURE_CUBE_MAP,
            self.ibl_controller.env_cubemap_texture
        )
        glUniform1i(shader_prog.locations.get("environmentMap", -1), RenderingDefaults.TEXTURE_UNIT_ENVIRONMENT)

        # Irradiance Map
        if self.ibl_controller.irradiance_map_texture is not None:
            self._renderer_activate_and_bind_texture(
                RenderingDefaults.TEXTURE_UNIT_IRRADIANCE_MAP,
                GL_TEXTURE_CUBE_MAP,
                self.ibl_controller.irradiance_map_texture
            )
            glUniform1i(shader_prog.locations.get("irradianceMap", -1), RenderingDefaults.TEXTURE_UNIT_IRRADIANCE_MAP)

        # Prefilter Map
        if self.ibl_controller.prefilter_map_texture is not None:
            self._renderer_activate_and_bind_texture(
                RenderingDefaults.TEXTURE_UNIT_PREFILTER_MAP,
                GL_TEXTURE_CUBE_MAP,
                self.ibl_controller.prefilter_map_texture
            )
            glUniform1i(shader_prog.locations.get("prefilterMap", -1), RenderingDefaults.TEXTURE_UNIT_PREFILTER_MAP)

        # BRDF LUT
        if self.ibl_controller.brdf_lut_texture is not None:
            self._renderer_activate_and_bind_texture(
                RenderingDefaults.TEXTURE_UNIT_BRDF_LUT,
                GL_TEXTURE_2D,
                self.ibl_controller.brdf_lut_texture
            )
            glUniform1i(shader_prog.locations.get("brdfLUT", -1), RenderingDefaults.TEXTURE_UNIT_BRDF_LUT)

        glUniform1f(shader_prog.locations.get("ambientIntensity", -1), self.ibl_controller.ambient_intensity)
        glUniform1f(shader_prog.locations.get("specularIntensity", -1), self.ibl_controller.specular_intensity)
        glUniform1f(shader_prog.locations.get("baseRoughness", -1), self.ibl_controller.base_roughness)

        glUniform1i(shader_prog.locations.get("useTexture", -1), int(self.base_shader_controller.use_texture))
        if self.base_shader_controller.use_texture and self.base_shader_controller.diffuse_texture_id is not None:
            # SceneRenderer binds this texture as BaseShaderController.apply_uniforms is not called for ibl_triangle_shader path
            self._renderer_activate_and_bind_texture(
                RenderingDefaults.TEXTURE_UNIT_DIFFUSE,
                GL_TEXTURE_2D,
                self.base_shader_controller.diffuse_texture_id
            )
            glUniform1i(shader_prog.locations.get("diffuseTexture", -1), RenderingDefaults.TEXTURE_UNIT_DIFFUSE)

        glUniform3fv(shader_prog.locations.get("materialDiffuse", -1), 1, self.base_shader_controller.material_diffuse)
        glUniform3fv(shader_prog.locations.get("materialSpecular", -1), 1, self.base_shader_controller.material_specular)
        glUniform1f(shader_prog.locations.get("materialAlpha", -1), self.base_shader_controller.material_alpha)

        glUniform1i(shader_prog.locations.get("enableInstancing", -1), int(self.base_shader_controller.enable_instancing))
        glUniform1i(shader_prog.locations.get("enableDistanceFade", -1), int(self.base_shader_controller.enable_distance_fade))
        glUniform1f(shader_prog.locations.get("fadeDistance", -1), self.base_shader_controller.fade_distance)

    def _set_shader_matrices(self, shader_prog: 'ShaderProgram',
                             view_matrix: np.ndarray, # (4, 4)
                             proj_matrix: np.ndarray  # (4, 4)
                             ) -> None:
        """
        Set common transformation matrices (model, view, projection, normalMatrix)
        for a given shader program using cached uniform locations.

        Args:
            shader_prog: The ShaderProgram to set uniforms for.
            view_matrix: The view matrix (4x4 NumPy array).
            proj_matrix: The projection matrix (4x4 NumPy array).
        """
        glUniformMatrix4fv(shader_prog.locations.get("model", -1),
                          1, GL_FALSE, self.camera.model_matrix)
        glUniformMatrix4fv(shader_prog.locations.get("view", -1),
                          1, GL_FALSE, view_matrix)
        glUniformMatrix4fv(shader_prog.locations.get("projection", -1),
                          1, GL_FALSE, proj_matrix)

        normal_matrix_loc = shader_prog.locations.get("normalMatrix", -1)
        if normal_matrix_loc != -1:
            model_3x3 = pyrr.matrix33.create_from_matrix44(self.camera.model_matrix)
            try:
                normal_matrix_val = np.linalg.inv(model_3x3).T
                glUniformMatrix3fv(normal_matrix_loc, 1, GL_FALSE, normal_matrix_val)
            except np.linalg.LinAlgError:
                # Use identity matrix if model matrix is not invertible
                identity_3x3 = np.eye(3, dtype=np.float32)
                glUniformMatrix3fv(normal_matrix_loc, 1, GL_FALSE, identity_3x3)

    def _draw_points_and_lines(self, view_matrix: np.ndarray, proj_matrix: np.ndarray) -> None: # (4,4) each
        """
        Draw points and lines using the base shader.

        Args:
            view_matrix: Current view matrix.
            proj_matrix: Current projection matrix.
        """
        if self.base_shader is None or self.base_shader.program_id is None:
            return

        self._bind_program(self.base_shader.program_id) # Use cached program binding
        self._set_shader_matrices(self.base_shader, view_matrix, proj_matrix)
        if not self.base_shader_controller.apply_uniforms(self.base_shader, self.camera.eye):
            print("Warning: Failed to apply shader uniforms for points/lines")

        self._draw_geometry_type(self.points_buffer, GL_POINTS)
        self._draw_geometry_type(self.lines_buffer, GL_LINES)

    def _draw_triangles(self, view_matrix: np.ndarray, proj_matrix: np.ndarray) -> None: # (4,4) each
        """
        Draw triangles, selecting the appropriate shader (SSS, IBL, or base) based on settings.

        Args:
            view_matrix: Current view matrix.
            proj_matrix: Current projection matrix.
        """
        triangle_shader_prog: Optional['ShaderProgram'] = None
        use_sss_shader = False

        # SSS has highest priority if enabled
        if (self.base_shader_controller.enable_sss and self.sss_shader and self.sss_shader.program_id is not None):
            triangle_shader_prog = self.sss_shader
            use_sss_shader = True
        # Otherwise select IBL shader if IBL is enabled
        elif (self.ibl_controller.enabled and
              self.ibl_controller.env_cubemap_texture and
              self.ibl_triangle_shader and self.ibl_triangle_shader.program_id is not None):
            triangle_shader_prog = self.ibl_triangle_shader
        elif self.base_shader is not None and self.base_shader.program_id is not None:
            triangle_shader_prog = self.base_shader
        else: # Default to base shader if specific ones are not available/valid
            triangle_shader_prog = self.base_shader


        if triangle_shader_prog is None or triangle_shader_prog.program_id is None:
            warnings.warn("No suitable shader available for rendering triangles.", RuntimeWarning)
            return

        self._bind_program(triangle_shader_prog.program_id) # Use cached program binding
        self._set_shader_matrices(triangle_shader_prog, view_matrix, proj_matrix)

        if use_sss_shader:
            # Apply common uniforms (lighting, material, effects)
            self.base_shader_controller.apply_uniforms(triangle_shader_prog, self.camera.eye)
            # SSS-specific IBL sub-binding
            # Determine if IBL should be used with SSS; ensure boolean for int conversion
            use_ibl = bool(self.ibl_controller.enabled \
                           and self.ibl_controller.env_cubemap_texture\
                              and self.ibl_controller.brdf_lut_texture)
            loc_useibl = triangle_shader_prog.locations.get("useIBL", -1)
            if loc_useibl != -1:
                glUniform1i(loc_useibl, int(use_ibl))
            if use_ibl:
                # Bind environment and BRDF LUT for SSS IBL
                self._renderer_activate_and_bind_texture(
                    RenderingDefaults.TEXTURE_UNIT_ENVIRONMENT,
                    GL_TEXTURE_CUBE_MAP,
                    self.ibl_controller.env_cubemap_texture
                )
                glUniform1i(triangle_shader_prog.locations.get("environmentMap", -1), RenderingDefaults.TEXTURE_UNIT_ENVIRONMENT)
                self._renderer_activate_and_bind_texture(
                    RenderingDefaults.TEXTURE_UNIT_BRDF_LUT,
                    GL_TEXTURE_2D,
                    self.ibl_controller.brdf_lut_texture
                )
                glUniform1i(triangle_shader_prog.locations.get("brdfLUT", -1), RenderingDefaults.TEXTURE_UNIT_BRDF_LUT)
        elif triangle_shader_prog == self.ibl_triangle_shader:
            self._set_ibl_uniforms(triangle_shader_prog)
        else: # Base shader is used for triangles
            self.base_shader_controller.apply_uniforms(triangle_shader_prog, self.camera.eye)

        self._draw_geometry_type(self.triangles_buffer, GL_TRIANGLES)

    def _draw_skybox(self, view_matrix: np.ndarray, proj_matrix: np.ndarray) -> None: # (4,4) each
        """
        Draw the skybox if IBL is enabled and resources are available.

        Args:
            view_matrix: Current view matrix.
            proj_matrix: Current projection matrix.
        """
        if (not self.ibl_controller.enabled or not self.ibl_controller.env_cubemap_texture or
            not self.skybox_shader or self.skybox_shader.program_id is None or not self.ibl_controller.cube_vao):
            return

        try:
            glDepthFunc(GL_LEQUAL)
            self._bind_program(self.skybox_shader.program_id) # Use cached program binding

            skybox_view_matrix = pyrr.matrix44.create_from_matrix33(pyrr.matrix33.create_from_matrix44(view_matrix))
            glUniformMatrix4fv(self.skybox_shader.locations.get("view", -1),
                             1, GL_FALSE, skybox_view_matrix)
            glUniformMatrix4fv(self.skybox_shader.locations.get("projection", -1),
                             1, GL_FALSE, proj_matrix)

            self._renderer_activate_and_bind_texture(
                RenderingDefaults.TEXTURE_UNIT_ENVIRONMENT,
                GL_TEXTURE_CUBE_MAP,
                self.ibl_controller.env_cubemap_texture
            )
            glUniform1i(self.skybox_shader.locations.get("environmentMap", -1), RenderingDefaults.TEXTURE_UNIT_ENVIRONMENT)

            self._bind_vao(self.ibl_controller.cube_vao) # Use cached VAO binding
            glDrawArrays(GL_TRIANGLES, 0, 36)
            # self._bind_vao(0) # Unbind VAO - will be done at end of draw_scene

            glDepthFunc(GL_LESS) # Reset depth function to default
            self.check_gl_error("Skybox drawing")
        except Exception as e:
            print(f"Error drawing skybox: {e}")


    def draw_scene(self,
                   if_draw_points_lines: bool=True,
                   if_draw_triangles: bool=True,
                   if_draw_skybox: bool = True) -> None:
        """
        Draw the complete scene, including all geometry types and the skybox.
        Manages shader activation and uniform setting for each part of the scene.

        Args:
            skybox_visible: If True and IBL is enabled, the skybox will be rendered.
        """
        if not self._shaders_initialized:
            print("SceneRenderer shaders not initialized. Skipping draw_scene.")
            return

        # Use camera's current matrices
        current_view = self.camera.view_matrix
        current_proj = self.camera.projection_matrix

        # Verify VAOs are initialized (VAOs are managed by OpenGLWidget and associated buffers)
        # This check might be redundant if individual draw_geometry_type handles VAO None.
        # However, it's a good early exit.
        if (self.points_buffer.vao is None or self.lines_buffer.vao is None or
            self.triangles_buffer.vao is None):
            # This check might be too strict if a buffer is legitimately empty and thus has no VAO yet.
            # The individual _draw_geometry_type checks for buffer.vao is None are more granular.
            # However, it's a good early exit.
            pass # print("Warning: Some Geometry VAOs not initialized. Drawing will attempt for valid ones.")


        try:
            self._draw_points_and_lines(current_view, current_proj) if if_draw_points_lines else ...
            self._draw_triangles(current_view, current_proj) if if_draw_triangles else ...
            self._draw_skybox(current_view, current_proj) if if_draw_skybox and self.ibl_controller.enabled else ...
        except Exception as e:
            print(f"Error during scene drawing: {e}")
        finally:
            self._bind_vao(0) # Ensure VAO is unbound
            self._bind_program(0)      # Ensure shader program is unbound
            # Reset texture cache states for next frame, good practice
            self._renderer_active_texture_unit = -1
            self._renderer_bound_textures.clear()
            if self.base_shader_controller:
                self.base_shader_controller.reset_texture_binding_cache()

            self.check_gl_error("SceneRenderer draw_scene end")


def render_scene_to_images(widget: 'OpenGLWidget', # type: ignore[no-redef, unused-ignore]
                           scene_renderer: 'SceneRenderer',
                           width: int, height: int,
                           render_skybox: bool = False,
                           offscreen_float: Optional[bool] = None) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Render the current scene to offscreen buffers and return vertically flipped RGBA, normal, and depth arrays.

    Args:
        widget: The OpenGLWidget instance for context and buffer sync.
        scene_renderer: The SceneRenderer instance.
        width: Output image width.
        height: Output image height.
        render_skybox: Whether to render the skybox.
        offscreen_float: If True, use float32 RGBA; if False, use 8-bit RGBA; if None, use default from config.

    Returns:
        Tuple of (rgba_array, normals_array, depth_array) where:
        - rgba_array: RGBA image data as numpy array
        - normals_array: Normal vectors as numpy array, or None if unavailable
        - depth_array: Depth buffer data as numpy array

    Raises:
        RuntimeError: If framebuffer setup fails or OpenGL operations fail.
        ValueError: If width or height are invalid.
    """
    # Validate input parameters
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid dimensions: width={width}, height={height}. Must be positive integers.")

    if not hasattr(widget, 'makeCurrent') or not hasattr(widget, '_update_geometry_buffers'):
        raise ValueError("Invalid widget: must be an OpenGLWidget instance with required methods.")

    # Ensure OpenGL context and buffer sync
    try:
        widget.makeCurrent()
        widget._update_geometry_buffers()
    except Exception as e:
        raise RuntimeError(f"Failed to prepare OpenGL context: {e}")

    # Save current framebuffer and viewport
    prev_fbo: int = glGetIntegerv(GL_FRAMEBUFFER_BINDING)
    prev_viewport: list[int] = glGetIntegerv(GL_VIEWPORT)

    # save current GL state
    prev_blend_enabled = glIsEnabled(GL_BLEND)
    prev_blend_src     = glGetIntegerv(GL_BLEND_SRC)
    prev_blend_dst     = glGetIntegerv(GL_BLEND_DST)
    prev_read_buf      = glGetIntegerv(GL_READ_BUFFER)

    # Determine precision
    if offscreen_float is None:
        use_float = RenderingDefaults.OFFSCREEN_RENDER_FLOAT_RGBA
    else:
        use_float = offscreen_float

    rgba_internal_format = GL_RGBA32F if use_float else GL_RGBA8
    rgba_format = GL_RGBA
    rgba_type = GL_FLOAT if use_float else GL_UNSIGNED_BYTE
    rgba_dtype = np.float32 if use_float else np.uint8

    # Initialize resource tracking
    fbo: Optional[int] = None
    color_tex: Optional[int] = None
    normal_tex: Optional[int] = None
    depth_tex: Optional[int] = None

    try:
        # Create and bind FBO
        fbo = glGenFramebuffers(1)
        if not fbo:
            raise RuntimeError("Failed to generate framebuffer")
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        # enable blending for proper alpha writes
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Color attachment
        color_tex = glGenTextures(1)
        if not color_tex:
            raise RuntimeError("Failed to generate color texture")
        glBindTexture(GL_TEXTURE_2D, color_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, rgba_internal_format, width, height, 0, rgba_format, rgba_type, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_tex, 0)

        # Normal attachment
        normal_tex = glGenTextures(1)
        if not normal_tex:
            raise RuntimeError("Failed to generate normal texture")
        glBindTexture(GL_TEXTURE_2D, normal_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0, GL_RGB, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, normal_tex, 0)

        # Depth attachment
        depth_tex = glGenTextures(1)
        if not depth_tex:
            raise RuntimeError("Failed to generate depth texture")
        glBindTexture(GL_TEXTURE_2D, depth_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_tex, 0)

        # Configure draw buffers
        glDrawBuffers(2, [GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1])

        # Check FBO completeness
        fbo_status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if fbo_status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Framebuffer incomplete. Status: {fbo_status}")

        # Set viewport and clear
        glViewport(0, 0, width, height)
        # clear alpha to 0 for transparent background
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) # type: ignore[no-untyped-call]

        # Render scene with control flags
        scene_renderer.draw_scene(
            if_draw_points_lines=False,
            if_draw_triangles=True,
            if_draw_skybox=render_skybox)

        # Read and flip outputs
        glReadBuffer(GL_COLOR_ATTACHMENT0)
        rgba_data = glReadPixels(0, 0, width, height, rgba_format, rgba_type)
        if not isinstance(rgba_data, (bytes, bytearray)) and hasattr(rgba_data, 'tobytes'):
            rgba_data = rgba_data.tobytes() # type: ignore[no-untyped-call]
        elif not isinstance(rgba_data, (bytes, bytearray)):
            raise RuntimeError(f"Unexpected data type from glReadPixels: {type(rgba_data)}")
        rgba = np.frombuffer(rgba_data, dtype=rgba_dtype).reshape((height, width, 4))
        rgba = np.flipud(rgba)

        glReadBuffer(GL_COLOR_ATTACHMENT1)
        normals: Optional[np.ndarray] = None
        try:
            normals_data = glReadPixels(0, 0, width, height, GL_RGB, GL_FLOAT)
            if not isinstance(normals_data, (bytes, bytearray)) and hasattr(normals_data, 'tobytes'):
                normals_data = normals_data.tobytes()  # type: ignore[no-untyped-call]
            elif not isinstance(normals_data, (bytes, bytearray)):
                raise RuntimeError(f"Unexpected data type from glReadPixels: {type(normals_data)}")
            normals = np.frombuffer(normals_data, dtype=np.float32).reshape((height, width, 3))
            normals = np.flipud(normals)
            # Check if normals are all zero (no normal data available)
            if np.allclose(normals, 0.0):
                normals = None
        except Exception as e:
            print(f"Warning: Failed to read normal buffer: {e}")
            normals = None

        depth_data = glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT)
        if not isinstance(depth_data, (bytes, bytearray)) and hasattr(depth_data, 'tobytes'):
            depth_data = depth_data.tobytes()  # type: ignore[no-untyped-call]
        elif not isinstance(depth_data, (bytes, bytearray)):
            raise RuntimeError(f"Unexpected data type from glReadPixels: {type(depth_data)}")
        depth = np.frombuffer(depth_data, dtype=np.float32).reshape((height, width, 1))
        depth = np.flipud(depth)

        return rgba, normals, depth

    except Exception as e:
        raise RuntimeError(f"Failed to render scene to images: {e}")

    finally:
        # Ensure proper cleanup of resources and restore GL state
        try:
            # Restore FBO binding and viewport
            glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo)
            glViewport(*prev_viewport)
            # Restore GL state
            if prev_blend_enabled:
                glEnable(GL_BLEND)
            else:
                glDisable(GL_BLEND)
            glBlendFunc(int(prev_blend_src), int(prev_blend_dst))
            # Restore read buffer
            glReadBuffer(int(prev_read_buf))

            # Clean up textures
            textures_to_delete = []
            if color_tex:
                textures_to_delete.append(color_tex)
            if normal_tex:
                textures_to_delete.append(normal_tex)
            if depth_tex:
                textures_to_delete.append(depth_tex)

            if textures_to_delete:
                glDeleteTextures(textures_to_delete)

            # Clean up framebuffer
            if fbo:
                glDeleteFramebuffers(1, [fbo])

        except Exception as cleanup_error:
            print(f"Warning: Error during resource cleanup: {cleanup_error}")
