"""
Main OpenGL widget for 3D geometry visualization.
"""

import array
import ctypes
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import Imath
import numpy as np
import pyrr
from OpenGL.GL import *  # type: ignore[import-untyped, unused-ignore]
from OpenGL.GLU import gluErrorString
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QSurfaceFormat  # Import QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from .cameras import CameraController
from .config import RenderingDefaults
from .controller import BaseShaderController, IBLRenderer
from .geometries import (DisplayedLines, DisplayedPoints, DisplayedTriangles,
                        GeometryBuffer, GeometryObjectAttributes,
                        LineAttributes, PointAttributes, TriangleAttributes)
from .renderer import SceneRenderer
from .shader_utils import clear_shader_cache, get_shader_program


class Viewer3DWidget(QOpenGLWidget):
    """
    Enhanced 3D viewer widget using the display object system.
    """

    CameraUpdated = QtCore.Signal(dict)  # Signal to notify camera updates

    # Shader attribute layouts
    _BASE_ATTRIBUTE_LAYOUT = [
        (0, 3, 0),  # Vertex position
        (1, 3, 3),  # Color
        (2, 3, 6),  # Normal
        (3, 2, 9),  # Texcoord
        (4, 1, 11), # Point size
        (5, 1, 12), # Line width
        (6, 1, 13), # Shape type
        (7, 4, 14)  # Instance data
    ]
    _POINTS_LINES_STRIDE = RenderingDefaults.OPENGL_ATTRIBUTE_STRIDE_POINTS_LINES

    _TRIANGLE_ATTRIBUTE_LAYOUT_EXT = [
        (8, 3, 18), # Material [metallic, roughness, ao]
        (9, 4, 21), # SSS Params [strength, distortion, power, scale]
        (10, 3, 25) # SSS Color
    ]
    _TRIANGLES_STRIDE = RenderingDefaults.OPENGL_ATTRIBUTE_STRIDE_TRIANGLES

    # =============================================================================
    # 1. Initialization and Basic Setup
    # =============================================================================
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None,
                 exr_filepath: Optional[str] = None,
                 draw_objects_individually: bool = True) -> None:
        """
        Initialize OpenGL widget.

        Args:
            parent: Parent widget.
            exr_filepath: Path to HDR environment map for IBL, or None to disable.
            draw_objects_individually: If True, draw objects one by one (for culling). Default False.
        """
        super().__init__(parent)

        # Request a multisample format
        fmt = QSurfaceFormat()
        fmt.setSamples(RenderingDefaults.MSAA_SAMPLES)  # Request 4x MSAA, adjust as needed
        self.setFormat(fmt)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Display object system
        self.display_points = DisplayedPoints()
        self.display_lines = DisplayedLines()
        self.display_triangles = DisplayedTriangles()

        # Geometry buffers
        self.points_buffer = GeometryBuffer(stride=self._POINTS_LINES_STRIDE)
        self.lines_buffer = GeometryBuffer(stride=self._POINTS_LINES_STRIDE)
        self.triangles_buffer = GeometryBuffer(stride=self._TRIANGLES_STRIDE)

        # Default rendering parameters
        self.defaults = RenderingDefaults()

        # Controllers
        self.camera = CameraController()
        self.ibl_controller = IBLRenderer(exr_filepath)
        self.base_shader_controller = BaseShaderController()
        # Ensure BaseShaderController can bind IBL textures
        self.base_shader_controller.ibl_controller = self.ibl_controller

        # Store the flag
        self._draw_objects_individually = draw_objects_individually

        # Scene renderer
        self.scene_renderer = SceneRenderer(
            camera=self.camera,
            ibl_controller=self.ibl_controller,
            base_shader_controller=self.base_shader_controller,
            points_buffer=self.points_buffer,
            lines_buffer=self.lines_buffer,
            triangles_buffer=self.triangles_buffer,
            draw_objects_individually=self._draw_objects_individually # Pass to SceneRenderer
        )        # Rendering state
        self.point_size = RenderingDefaults.DEFAULT_POINT_SIZE
        self._is_initialized = False

    def initializeGL(self) -> None:
        """Initialize OpenGL state, shaders, VAOs, and IBL resources."""
        try:
            # Clear shader cache in case of re-initialization
            clear_shader_cache()

            # Initialize SceneRenderer (which loads shaders)
            self.scene_renderer.initialize_shaders()
            if not self.scene_renderer.check_gl_error("SceneRenderer shader initialization"):
                raise RuntimeError("Failed to initialize SceneRenderer shaders.")

            # Setup VAOs for geometry buffers using class-defined layouts
            if not self._setup_geometry_vao(self.points_buffer, self._BASE_ATTRIBUTE_LAYOUT):
                raise RuntimeError("Failed to setup points VAO")
            if not self._setup_geometry_vao(self.lines_buffer, self._BASE_ATTRIBUTE_LAYOUT):
                raise RuntimeError("Failed to setup lines VAO")

            triangle_full_layout = self._BASE_ATTRIBUTE_LAYOUT + self._TRIANGLE_ATTRIBUTE_LAYOUT_EXT
            if not self._setup_geometry_vao(self.triangles_buffer, triangle_full_layout):
                raise RuntimeError("Failed to setup triangles VAO")            # Initialize IBL controller (creates textures, etc.)
            # Ensure shaders are available before initializing IBL resources
            if self.ibl_controller.enabled: # Check if IBL is enabled
                if not self.ibl_controller.setup_cube_geometry():
                    print("Warning: Failed to setup IBL cube geometry.")
                if not self.ibl_controller.setup_quad_geometry():
                    print("Warning: Failed to setup IBL quad geometry.")

                if (self.scene_renderer.equirect_to_cubemap_shader is not None and
                    self.scene_renderer.brdf_lut_shader is not None):

                    if not self.ibl_controller.create_environment_cubemap(self.scene_renderer.equirect_to_cubemap_shader):
                        print("Warning: Failed to create environment cubemap for IBL.")
                    if not self.scene_renderer.check_gl_error("IBL cubemap creation"):
                        print("Warning: OpenGL error after IBL cubemap creation.")

                    # Generate irradiance map for diffuse IBL
                    if (self.scene_renderer.irradiance_convolution_shader is not None):
                        if not self.ibl_controller.create_irradiance_map(self.scene_renderer.irradiance_convolution_shader):
                            print("Warning: Failed to create irradiance map for IBL.")
                        if not self.scene_renderer.check_gl_error("IBL irradiance map creation"):
                            print("Warning: OpenGL error after IBL irradiance map creation.")
                    else:
                        print("Warning: Irradiance convolution shader not available.")

                    # Generate prefiltered environment map for specular IBL
                    if (self.scene_renderer.prefilter_env_map_shader is not None):
                        if not self.ibl_controller.create_prefilter_map(self.scene_renderer.prefilter_env_map_shader):
                            print("Warning: Failed to create prefilter map for IBL.")
                        if not self.scene_renderer.check_gl_error("IBL prefilter map creation"):
                            print("Warning: OpenGL error after IBL prefilter map creation.")
                    else:
                        print("Warning: Prefilter environment map shader not available.")

                    if not self.ibl_controller.generate_brdf_lut(self.scene_renderer.brdf_lut_shader):
                        print("Warning: Failed to generate BRDF LUT for IBL.")
                    if not self.scene_renderer.check_gl_error("IBL BRDF LUT generation"):
                        print("Warning: OpenGL error after IBL BRDF LUT generation.")
                else:
                    print("Warning: Skipping IBL cubemap and BRDF LUT generation due to missing shaders.")
            else:
                print("IBL is disabled. Skipping IBL resource initialization.")


            # Standard OpenGL setup
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_PROGRAM_POINT_SIZE) # Allow shaders to set point size
            glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS) # For seamless cubemap filtering
            glDepthFunc(GL_LESS) # Standard depth testing

            # Enable multisampling and line smoothing
            glEnable(GL_MULTISAMPLE)
            glEnable(GL_LINE_SMOOTH)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)


            if not self.check_gl_error("OpenGL state setup"):
                raise RuntimeError("OpenGL error during state setup.")

            self._is_initialized = True  # Mark as initialized only if all steps succeed

        except RuntimeError as e:
            print(f"RuntimeError during OpenGLWidget initialization: {e}")
            self._is_initialized = False # Ensure it's marked as not initialized on error
            # Re-raise if you want to halt execution, or handle appropriately
            # For now, we print and mark as uninitialized.
        except Exception as e:
            print(f"Unexpected error during OpenGLWidget initialization: {e}")
            self._is_initialized = False # Ensure it's marked as not initialized on error
            # Re-raise or handle

    def resizeGL(self, w: int, h: int) -> None:
        """
        Handle widget resize event. Updates viewport and camera projection.

        Args:
            w: New width of the widget.
            h: New height of the widget.
        """
        glViewport(0, 0, w, h)
        self._update_camera()

    def paintGL(self) -> None:
        """Render the scene. Called by Qt when the widget needs to be repainted."""
        if not self._is_initialized: # Check if initialization was successful
            # Optionally, render a blank screen or an error message
            glClearColor(0.5, 0.0, 0.0, 1.0) # Red background for error
            glClear(int(GL_COLOR_BUFFER_BIT) | int(GL_DEPTH_BUFFER_BIT)) # Corrected: Cast to int
            return

        # Proceed with normal rendering
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glClear(int(GL_COLOR_BUFFER_BIT) | int(GL_DEPTH_BUFFER_BIT))
        self._update_geometry_buffers() # Ensure GPU buffers are up-to-date        # Delegate drawing to SceneRenderer
        self.scene_renderer.draw_scene(if_draw_skybox=self.ibl_controller.enabled)

    def _cleanup_gl_resources(self) -> None:
        """
        Clean up OpenGL resources.
        This should be called before the OpenGL context is destroyed.
        """
        print("Cleaning up OpenGLWidget resources...")
        # Ensure the context is current if necessary, though usually handled by Qt.
        # self.makeCurrent()

        # 1. Clear Geometry Buffers
        try:
            if hasattr(self, 'points_buffer') and self.points_buffer:
                self.points_buffer.clear()
                print("Points buffer cleared.")
        except Exception as e:
            print(f"Error clearing points_buffer: {e}")
        try:
            if hasattr(self, 'lines_buffer') and self.lines_buffer:
                self.lines_buffer.clear()
                print("Lines buffer cleared.")
        except Exception as e:
            print(f"Error clearing lines_buffer: {e}")
        try:
            if hasattr(self, 'triangles_buffer') and self.triangles_buffer:
                self.triangles_buffer.clear()
                print("Triangles buffer cleared.")
        except Exception as e:
            print(f"Error clearing triangles_buffer: {e}")

        # 2. Cleanup IBL Renderer
        try:
            if hasattr(self, 'ibl_controller') and self.ibl_controller:
                self.ibl_controller.cleanup()
                print("IBL controller cleaned up.")
        except Exception as e:
            print(f"Error cleaning up IBL controller: {e}")

        # 3. Clear Shader Cache
        try:
            clear_shader_cache()
            # Shader cache print is handled within clear_shader_cache itself
        except Exception as e:
            print(f"Error clearing shader cache during OpenGLWidget cleanup: {e}")

        # self.doneCurrent()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """
        Handle widget close event. Cleans up OpenGL resources.

        Args:
            event: Qt close event.
        """
        self._cleanup_gl_resources()  # Call cleanup before superclass handling
        super().closeEvent(event)
        print("OpenGLWidget closed.")

    # =============================================================================
    # 2. OpenGL/Resource Utility Methods
    # =============================================================================
    def check_gl_error(self, tag: str = "") -> bool:
        """
        Check for OpenGL errors and log them.

        Args:
            tag: Description tag for error context.
        Returns:
            True if no errors, False if errors occurred.
        """
        err = glGetError()
        if err != GL_NO_ERROR:
            err_str = gluErrorString(err)
            if isinstance(err_str, bytes):
                err_str = err_str.decode('utf-8', 'ignore')
            print(f"OpenGL Error ({tag}): Code {err} - {err_str}")
            return False
        return True

    def _setup_geometry_vao(self, buffer: GeometryBuffer,
                            attribute_layout: List[Tuple[int, int, int]]) -> bool:
        """
        Setup VAO for a geometry buffer with given attribute layout.

        Args:
            buffer: Geometry buffer to setup.
            attribute_layout: List of (attr_index, size, offset_in_floats) tuples.
                              Example: [(0, 3, 0), (1, 3, 3)] for pos and color.
        Returns:
            True if successful, False on error.
        """
        try:
            buffer.vao = glGenVertexArrays(1)
            buffer.vbo = glGenBuffers(1)

            glBindVertexArray(buffer.vao)
            glBindBuffer(GL_ARRAY_BUFFER, buffer.vbo)

            stride_bytes = buffer.stride * sizeof(GLfloat)

            for attr_index, size, offset in attribute_layout:
                glVertexAttribPointer(attr_index, size, GL_FLOAT, GL_FALSE,
                                    stride_bytes, ctypes.c_void_p(offset * sizeof(GLfloat)))
                glEnableVertexAttribArray(attr_index)

            glBindVertexArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

            return self.check_gl_error(f"VAO setup for stride {buffer.stride}")

        except Exception as e:
            print(f"Error setting up VAO: {e}")
            return False

    def _convert_attributes_to_buffer_data(self, attrs: GeometryObjectAttributes,
                                         default_color: np.ndarray, # (3,)
                                         is_triangle: bool = False) -> np.ndarray: # (N * stride,)
        """
        Convert object attributes to flattened buffer data format.

        Args:
            attrs: Object attributes to convert.
            default_color: Default color (RGB, shape (3,)) if none provided in attrs.
            is_triangle: Whether this is triangle data (affects layout and stride).

        Returns:
            Flattened buffer data as a 1D NumPy array (float32).
        """
        vertices = attrs.vertices # (N, 3)
        num_vertices = vertices.shape[0]

        # Process colors
        if attrs.colors is None:
            colors = np.tile(default_color, (num_vertices, 1))
        else:
            colors = attrs.colors
            if colors.shape != (num_vertices, 3):
                raise ValueError(f"Colors must be Nx3 array matching {num_vertices} vertices, got {colors.shape}")

        # Process normals
        if attrs.normals is None:
            normals = np.tile(self.defaults.DEFAULT_TRIANGLE_NORMAL, (num_vertices, 1))
        else:
            normals = attrs.normals
            if normals.shape != (num_vertices, 3):
                raise ValueError(f"Normals must be Nx3 array matching {num_vertices} vertices, got {normals.shape}")

        # Process texture coordinates
        if attrs.texcoords is None:
            texcoords = np.tile(self.defaults.DEFAULT_TEXCOORD, (num_vertices, 1))
        else:
            texcoords = attrs.texcoords
            if texcoords.shape != (num_vertices, 2):
                raise ValueError(f"Texcoords must be Nx2 array matching {num_vertices} vertices, got {texcoords.shape}")

        # Process enhanced attributes
        point_sizes = self._process_per_vertex_attribute(
            getattr(attrs, 'point_sizes', None), num_vertices, self.defaults.DEFAULT_POINT_SIZE, "point_sizes")

        line_widths = self._process_per_vertex_attribute(
            getattr(attrs, 'line_widths', None), num_vertices, self.defaults.DEFAULT_LINE_WIDTH, "line_widths")

        shape_types = self._process_per_vertex_attribute(
            getattr(attrs, 'shape_types', None), num_vertices, self.defaults.DEFAULT_SHAPE_TYPE, "shape_types", dtype=np.int32)

        # Process instance data
        if attrs.instance_data is None:
            instance_data = np.tile(self.defaults.DEFAULT_INSTANCE_DATA, (num_vertices, 1))
        else:
            instance_data = attrs.instance_data
            if instance_data.shape != (num_vertices, 4):
                raise ValueError(f"Instance data must be Nx4 array matching {num_vertices} vertices, got {instance_data.shape}")

        # Build interleaved data
        if is_triangle:
            materials = self._process_triangle_materials(getattr(attrs, 'materials', None), num_vertices)
            sss_params_data = self._process_sss_params(getattr(attrs, 'sss_params', None), num_vertices)
            sss_color_data = self._process_sss_color(getattr(attrs, 'sss_color', None), num_vertices)

            # Triangle format: enhanced layout + material(3) + sss_params(4) + sss_color(3)
            interleaved = np.hstack((
                vertices, colors, normals, texcoords, point_sizes,
                line_widths, shape_types.astype(np.float32), instance_data, materials,
                sss_params_data, sss_color_data
            ))
        else:
            # Points/Lines format: enhanced layout only
            interleaved = np.hstack((
                vertices, colors, normals, texcoords, point_sizes,
                line_widths, shape_types.astype(np.float32), instance_data
            ))

        return interleaved.astype(np.float32).flatten() # Return np.ndarray directly, ensure float32

    def _process_per_vertex_attribute(self, attr_data: Optional[np.ndarray],
                                    num_vertices: int, default_value: Union[float, int],
                                    attr_name: str, dtype: type = np.float32) -> np.ndarray: # (N, 1)
        """
        Process per-vertex attributes with validation and reshaping.

        Args:
            attr_data: Optional NumPy array of attribute data. Can be None, 1D (num_vertices,),
                       or 2D (num_vertices, 1).
            num_vertices: Number of vertices for validation.
            default_value: Default value if attr_data is None.
            attr_name: Name of the attribute for error messages.
            dtype: Desired NumPy dtype for the output array.

        Returns:
            A NumPy array of shape (num_vertices, 1) with the processed attribute data.
        """
        if attr_data is None:
            return np.full((num_vertices, 1), default_value, dtype=dtype)

        if attr_data.ndim == 1:
            if attr_data.shape[0] == num_vertices:
                return attr_data.reshape(-1, 1).astype(dtype)
            elif attr_data.shape[0] == 1:
                return np.full((num_vertices, 1), attr_data[0], dtype=dtype)
            else:
                raise ValueError(f"{attr_name} array size {attr_data.shape[0]} doesn't match vertex count {num_vertices}")
        elif attr_data.ndim == 2 and attr_data.shape == (num_vertices, 1):
            return attr_data.astype(dtype)
        else:
            raise ValueError(f"{attr_name} must be a 1D array or Nx1 array, got shape {attr_data.shape}")

    def _process_triangle_materials(self, materials: Optional[np.ndarray], num_vertices: int) -> np.ndarray: # (N, 3)
        """
        Process triangle material attributes ([metallic, roughness, ao]).

        Args:
            materials: Optional NumPy array. Can be None, 1D (3,) for uniform material,
                       or 2D (num_vertices, 3) for per-vertex materials.
            num_vertices: Number of vertices.

        Returns:
            A NumPy array of shape (num_vertices, 3) with material data.
        """
        if materials is None:
            default_material = np.array([self.defaults.DEFAULT_METALLIC, self.defaults.DEFAULT_ROUGHNESS, self.defaults.DEFAULT_AO])
            return np.tile(default_material, (num_vertices, 1))

        if materials.ndim == 1 and materials.shape[0] == 3:
            # Single material for all vertices
            return np.tile(materials, (num_vertices, 1))
        elif materials.shape == (num_vertices, 3):
            # Per-vertex materials
            return materials
        else:
            raise ValueError(f"Materials must be shape (3,) or ({num_vertices}, 3), got {materials.shape}")

    def _process_sss_params(self, sss_params: Optional[np.ndarray], num_vertices: int) -> np.ndarray: # (N, 4)
        """
        Process SSS parameters ([strength, distortion, power, scale]).

        Args:
            sss_params: Optional NumPy array. Can be None, 1D (4,) for uniform SSS,
                        or 2D (num_vertices, 4) for per-vertex SSS.
            num_vertices: Number of vertices.

        Returns:
            A NumPy array of shape (num_vertices, 4) with SSS parameter data.
        """
        if sss_params is None:
            # Default SSS: [strength, distortion, power, scale]
            default_sss = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32) # Default to no SSS effect
            return np.tile(default_sss, (num_vertices, 1))
        if sss_params.ndim == 1 and sss_params.shape[0] == 4:
            return np.tile(sss_params, (num_vertices, 1))
        elif sss_params.shape == (num_vertices, 4):
            return sss_params
        else:
            raise ValueError(f"SSS params must be shape (4,) or ({num_vertices}, 4), got {sss_params.shape}")

    def _process_sss_color(self, sss_color: Optional[np.ndarray], num_vertices: int) -> np.ndarray: # (N, 3)
        """
        Process SSS color.

        Args:
            sss_color: Optional NumPy array. Can be None, 1D (3,) for uniform SSS color,
                       or 2D (num_vertices, 3) for per-vertex SSS color.
            num_vertices: Number of vertices.

        Returns:
            A NumPy array of shape (num_vertices, 3) with SSS color data.
        """
        if sss_color is None:
            default_sss_color = np.array([1.0, 1.0, 1.0], dtype=np.float32) # Default SSS color
            return np.tile(default_sss_color, (num_vertices, 1))
        if sss_color.ndim == 1 and sss_color.shape[0] == 3:
            return np.tile(sss_color, (num_vertices, 1))
        elif sss_color.shape == (num_vertices, 3):
            return sss_color
        else:
            raise ValueError(f"SSS color must be shape (3,) or ({num_vertices}, 3), got {sss_color.shape}")


    def _update_display_buffers(self) -> None:
        """Update internal CPU-side geometry buffers based on display object changes."""
        try:
            # Update points buffer if there are changes
            if self.display_points.has_changes():
                self._update_single_buffer(self.display_points, self.points_buffer,
                                         self.defaults.DEFAULT_POINT_COLOR, False)

            # Update lines buffer if there are changes
            if self.display_lines.has_changes():
                self._update_single_buffer(self.display_lines, self.lines_buffer,
                                         self.defaults.DEFAULT_LINE_COLOR, False)

            # Update triangles buffer if there are changes
            if self.display_triangles.has_changes():
                self._update_single_buffer(self.display_triangles, self.triangles_buffer,
                                         self.defaults.DEFAULT_TRIANGLE_COLOR, True)

        except Exception as e:
            print(f"Error updating display buffers: {e}")
            # Consider more robust error handling or logging if this is critical for stability,
            # for example, re-raising or setting an error state

    def _update_single_buffer(self, display_obj: Any, buffer: GeometryBuffer,
                            default_color: np.ndarray, # (3,)
                            is_triangle: bool) -> None:
        """
        Update a single CPU-side geometry buffer from its corresponding display object changes.

        Args:
            display_obj: The display object manager (e.g., DisplayedPoints).
            buffer: The GeometryBuffer to update.
            default_color: Default color (RGB, shape (3,)) for objects without specified colors.
            is_triangle: True if processing triangle data, False otherwise.
        """
        # Handle deletions first
        for key in display_obj.deleted_objects:
            buffer.remove_object_data(key)

        # Handle additions and modifications
        for key in display_obj.dirty_objects:
            if key in display_obj.objects:
                attrs = display_obj.objects[key]
                if attrs.visible:
                    try:
                        buffer_data = self._convert_attributes_to_buffer_data(attrs, default_color, is_triangle)
                        vertex_count = attrs.vertices.shape[0]
                        buffer.update_object_data(key, buffer_data, vertex_count)
                    except Exception as e:
                        print(f"Error converting object {key} to buffer data: {e}")
                        buffer.remove_object_data(key)
                else:
                    buffer.remove_object_data(key)

        display_obj.mark_clean()

    def _update_geometry_buffers(self) -> None:
        """Update all CPU-side geometry buffers and then upload changes to GPU VBOs."""
        # Update display buffers first
        self._update_display_buffers()

        # Then update GPU buffers
        self.points_buffer.update_gpu_buffer()
        self.lines_buffer.update_gpu_buffer()
        self.triangles_buffer.update_gpu_buffer()

        self.check_gl_error("Geometry buffer updates")

    # =============================================================================
    # 3. Display Object Management (CRUD)
    # =============================================================================
    def add_points(self, vertices: np.ndarray, # (N, 3)
                   key: Optional[str] = None, **kwargs) -> str:
        """
        Add points using the display object system.

        Args:
            vertices: Nx3 NumPy array of point positions.
            key: Optional unique identifier for the point object. If None, one is generated.
            **kwargs: Additional point attributes (e.g., colors (N,3), point_sizes (N,1) or (N,),
                      shape_types (N,1) or (N,), instance_data (N,4), normals (N,3), texcoords (N,2)).

        Returns:
            The generated or provided key for the added points object.
        """
        result_key = self.display_points.add(key=key, vertices=vertices, **kwargs)
        if self.isVisible():
            self.update()
        return result_key

    def add_lines(self, vertices: np.ndarray, # (N, 3)
                  key: Optional[str] = None, **kwargs) -> str:
        """
        Add lines using the display object system. Vertices should be in pairs for line segments.

        Args:
            vertices: Nx3 NumPy array of line vertices.
            key: Optional unique identifier for the line object. If None, one is generated.
            **kwargs: Additional line attributes (e.g., colors (N,3), line_widths (N,1) or (N,),
                      instance_data (N,4), normals (N,3), texcoords (N,2)).

        Returns:
            The generated or provided key for the added lines object.
        """
        result_key = self.display_lines.add(key=key, vertices=vertices, **kwargs)
        if self.isVisible():
            self.update()
        return result_key

    def add_triangles(self, vertices: np.ndarray, # (N, 3)
                      key: Optional[str] = None, **kwargs) -> str:
        """
        Add triangles using the display object system. Vertices should be in triplets.

        Args:
            vertices: Nx3 NumPy array of triangle vertices.
            key: Optional unique identifier for the triangle object. If None, one is generated.
            **kwargs: Additional triangle attributes (e.g., colors (N,3), normals (N,3),
                      texcoords (N,2), materials (N,3) or (3,), sss_params (N,4) or (4,),
                      sss_color (N,3) or (3,), instance_data (N,4)).

        Returns:
            The generated or provided key for the added triangles object.
        """
        result_key = self.display_triangles.add(key=key, vertices=vertices, **kwargs)
        if self.isVisible():
            self.update()
        return result_key

    def _delete_object_type(self, display_manager: Any, key: str) -> bool:
        """
        Internal helper to delete an object from a specific display manager.

        Args:
            display_manager: The display manager instance (e.g., self.display_points).
            key: The unique key of the object to delete.

        Returns:
            True if the object was found and deleted, False otherwise.
        """
        result = display_manager.delete(key)
        if result and self.isVisible():
            self.update()
        return result

    def delete_points(self, key: str) -> bool:
        """
        Delete points object by key.

        Args:
            key: The unique key of the points object to delete.

        Returns:
            True if deleted, False if key not found.
        """
        return self._delete_object_type(self.display_points, key)

    def delete_lines(self, key: str) -> bool:
        """
        Delete lines object by key.

        Args:
            key: The unique key of the lines object to delete.

        Returns:
            True if deleted, False if key not found.
        """
        return self._delete_object_type(self.display_lines, key)

    def delete_triangles(self, key: str) -> bool:
        """
        Delete triangles object by key.

        Args:
            key: The unique key of the triangles object to delete.

        Returns:
            True if deleted, False if key not found.
        """
        return self._delete_object_type(self.display_triangles, key)

    def _modify_object_type(self, display_manager: Any, key: str, **kwargs) -> bool:
        """
        Internal helper to modify an object in a specific display manager.

        Args:
            display_manager: The display manager instance (e.g., self.display_points).
            key: The unique key of the object to modify.
            **kwargs: Attributes to modify.

        Returns:
            True if the object was found and modified, False otherwise.
        """
        result = display_manager.modify(key, **kwargs)
        if result and self.isVisible():
            self.update()
        return result

    def modify_points(self, key: str, **kwargs) -> bool:
        """
        Modify points object attributes by key.

        Args:
            key: The unique key of the points object to modify.
            **kwargs: Attributes to modify (e.g., vertices=new_vertices, colors=new_colors).

        Returns:
            True if modified, False if key not found or validation failed.
        """
        return self._modify_object_type(self.display_points, key, **kwargs)

    def modify_lines(self, key: str, **kwargs) -> bool:
        """
        Modify lines object attributes by key.

        Args:
            key: The unique key of the lines object to modify.
            **kwargs: Attributes to modify (e.g., vertices=new_vertices, colors=new_colors).

        Returns:
            True if modified, False if key not found or validation failed.
        """
        return self._modify_object_type(self.display_lines, key, **kwargs)

    def modify_triangles(self, key: str, **kwargs) -> bool:
        """
        Modify triangles object attributes by key.

        Args:
            key: The unique key of the triangles object to modify.
            **kwargs: Attributes to modify (e.g., vertices=new_vertices, colors=new_colors).

        Returns:
            True if modified, False if key not found or validation failed.
        """
        return self._modify_object_type(self.display_triangles, key, **kwargs)

    def get_points(self, key: str) -> Optional[PointAttributes]:
        """
        Get points attributes by key.

        Args:
            key: The unique key of the points object.

        Returns:
            PointAttributes if found, None otherwise.
        """
        result = self.display_points.get(key)
        return result if result is None else PointAttributes(**result.__dict__)

    def get_lines(self, key: str) -> Optional[LineAttributes]:
        """
        Get lines attributes by key.

        Args:
            key: The unique key of the lines object.

        Returns:
            LineAttributes if found, None otherwise.
        """
        result = self.display_lines.get(key)
        return result if result is None else LineAttributes(**result.__dict__)

    def get_triangles(self, key: str) -> Optional[TriangleAttributes]:
        """
        Get triangles attributes by key.

        Args:
            key: The unique key of the triangles object.

        Returns:
            TriangleAttributes if found, None otherwise.
        """
        result = self.display_triangles.get(key)
        return result if result is None else TriangleAttributes(**result.__dict__)

    def list_points_keys(self) -> List[str]:
        """
        List all point object keys.

        Returns:
            List of point object keys.
        """
        return list(self.display_points.objects.keys())

    def list_lines_keys(self) -> List[str]:
        """
        List all line object keys.

        Returns:
            List of line object keys.
        """
        return list(self.display_lines.objects.keys())

    def list_triangles_keys(self) -> List[str]:
        """
        List all triangle object keys.

        Returns:
            List of triangle object keys.
        """
        return list(self.display_triangles.objects.keys())

    def clear_geometry(self, points: bool = True, lines: bool = True, triangles: bool = True) -> None:
        """
        Clear geometry data from specified display object managers.

        Args:
            points: If True, clear all points objects.
            lines: If True, clear all lines objects.
            triangles: If True, clear all triangles objects.
        """
        if points:
            self.display_points.clear()
        if lines:
            self.display_lines.clear()
        if triangles:
            self.display_triangles.clear()

        if self.isVisible():
            self.update()

    def set_draw_mode(self, draw_individually: bool) -> None:
        """
        Set the drawing mode for objects.

        Args:
            draw_individually: If True, objects are drawn one by one, enabling
                               potential per-object culling. If False, all objects
                               of a type are drawn in a single batch.
        """
        self._draw_objects_individually = draw_individually
        if self.scene_renderer:
            self.scene_renderer.draw_objects_individually = draw_individually
        if self.isVisible():
            self.update()

    # =============================================================================
    # 4. Rendering/Shader/Parameter Settings
    # =============================================================================
    def set_sss_enabled(self, enabled: bool) -> None:
        """
        Enable or disable Subsurface Scattering globally for relevant shaders.

        Args:
            enabled: True to enable SSS, False to disable.
        """
        self.base_shader_controller.set_sss_enabled(enabled)
        if self.isVisible():
            self.update()

    def set_use_texture(self, enabled: bool) -> None:
        """
        Enable or disable global texture usage for shaders that support it.

        Args:
            enabled: True to enable texture mapping, False to disable.
        """
        self.base_shader_controller.use_texture = enabled
        if self.isVisible():
            self.update()

    def set_diffuse_texture(self, texture_id: Optional[int]) -> None:
        """
        Set the global diffuse texture ID for shaders.

        Args:
            texture_id: OpenGL texture ID for the diffuse map, or None to unset.
        """
        self.base_shader_controller.diffuse_texture_id = texture_id
        if self.isVisible():
            self.update()

    def set_shape_texture(self, texture_id: Optional[int]) -> None:
        """
        Set the global shape texture ID (e.g., for custom point shapes).

        Args:
            texture_id: OpenGL texture ID for the shape map, or None to unset.
        """
        self.base_shader_controller.shape_texture_id = texture_id
        if self.isVisible():
            self.update()

    def set_sss_params_for_triangles(self, key: str,
                                   strength: float = 1.0,
                                   distortion: float = 0.2,
                                   power: float = 2.0,
                                   scale: float = 1.0,
                                   sss_color: Optional[List[float]] = None) -> bool: # List[float] (3,)
        """
        Set SSS parameters for a specific triangle object.

        Args:
            key: Triangle object key.
            strength: SSS effect strength.
            distortion: Light scattering distortion.
            power: Scattering power/falloff.
            scale: Overall scaling factor.
            sss_color: SSS color (RGB list/tuple of 3 floats), defaults to white if None.

        Returns:
            True if successful, False if object not found or modification failed.
        """
        if key not in self.display_triangles.objects:
            return False

        sss_params = np.array([strength, distortion, power, scale], dtype=np.float32)
        if sss_color is None:
            sss_color = [1.0, 1.0, 1.0]
        sss_color_array = np.array(sss_color, dtype=np.float32)

        return self.modify_triangles(key, sss_params=sss_params, sss_color=sss_color_array)

    def set_camera_params(self, eye: Optional[List[float]] = None, # (3,)
                         target: Optional[List[float]] = None, # (3,)
                         up: Optional[List[float]] = None, # (3,)
                         fov_y_degrees: Optional[float] = None,
                         near_plane: Optional[float] = None,
                         far_plane: Optional[float] = None) -> None:
        """
        Set multiple camera parameters with validation.

        Args:
            eye: Camera position in world space (list/tuple of 3 floats).
            target: Camera look-at point in world space (list/tuple of 3 floats).
            up: Camera world up vector (list/tuple of 3 floats).
            fov_y_degrees: Vertical field of view in degrees.
            near_plane: Near clipping plane distance.
            far_plane: Far clipping plane distance.
        """
        if target is not None:
            self.camera.target = pyrr.Vector3(target)
        if up is not None:
            self.camera.world_up = pyrr.Vector3(up)
        if eye is not None:
            self.camera.eye = pyrr.Vector3(eye)
            diff = self.camera.eye - self.camera.target
            self.camera.distance = float(np.linalg.norm(diff))
            if self.camera.distance > 1e-6:
                normalized_diff = diff / self.camera.distance
                self.camera.elevation_deg = np.degrees(np.arcsin(np.clip(normalized_diff.y, -1.0, 1.0)))
                self.camera.azimuth_deg = np.degrees(np.arctan2(normalized_diff.x, normalized_diff.z))
            else:
                self.camera.elevation_deg = self.camera.initial_elevation_deg
                self.camera.azimuth_deg = self.camera.initial_azimuth_deg
                self.camera.distance = max(self.camera.initial_distance, 1.0)
                self.camera._calculate_eye_position()

        if fov_y_degrees is not None:
            self.camera.fov_degrees = np.clip(fov_y_degrees, 1.0, 179.0)
        if near_plane is not None:
            self.camera.near_plane = max(0.001, near_plane)
        if far_plane is not None:
            self.camera.far_plane = max(self.camera.near_plane + 0.1, far_plane)

        self._update_camera()

    def _update_camera(self, aspect_ratio_override: Optional[float] = None) -> None:
        """
        Update camera matrices based on current parameters and trigger repaint if needed.

        Args:
            aspect_ratio_override: Optional aspect ratio to use. If None, uses widget's ratio.
        """
        if aspect_ratio_override is None:
            aspect_ratio = self.width() / self.height() if self.height() > 0 else 1.0
        else:
            aspect_ratio = aspect_ratio_override

        self.camera.update_matrices(aspect_ratio)

        if aspect_ratio_override is None and self.isVisible():
            self.update()
            self.CameraUpdated.emit(self.camera.get_camera_info())

    def set_lighting(self, position: Optional[List[float]] = None, # (3,)
                    color: Optional[List[float]] = None, # (3,)
                    ambient_strength: Optional[float] = None,
                    specular_strength: Optional[float] = None,
                    shininess: Optional[float] = None) -> None:
        """
        Set lighting parameters for the base shader.

        Args:
            position: Light position in world space (list/tuple of 3 floats).
            color: Light color (RGB list/tuple of 3 floats).
            ambient_strength: Ambient light intensity [0,1].
            specular_strength: Specular reflection intensity [0,1].
            shininess: Specular shininess exponent.
        """
        self.base_shader_controller.set_lighting_params(
            position=np.array(position) if position else None,
            color=np.array(color) if color else None,
            ambient=ambient_strength,
            specular=specular_strength,
            shininess=shininess
        )
        if self.isVisible():
            self.update()

    def set_fog(self, enabled: bool, color: Optional[List[float]] = None, # (3,)
               near: Optional[float] = None, far: Optional[float] = None) -> None:
        """
        Set fog parameters for the base shader.

        Args:
            enabled: True to enable fog, False to disable.
            color: Fog color (RGB list/tuple of 3 floats).
            near: Fog start distance.
            far: Fog end distance.
        """
        self.base_shader_controller.set_fog_params(
            enabled=enabled,
            color=np.array(color) if color else None,
            near=near,
            far=far
        )
        if self.isVisible():
            self.update()

    def set_wireframe(self, enabled: bool, color: Optional[List[float]] = None) -> None: # (3,)
        """
        Set wireframe effect parameters for the base shader.

        Args:
            enabled: True to enable wireframe overlay, False to disable.
            color: Wireframe line color (RGB list/tuple of 3 floats).
        """
        self.base_shader_controller.set_wireframe_params(
            enabled=enabled,
            color=np.array(color) if color else None        )
        if self.isVisible():
            self.update()

    def change_skybox(self, exr_filepath: str) -> bool:
        """
        Change the current skybox to the specified HDR .exr file at runtime.

        Args:
            exr_filepath: Path to a new .exr HDR environment map. If None or empty, IBL is disabled.

        Returns:
            True if operation initiated and likely successful, False on failure.
        """
        if not self._is_initialized:
            print("Error: Viewer3DWidget not initialized. Cannot change skybox.")
            return False

        if not self.ibl_controller:
            print("Error: IBLController not available. Cannot change skybox.")
            return False

        self.makeCurrent() # Ensure OpenGL context is current

        try:
            if not exr_filepath:
                if self.ibl_controller.enabled:
                    print("Disabling IBL and skybox.")
                    self.ibl_controller.enabled = False
                    self.ibl_controller.cleanup_environment_maps()
                    if self.isVisible():
                        self.update()
                else:
                    print("IBL already disabled.")
                return True # Successfully disabled or was already disabled

            print(f"Attempting to change skybox to: {exr_filepath}")

            # Check shader availability first
            if not self.scene_renderer.brdf_lut_shader:
                print("Error: BRDF LUT shader is not available in SceneRenderer.")
                return False
            if not self.scene_renderer.equirect_to_cubemap_shader:
                print("Error: Equirectangular to cubemap shader is not available in SceneRenderer.")
                return False
            if not self.scene_renderer.irradiance_convolution_shader:
                print("Error: Irradiance convolution shader is not available in SceneRenderer.")
                return False
            if not self.scene_renderer.prefilter_env_map_shader:
                print("Error: Prefilter environment map shader is not available in SceneRenderer.")
                return False

            # 1. Ensure IBL prerequisites are met (one-time setup)
            needs_one_time_setup = False
            if self.ibl_controller.cube_vao is None or self.ibl_controller.cube_vbo is None:
                needs_one_time_setup = True
            if self.ibl_controller.quad_vao is None or self.ibl_controller.quad_vbo is None:
                needs_one_time_setup = True
            if self.ibl_controller.brdf_lut_texture is None:
                needs_one_time_setup = True

            if needs_one_time_setup:
                print("Performing one-time IBL setup (geometry, BRDF LUT)...")
                if self.ibl_controller.cube_vao is None or self.ibl_controller.cube_vbo is None:
                    if not self.ibl_controller.setup_cube_geometry():
                        print("Error: Failed to setup IBL cube geometry during change_skybox.")
                        return False
                if self.ibl_controller.quad_vao is None or self.ibl_controller.quad_vbo is None:
                    if not self.ibl_controller.setup_quad_geometry():
                        print("Error: Failed to setup IBL quad geometry during change_skybox.")
                        return False
                if self.ibl_controller.brdf_lut_texture is None:
                    # Temporarily enable IBL so BRDF LUT generation can proceed
                    self.ibl_controller.enabled = True
                    if not self.ibl_controller.generate_brdf_lut(self.scene_renderer.brdf_lut_shader):
                        print("Error: Failed to generate BRDF LUT during change_skybox.")
                        return False
                print("One-time IBL setup completed.")

            # 2. Clean up previous IBL environment-specific resources
            self.ibl_controller.cleanup_environment_maps()

            # 3. Update the EXR filepath in the IBL controller
            self.ibl_controller.exr_filepath = exr_filepath

            # 4. Re-create environment cubemap from the new HDR
            if not self.ibl_controller.create_environment_cubemap(self.scene_renderer.equirect_to_cubemap_shader):
                print(f"Error: Failed to create environment cubemap from {exr_filepath}")
                self.ibl_controller.enabled = False
                return False
            if not self.scene_renderer.check_gl_error("New IBL cubemap creation"):
                self.ibl_controller.enabled = False
                return False
            print("  - New environment cubemap created.")

            # 5. Re-generate irradiance map
            if not self.ibl_controller.create_irradiance_map(self.scene_renderer.irradiance_convolution_shader):
                print("Error: Failed to create new irradiance map.")
                self.ibl_controller.enabled = False
                return False
            if not self.scene_renderer.check_gl_error("New IBL irradiance map creation"):
                self.ibl_controller.enabled = False
                return False
            print("  - New irradiance map created.")

            # 6. Re-generate prefiltered environment map
            if not self.ibl_controller.create_prefilter_map(self.scene_renderer.prefilter_env_map_shader):
                print("Error: Failed to create new prefilter map.")
                self.ibl_controller.enabled = False
                return False
            if not self.scene_renderer.check_gl_error("New IBL prefilter map creation"):
                self.ibl_controller.enabled = False
                return False
            print("  - New prefilter map created.")

            self.ibl_controller.enabled = True
            print(f"Skybox successfully changed to {exr_filepath} and IBL enabled.")

            if self.isVisible():
                self.update()
            return True

        except Exception as e:
            print(f"Critical error during change_skybox: {e}")
            if self.ibl_controller:
                self.ibl_controller.enabled = False
                self.ibl_controller.cleanup_environment_maps()
            if self.isVisible():
                self.update()
            return False

        finally:
            self.doneCurrent() # Release context

    # =============================================================================
    # 5. Interaction Event Handlers
    # =============================================================================
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """
        Handle mouse press events for camera interaction.

        Args:
            event: Qt mouse press event.
        """
        self.camera.handle_mouse_press(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        """
        Handle mouse release events for camera interaction.

        Args:
            event: Qt mouse release event.
        """
        self.camera.handle_mouse_release(event)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        """
        Handle mouse double-click events (e.g., reset camera).

        Args:
            event: Qt mouse double-click event.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            self.camera.reset_to_initial()
            self._update_camera()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """
        Handle mouse move events for camera rotation and panning.

        Args:
            event: Qt mouse move event.
        """
        if self.camera.handle_mouse_move(event):
            self._update_camera()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        """
        Handle mouse wheel events for camera zooming.

        Args:
            event: Qt wheel event.
        """
        if self.camera.handle_wheel(event):
            self._update_camera()


