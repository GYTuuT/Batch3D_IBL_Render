"""
Camera controller for 3D scene navigation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pyrr
from PySide6 import QtCore, QtGui
from PySide6.QtCore import Qt

from .config import RenderingDefaults


class CameraController:
    """
    Advanced camera controller for 3D scene navigation, supporting orbit, pan, and zoom.
    """    # Constants for automatic near/far plane adjustment
    MIN_NEAR_PLANE_FACTOR = 0.001
    DEFAULT_NEAR_PLANE_MIN = 0.001
    DEFAULT_FAR_PLANE_FACTOR = 10.0
    NEAR_PLANE_CLIP_MIN = 0.001
    NEAR_PLANE_CLIP_MAX = 10.0
    FAR_PLANE_CLIP_MAX_FACTOR = 10000.0

    # Additional constants for magic numbers
    AZIMUTH_NORMALIZATION_DEGREES = 360.0
    MIN_ELEVATION_CONSTRAINT = -90.0
    MAX_ELEVATION_CONSTRAINT = 90.0
    MIN_DISTANCE_CONSTRAINT = 0.001
    MAX_DISTANCE_CONSTRAINT_FACTOR = 0.1
    MIN_ASPECT_RATIO = 1.0
    DEFAULT_PADDING_FACTOR = 1.2
    NEAR_FAR_PLANE_SAFETY_FACTOR = 2.0

    def __init__(self,
                 initial_target: Optional[Union[List[float], np.ndarray, pyrr.Vector3]] = None, # (3,)
                 initial_distance: Optional[float] = None,
                 initial_azimuth_deg: Optional[float] = None,
                 initial_elevation_deg: Optional[float] = None,
                 initial_fov_degrees: Optional[float] = None,
                 initial_near_plane: Optional[float] = None,
                 initial_far_plane: Optional[float] = None,
                 world_up: Optional[Union[List[float], np.ndarray, pyrr.Vector3]] = None # (3,)
                 ):
        """
        Initialize camera controller.

        Args:
            initial_target: Initial look-at point (list/array/Vector3 of 3 floats).
            initial_distance: Initial distance from target to camera.
            initial_azimuth_deg: Initial horizontal angle around target (degrees).
            initial_elevation_deg: Initial vertical angle above target (degrees).
            initial_fov_degrees: Initial vertical field of view (degrees).
            initial_near_plane: Initial near clipping plane distance.
            initial_far_plane: Initial far clipping plane distance.
            world_up: World's up direction (list/array/Vector3 of 3 floats).
        """
        # Initialize camera controller with default or provided parameters
        self.initial_target = pyrr.Vector3(initial_target) if initial_target is not None else pyrr.Vector3([0.0, 0.0, 0.0])
        self.initial_distance = initial_distance if initial_distance is not None else 10.0
        self.initial_azimuth_deg = initial_azimuth_deg if initial_azimuth_deg is not None else 0.0
        self.initial_elevation_deg = initial_elevation_deg if initial_elevation_deg is not None else 20.0
        self.initial_fov_degrees = initial_fov_degrees if initial_fov_degrees is not None else 45.0
        self.initial_near_plane = initial_near_plane if initial_near_plane is not None else 0.1
        self.initial_far_plane = initial_far_plane if initial_far_plane is not None else 100.0
        self.initial_world_up = pyrr.Vector3(world_up) if world_up is not None else pyrr.Vector3([0.0, 1.0, 0.0])

        # Current camera state
        self.target = self.initial_target.copy()
        self.distance = self.initial_distance
        self.azimuth_deg = self.initial_azimuth_deg
        self.elevation_deg = self.initial_elevation_deg
        self.world_up = self.initial_world_up.copy()
        self.eye = pyrr.Vector3()

        # Camera parameters
        self.fov_degrees = self.initial_fov_degrees
        self.near_plane = self.initial_near_plane
        self.far_plane = self.initial_far_plane

        # Transformation matrices
        self.view_matrix = pyrr.matrix44.create_identity()
        self.projection_matrix = pyrr.matrix44.create_identity()
        self.model_matrix = pyrr.matrix44.create_identity()

        # Mouse interaction state
        self.last_mouse_pos: Optional[QtCore.QPointF] = None
        self.left_button_down = False
        self.right_button_down = False
        self.middle_button_down = False

        # Interaction sensitivity settings
        self.rotation_sensitivity = RenderingDefaults.CAMERA_ROTATION_SENSITIVITY
        self.pan_sensitivity = RenderingDefaults.CAMERA_PAN_SENSITIVITY
        self.zoom_sensitivity = RenderingDefaults.CAMERA_ZOOM_SENSITIVITY

        # Camera constraints
        self.min_distance = 0.1
        self.max_distance = 1000.0
        self.min_elevation_deg = -89.9
        self.max_elevation_deg = 89.9
        self.min_fov_degrees = 1.0
        self.max_fov_degrees = 179.0

        # Advanced features
        self.auto_adjust_near_far = True
        self.distance_based_pan_scaling = True
        self.smooth_transitions = False
        self.transition_speed = 5.0

        # Initialize eye position
        self._calculate_eye_position()

    def _calculate_eye_position(self) -> None:
        """
        Calculate Cartesian eye position from spherical coordinates (distance, azimuth, elevation)
        relative to the target point. Updates self.eye.
        """
        # Clamp elevation to prevent gimbal lock
        self.elevation_deg = np.clip(self.elevation_deg,
                                   self.min_elevation_deg,
                                   self.max_elevation_deg)

        # Clamp distance to reasonable bounds
        self.distance = np.clip(self.distance, self.min_distance, self.max_distance)

        # Convert to radians
        rad_azimuth = np.radians(self.azimuth_deg)
        rad_elevation = np.radians(self.elevation_deg)

        # Calculate eye position in spherical coordinates
        self.eye.x = (self.target.x +
                     self.distance * np.cos(rad_elevation) * np.sin(rad_azimuth))
        self.eye.y = self.target.y + self.distance * np.sin(rad_elevation)
        self.eye.z = (self.target.z +
                     self.distance * np.cos(rad_elevation) * np.cos(rad_azimuth))

    def _auto_adjust_near_far_planes(self) -> None:
        """
        Automatically adjust near and far clipping planes based on camera distance to target.
        This helps prevent z-fighting and ensures optimal depth buffer precision.
        Updates self.near_plane and self.far_plane if self.auto_adjust_near_far is True.
        """
        if not self.auto_adjust_near_far:
            return

        # Calculate optimal near/far based on distance to target
        scene_size = self.distance

        # Near plane: small fraction of distance, but not too small
        self.near_plane = max(self.DEFAULT_NEAR_PLANE_MIN, scene_size * self.MIN_NEAR_PLANE_FACTOR)

        # Far plane: multiple of distance to encompass scene
        self.far_plane = max(self.near_plane * self.NEAR_FAR_PLANE_SAFETY_FACTOR, scene_size * self.DEFAULT_FAR_PLANE_FACTOR)

        # Ensure reasonable bounds
        self.near_plane = np.clip(self.near_plane, self.NEAR_PLANE_CLIP_MIN, self.NEAR_PLANE_CLIP_MAX)
        max_far_val = max(self.near_plane * self.NEAR_FAR_PLANE_SAFETY_FACTOR, self.FAR_PLANE_CLIP_MAX_FACTOR)
        self.far_plane = np.clip(self.far_plane, self.near_plane * self.NEAR_FAR_PLANE_SAFETY_FACTOR, max_far_val)

    def update_matrices(self, aspect_ratio: float) -> None:
        """
        Update view and projection matrices based on current camera state.

        Args:
            aspect_ratio: Viewport width / height ratio. Must be > 0.
        """
        # Validate aspect ratio
        if aspect_ratio <= 0:
            aspect_ratio = self.MIN_ASPECT_RATIO

        # Calculate eye position
        self._calculate_eye_position()

        # Auto-adjust planes if enabled
        self._auto_adjust_near_far_planes()

        # Create view matrix
        try:
            self.view_matrix = pyrr.matrix44.create_look_at(
                self.eye, self.target, self.world_up)
        except Exception as e:
            print(f"Warning: Failed to create view matrix (eye: {self.eye}, target: {self.target}, up: {self.world_up}). Using identity. Error: {e}")
            self.view_matrix = pyrr.matrix44.create_identity()

        # Create projection matrix
        try:            self.projection_matrix = pyrr.matrix44.create_perspective_projection_matrix(
                self.fov_degrees, aspect_ratio, self.near_plane, self.far_plane)
        except ValueError as ve:
            print(f"Warning: Failed to create projection matrix (fov: {self.fov_degrees}, aspect: {aspect_ratio}, near: {self.near_plane}, far: {self.far_plane}). Using identity. Error: {ve}")
            self.projection_matrix = pyrr.matrix44.create_identity()
        except Exception as e:
            print(f"Warning: Unexpected error creating projection matrix. Using identity. Error: {e}")
            self.projection_matrix = pyrr.matrix44.create_identity()

    def reset_to_initial(self) -> None:
        """Reset camera to its initial position, orientation, and parameters."""
        self.target = self.initial_target.copy()
        self.distance = self.initial_distance
        self.azimuth_deg = self.initial_azimuth_deg
        self.elevation_deg = self.initial_elevation_deg
        self.world_up = self.initial_world_up.copy()
        self.fov_degrees = self.initial_fov_degrees
        self.near_plane = self.initial_near_plane
        self.far_plane = self.initial_far_plane

    def set_view_preset(self, preset: str) -> bool:
        """
        Set camera orientation to a predefined view preset.

        Args:
            preset: View preset name (e.g., 'front', 'back', 'left', 'right', 'top', 'bottom', 'iso').
                    Case-insensitive.

        Returns:
            True if preset was applied, False if preset name is unknown.
        """
        presets = {
            'front': RenderingDefaults.VIEW_PRESET_FRONT,
            'back': RenderingDefaults.VIEW_PRESET_BACK,
            'left': RenderingDefaults.VIEW_PRESET_LEFT,
            'right': RenderingDefaults.VIEW_PRESET_RIGHT,
            'top': RenderingDefaults.VIEW_PRESET_TOP,
            'bottom': RenderingDefaults.VIEW_PRESET_BOTTOM,
            'iso': RenderingDefaults.VIEW_PRESET_ISO,
            'iso2': RenderingDefaults.VIEW_PRESET_ISO2,
        }

        if preset.lower() in presets:
            self.azimuth_deg, self.elevation_deg = presets[preset.lower()]
            return True
        return False

    def frame_bounds(self, min_bounds: np.ndarray, # (3,)
                    max_bounds: np.ndarray, # (3,)
                    padding_factor: Optional[float] = None) -> None:
        """
        Adjust camera target and distance to frame a given bounding box.

        Args:
            min_bounds: Minimum corner of the bounding box (NumPy array, shape (3,)).
            max_bounds: Maximum corner of the bounding box (NumPy array, shape (3,)).
            padding_factor: Factor to increase calculated distance for padding (e.g., 1.2 for 20% padding).
        """
        if padding_factor is None:
            padding_factor = self.DEFAULT_PADDING_FACTOR

        # Calculate bounding box center and size
        center = (min_bounds + max_bounds) * 0.5
        size = np.linalg.norm(max_bounds - min_bounds)

        # Set target to center
        self.target = pyrr.Vector3(center)

        # Calculate distance based on size and FOV
        if size > 0:
            # Calculate distance to fit object in view
            half_fov_rad = np.radians(self.fov_degrees * 0.5)
            distance = (size * 0.5) / np.tan(half_fov_rad)
            self.distance = distance * padding_factor
        else:
            self.distance = self.initial_distance

    def get_camera_info(self) -> Dict[str, Any]:
        """
        Get a dictionary containing comprehensive current camera parameters.

        Returns:
            A dictionary with keys like 'target', 'eye', 'distance', 'fov_degrees', etc.
        """
        return {
            'target': self.target.tolist(),
            'eye': self.eye.tolist(),
            'distance': self.distance,
            'azimuth_deg': self.azimuth_deg,
            'elevation_deg': self.elevation_deg,
            'fov_degrees': self.fov_degrees,
            'near_plane': self.near_plane,
            'far_plane': self.far_plane,
            'world_up': self.world_up.tolist(),
        }

    def handle_mouse_press(self, event: QtGui.QMouseEvent) -> None:
        """
        Handle mouse press events for camera interaction.

        Args:
            event: Qt mouse press event (QtGui.QMouseEvent).
        """
        self.last_mouse_pos = event.position()

        if event.button() == Qt.MouseButton.LeftButton:
            self.left_button_down = True
        elif event.button() == Qt.MouseButton.RightButton:
            self.right_button_down = True
        elif event.button() == Qt.MouseButton.MiddleButton:
            self.middle_button_down = True

    def handle_mouse_release(self, event: QtGui.QMouseEvent) -> None:
        """
        Handle mouse release events for camera interaction.

        Args:
            event: Qt mouse release event (QtGui.QMouseEvent).
        """
        if event.button() == Qt.MouseButton.LeftButton:
            self.left_button_down = False
        elif event.button() == Qt.MouseButton.RightButton:
            self.right_button_down = False
        elif event.button() == Qt.MouseButton.MiddleButton:
            self.middle_button_down = False

        self.last_mouse_pos = None

    def handle_mouse_move(self, event: QtGui.QMouseEvent) -> bool:
        """
        Handle mouse move events for camera rotation and panning.

        Args:
            event: Qt mouse move event (QtGui.QMouseEvent).

        Returns:
            True if the camera state was updated due to the move, False otherwise.
        """
        if self.last_mouse_pos is None:
            return False

        # Calculate mouse delta
        current_pos = event.position()
        dx = current_pos.x() - self.last_mouse_pos.x()
        dy = current_pos.y() - self.last_mouse_pos.y()

        camera_updated = False

        if self.left_button_down:
            # Rotation around target
            self.azimuth_deg -= dx * self.rotation_sensitivity
            self.elevation_deg += dy * self.rotation_sensitivity

            # Normalize azimuth to [0, 360)
            self.azimuth_deg = self.azimuth_deg % self.AZIMUTH_NORMALIZATION_DEGREES

            camera_updated = True

        elif self.right_button_down or self.middle_button_down:
            # Panning in screen space
            camera_updated = self._handle_pan(dx, dy)

        self.last_mouse_pos = current_pos
        return camera_updated

    def _handle_pan(self, dx: float, dy: float) -> bool:
        """
        Internal: Handle camera panning based on mouse delta.

        Args:
            dx: Horizontal mouse delta in pixels.
            dy: Vertical mouse delta in pixels.

        Returns:
            True if camera was panned, False if an error occurred (e.g., degenerate view).
        """
        try:
            # Calculate current view vectors
            forward_vec = pyrr.vector.normalise(self.target - self.eye)
            right_vec = pyrr.vector.normalise(np.cross(forward_vec, self.world_up))
            up_vec = pyrr.vector.normalise(np.cross(right_vec, forward_vec))

            # Scale pan sensitivity based on distance if enabled
            if self.distance_based_pan_scaling:
                effective_pan_sensitivity = self.pan_sensitivity * max(self.distance, 1.0)
            else:
                effective_pan_sensitivity = self.pan_sensitivity

            # Calculate pan offset in world space
            pan_offset = (right_vec * (-dx * effective_pan_sensitivity) +
                         up_vec * (dy * effective_pan_sensitivity))

            # Apply pan to target
            self.target += pan_offset

            return True

        except Exception as e:
            print(f"Warning: Pan operation failed: {e}")
            return False

    def handle_wheel(self, event: QtGui.QWheelEvent) -> bool:
        """
        Handle mouse wheel events for zooming (adjusting camera distance).

        Args:
            event: Qt wheel event (QtGui.QWheelEvent).

        Returns:
            True if the camera distance was updated, False otherwise.
        """
        try:
            # Get wheel delta (typically ±120 per notch)
            delta_zoom = event.angleDelta().y() / RenderingDefaults.MOUSE_WHEEL_DELTA_PER_NOTCH

            # Apply zoom with constraints
            zoom_factor = 1.0 - delta_zoom * self.zoom_sensitivity
            new_distance = self.distance * zoom_factor

            # Clamp to distance bounds
            self.distance = np.clip(new_distance, self.min_distance, self.max_distance)

            return True

        except Exception as e:
            print(f"Warning: Wheel handling failed: {e}")
            return False

    def handle_key_press(self, event: QtGui.QKeyEvent) -> bool:
        """
        Handle keyboard shortcuts for camera control (view presets, FOV, reset).

        Args:
            event: Qt key press event (QtGui.QKeyEvent).

        Returns:
            True if a camera action was triggered and state updated, False otherwise.
        """
        key = event.key()

        # View presets
        if key == Qt.Key.Key_1:
            return self.set_view_preset('front')
        elif key == Qt.Key.Key_3:
            return self.set_view_preset('right')
        elif key == Qt.Key.Key_7:
            return self.set_view_preset('top')
        elif key == Qt.Key.Key_9:
            return self.set_view_preset('iso')
        elif key == Qt.Key.Key_Home:
            self.reset_to_initial()
            return True

        # FOV adjustment
        elif key == Qt.Key.Key_Plus or key == Qt.Key.Key_Equal:
            self.fov_degrees = np.clip(self.fov_degrees - RenderingDefaults.CAMERA_FOV_ADJUSTMENT_STEP,
                                     self.min_fov_degrees, self.max_fov_degrees)
            return True
        elif key == Qt.Key.Key_Minus:
            self.fov_degrees = np.clip(self.fov_degrees + RenderingDefaults.CAMERA_FOV_ADJUSTMENT_STEP,
                                     self.min_fov_degrees, self.max_fov_degrees)
            return True

        return False

    def set_sensitivity(self, rotation: Optional[float] = None,
                       pan: Optional[float] = None,
                       zoom: Optional[float] = None) -> None:
        """
        Set interaction sensitivity values for mouse controls.

        Args:
            rotation: Rotation sensitivity (degrees per pixel).
            pan: Pan sensitivity (world units per pixel, scaled by distance if enabled).
            zoom: Zoom sensitivity (factor per wheel notch).
        """
        if rotation is not None:
            self.rotation_sensitivity = max(0.01, rotation)
        if pan is not None:
            self.pan_sensitivity = max(0.0001, pan)
        if zoom is not None:
            self.zoom_sensitivity = np.clip(zoom, 0.01, 1.0)

    def set_constraints(self, min_distance: Optional[float] = None,
                       max_distance: Optional[float] = None,
                       min_elevation: Optional[float] = None, # degrees
                       max_elevation: Optional[float] = None  # degrees
                       ) -> None:
        """
        Set camera movement constraints.

        Args:
            min_distance: Minimum zoom distance from target.
            max_distance: Maximum zoom distance from target.
            min_elevation: Minimum elevation angle in degrees (e.g., -89.9).
            max_elevation: Maximum elevation angle in degrees (e.g., 89.9).
        """
        if min_distance is not None:
            self.min_distance = max(self.MIN_DISTANCE_CONSTRAINT, min_distance)
        if max_distance is not None:
            self.max_distance = max(self.min_distance + self.MAX_DISTANCE_CONSTRAINT_FACTOR, max_distance)
        if min_elevation is not None:
            self.min_elevation_deg = np.clip(min_elevation, self.MIN_ELEVATION_CONSTRAINT, 0.0)
        if max_elevation is not None:
            self.max_elevation_deg = np.clip(max_elevation, 0.0, self.MAX_ELEVATION_CONSTRAINT)

        # Ensure current values respect constraints
        self.distance = np.clip(self.distance, self.min_distance, self.max_distance)
        self.elevation_deg = np.clip(self.elevation_deg,
                                   self.min_elevation_deg, self.max_elevation_deg)

    def enable_features(self, auto_near_far: Optional[bool] = None,
                       distance_pan_scaling: Optional[bool] = None,
                       smooth_transitions: Optional[bool] = None) -> None:
        """
        Enable or disable advanced camera features.

        Args:
            auto_near_far: If True, automatically adjust near/far clipping planes based on distance.
            distance_pan_scaling: If True, scale pan speed with camera distance to target.
            smooth_transitions: If True, enable smooth camera transitions (not fully implemented).
        """
        if auto_near_far is not None:
            self.auto_adjust_near_far = auto_near_far
        if distance_pan_scaling is not None:
            self.distance_based_pan_scaling = distance_pan_scaling
        if smooth_transitions is not None:
            self.smooth_transitions = smooth_transitions
