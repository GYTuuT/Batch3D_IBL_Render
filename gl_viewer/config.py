"""
Configuration for default rendering parameters and constants.
"""

import numpy as np
from typing import Tuple, List

class RenderingDefaults:
    """
    Stores default values for various rendering parameters used throughout the viewer.
    These defaults are applied when specific attributes are not provided by the user
    for geometry objects or rendering settings.
    """
    # Default colors (RGB, float32, range [0,1])
    DEFAULT_POINT_COLOR: np.ndarray = np.array([0.8, 0.8, 0.8], dtype=np.float32)  # Light gray for points
    DEFAULT_LINE_COLOR: np.ndarray = np.array([0.7, 0.7, 0.7], dtype=np.float32)   # Medium gray for lines
    DEFAULT_TRIANGLE_COLOR: np.ndarray = np.array([0.6, 0.6, 0.6], dtype=np.float32) # Dark gray for triangles

    # Default geometry attributes
    DEFAULT_POINT_SIZE: float = 5.0  # Default size for points in pixels
    DEFAULT_LINE_WIDTH: float = 1.0  # Default width for lines in pixels
    DEFAULT_SHAPE_TYPE: int = 0      # Default shape type for points (0 typically means circle)
    DEFAULT_NORMAL: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=np.float32) # Default normal vector (pointing along Z-axis)
    DEFAULT_TRIANGLE_NORMAL: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=np.float32) # Default normal for triangles
    DEFAULT_TEXCOORD: np.ndarray = np.array([0.0, 0.0], dtype=np.float32) # Default texture coordinate (bottom-left)

    # Default material properties for PBR
    DEFAULT_METALLIC: float = 0.1    # Default metallic value (mostly non-metallic)
    DEFAULT_ROUGHNESS: float = 0.8   # Default roughness value (quite rough)
    DEFAULT_AO: float = 1.0          # Default ambient occlusion value (no occlusion)

    # Default SSS (Subsurface Scattering) parameters
    # [strength, distortion, power, scale]
    DEFAULT_SSS_PARAMS: np.ndarray = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32) # Default: SSS disabled
    DEFAULT_SSS_COLOR: np.ndarray = np.array([1.0, 1.0, 1.0], dtype=np.float32)      # Default SSS color (white)

    # Default instance data (can be used for per-instance transformations or properties)
    # Example: [translateX, translateY, translateZ, scale] or [R, G, B, Alpha_multiplier]
    DEFAULT_INSTANCE_DATA: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    # Default lighting parameters
    DEFAULT_LIGHT_POSITION: np.ndarray = np.array([5.0, 10.0, 5.0], dtype=np.float32) # Default light position in world space
    DEFAULT_LIGHT_COLOR: np.ndarray = np.array([1.0, 1.0, 1.0], dtype=np.float32)    # Default light color (white)
    DEFAULT_AMBIENT_STRENGTH: float = 0.2  # Default ambient light intensity
    DEFAULT_SPECULAR_STRENGTH: float = 0.5 # Default specular reflection intensity
    DEFAULT_SHININESS: float = 32.0        # Default specular shininess exponent

    # Default fog parameters
    DEFAULT_FOG_COLOR: np.ndarray = np.array([0.5, 0.5, 0.5], dtype=np.float32) # Default fog color (gray)
    DEFAULT_FOG_NEAR: float = 10.0         # Default fog start distance
    DEFAULT_FOG_FAR: float = 100.0         # Default fog end distance

    # Default wireframe parameters
    DEFAULT_WIREFRAME_COLOR: np.ndarray = np.array([1.0, 1.0, 1.0], dtype=np.float32) # Default wireframe color (black)

    # Default distance fade parameters
    DEFAULT_FADE_DISTANCE: float = 50.0    # Default distance for 50% alpha fade

    # IBL (Image-Based Lighting) defaults
    DEFAULT_IBL_AMBIENT_INTENSITY: float = 1.0
    DEFAULT_IBL_SPECULAR_INTENSITY: float = 1.0
    DEFAULT_IBL_BASE_ROUGHNESS: float = 0.0 # Additional base roughness for IBL materials
    DEFAULT_IBL_CUBEMAP_SIZE: int = 512     # Default resolution for generated environment cubemap faces
    DEFAULT_IBL_BRDF_LUT_SIZE: int = 512    # Default resolution for the BRDF lookup texture
    DEFAULT_IBL_BRDF_SAMPLE_COUNT: int = 1024 # Default number of samples for BRDF integration in LUT generation

    # Shader and rendering constants
    MSAA_SAMPLES: int = 4                   # Default MSAA samples for anti-aliasing
    OPENGL_ATTRIBUTE_STRIDE_POINTS_LINES: int = 18  # Stride for points/lines vertex data
    OPENGL_ATTRIBUTE_STRIDE_TRIANGLES: int = 28     # Stride for triangles vertex data

    # Geometry buffer constants
    BUFFER_GROWTH_FACTOR: float = 1.5       # Growth factor for dynamic buffers
    MIN_BUFFER_OBJECT_CAPACITY: int = 10  # Minimum buffer capacity in terms of storable objects, affects initial allocation and growth floor

    # Camera interaction constants
    CAMERA_ROTATION_SENSITIVITY: float = 0.2   # Default rotation sensitivity
    CAMERA_PAN_SENSITIVITY: float = 0.005      # Default pan sensitivity
    CAMERA_ZOOM_SENSITIVITY: float = 0.1       # Default zoom sensitivity
    CAMERA_FOV_ADJUSTMENT_STEP: float = 5.0    # FOV adjustment step in degrees

    # View preset angles (azimuth, elevation) in degrees
    VIEW_PRESET_FRONT: tuple = (0.0, 0.0)
    VIEW_PRESET_BACK: tuple = (180.0, 0.0)
    VIEW_PRESET_LEFT: tuple = (-90.0, 0.0)
    VIEW_PRESET_RIGHT: tuple = (90.0, 0.0)
    VIEW_PRESET_TOP: tuple = (0.0, 90.0)
    VIEW_PRESET_BOTTOM: tuple = (0.0, -90.0)
    VIEW_PRESET_ISO: tuple = (45.0, 35.0)
    VIEW_PRESET_ISO2: tuple = (-45.0, 35.0)

    # IBL cubemap generation constants
    IBL_CUBEMAP_FOV_DEGREES: float = 90.0       # FOV for cubemap face generation
    IBL_CUBEMAP_ASPECT_RATIO: float = 1.0       # Aspect ratio for cubemap faces
    IBL_CUBEMAP_NEAR_PLANE: float = 0.1         # Near plane for cubemap generation
    IBL_CUBEMAP_FAR_PLANE: float = 10.0         # Far plane for cubemap generation
    IBL_CUBEMAP_FACES_COUNT: int = 6            # Number of cubemap faces
    IBL_TRIANGLE_STRIP_VERTEX_COUNT: int = 4    # Vertices for fullscreen quad
    IBL_CUBE_VERTEX_COUNT: int = 36             # Vertices for cube geometry

    # Texture binding units
    TEXTURE_UNIT_DIFFUSE: int = 1               # Diffuse texture unit
    TEXTURE_UNIT_SHAPE: int = 2                 # Shape texture unit
    TEXTURE_UNIT_ENVIRONMENT: int = 3            # Environment map texture unit
    TEXTURE_UNIT_BRDF_LUT: int = 4              # BRDF LUT texture unit (changed from 1 to 3)
    TEXTURE_UNIT_IRRADIANCE_MAP: int = 5
    TEXTURE_UNIT_PREFILTER_MAP: int = 6

    # Mouse wheel constants
    MOUSE_WHEEL_DELTA_PER_NOTCH: float = 120.0  # Standard wheel delta per notch

    # Buffer management constants
    FLOAT_SIZE_BYTES: int = 4                    # Size of float32 in bytes

    # Add this variable for offscreen rendering precision
    OFFSCREEN_RENDER_FLOAT_RGBA = False  # If True, use float32 RGBA, else 8bit RGBA
