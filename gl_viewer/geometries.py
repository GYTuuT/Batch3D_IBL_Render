"""
Geometry management system for the OpenGL viewer.
"""

import time
import uuid
import warnings
import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW, GL_FLOAT,
    glBindBuffer, glBindVertexArray, glBufferData, glBufferSubData,
    glDeleteBuffers, glDeleteVertexArrays
)
from .config import RenderingDefaults


@dataclass
class GeometryObjectAttributes:
    """Base class for geometry object attributes that match shader inputs."""
    vertices: np.ndarray  # Nx3 vertex positions, shape (N, 3)
    colors: Optional[np.ndarray] = None  # Nx3 RGB colors, shape (N, 3)
    normals: Optional[np.ndarray] = None  # Nx3 normal vectors, shape (N, 3)
    texcoords: Optional[np.ndarray] = None  # Nx2 texture coordinates, shape (N, 2)
    instance_data: Optional[np.ndarray] = None  # Nx4 instancing data, shape (N, 4)
    visible: bool = True

    def __post_init__(self):
        """Validate vertex data and other attributes on initialization."""
        if not isinstance(self.vertices, np.ndarray):
            raise ValueError("Vertices must be a NumPy array.")

        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("Vertices must be an Nx3 NumPy array.")

        if self.vertices.shape[0] == 0:
            raise ValueError("Vertices array cannot be empty.")

        num_vertices = self.vertices.shape[0]

        if self.colors is not None:
            if not isinstance(self.colors, np.ndarray):
                raise ValueError("Colors must be a NumPy array if provided.")
            if self.colors.ndim != 2 or self.colors.shape[0] != num_vertices or self.colors.shape[1] != 3:
                raise ValueError(f"Colors must be an {num_vertices}x3 NumPy array.")

        if self.normals is not None:
            if not isinstance(self.normals, np.ndarray):
                raise ValueError("Normals must be a NumPy array if provided.")
            if self.normals.ndim != 2 or self.normals.shape[0] != num_vertices or self.normals.shape[1] != 3:
                raise ValueError(f"Normals must be an {num_vertices}x3 NumPy array.")

        if self.texcoords is not None:
            if not isinstance(self.texcoords, np.ndarray):
                raise ValueError("Texcoords must be a NumPy array if provided.")
            if self.texcoords.ndim != 2 or self.texcoords.shape[0] != num_vertices or self.texcoords.shape[1] != 2:
                raise ValueError(f"Texcoords must be an {num_vertices}x2 NumPy array.")

        if self.instance_data is not None:
            if not isinstance(self.instance_data, np.ndarray):
                raise ValueError("Instance data must be a NumPy array if provided.")
            if self.instance_data.ndim != 2 or self.instance_data.shape[0] != num_vertices or self.instance_data.shape[1] != 4:
                raise ValueError(f"Instance data must be an {num_vertices}x4 NumPy array.")


@dataclass
class PointAttributes(GeometryObjectAttributes):
    """Attributes for point rendering with enhanced shader features."""
    point_sizes: Optional[np.ndarray] = None  # Per-vertex point sizes, shape (N,) or (N, 1)
    shape_types: Optional[np.ndarray] = None  # Shape type indices, shape (N,) or (N, 1), dtype int

    def __post_init__(self):
        """Validate point-specific attributes after base class validation."""
        super().__post_init__()
        num_vertices = self.vertices.shape[0]
        if self.point_sizes is not None:
            if not isinstance(self.point_sizes, np.ndarray):
                raise ValueError("Point sizes must be a NumPy array if provided.")
            if not (self.point_sizes.ndim == 1 and self.point_sizes.shape[0] == num_vertices) and \
               not (self.point_sizes.ndim == 2 and self.point_sizes.shape[0] == num_vertices and self.point_sizes.shape[1] == 1):
                raise ValueError(f"Point sizes must be a 1D array of length {num_vertices} or an {num_vertices}x1 NumPy array.")

        if self.shape_types is not None:
            if not isinstance(self.shape_types, np.ndarray):
                raise ValueError("Shape types must be a NumPy array if provided.")
            if not (self.shape_types.ndim == 1 and self.shape_types.shape[0] == num_vertices) and \
               not (self.shape_types.ndim == 2 and self.shape_types.shape[0] == num_vertices and self.shape_types.shape[1] == 1):
                raise ValueError(f"Shape types must be a 1D array of length {num_vertices} or an {num_vertices}x1 NumPy array.")


@dataclass
class LineAttributes(GeometryObjectAttributes):
    """Attributes for line rendering with enhanced shader features."""
    line_widths: Optional[np.ndarray] = None  # Per-vertex line widths, shape (N,) or (N, 1)

    def __post_init__(self):
        """Validate line-specific attributes after base class validation."""
        super().__post_init__()
        num_vertices = self.vertices.shape[0]
        if self.line_widths is not None:
            if not isinstance(self.line_widths, np.ndarray):
                raise ValueError("Line widths must be a NumPy array if provided.")
            if not (self.line_widths.ndim == 1 and self.line_widths.shape[0] == num_vertices) and \
               not (self.line_widths.ndim == 2 and self.line_widths.shape[0] == num_vertices and self.line_widths.shape[1] == 1):
                raise ValueError(f"Line widths must be a 1D array of length {num_vertices} or an {num_vertices}x1 NumPy array.")


@dataclass
class TriangleAttributes(GeometryObjectAttributes):
    """Attributes for triangle rendering with PBR material support."""
    materials: Optional[np.ndarray] = None  # Material properties [metallic, roughness, ao]. Shape (N, 3) or (3,).
    sss_params: Optional[np.ndarray] = None  # SSS params [strength, distortion, power, scale]. Shape (N, 4) or (4,).
    sss_color: Optional[np.ndarray] = None   # SSS color (RGB). Shape (N, 3) or (3,).

    def __post_init__(self):
        """Validate triangle-specific attributes after base class validation."""
        super().__post_init__()
        num_vertices = self.vertices.shape[0]
        if self.materials is not None:
            if not isinstance(self.materials, np.ndarray):
                raise ValueError("Materials must be a NumPy array if provided.")
            if not (self.materials.ndim == 1 and self.materials.shape[0] == 3) and \
               not (self.materials.ndim == 2 and self.materials.shape[0] == num_vertices and self.materials.shape[1] == 3):
                raise ValueError(f"Materials must be a 1D array of length 3 or an {num_vertices}x3 NumPy array.")

        if self.sss_params is not None:
            if not isinstance(self.sss_params, np.ndarray):
                raise ValueError("SSS params must be a NumPy array if provided.")
            if not (self.sss_params.ndim == 1 and self.sss_params.shape[0] == 4) and \
               not (self.sss_params.ndim == 2 and self.sss_params.shape[0] == num_vertices and self.sss_params.shape[1] == 4):
                raise ValueError(f"SSS params must be a 1D array of length 4 or an {num_vertices}x4 NumPy array.")

        if self.sss_color is not None:
            if not isinstance(self.sss_color, np.ndarray):
                raise ValueError("SSS color must be a NumPy array if provided.")
            if not (self.sss_color.ndim == 1 and self.sss_color.shape[0] == 3) and \
               not (self.sss_color.ndim == 2 and self.sss_color.shape[0] == num_vertices and self.sss_color.shape[1] == 3):
                raise ValueError(f"SSS color must be a 1D array of length 3 or an {num_vertices}x3 NumPy array.")


class BaseDisplayedObject:
    """Base class for managing display objects with change tracking."""

    def __init__(self, object_type: str):
        """
        Initialize display object manager.

        Args:
            object_type: String identifier for the type of objects managed (e.g., "points").
        """
        self.object_type = object_type
        self.objects: Dict[str, GeometryObjectAttributes] = {} # Stores actual attribute objects by key

        # Change tracking for efficient updates
        self.dirty_objects: set[str] = set() # Keys of objects modified or newly added since last GPU update
        self.deleted_objects: set[str] = set() # Keys of objects marked for deletion since last GPU update
        self.is_dirty = False # General flag indicating any change (add, modify, delete) has occurred

        self._last_buffer_size = 0 # Number of objects after the last clean state (GPU buffer update)
        self._key_counter = 0 # Counter for generating unique keys

    def _generate_unique_key(self) -> str:
        """
        Generate a unique key using a counter and UUID.

        Returns:
            A string representing the unique key, e.g., "points_1_a1b2c3d4".
        """
        self._key_counter += 1
        return f"{self.object_type}_{self._key_counter}_{uuid.uuid4().hex[:8]}"

    def _validate_key(self, key: Optional[str]) -> str:
        """
        Validate a provided key or generate a new unique key if None.

        Args:
            key: The key to validate. If None, a new key is generated.

        Returns:
            A validated unique string key.

        Raises:
            ValueError: If the provided key already exists.
        """
        if key is None:
            return self._generate_unique_key()

        if key in self.objects:
            raise ValueError(f"Key '{key}' already exists in {self.object_type}")

        return key

    def add(self, key: Optional[str] = None, **kwargs) -> str:
        """
        Add a new geometry object with specified attributes.

        Args:
            key: Optional unique identifier for the object. If None, one is generated.
            **kwargs: Attributes for the object (e.g., vertices=np.array(...), colors=...).
                      Must include 'vertices'.

        Returns:
            The generated or provided key for the added object.

        Raises:
            ValueError: If key already exists or attribute creation fails.
        """
        validated_key = self._validate_key(key)

        try:
            attributes = self._create_attributes(**kwargs)
        except Exception as e:
            raise ValueError(f"Failed to create {self.object_type} attributes: {e}")

        self.objects[validated_key] = attributes
        self.dirty_objects.add(validated_key)
        self.is_dirty = True

        return validated_key

    def delete(self, key: str) -> bool:
        """
        Delete an object by its key.

        Args:
            key: The unique key of the object to delete.

        Returns:
            True if the object was found and deleted, False otherwise.
        """
        if key not in self.objects:
            return False

        del self.objects[key]
        self.dirty_objects.discard(key)
        self.deleted_objects.add(key)
        self.is_dirty = True

        return True

    def modify(self, key: str, **kwargs) -> bool:
        """
        Modify existing object attributes.
        Validates changes against the attribute class's __post_init__ method.
        If validation passes, attributes are updated and the object is marked dirty.

        Args:
            key: The unique key of the object to modify.
            **kwargs: Attributes to modify (e.g., vertices=new_vertices, colors=new_colors).
                      These should match field names in the corresponding Attributes dataclass.

        Returns:
            True if the object was found and modified successfully, False otherwise.

        Raises:
            ValueError: If an invalid attribute name is provided or attribute validation fails
                        (e.g., incorrect shape, type).
        """
        if key not in self.objects:
            warnings.warn(f"Attempted to modify non-existent object with key '{key}' in {self.object_type}.")
            return False

        current_attrs = self.objects[key]
        # Create a dictionary representing the full state of attributes if modifications were applied
        updated_attrs_dict = {field.name: getattr(current_attrs, field.name) for field in dataclasses.fields(current_attrs)}

        for attr_name, value in kwargs.items():
            if not hasattr(current_attrs, attr_name): # Check if the attribute exists on the current object
                raise ValueError(f"Invalid attribute '{attr_name}' for {self.object_type} object with key '{key}'.")
            updated_attrs_dict[attr_name] = value # Tentatively apply the change to the dictionary

        try:
            # Validate the entire proposed attribute state by creating a temporary instance.
            # This leverages the __post_init__ validation of the dataclass.
            _ = type(current_attrs)(**updated_attrs_dict)

            # If validation passed, apply the changes to the actual object
            if kwargs: # Only mark as dirty if actual changes were made
                for attr_name, value in kwargs.items():
                    setattr(current_attrs, attr_name, value)

                self.dirty_objects.add(key)
                self.is_dirty = True
            return True

        except ValueError as e: # Catch validation errors from __post_init__
            raise ValueError(f"Failed to modify attributes for object '{key}' ({self.object_type}): {e}")

    def get(self, key: str) -> Optional[GeometryObjectAttributes]:
        """
        Get object attributes by key.

        Args:
            key: The unique key of the object.

        Returns:
            The GeometryObjectAttributes instance if found, None otherwise.
        """
        return self.objects.get(key)

    def get_all_keys(self) -> List[str]:
        """
        Get a list of all unique keys for currently managed objects.

        Returns:
            A list of string keys.
        """
        return list(self.objects.keys())

    def clear(self) -> None:
        """Clear all objects, marking them for deletion in the next buffer update."""
        if self.objects:
            self.deleted_objects.update(self.objects.keys())
            self.objects.clear()
            self.dirty_objects.clear()
            self.is_dirty = True

    def mark_clean(self) -> None:
        """Mark all pending changes (dirty, deleted) as processed, typically after GPU update."""
        self.dirty_objects.clear()
        self.deleted_objects.clear()
        self.is_dirty = False
        self._last_buffer_size = len(self.objects)

    def has_changes(self) -> bool:
        """
        Check if there are pending changes (dirty or deleted objects, or general dirty flag)
        requiring GPU buffer update.

        Returns:
            True if changes are pending, False otherwise.
        """
        return (self.is_dirty or
                len(self.dirty_objects) > 0 or
                len(self.deleted_objects) > 0)

    def get_statistics(self) -> Dict[str, int]:
        """
        Get statistics about the managed objects for debugging or monitoring.

        Returns:
            A dictionary containing counts of total, dirty, and deleted objects,
            and the size of the buffer at the last clean state.
        """
        return {
            'total_objects': len(self.objects),
            'dirty_objects': len(self.dirty_objects),
            'deleted_objects': len(self.deleted_objects),
            'last_buffer_size': self._last_buffer_size
        }

    def _create_attributes(self, vertices: np.ndarray, **kwargs) -> GeometryObjectAttributes: # type: ignore[empty-body]
        """
        Create an appropriate attributes object for the specific display type.
        Must be overridden by subclasses.

        Args:
            vertices: Nx3 NumPy array of vertex positions.
            **kwargs: Additional attributes specific to the geometry type.

        Returns:
            An instance of a GeometryObjectAttributes subclass.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement _create_attributes")


class DisplayedPoints(BaseDisplayedObject):
    """Manages point objects with shader-compatible attributes."""

    def __init__(self):
        super().__init__("points")

    def _create_attributes(self, vertices: np.ndarray, **kwargs) -> PointAttributes:
        """
        Create PointAttributes instance with validation.

        Args:
            vertices: Nx3 NumPy array of point positions.
            **kwargs: Additional attributes for PointAttributes (e.g., colors, point_sizes).

        Returns:
            A PointAttributes instance.
        """
        return PointAttributes(vertices=vertices, **kwargs)


class DisplayedLines(BaseDisplayedObject):
    """Manages line objects with shader-compatible attributes."""

    def __init__(self):
        super().__init__("lines")

    def _create_attributes(self, vertices: np.ndarray, **kwargs) -> LineAttributes:
        """
        Create LineAttributes instance with validation.

        Args:
            vertices: Nx3 NumPy array of line vertices.
            **kwargs: Additional attributes for LineAttributes (e.g., colors, line_widths).

        Returns:
            A LineAttributes instance.
        """
        return LineAttributes(vertices=vertices, **kwargs)


class DisplayedTriangles(BaseDisplayedObject):
    """Manages triangle objects with shader-compatible attributes."""

    def __init__(self):
        super().__init__("triangles")

    def _create_attributes(self, vertices: np.ndarray, **kwargs) -> TriangleAttributes:
        """
        Create TriangleAttributes instance with validation.

        Args:
            vertices: Nx3 NumPy array of triangle vertices.
            **kwargs: Additional attributes for TriangleAttributes (e.g., colors, normals, materials).

        Returns:
            A TriangleAttributes instance.
        """
        return TriangleAttributes(vertices=vertices, **kwargs)


class GeometryBuffer:
    """Manages OpenGL vertex buffer objects with efficient incremental updates using free block list and defragmentation."""

    DEFRAGMENT_FREE_BLOCK_COUNT_THRESHOLD: int = 10
    DEFRAGMENT_WASTED_SPACE_RATIO_THRESHOLD: float = 0.25


    def __init__(self, stride: int):
        """
        Initialize geometry buffer.

        Args:
            stride: Number of float elements per vertex (e.g., 3 for position, 3 for color = 6).
        """
        self.stride = stride  # Number of float elements per vertex
        self.data: Optional[np.ndarray] = None  # Raw float data buffer (potentially larger than used, may have gaps)
        self.current_elements_count = 0  # Total number of *valid* float elements for active objects
        self.cpu_capacity_elements = 0   # Total allocated float elements in self.data (actual length of np.array)
        self.buffer_high_water_mark_elements = 0 # Marks the end of the last used block in self.data

        # OpenGL resources
        self.vao: Optional[int] = None
        self.vbo: Optional[int] = None

        # Change tracking
        self.dirty = False
        self.object_ranges: Dict[str, Tuple[int, int]] = {}  # key -> (start_vertex, vertex_count)

        # Free block management
        self.free_blocks: List[Tuple[int, int]] = [] # List of (start_element_idx, element_count) for free blocks, sorted

        # Buffer management
        self.gpu_buffer_capacity_bytes = 0 # Current capacity of the GPU VBO in bytes
        self.growth_factor = RenderingDefaults.BUFFER_GROWTH_FACTOR

    def _ensure_cpu_capacity(self, required_high_water_mark_elements: int) -> None:
        """
        Ensures the CPU-side self.data buffer has at least required_high_water_mark_elements capacity.
        Grows the buffer if necessary.

        Args:
            required_high_water_mark_elements: The minimum number of float elements the buffer must hold up to its high water mark.
        """
        if required_high_water_mark_elements > self.cpu_capacity_elements:
            min_absolute_elements = self.stride * RenderingDefaults.MIN_BUFFER_OBJECT_CAPACITY
            new_capacity = int(max(required_high_water_mark_elements,
                                 self.cpu_capacity_elements * self.growth_factor,
                                 min_absolute_elements))
            new_data = np.empty(new_capacity, dtype=np.float32)
            if self.data is not None and self.buffer_high_water_mark_elements > 0: # Copy up to high water mark
                new_data[:self.buffer_high_water_mark_elements] = self.data[:self.buffer_high_water_mark_elements]
            self.data = new_data
            self.cpu_capacity_elements = new_capacity

    def _add_free_block(self, start_element: int, element_count: int) -> None:
        """Adds a block to the free list and merges adjacent/overlapping blocks."""
        if element_count <= 0:
            return

        self.free_blocks.append((start_element, element_count))
        self.free_blocks.sort(key=lambda x: x[0]) # Keep sorted by start index

        merged_blocks: List[Tuple[int, int]] = []
        if not self.free_blocks:
            return # Should not happen if we just appended

        current_block = self.free_blocks[0]
        for i in range(1, len(self.free_blocks)):
            next_block = self.free_blocks[i]
            # If current_block and next_block are adjacent or overlapping
            if current_block[0] + current_block[1] >= next_block[0]:
                # Merge them
                merged_end = max(current_block[0] + current_block[1], next_block[0] + next_block[1])
                current_block = (current_block[0], merged_end - current_block[0])
            else:
                # No overlap, current_block is final for now
                merged_blocks.append(current_block)
                current_block = next_block

        merged_blocks.append(current_block) # Add the last processed block
        self.free_blocks = merged_blocks

        self._try_truncate_buffer_high_water_mark()

    def _try_truncate_buffer_high_water_mark(self) -> None:
        """If the last block(s) in the buffer are free, reduce the high water mark."""
        while self.free_blocks:
            last_free_block_start, last_free_block_count = self.free_blocks[-1]
            if last_free_block_start + last_free_block_count == self.buffer_high_water_mark_elements:
                self.buffer_high_water_mark_elements = last_free_block_start
                self.free_blocks.pop()
            else:
                break

        if self.buffer_high_water_mark_elements == 0 and not self.object_ranges: # Buffer is completely empty
            self.data = None # Allow GC to collect if configured
            self.cpu_capacity_elements = 0 # Reflect that data array might be gone or zero-sized
            self.current_elements_count = 0 # Ensure consistency

    def _find_free_block(self, required_elements: int) -> Optional[Tuple[int, int, int]]: # (original_idx, start_elements, elements_in_block)
        """Finds a suitable free block (best fit). Returns its original index, start, and size."""
        best_fit_original_idx = -1
        best_fit_size_diff = float('inf')
        found_block_data: Optional[Tuple[int, int]] = None

        for i, (start, count) in enumerate(self.free_blocks):
            if count >= required_elements:
                size_diff = count - required_elements
                if size_diff < best_fit_size_diff: # Prioritize smaller suitable blocks (best fit)
                    best_fit_size_diff = size_diff
                    best_fit_original_idx = i
                    found_block_data = (start, count)
                if size_diff == 0: # Perfect fit found
                    break

        if best_fit_original_idx != -1 and found_block_data is not None:
            self.free_blocks.pop(best_fit_original_idx) # Remove the block from the list
            return best_fit_original_idx, found_block_data[0], found_block_data[1]
        return None

    def needs_defragmentation(self) -> bool:
        """Checks if the buffer is fragmented enough to warrant defragmentation."""
        if not self.free_blocks:
            return False

        num_free_blocks = len(self.free_blocks)
        if num_free_blocks == 0: return False

        total_free_elements = sum(count for _, count in self.free_blocks)

        if self.buffer_high_water_mark_elements == 0:
             return False # Empty buffer or fully truncated, no defrag needed

        wasted_space_ratio = total_free_elements / self.buffer_high_water_mark_elements

        return (num_free_blocks > self.DEFRAGMENT_FREE_BLOCK_COUNT_THRESHOLD or
                wasted_space_ratio > self.DEFRAGMENT_WASTED_SPACE_RATIO_THRESHOLD)

    def defragment(self) -> None:
        """Compacts the buffer by removing free blocks and shifting data."""
        if not self.free_blocks: # No fragmentation to fix
            return

        if self.data is None or self.current_elements_count == 0: # Nothing to defragment
            self.free_blocks.clear()
            self.buffer_high_water_mark_elements = 0
            self.current_elements_count = 0
            if self.data is not None and self.cpu_capacity_elements > 0 : # if data exists but current_elements is 0
                 self.data = np.empty(0, dtype=np.float32) # Make it an empty array
                 self.cpu_capacity_elements = 0
            self.dirty = True # Mark dirty as state changed (became empty)
            return

        new_data = np.empty(self.current_elements_count, dtype=np.float32)
        current_new_idx_elements = 0

        # Sort objects by their current start position to process them in order
        # This ensures data is copied contiguously into new_data
        sorted_object_keys = sorted(self.object_ranges.keys(), key=lambda k: self.object_ranges[k][0])

        new_object_ranges: Dict[str, Tuple[int, int]] = {}

        for key in sorted_object_keys:
            start_vertex, vertex_count = self.object_ranges[key]
            start_elements = start_vertex * self.stride
            elements_count = vertex_count * self.stride

            if self.data is not None: # Should be true
                 new_data[current_new_idx_elements : current_new_idx_elements + elements_count] = \
                    self.data[start_elements : start_elements + elements_count]

            new_start_vertex = current_new_idx_elements // self.stride
            new_object_ranges[key] = (new_start_vertex, vertex_count)
            current_new_idx_elements += elements_count

        self.data = new_data
        self.object_ranges = new_object_ranges
        self.cpu_capacity_elements = self.current_elements_count # Buffer is now perfectly sized
        self.buffer_high_water_mark_elements = self.current_elements_count
        self.free_blocks.clear()
        self.dirty = True
        # print(f"Defragmentation complete. New HWM: {self.buffer_high_water_mark_elements}")

    def _add_new_object_data(self, key: str, new_data_flat: np.ndarray, vertex_count: int) -> None:
        """Internal: Adds a new object, trying to use free blocks first, then appending."""
        new_elements_count = vertex_count * self.stride

        if self.data is None: # First object, or buffer was completely cleared and data became None
            self._ensure_cpu_capacity(new_elements_count) # This will create self.data

        # Try to find a free block
        found_block_info = self._find_free_block(new_elements_count)

        if found_block_info:
            _, block_start_elements, block_total_elements = found_block_info
            if self.data is not None:
                self.data[block_start_elements : block_start_elements + new_elements_count] = new_data_flat

            self.object_ranges[key] = (block_start_elements // self.stride, vertex_count)

            remaining_elements_in_block = block_total_elements - new_elements_count
            if remaining_elements_in_block > 0:
                self._add_free_block(block_start_elements + new_elements_count, remaining_elements_in_block)
        else: # No suitable free block found
            if self.needs_defragmentation():
                # print(f"Defragmenting (needed) before adding new object {key}...")
                self.defragment()
                # Try finding a block again after defragmentation
                found_block_info_after_defrag = self._find_free_block(new_elements_count)
                if found_block_info_after_defrag:
                    _, block_start_elements, block_total_elements = found_block_info_after_defrag
                    if self.data is not None: # data would have been re-created by defrag
                        self.data[block_start_elements : block_start_elements + new_elements_count] = new_data_flat
                    self.object_ranges[key] = (block_start_elements // self.stride, vertex_count)
                    remaining_elements_in_block = block_total_elements - new_elements_count
                    if remaining_elements_in_block > 0:
                        self._add_free_block(block_start_elements + new_elements_count, remaining_elements_in_block)
                    # current_elements_count updated by caller (update_object_data)
                    self.dirty = True
                    return # Successfully added after defrag

            # Append to the end of the buffer
            append_start_elements = self.buffer_high_water_mark_elements
            self._ensure_cpu_capacity(append_start_elements + new_elements_count)
            if self.data is not None: # Should be true after _ensure_cpu_capacity
                self.data[append_start_elements : append_start_elements + new_elements_count] = new_data_flat

            self.object_ranges[key] = (append_start_elements // self.stride, vertex_count)
            self.buffer_high_water_mark_elements = append_start_elements + new_elements_count

        # current_elements_count is updated by the caller (update_object_data)
        self.dirty = True

    def _replace_object_data(self, key: str, new_data_flat: np.ndarray, new_vertex_count: int) -> None:
        """Internal: Replaces an existing object's data. Assumes current_elements_count already reflects the net change."""
        old_start_vertex, old_vertex_count = self.object_ranges[key]
        old_start_elements = old_start_vertex * self.stride
        old_elements_count = old_vertex_count * self.stride

        new_elements_count = new_vertex_count * self.stride

        if new_elements_count <= old_elements_count:
            # New data fits in or is smaller than the old block
            if self.data is not None:
                self.data[old_start_elements : old_start_elements + new_elements_count] = new_data_flat
            self.object_ranges[key] = (old_start_vertex, new_vertex_count) # Start vertex remains the same

            remaining_elements_in_old_block = old_elements_count - new_elements_count
            if remaining_elements_in_old_block > 0:
                self._add_free_block(old_start_elements + new_elements_count, remaining_elements_in_old_block)
        else:
            # New data is larger. Release old block and find new space.
            self._add_free_block(old_start_elements, old_elements_count)

            # Find new space for the larger data (similar to _add_new_object_data's logic)
            found_block_info = self._find_free_block(new_elements_count)
            if found_block_info:
                _, block_start_elements, block_total_elements = found_block_info
                if self.data is not None:
                    self.data[block_start_elements : block_start_elements + new_elements_count] = new_data_flat
                self.object_ranges[key] = (block_start_elements // self.stride, new_vertex_count)
                remaining_elements_in_block = block_total_elements - new_elements_count
                if remaining_elements_in_block > 0:
                    self._add_free_block(block_start_elements + new_elements_count, remaining_elements_in_block)
            else:
                if self.needs_defragmentation():
                    # print(f"Defragmenting (needed) before replacing object {key} with larger data...")
                    self.defragment()
                    found_block_info_after_defrag = self._find_free_block(new_elements_count)
                    if found_block_info_after_defrag:
                        _, block_start_elements, block_total_elements = found_block_info_after_defrag
                        if self.data is not None:
                             self.data[block_start_elements : block_start_elements + new_elements_count] = new_data_flat
                        self.object_ranges[key] = (block_start_elements // self.stride, new_vertex_count)
                        remaining_elements_in_block = block_total_elements - new_elements_count
                        if remaining_elements_in_block > 0:
                            self._add_free_block(block_start_elements + new_elements_count, remaining_elements_in_block)
                        self.dirty = True
                        return # Successfully replaced after defrag

                # Append to the end of the buffer
                append_start_elements = self.buffer_high_water_mark_elements
                self._ensure_cpu_capacity(append_start_elements + new_elements_count)
                if self.data is not None:
                    self.data[append_start_elements : append_start_elements + new_elements_count] = new_data_flat
                self.object_ranges[key] = (append_start_elements // self.stride, new_vertex_count)
                self.buffer_high_water_mark_elements = append_start_elements + new_elements_count

        self.dirty = True

    def update_object_data(self, key: str, new_data_flat: np.ndarray, vertex_count: int) -> None:
        """
        Update or add vertex data for a specific object.
        The data is stored in a CPU-side buffer and marked as dirty for GPU upload.

        Args:
            key: Unique identifier for the object.
            new_data_flat: 1D NumPy array (float32) of interleaved vertex attributes.
                           Its size must be vertex_count * self.stride.
            vertex_count: Number of vertices in new_data_flat.
        """
        if new_data_flat.dtype != np.float32:
            new_data_flat = new_data_flat.astype(np.float32)

        new_elements_count = vertex_count * self.stride
        if new_data_flat.size != new_elements_count:
            raise ValueError(f"Data element count mismatch for key '{key}': expected {new_elements_count}, got {new_data_flat.size}")

        if key in self.object_ranges:
            old_vertex_count = self.object_ranges[key][1]
            old_elements_count = old_vertex_count * self.stride
            self.current_elements_count = self.current_elements_count - old_elements_count + new_elements_count
            self._replace_object_data(key, new_data_flat, vertex_count)
        else:
            self.current_elements_count += new_elements_count # Add count for new object
            self._add_new_object_data(key, new_data_flat, vertex_count)

        # self.dirty is set by _replace_object_data or _add_new_object_data

    def remove_object_data(self, key: str) -> None:
        """
        Remove object data from the CPU buffer by marking its space as free.
        Marks the buffer as dirty.

        Args:
            key: Unique identifier of the object to remove.
        """
        if key not in self.object_ranges:
            return

        start_vertex, count_vertex = self.object_ranges.pop(key)
        start_elements = start_vertex * self.stride
        elements_to_remove = count_vertex * self.stride

        self._add_free_block(start_elements, elements_to_remove) # This also calls _try_truncate_buffer_high_water_mark
        self.current_elements_count -= elements_to_remove

        if self.current_elements_count == 0 and not self.object_ranges: # Buffer is completely empty
            # _try_truncate_buffer_high_water_mark should have handled setting data to None if HWM became 0
            # and free_blocks became empty.
            # Explicitly ensure consistency here.
            self.data = None
            self.cpu_capacity_elements = 0
            self.buffer_high_water_mark_elements = 0
            self.free_blocks.clear() # Should be empty if HWM is 0 after truncation

        self.dirty = True

    def clear(self) -> None:
        """Clear all data from CPU and GPU buffers, and delete OpenGL resources (VAO, VBO)."""
        if self.vbo is not None:
            try:
                glDeleteBuffers(1, [self.vbo])
            except Exception as e: # Catch error if context is already gone
                print(f"Warning: Error deleting VBO: {e}")
            self.vbo = None
        if self.vao is not None:
            try:
                glDeleteVertexArrays(1, [self.vao])
            except Exception as e:
                print(f"Warning: Error deleting VAO: {e}")
            self.vao = None

        self.data = None
        self.object_ranges.clear()
        self.free_blocks.clear()
        self.gpu_buffer_capacity_bytes = 0
        self.current_elements_count = 0
        self.cpu_capacity_elements = 0
        self.buffer_high_water_mark_elements = 0
        self.dirty = True

    def update_gpu_buffer(self) -> None:
        """
        Upload CPU buffer data (up to high water mark) to the GPU VBO if dirty.
        Manages VBO capacity using a growth factor to minimize reallocations.
        """
        if not self.dirty or self.vao is None or self.vbo is None:
            return

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        data_ptr: Optional[np.ndarray] = None
        data_bytes: int = 0

        if self.buffer_high_water_mark_elements > 0 and self.data is not None:
            # Upload data up to the high water mark. Shaders will use object_ranges for correct access.
            data_to_upload = self.data[:self.buffer_high_water_mark_elements]
            data_bytes = data_to_upload.nbytes
            data_ptr = data_to_upload
        else: # Buffer is logically empty or self.data is None
            data_bytes = 0
            data_ptr = None


        if data_bytes == 0: # Handles case where buffer becomes empty
            glBufferData(GL_ARRAY_BUFFER, 0, None, GL_DYNAMIC_DRAW)
            self.gpu_buffer_capacity_bytes = 0
        else:
            if data_bytes > self.gpu_buffer_capacity_bytes:
                # Grow GPU buffer
                new_gpu_capacity = int(data_bytes * self.growth_factor)
                if new_gpu_capacity < data_bytes : new_gpu_capacity = data_bytes
                glBufferData(GL_ARRAY_BUFFER, new_gpu_capacity, data_ptr, GL_DYNAMIC_DRAW)
                self.gpu_buffer_capacity_bytes = new_gpu_capacity
            else:
                # Update existing GPU buffer
                glBufferSubData(GL_ARRAY_BUFFER, 0, data_bytes, data_ptr)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        self.dirty = False

    def get_vertex_count(self) -> int:
        """
        Get the total number of actual vertices currently stored in active objects.
        """
        return self.current_elements_count // self.stride if self.stride > 0 else 0

    def has_fragmentation(self) -> bool:
        """Checks if there are any free blocks, indicating fragmentation."""
        return bool(self.free_blocks)

    def get_object_count(self) -> int:
        """
        Get the number of distinct objects managed by this buffer.

        Returns:
            The count of objects.
        """
        return len(self.object_ranges)

    def get_memory_usage(self) -> Dict[str, int]:
        """
        Get memory usage statistics for debugging and monitoring.

        Returns:
            A dictionary containing CPU allocated/used bytes, GPU capacity/used bytes,
            vertex count, object count, and CPU element counts.
        """
        cpu_allocated_bytes = self.cpu_capacity_elements * RenderingDefaults.FLOAT_SIZE_BYTES
        cpu_used_bytes = self.current_elements_count * RenderingDefaults.FLOAT_SIZE_BYTES # Actual data elements
        gpu_used_bytes = self.buffer_high_water_mark_elements * RenderingDefaults.FLOAT_SIZE_BYTES # Data uploaded to GPU

        total_free_elements = sum(count for _, count in self.free_blocks)
        free_bytes_in_fragmentation = total_free_elements * RenderingDefaults.FLOAT_SIZE_BYTES

        return {
            'cpu_allocated_bytes': cpu_allocated_bytes,
            'cpu_used_bytes_active_objects': cpu_used_bytes,
            'cpu_high_water_mark_bytes': self.buffer_high_water_mark_elements * RenderingDefaults.FLOAT_SIZE_BYTES,
            'cpu_free_bytes_in_fragmentation': free_bytes_in_fragmentation,
            'gpu_capacity_bytes': self.gpu_buffer_capacity_bytes,
            'gpu_used_bytes_uploaded': gpu_used_bytes,
            'vertex_count_active': self.get_vertex_count(),
            'object_count': self.get_object_count(),
            'cpu_current_elements_active': self.current_elements_count,
            'cpu_capacity_elements_allocated': self.cpu_capacity_elements,
            'cpu_high_water_mark_elements': self.buffer_high_water_mark_elements,
            'free_block_count': len(self.free_blocks)
        }
