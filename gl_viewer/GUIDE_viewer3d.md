# gl_viewer 用户手册

## 1. 概览与整体结构

`gl_viewer` 是基于 PyOpenGL 和 PySide6 构建的模块化 3D 几何物体查看器，提供渲染、交互、着色器管理、相机控制等功能。主要目录及功能：

### config.py
- Class `RenderingDefaults`:
  - 全局常量，用于默认渲染参数。
  - 属性示例：
    - 颜色：`DEFAULT_POINT_COLOR`, `DEFAULT_LINE_COLOR`, `DEFAULT_TRIANGLE_COLOR`
    - 几何属性：`DEFAULT_POINT_SIZE`, `DEFAULT_LINE_WIDTH`, `DEFAULT_SHAPE_TYPE`, `DEFAULT_NORMAL`, `DEFAULT_TEXCOORD`
    - PBR 材质：`DEFAULT_METALLIC`, `DEFAULT_ROUGHNESS`, `DEFAULT_AO`
    - SSS：`DEFAULT_SSS_PARAMS` (strength, distortion, power, scale), `DEFAULT_SSS_COLOR`
    - IBL、雾、线框、相机灵敏度、缓冲区常量等 (>50 个常量)

### cameras.py
- Class `CameraController`:
  - __init__(initial_target, initial_distance, initial_azimuth_deg, initial_elevation_deg, initial_fov_degrees, initial_near_plane, initial_far_plane, world_up)
  - 私有方法：
    - `_calculate_eye_position()`：计算相机 `eye` 位置
    - `_auto_adjust_near_far_planes()`：根据距离自动调整裁剪面
  - 公共方法：
    - `update_matrices(aspect_ratio)`：更新 `view_matrix` 和 `projection_matrix`
    - `reset_to_initial()`：重置到初始状态
    - `set_view_preset(preset: str) -> bool`：应用预设视角
    - `frame_bounds(min_bounds, max_bounds, padding_factor=None)`：根据包围盒设置目标和距离
    - `get_camera_info() -> dict`：返回当前相机参数
    - 事件处理：
      - `handle_mouse_press(event)`, `handle_mouse_release(event)`, `handle_mouse_move(event)`
      - `handle_wheel(event)`：缩放

### geometries.py
- Class `GeometryObjectAttributes`: 数据容器 (vertices, colors, normals, texcoords, point_sizes, line_widths, shape_types, instance_data, materials, sss_params, sss_color)
- Class `GeometryBuffer`:
  - `__init__(stride: int)`
  - `add_object_data(key: str, attrs: GeometryObjectAttributes)`
  - `remove_object_data(key: str)`
  - `update_gpu_buffer()`：上传到 VBO
- Classes `DisplayedPoints`, `DisplayedLines`, `DisplayedTriangles`:
  - `add(key, vertices, **attrs) -> str`
  - `modify(key, **attrs) -> bool`
  - `remove(key)`, `list_keys() -> List[str]`
  - `mark_render_dirty()`, `mark_clean()`

### shader_utils.py
- `get_shader_program(vertex_path: str, fragment_path: str) -> int`：编译并缓存 GLSL 程序
- `clear_shader_cache() -> None`：清除已编译着色器

### controller.py
- Class `BaseShaderController`:
  - 控制基本 PBR+SSS 着色器参数
  - 方法：`set_sss_enabled(enabled: bool)`, `bind_uniforms(program: int)`, `set_material(metallic, roughness, ao)`, 等
- Class `IBLRenderer`:
  - __init__(exr_filepath: Optional[str])
  - `load_hdr_texture()`：加载 HDR 图像
  - `generate_cubemaps()`：生成环境贴图、漫反射立方体贴图和预滤波贴图
  - `bind_ibl(program: int)`: 绑定 IBL 相关采样器

### renderer.py
- Class `SceneRenderer`:
  - __init__(camera: CameraController, ibl_controller: IBLRenderer, base_shader_controller: BaseShaderController, points_buffer: GeometryBuffer, lines_buffer: GeometryBuffer, triangles_buffer: GeometryBuffer, draw_objects_individually: bool)
  - `render()`：绘制所有几何对象并处理后期效果
  - 私有方法：`_render_points()`, `_render_lines()`, `_render_triangles()`, `_render_skybox()`
  - `render_to_textures(width: int, height: int, render_skybox: bool=False) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]`
- 独立函数：
  - `render_scene_to_images(widget, renderer, width, height, render_skybox=False) -> (rgba, normals, depth)`

### viewer3d.py
- Class `Viewer3DWidget(QOpenGLWidget)`:
  - __init__(parent: QWidget=None, exr_filepath: str=None, draw_objects_individually: bool=False)
  - `check_gl_error(tag: str) -> bool`
  - 添加/修改/查询/移除：
    - `add_points(vertices, key=None, **kwargs) -> str`
    - `add_lines(vertices, key=None, **kwargs) -> str`
    - `add_triangles(vertices, key=None, **kwargs) -> str`
    - `modify_points(key, **kwargs) -> bool`, `modify_lines(...)`, `modify_triangles(...)`
    - `list_points_keys()`, `list_lines_keys()`, `list_triangles_keys()`
  - 场景设置：`set_lighting(...)`, `set_fog(...)`, `set_wireframe(...)`
  - 相机访问：`camera` 属性，可直接调用其方法
  - 内部：`_setup_geometry_vao()`, `_convert_attributes_to_buffer_data()`, `_update_geometry_buffers()` 等
- 无独立函数

### shaders 目录
- base.vert / base.frag：核心 PBR 顶点/片段着色器，实现 Blinn-Phong 光照、法线、漫反射、高光及基础材质
- sss.vert / sss.frag：次表面散射着色器，多次采样模拟散射效果
- skybox.vert / skybox.frag：天空盒渲染，将立方体贴图渲染为背景
- equirect_to_cubemap.vert / .frag：将 2D HDR 环境贴图转换为立方体贴图
- irradiance_convolution.vert / .frag：计算漫反射环境贴图（irradiance map）
- prefilter_env_map.vert / .frag：预滤波环境贴图，适配不同粗糙度采样
- brdf.vert / brdf.frag：渲染 BRDF LUT，用于 PBR 运行时近似
- ibl_triangle.vert / .frag：辅助渲染三角形时应用 IBL 采样

## 2. Viewer3DWidget 使用指南

### 初始化
```python
widget = Viewer3DWidget(
    parent: Optional[QWidget] = None,
    exr_filepath: Optional[str] = None,         # HDR 环境贴图路径，None 则关闭 IBL
    draw_objects_individually: bool = False     # 是否单独绘制所有对象（便于剔除）
)
```

### 核心 API

#### 1. 添加几何

- `add_points(vertices, key=None, **kwargs)`
  - 参数:
    - `vertices` (Nx3 `np.ndarray`): 点位置
    - `key` (`str`, 可选): 对象标识
    - 支持 `kwargs`:
      - `colors` (Nx3 或 (3,)): 每顶点颜色
      - `point_sizes` (N或Nx1): 每顶点点大小
      - `shape_types` (N或Nx1): 点形状 (0=圆,1=正方形)
      - `instance_data` (Nx4): 自定义实例化数据
      - `normals` (Nx3): 法线，用于光照或法线可视化
      - `texcoords` (Nx2): 纹理坐标
  - 返回: `key` (`str`)

示例:
```python
import numpy as np
# 添加三角位置的调试点
pts = np.array([[0,0,0],[1,1,1],[2,0,2]], dtype=np.float32)
pt_colors = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
pt_sizes = np.array([10, 20, 5], dtype=np.float32)
points_key = widget.add_points(
    vertices=pts,
    colors=pt_colors,
    point_sizes=pt_sizes,
    shape_types=np.array([0,1,0]),
    key="debug_points"
)
```

- `add_lines(vertices, key=None, **kwargs)`
  - 与 `add_points` 类似，要求 `vertices` 中的点成对构成线段
  - 支持 `line_widths` 参数

示例:
```python
# 添加一条红色粗线
line_vs = np.array([[0,0,0],[1,0,1]], dtype=np.float32)
line_key = widget.add_lines(
    vertices=line_vs,
    colors=np.array([[1,0,0],[1,0,0]],dtype=np.float32),
    line_widths=5.0,
    key="red_line"
)
```

- `add_triangles(vertices, key=None, **kwargs)`
  - `vertices` 中每 3 个点构成一个三角面
  - 支持 `colors`, `normals`, `texcoords`,
    - `materials` (3,)或(N×3): [metallic, roughness, ao]
    - `sss_params` (4,)或(N×4): SSS 参数 [strength, distortion, power, scale]
    - `sss_color` (3,)或(N×3): 次表面散射颜色
    - `instance_data` (N×4)

示例:
```python
# 添加带 PBR 材质和 SSS 的球体
from gl_viewer.config import RenderingDefaults
# 假设 verts,norms,cols,sss_p,sss_c 已由 create_sphere 返回
sphere_key = widget.add_triangles(
    vertices=verts,
    colors=cols,
    normals=norms,
    materials=np.array([0.5,0.3,1.0],dtype=np.float32),
    sss_params=np.tile(sss_p[0], (verts.shape[0],1)),
    sss_color=np.tile(sss_c[0], (verts.shape[0],1)),
    key="sphere_sss"
)
```

##### 属性详解
- `colors` (np.ndarray N×3 或 (3,)): 基础颜色，RGB 值范围 [0,1]。
- `normals` (np.ndarray N×3): 法线向量，用于光照计算。
- `texcoords` (np.ndarray N×2): 纹理坐标 (u,v)，用于贴图映射。
- `materials` (np.ndarray (3,) 或 (N×3)): PBR 材质参数 [metallic, roughness, ao]：
  - `metallic` (0.0 非金属 ~ 1.0 金属)
  - `roughness` (0.0 光滑 ~ 1.0 粗糙)
  - `ao` 环境遮蔽强度
- `sss_params` (np.ndarray (4,) 或 (N×4)): 次表面散射参数 [strength, distortion, power, scale]：
  - `strength` 散射强度
  - `distortion` 散射失真
  - `power` 散射幂次
  - `scale` 散射范围缩放
- `sss_color` (np.ndarray (3,) 或 (N×3)): 散射颜色，RGB。
- `point_sizes` (标量 或 np.ndarray N×1): 点大小 (像素)。
- `line_widths` (标量 或 np.ndarray N×1): 线宽 (像素)。
- `shape_types` (整数 或 np.ndarray N×1): 点形状 (0=圆形, 1=方形)。
- `instance_data` (np.ndarray N×4): 每顶点自定义数据，可用于变换或额外属性。

#### 2. 查询当前对象

- `list_points_keys() -> List[str]`
- `list_lines_keys() -> List[str]`
- `list_triangles_keys() -> List[str]`

示例:
```python
print(widget.list_points_keys())      # ['debug_points']
print(widget.list_triangles_keys())   # ['sphere_sss']
```

#### 3. 修改几何

- **modify_points(key, **kwargs) -> bool**
- **modify_lines(key, **kwargs) -> bool**
- **modify_triangles(key, **kwargs) -> bool**

支持的可修改属性:
- `visible` (`bool`): 显示/隐藏
- 顶点属性: `colors`, `normals`, `texcoords`, `point_sizes`, `line_widths`, `materials`, `sss_params`, `sss_color`, `instance_data`

示例:
```python
# 隐藏并延迟恢复
widget.modify_triangles('sphere_sss', visible=False)
QtCore.QTimer.singleShot(1000, lambda: widget.modify_triangles(
    'sphere_sss',
    visible=True,
    colors=np.array([[0,1,1]]),
    sss_color=np.array([[1,0,0]])
))
```

#### 4. 场景设置

- `set_lighting(position, color, ambient_strength, specular_strength)`
- `set_fog(enabled, color, near, far)`
- `set_wireframe(enabled, color)`

示例:
```python
widget.set_lighting([5,5,5], [1,1,1], 0.2, 0.8)
widget.set_fog(True, [0.2,0.2,0.2], 5, 50)
widget.set_wireframe(False, [1,0,0])
```

#### 5. 相机操作

- `camera.set_view_preset(preset)`
- `camera.reset_to_initial()`
- `camera.frame_bounds(min_bounds, max_bounds)`
- `camera.get_camera_info()` -> dict

示例:
```python
widget.camera.set_view_preset('iso')
info = widget.camera.get_camera_info()
print(info['eye'], info['target'])
```

#### 6. 调试与离屏渲染

- `check_gl_error(tag)`
- 使用 `render_scene_to_images(widget, widget.scene_renderer, w, h)` 获取 `(rgba, normals, depth)`

示例:
```python
rgba, normals, depth = render_scene_to_images(widget, widget.scene_renderer, 800,600)
print(rgba.shape)
```

#### 7. 事件响应

- 自动响应鼠标和键盘事件，可重载 Qt 事件处理以定制交互

## 2. config.py 参数说明

`RenderingDefaults` 中定义了一系列全局常量，用于初始化与回退：

1. **颜色**
   - `DEFAULT_POINT_COLOR`, `DEFAULT_LINE_COLOR`, `DEFAULT_TRIANGLE_COLOR`：默认 RGB。
2. **几何属性**
   - `DEFAULT_POINT_SIZE`, `DEFAULT_LINE_WIDTH`, `DEFAULT_SHAPE_TYPE`, `DEFAULT_NORMAL`, `DEFAULT_TRIANGLE_NORMAL`, `DEFAULT_TEXCOORD`。
3. **PBR 材质**
   - `DEFAULT_METALLIC`, `DEFAULT_ROUGHNESS`, `DEFAULT_AO`。
4. **SSS（次表面散射）**
   - `DEFAULT_SSS_PARAMS = [strength, distortion, power, scale]`，`DEFAULT_SSS_COLOR`。
5. **实例化数据**
   - `DEFAULT_INSTANCE_DATA`：可用于每顶点附加变换或额外属性。
6. **光照**
   - `DEFAULT_LIGHT_POSITION`, `DEFAULT_LIGHT_COLOR`, `DEFAULT_AMBIENT_STRENGTH`, `DEFAULT_SPECULAR_STRENGTH`, `DEFAULT_SHININESS`。
7. **雾**
   - `DEFAULT_FOG_COLOR`, `DEFAULT_FOG_NEAR`, `DEFAULT_FOG_FAR`。
8. **线框**
   - `DEFAULT_WIREFRAME_COLOR`。
9. **距离淡出**
   - `DEFAULT_FADE_DISTANCE`。
10. **IBL**
   - `DEFAULT_IBL_AMBIENT_INTENSITY`, `DEFAULT_IBL_SPECULAR_INTENSITY`, `DEFAULT_IBL_BASE_ROUGHNESS`
   - `DEFAULT_IBL_CUBEMAP_SIZE`, `DEFAULT_IBL_BRDF_LUT_SIZE`, `DEFAULT_IBL_BRDF_SAMPLE_COUNT`。
11. **OpenGL 常量**
   - `MSAA_SAMPLES`, `OPENGL_ATTRIBUTE_STRIDE_POINTS_LINES`, `OPENGL_ATTRIBUTE_STRIDE_TRIANGLES`。
12. **缓冲区管理**
   - `BUFFER_GROWTH_FACTOR`, `MIN_BUFFER_OBJECT_CAPACITY`。
13. **相机交互敏感度**
   - `CAMERA_ROTATION_SENSITIVITY`, `CAMERA_PAN_SENSITIVITY`, `CAMERA_ZOOM_SENSITIVITY`, `CAMERA_FOV_ADJUSTMENT_STEP`。
14. **视图预设**
   - `VIEW_PRESET_FRONT/BACK/LEFT/RIGHT/TOP/BOTTOM/ISO/ISO2`。
15. **IBL 生成常量**
   - `IBL_CUBEMAP_FOV_DEGREES`, `IBL_CUBEMAP_ASPECT_RATIO`, `IBL_CUBEMAP_NEAR_PLANE`, `IBL_CUBEMAP_FAR_PLANE`, `IBL_CUBEMAP_FACES_COUNT`。
16. **纹理单元**
   - `TEXTURE_UNIT_DIFFUSE`, `TEXTURE_UNIT_SHAPE`, `TEXTURE_UNIT_ENVIRONMENT`, `TEXTURE_UNIT_BRDF_LUT`, `TEXTURE_UNIT_IRRADIANCE_MAP`, `TEXTURE_UNIT_PREFILTER_MAP`。
17. **鼠标滚轮**
   - `MOUSE_WHEEL_DELTA_PER_NOTCH`。
18. **缓冲管理**
   - `FLOAT_SIZE_BYTES`。
19. **离屏渲染精度**
   - `OFFSCREEN_RENDER_FLOAT_RGBA`: 是否使用 float32 RGBA。

### SSS 参数调优

为了获得理想的散射半径和强度，请按以下方法调整 `sss_params`：

- **strength（散射强度）**: 控制整体次表面散射量。值越大，光线散射越明显，但过大会导致过度泛白。推荐范围：0.2 – 1.0。
- **distortion（散射失真）**: 模拟光线在介质内部的扩散偏折，值越大，光晕边缘越模糊。推荐范围：0.0 – 0.5。
- **power（散射衰减幂次）**: 定义散射衰减曲线的陡峭程度，较大值产生更快速的边缘衰减，较小值更平滑延展。推荐范围：0.5 – 2.0。
- **scale（散射范围缩放）**: 控制散射半径的全局缩放，值越大光线渗透范围越广。推荐范围：0.2 – 1.0。

调整流程示例：

1. 从中等参数开始，例如 `[0.5, 0.1, 1.0, 0.5]`，渲染并观察效果。
2. 若需扩大散射半径，则增大 `scale`；若散射整体不足，则提升 `strength`。
3. 如需更柔和边缘光晕，可增大 `distortion`；如需更陡峭衰减，可增大 `power`。
4. 重复上述步骤，直至达到预期视觉效果。

---

此手册可作为 `gl_viewer` 的快速入门与 API 参考，用于快速集成与二次开发。
