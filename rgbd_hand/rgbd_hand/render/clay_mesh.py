"""固定拓扑的米制泥坯；局部 +Y 为轴向，原点为几何中心。

所有 getter 返回副本。D 修改 vertices_local 后显式 set_vertices()；
E 使用 mesh_local()/mesh_scene() 和 version 获取同一版本的拾取几何。
"""
from __future__ import annotations

import numpy as np
import open3d as o3d

CLAY_COLOR = (0.70, 0.43, 0.29)


class ClayMesh:
    def __init__(self, radius=0.075, height=0.18, radial_segments=96,
                 height_segments=48, cap_segments=4):
        for name, value in (("radius", radius), ("height", height)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name, value, minimum in (("radial_segments", radial_segments, 3),
                                     ("height_segments", height_segments, 1),
                                     ("cap_segments", cap_segments, 1)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        self.radius, self.height = float(radius), float(height)
        self.radial_segments = n = int(radial_segments)
        self.height_segments = h = int(height_segments)
        self.cap_segments = int(cap_segments)
        theta = np.arange(n) * (2 * np.pi / n)
        circle = np.column_stack((np.cos(theta), np.zeros(n), np.sin(theta)))
        vertices, faces = [], []
        for y in np.linspace(-height / 2, height / 2, h + 1):
            ring = radius * circle.copy()
            ring[:, 1] = y
            vertices.extend(ring)
        for row in range(h):
            for j in range(n):
                a, b = row * n + j, row * n + (j + 1) % n
                faces.extend(((a, a + n, b), (b, a + n, b + n)))
        # 端面外环复用侧面顶点；角度取模闭合接缝，没有重合的重复点。
        for top in (False, True):
            outer = np.arange(n) + (h * n if top else 0)
            y = height / 2 if top else -height / 2
            for level in range(self.cap_segments - 1, 0, -1):
                inner = np.arange(n) + len(vertices)
                ring = radius * level / self.cap_segments * circle.copy()
                ring[:, 1] = y
                vertices.extend(ring)
                for j in range(n):
                    k = (j + 1) % n
                    pair = ((outer[j], inner[j], outer[k]),
                            (outer[k], inner[j], inner[k]))
                    faces.extend(pair if top else tuple(tuple(reversed(f)) for f in pair))
                outer = inner
            center = len(vertices)
            vertices.append((0, y, 0))
            for j in range(n):
                face = (outer[j], center, outer[(j + 1) % n])
                faces.append(face if top else tuple(reversed(face)))
        self._vertices = np.asarray(vertices, dtype=np.float64)
        self._initial = self._vertices.copy()
        self._triangles = np.asarray(faces, dtype=np.int32)
        self._pose = np.eye(4)
        self.shape_version = 0
        self.version = 0  # 形状或位姿变化均递增，供显示与拾取同步。

    @property
    def vertices_local(self):
        return self._vertices.copy()

    @property
    def initial_vertices(self):
        return self._initial.copy()

    @property
    def triangles(self):
        return self._triangles.copy()

    @property
    def side_vertex_indices(self):
        return np.arange(self.radial_segments * (self.height_segments + 1))

    @property
    def pose_scene(self):
        return self._pose.copy()

    def set_vertices(self, vertices):
        vertices = np.asarray(vertices, dtype=np.float64)
        if vertices.shape != self._vertices.shape or not np.isfinite(vertices).all():
            raise ValueError(f"vertices must be finite with shape {self._vertices.shape}")
        if not np.array_equal(vertices, self._vertices):
            self._vertices = vertices.copy()
            self.shape_version += 1
            self.version += 1

    def set_pose(self, pose_scene):
        """刚体位姿：局部坐标 -> Open3D 场景坐标。禁止缩放和镜像。"""
        pose = np.asarray(pose_scene, dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("pose must be a finite 4x4 rigid transform")
        r = pose[:3, :3]
        if (not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
                or not np.allclose(r.T @ r, np.eye(3), atol=1e-8, rtol=0)
                or not np.isclose(np.linalg.det(r), 1, atol=1e-8, rtol=0)):
            raise ValueError("pose must contain a proper rotation and translation")
        if not np.array_equal(pose, self._pose):
            self._pose = pose.copy()
            self.version += 1

    def reset(self):
        """恢复初始形状，保留位姿；场景复位由 Scene.reset_clay() 完成。"""
        self.set_vertices(self._initial)

    def mesh_local(self, initial=False):
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(self._initial if initial else self._vertices),
            o3d.utility.Vector3iVector(self._triangles))
        mesh.paint_uniform_color(CLAY_COLOR)
        mesh.compute_vertex_normals()
        return mesh

    def mesh_scene(self):
        mesh = self.mesh_local()
        mesh.transform(self._pose)
        return mesh
