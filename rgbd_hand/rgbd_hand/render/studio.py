"""陶艺工作室的静态几何。所有尺寸沿用场景米制坐标。"""
from pathlib import Path

import numpy as np
import open3d as o3d

TURNTABLE_HEIGHT = .028
STUDIO_BACKGROUND = np.array([.1255, .1451, .1373])  # #202523


def clay_studio_pose(clay, table_top=.18, distance=.5):
    pose = np.eye(4)
    pose[:3, 3] = [0, -table_top + TURNTABLE_HEIGHT + clay.height / 2, -distance]
    return pose


def configure_studio_lighting(option):
    # legacy Visualizer 的光源参数通过其原生 RenderOption JSON 设置。
    option.load_from_json(str(Path(__file__).with_name("studio_render.json")))


def cylinder_y(radius, height, center, color):
    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius, height, resolution=128, split=1)
    mesh.rotate(o3d.geometry.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0)), center=(0, 0, 0))
    mesh.translate(center)
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def add_studio(vis, clay, table_top):
    """木桌、两层转盘和刻度；几何保存引用，不参与雕刻或拾取。"""
    geometries = []
    x, _, z = clay.pose_scene[:3, 3]
    y = -table_top
    sx, sz = .78, .56
    slab = o3d.geometry.TriangleMesh.create_box(sx, .035, sz)
    slab.translate([x - sx / 2, y - .035, z - sz / 2])
    slab.paint_uniform_color([.39, .31, .245])
    slab.compute_vertex_normals()
    geometries.append(slab)

    # 稠密顶点颜色给桌面轻微木纹，避免图像资源和高对比重复纹理。
    nx, nz = 100, 44
    xx, zz = np.meshgrid(np.linspace(x-sx/2, x+sx/2, nx), np.linspace(z-sz/2, z+sz/2, nz))
    vertices = np.column_stack((xx.ravel(), np.full(xx.size, y+.0002), zz.ravel()))
    faces = []
    for row in range(nz-1):
        for col in range(nx-1):
            a = row*nx+col
            faces.extend(((a, a+nx, a+1), (a+1, a+nx, a+nx+1)))
    top = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
    grain = .004*np.sin(xx.ravel()*220 + .6*np.sin(zz.ravel()*13)) + .0015*np.sin(xx.ravel()*520)
    colors = np.array([.59, .48, .37])[None, :] + grain[:, None]
    top.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
    top.compute_vertex_normals()
    geometries.append(top)

    # 深色底座与石灰色盘面；盘顶与泥坯底面严格对齐。
    geometries.append(cylinder_y(.122, .016, [x, y+.008, z], [.27, .29, .27]))
    geometries.append(cylinder_y(.142, .012, [x, y+.022, z], [.50, .51, .47]))
    # 低对比环线和少量径向刻度，表现陶艺转盘而非游戏底座。
    points, lines = [], []
    for radius in (.118, .135):
        start = len(points)
        for angle in np.linspace(0, 2*np.pi, 128, endpoint=False):
            points.append([x + radius*np.cos(angle), y+TURNTABLE_HEIGHT+.00015, z+radius*np.sin(angle)])
        lines.extend((start+i, start+(i+1)%128) for i in range(128))
    for angle in np.linspace(0, 2*np.pi, 24, endpoint=False):
        start = len(points)
        for radius in (.128, .134):
            points.append([x+radius*np.cos(angle), y+TURNTABLE_HEIGHT+.0002, z+radius*np.sin(angle)])
        lines.append((start, start+1))
    rings = o3d.geometry.LineSet(o3d.utility.Vector3dVector(points), o3d.utility.Vector2iVector(lines))
    rings.paint_uniform_color([.36, .38, .34])
    geometries.append(rings)
    for geometry in geometries:
        vis.add_geometry(geometry, reset_bounding_box=False)
    return geometries
