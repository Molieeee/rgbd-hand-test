"""B02/B03 无摄像头场景。python clay_demo.py；--verify-dir 保存验收证据。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import open3d as o3d

from render import ClayMesh, Scene


def demo_dent(clay, front_sign=-1, brush_radius=None, depth=None):
    """固定局部坐标的显示夹具，不是 D 的连续笔刷算法。"""
    vertices = clay.initial_vertices
    center = np.array([0, 0, front_sign * clay.radius])
    distance = np.linalg.norm(vertices - center, axis=1)
    brush_radius = brush_radius if brush_radius is not None else min(clay.radius * .65, clay.height * .35)
    weight = np.maximum(0, 1 - (distance / brush_radius) ** 2) ** 2
    # 仅前侧侧面；端面及半径外顶点保持不变。
    selected = np.zeros(len(vertices), dtype=bool)
    selected[clay.side_vertex_indices] = True
    selected &= (front_sign * vertices[:, 2] > 0) & (np.abs(vertices[:, 1]) < clay.height / 2)
    depth = depth if depth is not None else min(.022, clay.radius * .3)
    vertices[selected, 2] -= front_sign * depth * weight[selected]
    return vertices


def verify(scene, output):
    output.mkdir(parents=True, exist_ok=True)
    clay = scene.clay
    report = {"open3d": o3d.__version__, "vertices": len(clay.vertices_local),
              "triangles": len(clay.triangles), "input": "fixed local vertices; no camera"}
    display_id = id(scene.clay_visual)

    def capture(name):
        for _ in range(8):
            if not scene.render():
                raise RuntimeError("Window closed during verification")
        scene.vis.capture_screen_image(str(output / f"{name}.png"), do_render=True)

    capture("01-initial")
    edited = demo_dent(clay, -1 if scene.first_person else 1)
    started = time.perf_counter()
    scene.update_clay(edited)
    report["shape_update_ms"] = (time.perf_counter() - started) * 1000
    changed = np.linalg.norm(edited - clay.initial_vertices, axis=1)
    assert np.count_nonzero(changed) > 0
    report["changed_vertices"] = int(np.count_nonzero(changed))
    report["max_displacement_m"] = float(changed.max())
    capture("02-dent")
    pose = clay.pose_scene
    pose[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz((0, .55, 0))
    scene.update_clay(pose_scene=pose)
    capture("03-posed")
    scene.vis.get_view_control().rotate(90, 25)
    capture("04-orbit")
    np.testing.assert_array_equal(clay.vertices_local, edited)
    expected = edited @ pose[:3, :3].T + pose[:3, 3]
    np.testing.assert_allclose(np.asarray(scene.clay_visual.vertices), expected, atol=1e-12)
    np.testing.assert_allclose(np.asarray(clay.mesh_scene().vertices), expected, atol=1e-12)
    report["display_matches_current_mesh"] = True
    report["dent_survives_pose_and_view"] = True
    assert id(scene.clay_visual) == display_id
    scene.reset_clay()
    np.testing.assert_array_equal(clay.vertices_local, clay.initial_vertices)
    np.testing.assert_array_equal(clay.pose_scene, scene._initial_clay_pose)
    scene._frame_view()
    capture("05-reset")
    report["reset_restores_shape_and_pose"] = True
    report["display_geometry_reused"] = True
    report["final_mesh_version"] = clay.version
    (output / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Camera-free editable clay scene")
    parser.add_argument("--radial-segments", type=int, default=96)
    parser.add_argument("--height-segments", type=int, default=48)
    parser.add_argument("--cap-segments", type=int, default=4)
    parser.add_argument("--radius", type=float, default=.075)
    parser.add_argument("--height", type=float, default=.18)
    parser.add_argument("--view", choices=["first", "camera"], default="first")
    parser.add_argument("--verify-dir", type=Path)
    args = parser.parse_args()
    clay = ClayMesh(args.radius, args.height, args.radial_segments,
                    args.height_segments, args.cap_segments)
    pose = np.eye(4)
    pose[:3, 3] = [0, -.18 + clay.height / 2, -.50]
    clay.set_pose(pose)
    scene = Scene(window_name="Clay Workbench | D: dent  R: reset  A/F: turn  V: view  Q: quit",
                  clay=clay, focus_distance=.50, look_at_y=-pose[1, 3],
                  first_person=args.view == "first")
    try:
        print(f"Clay: {len(clay.vertices_local)} vertices, {len(clay.triangles)} triangles")
        print("D demo dent | R reset shape/pose | A/F turn clay | V reset view | W wireframe | Q quit")
        if args.verify_dir:
            verify(scene, args.verify_dir)
            return

        def dent(vis):
            scene.update_clay(demo_dent(clay, -1 if scene.first_person else 1))
            return False

        def reset(vis):
            scene.reset_clay()
            return False

        def turn(angle):
            def callback(vis):
                pose = clay.pose_scene
                pose[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz((0, angle, 0)) @ pose[:3, :3]
                scene.update_clay(pose_scene=pose)
                return False
            return callback

        def view(vis):
            scene._frame_view()
            return False

        scene.vis.register_key_callback(ord("D"), dent)
        scene.vis.register_key_callback(ord("R"), reset)
        scene.vis.register_key_callback(ord("A"), turn(-.15))
        scene.vis.register_key_callback(ord("F"), turn(.15))
        scene.vis.register_key_callback(ord("V"), view)
        while scene.render():
            time.sleep(.01)
    finally:
        scene.close()


if __name__ == "__main__":
    main()
