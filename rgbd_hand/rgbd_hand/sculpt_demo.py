"""D01/D02 无摄像头雕刻演示。

    python sculpt_demo.py
    python sculpt_demo.py --verify-dir D:/tmp/sculpt_verify

C 用固定局部坐标压一个凹坑，S 在当前位置平滑，R 复位，A/F 转动泥坯，
V 恢复视角，Q 退出。--verify-dir 会保存初始/雕刻/背面/复位截图和
verification.json，不修改 B 的 clay_demo.py。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import open3d as o3d

from interaction.sculpt import (
    MAX_STEP_DEPTH_M,
    MAX_STROKE_DEPTH_M,
    BrushCommand,
    PickResult,
    Topology,
    apply_brush,
    select_vertices,
)
from render import ClayMesh, Scene


def _front_axis(first_person: bool) -> tuple[np.ndarray, np.ndarray, float]:
    sign = -1.0 if first_person else 1.0
    return (
        np.array([0.0, 0.0, sign]),
        np.array([0.0, 0.0, sign]),
        sign,
    )


def verify(scene: Scene, output: Path, brush_radius: float, strength: float,
           steps: int, dt: float) -> None:
    output.mkdir(parents=True, exist_ok=True)
    clay = scene.clay
    assert clay is not None
    vertices = clay.vertices_local
    triangles = clay.triangles
    topology = Topology(vertices, triangles)
    normal_axis, normal, front_sign = _front_axis(scene.first_person)
    front_point = normal_axis * clay.radius
    center = topology.nearest_vertex(front_point, vertices)
    report = {
        "open3d": o3d.__version__,
        "input": "fixed clay-local point/normal; no camera",
        "vertices": len(vertices),
        "triangles": len(triangles),
        "clay_radius_m": clay.radius,
        "clay_height_m": clay.height,
        "brush_radius_m": brush_radius,
        "strength": strength,
        "dt_s": dt,
        "steps": steps,
        "max_step_depth_m": MAX_STEP_DEPTH_M,
        "max_stroke_depth_m": MAX_STROKE_DEPTH_M,
        "front_sign": front_sign,
        "front_center_index": center,
        "front_center_initial": vertices[center].tolist(),
    }
    display_id = id(scene.clay_visual)
    capture_times = []

    def capture(name: str) -> None:
        for _ in range(8):
            if not scene.render():
                raise RuntimeError("Window closed during verification")
        scene.vis.capture_screen_image(str(output / f"{name}.png"), do_render=True)

    def record_step(step: int) -> dict:
        current = clay.vertices_local
        point = current[center].copy()
        pick = PickResult(True, 0, point, normal, 0.0, clay.version)
        command = BrushCommand("carve", brush_radius, strength, dt, stroke_id=1)
        result = apply_brush(
            current, pick, command, topology, stroke_start=stroke_start
        )
        if not result.applied:
            raise RuntimeError(f"carve rejected: {result.reason}")
        started = time.perf_counter()
        scene.update_clay(result.vertices)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        capture_times.append(elapsed_ms)
        total = np.linalg.norm(clay.vertices_local - stroke_start, axis=1)
        return {
            "step": step + 1,
            "selected_vertices": result.selected_count,
            "changed_vertices": int(np.count_nonzero(result.changed_mask)),
            "step_depth_m": result.step_depth_m,
            "max_displacement_m": result.max_displacement_m,
            "max_total_displacement_m": float(total.max()),
            "display_update_ms": elapsed_ms,
            "mesh_version": clay.version,
        }

    capture("01-initial")
    stroke_start = vertices.copy()
    step_reports = [record_step(step) for step in range(steps)]
    report["steps_report"] = step_reports
    capture("02-carve-final")

    final = clay.vertices_local
    total = np.linalg.norm(final - stroke_start, axis=1)
    changed = total > 0.0
    side_count = clay.radial_segments * (clay.height_segments + 1)
    cap_only = np.arange(side_count, len(final))
    back = front_sign * vertices[:, 2] <= 0.0
    report.update(
        changed_vertices=int(np.count_nonzero(changed)),
        max_total_displacement_m=float(total.max()),
        backside_changed_vertices=int(np.count_nonzero(changed & back)),
        cap_changed_vertices=int(np.count_nonzero(changed[cap_only])),
        backside_unchanged=not bool(np.any(changed & back)),
        caps_unchanged=not bool(np.any(changed[cap_only])),
        display_update_ms_mean=float(np.mean(capture_times)),
        display_update_ms_max=float(np.max(capture_times)),
    )
    assert report["backside_unchanged"]
    assert report["caps_unchanged"]
    np.testing.assert_allclose(
        report["max_total_displacement_m"], MAX_STROKE_DEPTH_M, atol=1e-9
    )

    # 命中点左右邻点应完全对称。
    n = clay.radial_segments
    ring = center // n
    column = center % n
    left = ring * n + (column - 1) % n
    right = ring * n + (column + 1) % n
    neighbor_left = float(np.linalg.norm(final[left] - stroke_start[left]))
    neighbor_right = float(np.linalg.norm(final[right] - stroke_start[right]))
    report["front_neighbors_symmetric"] = bool(
        np.isclose(neighbor_left, neighbor_right, atol=1e-12)
    )
    report["front_neighbor_displacements_m"] = [neighbor_left, neighbor_right]
    assert report["front_neighbors_symmetric"]

    # D02 接缝：命中 j=0 时，j=1 与 j=n-1 的权重必须相同。
    seam_point = np.array([clay.radius, 0.0, 0.0])
    seam_normal = np.array([1.0, 0.0, 0.0])
    seam_selection = select_vertices(
        vertices, seam_point, seam_normal, brush_radius, topology
    )
    seam_row = clay.height_segments // 2
    seam_center = seam_row * n
    seam_left = seam_row * n + n - 1
    seam_right = seam_row * n + 1
    report["seam_selection"] = {
        "center_inside": bool(seam_selection.mask[seam_center]),
        "left_inside": bool(seam_selection.mask[seam_left]),
        "right_inside": bool(seam_selection.mask[seam_right]),
        "left_weight": float(seam_selection.weights[seam_left]),
        "right_weight": float(seam_selection.weights[seam_right]),
    }
    report["seam_selection_symmetric"] = bool(
        np.isclose(
            seam_selection.weights[seam_left],
            seam_selection.weights[seam_right],
            atol=1e-12,
        )
    )
    assert report["seam_selection"]["center_inside"]
    assert report["seam_selection"]["left_inside"]
    assert report["seam_selection"]["right_inside"]
    assert report["seam_selection_symmetric"]

    # 几何质量：拓扑不变、闭合、无自交、侧面没有翻面。
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(final),
        o3d.utility.Vector3iVector(triangles),
    )
    mesh.compute_vertex_normals()
    a, b, c = (final[triangles[:, k]] for k in range(3))
    cross = np.cross(b - a, c - a)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    centers = (a + b + c) / 3.0
    side_faces = np.abs(centers[:, 1]) < clay.height / 2 - 1e-9
    outward = np.sum(
        cross[side_faces][:, [0, 2]] * centers[side_faces][:, [0, 2]], axis=1
    )
    report["geometry"] = {
        "watertight": bool(mesh.is_watertight()),
        "orientable": bool(mesh.is_orientable()),
        "self_intersecting": bool(mesh.is_self_intersecting()),
        "min_triangle_area_m2": float(areas.min()),
        "min_side_outward_dot": float(outward.min()),
        "all_finite": bool(np.isfinite(final).all()),
    }
    assert report["geometry"]["watertight"]
    assert report["geometry"]["orientable"]
    assert not report["geometry"]["self_intersecting"]
    assert report["geometry"]["min_triangle_area_m2"] > 0
    assert report["geometry"]["min_side_outward_dot"] > 0
    assert report["geometry"]["all_finite"]

    # 当前形状与显示几何一致，且显示对象没有重建。
    pose = clay.pose_scene
    expected = clay.vertices_local @ pose[:3, :3].T + pose[:3, 3]
    np.testing.assert_allclose(
        np.asarray(scene.clay_visual.vertices), expected, atol=1e-12
    )
    report["display_matches_current_mesh"] = True
    report["display_geometry_reused"] = id(scene.clay_visual) == display_id
    assert report["display_geometry_reused"]

    # 转到背面截图：背面没有凹坑/凸起。
    pose = clay.pose_scene
    pose[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz((0, np.pi, 0)) @ pose[:3, :3]
    scene.update_clay(pose_scene=pose)
    capture("03-backside")
    report["backside_screenshot"] = "03-backside.png"

    scene.reset_clay()
    scene._frame_view()
    np.testing.assert_array_equal(clay.vertices_local, stroke_start)
    capture("04-reset")
    report["reset_restores_initial_shape"] = True
    report["final_mesh_version"] = clay.version
    (output / "verification.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="D01/D02 clay sculpting demo")
    parser.add_argument("--radius", type=float, default=0.075)
    parser.add_argument("--height", type=float, default=0.18)
    parser.add_argument("--radial-segments", type=int, default=96)
    parser.add_argument("--height-segments", type=int, default=48)
    parser.add_argument("--cap-segments", type=int, default=4)
    parser.add_argument("--brush-radius", type=float, default=0.024)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--view", choices=["first", "camera"], default="first")
    parser.add_argument("--verify-dir", type=Path)
    parser.add_argument("--verify-steps", type=int, default=3)
    parser.add_argument("--verify-dt", type=float, default=0.1)
    args = parser.parse_args()

    clay = ClayMesh(
        args.radius, args.height, args.radial_segments,
        args.height_segments, args.cap_segments,
    )
    pose = np.eye(4)
    pose[:3, 3] = [0.0, -0.18 + clay.height / 2.0, -0.50]
    clay.set_pose(pose)
    scene = Scene(
        window_name=(
            "D Sculpt | C carve  S smooth  R reset  A/F turn  V view  Q quit"
        ),
        clay=clay,
        focus_distance=0.50,
        look_at_y=-pose[1, 3],
        first_person=args.view == "first",
    )
    try:
        print(
            f"Clay: {len(clay.vertices_local)} vertices, "
            f"{len(clay.triangles)} triangles"
        )
        print(
            "C carve | S smooth | R reset | A/F turn | V view | Q quit\n"
            f"brush radius={args.brush_radius:.3f} m, strength={args.strength:.2f}"
        )
        if args.verify_dir:
            verify(
                scene, args.verify_dir, args.brush_radius, args.strength,
                args.verify_steps, args.verify_dt,
            )
            return

        vertices = clay.vertices_local
        topology = Topology(vertices, clay.triangles)
        normal_axis, normal, _ = _front_axis(scene.first_person)
        center = topology.nearest_vertex(normal_axis * clay.radius, vertices)
        state = {
            "tool": None,
            "stroke_id": 0,
            "stroke_start": vertices.copy(),
        }

        def begin_stroke(tool: str) -> None:
            if state["tool"] != tool:
                state["tool"] = tool
                state["stroke_id"] += 1
                state["stroke_start"] = clay.vertices_local

        def apply_tool(tool: str, dt: float) -> None:
            begin_stroke(tool)
            current = clay.vertices_local
            pick = PickResult(
                True, 0, current[center].copy(), normal, 0.0, clay.version
            )
            command = BrushCommand(
                tool, args.brush_radius, args.strength, dt, state["stroke_id"]
            )
            result = apply_brush(
                current, pick, command, topology,
                stroke_start=state["stroke_start"],
            )
            if not result.applied:
                print(f"{tool}: not applied ({result.reason})")
                return
            scene.update_clay(result.vertices)
            total = np.linalg.norm(
                clay.vertices_local - state["stroke_start"], axis=1
            ).max()
            print(
                f"{tool}: selected={result.selected_count} "
                f"changed={len(result.changed_indices)} "
                f"step={result.max_displacement_m * 1000:.2f} mm "
                f"total={total * 1000:.2f} mm"
            )

        def carve_callback(vis) -> bool:
            apply_tool("carve", 1.0 / 30.0)
            return False

        def smooth_callback(vis) -> bool:
            apply_tool("smooth", 1.0 / 30.0)
            return False

        def reset_callback(vis) -> bool:
            scene.reset_clay()
            state["tool"] = None
            state["stroke_id"] = 0
            state["stroke_start"] = clay.vertices_local
            print("reset: shape and pose restored")
            return False

        def turn_callback(angle: float):
            def callback(vis) -> bool:
                pose = clay.pose_scene
                pose[:3, :3] = (
                    o3d.geometry.get_rotation_matrix_from_xyz((0, angle, 0))
                    @ pose[:3, :3]
                )
                scene.update_clay(pose_scene=pose)
                return False
            return callback

        def view_callback(vis) -> bool:
            scene._frame_view()
            return False

        def quit_callback(vis) -> bool:
            vis.close()
            return False

        scene.vis.register_key_callback(ord("C"), carve_callback)
        scene.vis.register_key_callback(ord("S"), smooth_callback)
        scene.vis.register_key_callback(ord("R"), reset_callback)
        scene.vis.register_key_callback(ord("A"), turn_callback(-0.15))
        scene.vis.register_key_callback(ord("F"), turn_callback(0.15))
        scene.vis.register_key_callback(ord("V"), view_callback)
        scene.vis.register_key_callback(ord("Q"), quit_callback)
        while scene.render():
            time.sleep(0.01)
    finally:
        scene.close()


if __name__ == "__main__":
    main()
