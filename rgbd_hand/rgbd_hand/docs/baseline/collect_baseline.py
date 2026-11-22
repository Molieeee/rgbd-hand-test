"""Run the existing webcam entry point and collect B01 scene measurements.

Run from the business-code directory:
    python docs/baseline/collect_baseline.py --view first --runs 2

Instrumentation is limited to this process. The application's modules are restored
after each run. Raw camera images are displayed by main.py and are not saved here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
import main as app
from render.scene import to_scene


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def render_fixture(args, output):
    """Verify HandVisual with known points, independently of camera detection."""
    from workbench_ui import demo_hand

    points = np.asarray(demo_hand(x=-0.09, y=0.05, z=0.52), dtype=np.float64)
    scene = app.Scene(
        width=args.width, height=args.height, first_person=args.view == "first",
        focus_distance=app.OBJECT_CENTER[2], look_at_y=app.OBJECT_CENTER[1],
    )
    try:
        scene.update_object(app.initial_object_T(app.OBJECT_CENTER))
        scene.update_hands([points], [False])
        expected = to_scene(points)
        actual = np.asarray(scene.hands[0].skeleton.points)
        joint_centers = np.array([np.asarray(joint.vertices).mean(axis=0)
                                  for joint in scene.hands[0].joints])
        np.testing.assert_allclose(actual, expected, atol=1e-10)
        np.testing.assert_allclose(joint_centers, expected, atol=1e-10)
        for _ in range(20):
            if not scene.render():
                raise RuntimeError("Fixture window closed before capture")
        scene.vis.capture_screen_image(str(output / f"{args.view}-fixture.png"), do_render=True)
        save_json(output / f"{args.view}-fixture.json", {
            "input": "workbench_ui.demo_hand fixed synthetic points",
            "view": args.view, "points_camera_m": points.tolist(),
            "skeleton_point_count": len(actual),
            "skeleton_line_count": len(scene.hands[0].skeleton.lines),
            "joint_count": len(scene.hands[0].joints),
            "skeleton_max_abs_error_m": float(np.max(np.abs(actual - expected))),
            "joint_center_max_abs_error_m": float(np.max(np.abs(joint_centers - expected))),
        })
    finally:
        scene.close()


def measure(args, output, run_number):
    label = f"{args.view}-{run_number}"
    started = time.perf_counter()
    original_scene = app.Scene
    original_source = app.build_source
    original_wait_key = cv2.waitKey
    samples = []
    details = {"view": args.view, "run": run_number, "started_at": datetime.now().astimezone().isoformat()}
    state = {"last_render": None, "first_render": None, "hands": 0, "max_hands": 0, "hand_capture": False}

    class SourceProbe:
        def __init__(self, source):
            self.source = source
            details["camera_backend"] = source.cap.getBackendName()
            details["camera_reported_fps"] = source.cap.get(cv2.CAP_PROP_FPS)

        def read(self):
            frame = self.source.read()
            if frame is not None:
                details["frame_size"] = [frame.color.shape[1], frame.color.shape[0]]
                details["has_depth"] = frame.has_depth
            return frame

        def close(self):
            self.source.close()

    class SceneProbe(original_scene):
        def __init__(self, *positional, **keywords):
            super().__init__(*positional, **keywords)
            details["requested_window_size"] = [self.width, self.height]
            details["cube_vertices"] = len(self.box.vertices)
            details["cube_triangles"] = len(self.box.triangles)
            details["joint_vertices_each"] = len(self.hands[0].joints[0].vertices)
            details["joint_triangles_each"] = len(self.hands[0].joints[0].triangles)

        def update_hands(self, hands, pinching=None):
            state["hands"] = len(hands)
            state["max_hands"] = max(state["max_hands"], len(hands))
            return super().update_hands(hands, pinching)

        def render(self):
            before = time.perf_counter()
            alive = super().render()
            after = time.perf_counter()
            if state["first_render"] is None:
                state["first_render"] = after
                details["startup_to_first_render_s"] = after - started
                params = self.vis.get_view_control().convert_to_pinhole_camera_parameters()
                details["intrinsic"] = np.asarray(params.intrinsic.intrinsic_matrix).tolist()
                details["extrinsic"] = np.asarray(params.extrinsic).tolist()
                details["viewport_size"] = [params.intrinsic.width, params.intrinsic.height]
                details["camera_preview_requested_size"] = [min(args.width, 960), min(args.height, 540)]
                details["camera_preview_image_rect"] = list(cv2.getWindowImageRect("RGBD Hand - camera"))
                self.vis.capture_screen_image(str(output / f"{label}-initial.png"), do_render=True)
                print(f"BASELINE READY {label}", flush=True)
            elapsed = after - state["first_render"]
            if state["hands"] and not state["hand_capture"] and elapsed >= args.warmup:
                self.vis.capture_screen_image(str(output / f"{label}-hand.png"), do_render=True)
                state["hand_capture"] = True
            if state["last_render"] is not None and elapsed >= args.warmup:
                samples.append({
                    "elapsed_s": elapsed,
                    "loop_ms": (before - state["last_render"]) * 1000,
                    "render_ms": (after - before) * 1000,
                    "hands": state["hands"],
                })
            # Keep screenshot readback outside the following loop-time sample.
            state["last_render"] = time.perf_counter()
            return alive

    def wait_key(delay):
        key = original_wait_key(delay)
        if state["first_render"] is not None:
            elapsed = time.perf_counter() - state["first_render"]
            if elapsed >= args.warmup + args.seconds:
                return ord("q")
        return key

    app.Scene = SceneProbe
    app.build_source = lambda options: SourceProbe(original_source(options))
    cv2.waitKey = wait_key
    try:
        details["exit_code"] = app.main([
            "--source", "webcam", "--camera", str(args.camera), "--view", args.view,
            "--width", str(args.width), "--height", str(args.height),
        ])
    finally:
        app.Scene = original_scene
        app.build_source = original_source
        cv2.waitKey = original_wait_key
        details["max_detected_hands"] = state["max_hands"]
        details["measured_samples"] = len(samples)
        details["warmup_s"] = args.warmup
        details["requested_sample_duration_s"] = args.seconds
        if samples:
            loop = np.array([s["loop_ms"] + s["render_ms"] for s in samples])
            render = np.array([s["render_ms"] for s in samples])
            details["mean_loop_fps"] = float(1000 / np.mean(loop))
            details["loop_ms_mean"] = float(np.mean(loop))
            details["loop_ms_p95"] = float(np.percentile(loop, 95))
            details["render_ms_mean"] = float(np.mean(render))
            details["render_ms_p95"] = float(np.percentile(render, 95))
            details["frames_with_hands"] = sum(s["hands"] > 0 for s in samples)
        save_json(output / f"{label}.json", details)
        if samples:
            with (output / f"{label}.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
                writer.writeheader()
                writer.writerows(samples)
    return details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", choices=["first", "camera"], default="first")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--warmup", type=float, default=3)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--fixture", action="store_true", help="render fixed synthetic hand points without a camera")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "2026-11-20")
    args = parser.parse_args()
    if args.runs < 1 or args.seconds <= 0 or args.warmup < 0:
        parser.error("runs/seconds must be positive and warmup must be nonnegative")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.fixture:
        render_fixture(args, output)
        return
    environment = {
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "versions": {name: version(name) for name in ["open3d", "mediapipe", "numpy", "opencv-python"]},
        "source_sha256": {name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest()
                          for name in ["main.py", "render/scene.py", "assets/hand_landmarker.task"]},
    }
    save_json(output / "environment.json", environment)
    results = []
    for index in range(1, args.runs + 1):
        result = measure(args, output, index)
        results.append(result)
        print(json.dumps(result, ensure_ascii=True), flush=True)
    if len(results) > 1:
        comparisons = []
        reference = cv2.imdecode(np.fromfile(output / f"{args.view}-1-initial.png", dtype=np.uint8), cv2.IMREAD_COLOR)
        for index, result in enumerate(results[1:], start=2):
            current = cv2.imdecode(np.fromfile(output / f"{args.view}-{index}-initial.png", dtype=np.uint8), cv2.IMREAD_COLOR)
            comparisons.append({
                "reference_run": 1, "compared_run": index,
                "intrinsic_max_abs_difference": float(np.max(np.abs(np.array(results[0]["intrinsic"]) - result["intrinsic"]))),
                "extrinsic_max_abs_difference": float(np.max(np.abs(np.array(results[0]["extrinsic"]) - result["extrinsic"]))),
                "initial_screenshots_equal": bool(np.array_equal(reference, current)),
            })
        save_json(output / f"{args.view}-repeatability.json", comparisons)


if __name__ == "__main__":
    main()
