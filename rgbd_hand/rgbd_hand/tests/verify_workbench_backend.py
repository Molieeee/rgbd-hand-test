"""真实 Open3D + spawn + 队列集成测试，相机使用明确的合成夹具。"""
import json
import multiprocessing as mp
from pathlib import Path
import queue
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def worker(args, snapshots, actions, stop, cleanup):
    import numpy as np
    import main
    import perception
    from capture.base import Frame
    from clay_runtime import run_clay
    from workbench_app import drain, latest_put
    from workbench_ui import demo_hand
    resources = []
    original_source = main.build_source

    class SyntheticSource:
        fps = 10
        def __init__(self):
            self.count = 0
            self.closed = False
            resources.append(self)

        def read(self):
            time.sleep(.06)
            self.count += 1
            if self.count > 6:
                return None
            return Frame(np.full((120, 160, 3), [70, 110, 150], dtype=np.uint8),
                         timestamp_ms=self.count * 60)

        def close(self):
            self.closed = True

    class SyntheticLandmarker:
        def __init__(self, **kwargs):
            self.closed = False
            resources.append(self)

        def detect(self, image, stamp):
            points = [SimpleNamespace(x=.4 + i * .005, y=.4 + i * .005) for i in range(21)]
            return SimpleNamespace(hand_landmarks=[points], hand_world_landmarks=[points])

        def close(self):
            self.closed = True

    class SyntheticProvider:
        def __init__(self, **kwargs):
            pass

        def to_camera3d(self, *args):
            return np.asarray(demo_hand(x=-.1, y=.02, z=.5))

    main.build_source = lambda config: SyntheticSource() if config.source == "webcam" else original_source(config)
    perception.HandLandmarkerWrapper = SyntheticLandmarker
    perception.WebcamHand3D = SyntheticProvider
    try:
        run_clay(args, lambda snap: latest_put(snapshots, snap), lambda: drain(actions), stop.is_set)
    finally:
        cleanup.put({"resources_closed": all(resource.closed for resource in resources),
                     "resource_count": len(resources)})


def main():
    import numpy as np
    from main import parse_args
    args = parse_args([])
    args.source, args.width, args.height = "demo", 640, 480
    context = mp.get_context("spawn")
    snapshots, actions, cleanup = context.Queue(2), context.Queue(128), context.Queue()
    stop = context.Event()
    process = context.Process(target=worker, args=(args, snapshots, actions, stop, cleanup))
    process.start()
    phase = 0
    saw_hand = False
    initial = None
    report = {}
    deadline = time.monotonic() + 40
    try:
        while phase < 8 and time.monotonic() < deadline:
            try:
                snap = snapshots.get(timeout=2)
            except queue.Empty:
                assert process.is_alive(), "Render process exited unexpectedly"
                continue
            assert snap["scene_rgb"].shape == (480, 640, 3)
            assert snap["scene_rgb"].std() > 10
            if phase == 0:
                initial = snap["scene_rgb"]
                actions.put({"type": "source", "source": "replay"})
                phase = 1
            elif phase == 1 and snap["error"]:
                report["input_failure_preserves_scene"] = True
                actions.put({"type": "configure", "brush_radius_m": .03, "demo_depth_m": .01})
                actions.put({"type": "dent"})
                actions.put({"type": "pause"})
                phase = 2
            elif phase == 2 and not snap["running"] and snap["changed_vertices"]:
                assert not np.array_equal(initial, snap["scene_rgb"])
                assert snap["brush_radius_m"] == .03
                report["demo_parameters_dent_and_pause"] = True
                actions.put({"type": "reset"})
                phase = 3
            elif phase == 3 and not snap["changed_vertices"]:
                assert not snap["running"]
                report["reset_while_paused"] = True
                actions.put({"type": "turn", "angle": .3})
                actions.put({"type": "orbit", "dx": 30, "dy": 10})
                actions.put({"type": "zoom", "factor": .9})
                actions.put({"type": "source", "source": "demo"})
                phase = 4
            elif phase == 4 and snap["source"] == "demo" and not snap["error"]:
                report["source_recovery_and_observation"] = True
                actions.put({"type": "start"})
                actions.put({"type": "source", "source": "webcam"})
                phase = 5
            elif phase == 5 and snap["source"] == "webcam":
                if snap["hands"]:
                    saw_hand = True
                    assert len(snap["hands"][0]["points_3d_m"]) == 21
                    assert snap["camera_bgr"].shape == (120, 160, 3)
                if snap["error"]:
                    assert saw_hand, "Synthetic hand/preview was never published"
                    assert not snap["hands"] and snap["camera_bgr"] is None
                    report["synthetic_input_hand_preview_and_end_cleanup"] = True
                    actions.put({"type": "source", "source": "demo"})
                    phase = 6
            elif phase == 6 and snap["source"] == "demo" and not snap["error"]:
                actions.put({"type": "quit"})
                phase = 8
        assert phase == 8, f"Timed out at phase {phase}"
    finally:
        stop.set()
        process.join(5)
        if process.is_alive():
            process.terminate()
            process.join(3)
        assert process.exitcode == 0, f"Unexpected backend exit {process.exitcode}"
    report.update(cleanup.get(timeout=2))
    assert report["resources_closed"] and report["resource_count"] == 2
    report["backend_exits_cleanly"] = True
    # 使用实际工作台 worker 再验证一次带大图像的正常退出，避免 feeder
    # 提前终止后留下半个图像包、使 UI Queue.get 永久等待的回归。
    from workbench_app import backend
    production_stop = context.Event()
    production_snapshots, production_actions = context.Queue(2), context.Queue(128)
    production = context.Process(target=backend, args=(args, production_snapshots, production_actions, production_stop))
    production.start()
    try:
        first = production_snapshots.get(timeout=15)
        assert first["scene_kind"] == "clay"
        production_actions.put({"type": "quit"})
        deadline = time.monotonic() + 10
        while production.is_alive() and time.monotonic() < deadline:
            try:
                production_snapshots.get(timeout=.2)
            except queue.Empty:
                pass
            production.join(.05)
        assert not production.is_alive() and production.exitcode == 0
        report["production_worker_flushes_frames_and_exits"] = True
    finally:
        production_stop.set()
        if production.is_alive():
            production.terminate()
            production.join(3)
    print(json.dumps(report, indent=2))
    for channel in (snapshots, actions, cleanup, production_snapshots, production_actions):
        channel.close()


if __name__ == "__main__":
    main()
