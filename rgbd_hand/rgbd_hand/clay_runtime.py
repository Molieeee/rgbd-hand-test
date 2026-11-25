"""泥坯后端：在创建 Open3D 窗口的进程内维护网格与输入源。"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from clay_demo import demo_dent
from render import ClayMesh, Scene
from render.studio import clay_studio_pose


def preview_frame(image, previous=None):
    """最小化窗口可能返回空帧或纯背景，保留上一张有效画面。"""
    image = np.asarray(image)
    valid = (image.ndim == 3 and image.shape[0] > 16 and image.shape[1] > 16
             and image.shape[2] == 3 and np.isfinite(image[::16, ::16]).all()
             and np.max(np.std(image[::16, ::16], axis=(0, 1))) > .003)
    return (np.clip(image * 255, 0, 255).astype(np.uint8) if valid else previous), not valid


class ClaySession:
    """UI 与独立窗口共享的状态；固定凹坑仅用于显示联调。"""
    def __init__(self, table_top=.18):
        self.clay = ClayMesh(edge_rounding=.004)
        pose = clay_studio_pose(self.clay, table_top)
        self.clay.set_pose(pose)
        self.initial_pose = pose.copy()
        self.running = True
        self.radius_m = .04875
        self.depth_m = .022
        self.skeleton = True
        self.grid = False
        self.message = "拖动观察泥坯，或点击下方「演示凹坑」体验形状变化。"

    def action(self, action, first_person=True):
        kind = action.get("type")
        if kind in ("start", "pause"):
            self.running = kind == "start"
        elif kind == "configure":
            radius = float(action.get("brush_radius_m", self.radius_m))
            depth = float(action.get("demo_depth_m", self.depth_m))
            if not np.isfinite([radius, depth]).all():
                raise ValueError("演示参数必须为有限数值")
            self.radius_m = float(np.clip(radius, .01, .07))
            self.depth_m = float(np.clip(depth, .001, .03))
            self.skeleton = bool(action.get("skeleton", self.skeleton))
            self.grid = bool(action.get("grid", self.grid))
        elif kind == "dent":
            if not self.running:
                self.message = "已暂停；开始后才能放置演示凹坑"
                return
            self.clay.set_vertices(demo_dent(
                self.clay, -1 if first_person else 1,
                brush_radius=self.radius_m, depth=self.depth_m))
            self.message = "已放置固定演示凹坑；重复点击不累加"
        elif kind == "reset":
            self.clay.reset()
            self.clay.set_pose(self.initial_pose)
            self.message = "已恢复初始圆柱和位姿"
        elif kind == "turn":
            angle = float(action.get("angle", .15))
            if not np.isfinite(angle):
                raise ValueError("旋转角必须为有限数值")
            pose = self.clay.pose_scene
            c, s = np.cos(angle), np.sin(angle)
            pose[:3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) @ pose[:3, :3]
            self.clay.set_pose(pose)

    def snapshot(self):
        center = self.clay.pose_scene[:3, 3] * [1, -1, -1]
        changed = np.linalg.norm(self.clay.vertices_local - self.clay.initial_vertices, axis=1)
        return {"scene_kind": "clay", "running": self.running,
                "brush_radius_m": self.radius_m, "demo_depth_m": self.depth_m,
                "mesh_version": self.clay.version, "vertex_count": len(changed),
                "changed_vertices": int(np.count_nonzero(changed)),
                "message": self.message, "hit": False, "sculpting": False,
                "object": {"center_m": center.tolist(), "kind": "clay"}}


def run_clay(args, publish=None, poll_actions=None, should_stop=None):
    """单独运行或由工作台子进程调用；publish/poll_actions 均为非阻塞接口。"""
    session = ClaySession(args.table_top)
    scene = source = landmarker = recorder = None
    selected_source = args.source
    error = ""
    hands, camera = [], None
    preview = None
    depth_enabled = False
    frames = 0
    started = last_publish = time.perf_counter()
    filters, fsms = [], []
    provider = depth_provider = None
    next_read = 0.0

    def close_input():
        nonlocal source, landmarker, recorder, hands, camera, depth_enabled
        for resource in (recorder, landmarker, source):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    print(f"输入资源关闭失败：{exc}", flush=True)
        source = landmarker = recorder = None
        hands, camera, depth_enabled = [], None, False

    def open_input(name):
        nonlocal source, landmarker, recorder, filters, fsms, provider, depth_provider
        nonlocal selected_source, error, frames, started, next_read
        close_input()
        selected_source, error = name, ""
        frames, started, next_read = 0, time.perf_counter(), 0.0
        if name == "demo":
            return
        try:
            from main import build_source
            from capture import Recorder
            from perception import EMAFilter, HandLandmarkerWrapper, WebcamHand3D
            from interaction import GestureFSM
            config = SimpleNamespace(**vars(args))
            config.source = name
            source = build_source(config)
            landmarker = HandLandmarkerWrapper(model_path=args.model)
            recorder = Recorder(args.record) if args.record else None
            provider = WebcamHand3D(anchor_z=.5, aspect=args.width / args.height)
            depth_provider = None
            filters = [EMAFilter(alpha=args.ema) for _ in range(2)]
            fsms = [GestureFSM() for _ in range(2)]
        except (Exception, SystemExit) as exc:
            close_input()
            error = f"输入源 {name} 启动失败：{exc}"
            print(error, flush=True)

    try:
        scene = Scene(window_name="Clay Scene | D: demo dent  R: reset  V: view  Q: quit",
                      clay=session.clay, width=args.width, height=args.height,
                      focus_distance=.5, look_at_y=args.table_top - session.clay.height / 2,
                      table_top=args.table_top, first_person=args.view == "first",
                      eye_distance=args.eye_distance, eye_height=args.eye_height)

        def apply(action):
            if action.get("type") == "source":
                name = action.get("source")
                if name not in ("demo", "webcam", "orbbec", "replay"):
                    raise ValueError("未知输入源")
                open_input(name)
            elif action.get("type") == "view":
                scene._frame_view()
            elif action.get("type") == "orbit":
                scene.vis.get_view_control().rotate(float(action.get("dx", 0)), float(action.get("dy", 0)))
            elif action.get("type") == "zoom":
                scene.vis.get_view_control().scale(1.0 if float(action.get("factor", 1)) < 1 else -1.0)
            else:
                session.action(action, scene.first_person)
                scene.set_grid_visible(session.grid)
                scene.update_clay()

        def key(action):
            def callback(vis):
                apply(action)
                return False
            return callback

        for letter, action in (("D", {"type": "dent"}), ("R", {"type": "reset"}),
                               ("V", {"type": "view"}), ("A", {"type": "turn", "angle": -.15}),
                               ("F", {"type": "turn", "angle": .15})):
            scene.vis.register_key_callback(ord(letter), key(action))
        open_input(selected_source)
        print("泥坯场景已启动：D 演示凹坑，R 复位，A/F 转动，V 恢复视角，Q 退出", flush=True)
        while not (should_stop and should_stop()):
            quit_requested = False
            for action in (poll_actions() if poll_actions else []):
                if action.get("type") == "quit":
                    quit_requested = True
                    break
                try:
                    apply(action)
                except Exception as exc:
                    session.message = str(exc)
            if quit_requested:
                break
            now = time.perf_counter()
            if session.running and source is not None and now >= next_read:
                try:
                    frame = source.read()
                    if frame is None:
                        close_input()
                        error = "输入源结束或读取失败；泥坯仍可观察。可重新选择输入源。"
                    else:
                        if recorder is not None:
                            recorder.write(frame)
                        result = landmarker.detect(frame.color, frame.timestamp_ms)
                        depth_enabled = frame.has_depth
                        if frame.has_depth:
                            from perception import DepthHand3D
                            if depth_provider is None:
                                depth_provider = DepthHand3D()
                            current_provider = depth_provider
                        else:
                            provider.aspect = frame.color.shape[1] / frame.color.shape[0]
                            current_provider = provider
                        hands = []
                        for i, (landmarks, world) in enumerate(zip(
                                result.hand_landmarks, result.hand_world_landmarks)):
                            if i >= len(filters):
                                break
                            points = filters[i].update(current_provider.to_camera3d(
                                landmarks, world, frame.depth, frame.K))
                            state = fsms[i].update(points)
                            hands.append({"slot": i, "points_3d_m": points,
                                          "pinching": state.pinching, "open_palm": state.open_palm,
                                          "fist": state.fist, "pinch_ratio": state.pinch_ratio})
                        for i in range(len(hands), len(filters)):
                            filters[i].reset()
                            fsms[i].reset()
                        from main import draw_hand_2d
                        mirror = args.view == "first" and not args.no_mirror
                        camera = frame.color[:, ::-1].copy() if mirror else frame.color.copy()
                        for landmarks, hand in zip(result.hand_landmarks, hands):
                            draw_hand_2d(camera, landmarks, hand["pinching"], mirror)
                        frames += 1
                        if selected_source == "replay":
                            next_read = now + 1 / max(getattr(source, "fps", 30), 1)
                except Exception as exc:
                    close_input()
                    error = f"输入处理失败：{exc}"
            scene.update_hands([h["points_3d_m"] for h in hands] if session.skeleton else [],
                               [h["pinching"] for h in hands] if session.skeleton else [])
            if not scene.render():
                break
            if publish and now - last_publish >= .1:
                snap = session.snapshot()
                image = np.asarray(scene.vis.capture_screen_float_buffer(do_render=False))
                preview, view_suspended = preview_frame(image, preview)
                snap.update(source=selected_source, hands=hands, camera_bgr=camera,
                            depth_enabled=depth_enabled, error=error,
                            fps=frames / max(now - started, .001),
                            scene_rgb=preview, view_suspended=view_suspended)
                publish(snap)
                last_publish = now
            time.sleep(.005)
    finally:
        try:
            close_input()
        finally:
            if scene is not None:
                scene.close()
    return 0
