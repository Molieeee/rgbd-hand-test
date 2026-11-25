"""泥坯工作台与 RGB-D 手部三维交互。

默认启动 Tk 工作台和独立 Open3D 泥坯场景，不访问摄像头。
--source webcam/orbbec/replay 将真实输入骨架显示到固定泥坯旁；
--no-ui 仅启动三维窗口；--scene cube 保留原立方体抓取演示。

数据流：

    相机 (Frame) -> MediaPipe 关键点 -> 3D 坐标 (m)
                 -> EMA 平滑 -> 手势状态机 -> 抓取/移动/旋转
                 -> Open3D 虚拟手 + 可抓取立方体

没有深度相机时用普通摄像头：3D 坐标来自 MediaPipe world landmarks
（公制、相机轴对齐），手腕固定在相机前方 0.5 m 处。
接上 Orbbec 后，同一套上层代码改用 depth + 内参反投影。

运行：
    python main.py
    python main.py --source webcam
    python main.py --scene cube --source webcam
    python main.py --source replay --replay-dir data/demo
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from capture import Recorder, ReplaySource, WebcamSource
from interaction import FallingObject, GestureFSM, Manipulator
from perception import DepthHand3D, EMAFilter, HandLandmarkerWrapper, WebcamHand3D
from perception.hand3d import HAND_CONNECTIONS, INDEX_TIP, THUMB_TIP
from render import Scene

ESC = 27
TIP_SET = (THUMB_TIP, INDEX_TIP)

OBJECT_SIZE = 0.05          # 立方体边长 (m)
TABLE_TOP = 0.18            # 桌面上表面在相机系中的 y (m，向下为正)
TABLE_SIZE = (0.9, 0.60)    # 桌面 (x, z) 尺寸 (m)
TABLE_CENTER_Z = 0.55       # 桌面中心的相机 z (m)
# 方块初始悬停在桌面上方、虚拟手自然捏合的高度（≈ 手部实测值）。
# Webcam 模式下捏合点的 z 比手腕靠前约 0.08 m，y 比手腕高约 0.08 m，
# 把方块放在这里，捏合点基本就在方块中心，不用“够”得很远。
OBJECT_CENTER = (0.0, 0.05, 0.42)


def build_source(args):
    if args.source == "webcam":
        return WebcamSource(args.camera, args.width, args.height)
    if args.source == "orbbec":
        from capture.orbbec_source import OrbbecSource

        return OrbbecSource(width=args.width, height=args.height)
    if args.source == "replay":
        if not args.replay_dir:
            raise SystemExit("--source replay 需要 --replay-dir")
        return ReplaySource(args.replay_dir, loop=args.loop)
    raise SystemExit(f"未知数据源: {args.source}")


def draw_hand_2d(image: np.ndarray, landmarks, pinching: bool, mirror: bool = False) -> None:
    h, w = image.shape[:2]
    pts = [
        (int((1.0 - lm.x if mirror else lm.x) * w), int(lm.y * h))
        for lm in landmarks
    ]

    line_color = (70, 220, 120) if pinching else (200, 200, 200)
    for a, b in HAND_CONNECTIONS:
        cv2.line(image, pts[a], pts[b], line_color, 2, cv2.LINE_AA)

    for i, p in enumerate(pts):
        if pinching and i in TIP_SET:
            color = (60, 240, 90)
            radius = 7
        elif i == 0:
            color = (40, 160, 250)
            radius = 6
        else:
            color = (240, 180, 60)
            radius = 4
        cv2.circle(image, p, radius, color, -1, cv2.LINE_AA)


def initial_object_T(position) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = np.asarray(position, dtype=np.float64)
    return T


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="泥坯工作台（默认无相机）；--scene cube 使用原抓取演示")
    p.add_argument("--source", choices=["demo", "webcam", "orbbec", "replay"], default=None,
                   help="泥坯默认 demo；原立方体演示默认 webcam")
    p.add_argument("--scene", choices=["clay", "cube"], default="clay")
    p.add_argument("--no-ui", action="store_true", help="仅显示独立 Open3D 窗口")
    p.add_argument("--camera", type=int, default=0, help="webcam 设备号")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--model", type=str, default=None, help="hand_landmarker.task 路径")
    p.add_argument("--record", type=str, default=None, help="把帧录到该目录")
    p.add_argument("--replay-dir", type=str, default=None)
    p.add_argument("--loop", action="store_true", help="回放循环")
    p.add_argument("--ema", type=float, default=0.7, help="EMA 系数，越大越平滑")
    p.add_argument("--grab-radius", type=float, default=0.12, help="抓取判定半径 (m)")
    p.add_argument("--object-pos", type=float, nargs=3, default=None,
                   help="物体初始位置（相机坐标，米）；默认悬停在桌面上方")
    p.add_argument("--table-top", type=float, default=TABLE_TOP,
                   help="桌面上表面在相机系中的 y (m，向下为正)")
    p.add_argument("--view", choices=["first", "camera"], default="first",
                   help="first: 第一人称（手伸进屏幕）；camera: 摄像头视角")
    p.add_argument("--eye-distance", type=float, default=0.50,
                   help="第一人称视角下眼睛到物体的水平距离 (m)")
    p.add_argument("--eye-height", type=float, default=0.30,
                   help="第一人称视角下眼睛高出物体的距离 (m)")
    p.add_argument("--no-mirror", action="store_true",
                   help="摄像头预览不做自拍镜像（默认与 3D 视角保持一致而镜像）")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.source is None:
        args.source = "demo" if args.scene == "clay" else "webcam"
    if args.scene == "clay":
        if args.no_ui:
            from clay_runtime import run_clay
            return run_clay(args)
        from workbench_app import launch
        return launch(args)
    if args.source == "demo":
        raise SystemExit("立方体模式请使用 webcam、orbbec 或 replay 数据源")

    first_person = args.view == "first"
    # 第一人称 3D 视角下，左边就是用户身体的左边；预览也跟着镜像，
    # 两个窗口才不会左右相反。第三人称视角则保持摄像头原图。
    mirror = first_person and not args.no_mirror

    # 方块默认悬停在桌面上方、虚拟手自然停留的位置（见 OBJECT_CENTER）。
    # 第一次抓住之前不受重力，松手后才会掉到桌面上。
    object_pos = list(args.object_pos) if args.object_pos is not None else list(OBJECT_CENTER)

    source = build_source(args)
    recorder = Recorder(args.record) if args.record else None
    landmarker = HandLandmarkerWrapper(model_path=args.model)
    scene = Scene(
        width=args.width,
        height=args.height,
        focus_distance=object_pos[2],
        first_person=first_person,
        eye_distance=args.eye_distance,
        eye_height=args.eye_height,
        look_at_y=object_pos[1],
        object_size=OBJECT_SIZE,
        table_top=args.table_top,
        table_size=TABLE_SIZE,
        table_center_z=TABLE_CENTER_Z,
    )

    webcam_provider = WebcamHand3D(anchor_z=0.5, aspect=args.width / args.height)
    depth_provider: DepthHand3D | None = None

    filters = [EMAFilter(alpha=args.ema) for _ in range(2)]
    fsms = [GestureFSM() for _ in range(2)]
    manipulator = Manipulator(grab_radius=args.grab_radius)
    falling = FallingObject(
        table_top=args.table_top,
        table_x=(-TABLE_SIZE[0] / 2.0, TABLE_SIZE[0] / 2.0),
        table_z=(TABLE_CENTER_Z - TABLE_SIZE[1] / 2.0,
                 TABLE_CENTER_Z + TABLE_SIZE[1] / 2.0),
    )

    object_T = initial_object_T(object_pos)
    object_T0 = object_T.copy()

    cv2.namedWindow("RGBD Hand - camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RGBD Hand - camera", min(args.width, 960), min(args.height, 540))

    fps = 0.0
    last_time = time.perf_counter()
    frame_count = 0
    replay_dt = 1.0 / max(getattr(source, "fps", 30), 1) if args.source == "replay" else 0.0

    try:
        while True:
            loop_start = time.perf_counter()
            frame = source.read()
            if frame is None:
                print("数据源结束")
                break

            if recorder is not None:
                recorder.write(frame)

            now = time.perf_counter()
            dt = max(now - last_time, 1e-6)
            last_time = now
            inst_fps = 1.0 / dt
            fps = inst_fps if fps == 0.0 else 0.9 * fps + 0.1 * inst_fps

            webcam_provider.aspect = frame.color.shape[1] / frame.color.shape[0]
            result = landmarker.detect(frame.color, frame.timestamp_ms)

            if frame.has_depth:
                if depth_provider is None:
                    depth_provider = DepthHand3D()
                    print("检测到深度数据，切换到 DepthHand3D（反投影 + 内参）")
                provider = depth_provider
            else:
                provider = webcam_provider

            hands = []
            for i, (landmarks, world) in enumerate(
                zip(result.hand_landmarks, result.hand_world_landmarks)
            ):
                if i >= len(filters):
                    break
                pts = provider.to_camera3d(landmarks, world, frame.depth, frame.K)
                pts = filters[i].update(pts)
                state = fsms[i].update(pts)
                hands.append((pts, state))

            # 消失的手要清掉平滑器和状态，避免再次出现时跳变
            for i in range(len(filters)):
                if i >= len(hands):
                    filters[i].reset()
                    fsms[i].reset()

            primary = None
            for pts, state in hands:
                if state.pinching:
                    primary = (pts, state)
                    break
            if primary is None and hands:
                primary = hands[0]

            if primary is not None:
                _, state = primary
                object_T, grabbed = manipulator.update(
                    state.hand_T, state.pinching, object_T, dt
                )
            else:
                # MediaPipe 偶尔会丢一两帧。短暂丢失时保持抓取、物体停在
                # 最后的位置，避免“一直捏着却突然掉下去”。
                object_T, grabbed = manipulator.update_missing(object_T, dt)

            # 捏合点到方块中心的距离：HUD 里显示，方块进入抓取范围时会变色。
            pinch_dist = None
            if primary is not None and primary[1].pinch_point is not None:
                pinch_dist = float(
                    np.linalg.norm(primary[1].pinch_point - object_T[:3, 3])
                )
            near = (not grabbed) and pinch_dist is not None and pinch_dist < args.grab_radius

            display = cv2.flip(frame.color, 1) if mirror else frame.color.copy()
            for landmarks, (_, state) in zip(result.hand_landmarks, hands):
                draw_hand_2d(display, landmarks, state.pinching, mirror=mirror)

            # 松手后重力接管，落到桌面上停住；抓取中速度清零。
            object_T = falling.update(object_T, grabbed, dt, OBJECT_SIZE / 2.0)

            scene.update_hands([pts for pts, _ in hands], [s.pinching for _, s in hands])
            scene.update_object(object_T, grabbed, near)

            pinch_ratio = primary[1].pinch_ratio if primary is not None else 0.0
            mode = "GRAB" if grabbed else ("PINCH" if primary is not None and primary[1].pinching else "IDLE")
            dist_text = f"{pinch_dist:.3f}" if pinch_dist is not None else " --  "
            hud = [
                f"source: {args.source}   depth: {'ON' if frame.has_depth else 'OFF'}",
                f"fps: {fps:5.1f}   hands: {len(hands)}   pinch_ratio: {pinch_ratio:.2f}",
                f"state: {mode}   dist: {dist_text} m / {args.grab_radius:.2f}",
                f"object: ({object_T[0, 3]:.2f}, {object_T[1, 3]:.2f}, {object_T[2, 3]:.2f})"
                f"   gravity: {'on' if falling.active else 'off'}",
                "q/ESC: quit   r: reset object",
            ]
            for k, text in enumerate(hud):
                cv2.putText(display, text, (12, 28 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(display, text, (12, 28 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("RGBD Hand - camera", display)
            if not scene.render():
                print("Open3D 窗口已关闭")
                break

            frame_count += 1
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ESC):
                break
            if key == ord("r"):
                object_T = object_T0.copy()
                manipulator.reset()
                falling.reset()
                for fsm in fsms:
                    fsm.reset()
                print("物体位姿已重置")

            if replay_dt:
                remain = replay_dt - (time.perf_counter() - loop_start)
                if remain > 0:
                    time.sleep(remain)
    finally:
        if recorder is not None:
            recorder.close()
            print(f"已录制 {recorder.index} 帧到 {recorder.root}")
        source.close()
        landmarker.close()
        scene.close()
        cv2.destroyAllWindows()

    print(f"共处理 {frame_count} 帧")
    return 0


if __name__ == "__main__":
    sys.exit(main())
