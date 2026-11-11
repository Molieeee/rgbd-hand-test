"""Orbbec Gemini 2 / Astra 深度相机后端。

依赖官方 pyorbbecsdk2（从 GitHub release 安装，不要用 PyPI 上的
`pyorbbecsdk`，那个包是误标的 macOS 版本）。

本文件在拿到相机前无法实机验证，API 名称已对照 pyorbbecsdk2 2.1.2 检查过。
拿到相机后，如果 SDK 版本不同，优先对照官方 example 调整：
    python -c "import pyorbbecsdk; print(dir(pyorbbecsdk))"
"""
from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from .base import Frame


def _as_array(data, dtype):
    """get_data() 在不同版本里可能返回 ndarray 或 bytes/memoryview。"""
    if isinstance(data, np.ndarray):
        return data.astype(dtype, copy=False)
    return np.frombuffer(data, dtype=dtype)


class OrbbecSource:
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        align_mode: str = "hw",
    ):
        try:
            import pyorbbecsdk as ob
        except ImportError as exc:
            raise RuntimeError(
                "未安装 pyorbbecsdk2。请从 https://github.com/orbbec/pyorbbecsdk/releases "
                "下载对应 Python 版本的 Windows wheel 安装。"
            ) from exc

        self.ob = ob
        self._closed = False
        self._start = time.time()

        ctx = ob.Context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise RuntimeError(
                "没有检测到 Orbbec 相机。检查 USB 连接、驱动，"
                "或确认设备没有被其他程序占用。"
            )

        self.pipeline = ob.Pipeline()
        config = ob.Config()
        config.enable_video_stream(
            ob.OBSensorType.COLOR_SENSOR, width, height, fps, ob.OBFormat.RGB
        )
        config.enable_video_stream(
            ob.OBSensorType.DEPTH_SENSOR, width, height, fps, ob.OBFormat.Y16
        )

        # 硬件 D2C 对齐：depth 直接落到 color 的像素网格上。
        mode = ob.OBAlignMode.HW_MODE if align_mode == "hw" else ob.OBAlignMode.SW_MODE
        try:
            config.set_align_mode(mode)
        except Exception:
            # 某些型号/固件不支持硬件对齐，退回软件对齐。
            config.set_align_mode(ob.OBAlignMode.SW_MODE)

        self.pipeline.start(config)

    def read(self) -> Optional[Frame]:
        frames = self.pipeline.wait_for_frames(1000)
        if frames is None:
            return None

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame is None or depth_frame is None:
            return None

        h = color_frame.get_height()
        w = color_frame.get_width()
        color = _as_array(color_frame.get_data(), np.uint8).reshape(h, w, 3)
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

        dh = depth_frame.get_height()
        dw = depth_frame.get_width()
        depth = _as_array(depth_frame.get_data(), np.uint16).reshape(dh, dw)

        # SDK 的 depth 单位可能是 0.1mm/1mm，统一换算成毫米。
        scale = float(depth_frame.get_depth_scale())
        if abs(scale - 1.0) > 1e-6:
            depth = np.clip(depth.astype(np.float32) * scale, 0, 65535).astype(np.uint16)

        if (dh, dw) != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

        profile = color_frame.get_stream_profile().as_video_stream_profile()
        intr = profile.get_intrinsic()
        K = np.array(
            [[intr.fx, 0.0, intr.cx], [0.0, intr.fy, intr.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        ts = int((time.time() - self._start) * 1000)
        return Frame(color=color, depth=depth, K=K, timestamp_ms=ts)

    def close(self) -> None:
        if not self._closed:
            try:
                self.pipeline.stop()
            finally:
                self._closed = True
