"""普通 RGB 摄像头后端。

没有深度，也没有内参。为了让上层代码完全一致，Frame 里 depth/K 留空，
3D 坐标由 perception.hand3d.WebcamHand3D 用 MediaPipe 的 world landmarks 提供。
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from .base import Frame


class WebcamSource:
    def __init__(self, device: int = 0, width: int = 1280, height: int = 720):
        self.cap = cv2.VideoCapture(device)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 device={device}，试试换 --camera 1")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._start = time.time()

    def read(self) -> Frame | None:
        ok, color = self.cap.read()
        if not ok:
            return None
        ts = int((time.time() - self._start) * 1000)
        return Frame(color=color, depth=None, K=None, timestamp_ms=ts)

    def close(self) -> None:
        self.cap.release()
