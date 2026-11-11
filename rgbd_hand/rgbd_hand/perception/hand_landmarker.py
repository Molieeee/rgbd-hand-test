"""MediaPipe HandLandmarker 封装。

用 Tasks API（不是旧的 mp.solutions），输出：
- hand_landmarks:       21 个归一化图像坐标 (x, y, z)
- hand_world_landmarks: 21 个公制手部坐标 (米)，原点在手部几何中心

注意：Windows 上 MediaPipe 无法打开含中文的路径，所以必须把模型读成
bytes 后通过 model_asset_buffer 传入。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "assets" / "hand_landmarker.task"


class HandLandmarkerWrapper:
    def __init__(
        self,
        model_path: str | Path | None = None,
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not model_path.exists():
            raise FileNotFoundError(f"找不到手部模型: {model_path}")

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model_path.read_bytes()),
            num_hands=num_hands,
            running_mode=vision.RunningMode.VIDEO,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def detect(self, color_bgr, timestamp_ms: int):
        # detect_for_video 要求时间戳严格递增。
        ts = max(int(timestamp_ms), self._last_ts + 1)
        self._last_ts = ts
        rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self.landmarker.detect_for_video(image, ts)

    def close(self) -> None:
        self.landmarker.close()
