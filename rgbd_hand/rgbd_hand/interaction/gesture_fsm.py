"""基于 3D 关节的手势状态机。

所有判定都使用米制 3D 坐标，并且用“手部尺寸”做归一化，
所以换深度相机后阈值基本不用改。

手势：
- pinch : 拇指尖与食指尖靠近（比值 < PINCH_ON_RATIO）
- fist  : 四个指尖都收向手腕
- palm  : 四个指尖都明显伸展
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from perception.hand3d import INDEX_MCP, INDEX_TIP, MIDDLE_MCP, PINKY_MCP, THUMB_TIP, WRIST

# 归一化阈值（除以 wrist -> middle MCP 的距离）
PINCH_ON_RATIO = 0.50
# 松开阈值要比捏合阈值高很多：world landmarks 的深度（z）有噪声，
# 明明捏着的时候 |拇指尖-食指尖| 的三维距离也可能偏大。松开要求
# “明显张开”，并且连续多帧，避免捏着捏着被误判成松开。
PINCH_OFF_RATIO = 0.85
FIST_RATIO = 1.10
PALM_RATIO = 1.50
STABLE_FRAMES = 2
RELEASE_FRAMES = 10

TIP_IDS = [8, 12, 16, 20]


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def _hand_frame(pts: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """手掌坐标系，返回 4x4 齐次矩阵。

    y 轴：手腕 -> 中指 MCP（沿手掌方向）
    z 轴：掌面法向
    x 轴：y × z，构成右手系
    """
    y = _normalize(pts[MIDDLE_MCP] - pts[WRIST])
    normal = np.cross(pts[INDEX_MCP] - pts[WRIST], pts[PINKY_MCP] - pts[WRIST])
    if np.linalg.norm(normal) < 1e-9:
        normal = np.array([0.0, 0.0, 1.0])
    z = _normalize(normal)
    x = _normalize(np.cross(y, z))
    z = _normalize(np.cross(x, y))

    R = np.column_stack([x, y, z])
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1.0

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = origin
    return T


@dataclass
class GestureState:
    pinching: bool = False
    fist: bool = False
    open_palm: bool = False
    pinch_ratio: float = 0.0
    extended_ratios: tuple[float, ...] = ()
    pinch_point: Optional[np.ndarray] = None
    hand_T: Optional[np.ndarray] = None


class GestureFSM:
    def __init__(
        self,
        pinch_on: float = PINCH_ON_RATIO,
        pinch_off: float = PINCH_OFF_RATIO,
        fist_ratio: float = FIST_RATIO,
        palm_ratio: float = PALM_RATIO,
        stable_frames: int = STABLE_FRAMES,
        release_frames: int = RELEASE_FRAMES,
    ):
        self.pinch_on = pinch_on
        self.pinch_off = pinch_off
        self.fist_ratio = fist_ratio
        self.palm_ratio = palm_ratio
        self.stable_frames = stable_frames
        self.release_frames = release_frames

        self.pinching = False
        self._pinch_counter = 0
        self._release_counter = 0
        self._fist_counter = 0
        self._palm_counter = 0

    def reset(self) -> None:
        self.pinching = False
        self._pinch_counter = 0
        self._release_counter = 0
        self._fist_counter = 0
        self._palm_counter = 0

    def update(self, pts: np.ndarray) -> GestureState:
        pts = np.asarray(pts, dtype=np.float64)
        hand_size = float(np.linalg.norm(pts[MIDDLE_MCP] - pts[WRIST])) + 1e-6

        pinch_ratio = float(np.linalg.norm(pts[THUMB_TIP] - pts[INDEX_TIP])) / hand_size
        pinch_point = 0.5 * (pts[THUMB_TIP] + pts[INDEX_TIP])
        extended = tuple(
            float(np.linalg.norm(pts[t] - pts[WRIST])) / hand_size for t in TIP_IDS
        )

        # 捏合：带迟滞的开关。进入捏合只要连续 3 帧；退出捏合要求
        # 明显张开（> pinch_off）并持续 release_frames 帧，防止抖动掉手。
        if not self.pinching:
            if pinch_ratio < self.pinch_on:
                self._pinch_counter += 1
                if self._pinch_counter >= self.stable_frames:
                    self.pinching = True
                    self._pinch_counter = 0
                    self._release_counter = 0
            else:
                self._pinch_counter = 0
        else:
            if pinch_ratio > self.pinch_off:
                self._release_counter += 1
                if self._release_counter >= self.release_frames:
                    self.pinching = False
                    self._release_counter = 0
                    self._pinch_counter = 0
            else:
                self._release_counter = 0

        # 握拳 / 张掌：同样要求连续若干帧，减少抖动
        raw_fist = max(extended) < self.fist_ratio
        raw_palm = min(extended) > self.palm_ratio
        self._fist_counter = self._fist_counter + 1 if raw_fist else 0
        self._palm_counter = self._palm_counter + 1 if raw_palm else 0
        fist = self._fist_counter >= self.stable_frames
        palm = self._palm_counter >= self.stable_frames

        origin = pinch_point if self.pinching else np.mean(
            [pts[WRIST], pts[INDEX_MCP], pts[PINKY_MCP]], axis=0
        )

        return GestureState(
            pinching=self.pinching,
            fist=fist,
            open_palm=palm,
            pinch_ratio=pinch_ratio,
            extended_ratios=extended,
            pinch_point=pinch_point,
            hand_T=_hand_frame(pts, origin),
        )
