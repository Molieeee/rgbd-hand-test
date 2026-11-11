"""把 2D 手部关键点变成相机坐标系下的 3D 坐标（单位：米）。

两种实现，接口完全一致：

- WebcamHand3D : 没有深度时使用。用 MediaPipe 的 world landmarks（公制、
                 相机坐标轴对齐、原点在手部中心），再把手腕平移到固定锚点。
- DepthHand3D  : 有深度时使用。对每个关键点取邻域有效深度中值，用内参反投影；
                 深度缺失的关节用 world landmarks 做平移对齐后补全。

坐标约定（相机系）：+X 右，+Y 下，+Z 朝前（OpenCV / Orbbec 约定）。
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# MediaPipe 手部 21 点的骨架连接
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm base
]

WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
PINKY_MCP = 17


def _world_array(world_landmarks) -> np.ndarray:
    return np.array([[p.x, p.y, p.z] for p in world_landmarks], dtype=np.float64)


class Hand3DProvider:
    def to_camera3d(self, hand_landmarks, world_landmarks, depth, K) -> np.ndarray:
        raise NotImplementedError


class WebcamHand3D(Hand3DProvider):
    """无深度模式。

    world landmarks 只提供手的形状和姿态（它们以手为中心，平移量已被去掉），
    平移用 2D 图像坐标估计：假设手在固定深度 anchor_z 处，按近似 FOV 反投影。
    这样手会跟着人在画面里的位置移动，而不是永远钉在画面中心。
    """

    def __init__(self, anchor_z: float = 0.5, hfov_deg: float = 60.0, aspect: float = 16.0 / 9.0):
        self.anchor_z = float(anchor_z)
        self.hfov_deg = float(hfov_deg)
        self.aspect = float(aspect)

    def _wrist_target(self, wrist_lm) -> np.ndarray:
        tan_h = np.tan(np.radians(self.hfov_deg) / 2.0)
        tan_v = tan_h / self.aspect
        x = (wrist_lm.x - 0.5) * 2.0 * tan_h * self.anchor_z
        y = (wrist_lm.y - 0.5) * 2.0 * tan_v * self.anchor_z
        return np.array([x, y, self.anchor_z], dtype=np.float64)

    def to_camera3d(self, hand_landmarks, world_landmarks, depth=None, K=None) -> np.ndarray:
        pts = _world_array(world_landmarks)
        # world landmarks 以手为中心；把手腕平移到由图像位置估计出的相机坐标。
        return pts + (self._wrist_target(hand_landmarks[WRIST]) - pts[WRIST])


class DepthHand3D(Hand3DProvider):
    """深度模式：邻域中值深度 + 内参反投影。"""

    def __init__(
        self,
        median_ksize: int = 5,
        min_depth_m: float = 0.1,
        max_depth_m: float = 3.0,
        fallback_anchor: Sequence[float] = (0.0, 0.0, 0.5),
    ):
        self.median_ksize = int(median_ksize)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.fallback_anchor = np.asarray(fallback_anchor, dtype=np.float64)

    def _sample_depth_m(self, depth_mm: np.ndarray, u: float, v: float) -> float:
        h, w = depth_mm.shape[:2]
        k = self.median_ksize
        x = int(round(u))
        y = int(round(v))
        x0, x1 = max(0, x - k // 2), min(w, x + k // 2 + 1)
        y0, y1 = max(0, y - k // 2), min(h, y + k // 2 + 1)
        patch = depth_mm[y0:y1, x0:x1]
        valid = patch[patch > 0]
        if valid.size == 0:
            return 0.0
        return float(np.median(valid)) / 1000.0

    def to_camera3d(self, hand_landmarks, world_landmarks, depth, K) -> np.ndarray:
        if depth is None or K is None:
            return WebcamHand3D(anchor_z=float(self.fallback_anchor[2])).to_camera3d(
                hand_landmarks, world_landmarks
            )

        h, w = depth.shape[:2]
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        pts = np.full((21, 3), np.nan, dtype=np.float64)
        for i, lm in enumerate(hand_landmarks):
            u = lm.x * w
            v = lm.y * h
            z = self._sample_depth_m(depth, u, v)
            if not (self.min_depth_m <= z <= self.max_depth_m):
                continue
            pts[i] = [(u - cx) * z / fx, (v - cy) * z / fy, z]

        valid = ~np.isnan(pts[:, 2])
        world = _world_array(world_landmarks)

        if valid.any():
            # world landmarks 与深度点之间只差一个平移：用有效点求偏移。
            offset = np.mean(pts[valid] - world[valid], axis=0)
            filled = world + offset
        else:
            # 整只手都没有有效深度：退回到“图像位置 + 固定深度”的估计。
            return WebcamHand3D(anchor_z=float(self.fallback_anchor[2])).to_camera3d(
                hand_landmarks, world_landmarks
            )

        filled[valid] = pts[valid]
        return filled
