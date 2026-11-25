"""简单的桌面/重力物理。

只做课程演示需要的部分：物体松手后自由落体，落到桌面上停住。
坐标系是相机系（+x 右、+y 下、+z 前，单位米），所以重力方向是 +y。

物体的姿态在落地前保持不变；落点高度按旋转后立方体的最低点计算，
因此斜着落地也不会插进桌面。
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


_CUBE_SIGNS = np.array(
    [
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)


class FallingObject:
    def __init__(
        self,
        table_top: float = 0.20,
        table_x: Tuple[float, float] = (-0.45, 0.45),
        table_z: Tuple[float, float] = (0.25, 0.85),
        floor_top: float = 0.85,
        gravity: float = 9.81,
        restitution: float = 0.15,
        contact_epsilon: float = 1e-3,
    ):
        self.table_top = float(table_top)
        self.table_x = tuple(map(float, table_x))
        self.table_z = tuple(map(float, table_z))
        self.floor_top = float(floor_top)
        self.gravity = float(gravity)
        self.restitution = float(restitution)
        # Keep a tiny visible gap at contact. Exact coplanar faces can show
        # z-fighting in Open3D and look like the cube is slightly sunk.
        self.contact_epsilon = max(float(contact_epsilon), 0.0)
        self.velocity = np.zeros(3, dtype=np.float64)
        # Webcam 模式下手的可达高度有限，方块先悬停在桌面之上；
        # 第一次被抓住之后才开始受重力，松手就会掉到桌面上。
        self.active = False

    def reset(self) -> None:
        self.velocity[:] = 0.0
        self.active = False

    def _support_radius(self, R: np.ndarray, half_extent: float) -> float:
        """立方体中心到最低点的竖直距离（相机 +y 方向）。"""
        return half_extent * float(np.sum(np.abs(R[1, :])))

    def _ground_height(self, object_T: np.ndarray, half_extent: float) -> float:
        """Return the support height under the cube's lowest horizontal point."""
        R = object_T[:3, :3]
        pos = object_T[:3, 3]
        corners = (_CUBE_SIGNS * half_extent) @ R.T + pos
        lowest_y = float(np.min(corners[:, 1]))
        lowest_corners = corners[
            np.isclose(corners[:, 1], lowest_y, rtol=0.0, atol=1e-9)
        ]
        inside_table = any(
            self.table_x[0] <= x <= self.table_x[1]
            and self.table_z[0] <= z <= self.table_z[1]
            for x, _, z in lowest_corners
        )
        return self.table_top if inside_table else self.floor_top

    def _resolve_support(
        self,
        object_T: np.ndarray,
        half_extent: float,
        *,
        bounce: bool,
    ) -> np.ndarray:
        """Clamp the cube above its support plane without tunnelling."""
        support = self._support_radius(object_T[:3, :3], half_extent)
        ground = self._ground_height(object_T, half_extent)
        contact_center_y = ground - support - self.contact_epsilon

        if float(object_T[1, 3]) < contact_center_y:
            return object_T

        resolved = object_T.copy()
        resolved[1, 3] = contact_center_y
        if bounce:
            if self.velocity[1] > 0.5:
                self.velocity[1] = -self.velocity[1] * self.restitution
            else:
                self.velocity[:] = 0.0
        return resolved

    def update(
        self,
        object_T: np.ndarray,
        grabbed: bool,
        dt: float,
        half_extent: float,
    ) -> np.ndarray:
        if grabbed:
            # 第一次抓住：从这一刻起才启用重力，松手后就会往下掉。
            self.active = True
            self.velocity[:] = 0.0
            # The hand must not be able to drag the cube through the table.
            return self._resolve_support(object_T, half_extent, bounce=False)

        if not self.active:
            return object_T

        # 掉帧时不要用太大的步长，避免穿透桌面。
        dt = float(np.clip(dt, 0.0, 0.05))
        object_T = object_T.copy()
        pos = object_T[:3, 3]

        self.velocity[1] += self.gravity * dt
        pos += self.velocity * dt

        object_T[:3, 3] = pos
        return self._resolve_support(object_T, half_extent, bounce=True)
