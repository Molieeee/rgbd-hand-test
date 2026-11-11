"""简单的桌面/重力物理。

只做课程演示需要的部分：物体松手后自由落体，落到桌面上停住。
坐标系是相机系（+x 右、+y 下、+z 前，单位米），所以重力方向是 +y。

物体的姿态在落地前保持不变；落点高度按旋转后立方体的最低点计算，
因此斜着落地也不会插进桌面。
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


class FallingObject:
    def __init__(
        self,
        table_top: float = 0.20,
        table_x: Tuple[float, float] = (-0.45, 0.45),
        table_z: Tuple[float, float] = (0.25, 0.85),
        floor_top: float = 0.85,
        gravity: float = 9.81,
        restitution: float = 0.15,
    ):
        self.table_top = float(table_top)
        self.table_x = tuple(map(float, table_x))
        self.table_z = tuple(map(float, table_z))
        self.floor_top = float(floor_top)
        self.gravity = float(gravity)
        self.restitution = float(restitution)
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

    def _ground_height(self, pos: np.ndarray) -> float:
        inside_table = (
            self.table_x[0] <= pos[0] <= self.table_x[1]
            and self.table_z[0] <= pos[2] <= self.table_z[1]
        )
        return self.table_top if inside_table else self.floor_top

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
            return object_T

        if not self.active:
            return object_T

        # 掉帧时不要用太大的步长，避免穿透桌面。
        dt = float(np.clip(dt, 0.0, 0.05))
        object_T = object_T.copy()
        pos = object_T[:3, 3]

        self.velocity[1] += self.gravity * dt
        pos += self.velocity * dt

        support = self._support_radius(object_T[:3, :3], half_extent)
        ground = self._ground_height(pos)
        if pos[1] + support >= ground:
            pos[1] = ground - support
            if self.velocity[1] > 0.5:
                self.velocity[1] = -self.velocity[1] * self.restitution
            else:
                self.velocity[:] = 0.0

        object_T[:3, 3] = pos
        return object_T
