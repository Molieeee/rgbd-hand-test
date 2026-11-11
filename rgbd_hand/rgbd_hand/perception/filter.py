"""3D 关节的时序平滑。

先用简单的 EMA；后续可以替换成 One Euro Filter（静止更稳、快速运动延迟更低），
对外接口保持不变即可。
"""
from __future__ import annotations

import numpy as np


class EMAFilter:
    def __init__(self, alpha: float = 0.75):
        if not 0.0 <= alpha < 1.0:
            raise ValueError("alpha 应在 [0, 1) 之间")
        self.alpha = float(alpha)
        self.state: np.ndarray | None = None

    def update(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if self.state is None or self.state.shape != points.shape:
            self.state = points.copy()
        else:
            self.state = self.alpha * self.state + (1.0 - self.alpha) * points
        return self.state.copy()

    def reset(self) -> None:
        self.state = None
