"""相机数据接口。

上层（感知 / 交互 / 渲染）只依赖这里定义的 Frame，
换相机时只需要新增一个 CameraSource 实现。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np


@dataclass
class Frame:
    """一帧数据，统一约定：

    - color: BGR, uint8, HxWx3
    - depth: uint16, 毫米, 已经对齐到 color 的像素网格；没有深度时为 None
    - K:     3x3 相机内参；没有标定/深度时为 None
    - timestamp_ms: 毫秒时间戳，必须单调递增
    """

    color: np.ndarray
    depth: Optional[np.ndarray] = None
    K: Optional[np.ndarray] = None
    timestamp_ms: int = 0

    @property
    def has_depth(self) -> bool:
        return self.depth is not None and self.K is not None


class CameraSource(Protocol):
    """所有相机后端的统一接口。"""

    def read(self) -> Optional[Frame]:
        """返回一帧；流结束或读取失败时返回 None。"""
        ...

    def close(self) -> None:
        ...
