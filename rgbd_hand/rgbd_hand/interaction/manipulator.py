"""抓取 / 移动 / 旋转 / 释放。

思路：
- 捏合点靠近物体且处于捏合状态 -> 抓取
- 抓取瞬间记录物体相对于手部坐标系的变换 T_hand_object
- 之后每帧 object_T = hand_T @ T_hand_object
- 松开捏合 -> 释放，物体保持当前位姿

为了演示时“一直捏着却掉”的情况，抓取状态额外做了两处防抖：

- 手部检测偶尔丢帧（MediaPipe 在遮挡/运动模糊时会这样），短时间内
  保持抓取状态，物体停在最后的位置，等手重新出现后继续跟随；
- 手重新出现后手势状态机需要几帧才会重新判定为捏合，这段时间也不释放。

所有矩阵都在相机坐标系（米）下，渲染时再统一转到场景坐标。
"""
from __future__ import annotations

import numpy as np


class Manipulator:
    def __init__(
        self,
        grab_radius: float = 0.12,
        hold_seconds: float = 0.50,
        release_delay: float = 0.20,
    ):
        self.grab_radius = float(grab_radius)
        # 手完全检测不到时，最多继续“抓”这么久（秒）。
        self.hold_seconds = float(hold_seconds)
        # 手在画面里、但状态机暂时判为非捏合时，延迟这么久再释放（秒）。
        self.release_delay = float(release_delay)

        self.grabbed = False
        self.T_hand_object: np.ndarray | None = None
        self._missing = 0.0
        self._unpinched = 0.0

    def update(
        self,
        hand_T: np.ndarray,
        pinching: bool,
        object_T: np.ndarray,
        dt: float = 0.0,
    ):
        self._missing = 0.0
        if not self.grabbed:
            self._unpinched = 0.0
            if pinching:
                dist = np.linalg.norm(hand_T[:3, 3] - object_T[:3, 3])
                if dist < self.grab_radius:
                    self.grabbed = True
                    self.T_hand_object = np.linalg.inv(hand_T) @ object_T
            return object_T, self.grabbed

        if pinching:
            self._unpinched = 0.0
            return hand_T @ self.T_hand_object, True

        # 状态机刚重新出现时会有几帧不判定为捏合，先等一下再松手。
        self._unpinched += float(dt)
        if self._unpinched >= self.release_delay:
            self.grabbed = False
            self.T_hand_object = None
            self._unpinched = 0.0
            return object_T, False
        return object_T, True

    def update_missing(self, object_T: np.ndarray, dt: float):
        """当前帧完全没检测到手时调用。"""
        if not self.grabbed:
            return object_T, False
        self._missing += float(dt)
        if self._missing >= self.hold_seconds:
            self.grabbed = False
            self.T_hand_object = None
            self._missing = 0.0
            return object_T, False
        return object_T, True

    def reset(self) -> None:
        self.grabbed = False
        self.T_hand_object = None
        self._missing = 0.0
        self._unpinched = 0.0
