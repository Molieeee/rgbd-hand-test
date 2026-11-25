"""抓取 / 移动 / 旋转 / 释放。

思路：
- 捏合点靠近物体且处于捏合状态 -> 抓取
- 抓取后短时间内把物块中心吸附到捏合点
- 之后每帧 object_T = hand_T @ T_hand_object
- 松开捏合 -> 释放，物体保持当前位姿

为了演示时“一直捏着却掉”的情况，抓取状态额外做了两处防抖：

- 手部检测偶尔丢帧（MediaPipe 在遮挡/运动模糊时会这样），短时间内
  保持抓取状态，物体停在最后位置，等手重新出现后继续跟随；
- 手重新出现后手势状态机需要几帧才会重新判定为捏合，这段时间也不释放。

所有矩阵都在相机坐标系（米）下，渲染时再统一转到场景坐标。
"""
from __future__ import annotations

import numpy as np


class Manipulator:
    def __init__(
        self,
        grab_radius: float = 0.08,
        hold_seconds: float = 0.80,
        release_delay: float = 0.35,
        snap_seconds: float = 0.12,
    ):
        self.grab_radius = float(grab_radius)
        # 手完全检测不到时，最多继续“抓”这么久（秒）。
        self.hold_seconds = float(hold_seconds)
        # 手在画面里、但状态机暂时判为非捏合时，延迟这么久再释放（秒）。
        self.release_delay = float(release_delay)
        # 抓取后把物块中心平滑移到 pinch_point 的时间。
        self.snap_seconds = max(float(snap_seconds), 0.0)

        self.grabbed = False
        self.T_hand_object: np.ndarray | None = None
        self._grab_offset = np.zeros(3, dtype=np.float64)
        self._snap_elapsed = 0.0
        self._missing = 0.0
        self._unpinched = 0.0

    def _attached_object_T(self, hand_T: np.ndarray, dt: float) -> np.ndarray:
        if self.T_hand_object is None:
            raise RuntimeError("抓取变换尚未初始化")

        self._snap_elapsed += max(float(dt), 0.0)
        if self.snap_seconds <= 0.0:
            progress = 1.0
        else:
            progress = min(self._snap_elapsed / self.snap_seconds, 1.0)

        # 旋转关系保持不变，只把初始平移偏移平滑缩到零。
        self.T_hand_object[:3, 3] = self._grab_offset * (1.0 - progress)
        return hand_T @ self.T_hand_object

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
            if not pinching:
                return object_T, False

            dist = np.linalg.norm(hand_T[:3, 3] - object_T[:3, 3])
            if dist < self.grab_radius:
                self.grabbed = True
                # 记录抓取瞬间的相对旋转和中心偏移，后续把偏移平滑降为零。
                self.T_hand_object = np.linalg.inv(hand_T) @ object_T
                self._grab_offset = self.T_hand_object[:3, 3].copy()
                self._snap_elapsed = 0.0
            return object_T, self.grabbed

        if pinching:
            self._unpinched = 0.0
            return self._attached_object_T(hand_T, dt), True

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
        self._grab_offset[:] = 0.0
        self._snap_elapsed = 0.0
        self._missing = 0.0
        self._unpinched = 0.0