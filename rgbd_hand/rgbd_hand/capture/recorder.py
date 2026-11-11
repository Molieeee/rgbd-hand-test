"""把 Frame 序列写到磁盘，供回放 / 提交测试数据使用。

注意：路径可能包含中文，不能用 cv2.imwrite（Windows 下使用窄字符 API），
这里统一用 cv2.imencode + Path.write_bytes。
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .base import Frame


class Recorder:
    def __init__(self, root: str | Path, save_depth: bool = True):
        self.root = Path(root)
        self.color_dir = self.root / "color"
        self.depth_dir = self.root / "depth"
        self.color_dir.mkdir(parents=True, exist_ok=True)
        self.save_depth = save_depth
        self.index = 0
        self.K = None
        self.fps = 30
        self._timestamps: list[int] = []
        if save_depth:
            self.depth_dir.mkdir(parents=True, exist_ok=True)

    def write(self, frame: Frame) -> None:
        name = f"{self.index:06d}"
        ok, buf = cv2.imencode(".png", frame.color)
        if ok:
            (self.color_dir / f"{name}.png").write_bytes(buf.tobytes())

        if self.save_depth and frame.depth is not None:
            np.save(self.depth_dir / f"{name}.npy", frame.depth)

        if frame.K is not None:
            self.K = frame.K.tolist()
        self._timestamps.append(frame.timestamp_ms)
        self.index += 1

    def close(self) -> None:
        if len(self._timestamps) > 1:
            duration_s = (self._timestamps[-1] - self._timestamps[0]) / 1000.0
            if duration_s > 0:
                self.fps = int(round((len(self._timestamps) - 1) / duration_s))
        meta = {
            "frame_count": self.index,
            "fps": max(1, self.fps),
            "has_depth": bool(self.save_depth and self.K is not None),
            "K": self.K,
        }
        (self.root / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if self._timestamps:
            lines = ["timestamp_ms"] + [str(t) for t in self._timestamps]
            (self.root / "timestamps.csv").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
