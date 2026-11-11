"""回放后端：从磁盘读取录制好的序列。

目录格式（Recorder 写出的就是这种）：

    <root>/
      color/000000.png, 000001.png, ...
      depth/000000.npy, ...        # 可选
      meta.json                    # {"fps": 30, "K": null 或 3x3, ...}
      timestamps.csv               # 可选
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .base import Frame


class ReplaySource:
    def __init__(self, root: str | Path, loop: bool = False):
        self.root = Path(root)
        color_dir = self.root / "color"
        if not color_dir.is_dir():
            raise FileNotFoundError(f"回放目录不存在: {color_dir}")

        self.color_paths = sorted(
            [p for p in color_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        )
        if not self.color_paths:
            raise FileNotFoundError(f"回放目录里没有图片: {color_dir}")

        meta_path = self.root / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        k = meta.get("K")
        self.K = np.array(k, dtype=np.float64) if k else None
        self.fps = int(meta.get("fps", 30))
        self.loop = loop
        self.index = 0
        self._start_ts = 0

    def _depth_path(self, color_path: Path) -> Path:
        return self.root / "depth" / (color_path.stem + ".npy")

    def read(self) -> Optional[Frame]:
        if self.index >= len(self.color_paths):
            if not self.loop:
                return None
            self.index = 0

        color_path = self.color_paths[self.index]
        color = cv2.imdecode(np.fromfile(str(color_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if color is None:
            return None

        depth = None
        depth_path = self._depth_path(color_path)
        if depth_path.exists():
            depth = np.load(depth_path)

        ts = int(self.index * 1000 / max(self.fps, 1))
        frame = Frame(
            color=color,
            depth=depth,
            K=self.K if depth is not None else None,
            timestamp_ms=ts,
        )
        self.index += 1
        return frame

    def close(self) -> None:
        pass
