"""D01-D05：泥坯局部网格雕刻与平滑。

坐标约定：本模块的输入和输出全部是泥坯**局部坐标**（米），不接触相机坐标
或场景位姿。E 负责射线拾取并把 ``PickResult.point_local`` /
``normal_local`` 传进来；D 只修改顶点数组；B 负责 ``set_vertices()``、
法线重算和显示刷新。每一帧只应提交一次结果。

主要接口::

    topo = Topology(vertices, triangles)          # 固定拓扑，构造一次
    pick = PickResult(True, 0, point, normal, d, clay.version)
    cmd  = BrushCommand("carve", 0.024, 1.0, dt, stroke_id)
    res  = apply_brush(vertices, pick, cmd, topo, stroke_start=start)
    clay.set_vertices(res.vertices)                # 一帧一次

算法要点：

* D01 手动局部点：沿 ``-pick.normal_local`` 压入表面（法线朝外）。
* D02 邻域：欧氏半径 ``d <= r``；切平面侧 ``(v-p)·n >= -tol*r``；
  顶点法线侧 ``n_v·n >= cos_min``；可选 Dijkstra 测地距限制。
  B 的圆柱接缝已经焊接，端面外环复用侧面顶点，因此接缝和端面天然无重复点。
* D03 深度 ``rate * strength * dt``，逐帧、逐笔画双重限幅。
* D04 预更新副本上的邻域均值：``v' = v + α w (mean(N(v)) - v)``。
* D05 笔画插值按笔刷半径的比例步进；``snapshot``/``restore`` 供撤销使用，
  撤销历史本身由 E 维护。
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import heapq
import math
from typing import Any

import numpy as np

# ---- D03/D04 参数（米、秒） ------------------------------------------------

# 满强度按住 1 秒、命中中心时的理论下压深度。
CARVE_RATE_M_PER_S = 0.060
# 单次 apply_brush 的最大位移；防止掉帧时一帧挖穿。
MAX_STEP_DEPTH_M = 0.004
# 同一笔画（stroke_start 快照）累计最大位移。
MAX_STROKE_DEPTH_M = 0.012
# 平滑混合速率和单步上限；边界权重由 smoothstep 自然压到 0。
SMOOTH_RATE_PER_S = 4.0
MAX_SMOOTH_ALPHA = 0.40
MAX_SMOOTH_STEP_M = 0.0015
# 选择参数：允许切平面后方一个半径以内的点（凹坑侧壁），背面仍被顶点法线排除。
DEFAULT_SIDE_TOLERANCE = 1.0
DEFAULT_NORMAL_COS_MIN = 0.10
DEFAULT_GEODESIC_FACTOR = 1.35
# 笔画插值：相邻印章间距 = 笔刷半径 * 0.40，且不小于 1 mm。
DEFAULT_STAMP_SPACING_FACTOR = 0.40
MIN_STAMP_SPACING_M = 0.001


# ---- 接口数据 ---------------------------------------------------------------


@dataclass
class PickResult:
    """E 的拾取结果；字段名与分工文档一致，长度单位米。"""

    valid: bool
    hand_id: int | None = None
    point_local: np.ndarray | None = None
    normal_local: np.ndarray | None = None
    distance_m: float = float("nan")
    mesh_version: int = -1

    @classmethod
    def miss(cls, hand_id: int | None = None, mesh_version: int = -1) -> "PickResult":
        """手/射线没有命中时使用，避免调用方到处填 None。"""
        return cls(False, hand_id, None, None, float("nan"), mesh_version)


@dataclass
class BrushCommand:
    """A/E 下发的笔刷指令；strength 在 [0, 1]，dt 单位秒。"""

    tool: str
    radius_m: float
    strength: float = 1.0
    dt: float = 0.0
    stroke_id: int = 0


@dataclass(frozen=True)
class BrushResult:
    """一次笔刷应用的结果；``vertices`` 总是新数组，输入不会被修改。"""

    vertices: np.ndarray
    changed_mask: np.ndarray
    changed_indices: np.ndarray
    max_displacement_m: float
    applied: bool
    reason: str = ""
    selected_count: int = 0
    step_depth_m: float = 0.0
    stroke_id: int = 0
    tool: str = ""


@dataclass(frozen=True)
class Selection:
    """笔刷邻域；``mask`` 只包含实际会移动的顶点（权重 > 0）。"""

    mask: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    distances_m: np.ndarray
    source_index: int
    geodesic_m: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.indices.size)


@dataclass(frozen=True)
class Stamp:
    """笔画插值产生的一个笔刷位置。"""

    point_local: np.ndarray
    normal_local: np.ndarray


# ---- 基础校验 ---------------------------------------------------------------


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _as_vertices(vertices: Any, *, name: str = "vertices") -> np.ndarray:
    try:
        array = np.asarray(vertices, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an (N, 3) array") from exc
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty (N, 3) array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _as_point(value: Any, *, name: str = "point_local") -> np.ndarray:
    try:
        point = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a length-3 vector") from exc
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError(f"{name} must be a finite length-3 vector")
    return point


def _as_unit_normal(value: Any, *, name: str = "normal_local") -> np.ndarray:
    normal = _as_point(value, name=name)
    length = float(np.linalg.norm(normal))
    if length < 1e-12:
        raise ValueError(f"{name} must be non-zero")
    return normal / length


def _runtime_point_normal(point: Any, normal: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """运行时输入（手部/射线）可能缺失：返回 None 而不是抛异常。"""
    try:
        p = np.asarray(point, dtype=np.float64)
        n = np.asarray(normal, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if p.shape != (3,) or n.shape != (3,):
        return None
    if not np.isfinite(p).all() or not np.isfinite(n).all():
        return None
    length = float(np.linalg.norm(n))
    if length < 1e-12:
        return None
    return p, n / length


def _runtime_pick(pick: Any) -> tuple[np.ndarray, np.ndarray] | None:
    if pick is None:
        return None
    try:
        valid = bool(_field(pick, "valid", False))
    except (TypeError, ValueError):
        return None
    if not valid:
        return None
    return _runtime_point_normal(
        _field(pick, "point_local", None),
        _field(pick, "normal_local", None),
    )


def _command_fields(command: Any) -> tuple[str, Any, Any, Any, int] | None:
    if command is None:
        return None
    tool = _field(command, "tool", None)
    if tool is None:
        return None
    try:
        stroke_id = int(_field(command, "stroke_id", 0))
    except (TypeError, ValueError):
        stroke_id = 0
    return (
        str(tool),
        _field(command, "radius_m", None),
        _field(command, "strength", 1.0),
        _field(command, "dt", 0.0),
        stroke_id,
    )


def _depth_limit(value: Any, name: str) -> float | None:
    if value is None:
        return None
    limit = _finite_float(value)
    if limit is None or limit <= 0:
        raise ValueError(f"{name} must be positive and finite, or None")
    return limit


def _unchanged(
    vertices: np.ndarray,
    reason: str,
    *,
    selected_count: int = 0,
    stroke_id: int = 0,
    tool: str = "",
) -> BrushResult:
    n = len(vertices)
    return BrushResult(
        vertices=np.array(vertices, dtype=np.float64, copy=True),
        changed_mask=np.zeros(n, dtype=bool),
        changed_indices=np.empty(0, dtype=np.int64),
        max_displacement_m=0.0,
        applied=False,
        reason=reason,
        selected_count=selected_count,
        step_depth_m=0.0,
        stroke_id=stroke_id,
        tool=tool,
    )


def _step_result(
    vertices: np.ndarray,
    new_vertices: np.ndarray,
    selection: Selection,
    step_depth_m: float,
    stroke_id: int,
    tool: str,
) -> BrushResult:
    delta = new_vertices - vertices
    changed_mask = np.any(delta != 0.0, axis=1)
    if not changed_mask.any():
        return _unchanged(
            vertices,
            "no displacement (zero weight or already at clamp)",
            selected_count=len(selection),
            stroke_id=stroke_id,
            tool=tool,
        )
    if not np.isfinite(new_vertices).all():
        return _unchanged(
            vertices,
            "non-finite result",
            selected_count=len(selection),
            stroke_id=stroke_id,
            tool=tool,
        )
    return BrushResult(
        vertices=new_vertices,
        changed_mask=changed_mask,
        changed_indices=np.flatnonzero(changed_mask).astype(np.int64, copy=False),
        max_displacement_m=float(np.linalg.norm(delta, axis=1).max()),
        applied=True,
        selected_count=len(selection),
        step_depth_m=float(step_depth_m),
        stroke_id=stroke_id,
        tool=tool,
    )


def _validate_selection(selection: Selection, n: int) -> Selection:
    if not isinstance(selection, Selection):
        raise TypeError("selection must be a sculpt.Selection")
    if selection.weights.shape != (n,) or selection.mask.shape != (n,):
        raise ValueError("selection does not match the vertex count")
    indices = np.asarray(selection.indices)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("selection.indices must be a 1-D integer array")
    if indices.size and (indices.min() < 0 or indices.max() >= n):
        raise ValueError("selection.indices out of range")
    if not np.isfinite(selection.weights).all() or not np.isfinite(
        selection.distances_m
    ).all():
        raise ValueError("selection weights/distances must be finite")
    return selection


def _validate_stroke_start(stroke_start: Any, n: int) -> np.ndarray | None:
    if stroke_start is None:
        return None
    start = _as_vertices(stroke_start, name="stroke_start")
    if len(start) != n:
        raise ValueError("stroke_start must have the same vertex count")
    return start


def _clamp_total(
    vertices: np.ndarray, stroke_start: np.ndarray, limit_m: float | None
) -> np.ndarray:
    """把每个顶点相对笔画起点的总位移限制在 limit_m 内（原地修改副本）。"""
    if limit_m is None:
        return vertices
    total = vertices - stroke_start
    magnitude = np.linalg.norm(total, axis=1)
    over = magnitude > limit_m
    if over.any():
        scale = limit_m / magnitude[over]
        vertices[over] = stroke_start[over] + total[over] * scale[:, None]
    return vertices


# ---- D02 拓扑与邻域 ---------------------------------------------------------


class Topology:
    """由 B 的固定三角形数组构造；顶点坐标可以变化，邻接关系不变。"""

    def __init__(self, vertices: Any, triangles: Any):
        vertices = _as_vertices(vertices)
        try:
            triangles = np.asarray(triangles)
        except (TypeError, ValueError) as exc:
            raise ValueError("triangles must be an (M, 3) integer array") from exc
        if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.shape[0] == 0:
            raise ValueError("triangles must be a non-empty (M, 3) array")
        if not np.issubdtype(triangles.dtype, np.integer):
            if not np.isfinite(triangles).all() or not np.allclose(
                triangles, np.round(triangles)
            ):
                raise ValueError("triangles must contain integer vertex indices")
        triangles = triangles.astype(np.int64, copy=False)
        n = len(vertices)
        if triangles.min() < 0 or triangles.max() >= n:
            raise ValueError("triangle index out of range")
        if np.any(
            (triangles[:, 0] == triangles[:, 1])
            | (triangles[:, 1] == triangles[:, 2])
            | (triangles[:, 2] == triangles[:, 0])
        ):
            raise ValueError("triangles must reference three distinct vertices")

        self.vertex_count = n
        self.triangles = triangles.copy()
        raw_edges = np.concatenate(
            (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]),
            axis=0,
        )
        raw_edges = np.sort(raw_edges, axis=1)
        edges, counts = np.unique(raw_edges, axis=0, return_counts=True)
        self.edges = edges.astype(np.int64, copy=False)
        self.boundary_edges = self.edges[counts == 1]
        self.nonmanifold_edges = self.edges[counts > 2]

        adjacency: list[list[int]] = [[] for _ in range(n)]
        for i, j in self.edges:
            adjacency[int(i)].append(int(j))
            adjacency[int(j)].append(int(i))
        self.adjacency = tuple(
            np.asarray(sorted(row), dtype=np.int64) for row in adjacency
        )

    def _check_vertices(self, vertices: Any) -> np.ndarray:
        vertices = _as_vertices(vertices)
        if len(vertices) != self.vertex_count:
            raise ValueError(
                f"expected {self.vertex_count} vertices, got {len(vertices)}"
            )
        return vertices

    def max_edge_length(self, vertices: Any) -> float:
        vertices = self._check_vertices(vertices)
        if len(self.edges) == 0:
            return 0.0
        lengths = np.linalg.norm(
            vertices[self.edges[:, 0]] - vertices[self.edges[:, 1]], axis=1
        )
        return float(lengths.max())

    def vertex_normals(self, vertices: Any) -> np.ndarray:
        """面积加权顶点法线；B 的三角形绕向保证初始圆柱法线朝外。"""
        vertices = self._check_vertices(vertices)
        tri = self.triangles
        a = vertices[tri[:, 0]]
        b = vertices[tri[:, 1]]
        c = vertices[tri[:, 2]]
        face_normals = np.cross(b - a, c - a)
        normals = np.zeros_like(vertices)
        for k in range(3):
            np.add.at(normals, tri[:, k], face_normals)
        lengths = np.linalg.norm(normals, axis=1)
        nonzero = lengths > 1e-15
        normals[nonzero] /= lengths[nonzero, None]
        normals[~nonzero] = 0.0
        return normals

    def nearest_vertex(
        self, point_local: Any, vertices: Any, candidates: Any = None
    ) -> int:
        point = _as_point(point_local)
        vertices = self._check_vertices(vertices)
        if candidates is None:
            indices = np.arange(len(vertices), dtype=np.int64)
        else:
            candidates = np.asarray(candidates)
            if candidates.dtype == bool:
                if candidates.shape != (len(vertices),):
                    raise ValueError("boolean candidates must match vertex count")
                indices = np.flatnonzero(candidates).astype(np.int64, copy=False)
            else:
                indices = candidates.astype(np.int64, copy=False).ravel()
                if indices.size and (
                    indices.min() < 0 or indices.max() >= len(vertices)
                ):
                    raise ValueError("candidate index out of range")
        if indices.size == 0:
            return -1
        distances = np.linalg.norm(vertices[indices] - point, axis=1)
        return int(indices[int(np.argmin(distances))])

    def geodesic_distances(
        self, source: int, vertices: Any, limit_m: float
    ) -> np.ndarray:
        """从 source 出发的 Dijkstra 边长测地距；超过 limit_m 的顶点为 inf。

        只访问局部区域，因此不会因为整张 5282 顶点网格而变慢。
        """
        vertices = self._check_vertices(vertices)
        limit = _finite_float(limit_m)
        if limit is None or limit <= 0:
            raise ValueError("limit_m must be positive and finite")
        if not 0 <= int(source) < len(vertices):
            raise ValueError("source vertex out of range")

        distances = np.full(len(vertices), np.inf, dtype=np.float64)
        source = int(source)
        distances[source] = 0.0
        queue: list[tuple[float, int]] = [(0.0, source)]
        adjacency = self.adjacency
        while queue:
            distance, u = heapq.heappop(queue)
            if distance > distances[u] + 1e-15:
                continue
            if distance > limit:
                break
            for v in adjacency[u]:
                step = float(np.linalg.norm(vertices[int(v)] - vertices[u]))
                new_distance = distance + step
                if new_distance < distances[int(v)] and new_distance <= limit:
                    distances[int(v)] = new_distance
                    heapq.heappush(queue, (new_distance, int(v)))
        return distances


def select_vertices(
    vertices: Any,
    point_local: Any,
    normal_local: Any,
    radius_m: float,
    topology: Topology | None = None,
    *,
    side_tolerance: float = DEFAULT_SIDE_TOLERANCE,
    normal_cos_min: float = DEFAULT_NORMAL_COS_MIN,
    geodesic: bool = True,
    geodesic_factor: float = DEFAULT_GEODESIC_FACTOR,
) -> Selection:
    """D02：欧氏半径 + 切平面侧 + 顶点法线侧 + 可选测地距的邻域选择。

    权重为 smoothstep：``u = 1 - d/r``，``w = u²(3-2u)``；半径边缘权重为 0，
    因此不会在笔刷边界产生台阶。

    切平面侧允许 ``side_tolerance * r`` 的后方深度：凹坑的侧壁在切平面后方，
    不能简单要求 ``(v-p)·n >= 0``；真正的背面由顶点法线项排除。
    """
    vertices = _as_vertices(vertices)
    point = _as_point(point_local)
    normal = _as_unit_normal(normal_local)
    radius = _finite_float(radius_m)
    if radius is None or radius <= 0:
        raise ValueError("radius_m must be positive and finite")
    tolerance = _finite_float(side_tolerance)
    if tolerance is None or tolerance < 0:
        raise ValueError("side_tolerance must be non-negative and finite")
    cos_min = _finite_float(normal_cos_min)
    if cos_min is None or not -1.0 <= cos_min <= 1.0:
        raise ValueError("normal_cos_min must be in [-1, 1]")
    geodesic_factor = _finite_float(geodesic_factor)
    if geodesic_factor is None or geodesic_factor <= 0:
        raise ValueError("geodesic_factor must be positive and finite")
    if topology is not None:
        vertices = topology._check_vertices(vertices)

    delta = vertices - point
    distances = np.linalg.norm(delta, axis=1)
    mask = distances <= radius
    mask &= (delta @ normal) >= -tolerance * radius

    if topology is not None:
        # 背面顶点的法线与拾取法线相反，因此这一项比切平面符号更稳健，
        # 也天然处理端面和接缝（接缝没有重复点）。
        vertex_normals = topology.vertex_normals(vertices)
        mask &= (vertex_normals @ normal) >= cos_min

    geodesic_distances = None
    if mask.any():
        candidates = np.flatnonzero(mask)
        source = int(candidates[int(np.argmin(distances[candidates]))])
        if topology is not None and geodesic:
            limit = radius * geodesic_factor + topology.max_edge_length(vertices)
            geodesic_distances = topology.geodesic_distances(source, vertices, limit)
            mask &= np.isfinite(geodesic_distances)
    else:
        source = -1

    u = np.clip(1.0 - distances / radius, 0.0, 1.0)
    weights = np.where(mask, u * u * (3.0 - 2.0 * u), 0.0)
    mask &= weights > 0.0
    indices = np.flatnonzero(mask).astype(np.int64, copy=False)
    if indices.size:
        source = int(indices[int(np.argmin(distances[indices]))])
    else:
        source = -1
    return Selection(
        mask=mask,
        indices=indices,
        weights=weights,
        distances_m=distances,
        source_index=source,
        geodesic_m=geodesic_distances,
    )


# ---- D01/D03 雕刻与 D04 平滑 ------------------------------------------------


def carve(
    vertices: Any,
    pick: Any,
    command: Any,
    topology: Topology | None = None,
    *,
    selection: Selection | None = None,
    stroke_start: Any = None,
    side_tolerance: float = DEFAULT_SIDE_TOLERANCE,
    normal_cos_min: float = DEFAULT_NORMAL_COS_MIN,
    geodesic: bool = True,
    max_step_depth_m: float | None = MAX_STEP_DEPTH_M,
    max_stroke_depth_m: float | None = MAX_STROKE_DEPTH_M,
) -> BrushResult:
    """D01/D02/D03：沿拾取法线的反方向压入，深度受单步和笔画累计双重限幅。"""
    vertices = _as_vertices(vertices)
    point_normal = _runtime_pick(pick)
    if point_normal is None:
        return _unchanged(vertices, "pick invalid")
    fields = _command_fields(command)
    if fields is None:
        return _unchanged(vertices, "command missing")
    tool, radius_value, strength_value, dt_value, stroke_id = fields
    if tool != "carve":
        return _unchanged(vertices, f"tool {tool!r} is not carve", stroke_id=stroke_id)

    radius = _finite_float(radius_value)
    if radius is None or radius <= 0:
        return _unchanged(vertices, "radius_m invalid", stroke_id=stroke_id, tool=tool)
    strength = _finite_float(strength_value)
    if strength is None or strength <= 0:
        return _unchanged(vertices, "strength invalid", stroke_id=stroke_id, tool=tool)
    dt = _finite_float(dt_value)
    if dt is None or dt <= 0:
        return _unchanged(vertices, "dt invalid", stroke_id=stroke_id, tool=tool)
    step_limit = _depth_limit(max_step_depth_m, "max_step_depth_m")
    stroke_limit = _depth_limit(max_stroke_depth_m, "max_stroke_depth_m")
    stroke_start = _validate_stroke_start(stroke_start, len(vertices))

    point, normal = point_normal
    if selection is None:
        selection = select_vertices(
            vertices,
            point,
            normal,
            radius,
            topology,
            side_tolerance=side_tolerance,
            normal_cos_min=normal_cos_min,
            geodesic=geodesic,
        )
    else:
        selection = _validate_selection(selection, len(vertices))
    if len(selection) == 0:
        return _unchanged(
            vertices, "no vertices in brush", stroke_id=stroke_id, tool=tool
        )

    step_depth = CARVE_RATE_M_PER_S * min(strength, 1.0) * dt
    if step_limit is not None:
        step_depth = min(step_depth, step_limit)
    if step_depth <= 0:
        return _unchanged(
            vertices, "step depth is zero", stroke_id=stroke_id, tool=tool
        )

    new_vertices = np.array(vertices, dtype=np.float64, copy=True)
    weights = selection.weights[selection.indices]
    # PickResult.normal_local 是朝外的表面法线；刻刀要压入泥坯，所以取负方向。
    new_vertices[selection.indices] -= (
        step_depth * weights
    )[:, None] * normal[None, :]
    if stroke_start is not None:
        _clamp_total(new_vertices, stroke_start, stroke_limit)
    return _step_result(
        vertices, new_vertices, selection, step_depth, stroke_id, tool
    )


def smooth(
    vertices: Any,
    pick: Any,
    command: Any,
    topology: Topology | None = None,
    *,
    selection: Selection | None = None,
    stroke_start: Any = None,
    side_tolerance: float = DEFAULT_SIDE_TOLERANCE,
    normal_cos_min: float = DEFAULT_NORMAL_COS_MIN,
    geodesic: bool = True,
    max_stroke_depth_m: float | None = MAX_STROKE_DEPTH_M,
    max_smooth_step_m: float | None = MAX_SMOOTH_STEP_M,
) -> BrushResult:
    """D04：邻域均值平滑。

    ``mean(N(v))`` 从**更新前**的顶点副本计算，避免同一帧内前面的顶点影响
    后面的顶点。只对笔刷内顶点施加位移，笔刷外邻居保持不动，给边界提供锚点；
    权重在半径边缘为 0，进一步避免边界台阶。
    """
    vertices = _as_vertices(vertices)
    point_normal = _runtime_pick(pick)
    if point_normal is None:
        return _unchanged(vertices, "pick invalid")
    fields = _command_fields(command)
    if fields is None:
        return _unchanged(vertices, "command missing")
    tool, radius_value, strength_value, dt_value, stroke_id = fields
    if tool != "smooth":
        return _unchanged(vertices, f"tool {tool!r} is not smooth", stroke_id=stroke_id)
    if topology is None:
        return _unchanged(
            vertices,
            "smoothing requires topology",
            stroke_id=stroke_id,
            tool=tool,
        )

    radius = _finite_float(radius_value)
    if radius is None or radius <= 0:
        return _unchanged(vertices, "radius_m invalid", stroke_id=stroke_id, tool=tool)
    strength = _finite_float(strength_value)
    if strength is None or strength <= 0:
        return _unchanged(vertices, "strength invalid", stroke_id=stroke_id, tool=tool)
    dt = _finite_float(dt_value)
    if dt is None or dt <= 0:
        return _unchanged(vertices, "dt invalid", stroke_id=stroke_id, tool=tool)
    stroke_limit = _depth_limit(max_stroke_depth_m, "max_stroke_depth_m")
    step_limit = _depth_limit(max_smooth_step_m, "max_smooth_step_m")
    stroke_start = _validate_stroke_start(stroke_start, len(vertices))

    point, normal = point_normal
    if selection is None:
        selection = select_vertices(
            vertices,
            point,
            normal,
            radius,
            topology,
            side_tolerance=side_tolerance,
            normal_cos_min=normal_cos_min,
            geodesic=geodesic,
        )
    else:
        selection = _validate_selection(selection, len(vertices))
    if len(selection) == 0:
        return _unchanged(
            vertices, "no vertices in brush", stroke_id=stroke_id, tool=tool
        )

    alpha = min(MAX_SMOOTH_ALPHA, SMOOTH_RATE_PER_S * min(strength, 1.0) * dt)
    if alpha <= 0:
        return _unchanged(vertices, "smoothing alpha is zero", stroke_id=stroke_id, tool=tool)

    indices = selection.indices
    means = np.zeros((len(indices), 3), dtype=np.float64)
    has_neighbors = np.zeros(len(indices), dtype=bool)
    for k, vertex_index in enumerate(indices):
        neighbors = topology.adjacency[int(vertex_index)]
        if neighbors.size:
            means[k] = vertices[neighbors].mean(axis=0)
            has_neighbors[k] = True

    # 预更新副本：means 全部来自 vertices，之后再统一写入 new_vertices。
    delta = (means - vertices[indices]) * (
        alpha * selection.weights[indices]
    )[:, None]
    delta[~has_neighbors] = 0.0
    if step_limit is not None:
        magnitude = np.linalg.norm(delta, axis=1)
        over = magnitude > step_limit
        if over.any():
            delta[over] *= (step_limit / magnitude[over])[:, None]

    new_vertices = np.array(vertices, dtype=np.float64, copy=True)
    new_vertices[indices] += delta
    if stroke_start is not None:
        _clamp_total(new_vertices, stroke_start, stroke_limit)
    return _step_result(vertices, new_vertices, selection, 0.0, stroke_id, tool)


def apply_brush(
    vertices: Any,
    pick: Any,
    command: Any,
    topology: Topology | None = None,
    *,
    selection: Selection | None = None,
    stroke_start: Any = None,
    side_tolerance: float = DEFAULT_SIDE_TOLERANCE,
    normal_cos_min: float = DEFAULT_NORMAL_COS_MIN,
    geodesic: bool = True,
    max_step_depth_m: float | None = MAX_STEP_DEPTH_M,
    max_stroke_depth_m: float | None = MAX_STROKE_DEPTH_M,
    max_smooth_step_m: float | None = MAX_SMOOTH_STEP_M,
) -> BrushResult:
    """统一笔刷入口；E 的 controller 每帧调用一次并提交 ``res.vertices``。"""
    vertices = _as_vertices(vertices)
    tool_value = _field(command, "tool", None) if command is not None else None
    tool = str(tool_value) if tool_value is not None else ""
    if tool in ("", "none", "idle"):
        return _unchanged(vertices, "no active tool")
    try:
        stroke_id = int(_field(command, "stroke_id", 0))
    except (TypeError, ValueError):
        stroke_id = 0
    if tool == "carve":
        return carve(
            vertices,
            pick,
            command,
            topology,
            selection=selection,
            stroke_start=stroke_start,
            side_tolerance=side_tolerance,
            normal_cos_min=normal_cos_min,
            geodesic=geodesic,
            max_step_depth_m=max_step_depth_m,
            max_stroke_depth_m=max_stroke_depth_m,
        )
    if tool == "smooth":
        return smooth(
            vertices,
            pick,
            command,
            topology,
            selection=selection,
            stroke_start=stroke_start,
            side_tolerance=side_tolerance,
            normal_cos_min=normal_cos_min,
            geodesic=geodesic,
            max_stroke_depth_m=max_stroke_depth_m,
            max_smooth_step_m=max_smooth_step_m,
        )
    return _unchanged(
        vertices, f"unknown tool {tool!r}", stroke_id=stroke_id, tool=tool
    )


# ---- D05 快照与笔画插值 -----------------------------------------------------


def snapshot(vertices: Any) -> np.ndarray:
    """返回一份独立快照；E 用它维护撤销栈。"""
    return np.array(_as_vertices(vertices), dtype=np.float64, copy=True)


def restore(snapshot_vertices: Any) -> np.ndarray:
    """从快照恢复出一个新数组，不共享内存。"""
    return np.array(
        _as_vertices(snapshot_vertices, name="snapshot"), dtype=np.float64, copy=True
    )


class StrokeSampler:
    """D05：在两帧命中点之间按笔刷半径插值出多个印章。

    典型用法::

        sampler = StrokeSampler()
        stamps = [sampler.begin(vertices, p0, n0, stroke_id=7)]
        ...
        stamps = sampler.step(p1, n1, radius_m)   # 中间点 + 端点
        start = sampler.end()                     # 交给 E 做撤销

    调用方把这一帧的 dt 平均分给所有印章；``begin`` 返回的起点也要应用。
    无效命中（None/NaN/零法线）会自动结束笔画，对应“手丢失/暂停/切换工具”
    时不应继续留痕。
    """

    def __init__(
        self,
        spacing_factor: float = DEFAULT_STAMP_SPACING_FACTOR,
        min_spacing_m: float = MIN_STAMP_SPACING_M,
        max_stamps_per_step: int = 64,
    ):
        spacing_factor = _finite_float(spacing_factor)
        min_spacing = _finite_float(min_spacing_m)
        if spacing_factor is None or spacing_factor <= 0:
            raise ValueError("spacing_factor must be positive and finite")
        if min_spacing is None or min_spacing <= 0:
            raise ValueError("min_spacing_m must be positive and finite")
        if (
            isinstance(max_stamps_per_step, (bool, np.bool_))
            or not isinstance(max_stamps_per_step, (int, np.integer))
            or max_stamps_per_step < 1
        ):
            raise ValueError("max_stamps_per_step must be an integer >= 1")
        self.spacing_factor = spacing_factor
        self.min_spacing_m = min_spacing
        self.max_stamps_per_step = int(max_stamps_per_step)
        self._active = False
        self._stroke_id: int | None = None
        self._start: np.ndarray | None = None
        self._last_point: np.ndarray | None = None
        self._last_normal: np.ndarray | None = None
        self._next_id = 1

    @property
    def active(self) -> bool:
        return self._active

    @property
    def stroke_id(self) -> int | None:
        return self._stroke_id

    @property
    def stroke_start(self) -> np.ndarray | None:
        return None if self._start is None else self._start.copy()

    def begin(
        self,
        vertices: Any,
        point_local: Any,
        normal_local: Any,
        stroke_id: int | None = None,
    ) -> Stamp:
        vertices = _as_vertices(vertices)
        point = _as_point(point_local)
        normal = _as_unit_normal(normal_local)
        if stroke_id is None:
            stroke_id = self._next_id
        else:
            try:
                stroke_id = int(stroke_id)
            except (TypeError, ValueError) as exc:
                raise ValueError("stroke_id must be an integer") from exc
        self._next_id = max(self._next_id, int(stroke_id) + 1)
        self._active = True
        self._stroke_id = int(stroke_id)
        self._start = np.array(vertices, dtype=np.float64, copy=True)
        self._last_point = point.copy()
        self._last_normal = normal.copy()
        return Stamp(point.copy(), normal.copy())

    def step(
        self, point_local: Any, normal_local: Any, radius_m: float
    ) -> list[Stamp]:
        if not self._active:
            raise RuntimeError("begin() must be called before step()")
        point_normal = _runtime_point_normal(point_local, normal_local)
        radius = _finite_float(radius_m)
        if point_normal is None or radius is None or radius <= 0:
            self.end()
            return []
        point, normal = point_normal
        spacing = max(self.min_spacing_m, radius * self.spacing_factor)
        delta = point - self._last_point
        distance = float(np.linalg.norm(delta))
        count = 1 if distance <= 0 else max(1, int(math.ceil(distance / spacing)))
        if count > self.max_stamps_per_step:
            # 命中点跳得太远，视为一次不连续，不把整段都压进泥坯。
            self.end()
            return []
        stamps: list[Stamp] = []
        for i in range(1, count + 1):
            t = i / count
            stamp_point = self._last_point + delta * t
            blended = (1.0 - t) * self._last_normal + t * normal
            length = float(np.linalg.norm(blended))
            if length < 1e-12:
                stamp_normal = normal if t >= 0.5 else self._last_normal
            else:
                stamp_normal = blended / length
            stamps.append(Stamp(stamp_point.copy(), stamp_normal.copy()))
        self._last_point = point.copy()
        self._last_normal = normal.copy()
        return stamps

    def end(self) -> np.ndarray | None:
        """结束笔画并返回起点快照（供 E 的撤销历史）；没有活动笔画时返回 None。"""
        start = None if self._start is None else self._start.copy()
        self._active = False
        self._stroke_id = None
        self._start = None
        self._last_point = None
        self._last_normal = None
        return start


__all__ = [
    "BrushCommand",
    "BrushResult",
    "PickResult",
    "Selection",
    "Stamp",
    "StrokeSampler",
    "Topology",
    "apply_brush",
    "carve",
    "restore",
    "select_vertices",
    "smooth",
    "snapshot",
]
