# D01/D02 局部网格雕刻验收（2026-11-25）

本文件记录 D（罗淇玥）负责的 `interaction/sculpt.py` 的第一阶段验收：
D01 手动局部坐标雕刻、D02 邻域/背面/端面/接缝处理。D03 连续深度与限幅、
D04 邻域均值平滑、D05 笔画插值和快照接口已经实现并有单元测试；D06 的
完整 benchmark 和实验图在后续补充。

## 运行

```bat
cd rgbd_hand\rgbd_hand
python -m unittest discover -s tests -v
python sculpt_demo.py
python sculpt_demo.py --verify-dir %TEMP%\sculpt_verify
```

窗口按键：

| 键 | 功能 |
| --- | --- |
| C | 在泥坯正面中心压一次（`dt = 1/30 s`） |
| S | 在同一位置做一次邻域均值平滑 |
| R | 恢复初始形状和位姿 |
| A / F | 绕 Y 轴转动泥坯 |
| V | 恢复视角 |
| Q | 退出 |

`--verify-dir` 会自动完成 3 次雕刻、背面检查、几何质量检查和复位，
并写出 `verification.json` 与 4 张截图。它不会修改 B 的 `clay_demo.py`。

## 接口（与分工文档一致）

```python
PickResult(valid, hand_id, point_local, normal_local, distance_m, mesh_version)
BrushCommand(tool, radius_m, strength, dt, stroke_id)   # tool: "carve" / "smooth"
apply_brush(vertices, pick, command, topology,
            selection=None, stroke_start=None) -> BrushResult
```

- `point_local` / `normal_local` 由 E 的 `picking.py` 从场景拾取结果反变换到
  泥坯局部坐标；`normal_local` 必须是**朝外的表面法线**。刻刀沿
  `-normal_local` 压入，不能把射线方向当法线传进来。
- D 返回完整的 `BrushResult.vertices`、`changed_mask`、`changed_indices`、
  `max_displacement_m`、`step_depth_m` 和 `reason`；E 每帧只调用一次
  `clay.set_vertices(res.vertices)` / `scene.update_clay(res.vertices)`。
- `Topology(vertices, triangles)` 在启动时构造一次；顶点坐标可以变，邻接关系
  不变。B 的 `ClayMesh.triangles` 直接传入。
- 撤销历史由 E 维护；D 提供 `snapshot()` / `restore()` 和
  `StrokeSampler.end()`（返回笔画起点快照）。

## 算法

### D02 邻域选择

对拾取点 `p`、朝外法线 `n`、笔刷半径 `r`：

1. 欧氏半径：`d_i = ||v_i - p|| <= r`。
2. 切平面侧：`(v_i - p)·n >= -side_tolerance * r`。默认
   `side_tolerance = 1.0`，允许凹坑侧壁落在切平面后方一个半径以内；
   背面仍会被下一项排除。
3. 顶点法线侧：`n_i·n >= normal_cos_min`（默认 0.1）。背面法线朝反方向，
   端面法线接近垂直，因此不会被侧面的笔刷带走。
4. 可选测地距：从最近的入选顶点做 Dijkstra，限
   `r * 1.35 + max_edge`。它防止薄结构两侧被欧氏球“穿透”，并让接缝
   自然连通。B 的圆柱接缝已经焊接，端面外环复用侧面顶点，没有重复点。
5. 权重用 smoothstep：`u = 1 - d/r`，`w = u²(3 - 2u)`。半径边缘 `w = 0`。

`Selection.mask` 只包含 `w > 0` 的顶点；`weights` 是全长数组，便于 B 做
笔刷高亮。

### D01/D03 雕刻深度

```text
raw_depth    = CARVE_RATE_M_PER_S * clamp(strength, 0, 1) * dt
step_depth   = min(raw_depth, MAX_STEP_DEPTH_M)          # 4 mm
total_depth  = min(||v - stroke_start||, MAX_STROKE_DEPTH_M)  # 12 mm
Δv_i         = -n * step_depth * w_i
```

- 单步限幅防止掉帧（`dt` 很大）时一帧挖穿。
- 累计限幅以 `stroke_start` 快照为基准；E 在 `begin stroke` 时保存快照，
  每帧把同一个快照传进来。当前实现按总位移的欧氏长度限幅，保守且对
  平滑也适用。
- 非法 `pick`（无效、NaN、零法线）、非正半径/`dt`/`strength`、未知工具
  都返回 `applied=False`，不会修改输入数组。

### D04 平滑

```text
mean_i = mean(v_j, j ∈ N(i))          # N(i) 用更新前的顶点副本
Δv_i   = α * w_i * (mean_i - v_i)
α      = min(0.40, SMOOTH_RATE_PER_S * strength * dt)
```

只移动笔刷内顶点；笔刷外邻居保持不动，作为边界锚点。权重在半径边缘为 0，
单步位移另有 `MAX_SMOOTH_STEP_M = 1.5 mm` 限幅。测试里检查了：
预更新副本（顺序更新会得到不同结果）、凹坑深度下降、单步不引起明显收缩。

### D05 笔画插值

`StrokeSampler` 在上一帧命中和当前帧命中之间按
`max(1 mm, 0.40 * radius)` 的间距插入印章，法线做线性插值后归一化。
`step()` 返回不含起点的中间点 + 端点；调用方把这一帧的 `dt` 平均分给
所有印章。无效命中会自动 `end()`，对应手丢失/暂停/切换工具时结束笔画；
单帧超过 `max_stamps_per_step`（默认 64）的跳变也会结束笔画，避免把整段
远距离移动都压进泥坯。

## 验收结果

固定局部点：`(0, 0, -0.075)`，法线 `(0, 0, -1)`，笔刷半径 24 mm，
强度 1.0；3 次 `dt = 0.1 s` 的雕刻（每次被限到 4 mm，累计 12 mm）。

| 项目 | 结果 |
| --- | --- |
| 网格 | 5282 顶点 / 10560 三角形（B 的 `ClayMesh` 默认参数） |
| 选中 / 改变顶点 | 97 / 97 |
| 单步深度 | 4.000 mm（`MAX_STEP_DEPTH_M`） |
| 累计深度 | 12.000 mm（`MAX_STROKE_DEPTH_M`） |
| 背面改变顶点 | 0 |
| 端面改变顶点 | 0 |
| 接缝 j=1 / j=n-1 权重 | 0.8916493274288931 / 0.8916493274288949 |
| 水密 / 可定向 / 自交 | true / true / false |
| 最小三角形面积 | 8.84e-6 m² |
| 侧面最小朝外点积 | 1.11e-6 > 0（没有翻面） |
| `update_clay` 耗时 | 0.90–1.65 ms，平均 1.37 ms |
| 复位 | 形状和位姿都恢复到初始值 |

`update_clay` 的耗时包含 `clay.set_vertices`、顶点法线重算和
`vis.update_geometry`，不含窗口渲染。显示对象 `clay_visual` 没有被重建。

验收时 `--verify-dir` 会生成 `01-initial.png`（初始）、`02-carve-final.png`
（12 mm 凹坑）、`03-backside.png`（转 180° 后的背面）和 `04-reset.png`
（复位），以及 `verification.json`。这些是报告用的证据，默认不提交仓库。

## 单元测试

`tests/test_sculpt.py` 覆盖：

- 拓扑：焊接/闭合/流形、邻接对称、法线朝外、端面中心法线。
- 选择：半径隔离、背面排除、端面排除、接缝左右权重一致、无拓扑退化。
- 雕刻：输入不变、方向（沿 `-n` 压入）、`dt` 缩放、单步/累计限幅、
  凹坑仍能选中周围顶点、非法输入无副作用。
- 几何：雕刻后水密、可定向、无自交、面积非零、侧面不翻面。
- 平滑：预更新副本、只影响笔刷内、凹坑变浅、单步收缩上界。
- 快照/笔画：快照独立、插值间距、法线单位化、零位移、无效命中和过大跳变
  都会结束笔画。

## 与队友的边界

- **B**：继续拥有 `render/clay_mesh.py`、`render/scene.py` 和显示/法线刷新；
  D 只返回新顶点数组。若要做笔刷高亮，使用 `BrushResult.changed_indices`。
- **E**：负责 `picking.py` 的拾取、场景→局部变换、`mesh_version` 校验、
  笔画开始/结束、撤销历史；调用 `apply_brush` 后每帧只提交一次。
- **C**：手丢失、暂停、切换工具时结束笔画；D 的 `StrokeSampler` 也会在
  无效输入时自动结束。
- **A**：参数名使用 `radius_m` / `strength` / `dt` / `tool`；工具值为
  `"carve"` 和 `"smooth"`。

## 已知限制 / 下一步

- 当前 demo 用固定局部点和固定法线，不需要摄像头；E 接入后应每帧用
  `PickResult` 重新拾取，命中点跟随凹坑中心（demo 的 verify 已经这样模拟）。
- 累计限幅按欧氏位移长度，不是严格沿法线的深度；对平滑更安全。
- `normal_cos_min=0.1` 对 12 mm 深、24 mm 半径的凹坑足够；更窄更深的坑
  可以把阈值调低，需在 D06 里做对比实验。
- 面数 5282/10560 时单次 `apply_brush` 远低于一帧；D06 会补不同半径、
  不同 `dt`、不同顶点数的 `benchmark.csv` 和曲线。
