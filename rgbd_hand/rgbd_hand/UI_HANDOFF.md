# UI 交接接口说明

> 项目：`D:\rgbd_hand\rgbd_hand`  
> 面向对象：负责 UI 制作的同学  
> 本文基于当前代码整理。文档中标注为“已有接口”的内容可以直接使用；标注为“建议接口/待接入”的内容目前尚未在 `main.py` 中实现，属于推荐的数据边界。

---

## 1. 交接目标与边界

建议 UI 只依赖一份稳定的“每帧状态快照”，不要直接访问或修改以下内部逻辑：

- MediaPipe 手部检测与 handedness 解析
- 3D 手部坐标计算
- EMA 平滑
- 捏合手势状态机
- 抓取与物块跟随
- 重力与桌面碰撞

建议 UI 负责：

- 摄像头预览与叠加层
- 手部状态、抓取状态、距离和 FPS 的可视化
- 操作提示与按钮/快捷键
- 物块状态、重置按钮、退出按钮
- 可选的独立 3D 视图或状态面板

当前程序是两个窗口：

1. OpenCV 摄像头窗口：`RGBD Hand - camera`
2. Open3D 3D 场景窗口：`RGBD Hand - 3D Scene`

主循环目前是单线程，OpenCV、MediaPipe、交互逻辑和 Open3D 都在同一个循环中串行执行。UI 不应长时间阻塞该循环。

---

## 2. 运行入口与当前命令行参数

入口：

```text
D:\rgbd_hand\rgbd_hand\main.py
```

常用启动命令：

```powershell
cd D:\rgbd_hand\rgbd_hand
.\.venv\Scripts\python.exe .\main.py
```

常用参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--source` | `webcam` | 数据源：`webcam` / `orbbec` / `replay` |
| `--camera` | `0` | Webcam 设备号 |
| `--width` | `1280` | 输入宽度 |
| `--height` | `720` | 输入高度 |
| `--model` | 自动 | `hand_landmarker.task` 路径 |
| `--record` | `None` | 录制目录 |
| `--replay-dir` | `None` | 回放目录 |
| `--loop` | `False` | 回放循环 |
| `--ema` | `0.7` | EMA 平滑系数，越大越平滑 |
| `--grab-radius` | `0.08` | 抓取判定半径，单位米 |
| `--object-pos` | `0 0.05 0.42` | 物块初始位置，相机坐标，单位米 |
| `--table-top` | `0.18` | 桌面上表面相机 y，单位米 |
| `--view` | `first` | `first` 第一人称 / `camera` 摄像头视角 |
| `--eye-distance` | `0.50` | 第一人称视角眼睛到物块水平距离 |
| `--eye-height` | `0.30` | 第一人称视角眼睛高度偏移 |
| `--no-mirror` | `False` | 关闭摄像头预览自拍镜像 |

当前按键：

| 按键 | 行为 |
|---|---|
| `q` / `Esc` | 退出 |
| `r` | 物块复位，并重置抓取、重力和手势状态 |

> 目前“切换数据源、视角、是否镜像”只能在启动时通过命令行指定，尚没有运行时切换接口。UI 如果需要运行时切换，需要增加配置适配层并重建相机/场景。

---

## 3. 数据流

```text
CameraSource.read()
    -> Frame
    -> HandLandmarkerWrapper.detect()
    -> MediaPipe HandLandmarkerResult
    -> Handedness 解析 + 左右手槽位
    -> Hand3DProvider.to_camera3d()
    -> EMAFilter.update()
    -> GestureFSM.update()
    -> Manipulator.update() / update_missing()
    -> FallingObject.update()
    -> Scene.update_hands() / update_object() / render()
```

左右手槽位固定为：

```text
slot 0 = left
slot 1 = right
```

如果 handedness 缺失或重复，会在两个槽位中按空闲槽位降级分配。

---

## 4. 坐标、单位与颜色约定

### 4.1 相机坐标系

所有手部 3D 点、物块位姿、抓取距离都以相机坐标系表示：

```text
+X = 右
+Y = 下
+Z = 前（远离相机）
单位 = 米（m）
```

### 4.2 Open3D 场景坐标系

场景通过以下矩阵转换：

```python
SCENE_M = diag([1.0, -1.0, -1.0])
```

即场景：

```text
+X = 右
+Y = 上
+Z = 朝向观察者
```

UI 如果直接使用相机系数据，不需要自行做这个转换；调用 `Scene` 时由 `Scene` 内部完成。

### 4.3 图像与深度

- `Frame.color`：BGR，`uint8`，形状 `H x W x 3`
- `Frame.depth`：`uint16`，单位毫米，已对齐到 color 像素网格；可能为 `None`
- `Frame.K`：3x3 彩色相机内参；可能为 `None`
- `Frame.timestamp_ms`：毫秒，单调递增

### 4.4 镜像

`mirror` 只影响摄像头预览：

- MediaPipe 始终接收未翻转的原始 `frame.color`
- 3D 手部坐标始终使用原始 landmarks
- 预览通过 `cv2.flip(frame.color, 1)` 镜像
- 2D 骨架绘制横坐标通过 `1 - lm.x` 同步镜像

因此 `--view first` 和 `--no-mirror` 不应改变 3D 坐标，只改变预览方向。

---

## 5. 当前已有接口

## 5.1 `Frame` / `CameraSource`

文件：`capture/base.py`

```python
@dataclass
class Frame:
    color: np.ndarray
    depth: Optional[np.ndarray] = None
    K: Optional[np.ndarray] = None
    timestamp_ms: int = 0

    @property
    def has_depth(self) -> bool: ...
```

```python
class CameraSource(Protocol):
    def read(self) -> Optional[Frame]: ...
    def close(self) -> None: ...
```

### 相机实现

| 类 | 文件 | 构造参数 |
|---|---|---|
| `WebcamSource` | `capture/webcam_source.py` | `device=0, width=1280, height=720` |
| `ReplaySource` | `capture/replay_source.py` | `root, loop=False` |
| `OrbbecSource` | `capture/orbbec_source.py` | `width=640, height=480, fps=30, align_mode="hw"` |

### 录制

文件：`capture/recorder.py`

```python
Recorder(root, save_depth=True)
Recorder.write(frame: Frame) -> None
Recorder.close() -> None
```

---

## 5.2 `HandLandmarkerWrapper`

文件：`perception/hand_landmarker.py`

```python
HandLandmarkerWrapper(
    model_path=None,
    num_hands=2,
    min_detection_confidence=0.5,
    min_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
```

```python
def detect(color_bgr, timestamp_ms: int):
    """返回 MediaPipe HandLandmarkerResult"""
```

返回结果中 UI 可能用到：

```text
result.handedness              # 每只手的左右手分类和置信度
result.hand_landmarks          # 21 个归一化图像点，x/y/z
result.hand_world_landmarks    # 21 个以手为中心的米制 3D 点
```

> 不建议 UI 直接消费 MediaPipe 原始结果。优先使用下文的 `FrameSnapshot` 建议接口。

---

## 5.3 Handedness 工具

文件：`perception/handedness.py`

```python
handedness_label(categories) -> "left" | "right" | None
assign_hand_slots(labels, max_hands=2) -> list[int | None]
```

规则：

```text
left  -> slot 0
right -> slot 1
```

当标签缺失或重复时，未占用的槽位会被分配为兜底槽位。

---

## 5.4 3D 手部 provider

文件：`perception/hand3d.py`

统一接口：

```python
class Hand3DProvider:
    def to_camera3d(
        self,
        hand_landmarks,
        world_landmarks,
        depth,
        K,
    ) -> np.ndarray:
        """返回 shape=(21, 3), dtype=float64, 单位米, 相机坐标系"""
```

### `WebcamHand3D`

```python
WebcamHand3D(
    anchor_z=0.5,
    hfov_deg=60.0,
    aspect=16.0 / 9.0,
)
```

- 无真实深度时使用
- 手部形状来自 `hand_world_landmarks`
- 绝对位置由手腕 2D 位置、固定 0.5 m 深度和假设的 60° 水平 FOV 估计

### `DepthHand3D`

```python
DepthHand3D(
    median_ksize=5,
    min_depth_m=0.1,
    max_depth_m=3.0,
    fallback_anchor=(0.0, 0.0, 0.5),
)
```

- 有深度和相机内参时使用
- 对每个关键点取邻域深度中值，再用内参反投影
- 无效深度关节用 world landmarks 做平移对齐补全

---

## 5.5 `EMAFilter`

文件：`perception/filter.py`

```python
EMAFilter(alpha=0.75)
EMAFilter.update(points: np.ndarray) -> np.ndarray
EMAFilter.reset() -> None
```

- `points` 形状为 `(21, 3)`
- 主程序默认使用 `alpha=0.7`

---

## 5.6 `GestureState` / `GestureFSM`

文件：`interaction/gesture_fsm.py`

```python
@dataclass
class GestureState:
    pinching: bool = False
    fist: bool = False
    open_palm: bool = False
    pinch_ratio: float = 0.0
    extended_ratios: tuple[float, ...] = ()
    pinch_point: Optional[np.ndarray] = None
    hand_T: Optional[np.ndarray] = None
```

```python
GestureFSM(
    pinch_on=0.50,
    pinch_off=0.85,
    fist_ratio=1.10,
    palm_ratio=1.50,
    stable_frames=2,
    release_frames=10,
)
```

```python
GestureFSM.update(points: np.ndarray) -> GestureState
GestureFSM.reset() -> None
```

字段含义：

| 字段 | 含义 |
|---|---|
| `pinching` | 当前是否判定为捏合 |
| `fist` | 当前是否判定为握拳 |
| `open_palm` | 当前是否判定为张掌 |
| `pinch_ratio` | 拇指尖到食指尖的距离 / 手部尺度 |
| `extended_ratios` | 四个指尖到手腕的归一化距离 |
| `pinch_point` | 拇指尖和食指尖的中点，shape=(3,)，相机系，米 |
| `hand_T` | 4x4 手掌坐标系；捏合时原点为 `pinch_point` |

---

## 5.7 `Manipulator`

文件：`interaction/manipulator.py`

```python
Manipulator(
    grab_radius=0.08,
    hold_seconds=0.80,
    release_delay=0.35,
    snap_seconds=0.12,
)
```

```python
Manipulator.update(
    hand_T: np.ndarray,
    pinching: bool,
    object_T: np.ndarray,
    dt: float = 0.0,
) -> tuple[np.ndarray, bool]
```

返回：

```text
(object_T, grabbed)
```

```python
Manipulator.update_missing(
    object_T: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, bool]
```

```python
Manipulator.reset() -> None
```

状态：

| 属性 | 类型 | 含义 |
|---|---|---|
| `grabbed` | `bool` | 当前是否抓取物块 |
| `T_hand_object` | `np.ndarray | None` | 物块相对手掌的 4x4 变换 |

行为：

- 物块中心进入抓取半径且处于捏合状态时抓取
- 抓取后约 `0.12 s` 把物块中心平滑吸附到 `pinch_point`
- 抓取期间物块跟随 `hand_T`
- 短暂丢手保持 `0.80 s`
- 短暂非捏合保持 `0.35 s`

---

## 5.8 `FallingObject`

文件：`interaction/physics.py`

```python
FallingObject(
    table_top=0.20,
    table_x=(-0.45, 0.45),
    table_z=(0.25, 0.85),
    floor_top=0.85,
    gravity=9.81,
    restitution=0.15,
    contact_epsilon=1e-3,
)
```

```python
FallingObject.update(
    object_T: np.ndarray,
    grabbed: bool,
    dt: float,
    half_extent: float,
) -> np.ndarray
```

```python
FallingObject.reset() -> None
```

输出仍为相机坐标系下的 4x4 物块位姿。

可用状态：

| 属性 | 含义 |
|---|---|
| `active` | 是否已经首次抓取并启用重力 |
| `velocity` | 当前速度，shape=(3,) |
| `table_top` | 桌面高度 |
| `floor_top` | 桌面范围外的概念地板高度 |

---

## 5.9 `Scene`

文件：`render/scene.py`

```python
Scene(
    window_name="RGBD Hand - 3D Scene",
    width=1280,
    height=720,
    max_hands=2,
    joint_radius=0.009,
    object_size=0.05,
    focus_distance=0.5,
    first_person=True,
    eye_distance=0.50,
    eye_height=0.30,
    look_at_y=0.0,
    table_top=0.18,
    table_size=(0.9, 0.60),
    table_center_z=0.55,
    hfov_deg=60.0,
)
```

```python
Scene.update_hands(
    hands: Sequence[np.ndarray],
    pinching: Sequence[bool] | None = None,
) -> None
```

- 每个手部数组形状 `(21, 3)`，相机系，单位米
- 当前实现会把传入列表按顺序放入可视化槽位

```python
Scene.update_object(
    object_T_cam: np.ndarray,
    grabbed: bool = False,
    near: bool = False,
) -> None
```

- `object_T_cam` 为 4x4 相机系齐次变换
- `grabbed=True` 时物块显示绿色
- `grabbed=False, near=True` 时显示黄色
- 其他情况显示红色

```python
Scene.render() -> bool
Scene.close() -> None
```

`render()` 内部调用：

```python
vis.poll_events()
vis.update_renderer()
```

返回值 `False` 表示 Open3D 窗口已关闭。

---

## 6. 主循环中当前可用于 UI 的每帧数据

文件：`main.py`

主循环中以下变量是 UI 最关心的数据：

| 变量 | 类型/形状 | 来源 | 用途 |
|---|---|---|---|
| `frame` | `Frame` | `source.read()` | 图像、深度、时间戳 |
| `fps` | `float` | 主循环平滑估算 | HUD |
| `dt` | `float` | 帧间隔，秒 | 动画/状态显示 |
| `hands` | `list[tuple]` | MediaPipe + 3D + FSM | 手部显示 |
| `primary` | `tuple | None` | 主手选择 | HUD/抓取 |
| `primary_slot` | `int | None` | 0=左，1=右 | 主手标识 |
| `object_T` | `np.ndarray (4,4)` | 抓取/物理 | 物块位姿 |
| `grabbed` | `bool` | `Manipulator` | 绿色状态 |
| `near` | `bool` | 抓取距离判断 | 黄色状态 |
| `pinch_dist` | `float | None` | 捏合点到物块中心距离 | HUD |
| `pinch_ratio` | `float` | 主手手势状态 | HUD |
| `falling.active` | `bool` | 重力学 | gravity on/off |
| `mode` | `"GRAB" / "PINCH" / "IDLE"` | 主循环组合得到 | HUD |

当前 `hands` 中每个元素的结构为：

```python
(landmarks, points_cam, gesture_state)
```

其中：

```text
landmarks       : MediaPipe NormalizedLandmark 列表，长度 21
points_cam      : np.ndarray, shape=(21, 3), 相机系，米
gesture_state   : GestureState
```

当前 HUD 字段：

```text
source: webcam / orbbec / replay
depth: ON / OFF
fps
hands: 检测到的手数量
pinch_ratio
state: GRAB / PINCH / IDLE
dist: 捏合点到物块中心距离 / 当前抓取半径
object: 物块中心相机坐标
gravity: on / off
```

---

## 7. 建议的 UI 边界：`FrameSnapshot`

> 以下接口目前尚未在代码中实现，是建议给 UI 使用的稳定数据契约。实现时建议新增 `ui/contract.py` 或 `ui_bridge.py`，不要修改感知/交互内部接口。

```python
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

Handedness = Literal["left", "right"]
InteractionMode = Literal["IDLE", "PINCH", "GRAB"]


@dataclass(frozen=True)
class HandSnapshot:
    slot: int                       # 0=left, 1=right
    handedness: Optional[Handedness]
    landmarks_2d: tuple[tuple[float, float, float], ...]  # 21 个归一化点
    points_3d_m: tuple[tuple[float, float, float], ...]   # 21 个相机系点
    pinching: bool
    fist: bool
    open_palm: bool
    pinch_ratio: float
    pinch_point_m: Optional[tuple[float, float, float]]
    hand_T: Optional[tuple[tuple[float, ...], ...]]


@dataclass(frozen=True)
class ObjectSnapshot:
    pose_cam: tuple[tuple[float, ...], ...]  # 4x4
    center_m: tuple[float, float, float]
    grabbed: bool
    near: bool
    gravity_active: bool


@dataclass(frozen=True)
class FrameSnapshot:
    timestamp_ms: int
    source: str
    fps: float
    dt_s: float
    depth_enabled: bool
    view: Literal["first", "camera"]
    mirrored: bool
    primary_slot: Optional[int]
    mode: InteractionMode
    pinch_dist_m: Optional[float]
    grab_radius_m: float
    hands: tuple[HandSnapshot, ...]
    object: ObjectSnapshot
    camera_bgr: Optional[np.ndarray] = None  # UI 跨线程使用时应复制
```

推荐约定：

- `hands` 按 `slot` 排序，左手在前、右手在后
- 未知 handedness 时保留实际分配的 `slot`，`handedness=None`
- `points_3d_m` 使用相机坐标系，不需要 UI 再做 `SCENE_M` 转换
- `camera_bgr` 交给 UI 前建议 `copy()`，避免主循环复用/修改
- UI 只能读取快照，不直接写 `object_T`、`GestureState` 或 MediaPipe 数据

---

## 8. 建议的 UI 回调接口

建议用一个最小回调协议隔离主循环与 UI：

```python
from typing import Protocol, Optional


class UIAction(Protocol):
    reset_object: bool
    quit: bool


class UIBridge(Protocol):
    def on_frame(self, snapshot: FrameSnapshot) -> Optional[UIAction]:
        """每帧调用一次，UI 不应阻塞超过一帧预算。"""
        ...
```

主循环建议插入位置：

```text
GestureFSM.update
-> Manipulator.update
-> FallingObject.update
-> 构建 FrameSnapshot
-> UIBridge.on_frame(snapshot)
-> Scene.update_hands / update_object / render
```

动作处理建议：

```python
action = ui.on_frame(snapshot)
if action and action.reset_object:
    object_T = object_T0.copy()
    primary_slot = None
    manipulator.reset()
    falling.reset()
    for fsm in fsms:
        fsm.reset()

if action and action.quit:
    break
```

> 注意：当前 `main.py` 尚未实现 `FrameSnapshot` 和 `UIBridge`。如果 UI 同学需要直接接入，需要先把上述适配层实现到主循环中。

---

## 9. UI 可展示的推荐状态

### 9.1 顶部状态栏

```text
数据源：Webcam / Orbbec / Replay
深度：ON / OFF
FPS：xx.x
检测手数：0 / 1 / 2
视角：第一人称 / 摄像头视角
镜像：ON / OFF
```

### 9.2 手部状态

每只手建议显示：

```text
左手 / 右手
PINCH / IDLE
pinch_ratio
pinch_point 相机坐标
```

### 9.3 交互状态

```text
IDLE：未检测到捏合
PINCH：已捏合，但未抓取物块
GRAB：已抓住物块
near：物块在抓取范围内
dist：捏合点到物块中心距离
grab_radius：当前抓取半径
```

颜色语义建议与现有 Open3D 保持一致：

| 状态 | 颜色 |
|---|---|
| 可抓取但未抓取 | 黄色 |
| 已抓取 | 绿色 |
| 普通状态 | 红色 / 蓝色 |

### 9.4 物块状态

```text
中心坐标 (x, y, z)，相机系，米
4x4 位姿（高级/调试面板）
grabbed
gravity_active
```

### 9.5 操作区

当前可用的最小动作：

```text
重置物块
退出
```

如果 UI 要新增：

```text
开始/停止录制
切换数据源
切换视角
调整 EMA
调整抓取半径
```

需要配套改主循环或重建相机/场景，不能只改 UI 文本。

---

## 10. 三种 UI 接入方案

### 方案 A：OpenCV 叠加层（最小改动）

在现有 `display` 图像上绘制 UI：

- 改动小
- 适合快速演示
- 与现有 HUD 逻辑兼容
- 但 UI 表现力受 OpenCV 限制

建议插入点：`main.py` 中 `display = ...` 之后、`cv2.imshow()` 之前。

### 方案 B：独立 UI + `FrameSnapshot`（推荐）

- UI 使用 PyQt/Tk/Web 等独立界面
- 主循环每帧构建不可变快照
- UI 只读取快照并返回动作
- 核心感知和交互逻辑保持独立
- 需要新增轻量适配层

推荐用于正式 UI 制作。

### 方案 C：完全接管 Open3D 窗口

- UI 自己实现 3D 渲染
- 直接消费 `points_3d_m` 和 `object.pose_cam`
- 灵活但工作量大
- 不建议与现有 Open3D 窗口同时运行两个事件循环

---

## 11. 文件与职责对照

| 路径 | 职责 | UI 是否直接依赖 |
|---|---|---|
| `capture/base.py` | `Frame` / 相机接口 | 通过快照间接依赖 |
| `capture/webcam_source.py` | RGB 摄像头 | 否 |
| `capture/orbbec_source.py` | RGB-D 摄像头 | 否 |
| `capture/replay_source.py` | 回放 | 否 |
| `capture/recorder.py` | 录制 | 可选 |
| `perception/hand_landmarker.py` | MediaPipe 检测 | 否 |
| `perception/handedness.py` | 左右手标签/槽位 | 否 |
| `perception/hand3d.py` | 3D 坐标 | 否 |
| `perception/filter.py` | EMA 平滑 | 否 |
| `interaction/gesture_fsm.py` | 手势状态机 | 通过 `GestureState` 间接依赖 |
| `interaction/manipulator.py` | 抓取/跟随/释放 | 通过快照间接依赖 |
| `interaction/physics.py` | 重力/桌面碰撞 | 通过快照间接依赖 |
| `render/scene.py` | Open3D 场景 | 可选 |
| `main.py` | 主循环/HUD | 需要接入点 |

---

## 12. 交接验收清单

UI 同学完成后至少验证：

- [ ] 摄像头预览方向与 `mirror` 参数一致
- [ ] `hands` 数量正确显示
- [ ] 左手/右手标识不因检测顺序变化而互换
- [ ] `pinch_ratio`、`dist`、`state` 实时更新
- [ ] 物块进入抓取范围的黄色提示正确
- [ ] 抓取成功后的绿色提示正确
- [ ] 重置按钮/快捷键能复位物块和交互状态
- [ ] 退出按钮/快捷键能正常释放相机、MediaPipe 和 Open3D 资源
- [ ] `--view first` 与 `--no-mirror` 下 3D 坐标不因镜像改变
- [ ] UI 不会阻塞主循环导致 FPS 明显下降
- [ ] 两只手同时出现时，主手与物块状态不互相串用

---

## 13. 已知限制

1. **Webcam 模式没有真实深度**  
   3D 绝对位置使用固定深度和近似 FOV，精度依赖摄像头和手部距离。

2. **镜像只影响显示**  
   不参与 3D 坐标；UI 不要对 `points_3d_m` 再次做镜像。

3. **当前主循环是单线程**  
   UI 回调中不要执行耗时推理、网络请求或大规模磁盘 IO。

4. **Open3D 与 OpenCV 都有自己的窗口事件循环**  
   如果改用 Qt/Tk，需要避免同时阻塞两套事件循环。

5. **运行时切换数据源/视角尚未实现**  
   当前只能通过启动参数选择；UI 若需要运行时切换，需要新增重建逻辑。

6. **当前建议的 `FrameSnapshot`/`UIBridge` 尚未接入**  
   需要和主循环负责人一起完成适配层后，UI 才能按该接口直接开发。

---

## 14. 最小对接代码示例

### 读取单帧并调用现有管线

```python
from capture import WebcamSource
from perception import HandLandmarkerWrapper, WebcamHand3D, EMAFilter
from interaction import GestureFSM

source = WebcamSource(0, 1280, 720)
landmarker = HandLandmarkerWrapper()
provider = WebcamHand3D(anchor_z=0.5, aspect=16 / 9)
filter_ = EMAFilter(alpha=0.7)
fsm = GestureFSM()

try:
    frame = source.read()
    if frame is not None:
        result = landmarker.detect(frame.color, frame.timestamp_ms)
        if result.hand_landmarks and result.hand_world_landmarks:
            landmarks = result.hand_landmarks[0]
            world = result.hand_world_landmarks[0]
            points = provider.to_camera3d(landmarks, world, frame.depth, frame.K)
            points = filter_.update(points)
            state = fsm.update(points)

            print("pinching:", state.pinching)
            print("pinch_point:", state.pinch_point)
            print("hand_T:\n", state.hand_T)
finally:
    source.close()
    landmarker.close()
```

### UI 每帧读取建议快照

```python
def ui_loop(ui: UIBridge, snapshot_provider):
    while True:
        snapshot = snapshot_provider()
        if snapshot is None:
            break

        action = ui.on_frame(snapshot)

        if action and action.reset_object:
            snapshot_provider.reset_object()

        if action and action.quit:
            break
```

---

## 15. 交接结论

UI 同学最需要依赖的是：

1. `FrameSnapshot`
2. `HandSnapshot`
3. `ObjectSnapshot`
4. `UIBridge.on_frame()`
5. `UIAction.reset_object / quit`

其余 MediaPipe、3D 反投影、EMA、手势状态机、抓取和物理细节应保持在核心模块内部。这样 UI 可以独立迭代，不会破坏手部识别和抓取逻辑。