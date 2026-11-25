# rgbd_hand —— 深度相机 + MediaPipe 的三维手部交互

课程大作业的工程骨架。核心链路是 **真实深度 → 相机内参 → 手部三维坐标 → 三维交互**，
不是二维手势分类，也不是现成的 3D 模型浏览器。

```
相机 Frame ──► MediaPipe 21 关键点 ──► 3D 坐标 (m) ──► EMA 平滑
                                                          │
                      Open3D 虚拟手 ◄── 手势状态机 ◄───────┘
                             ▲              │
                             └──── 抓取/移动/旋转/释放 ◄── 立方体
```

## 运行

```bash
# 第一次在新机器上运行（需要 Python 3.10，mediapipe 0.10.21 有这个版本的 wheel）
conda create -n handgesture python=3.10 -y
conda activate handgesture
cd rgbd_hand/rgbd_hand  # 从仓库根目录进入项目
pip install -r requirements.txt

# 现在没有深度相机：用普通摄像头（MediaPipe world landmarks 提供米制 3D）
python main.py --source webcam

# 以后接上 Orbbec
python main.py --source orbbec --width 640 --height 480

# 回放录制的测试数据
python main.py --source replay --replay-dir data/demo --loop
```

按键：`q` / `ESC` 退出，`r` 把物体复位。

## 视角：第一人称“手伸进屏幕”

默认 `--view first`：3D 相机放在手和物体靠近用户的一侧、朝屏幕方向看，
相当于用你自己的眼睛看这只手。移动右手，虚拟手在画面里也向右移动，
不需要在脑子里做左右翻转。

这是真正的三维视角变换（相机绕 y 轴转 180°），**不是镜像**，
所以左右手的几何关系（拇指在哪一侧）仍然是对的。
摄像头预览窗口会同时做自拍镜像，保证两个窗口的左右方向一致；
不想要镜像可以加 `--no-mirror`。

`--view camera` 是之前的第三人称视角（相当于别人从摄像头位置看你），
调试关键点时可以和第一人称视角对比。

两个窗口的视场角是对齐的：`WebcamHand3D` 按 60° **水平**视场把图像位置换算成 3D，
而 Open3D 窗口默认竖直 FOV 是 60°（16:9 下水平约 91.5°）。如果不处理，虚拟手在 3D
画面里只有摄像头画面里一半的活动范围，看起来“手怎么动都够不到”。`Scene` 现在把
3D 相机的水平 FOV 也设成 60°，手在摄像头里移动多少，虚拟手在 3D 画面里就移动多少。

```bash
python main.py --source webcam --view first   # 默认
python main.py --source webcam --view camera
python main.py --source webcam --eye-distance 0.55   # 手看起来更远
python main.py --source webcam --eye-height 0.35     # 俯视桌面更明显
```

## 桌面与掉落

场景下方有一块 0.9 m × 0.6 m 的桌子，桌面在相机 y = 0.18 m 处。方块一开始**悬停**
在桌面上方 10 cm（相机 y = 0.05 m、z = 0.42 m），正好是虚拟手自然捏合的位置：
Webcam 模式下 MediaPipe 的 world landmarks 让捏合点比手腕靠近摄像头约 8 cm、
比手腕高约 8 cm，把方块放在这里，捏合点几乎就在方块中心，不用把手伸到画面边缘去够。

方块在第一次被抓住之前不受重力（HUD 里 `gravity: off`），方便先对准；抓住后改
`gravity: on`，松手就会自由落体，掉到桌面上停住（10 cm 的落差看得很清楚）。
物理只做课程演示需要的部分：重力（相机 +y 方向）+ 立方体最低点与桌面的碰撞，姿态在
落地前保持不变，所以斜着落地也不会插进桌面；桌面范围之外会落到更低的“地面”。

当捏合点进入抓取半径时，方块会变成**黄色**提示“可以抓了”；HUD 里也会显示
`dist`（捏合点到方块中心的距离）和当前半径，方便判断是距离不够还是没捏合。
抓取成功后，物块中心会在约 0.12 秒内平滑吸附到捏合点，避免手和物块之间保留抓取
瞬间的距离。方块落到桌面后，需要重新把捏合点移动到方块中心 8 cm 以内再抓取。

```bash
python main.py --source webcam --table-top 0.15        # 桌面抬高 3 cm，掉落后更好捡
python main.py --source webcam --object-pos 0 0.02 0.5 # 换一个悬浮位置
```

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--table-top` | `0.18` | 桌面上表面在相机系中的 y（米，向下为正） |
| `--object-pos` | `0 0.05 0.42` | 方块初始中心位置（相机坐标，米），默认悬停在桌面上方 |
| `--eye-distance` | `0.50` | 第一人称眼睛到方块的水平距离 |
| `--eye-height` | `0.30` | 眼睛高出方块的距离（俯视角） |
| `--grab-radius` | `0.08` | 捏合点到方块中心多近可以抓 |

按 `r` 会把方块放回初始悬浮位置，并清零速度和重力状态。

## 两种 3D 来源

| 模式 | 3D 坐标 | 用途 |
| --- | --- | --- |
| `WebcamHand3D` | 形状/姿态来自 `hand_world_landmarks`（米，相机轴对齐），平移由 2D 图像位置按固定深度 0.5 m 和近似 FOV 估计 | 没有深度相机时先把框架跑通 |
| `DepthHand3D` | 每个关键点取 5×5 邻域有效深度中值，再用内参反投影 | 最终方案，使用真实深度 |

两者接口一致，`main.py` 根据 `Frame.has_depth` 自动选择。
深度模式下，深度缺失的关节用 world landmarks 做整体平移对齐后补全。

## 目录

```
capture/         相机抽象 Frame / CameraSource，webcam、orbbec、回放、录制
perception/      MediaPipe 封装、2D->3D、EMA 平滑
interaction/     手势状态机（捏合/握拳/张掌）、抓取与位姿跟随、重力/桌面碰撞
render/          Open3D 场景：骨架、关节球、可抓取立方体、桌面
assets/          hand_landmarker.task（MediaPipe 模型，启动必需）
requirements.txt 依赖版本（Python 3.10）
main.py          主循环 + HUD
workbench_ui.py  独立的 Tkinter 手势交互工作台
```

## 无摄像头圆柱泥坯场景 B02 B03

在本目录运行 `python clay_demo.py`：显示可编辑圆柱与桌面，按 `D` 放置演示凹坑，`A/F` 转动泥坯，`R` 复位，`V` 恢复视角，鼠标拖动观察、滚轮缩放，`Q` 退出。使用 `--view camera` 可检查第二视角。该入口不需要摄像头，原 `main.py` 保留立方体抓取演示。

圆柱支持细分、局部顶点提交、场景位姿更新和版本标识。凹坑经过刷新和旋转后保留，供后续雕刻与拾取模块接入。接口、拓扑约定和截图见 [B02 B03 验收记录](docs/clay/README.md)。

## 手势交互工作台

在 `main.py` 所在目录运行 `python workbench_ui.py`，即可用内置模拟手和立方体预览界面。需要 Python 的 `tkinter` 模块，显示真实摄像头帧时还需要 Pillow。

`python workbench_ui.py --external` 会等待外部数据。接入时由调用方在主线程运行 Tk 窗口，通过 `publish_snapshot()` 提交快照，并用 `poll_actions()` 读取开始、暂停、复位等操作。现有 `main.py` 仍运行原来的 Open3D 场景。

## 「隔空捏泥人」网页方案

- `capture/` 和 `perception/` 可提供相机坐标系下的 21 个三维手部关键点，`interaction/gesture_fsm.py` 可提供捏合和张掌状态。
- 网页端计划用 Three.js 生成圆柱泥坯，把指尖或掌心从相机坐标系转换到场景坐标系，作为 `Raycaster` 的射线起点。射线命中网格后，把交点转换到网格局部坐标，按距离加权修改附近顶点。捏合向内刻出凹坑，张掌平滑局部凸起。修改后更新顶点缓冲、法线和包围体。
- 当前仓库已有 Python 手势输入链路、Open3D 抓取演示和 Tkinter UI 原型。浏览器数据接入与 Three.js 网格雕刻待实现。

## 关键实现说明

- **相机系**：+X 右、+Y 下、+Z 朝前（OpenCV / Orbbec 约定），单位米。
- **场景系**：通过 `M = diag(1, -1, -1)` 转到 Open3D 的 y 轴向上坐标。
- **视角**：第一人称相机放在场景 z 更小的一侧（靠近用户），朝 +z 看；
  因为场景 z 与相机 z 反号，这等价于从用户眼睛看手，右手仍在画面右侧。
  相机再高出方块 `eye_height`，视线对准方块中心，因此能看到桌面和手。
- **手势判定**（尺度无关）：`pinch_ratio = |p4 - p8| / |p9 - p0|`，
  进入捏合阈值 0.50（连续 2 帧），松开阈值 0.85（连续 10 帧）。
  松开阈值故意放得很宽：world landmarks 的深度（z）有噪声，捏着的时候
  三维指尖距离也可能偏大，阈值太接近就会“捏着捏着被判成松开”。
- **抓取防抖**：手部检测偶尔丢帧时，0.8 s 内保持抓取（物体停在最后位置，
  等手重新出现后继续跟随）；手重新出现后状态机需要几帧才重新判定为捏合，
  这 0.35 s 内也不释放，避免“一直捏着却突然掉下去”。
- **抓取**：捏合点与物体中心距离 < 8 cm 时抓取（方块变黄表示进入范围）。抓取
  瞬间保留相对旋转，并将物块中心的平移偏移在约 0.12 秒内平滑吸附到 pinch point；
  之后 `T_object = T_hand @ T_hand_object`，移动和旋转都跟随手掌坐标系。抓取期间
  桌面约束仍会限制物块高度，但不会因此主动取消抓取。
- **重力**：第一次抓住后启用（之前方块悬停）；松手后 `velocity.y += g·dt`，位置按速度积分；立方体的支撑半径取
  `half_extent · Σ|R[1, :]|`（旋转后最低点到中心的竖直距离），
  中心 y + 支撑半径碰到桌面高度时贴回桌面并做很小的反弹（恢复系数 0.15）。
- **平滑**：21 个点做 EMA；后续可无缝替换为 One Euro Filter。

## 测试数据 / 提交材料

```bash
# 录一段（带深度时会同时存 depth/*.npy 与 meta.json 中的 K）
python main.py --source webcam --record data/demo
```

`ReplaySource` 使用同样的目录格式，因此演示视频和“可执行程序 + 测试数据”可以直接
复用这套录制/回放。

## 拿到相机后要做的验证

1. `pyorbbecsdk` 的流配置、D2C 对齐、`get_depth_scale()`、内参是否与
   [capture/orbbec_source.py](capture/orbbec_source.py) 中的假设一致。
2. 深度噪声：关节深度用邻域中值 + 时域滤波（One Euro）。
3. 精度评估：固定几个姿态，用棋盘格/已知点比较反投影误差。
4. 相机标定：如果 SDK 内参与实际不符，用 OpenCV 重新标定 color 相机。
