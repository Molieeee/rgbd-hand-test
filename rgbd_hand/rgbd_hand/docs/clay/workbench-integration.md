# 泥坯工作台接入

2026年11月25日，`main.py` 与 `workbench_ui.py` 已接入同一份泥坯后端。默认无需摄像头，启动后同时打开 Tk 工作台和独立 Open3D 窗口；工作台中央显示 Open3D 渲染画面，不再用立方体投影代替泥坯。

## 启动

在业务目录运行：

```powershell
cd D:\rgbd-hand-test\rgbd_hand\rgbd_hand
$pythonExe = 'C:/Users/20981/.codex/environments/rgbd-hand-baseline/Scripts/python.exe'
& $pythonExe -B -X utf8 main.py
# 或：& $pythonExe -B -X utf8 workbench_ui.py
```

默认 `--scene clay --source demo`。`--source webcam`、`--source orbbec` 或 `--source replay --replay-dir 路径 --loop` 会打开相应输入源。输入源可在工作台侧栏切换，回放使用启动时指定的目录。失败时显示错误，仍可观察泥坯或切回 demo；不会悄悄把真实输入替换成模拟手。

`--no-ui` 仅打开独立 Open3D 场景；`--scene cube --source webcam` 保留原抓取演示。`workbench_ui.py --legacy-demo` 保留原 Canvas 立方体模拟，`--external` 保留原快照 UI 接口。

## 工作台操作

- 开始/暂停：控制输入处理与演示凹坑操作。暂停时仍可转动观察、复位，并看到新画面。
- 放置演示凹坑：在泥坯局部固定位置生成真实几何变形；重复点击从初始形状计算，不累加。
- 演示半径/深度：分别使用 cm 显示，后端使用米。点击“应用演示参数”后再放置凹坑；“详细数据”显示实际生效值。
- 左转/右转：改变泥坯位姿，保留局部变形；复位视角只改变观察相机。
- 重置场景：恢复初始形状及泥坯位姿。暂停时同样可执行。
- 鼠标拖动中央画面、滚轮：转动观察和缩放；独立三维窗口也可使用原生鼠标交互。
- 显示空间网格/手骨架：控制真实 Open3D 几何显示。

独立三维窗口快捷键为 `D` 凹坑、`R` 复位、`A/F` 转动、`V` 恢复视角、`Q/Esc` 退出。关闭工作台会停止并回收渲染进程；先关闭三维窗口时，后端释放资源，工作台保留最后画面并提示重新启动。

当前真实输入只驱动手骨架与相机预览，尚无射线命中、手势挖坑和平滑；界面对此明确标注。泥坯固定在桌面上，不应用立方体重力模型。左右手稳定跟踪优化仍在独立草稿 PR 中，本次没有自动合并。

## 模块与数据

`clay_runtime.py` 提供 `ClaySession`（运行状态、参数、当前网格）与 `run_clay()`（场景/输入源生命周期）。`main.py` 根据 scene/no-ui 选择后端或旧场景。`workbench_app.py` 创建 spawn 子进程，Tk 主线程以 50 ms 周期处理队列。

场景快照约 10 Hz，包含 `scene_kind=clay`、`scene_rgb`、`running`、`source`、`error`、`hands`、`camera_bgr`、`depth_enabled`、`mesh_version`、`changed_vertices`、`brush_radius_m` 和 `demo_depth_m`。队列只保留至多两帧，画面快照不会作为笔刷输入。输入 FPS 是输入处理速率，暂停时间包含在计时内，不作为渲染性能基准。

动作包括 `start/pause/reset/dent/turn/view/orbit/zoom/configure/source/quit`。顶点/位姿/法线更新仍通过 `ClayMesh` 和 `Scene.update_clay()`，复用 B02/B03 接口。Tk 中使用 Pillow 按原图宽高比显示实际帧，没有把原生窗口嵌入 Tk。

## 验证

```powershell
& $pythonExe -B -X utf8 -m unittest discover -s tests -v
& $pythonExe -B -X utf8 tests/verify_workbench_backend.py
```

单元测试覆盖原网格拓扑、暂停时拒绝修改、复位、参数及启动路由。集成脚本通过真实 Open3D 窗口、spawn 子进程和有界队列验证凹坑画面变化、暂停复位、观察动作、输入源错误恢复；相机输入使用明确的合成 21 点和 BGR 图像夹具，验证骨架/预览快照、数据结束后清除旧手势与预览、资源关闭和后端正常退出，不计作真实相机识别验收。

本机另在真实 Tk 窗口点击凹坑、暂停、暂停期间复位，确认状态与实际场景同步；以缺失回放目录启动工作台时，错误提示和圆柱桌面都正常显示。实际工作台 worker 另经过图像发送完成后正常退出的回归检查，避免提前终止队列 feeder 留下不完整帧。原 B01 采集脚本已显式使用 `--scene cube --no-ui`，避免默认入口变化后意外启动工作台。
