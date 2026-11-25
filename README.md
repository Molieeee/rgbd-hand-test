# rgbd-hand-test

在项目 Python 3.10 环境运行泥坯工作台：

```powershell
cd D:\rgbd-hand-test\rgbd_hand\rgbd_hand
python main.py
```

默认无需摄像头，同时显示工作台与 Open3D 泥坯场景。`python workbench_ui.py` 为同一入口；`python main.py --source webcam` 接入真实手骨架，`python main.py --scene cube --source webcam` 保留原立方体抓取演示。

环境安装见 [运行说明](rgbd_hand/rgbd_hand/README.md)，界面操作见 [泥坯工作台接入](rgbd_hand/rgbd_hand/docs/clay/workbench-integration.md)。

