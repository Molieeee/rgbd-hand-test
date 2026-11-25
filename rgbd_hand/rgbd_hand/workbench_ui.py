"""泥坯工作台 UI；默认通过 workbench_app 连接真实 Open3D 后端。

运行 python workbench_ui.py 或 python main.py 均启动泥坯工作台。
--legacy-demo 保留原立方体 Canvas 模拟；--external 保留快照接入原型。

运行：python workbench_ui.py
默认需要项目 Open3D/NumPy 环境、tkinter 与 Pillow；原立方体 Canvas 原型仅需 tkinter。

接入方式（Tk 窗口及 mainloop 必须在主线程运行）：
    app = HandWorkbench(demo=False)
    # 算法线程每帧调用，接受 UI_HANDOFF.md 中的快照对象或同名字段的 dict：
    app.publish_snapshot(snapshot)
    # 算法线程定期读取；UI 本身不改变算法对象：
    for action in app.poll_actions():
        ...  # action['type']: start / pause / reset / configure / source / quit
    app.mainloop()

publish_snapshot 保留最新一帧，不累积排队；camera_bgr 会在提交时复制。
configure 的 grab_radius_m/grid/skeleton 是请求，后端自行接收并应用。
原型的 --external 模式需由集成代码提交数据。默认模式由 workbench_app 启动后端。
只有原立方体原型使用 Canvas 投影；泥坯工作台显示 scene_rgb 实际渲染帧。
Windows 独立启动时自动启用系统 DPI 感知；字号以 point 绘制，布局按 DPI 缩放。
更改系统缩放后请重启窗口；嵌入已有 Tk 程序时保留宿主的 DPI 模式。
"""
from __future__ import annotations

import argparse
import math
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import font as tkfont
from collections.abc import Mapping


BG = "#101419"
PANEL = "#191f27"
CARD = "#202833"
LINE = "#303b48"
TEXT = "#edf2f7"
MUTED = "#93a2b5"
ACCENT = "#7de0c3"
YELLOW = "#f0cd7b"
FONT = "Microsoft YaHei UI"
# MediaPipe 的手掌与五指连接；不用依赖 perception 模块。
CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14),
    (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


def field(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, Mapping) else getattr(obj, name, default)


def shade(color, factor):
    return "#" + "".join(f"{max(0, min(255, int(int(color[i:i + 2], 16) * factor))):02x}"
                         for i in (1, 3, 5))


def enable_high_dpi():
    """在第一个 Tk 窗口创建前启用系统 DPI 感知；不覆盖宿主已有设置。"""
    if sys.platform != "win32" or tk._default_root is not None:
        return
    import ctypes
    try:
        set_context = ctypes.windll.user32.SetProcessDpiAwarenessContext
        set_context.argtypes = [ctypes.c_void_p]
        set_context.restype = ctypes.c_int
        if set_context(ctypes.c_void_p(-2)):
            return
        # ERROR_ACCESS_DENIED 表示宿主已经设置过 DPI 模式，应保留宿主设置。
        if ctypes.windll.kernel32.GetLastError() == 5:
            return
    except AttributeError:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        ctypes.windll.user32.SetProcessDPIAware()


def demo_hand(x=0.0, y=0.0, z=0.46, pinching=False):
    points = [(0.0, 0.105, 0.01)]
    fingers = [
        [(-.045, .075), (-.075, .048), (-.09, .017), (-.085, -.01)],
        [(-.035, .025), (-.04, -.02), (-.039, -.059), (-.035, -.087)],
        [(0, .015), (.002, -.04), (.003, -.084), (.004, -.115)],
        [(.03, .025), (.04, -.025), (.043, -.064), (.044, -.09)],
        [(.057, .046), (.073, .005), (.081, -.023), (.084, -.047)],
    ]
    for finger in fingers:
        for px, py in finger:
            points.append((px, py, -.008))
    if pinching:
        points[3] = (-.036, -.018, -.025)
        points[4] = (-.018, -.028, -.03)
        points[6] = (-.032, -.025, -.02)
        points[7] = (-.026, -.035, -.028)
        points[8] = (-.016, -.029, -.03)
    return [(px + x, py + y, pz + z) for px, py, pz in points]


class HandWorkbench(tk.Tk):
    """可独立运行，也可由现有主程序导入的窗口。"""

    def __init__(self, demo=True, clay_mode=False):
        enable_high_dpi()
        super().__init__()
        self.ui_scale = max(1.0, round(self.winfo_fpixels("1i") / 96.0, 2))
        self.tk.call("tk", "scaling", self.ui_scale * 96.0 / 72.0)
        available = set(tkfont.families(self))
        self.font_family = next((name for name in (FONT, "Microsoft YaHei", "Segoe UI")
                                 if name in available), "TkDefaultFont")
        self.title("Handspace · 手势交互工作台")
        self.configure(bg=BG)
        width = min(self.px(1440), self.winfo_screenwidth() - self.px(80))
        height = min(self.px(900), self.winfo_screenheight() - self.px(100))
        self.geometry(f"{width}x{height}+{self.px(30)}+{self.px(30)}")
        self.minsize(min(self.px(1000), width), min(self.px(680), height))
        self.demo = demo
        self.clay_mode = clay_mode
        self.running = demo
        self.snapshot = None
        self._pending = None
        self._lock = threading.Lock()
        self._actions = queue.SimpleQueue()
        self._last_frame = 0.0
        self._closed = False
        self._timer = None
        self._elapsed = 0.0
        self._clock = time.perf_counter()
        self._drag = None
        self._hover = False
        self._cube_hitbox = None
        self._camera_image = None
        self._camera_frame = None
        self._camera_error = False
        self._scene_image = None
        self._scene_image_key = None
        self._scene_frame = None
        self._settings = None
        self._toast = ""
        self._toast_until = 0.0
        self.panel_visible = True
        self.radius = tk.DoubleVar(value=8.0)
        self.show_grid = tk.BooleanVar(value=True)
        self.show_hand = tk.BooleanVar(value=True)
        self.auto_demo = tk.BooleanVar(value=True)
        self.brush_radius = tk.DoubleVar(value=4.875)
        self.demo_depth = tk.DoubleVar(value=2.2)
        self.input_source = tk.StringVar(value="demo")
        self.angle = 0.25
        self.object_pos = [0.0, .05, .42]
        self._build()
        self._scale_layout(self)
        self.bind("<space>", self._space)
        self.bind("r", lambda event: self.reset_scene())
        self.bind("R", lambda event: self.reset_scene())
        self.bind("<Tab>", self._tab)
        self.bind("<Escape>", lambda event: self.close())
        self.protocol("WM_DELETE_WINDOW", self.close)
        self._tick()

    def publish_snapshot(self, snapshot):
        """线程安全地提交快照；除图像外，调用方应保持快照只读。"""
        camera = field(snapshot, "camera_bgr")
        camera = camera.copy() if camera is not None else None
        with self._lock:
            self._pending = (snapshot, camera, time.perf_counter())

    def poll_actions(self):
        """非阻塞取出用户操作，供算法线程处理。"""
        actions = []
        while True:
            try:
                actions.append(self._actions.get_nowait())
            except queue.Empty:
                return actions

    def _action(self, kind, **values):
        self._actions.put({"type": kind, **values})

    def px(self, value):
        return round(value * self.ui_scale)

    def font(self, size=10, weight="normal"):
        return (self.font_family, max(10, size), weight)

    def _layout_distance(self, widget, value):
        # Tk 默认 Canvas 尺寸用厘米（如 7c），已经随 DPI 换算，不能再次放大。
        try:
            return self.px(float(value))
        except (TypeError, ValueError):
            return widget.winfo_pixels(value)

    def _scale_layout(self, parent):
        """字体由 Tk 以 point 绘制，像素尺寸与布局间距单独按相同比例换算。"""
        for widget in parent.winfo_children():
            options = {}
            for name in ("padx", "pady", "wraplength", "highlightthickness", "borderwidth"):
                if name in widget.keys():
                    options[name] = self._layout_distance(widget, widget.cget(name))
            if isinstance(widget, (tk.Frame, tk.Canvas, tk.Scrollbar)):
                for name in ("width", "height"):
                    if name in widget.keys():
                        options[name] = self._layout_distance(widget, widget.cget(name))
            widget.configure(**options)
            if widget.winfo_manager() == "pack":
                info = widget.pack_info()
                padding = {}
                for name in ("padx", "pady", "ipadx", "ipady"):
                    raw = info[name]
                    values = widget.tk.splitlist(raw) if isinstance(raw, str) else raw if isinstance(raw, tuple) else (raw,)
                    padding[name] = tuple(self.px(widget.winfo_pixels(v)) for v in values)
                widget.pack_configure(**padding)
            if isinstance(widget, tk.Canvas):
                widget.scale("all", 0, 0, self.ui_scale, self.ui_scale)
                for item in widget.find_all():
                    if widget.type(item) in ("line", "polygon", "oval", "rectangle"):
                        widget.itemconfigure(item, width=self.px(float(widget.itemcget(item, "width"))))
            self._scale_layout(widget)

    def label(self, parent, text="", size=10, color=TEXT, bg=PANEL, **kw):
        return tk.Label(parent, text=text, font=self.font(size), fg=color, bg=bg, **kw)

    def button(self, parent, text, command, primary=False, compact=False):
        return tk.Button(parent, text=text, command=command,
                         font=self.font(10, "bold" if primary else "normal"),
                         bg=ACCENT if primary else CARD, fg=BG if primary else TEXT,
                         activebackground="#a2ead6" if primary else LINE,
                         activeforeground=BG if primary else TEXT,
                         relief="flat", bd=0, cursor="hand2", takefocus=True,
                         padx=12 if compact else 18, pady=8, highlightthickness=0)

    def _build(self):
        top = tk.Frame(self, bg=PANEL, height=78)
        top.pack(fill="x")
        top.pack_propagate(False)
        mark = tk.Canvas(top, width=42, height=42, bg=PANEL, highlightthickness=0)
        mark.pack(side="left", padx=(24, 12))
        for x, y, h in [(10, 15, 17), (18, 8, 24), (26, 12, 20)]:
            mark.create_line(x, y, x, y + h, fill=ACCENT, width=4, capstyle="round")
        mark.create_line(10, 32, 28, 32, 34, 23, fill=ACCENT, width=4, capstyle="round")
        branding = tk.Frame(top, bg=PANEL)
        branding.pack(side="left")
        self.label(branding, "HANDSPACE", 16).pack(anchor="w")
        self.label(branding, "手势交互工作台", 9, MUTED).pack(anchor="w")
        self.button(top, "设置", self.open_settings, compact=True).pack(side="right", padx=(8, 24))
        self.run_button = self.button(top, "暂停演示" if self.demo else "开始", self.toggle_run, True)
        self.run_button.pack(side="right", padx=(12, 0))
        self.source_badge = self.label(top, "●  演示模式" if self.demo else "○  等待外部输入", 10, YELLOW)
        self.source_badge.pack(side="right", padx=14)
        tk.Frame(self, height=1, bg=LINE).pack(fill="x")

        self.body = tk.Frame(self, bg=BG)
        self.body.pack(fill="both", expand=True, padx=22, pady=(20, 14))
        self.side = tk.Frame(self.body, bg=PANEL, width=300)
        self.side.pack(side="right", fill="y", padx=(16, 0))
        self.side.pack_propagate(False)
        self.center = tk.Frame(self.body, bg=BG)
        self.center.pack(side="left", fill="both", expand=True)
        heading = tk.Frame(self.center, bg=BG)
        heading.pack(fill="x", pady=(0, 14))
        titles = tk.Frame(heading, bg=BG)
        titles.pack(side="left")
        self.label(titles, "交互空间", 21, bg=BG).pack(anchor="w")
        self.label(titles, "用双手，探索屏幕里的空间。", 10, MUTED, BG).pack(anchor="w", pady=(4, 0))
        self.panel_button = self.button(heading, "收起面板  ›", self.toggle_panel, compact=True)
        self.panel_button.pack(side="right", anchor="s")

        frame = tk.Frame(self.center, bg=LINE, padx=1, pady=1)
        frame.pack(fill="both", expand=True)
        tools = tk.Frame(frame, bg=PANEL, height=44)
        tools.pack(fill="x")
        self.label(tools, "  ◇  泥坯与桌面" if self.clay_mode else "  ◇  方块交互", 10).pack(side="left", padx=10, pady=12)
        self.view_label = self.label(tools, "空间预览  /  演示" if self.demo else "空间预览  /  外部输入", 9, MUTED)
        self.view_label.pack(side="right", padx=16)
        self.scene = tk.Canvas(frame, width=394, height=276, bg="#131b24", highlightthickness=0, cursor="hand2")
        self.scene.pack(fill="both", expand=True)
        self.scene.bind("<ButtonPress-1>", self._press)
        self.scene.bind("<B1-Motion>", self._move)
        self.scene.bind("<ButtonRelease-1>", self._release)
        self.scene.bind("<Motion>", self._motion)
        self.scene.bind("<MouseWheel>", self._wheel)
        self.scene.bind("<Configure>", lambda event: self.draw_scene())
        footer = tk.Frame(frame, bg=PANEL)
        footer.pack(fill="x")
        self.label(footer, "  拖动观察 · 滚轮缩放 · Open3D 同步画面" if self.clay_mode else "  拖动物体 · 滚轮旋转", 9, MUTED).pack(side="left", padx=10, pady=10)
        self.button(footer, "重置场景  R", self.reset_scene, compact=True).pack(side="right", padx=8, pady=5)
        self._build_sidebar()

        bottom = tk.Frame(self, bg=PANEL, height=34)
        bottom.pack(fill="x", side="bottom")
        self.status = self.label(bottom, "", 9, MUTED)
        self.status.pack(side="left", padx=22, pady=8)
        self.label(bottom, "空格  开始 / 暂停     ·     Tab  侧栏", 9, MUTED).pack(side="right", padx=22)

    def _build_sidebar(self):
        self.side_scroll = tk.Canvas(self.side, bg=PANEL, highlightthickness=0, width=288, height=276)
        scrollbar = tk.Scrollbar(self.side, orient="vertical", command=self.side_scroll.yview,
                                 width=10, bg=PANEL, troughcolor=BG, bd=0)
        scrollbar.pack(side="right", fill="y")
        self.side_scroll.pack(side="left", fill="both", expand=True)
        self.side_scroll.configure(yscrollcommand=scrollbar.set)
        self.side_content = panel = tk.Frame(self.side_scroll, bg=PANEL)
        window = self.side_scroll.create_window(0, 0, anchor="nw", window=panel)
        panel.bind("<Configure>", lambda event: self.side_scroll.configure(scrollregion=self.side_scroll.bbox("all")))
        self.side_scroll.bind("<Configure>", lambda event: self.side_scroll.itemconfigure(window, width=event.width))
        self.bind("<MouseWheel>", self._sidebar_wheel, add="+")
        self.label(panel, "观察与控制", 12).pack(anchor="w", padx=18, pady=(18, 12))
        self.preview = tk.Canvas(panel, height=138, bg="#10171e", highlightthickness=0)
        self.preview.pack(fill="x", padx=16)
        self.preview.bind("<Configure>", lambda event: self.draw_preview())
        self.preview_caption = self.label(panel, "示意画面 · 未使用摄像头", 9, MUTED)
        self.preview_caption.pack(anchor="w", padx=18, pady=(7, 14))
        self._rule()
        row = tk.Frame(panel, bg=PANEL)
        row.pack(fill="x", padx=18, pady=(14, 10))
        self.label(row, "实时状态", 11).pack(side="left")
        self.live_label = self.label(row, "●  模拟数据", 9, ACCENT)
        self.live_label.pack(side="right")
        self.hand_label = self._stat("追踪手部", "右手 · 示意")
        self.gesture_label = self._stat("当前手势", "张掌")
        self.object_label = self._stat("泥坯状态" if self.clay_mode else "物体状态", "等待交互")
        self.distance_label = self._stat("拾取状态" if self.clay_mode else "抓取距离", "—")
        self._rule(pady=12)
        self.label(panel, "下一步", 10, MUTED).pack(anchor="w", padx=18)
        self.hint_label = self.label(panel, "", 11, TEXT, justify="left", wraplength=240)
        self.hint_label.pack(anchor="w", padx=18, pady=(8, 6))
        self.hint_detail = self.label(panel, "", 9, MUTED, justify="left", wraplength=240)
        self.hint_detail.pack(anchor="w", padx=18, pady=(0, 12))
        self._rule()
        if self.clay_mode:
            self.label(panel, "输入源", 10, MUTED).pack(anchor="w", padx=18, pady=(12, 4))
            menu = tk.OptionMenu(panel, self.input_source, "demo", "webcam", "orbbec", "replay",
                                 command=lambda value: self._action("source", source=value))
            menu.configure(bg=CARD, fg=TEXT, highlightthickness=0, font=self.font(10))
            menu.pack(fill="x", padx=18)
            self.label(panel, "demo 无相机；replay 使用启动时指定的目录", 8, MUTED,
                       wraplength=240).pack(anchor="w", padx=18, pady=4)
            self.dent_button = self.button(panel, "放置演示凹坑", lambda: self._action("dent"), True)
            self.dent_button.pack(fill="x", padx=18, pady=(8, 4))
            row = tk.Frame(panel, bg=PANEL)
            row.pack(fill="x", padx=18)
            self.button(row, "左转", lambda: self._action("turn", angle=-.15), compact=True).pack(side="left")
            self.button(row, "右转", lambda: self._action("turn", angle=.15), compact=True).pack(side="left", padx=4)
            self.button(row, "复位视角", lambda: self._action("view"), compact=True).pack(side="left")
            for title, variable, low, high in (("演示凹坑半径 / cm", self.brush_radius, 1, 7),
                                                ("演示凹陷深度 / cm", self.demo_depth, .1, 3)):
                self.label(panel, title, 9, MUTED).pack(anchor="w", padx=18, pady=(10, 0))
                tk.Scale(panel, from_=low, to=high, resolution=.1, orient="horizontal", variable=variable,
                         bg=PANEL, fg=TEXT, troughcolor=BG, highlightthickness=0, font=self.font(9)).pack(fill="x", padx=18)
            self.button(panel, "应用演示参数", self.configure_request, compact=True).pack(fill="x", padx=18, pady=8)
        switches = tk.Frame(panel, bg=PANEL)
        switches.pack(fill="x", padx=14, pady=9)
        for title, var in [("显示空间网格", self.show_grid), ("显示手部骨架", self.show_hand)]:
            tk.Checkbutton(switches, text=title, variable=var, command=self.configure_request,
                           bg=PANEL, fg=MUTED, activebackground=PANEL, activeforeground=TEXT,
                           selectcolor=BG, font=self.font(9), bd=0, highlightthickness=0).pack(anchor="w")
        self.debug_button = self.button(panel, "查看详细数据  +", self.toggle_debug, compact=True)
        self.debug_button.pack(fill="x", padx=16, pady=(0, 8))
        self.debug_label = self.label(panel, "", 9, MUTED, justify="left", anchor="w")
        self.note = self.label(panel, "手靠近物体 → 捏合抓取 → 张开释放", 9, MUTED,
                               wraplength=260, justify="left")
        self.note.pack(side="bottom", anchor="w", padx=18, pady=14)

    def _rule(self, pady=0):
        tk.Frame(self.side_content, height=1, bg=LINE).pack(fill="x", padx=18, pady=pady)

    def _stat(self, title, value):
        row = tk.Frame(self.side_content, bg=PANEL)
        row.pack(fill="x", padx=18, pady=5)
        self.label(row, title, 10, MUTED).pack(side="left")
        label = self.label(row, value, 10)
        label.pack(side="right")
        return label

    def _sidebar_wheel(self, event):
        if str(event.widget).startswith(str(self.side)):
            self.side_scroll.yview_scroll(-1 if event.delta > 0 else 1, "units")
            return "break"

    def toggle_debug(self):
        if self.debug_label.winfo_manager():
            self.debug_label.pack_forget()
            self.debug_button.configure(text="查看详细数据  +")
        else:
            self.debug_label.pack(fill="x", padx=self.px(18), pady=(0, self.px(8)))
            self.debug_button.configure(text="收起详细数据  −")

    def toggle_panel(self):
        self.panel_visible = not self.panel_visible
        if self.panel_visible:
            self.side.pack(side="right", fill="y", padx=(self.px(16), 0), before=self.center)
        else:
            self.side.pack_forget()
        self.panel_button.configure(text="收起面板  ›" if self.panel_visible else "‹  显示面板")

    def _space(self, event):
        if event.widget is self or event.widget is self.scene:
            self.toggle_run()
            return "break"

    def _tab(self, event):
        if event.widget is self or event.widget is self.scene:
            self.toggle_panel()
            return "break"

    def toggle_run(self):
        self.running = not self.running
        self._action("start" if self.running else "pause")
        self.run_button.configure(text=("暂停演示" if self.demo else "暂停") if self.running else "开始")
        if not self.demo:
            self.notify("已发送开始请求，等待数据" if self.running else "显示已暂停，已发送暂停请求")

    def reset_scene(self):
        self._action("reset", reset_object=True)
        self.object_pos = [0.0, .05, .42]
        self.angle = .25
        self._elapsed = 0.0
        self._drag = None
        self.notify("演示场景已复位" if self.demo else "已发送场景复位请求")

    def configure_request(self):
        if self.clay_mode:
            self._action("configure", brush_radius_m=self.brush_radius.get() / 100,
                         demo_depth_m=self.demo_depth.get() / 100,
                         grid=self.show_grid.get(), skeleton=self.show_hand.get())
            return
        self._action("configure", grab_radius_m=self.radius.get() / 100,
                     grid=self.show_grid.get(), skeleton=self.show_hand.get())
        self.draw_scene()

    def notify(self, text):
        self._toast = text
        self._toast_until = time.perf_counter() + 3

    def open_settings(self):
        if self.clay_mode:
            self.notify("在右侧选择输入源、应用演示参数；真实手输入当前用于骨架显示，尚未接入雕刻")
            return
        if self._settings is not None and self._settings.winfo_exists():
            self._settings.lift()
            return
        win = self._settings = tk.Toplevel(self)
        win.title("工作台设置")
        win.configure(bg=PANEL)
        win.geometry(f"{self.px(440)}x{self.px(420)}")
        win.resizable(False, False)
        win.transient(self)
        self.label(win, "工作台设置", 17).pack(anchor="w", padx=24, pady=(24, 14))
        self.label(win, "输入方式", 10, MUTED).pack(anchor="w", padx=24)
        choice = tk.StringVar(value="demo" if self.demo else "external")
        for title, value in [("内置演示（模拟手部与物体）", "demo"), ("外部状态快照（由主程序提交）", "external")]:
            tk.Radiobutton(win, text=title, variable=choice, value=value, bg=PANEL, fg=TEXT,
                           activebackground=PANEL, activeforeground=ACCENT,
                           selectcolor=BG, font=self.font(10)).pack(anchor="w", padx=22, pady=4)
        self.label(win, "抓取半径 / cm", 10, MUTED).pack(anchor="w", padx=24, pady=(16, 0))
        tk.Scale(win, from_=3, to=15, resolution=1, orient="horizontal", variable=self.radius,
                 bg=PANEL, fg=TEXT, troughcolor=BG, highlightthickness=0,
                 activebackground=ACCENT, font=self.font(9)).pack(fill="x", padx=24)
        tk.Checkbutton(win, text="自动循环演示抓取过程", variable=self.auto_demo,
                       bg=PANEL, fg=TEXT, selectcolor=BG, activebackground=PANEL,
                       activeforeground=TEXT, font=self.font(10)).pack(anchor="w", padx=22, pady=8)

        def apply():
            demo = choice.get() == "demo"
            if demo != self.demo:
                self.demo = demo
                self.snapshot = None
                self._last_frame = 0.0
                self._camera_frame = None
                self._camera_image = None
                self._camera_error = False
                self._drag = None
                self.running = demo
                self._elapsed = 0
                with self._lock:
                    self._pending = None
                self._action("source", source="demo" if demo else "external")
                self.run_button.configure(text="暂停演示" if demo else "开始")
            self.configure_request()
            win.destroy()
            self.notify("设置已应用" if self.demo else "已切换外部输入，等待主程序提交数据")

        self.button(win, "应用设置", apply, True).pack(anchor="e", padx=24, pady=12)
        self._scale_layout(win)

    def _demo_snapshot(self):
        phase = self._elapsed % 12
        automatic = self.auto_demo.get() and self._drag is None
        grabbed = self._drag is not None or (automatic and 4 <= phase < 8)
        near = not grabbed and (self._hover or (automatic and 2 <= phase < 4))
        if self._drag is not None:
            pos = self.object_pos[:]
        elif automatic:
            if 4 <= phase < 8:
                t = (phase - 4) / 4
                pos = [.12 * math.sin(t * math.pi), .05 - .11 * math.sin(t * math.pi), .42]
            elif 8 <= phase < 9:
                t = phase - 8
                pos = [0, .05 + .105 * t * t, .42]
            elif phase >= 9:
                pos = [0, .155, .42]
            else:
                pos = [0, .05, .42]
            self.object_pos = pos[:]
        else:
            pos = self.object_pos[:]
        hx = pos[0] + (.017 if grabbed else -.10 + .06 * math.sin(self._elapsed * .6))
        hy = pos[1] + (.029 if grabbed else .04)
        hand = demo_hand(hx, hy, pos[2] + .03, grabbed)
        pinch = tuple((hand[4][i] + hand[8][i]) / 2 for i in range(3))
        distance = math.sqrt(sum((pinch[i] - pos[i]) ** 2 for i in range(3)))
        return {"source": "demo", "fps": 0, "depth_enabled": False,
                "mode": "GRAB" if grabbed else "IDLE", "pinch_dist_m": distance,
                "grab_radius_m": self.radius.get() / 100, "primary_slot": 1,
                "hands": [{"slot": 1, "handedness": "right", "points_3d_m": hand,
                           "pinching": grabbed, "open_palm": not grabbed, "fist": False,
                           "pinch_ratio": .25 if grabbed else .95}],
                "object": {"center_m": pos, "grabbed": grabbed, "near": near,
                           "gravity_active": automatic and phase >= 8}}

    def _tick(self):
        if self._closed:
            return
        now = time.perf_counter()
        dt = min(now - self._clock, .1)
        self._clock = now
        with self._lock:
            pending = self._pending
            self._pending = None
        if pending and not self.demo:
            snapshot, camera, stamp = pending
            self._last_frame = stamp
            if self.running or self.clay_mode:
                self.snapshot = snapshot
                if self.clay_mode:
                    self.running = bool(field(snapshot, "running", False))
                    self.run_button.configure(text="暂停" if self.running else "开始")
                self._camera_frame = camera
                self._camera_error = False
                self._update_camera_image()
        if self.demo:
            if self.running:
                self._elapsed += dt
                if self._drag is None:
                    self.angle += dt * .12
            self.snapshot = self._demo_snapshot()
        self._refresh_labels()
        self.draw_scene()
        self.draw_preview()
        self._timer = self.after(33, self._tick)

    def _stale(self):
        return not self.demo and self.running and self._last_frame > 0 and time.perf_counter() - self._last_frame > 1.5

    def _refresh_labels(self):
        if self.clay_mode:
            self._refresh_clay_labels()
            return
        snap = self.snapshot
        hands = field(snap, "hands", ())
        obj = field(snap, "object", {})
        stale = self._stale()
        grabbed = field(obj, "grabbed", False)
        near = field(obj, "near", False)
        self.source_badge.configure(text="●  演示模式" if self.demo else "●  外部输入", fg=YELLOW if self.demo else ACCENT)
        self.view_label.configure(text="空间预览  /  " + ("演示" if self.demo else "外部输入"))
        live = "模拟数据" if self.demo else ("数据中断" if stale else ("等待输入" if snap is None else "数据已接入"))
        self.live_label.configure(text="●  " + live, fg=YELLOW if stale or snap is None else ACCENT)
        labels = {"left": "左手", "right": "右手"}
        names = [labels.get(field(h, "handedness"), "未知手") for h in hands]
        self.hand_label.configure(text="、".join(names) if names else "未检测到")
        gestures = ["捏合" if field(h, "pinching", False) else "握拳" if field(h, "fist", False)
                    else "张掌" if field(h, "open_palm", False) else "自然姿态" for h in hands]
        self.gesture_label.configure(text=" / ".join(gestures) if gestures else "—")
        self.object_label.configure(text="已抓取" if grabbed else "可抓取" if near else "等待交互",
                                    fg=ACCENT if grabbed else YELLOW if near else TEXT)
        dist = field(snap, "pinch_dist_m")
        self.distance_label.configure(text=f"{dist * 100:.1f} cm" if dist is not None else "—")
        if not self.running:
            hint, detail = "已暂停", "点击开始或按空格，继续当前体验。"
        elif stale:
            hint, detail = "输入数据已中断", "画面保留最后一帧，请检查输入设备和算法线程。"
        elif snap is None:
            hint, detail = "等待输入数据", "连接主程序后，这里会显示真实的手部与场景状态。"
        elif not hands:
            hint, detail = "把手放进摄像头画面", "让手掌保持完整可见，避免遮挡。"
        elif grabbed:
            hint, detail = "已抓住，试着移动它", "真实交互中，张开拇指和食指即可释放物体。"
        elif near:
            hint, detail = "进入范围，可以捏合了", "拇指与食指靠拢，抓住高亮的物体。"
        else:
            hint, detail = "将手靠近方块", "移动到黄色提示出现，再捏合抓取。"
        self.hint_label.configure(text=hint)
        self.hint_detail.configure(text=detail)
        self.preview_caption.configure(text="示意画面 · 未使用摄像头" if self.demo else
                                       "图像显示需安装 Pillow" if self._camera_error else
                                       "摄像头预览 · 外部输入" if self._camera_frame is not None else "等待摄像头画面")
        self.note.configure(text="鼠标可拖动方块，滚轮可旋转。\n手部识别与掉落为模拟演示。" if self.demo
                            else "手靠近物体 → 捏合抓取 → 张开释放")
        source = field(snap, "source", "—")
        depth = "真实深度" if field(snap, "depth_enabled", False) else "无真实深度"
        fps = field(snap, "fps", 0.0)
        status = "演示数据 · 未连接设备" if self.demo else f"{source}  ·  {depth}  ·  {fps:.1f} FPS"
        if self._toast_until > time.perf_counter():
            status = self._toast
        elif stale:
            status += "  ·  输入中断，当前为最后一帧"
        elif not self.running:
            status += "  ·  已暂停"
        self.status.configure(text=status)
        pos = field(obj, "center_m", (0, 0, 0))
        radius = field(snap, "grab_radius_m", self.radius.get() / 100)
        self.debug_label.configure(text=f"物体 / m    {pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}\n"
                                       f"抓取半径    {radius * 100:.0f} cm\n"
                                       f"重力           {'已启用' if field(obj, 'gravity_active', False) else '未启用'}")

    def _refresh_clay_labels(self):
        snap = self.snapshot
        error = field(snap, "error", "")
        source = field(snap, "source", self.input_source.get())
        hands = field(snap, "hands", ()) if not self._stale() else ()
        self.source_badge.configure(text="●  无相机演示" if source == "demo" else f"●  {source}",
                                    fg=YELLOW if source == "demo" else ACCENT)
        self.view_label.configure(text="Open3D 实际场景 · 同一泥坯")
        self.live_label.configure(text="●  输入异常" if error else "●  场景已连接" if field(snap, "scene_rgb") is not None else "○  正在启动",
                                  fg=YELLOW if error else ACCENT)
        self.hand_label.configure(text=f"{len(hands)} 只手" if hands else "无输入" if source == "demo" else "未检测到")
        gestures = ["捏合" if field(h, "pinching") else "张掌" if field(h, "open_palm") else "自然姿态" for h in hands]
        self.gesture_label.configure(text=" / ".join(gestures) or "—")
        changed = field(snap, "changed_vertices", 0)
        self.object_label.configure(text="已变形" if changed else "初始圆柱", fg=ACCENT)
        self.distance_label.configure(text="未接入拾取")
        if error:
            hint, detail = "输入源需要检查", error
        elif self._stale():
            hint, detail = "场景快照已中断", "当前保留最后画面，请检查独立三维窗口。"
        elif not self.running:
            hint, detail = "已暂停", "可继续观察和复位；开始后可放置演示凹坑。"
        else:
            hint, detail = "泥坯与桌面已就绪", field(snap, "message", "正在连接三维场景…")
        self.hint_label.configure(text=hint)
        self.hint_detail.configure(text=detail)
        self.dent_button.configure(state="normal" if self.running and not self._stale() else "disabled")
        self.preview_caption.configure(text="无相机演示 · 未使用真实手" if source == "demo" else
                                       "摄像头预览" if self._camera_frame is not None else "输入源未提供图像")
        self.note.configure(text="真实手当前只显示骨架。\n凹坑按钮用于场景联调，非手势雕刻。")
        self.debug_label.configure(text=f"顶点数        {field(snap, 'vertex_count', 0)}\n"
                                       f"网格版本     {field(snap, 'mesh_version', 0)}\n"
                                       f"变化顶点     {changed}\n"
                                       f"实际半径     {field(snap, 'brush_radius_m', 0) * 100:.1f} cm\n"
                                       f"实际深度     {field(snap, 'demo_depth_m', 0) * 100:.1f} cm")
        status = f"{source} · {'真实深度' if field(snap, 'depth_enabled', False) else '无真实深度'} · 网格 v{field(snap, 'mesh_version', 0)}"
        if source != "demo":
            status += f" · 输入 {field(snap, 'fps', 0):.1f} FPS"
        if not self.running:
            status += " · 已暂停"
        if error:
            status += " · " + error
        if self._toast_until > time.perf_counter():
            status = self._toast
        self.status.configure(text=status)

    def _draw_clay_scene(self):
        canvas = self.scene
        canvas.delete("all")
        width, height = canvas.winfo_width(), canvas.winfo_height()
        frame = field(self.snapshot, "scene_rgb")
        if frame is None or width < 20 or height < 20:
            canvas.create_text(width / 2, height / 2, text=field(self.snapshot, "error") or "正在启动泥坯场景…",
                               fill=TEXT, font=self.font(13), width=max(20, width - 40))
            return
        try:
            from PIL import Image, ImageTk
            key = (id(frame), width, height)
            if key != self._scene_image_key:
                image = Image.fromarray(frame)
                image.thumbnail((width, height), Image.Resampling.LANCZOS)
                self._scene_image = ImageTk.PhotoImage(image)
                self._scene_image_key = key
                self._scene_frame = frame
            canvas.create_image(width / 2, height / 2, image=self._scene_image)
            if not self.running:
                canvas.create_text(width / 2, self.px(24), text="已暂停 · 可观察 / 复位", fill=YELLOW, font=self.font(10))
        except (ImportError, ValueError, TypeError) as exc:
            canvas.create_text(width / 2, height / 2, text=f"场景预览不可用：{exc}\n请在独立 Open3D 窗口查看",
                               fill=YELLOW, font=self.font(11), width=max(20, width - 40))

    def project(self, point):
        w, h = self.scene.winfo_width(), self.scene.winfo_height()
        scale = min(w / 1.15, h / .78)
        x, y, z = point
        return w * .50 + (x + (z - .5) * .38) * scale, h * .53 + (y + (z - .5) * .52) * scale

    def draw_scene(self):
        if self.clay_mode:
            self._draw_clay_scene()
            return
        c = self.scene
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if w < 20 or h < 20:
            return
        c.create_text(self.px(22), self.px(24), anchor="w", text="WORKSPACE  /  01", fill="#7a91a8", font=self.font(9))
        c.create_text(w - self.px(22), self.px(24), anchor="e", text="模拟演示" if self.demo else "外部场景数据",
                      fill=YELLOW if self.demo else ACCENT, font=self.font(9))
        table = [self.project(p) for p in [(-.45, .18, .25), (.45, .18, .25), (.45, .18, .85), (-.45, .18, .85)]]
        c.create_polygon(*[v for p in table for v in p], fill="#1b2733", outline="#344859", width=1)
        if self.show_grid.get():
            for i in range(13):
                x = -.45 + i * .075
                c.create_line(*self.project((x, .18, .25)), *self.project((x, .18, .85)), fill="#2c3b4a")
            for i in range(9):
                z = .25 + i * .075
                c.create_line(*self.project((-.45, .18, z)), *self.project((.45, .18, z)), fill="#2c3b4a")
        snap = self.snapshot
        self._cube_hitbox = None
        if snap is not None:
            obj = field(snap, "object", {})
            pos = field(obj, "center_m", (0, .05, .42))
            grabbed, near = field(obj, "grabbed", False), field(obj, "near", False)
            color = ACCENT if grabbed else YELLOW if near else "#79b5df"
            sx, sy = self.project((pos[0], .179, pos[2]))
            c.create_oval(sx - self.px(42), sy - self.px(8), sx + self.px(42), sy + self.px(8), fill="#121d27", outline="")
            cx, cy = self.project(pos)
            c.create_line(cx, cy, sx, sy, fill="#415567", dash=(self.px(3), self.px(5)))
            if self.show_hand.get():
                for hand in field(snap, "hands", ()):
                    self._draw_hand(c, field(hand, "points_3d_m", ()), self.project,
                                    ACCENT if field(hand, "pinching", False) else "#809caf")
            self._draw_cube(pos, color, field(obj, "pose_cam"))
            if grabbed or near:
                c.create_text(cx, cy - self.px(57), text="已抓取" if grabbed else "可抓取", fill=color, font=self.font(10, "bold"))
        else:
            c.create_text(w / 2, h * .38, text="交互空间已就绪", fill=TEXT, font=self.font(20))
            c.create_text(w / 2, h * .38 + self.px(38), text="等待主程序输入，或在设置中体验演示", fill=MUTED, font=self.font(10))
        c.create_line(self.px(28), h - self.px(38), self.px(64), h - self.px(38), fill="#c98282", width=self.px(2))
        c.create_line(self.px(28), h - self.px(38), self.px(28), h - self.px(72), fill=ACCENT, width=self.px(2))
        c.create_text(self.px(74), h - self.px(38), text="X", fill=MUTED, font=self.font(9))
        c.create_text(self.px(28), h - self.px(83), text="Y", fill=MUTED, font=self.font(9))
        text = "鼠标拖动 · 滚轮旋转  |  手部与物理为模拟数据" if self.demo else "相机坐标 · 米制空间投影"
        c.create_text(w / 2, h - self.px(24), text=text, fill="#71869b", font=self.font(9))
        if self._stale() or not self.running:
            text = "输入中断 · 保留最后一帧" if self._stale() else "已暂停"
            c.create_rectangle(w / 2 - self.px(120), self.px(48), w / 2 + self.px(120), self.px(80), fill=PANEL, outline=LINE)
            c.create_text(w / 2, self.px(64), text=text, fill=YELLOW, font=self.font(10))

    def _draw_cube(self, pos, color, pose):
        verts = []
        for x, y, z in [(-1,-1,-1), (1,-1,-1), (1,1,-1), (-1,1,-1),
                        (-1,-1,1), (1,-1,1), (1,1,1), (-1,1,1)]:
            local = [x * .025, y * .025, z * .025]
            if pose is not None:
                p = [sum(float(pose[i][j]) * local[j] for j in range(3)) + pos[i] for i in range(3)]
            else:
                a = self.angle
                p = [pos[0] + local[0] * math.cos(a) + local[2] * math.sin(a),
                     pos[1] + local[1], pos[2] - local[0] * math.sin(a) + local[2] * math.cos(a)]
            verts.append(p)
        faces = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
                 (3, 2, 6, 7), (0, 3, 7, 4), (1, 5, 6, 2)]
        # 投影深度 n=(.38,.52,-1)，由远及近绘制，避免依赖 GPU。
        faces.sort(key=lambda face: sum(.38 * verts[i][0] + .52 * verts[i][1] - verts[i][2] for i in face))
        pixels = [self.project(p) for p in verts]
        for face in faces:
            factor = .72 if face in [(0, 3, 7, 4), (1, 5, 6, 2)] else 1.06 if face == (0, 4, 5, 1) else .88
            self.scene.create_polygon(*[v for i in face for v in pixels[i]], fill=shade(color, factor),
                                      outline=shade(color, 1.15), width=self.px(1.5))
        self._cube_hitbox = (min(p[0] for p in pixels) - self.px(12), min(p[1] for p in pixels) - self.px(12),
                             max(p[0] for p in pixels) + self.px(12), max(p[1] for p in pixels) + self.px(12))

    def _draw_hand(self, canvas, points, project, color):
        if len(points) != 21:
            return
        pixels = [project(p) for p in points]
        for a, b in CONNECTIONS:
            canvas.create_line(*pixels[a], *pixels[b], fill=color, width=self.px(2), capstyle="round")
        for i, (x, y) in enumerate(pixels):
            r = self.px(3.5 if i in (0, 4, 8) else 2.4)
            canvas.create_oval(x-r, y-r, x+r, y+r, fill=ACCENT if i in (4, 8) else color, outline="")

    def _update_camera_image(self):
        self._camera_image = None
        if self._camera_frame is None:
            return
        try:
            from PIL import Image, ImageTk
            image = Image.fromarray(self._camera_frame[:, :, ::-1])
            image.thumbnail((max(1, self.preview.winfo_width()), self.px(138)))
            self._camera_image = ImageTk.PhotoImage(image)
        except (ImportError, ValueError, TypeError):
            self._camera_error = True

    def draw_preview(self):
        c = self.preview
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if self._camera_image is not None and not self.demo:
            c.create_image(w / 2, h / 2, image=self._camera_image)
            return
        for x in range(0, w, self.px(28)):
            c.create_line(x, 0, x, h, fill="#18242e")
        for y in range(0, h, self.px(28)):
            c.create_line(0, y, w, y, fill="#18242e")
        edge = self.px(12)
        for x, y, dx, dy in [(edge, edge, 1, 1), (w-edge, edge, -1, 1), (edge, h-edge, 1, -1), (w-edge, h-edge, -1, -1)]:
            c.create_line(x, y + dy*edge, x, y, x + dx*edge, y, fill="#50677a", width=self.px(2))
        if self.demo:
            obj = field(self.snapshot, "object", {})
            points = demo_hand(pinching=field(obj, "grabbed", False))
            self._draw_hand(c, points, lambda p: (w*.48 + p[0]*self.px(510), h*.59 + p[1]*self.px(510)), "#769ca8")
            c.create_text(w-self.px(15), self.px(18), anchor="e", text="DEMO", fill=YELLOW, font=self.font(8))
        else:
            c.create_text(w/2, h/2, text="等待图像输入" if not self._camera_error else "无法显示当前图像",
                          fill=MUTED, font=self.font(10))

    def _hit(self, event):
        b = self._cube_hitbox
        return b is not None and b[0] <= event.x <= b[2] and b[1] <= event.y <= b[3]

    def _motion(self, event):
        self._hover = self._hit(event)

    def _press(self, event):
        self.scene.focus_set()
        if self.clay_mode:
            self._drag = (event.x, event.y)
            return
        if self.demo and self.running and self._hit(event):
            self._drag = (event.x, event.y, self.object_pos[:])

    def _move(self, event):
        if self.clay_mode:
            if self._drag is not None:
                x, y = self._drag
                self._action("orbit", dx=event.x - x, dy=event.y - y)
                self._drag = (event.x, event.y)
            return
        if self._drag is None or not self.running:
            return
        x, y, pos = self._drag
        scale = min(self.scene.winfo_width() / 1.15, self.scene.winfo_height() / .78)
        self.object_pos = [max(-.40, min(.40, pos[0] + (event.x - x) / scale)),
                           max(-.23, min(.155, pos[1] + (event.y - y) / scale)), pos[2]]

    def _release(self, event):
        if self.clay_mode:
            self._drag = None
            return
        if self._drag is not None:
            self._drag = None
            self.auto_demo.set(False)
            self.notify("已释放演示物体；可在设置中恢复自动演示")

    def _wheel(self, event):
        if self.clay_mode:
            self._action("zoom", factor=.9 if event.delta > 0 else 1.1)
            return "break"
        if self.demo and self.running and self._hit(event):
            self.angle += .12 if event.delta > 0 else -.12

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._action("quit", quit=True)
        if self._timer is not None:
            self.after_cancel(self._timer)
        self.destroy()


def main():
    parser = argparse.ArgumentParser(description="默认启动 Open3D 泥坯工作台；可附加 main.py 的输入源参数")
    parser.add_argument("--external", action="store_true", help="以等待外部快照的模式启动")
    parser.add_argument("--legacy-demo", action="store_true", help="查看原立方体 Canvas 原型")
    args, remaining = parser.parse_known_args()
    if args.external or args.legacy_demo:
        if remaining:
            parser.error("原型模式不接受其他参数")
        HandWorkbench(demo=not args.external).mainloop()
    else:
        from main import main as run
        return run(remaining)


if __name__ == "__main__":
    main()
