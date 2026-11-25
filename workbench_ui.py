"""手势交互工作台：独立 UI 原型，不导入或修改现有业务模块。

运行：python workbench_ui.py
仅需要 Python 自带 tkinter；显示真实 camera_bgr 时额外使用 Pillow。

接入方式（Tk 窗口及 mainloop 必须在主线程运行）：
    app = HandWorkbench(demo=False)
    # 算法线程每帧调用，接受 UI_HANDOFF.md 中的快照对象或同名字段的 dict：
    app.publish_snapshot(snapshot)
    # 算法线程定期读取；UI 本身不改变算法对象：
    for action in app.poll_actions():
        ...  # action['type']: start / pause / reset / configure / tool_change / brush_config / source / quit
    app.mainloop()

publish_snapshot 保留最新一帧，不累积排队；camera_bgr 会在提交时复制。
configure 的 grid/skeleton 是显示开关请求，后端自行接收并应用。
切换到外部输入后，需由集成代码提交数据；本文件不启动摄像头或 MediaPipe。
内置 3D 为 Canvas 投影预览，可由队友替换为自己的渲染视图。
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

    def __init__(self, demo=True):
        enable_high_dpi()
        super().__init__()
        self.ui_scale = max(1.0, round(self.winfo_fpixels("1i") / 96.0, 2))
        self.tk.call("tk", "scaling", self.ui_scale * 96.0 / 72.0)
        available = set(tkfont.families(self))
        self.font_family = next((name for name in (FONT, "Microsoft YaHei", "Segoe UI")
                                 if name in available), "TkDefaultFont")
        self.title("隔空捏泥人 · 三维雕刻系统")
        self.configure(bg=BG)
        width = min(self.px(1440), self.winfo_screenwidth() - self.px(80))
        height = min(self.px(900), self.winfo_screenheight() - self.px(100))
        self.geometry(f"{width}x{height}+{self.px(30)}+{self.px(30)}")
        self.minsize(min(self.px(1000), width), min(self.px(680), height))
        self.demo = demo
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
        self._camera_image = None
        self._camera_frame = None
        self._camera_error = False
        self._settings = None
        self._toast = ""
        self._toast_until = 0.0
        self.panel_visible = True
        self.tool = tk.StringVar(value="carve")
        self.tool_buttons = {}
        self.brush_radius = tk.DoubleVar(value=5.0)
        self.brush_strength = tk.DoubleVar(value=50.0)
        self.smooth_level = tk.DoubleVar(value=30.0)
        self.show_grid = tk.BooleanVar(value=True)
        self.show_hand = tk.BooleanVar(value=True)
        self.auto_demo = tk.BooleanVar(value=True)
        self.demo_error = tk.StringVar(value="none")
        self.clay_radius = 0.13
        self.clay_center = (0.0, 0.5)
        self.clay_y_top = -0.18
        self.clay_y_bot = 0.18
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
        self.label(branding, "隔空捏泥人", 16).pack(anchor="w")
        self.label(branding, "三维雕刻系统", 9, MUTED).pack(anchor="w")
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
        self.label(titles, "雕刻工作台", 21, bg=BG).pack(anchor="w")
        self.label(titles, "用手势在虚拟泥坯上刻、抹、旋转。", 10, MUTED, BG).pack(anchor="w", pady=(4, 0))
        self.panel_button = self.button(heading, "收起面板  ›", self.toggle_panel, compact=True)
        self.panel_button.pack(side="right", anchor="s")

        frame = tk.Frame(self.center, bg=LINE, padx=1, pady=1)
        frame.pack(fill="both", expand=True)
        tools = tk.Frame(frame, bg=PANEL, height=44)
        tools.pack(fill="x")
        tools.pack_propagate(False)
        group = tk.Frame(tools, bg=PANEL)
        group.pack(side="left", padx=10, pady=7)
        for text, value in [("刻刀", "carve"), ("抹刀", "smooth"), ("旋转", "orbit")]:
            self._tool_button(group, text, value)
        self.button(group, "重置", self.reset_scene, compact=True).pack(side="left", padx=(8, 0))
        self.view_label = self.label(tools, "空间预览  /  演示" if self.demo else "空间预览  /  外部输入", 9, MUTED)
        self.view_label.pack(side="right", padx=16)
        self._sync_tool_buttons()
        self.scene = tk.Canvas(frame, width=394, height=276, bg="#131b24", highlightthickness=0, cursor="hand2")
        self.scene.pack(fill="both", expand=True)
        self.scene.bind("<Configure>", lambda event: self.draw_scene())
        footer = tk.Frame(frame, bg=PANEL)
        footer.pack(fill="x")
        self.label(footer, "  捏合雕刻 · 张掌抹平 · 滚轮旋转视角", 9, MUTED).pack(side="left", padx=10, pady=10)
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
        self.left_hand_label = self._stat("左手", "未检测到")
        self.right_hand_label = self._stat("右手", "未检测到")
        self.gesture_label = self._stat("当前手势", "—")
        self.pick_label = self._stat("拾取状态", "未命中")
        self.tool_label = self._stat("当前工具", "刻刀")
        self._rule(pady=12)
        self._build_brush_panel()
        self.label(panel, "下一步", 10, MUTED).pack(anchor="w", padx=18)
        self.hint_label = self.label(panel, "", 11, TEXT, justify="left", wraplength=240)
        self.hint_label.pack(anchor="w", padx=18, pady=(8, 6))
        self.hint_detail = self.label(panel, "", 9, MUTED, justify="left", wraplength=240)
        self.hint_detail.pack(anchor="w", padx=18, pady=(0, 12))
        self._rule()
        switches = tk.Frame(panel, bg=PANEL)
        switches.pack(fill="x", padx=14, pady=9)
        for title, var in [("显示空间网格", self.show_grid), ("显示手部骨架", self.show_hand)]:
            tk.Checkbutton(switches, text=title, variable=var, command=self.configure_request,
                           bg=PANEL, fg=MUTED, activebackground=PANEL, activeforeground=TEXT,
                           selectcolor=BG, font=self.font(9), bd=0, highlightthickness=0).pack(anchor="w")
        self.debug_button = self.button(panel, "查看详细数据  +", self.toggle_debug, compact=True)
        self.debug_button.pack(fill="x", padx=16, pady=(0, 8))
        self.debug_label = self.label(panel, "", 9, MUTED, justify="left", anchor="w")
        self.note = self.label(panel, "捏合使用刻刀挖坑，张掌使用抹刀平滑，旋转工具调整视角。", 9, MUTED,
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

    def _build_brush_panel(self):
        self._rule(pady=12)
        self.label(self.side_content, "笔刷参数", 11).pack(anchor="w", padx=18, pady=(4, 10))
        self._slider_row("笔刷半径 / cm", self.brush_radius, 1, 15, 1, "{:.0f}")
        self._slider_row("雕刻强度", self.brush_strength, 0, 100, 1, "{:.0f}%")
        self._slider_row("平滑程度", self.smooth_level, 0, 100, 1, "{:.0f}%")

    def _slider_row(self, title, variable, low, high, step, fmt):
        box = tk.Frame(self.side_content, bg=PANEL)
        box.pack(fill="x", padx=18, pady=(0, 12))
        top = tk.Frame(box, bg=PANEL)
        top.pack(fill="x")
        self.label(top, title, 10, MUTED).pack(side="left")
        value_label = self.label(top, "", 10, ACCENT)
        value_label.pack(side="right")

        def show(value):
            value_label.configure(text=fmt.format(float(value)))

        def on_change(value):
            show(value)
            self.send_brush_config()

        scale = tk.Scale(box, from_=low, to=high, resolution=step, orient="horizontal",
                         variable=variable, command=on_change, showvalue=False,
                         bg=PANEL, fg=TEXT, troughcolor=BG, highlightthickness=0,
                         activebackground=ACCENT, font=self.font(9))
        scale.pack(fill="x")
        show(variable.get())

    def send_brush_config(self):
        self._action("brush_config",
                     radius_cm=self.brush_radius.get(),
                     strength=self.brush_strength.get() / 100,
                     smooth=self.smooth_level.get() / 100)

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
        self._elapsed = 0.0
        self.notify("演示场景已复位" if self.demo else "已发送场景复位请求")

    def configure_request(self):
        self._action("configure", grid=self.show_grid.get(),
                     skeleton=self.show_hand.get())
        self.draw_scene()

    def _tool_button(self, parent, text, value):
        btn = tk.Button(parent, text=text, command=lambda: self.select_tool(value),
                        font=self.font(10), relief="flat", bd=0, cursor="hand2",
                        padx=14, pady=6, highlightthickness=0)
        btn.pack(side="left", padx=4)
        self.tool_buttons[value] = btn

    def select_tool(self, value):
        self.tool.set(value)
        self._action("tool_change", tool=value)
        self._sync_tool_buttons()

    def _sync_tool_buttons(self):
        current = self.tool.get()
        for value, btn in self.tool_buttons.items():
            if value == current:
                btn.configure(bg=ACCENT, fg=BG, activebackground="#a2ead6", activeforeground=BG)
            else:
                btn.configure(bg=CARD, fg=TEXT, activebackground=LINE, activeforeground=TEXT)

    def notify(self, text):
        self._toast = text
        self._toast_until = time.perf_counter() + 3

    def open_settings(self):
        if self._settings is not None and self._settings.winfo_exists():
            self._settings.lift()
            return
        win = self._settings = tk.Toplevel(self)
        win.title("工作台设置")
        win.configure(bg=PANEL)
        win.geometry(f"{self.px(440)}x{self.px(560)}")
        win.resizable(False, False)
        win.transient(self)
        self.label(win, "工作台设置", 17).pack(anchor="w", padx=24, pady=(24, 14))
        self.label(win, "输入方式", 10, MUTED).pack(anchor="w", padx=24)
        choice = tk.StringVar(value="demo" if self.demo else "external")
        for title, value in [("内置演示（模拟手部与物体）", "demo"), ("外部状态快照（由主程序提交）", "external")]:
            tk.Radiobutton(win, text=title, variable=choice, value=value, bg=PANEL, fg=TEXT,
                           activebackground=PANEL, activeforeground=ACCENT,
                           selectcolor=BG, font=self.font(10)).pack(anchor="w", padx=22, pady=4)
        tk.Checkbutton(win, text="自动循环演示手势", variable=self.auto_demo,
                       bg=PANEL, fg=TEXT, selectcolor=BG, activebackground=PANEL,
                       activeforeground=TEXT, font=self.font(10)).pack(anchor="w", padx=22, pady=8)
        self.label(win, "模拟错误状态（仅演示模式）", 10, MUTED).pack(anchor="w", padx=24, pady=(16, 0))
        error_choice = tk.StringVar(value=self.demo_error.get())
        for title, value in [("无错误", "none"),
                             ("摄像头权限被拒绝", "permission_denied"),
                             ("识别中断", "interrupted")]:
            tk.Radiobutton(win, text=title, variable=error_choice, value=value,
                           command=lambda v=value: self.set_demo_error(v),
                           bg=PANEL, fg=TEXT, activebackground=PANEL, activeforeground=ACCENT,
                           selectcolor=BG, font=self.font(10)).pack(anchor="w", padx=22, pady=4)

        def apply():
            demo = choice.get() == "demo"
            if demo != self.demo:
                self.demo = demo
                self.snapshot = None
                self._last_frame = 0.0
                self._camera_frame = None
                self._camera_image = None
                self._camera_error = False
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
        pinching = self.auto_demo.get() and 4 <= phase < 8
        sway = .05 * math.sin(self._elapsed * .6)
        hand = demo_hand(sway, -.05, .66, pinching)
        point = (sway * .5, -.05, .5 + self.clay_radius)
        return {"source": "demo", "fps": 0.0, "depth_enabled": False,
                "hands": [{"slot": 1, "handedness": "right", "points_3d_m": hand,
                           "pinching": pinching, "open_palm": not pinching,
                           "fist": False,
                           "pinch_ratio": .25 if pinching else .95}],
                "pick": {"hit": True, "point_m": point, "normal_m": (0, 0, 1)}}

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
            if self.running:
                self.snapshot = snapshot
                self._camera_frame = camera
                self._camera_error = False
                self._update_camera_image()
        if self.demo:
            if self.running:
                self._elapsed += dt
            self.snapshot = self._demo_snapshot()
        self._refresh_labels()
        self.draw_scene()
        self.draw_preview()
        self._timer = self.after(33, self._tick)

    def _stale(self):
        return not self.demo and self.running and self._last_frame > 0 and time.perf_counter() - self._last_frame > 1.5

    def current_camera_error(self):
        if self.demo:
            value = self.demo_error.get()
            return None if value == "none" else value
        return field(self.snapshot, "camera_error", None)

    def set_demo_error(self, value):
        self.demo_error.set(value)
        self._refresh_labels()
        self.draw_preview()
        self.draw_scene()

    def _refresh_labels(self):
        snap = self.snapshot
        hands = field(snap, "hands", ())
        stale = self._stale()
        pick = field(snap, "pick", {})
        self.source_badge.configure(text="●  演示模式" if self.demo else "●  外部输入", fg=YELLOW if self.demo else ACCENT)
        self.view_label.configure(text="空间预览  /  " + ("演示" if self.demo else "外部输入"))
        error = self.current_camera_error()
        if error == "permission_denied":
            live, warn = "权限拒绝", True
        elif error == "interrupted":
            live, warn = "数据中断", True
        elif self.demo:
            live, warn = "模拟数据", False
        elif stale:
            live, warn = "数据中断", True
        elif snap is None:
            live, warn = "等待输入", True
        else:
            live, warn = "数据已接入", False
        self.live_label.configure(text="●  " + live, fg=YELLOW if warn else ACCENT)
        handed = [field(h, "handedness") for h in hands]
        left_found = "left" in handed
        right_found = "right" in handed
        self.left_hand_label.configure(text="检测到" if left_found else "未检测到",
                                       fg=ACCENT if left_found else MUTED)
        self.right_hand_label.configure(text="检测到" if right_found else "未检测到",
                                        fg=ACCENT if right_found else MUTED)
        if hands:
            active_hand = hands[0]
            if field(active_hand, "pinching", False):
                gesture = "捏合"
            elif field(active_hand, "open_palm", False):
                gesture = "张掌"
            else:
                gesture = "空闲"
        else:
            gesture = "—"
        self.gesture_label.configure(text=gesture)
        hit = field(pick, "hit", False)
        self.pick_label.configure(text="命中" if hit else "未命中",
                                  fg=ACCENT if hit else MUTED)
        tool_names = {"carve": "刻刀", "smooth": "抹刀", "orbit": "旋转"}
        self.tool_label.configure(text=tool_names.get(self.tool.get(), "—"))
        if not self.running:
            hint, detail = "已暂停", "点击开始或按空格，继续当前体验。"
        elif stale:
            hint, detail = "输入数据已中断", "画面保留最后一帧，请检查输入设备和算法线程。"
        elif snap is None:
            hint, detail = "等待输入数据", "连接主程序后，这里会显示真实的手部与场景状态。"
        elif not hands:
            hint, detail = "把手放进摄像头画面", "让手掌保持完整可见，避免遮挡。"
        elif hit:
            if field(hands[0], "pinching", False):
                hint, detail = "正在雕刻", "保持捏合并缓慢移动，形成连续凹槽。"
            elif field(hands[0], "open_palm", False):
                hint, detail = "抹刀平滑中", "保持张掌移动，消减局部不平整。"
            else:
                hint, detail = "已对准泥坯", "捏合手指开始雕刻，或张掌进行平滑。"
        else:
            hint, detail = "移动手部对准泥坯", "让刻刀光圈落在圆柱表面，再开始手势。"
        self.hint_label.configure(text=hint)
        self.hint_detail.configure(text=detail)
        self.preview_caption.configure(text="示意画面 · 未使用摄像头" if self.demo else
                                       "图像显示需安装 Pillow" if self._camera_error else
                                       "摄像头预览 · 外部输入" if self._camera_frame is not None else "等待摄像头画面")
        self.note.configure(text="工具按钮可切换，参数与雕刻效果为模拟演示。" if self.demo
                            else "捏合使用刻刀挖坑，张掌使用抹刀平滑，旋转工具调整视角。")
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
        point = field(pick, "point_m")
        if point is not None:
            point_text = f"{point[0]:+.3f}, {point[1]:+.3f}, {point[2]:+.3f}"
        else:
            point_text = "—"
        self.debug_label.configure(text=f"笔刷半径    {self.brush_radius.get():.0f} cm\n"
                                       f"雕刻强度    {self.brush_strength.get():.0f}%\n"
                                       f"平滑程度    {self.smooth_level.get():.0f}%\n"
                                       f"命中点 / m  {point_text}")

    def project(self, point):
        w, h = self.scene.winfo_width(), self.scene.winfo_height()
        scale = min(w / 1.15, h / .78)
        x, y, z = point
        return w * .50 + (x + (z - .5) * .38) * scale, h * .53 + (y + (z - .5) * .52) * scale

    def draw_scene(self):
        c = self.scene
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if w < 20 or h < 20:
            return
        c.create_text(self.px(22), self.px(24), anchor="w", text="WORKSPACE  /  01",
                      fill="#7a91a8", font=self.font(9))
        c.create_text(w - self.px(22), self.px(24), anchor="e",
                      text="模拟演示" if self.demo else "外部场景数据",
                      fill=YELLOW if self.demo else ACCENT, font=self.font(9))
        table = [self.project(p) for p in [(-.45, .18, .25), (.45, .18, .25),
                                           (.45, .18, .85), (-.45, .18, .85)]]
        c.create_polygon(*[v for p in table for v in p], fill="#1b2733",
                         outline="#344859", width=1)
        if self.show_grid.get():
            for i in range(13):
                x = -.45 + i * .075
                c.create_line(*self.project((x, .18, .25)),
                              *self.project((x, .18, .85)), fill="#2c3b4a")
            for i in range(9):
                z = .25 + i * .075
                c.create_line(*self.project((-.45, .18, z)),
                              *self.project((.45, .18, z)), fill="#2c3b4a")
        snap = self.snapshot
        if snap is not None:
            self._draw_clay()
            self._draw_brush_ring(field(snap, "pick", {}))
            if self.show_hand.get():
                for hand in field(snap, "hands", ()):
                    self._draw_hand(c, field(hand, "points_3d_m", ()), self.project,
                                    ACCENT if field(hand, "pinching", False) else "#809caf")
        else:
            c.create_text(w / 2, h * .38, text="雕刻场景已就绪", fill=TEXT, font=self.font(20))
            c.create_text(w / 2, h * .38 + self.px(38),
                          text="等待主程序输入，或在设置中体验演示",
                          fill=MUTED, font=self.font(10))
        c.create_line(self.px(28), h - self.px(38), self.px(64), h - self.px(38),
                      fill="#c98282", width=self.px(2))
        c.create_line(self.px(28), h - self.px(38), self.px(28), h - self.px(72),
                      fill=ACCENT, width=self.px(2))
        c.create_text(self.px(74), h - self.px(38), text="X", fill=MUTED, font=self.font(9))
        c.create_text(self.px(28), h - self.px(83), text="Y", fill=MUTED, font=self.font(9))
        text = "手势与光圈为模拟演示" if self.demo else "相机坐标 · 米制空间投影"
        c.create_text(w / 2, h - self.px(24), text=text, fill="#71869b", font=self.font(9))
        error = self.current_camera_error()
        if error == "permission_denied":
            banner = "摄像头权限被拒绝 · 请检查权限设置"
        elif error == "interrupted" or self._stale():
            banner = "输入中断 · 保留最后一帧"
        elif not self.running:
            banner = "已暂停"
        else:
            banner = None
        if banner:
            c.create_rectangle(w/2 - self.px(150), self.px(48),
                               w/2 + self.px(150), self.px(80),
                               fill=PANEL, outline=LINE)
            c.create_text(w/2, self.px(64), text=banner, fill=YELLOW, font=self.font(10))

    def _clay_ring(self, y, radius, n=36):
        cx, cz = self.clay_center
        ring = []
        for i in range(n):
            angle = 2 * math.pi * i / n
            ring.append(self.project((cx + radius * math.cos(angle), y,
                                      cz + radius * math.sin(angle))))
        return ring

    def _draw_clay(self):
        top_ring = self._clay_ring(self.clay_y_top, self.clay_radius)
        bot_ring = self._clay_ring(self.clay_y_bot, self.clay_radius)
        tl = min(top_ring, key=lambda p: p[0])
        tr = max(top_ring, key=lambda p: p[0])
        bl = min(bot_ring, key=lambda p: p[0])
        br = max(bot_ring, key=lambda p: p[0])
        clay = "#c98a5b"
        self.scene.create_polygon(*[v for p in bot_ring for v in p], fill="#8f5f3d", outline="")
        self.scene.create_polygon(tl[0], tl[1], tr[0], tr[1], br[0], br[1], bl[0], bl[1],
                                 fill=clay, outline="")
        count = len(bot_ring)
        for fraction in (0.33, 0.66):
            y = self.clay_y_top + fraction * (self.clay_y_bot - self.clay_y_top)
            ring = self._clay_ring(y, self.clay_radius)
            for i in range(count):
                if math.sin(2 * math.pi * i / count) > 0:
                    a, b = ring[i], ring[(i + 1) % count]
                    self.scene.create_line(a[0], a[1], b[0], b[1], fill="#a96e44")
        self.scene.create_polygon(*[v for p in top_ring for v in p],
                                 fill="#d9a074", outline="#b5794c")
        self.scene.create_line(tl[0], tl[1], bl[0], bl[1], fill="#8f5f3d")
        self.scene.create_line(tr[0], tr[1], br[0], br[1], fill="#8f5f3d")
        for i in range(count):
            if math.sin(2 * math.pi * i / count) > 0:
                a, b = bot_ring[i], bot_ring[(i + 1) % count]
                self.scene.create_line(a[0], a[1], b[0], b[1], fill="#8f5f3d")

    def _draw_brush_ring(self, pick):
        if not field(pick, "hit", False):
            return
        point = field(pick, "point_m")
        if point is None:
            return
        center_x, center_y = self.project(point)
        radius = self.px(14)
        self.scene.create_oval(center_x - radius, center_y - radius,
                               center_x + radius, center_y + radius,
                               outline=ACCENT, width=self.px(2))
        dot = self.px(2)
        self.scene.create_oval(center_x - dot, center_y - dot,
                               center_x + dot, center_y + dot,
                               fill=ACCENT, outline="")

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
        error = self.current_camera_error()
        if error == "permission_denied":
            c.create_text(w/2, h*.36, text="摄像头权限被拒绝", fill=YELLOW, font=self.font(11, "bold"))
            c.create_text(w/2, h*.58, text="请允许访问摄像头后重试", fill=MUTED, font=self.font(9))
            c.create_text(w/2, h*.76, text="错误代码 permission_denied", fill="#5d7183", font=self.font(8))
        elif error == "interrupted":
            c.create_text(w/2, h*.36, text="识别中断", fill=YELLOW, font=self.font(11, "bold"))
            c.create_text(w/2, h*.58, text="正在尝试恢复，请保持手部在画面内", fill=MUTED, font=self.font(9))
            c.create_text(w/2, h*.76, text="错误代码 interrupted", fill="#5d7183", font=self.font(8))
        elif self.demo:
            hands = field(self.snapshot, "hands", ())
            pinching = any(field(h, "pinching", False) for h in hands)
            points = demo_hand(pinching=pinching)
            self._draw_hand(c, points,
                            lambda p: (w*.48 + p[0]*self.px(510), h*.59 + p[1]*self.px(510)),
                            "#769ca8")
            c.create_text(w-self.px(15), self.px(18), anchor="e", text="DEMO",
                          fill=YELLOW, font=self.font(8))
        else:
            c.create_text(w/2, h/2, text="等待图像输入" if not self._camera_error else "无法显示当前图像",
                          fill=MUTED, font=self.font(10))

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._action("quit", quit=True)
        if self._timer is not None:
            self.after_cancel(self._timer)
        self.destroy()


def main():
    parser = argparse.ArgumentParser(description="独立的手势交互工作台 UI 原型")
    parser.add_argument("--external", action="store_true", help="以等待外部快照的模式启动")
    args = parser.parse_args()
    HandWorkbench(demo=not args.external).mainloop()


if __name__ == "__main__":
    main()
