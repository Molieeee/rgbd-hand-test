"""Tk 在主进程运行，Open3D/相机在独立进程运行；队列只保留最新场景快照。"""
from __future__ import annotations

import multiprocessing as mp
import queue
import traceback
import time


def latest_put(channel, value):
    try:
        channel.put_nowait(value)
    except queue.Full:
        try:
            channel.get_nowait()
        except queue.Empty:
            pass
        try:
            channel.put_nowait(value)
        except queue.Full:
            pass


def drain(channel):
    values = []
    while True:
        try:
            values.append(channel.get_nowait())
        except queue.Empty:
            return values


def backend(args, snapshots, actions, stop):
    try:
        from clay_runtime import run_clay
        run_clay(args, lambda value: latest_put(snapshots, value),
                 lambda: drain(actions), stop.is_set)
    except Exception as exc:
        traceback.print_exc()
        latest_put(snapshots, {"scene_kind": "clay", "running": False,
                               "error": f"三维后端启动失败：{exc}", "hands": []})


def launch(args):
    from workbench_ui import HandWorkbench
    context = mp.get_context("spawn")
    snapshots = context.Queue(maxsize=2)
    actions = context.Queue(maxsize=128)
    stop = context.Event()
    app = HandWorkbench(demo=False, clay_mode=True)
    app.input_source.set(args.source)
    process = context.Process(target=backend, args=(args, snapshots, actions, stop),
                              name="clay-render-backend")
    process.start()
    last_snapshot = {"scene_kind": "clay", "running": False, "hands": [],
                     "message": "正在启动 Open3D 泥坯场景…"}

    def tick():
        nonlocal last_snapshot
        if app._closed:
            return
        for snapshot in drain(snapshots):
            last_snapshot = snapshot
            app.publish_snapshot(snapshot)
        for action in app.poll_actions():
            try:
                actions.put_nowait(action)
            except queue.Full:
                app.notify("操作较多，请稍候重试")
        if not process.is_alive():
            last_snapshot = dict(last_snapshot, running=False,
                                 error="三维窗口已关闭，请关闭工作台后重新启动")
            app.publish_snapshot(last_snapshot)
            app.run_button.configure(state="disabled")
        app.after(50, tick)

    app.publish_snapshot(last_snapshot)
    app.after(50, tick)
    try:
        app.mainloop()
    finally:
        stop.set()
        # 继续接收正在写入的图像，直到后台 feeder 完成；禁止提前
        # cancel_join_thread()，否则 Queue 内半个图像包会使接收方永久阻塞。
        deadline = time.monotonic() + 3
        while process.is_alive() and time.monotonic() < deadline:
            drain(snapshots)
            process.join(timeout=.05)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
        for channel in (snapshots, actions):
            channel.cancel_join_thread()
            channel.close()
    return 0


if __name__ == "__main__":
    from main import main
    raise SystemExit(main())
