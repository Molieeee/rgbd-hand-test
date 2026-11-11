"""Open3D 三维场景。

相机坐标系 (x 右, y 下, z 前) 与场景坐标系 (x 右, y 上, z 朝观察者) 之间
用一个旋转矩阵转换：

    M = diag(1, -1, -1)

det(M) = +1，所以它同时也可以直接作用在旋转矩阵上。

场景里目前包含：手部骨架（LineSet + 21 个球）、一个可被抓取的立方体、
以及下方带网格线的桌面（用来给掉落提供落点）。
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import open3d as o3d

from perception.hand3d import HAND_CONNECTIONS, INDEX_TIP, THUMB_TIP

SCENE_M = np.diag([1.0, -1.0, -1.0])

JOINT_COLOR = np.array([0.24, 0.72, 0.95])
WRIST_COLOR = np.array([0.95, 0.55, 0.15])
PINCH_COLOR = np.array([0.25, 0.95, 0.45])
LINE_COLOR = np.array([0.78, 0.84, 0.90])
OBJECT_COLOR = np.array([0.90, 0.35, 0.35])
OBJECT_NEAR_COLOR = np.array([0.95, 0.75, 0.25])
OBJECT_GRABBED_COLOR = np.array([0.35, 0.90, 0.50])
TABLE_COLOR = np.array([0.34, 0.28, 0.23])
GRID_COLOR = np.array([0.46, 0.40, 0.35])


def to_scene(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return (SCENE_M @ points.reshape(-1, 3).T).T


class HandVisual:
    """一只手：骨架线 + 关节球。"""

    def __init__(self, vis: o3d.visualization.Visualizer, joint_radius: float = 0.009):
        zeros = np.zeros((21, 3))
        self.skeleton = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(zeros),
            lines=o3d.utility.Vector2iVector(HAND_CONNECTIONS),
        )
        self.skeleton.colors = o3d.utility.Vector3dVector(
            np.tile(LINE_COLOR, (len(HAND_CONNECTIONS), 1))
        )

        self.joints = [o3d.geometry.TriangleMesh.create_sphere(joint_radius, resolution=10)
                       for _ in range(21)]
        for i, joint in enumerate(self.joints):
            joint.paint_uniform_color(WRIST_COLOR if i == 0 else JOINT_COLOR)
            joint.compute_vertex_normals()

        self.vis = vis
        self._positions = zeros.copy()
        self._pinching = False
        self.added = False

    def add(self) -> None:
        if self.added:
            return
        self.vis.add_geometry(self.skeleton, reset_bounding_box=False)
        for joint in self.joints:
            self.vis.add_geometry(joint, reset_bounding_box=False)
        self.added = True

    def remove(self) -> None:
        if not self.added:
            return
        self.vis.remove_geometry(self.skeleton, reset_bounding_box=False)
        for joint in self.joints:
            self.vis.remove_geometry(joint, reset_bounding_box=False)
        self.added = False

    def update(self, points_cam: np.ndarray, pinching: bool) -> None:
        points = to_scene(points_cam)
        if points.shape != (21, 3):
            return

        self.skeleton.points = o3d.utility.Vector3dVector(points)

        for i, joint in enumerate(self.joints):
            delta = points[i] - self._positions[i]
            if np.any(delta):
                joint.translate(delta, relative=True)
            self._positions[i] = points[i]

        if pinching != self._pinching:
            self._pinching = pinching
            color = PINCH_COLOR if pinching else JOINT_COLOR
            for i in (THUMB_TIP, INDEX_TIP):
                self.joints[i].paint_uniform_color(color)

        self.vis.update_geometry(self.skeleton)
        for joint in self.joints:
            self.vis.update_geometry(joint)


class Scene:
    def __init__(
        self,
        window_name: str = "RGBD Hand - 3D Scene",
        width: int = 1280,
        height: int = 720,
        max_hands: int = 2,
        joint_radius: float = 0.009,
        object_size: float = 0.05,
        focus_distance: float = 0.5,
        first_person: bool = True,
        eye_distance: float = 0.50,
        eye_height: float = 0.30,
        look_at_y: float = 0.0,
        table_top: float = 0.18,
        table_size: tuple[float, float] = (0.9, 0.60),
        table_center_z: float = 0.55,
        hfov_deg: float = 60.0,
    ):
        self.width = int(width)
        self.height = int(height)
        self.focus_distance = float(focus_distance)
        self.first_person = bool(first_person)
        self.eye_distance = float(eye_distance)
        self.eye_height = float(eye_height)
        self.look_at_y = float(look_at_y)
        self.table_top = float(table_top)
        self.table_size = tuple(map(float, table_size))
        self.table_center_z = float(table_center_z)
        self.hfov_deg = float(hfov_deg)
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=window_name, width=width, height=height)

        opt = self.vis.get_render_option()
        opt.background_color = np.array([0.08, 0.09, 0.12])
        opt.line_width = 3.0

        self.hands = [HandVisual(self.vis, joint_radius) for _ in range(max_hands)]

        # 立方体：几何中心在原点，便于直接用物体的 4x4 位姿摆放。
        box = o3d.geometry.TriangleMesh.create_box(object_size, object_size, object_size)
        box.translate(-np.array([object_size / 2.0] * 3))
        box.paint_uniform_color(OBJECT_COLOR)
        box.compute_vertex_normals()
        self.box = box
        self._box_vertices = np.asarray(box.vertices).copy()
        self._box_normals = np.asarray(box.vertex_normals).copy()
        self._object_grabbed = False
        self._object_near = False
        self._last_object_T: np.ndarray | None = None

        # 先把物体放到初始场景位置，再添加几何体。
        self.box.translate(np.array([0.0, 0.0, -self.focus_distance]), relative=True)
        self.vis.add_geometry(self.box, reset_bounding_box=True)
        self._add_table()
        self._frame_view()

    def _add_table(self) -> None:
        """桌面：一块薄板 + 顶面网格线，帮助判断高度。"""
        sx, sz = self.table_size
        thickness = 0.02
        top_scene_y = -self.table_top
        center = np.array([0.0, top_scene_y - thickness / 2.0, -self.table_center_z])

        slab = o3d.geometry.TriangleMesh.create_box(sx, thickness, sz)
        slab.translate(center - np.array([sx / 2.0, thickness / 2.0, sz / 2.0]))
        slab.paint_uniform_color(TABLE_COLOR)
        slab.compute_vertex_normals()
        self.vis.add_geometry(slab, reset_bounding_box=True)

        n = 6
        gx = np.linspace(-sx / 2.0, sx / 2.0, n)
        gz = np.linspace(-self.table_center_z - sz / 2.0, -self.table_center_z + sz / 2.0, n)
        y = top_scene_y + 0.001
        points, lines = [], []
        for z in gz:
            points.extend([[-sx / 2.0, y, z], [sx / 2.0, y, z]])
        for x in gx:
            points.extend([[x, y, -self.table_center_z - sz / 2.0],
                           [x, y, -self.table_center_z + sz / 2.0]])
        count = len(points) // 2
        lines = [[2 * i, 2 * i + 1] for i in range(count)]
        grid = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np.asarray(points)),
            lines=o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32)),
        )
        grid.colors = o3d.utility.Vector3dVector(np.tile(GRID_COLOR, (len(lines), 1)))
        self.vis.add_geometry(grid, reset_bounding_box=False)

    def _frame_view(self) -> None:
        """设置相机。

        first_person=True（默认）：相机放在手和物体靠近用户的一侧，
        朝屏幕方向看。这样看到的是“自己的手伸进屏幕”的第一人称视角：
        右手在画面右边，符合身体直觉，而且几何没有被镜像。

        first_person=False：相机在摄像头一侧，相当于第三人称（别人看你）。

        不能直接用 ViewControl.set_front/set_lookat/set_zoom：在当前 Open3D
        版本里它们会结合内部包围盒把相机推到很远的位置（画面全黑）。
        这里先用一个辅助包围盒让 Open3D 自己算出一个合法视角，再用
        PinholeCameraParameters 精确设置外参。
        """
        depth = self.focus_distance
        bounds = o3d.geometry.TriangleMesh.create_box(0.34, 0.34, 0.45)
        bounds.translate(np.array([-0.17, -0.17, -depth - 0.225]))
        self.vis.add_geometry(bounds, reset_bounding_box=True)
        self.vis.reset_view_point(True)
        self.vis.remove_geometry(bounds, reset_bounding_box=False)

        ctr = self.vis.get_view_control()
        params = ctr.convert_to_pinhole_camera_parameters()

        # Open3D 窗口默认竖直 FOV 是 60°（16:9 下水平约 91.5°），远大于
        # WebcamHand3D 假设的 60° 水平 FOV。不改的话，虚拟手的活动范围在
        # 画面里只占一半，看起来“手怎么动都够不到”。这里把 3D 视角的
        # 水平 FOV 也设成 60°，虚拟手和摄像头里的手移动幅度就一致了。
        fx = (self.width / 2.0) / np.tan(np.radians(self.hfov_deg) / 2.0)
        params.intrinsic.set_intrinsics(
            self.width, self.height, fx, fx,
            (self.width - 1) / 2.0, (self.height - 1) / 2.0,
        )

        # 视线对准方块中心（相机 y = look_at_y），而不是相机光轴平面。
        target = np.array([0.0, -self.look_at_y, -depth])
        if self.first_person:
            # 场景 z 轴与相机 z 轴相反：用户在 -z 一侧（更远离摄像头）。
            # 眼睛再抬高 eye_height，像人低头看自己的手和桌面。
            eye = target + np.array([0.0, self.eye_height, -self.eye_distance])
        else:
            eye = target + np.array([0.0, 0.0, 0.38])
        up = np.array([0.0, 1.0, 0.0])
        front = target - eye
        front /= np.linalg.norm(front)
        right = np.cross(front, up)
        right /= np.linalg.norm(right)
        down = np.cross(front, right)
        R = np.vstack([right, down, front])
        E = np.eye(4)
        E[:3, :3] = R
        E[:3, 3] = -R @ eye
        params.extrinsic = E
        # is_global=True 会跳过窗口尺寸检查；False 在部分 Open3D 版本上会失败。
        if not ctr.convert_from_pinhole_camera_parameters(params, True):
            ctr.convert_from_pinhole_camera_parameters(params, False)

    # ---- 手 --------------------------------------------------------------

    def update_hands(
        self, hands: Sequence[np.ndarray], pinching: Sequence[bool] | None = None
    ) -> None:
        pinching = list(pinching) if pinching is not None else [False] * len(hands)
        for i, visual in enumerate(self.hands):
            if i < len(hands) and hands[i] is not None:
                visual.add()
                visual.update(hands[i], bool(pinching[i]))
            else:
                visual.remove()

    # ---- 物体 ------------------------------------------------------------

    def update_object(
        self, object_T_cam: np.ndarray, grabbed: bool = False, near: bool = False
    ) -> None:
        object_T_cam = np.asarray(object_T_cam, dtype=np.float64)
        if (
            self._last_object_T is not None
            and np.allclose(object_T_cam, self._last_object_T, atol=1e-6)
            and grabbed == self._object_grabbed
            and near == self._object_near
        ):
            return
        self._last_object_T = object_T_cam.copy()

        R = SCENE_M @ object_T_cam[:3, :3]
        t = SCENE_M @ object_T_cam[:3, 3]

        self.box.vertices = o3d.utility.Vector3dVector(
            self._box_vertices @ R.T + t
        )
        self.box.vertex_normals = o3d.utility.Vector3dVector(
            self._box_normals @ R.T
        )

        if grabbed != self._object_grabbed or near != self._object_near:
            self._object_grabbed = grabbed
            self._object_near = near
            if grabbed:
                color = OBJECT_GRABBED_COLOR
            elif near:
                color = OBJECT_NEAR_COLOR
            else:
                color = OBJECT_COLOR
            self.box.paint_uniform_color(color)

        self.vis.update_geometry(self.box)

    # ---- 渲染 ------------------------------------------------------------

    def render(self) -> bool:
        if not self.vis.poll_events():
            return False
        self.vis.update_renderer()
        return True

    def close(self) -> None:
        self.vis.destroy_window()
