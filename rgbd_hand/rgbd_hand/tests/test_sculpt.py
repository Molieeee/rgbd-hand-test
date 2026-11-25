"""D01-D05 雕刻/平滑/笔画测试；运行 python -m unittest discover -s tests。"""
import unittest

import numpy as np
import open3d as o3d

from interaction.sculpt import (
    MAX_SMOOTH_ALPHA,
    MAX_SMOOTH_STEP_M,
    MAX_STEP_DEPTH_M,
    MAX_STROKE_DEPTH_M,
    SMOOTH_RATE_PER_S,
    BrushCommand,
    PickResult,
    Selection,
    StrokeSampler,
    Topology,
    apply_brush,
    carve,
    restore,
    select_vertices,
    smooth,
    snapshot,
)
from render import ClayMesh


class SculptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clay = ClayMesh()
        cls.vertices = cls.clay.vertices_local
        cls.triangles = cls.clay.triangles
        cls.topology = Topology(cls.vertices, cls.triangles)
        cls.radius = cls.clay.radius
        cls.front = np.array([0.0, 0.0, -cls.radius])
        cls.normal = np.array([0.0, 0.0, -1.0])
        cls.center = cls.topology.nearest_vertex(cls.front, cls.vertices)
        cls.side_count = cls.clay.radial_segments * (cls.clay.height_segments + 1)
        cls.cap_only = np.arange(cls.side_count, len(cls.vertices))

    def setUp(self):
        self.v = self.vertices.copy()

    def pick(self, point=None, normal=None, valid=True):
        return PickResult(
            valid, 0, self.front if point is None else point,
            self.normal if normal is None else normal, 0.0, 0,
        )

    # ---- D02 拓扑与邻域 --------------------------------------------------

    def test_topology_matches_welded_closed_mesh(self):
        topo = self.topology
        self.assertEqual(topo.vertex_count, len(self.vertices))
        self.assertEqual(len(topo.triangles), len(self.triangles))
        self.assertEqual(len(topo.edges), 3 * len(self.triangles) // 2)
        self.assertEqual(len(topo.boundary_edges), 0)
        self.assertEqual(len(topo.nonmanifold_edges), 0)
        self.assertGreater(topo.max_edge_length(self.vertices), 0)
        # 每条邻接边都对称，且没有自环。
        for i, neighbors in enumerate(topo.adjacency):
            for j in neighbors:
                self.assertNotEqual(int(j), i)
                self.assertIn(i, topo.adjacency[int(j)])
        # 侧面法线水平朝外；两个端面中心法线分别朝 +y/-y。
        normals = topo.vertex_normals(self.vertices)
        self.assertTrue(np.isfinite(normals).all())
        self.assertGreater(np.linalg.norm(normals, axis=1).min(), 0.99)
        side = self.clay.side_vertex_indices
        radial = self.vertices[side][:, [0, 2]]
        radial /= np.linalg.norm(radial, axis=1)[:, None]
        side_dots = np.sum(normals[side][:, [0, 2]] * radial, axis=1)
        # 端面外环复用了侧面顶点，法线会同时包含端面面积，因此只有内部
        # 侧面环是严格水平的；上下环仍应保持正的径向分量。
        self.assertGreater(side_dots.min(), 0.1)
        interior_rows = np.arange(
            self.clay.radial_segments,
            self.clay.radial_segments * self.clay.height_segments,
        )
        interior_radial = self.vertices[interior_rows][:, [0, 2]]
        interior_radial /= np.linalg.norm(interior_radial, axis=1)[:, None]
        interior_dots = np.sum(
            normals[interior_rows][:, [0, 2]] * interior_radial, axis=1
        )
        self.assertGreater(interior_dots.min(), 0.99)
        top_center = topo.nearest_vertex(
            np.array([0.0, self.clay.height / 2, 0.0]), self.vertices
        )
        bottom_center = topo.nearest_vertex(
            np.array([0.0, -self.clay.height / 2, 0.0]), self.vertices
        )
        self.assertAlmostEqual(normals[top_center, 1], 1.0, places=12)
        self.assertAlmostEqual(normals[bottom_center, 1], -1.0, places=12)

    def test_selection_excludes_backside_and_caps(self):
        selection = select_vertices(
            self.v, self.front, self.normal, 0.024, self.topology
        )
        self.assertGreater(len(selection), 30)
        self.assertTrue(selection.mask[self.center])
        self.assertAlmostEqual(selection.weights[self.center], 1.0, places=12)
        # 所有入选点都在半径内且权重为正。
        self.assertTrue(
            np.all(selection.distances_m[selection.mask] <= 0.024 + 1e-12)
        )
        self.assertTrue(np.all(selection.weights[selection.mask] > 0))
        self.assertEqual(len(selection.indices), int(selection.mask.sum()))
        # 半径外、背面、端面内圈都不在邻域中。
        outside = selection.distances_m > 0.024 + 1e-12
        self.assertFalse(selection.mask[outside].any())
        back = self.topology.nearest_vertex(np.array([0.0, 0.0, self.radius]), self.v)
        self.assertFalse(selection.mask[back])
        self.assertFalse(selection.mask[self.cap_only].any())
        # 测地距只可能进一步收紧选择，不能扩张。
        no_geo = select_vertices(
            self.v, self.front, self.normal, 0.024, self.topology, geodesic=False
        )
        self.assertGreaterEqual(len(no_geo), len(selection))
        self.assertTrue(selection.geodesic_m is not None)
        self.assertTrue(np.isfinite(selection.geodesic_m[selection.indices]).all())
        # 没有拓扑时仍能靠切平面项排除背面。
        no_topo = select_vertices(self.v, self.front, self.normal, 0.024, None)
        self.assertFalse(no_topo.mask[back])
        self.assertTrue(no_topo.mask[self.center])

    def test_selection_is_continuous_across_welded_seam(self):
        # 命中圆柱接缝 j=0：j=1 与 j=n-1 必须得到相同的权重，
        # 不能像未焊接网格那样在接缝处裂开。B 的网格没有重复顶点。
        seam_point = np.array([self.radius, 0.0, 0.0])
        seam_normal = np.array([1.0, 0.0, 0.0])
        selection = select_vertices(
            self.v, seam_point, seam_normal, 0.024, self.topology
        )
        row = self.clay.height_segments // 2
        n = self.clay.radial_segments
        center = row * n
        left = row * n + n - 1
        right = row * n + 1
        self.assertTrue(selection.mask[center])
        self.assertTrue(selection.mask[left])
        self.assertTrue(selection.mask[right])
        self.assertAlmostEqual(
            selection.weights[left], selection.weights[right], places=12
        )
        self.assertEqual(len(np.unique(self.v, axis=0)), len(self.v))

    def test_selection_rejects_invalid_parameters(self):
        for radius in (0.0, -0.01, float("nan")):
            with self.assertRaises(ValueError):
                select_vertices(self.v, self.front, self.normal, radius, self.topology)
        for normal in ([0.0, 0.0, 0.0], [np.nan, 0.0, 1.0]):
            with self.assertRaises(ValueError):
                select_vertices(self.v, self.front, normal, 0.024, self.topology)
        with self.assertRaises(ValueError):
            select_vertices(
                self.v, self.front, self.normal, 0.024, self.topology,
                side_tolerance=-0.1,
            )
        with self.assertRaises(ValueError):
            select_vertices(
                self.v, self.front, self.normal, 0.024, self.topology,
                normal_cos_min=2.0,
            )

    # ---- D01/D03 雕刻 ----------------------------------------------------

    def test_carve_displaces_front_only_and_scales_with_dt(self):
        result = carve(
            self.v, self.pick(), BrushCommand("carve", 0.024, 1.0, 1 / 30, 7),
            self.topology, stroke_start=self.v,
        )
        self.assertTrue(result.applied)
        self.assertIsNot(result.vertices, self.v)
        np.testing.assert_array_equal(self.v, self.vertices)  # 输入未被修改
        expected = 0.060 / 30.0
        self.assertAlmostEqual(result.max_displacement_m, expected, places=12)
        self.assertAlmostEqual(result.step_depth_m, expected, places=12)
        np.testing.assert_allclose(
            result.vertices[self.center] - self.v[self.center],
            [0.0, 0.0, expected], atol=1e-12,
        )
        self.assertGreater(len(result.changed_indices), 30)
        np.testing.assert_array_equal(
            result.changed_indices, np.flatnonzero(result.changed_mask)
        )
        # 背面（z >= 0）、端面内圈和三角拓扑都不变。
        self.assertFalse((result.changed_mask & (self.v[:, 2] >= 0)).any())
        self.assertFalse(result.changed_mask[self.cap_only].any())
        np.testing.assert_array_equal(self.topology.triangles, self.triangles)
        self.assertTrue(np.isfinite(result.vertices).all())
        self.assertEqual((result.tool, result.stroke_id), ("carve", 7))

    def test_carve_clamps_single_step_and_whole_stroke(self):
        # dt 很大时单步被限制到 4 mm。
        result = carve(
            self.v, self.pick(), BrushCommand("carve", 0.024, 1.0, 1.0, 1),
            self.topology,
        )
        self.assertAlmostEqual(result.step_depth_m, MAX_STEP_DEPTH_M, places=12)
        self.assertAlmostEqual(result.max_displacement_m, MAX_STEP_DEPTH_M, places=12)

        # 同一笔画累计最多 12 mm；命中点跟随凹坑中心，模拟每帧重新拾取。
        current = self.v.copy()
        start = self.v.copy()
        point = self.front.copy()
        for _ in range(8):
            result = carve(
                current, self.pick(point), BrushCommand("carve", 0.024, 1.0, 0.1, 3),
                self.topology, stroke_start=start,
            )
            current = result.vertices
            point = current[self.center].copy()
        total = np.linalg.norm(current - start, axis=1)
        self.assertAlmostEqual(total.max(), MAX_STROKE_DEPTH_M, places=9)
        self.assertTrue(np.isfinite(current).all())

    def test_concave_dent_still_selects_surrounding_vertices(self):
        current = self.v.copy()
        start = self.v.copy()
        point = self.front.copy()
        for _ in range(3):
            result = carve(
                current, self.pick(point), BrushCommand("carve", 0.024, 1.0, 0.1, 4),
                self.topology, stroke_start=start,
            )
            current = result.vertices
            point = current[self.center].copy()
        selection = select_vertices(
            current, current[self.center], self.normal, 0.024, self.topology
        )
        self.assertTrue(selection.mask[self.center])
        self.assertGreater(len(selection), 20)
        self.assertFalse(selection.mask[self.topology.nearest_vertex(
            np.array([0.0, 0.0, self.radius]), current)])

    def test_invalid_pick_and_command_are_noops(self):
        good = BrushCommand("carve", 0.024, 1.0, 1 / 30, 1)
        cases = [
            (PickResult.miss(), good),
            (self.pick(normal=[0.0, 0.0, 0.0]), good),
            (self.pick(point=[np.nan, 0.0, 0.0]), good),
            (self.pick(), BrushCommand("carve", 0.0, 1.0, 1 / 30, 1)),
            (self.pick(), BrushCommand("carve", -0.01, 1.0, 1 / 30, 1)),
            (self.pick(), BrushCommand("carve", 0.024, 0.0, 1 / 30, 1)),
            (self.pick(), BrushCommand("carve", 0.024, 1.0, 0.0, 1)),
            (self.pick(), BrushCommand("carve", 0.024, 1.0, -1.0, 1)),
            (self.pick(), BrushCommand("carve", 0.024, 1.0, np.nan, 1)),
            (self.pick(), BrushCommand("unknown", 0.024, 1.0, 1 / 30, 1)),
            (self.pick(), None),
            (None, good),
        ]
        for pick, command in cases:
            result = apply_brush(self.v, pick, command, self.topology)
            self.assertFalse(result.applied)
            np.testing.assert_array_equal(result.vertices, self.v)
            self.assertEqual(result.changed_indices.size, 0)
        np.testing.assert_array_equal(self.v, self.vertices)

    def test_carved_mesh_stays_closed_and_outward(self):
        current = self.v.copy()
        start = self.v.copy()
        point = self.front.copy()
        for _ in range(3):
            result = carve(
                current, self.pick(point), BrushCommand("carve", 0.024, 1.0, 0.1, 5),
                self.topology, stroke_start=start,
            )
            current = result.vertices
            point = current[self.center].copy()
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(current),
            o3d.utility.Vector3iVector(self.triangles),
        )
        mesh.compute_vertex_normals()
        self.assertTrue(mesh.is_watertight())
        self.assertTrue(mesh.is_orientable())
        self.assertFalse(mesh.is_self_intersecting())
        a, b, c = (current[self.triangles[:, k]] for k in range(3))
        cross = np.cross(b - a, c - a)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        self.assertGreater(areas.min(), 1e-12)
        centers = (a + b + c) / 3.0
        side_faces = np.abs(centers[:, 1]) < self.clay.height / 2 - 1e-9
        side_cross = cross[side_faces][:, [0, 2]]
        side_centers = centers[side_faces][:, [0, 2]]
        outward = np.sum(side_cross * side_centers, axis=1)
        self.assertGreater(outward.min(), 0)

    # ---- D04 平滑 --------------------------------------------------------

    def test_smooth_uses_preupdate_neighbors_and_stays_local(self):
        # 合成 2x3 平面条带，便于逐点手算。
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0]],
            dtype=np.float64,
        )
        triangles = np.array([[0, 3, 1], [1, 3, 4], [1, 4, 2], [2, 4, 5]])
        topology = Topology(vertices, triangles)
        original = vertices.copy()
        selection = Selection(
            mask=np.ones(6, dtype=bool),
            indices=np.arange(6, dtype=np.int64),
            weights=np.ones(6),
            distances_m=np.zeros(6),
            source_index=1,
        )
        command = BrushCommand("smooth", 10.0, 1.0, 0.01, 9)
        result = smooth(
            vertices, self.pick(vertices[1], [0, 0, 1]), command, topology,
            selection=selection, max_smooth_step_m=None, max_stroke_depth_m=None,
        )
        alpha = min(MAX_SMOOTH_ALPHA, SMOOTH_RATE_PER_S * 0.01)
        expected = vertices.copy()
        for i in range(6):
            neighbors = topology.adjacency[i]
            expected[i] += alpha * (vertices[neighbors].mean(axis=0) - vertices[i])
        # 如果实现是顺序更新，后面的顶点会用到前面已移动的邻居，结果不同。
        np.testing.assert_allclose(result.vertices, expected, atol=1e-12)
        self.assertTrue(result.applied)
        np.testing.assert_array_equal(vertices, original)

    def test_smooth_reduces_dent_and_anchors_boundary(self):
        carved = carve(
            self.v, self.pick(), BrushCommand("carve", 0.024, 1.0, 1 / 30, 1),
            self.topology,
        )
        current = carved.vertices
        selection = select_vertices(
            current, current[self.center], self.normal, 0.024, self.topology
        )
        result = smooth(
            current, self.pick(current[self.center]),
            BrushCommand("smooth", 0.024, 1.0, 1 / 30, 2),
            self.topology, selection=selection,
        )
        self.assertTrue(result.applied)
        self.assertGreater(len(result.changed_indices), 0)
        self.assertLessEqual(result.max_displacement_m, MAX_SMOOTH_STEP_M + 1e-12)
        # 笔刷外的顶点完全不动，边界权重为 0。
        self.assertFalse(result.changed_mask[~selection.mask].any())
        # 压入后 z 增大（朝 +z），平滑应让这个深度变小。
        before = current[self.center, 2] - self.v[self.center, 2]
        after = result.vertices[self.center, 2] - self.v[self.center, 2]
        self.assertLess(after, before)
        # 单步平滑不应造成明显整体收缩。
        side = self.clay.side_vertex_indices
        r_before = np.linalg.norm(current[side][:, [0, 2]], axis=1).mean()
        r_after = np.linalg.norm(result.vertices[side][:, [0, 2]], axis=1).mean()
        self.assertLess(abs(r_after - r_before), 5e-4)
        self.assertTrue(np.isfinite(result.vertices).all())

    def test_smooth_needs_topology(self):
        result = apply_brush(
            self.v, self.pick(), BrushCommand("smooth", 0.024, 1.0, 1 / 30, 1),
            None,
        )
        self.assertFalse(result.applied)
        self.assertIn("topology", result.reason)

    # ---- D05 快照与笔画 --------------------------------------------------

    def test_snapshot_restore_are_independent(self):
        source = self.v.copy()
        saved = snapshot(source)
        source[0] += 100.0
        np.testing.assert_array_equal(saved, self.vertices)
        saved[1] += 2.0
        restored = restore(saved)
        np.testing.assert_array_equal(restored, saved)
        restored[2] += 3.0
        np.testing.assert_array_equal(saved[1], self.vertices[1] + 2.0)
        np.testing.assert_array_equal(saved[2], self.vertices[2])

    def test_stroke_sampler_interpolates_and_ends(self):
        sampler = StrokeSampler(spacing_factor=0.4, min_spacing_m=0.001)
        p0 = np.array([0.0, 0.0, 0.0])
        n0 = np.array([0.0, 0.0, 1.0])
        first = sampler.begin(self.v, p0, n0, stroke_id=7)
        self.assertTrue(sampler.active)
        self.assertEqual(sampler.stroke_id, 7)
        np.testing.assert_array_equal(first.point_local, p0)
        start = sampler.stroke_start
        steps = sampler.step(np.array([0.1, 0.0, 0.0]), n0, 0.05)
        self.assertEqual(len(steps), 5)  # 间距 0.05*0.4 = 0.02
        np.testing.assert_allclose(steps[-1].point_local, [0.1, 0.0, 0.0])
        for stamp in steps:
            self.assertAlmostEqual(np.linalg.norm(stamp.normal_local), 1.0, places=12)
        ended = sampler.end()
        np.testing.assert_array_equal(ended, start)
        self.assertFalse(sampler.active)
        with self.assertRaises(RuntimeError):
            sampler.step(p0, n0, 0.05)

    def test_stroke_sampler_large_jump_ends_stroke(self):
        with self.assertRaises(ValueError):
            StrokeSampler(max_stamps_per_step=0)
        sampler = StrokeSampler(max_stamps_per_step=4)
        p0 = np.array([0.0, 0.0, 0.0])
        n0 = np.array([0.0, 0.0, 1.0])
        sampler.begin(self.v, p0, n0, stroke_id=1)
        # 1 m 的跳变 / 0.02 m 间距 = 50 个印章，超过上限，直接结束。
        self.assertEqual(sampler.step(np.array([1.0, 0.0, 0.0]), n0, 0.05), [])
        self.assertFalse(sampler.active)

    def test_stroke_sampler_zero_motion_and_invalid_hit(self):
        sampler = StrokeSampler()
        p0 = np.array([0.0, 0.0, 0.0])
        n0 = np.array([0.0, 0.0, 1.0])
        sampler.begin(self.v, p0, n0, stroke_id=1)
        self.assertEqual(len(sampler.step(p0, n0, 0.05)), 1)
        sampler.end()
        sampler.begin(self.v, p0, n0, stroke_id=2)
        self.assertEqual(sampler.step([np.nan, 0.0, 0.0], n0, 0.05), [])
        self.assertFalse(sampler.active)


if __name__ == "__main__":
    unittest.main()
