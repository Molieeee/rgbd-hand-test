"""B02 几何与 B03 数据契约；运行 python -m unittest discover -s tests。"""
import unittest

import numpy as np

from render import ClayMesh


class ClayMeshTests(unittest.TestCase):
    def test_closed_oriented_shared_topology(self):
        for n, h, c in ((8, 1, 1), (24, 8, 3), (96, 48, 4)):
            clay = ClayMesh(radial_segments=n, height_segments=h, cap_segments=c)
            mesh = clay.mesh_local()
            v, f = clay.vertices_local, clay.triangles
            self.assertEqual(len(v), n * (h + 1 + 2 * (c - 1)) + 2)
            self.assertEqual(len(f), 2 * n * h + 2 * n * (2 * c - 1))
            self.assertEqual(len(np.unique(v, axis=0)), len(v))
            self.assertTrue(mesh.is_watertight())
            self.assertTrue(mesh.is_orientable())
            self.assertFalse(mesh.is_self_intersecting())
            # 每条无向边恰好有两个面，且绕向相反。
            edges = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
            _, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
            np.testing.assert_array_equal(counts, 2)
            self.assertEqual(set(map(tuple, edges)), set(map(tuple, edges[:, ::-1])))
            a, b, d = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
            cross = np.cross(b - a, d - a)
            self.assertTrue((np.linalg.norm(cross, axis=1) > 0).all())
            # 对凸初始圆柱，外法线与面中心的点积必须为正。
            self.assertTrue((np.sum(cross * (a + b + d) / 3, axis=1) > 0).all())

    def test_edits_pose_snapshots_versions_reset(self):
        clay = ClayMesh(radial_segments=16, height_segments=4)
        initial = clay.initial_vertices
        vertices = clay.vertices_local
        vertices[20, 0] *= .8
        clay.set_vertices(vertices)
        triangles = clay.triangles
        pose = np.eye(4)
        pose[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
        pose[:3, 3] = [.1, -.09, -.5]
        clay.set_pose(pose)
        self.assertEqual((clay.shape_version, clay.version), (1, 2))
        np.testing.assert_array_equal(clay.vertices_local, vertices)
        np.testing.assert_allclose(np.asarray(clay.mesh_scene().vertices),
                                   vertices @ pose[:3, :3].T + pose[:3, 3])
        clay.set_vertices(vertices)
        clay.set_pose(pose)
        self.assertEqual(clay.version, 2)
        vertices[:] = 0
        pose[:] = 0
        initial[:] = 0
        self.assertTrue(clay.vertices_local.any())
        clay.reset()
        np.testing.assert_array_equal(clay.vertices_local, clay.initial_vertices)
        np.testing.assert_array_equal(clay.triangles, triangles)
        self.assertEqual(clay.pose_scene[2, 3], -.5)

    def test_reject_invalid_input_without_mutation(self):
        for kwargs in ({"radius": 0}, {"height": float("nan")},
                       {"radial_segments": 2}, {"height_segments": 1.5},
                       {"cap_segments": True}):
            with self.assertRaises(ValueError):
                ClayMesh(**kwargs)
        clay = ClayMesh()
        for vertices in (np.zeros((1, 3)), np.full_like(clay.vertices_local, np.nan)):
            with self.assertRaises(ValueError):
                clay.set_vertices(vertices)
        for pose in (np.zeros((4, 4)), np.diag([2, 1, 1, 1]), np.diag([-1, 1, 1, 1])):
            with self.assertRaises(ValueError):
                clay.set_pose(pose)
        self.assertEqual(clay.version, 0)


if __name__ == "__main__":
    unittest.main()
