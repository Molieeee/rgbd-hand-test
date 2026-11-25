"""合并前回归：左右手槽位、吸附释放、旋转方块的支撑点。"""
import unittest
from types import SimpleNamespace

import numpy as np

from interaction.manipulator import Manipulator
from interaction.physics import FallingObject
from perception.handedness import assign_hand_slots, handedness_label


class HandOptimizationTests(unittest.TestCase):
    def test_detection_order_does_not_change_label_slots(self):
        self.assertEqual(assign_hand_slots(['left', 'right']), [0, 1])
        self.assertEqual(assign_hand_slots(['right', 'left']), [1, 0])
        self.assertEqual(assign_hand_slots([None, 'left']), [1, 0])
        self.assertEqual(assign_hand_slots(['left', 'left', None]), [0, 1, None])
        self.assertEqual(assign_hand_slots(['right'], max_hands=0), [None])
        categories = [SimpleNamespace(category_name='Left', score=.2),
                      SimpleNamespace(category_name='Right', score=.9)]
        self.assertEqual(handedness_label(categories), 'right')
        self.assertIsNone(handedness_label([]))

    def test_snap_and_missing_timeout_then_reset(self):
        hand = np.eye(4)
        obj = np.eye(4)
        obj[0, 3] = .06
        manip = Manipulator()
        obj, grabbed = manip.update(hand, True, obj, .01)
        self.assertTrue(grabbed)
        obj, _ = manip.update(hand, True, obj, .06)
        self.assertAlmostEqual(obj[0, 3], .03)
        obj, _ = manip.update(hand, True, obj, .06)
        np.testing.assert_allclose(obj[:3, 3], hand[:3, 3])
        _, grabbed = manip.update_missing(obj, .4)
        self.assertTrue(grabbed)
        _, grabbed = manip.update(hand, True, obj, .01)
        self.assertTrue(grabbed)
        _, grabbed = manip.update_missing(obj, .79)
        self.assertTrue(grabbed)
        _, grabbed = manip.update_missing(obj, .02)
        self.assertFalse(grabbed)
        manip.reset()
        self.assertIsNone(manip.T_hand_object)
        self.assertFalse(manip.grabbed)

    def test_unpinch_has_release_delay(self):
        manip = Manipulator()
        pose = np.eye(4)
        manip.update(pose, True, pose)
        self.assertTrue(manip.update(pose, False, pose, .2)[1])
        self.assertFalse(manip.update(pose, False, pose, .16)[1])

    def test_rotated_lowest_corner_inside_table(self):
        physics = FallingObject()
        angle = 1.0
        pose = np.eye(4)
        pose[:3, :3] = [[np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        pose[:3, 3] = [.455, .3, .5]
        self.assertEqual(physics._ground_height(pose, .025), physics.table_top)
        resolved = physics.update(pose, True, .01, .025)
        self.assertAlmostEqual(resolved[1, 3] + physics._support_radius(pose[:3, :3], .025),
                               physics.table_top - physics.contact_epsilon)

    def test_rotated_lowest_corner_outside_table(self):
        physics = FallingObject()
        angle = -1.0
        pose = np.eye(4)
        pose[:3, :3] = [[np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        pose[:3, 3] = [.445, .3, .5]
        self.assertEqual(physics._ground_height(pose, .025), physics.floor_top)


if __name__ == '__main__':
    unittest.main()
