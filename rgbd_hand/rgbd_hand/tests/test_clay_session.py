import unittest
from unittest.mock import patch

import numpy as np

from clay_runtime import ClaySession
import main


class ClaySessionTests(unittest.TestCase):
    def test_pause_reset_and_parameter_application(self):
        session = ClaySession()
        session.action({"type": "configure", "brush_radius_m": .03, "demo_depth_m": .01})
        session.action({"type": "dent"})
        self.assertGreater(session.snapshot()["changed_vertices"], 0)
        self.assertAlmostEqual(float(np.max(np.linalg.norm(
            session.clay.vertices_local - session.clay.initial_vertices, axis=1))), .01)
        session.action({"type": "turn", "angle": .4})
        session.action({"type": "pause"})
        version = session.clay.version
        session.action({"type": "dent"})
        self.assertEqual(session.clay.version, version)
        session.action({"type": "reset"})
        self.assertFalse(session.running)
        self.assertEqual(session.snapshot()["changed_vertices"], 0)
        np.testing.assert_array_equal(session.clay.pose_scene, session.initial_pose)
        session.action({"type": "start"})
        session.action({"type": "dent"})
        self.assertGreater(session.snapshot()["changed_vertices"], 0)

    def test_parameters_are_clamped_and_invalid_values_rejected(self):
        session = ClaySession()
        session.action({"type": "configure", "brush_radius_m": 1, "demo_depth_m": -1})
        self.assertEqual(session.radius_m, .07)
        self.assertEqual(session.depth_m, .001)
        with self.assertRaises(ValueError):
            session.action({"type": "configure", "brush_radius_m": float("nan")})
        self.assertEqual(session.radius_m, .07)

    def test_default_main_launches_clay_workbench(self):
        with patch("workbench_app.launch", return_value=0) as launch, patch("main.build_source") as capture:
            self.assertEqual(main.main([]), 0)
            args = launch.call_args.args[0]
            self.assertEqual((args.scene, args.source), ("clay", "demo"))
            capture.assert_not_called()

    def test_no_ui_uses_same_clay_backend(self):
        with patch("clay_runtime.run_clay", return_value=0) as run:
            self.assertEqual(main.main(["--no-ui", "--source", "webcam"]), 0)
            self.assertEqual(run.call_args.args[0].source, "webcam")


if __name__ == "__main__":
    unittest.main()
