"""Known-scene subtraction checks without target poses, labels or physics."""
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from ..sensors import capture_observation, robot_feedback
from ..test_sensors import fixture
from .known_static_filter import checked_model, filter_depth, render_known_depth


class KnownStaticFilterTests(unittest.TestCase):
    def test_fixed_one_mm_match_preserves_unknown_foreground(self):
        raw = np.array([[.5000, .5091, .5100, np.nan, 0.]])
        known = np.array([[.5004, .5100, .5200, .5, .5]])
        filtered, removed = filter_depth(raw, known)
        np.testing.assert_array_equal(removed, [[True, True, False, False, False]])
        self.assertTrue(np.isnan(filtered[0, 0]))
        self.assertEqual(filtered[0, 2], raw[0, 2])

    def test_predictive_depth_uses_known_model_and_robot_feedback_only(self):
        original, data = fixture()
        observation = capture_observation(original, data)
        with tempfile.TemporaryDirectory() as directory:
            template = Path(directory) / "known_static.xml"
            mujoco.mj_saveLastXML(str(template), original)
            model, model_data = checked_model(template, robot_feedback(original, data))
            entry = observation["cameras"]["board_inspection"]
            prediction = render_known_depth(model, model_data, entry["calibration"])
            raw = entry["depth_m"].copy()
            valid = np.isfinite(prediction) & (prediction > 0)
            self.assertGreater(valid.sum(), 100)
            self.assertLess(float(np.max(abs(raw[valid] - prediction[valid]))), .0001)
            pixels = np.argwhere(valid)[::max(1, valid.sum() // 10)][:10]
            raw[tuple(pixels.T)] = prediction[tuple(pixels.T)] - .010
            _, removed = filter_depth(raw, prediction)
            self.assertFalse(removed[tuple(pixels.T)].any())
            remaining = valid.copy()
            remaining[tuple(pixels.T)] = False
            self.assertTrue(removed[remaining].all())


if __name__ == "__main__":
    unittest.main()
