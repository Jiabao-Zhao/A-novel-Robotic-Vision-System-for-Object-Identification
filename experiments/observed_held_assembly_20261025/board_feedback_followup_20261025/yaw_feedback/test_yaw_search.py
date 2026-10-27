"""Focused yaw-bank coverage, raw-depth ranking and file boundary checks."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .yaw_search import _forward_score, yaw_offsets
from .estimate_from_files import estimate_packet, digest


class YawSearchTests(unittest.TestCase):
    def test_complete_circle_and_bounded_phase_banks(self):
        for pitch, count in ((6., 60), (9., 40)):
            offsets = yaw_offsets("full_circle", pitch)
            self.assertEqual(len(offsets), count)
            np.testing.assert_allclose(np.diff(np.r_[offsets, 360.]), pitch)
            np.testing.assert_allclose(yaw_offsets("local_phase", pitch), [0, -pitch, pitch, -2*pitch, 2*pitch])
        with self.assertRaises(ValueError):
            yaw_offsets("full_circle", 5.)

    def test_projected_ranking_uses_raw_depth(self):
        class Surface:
            def render(self, *args, **kwargs):
                return {"roi": (0, 0, 8, 8), "depth": np.ones((8, 8)), "clipped": False}
        calibration = {"world_T_camera": np.eye(4).tolist(), "intrinsics": np.eye(3).tolist()}
        filtered = {"camera": {"depth_m": np.full((8, 8), np.nan), "known_static_depth_m": np.full((8, 8), np.inf)}}
        raw = {"camera": {"calibration": calibration, "depth_m": np.ones((8, 8))}}
        good, good_check = _forward_score(Surface(), np.eye(4), raw, filtered)
        raw["camera"]["depth_m"][:] = 1.01
        bad, bad_check = _forward_score(Surface(), np.eye(4), raw, filtered)
        self.assertEqual(good, 0.)
        self.assertGreater(bad, good)
        self.assertTrue(good_check["passed"])
        self.assertFalse(bad_check["passed"])

    def test_unknown_packet_field_rejected_before_input_read(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"packet.json"
            path.write_text(json.dumps({"mode": "held", "target_pose": np.eye(4).tolist()}))
            with self.assertRaises(ValueError):
                estimate_packet(path)

    def test_source_hash_tracks_fresh_content(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"source.py"
            path.write_text("first version\n")
            before = digest(path)
            path.write_text("second version\n")
            self.assertNotEqual(digest(path), before)


if __name__ == "__main__":
    unittest.main()
