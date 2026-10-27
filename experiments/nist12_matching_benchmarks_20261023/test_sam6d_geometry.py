"""Focused controls for the isolated SAM-6D ISM geometry adaptation."""

import ast
import json
from pathlib import Path
import types
import unittest

import numpy as np

from sam6d_geometry import bbox_iou_xyxy, compute_geometry

try:
    import torch
except ImportError:
    torch = None


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[10, 0, 2], [0, 10, 2], [0, 0, 1]], np.float32)
        self.depth = np.ones((5, 5), np.float32)
        self.mask = np.zeros((5, 5), bool)
        self.mask[1:4, 1:4] = True
        self.corners = np.array([[-.1, -.1, 0], [.1, -.1, 0],
                                 [-.1, .1, 0], [.1, .1, 0]], np.float32)

    def test_exact_native_scale_and_integer_pixel_backprojection(self):
        got = compute_geometry(self.corners, np.eye(3), self.mask, self.depth, self.K)
        self.assertEqual(got["status"], "available")
        np.testing.assert_allclose(got["translation_camera_m"], [0, 0, 1], atol=1e-7)
        self.assertEqual(got["predicted_bbox_xyxy"], [1, 1, 3, 3])
        self.assertEqual(got["observed_bbox_xyxy"], [1, 1, 3, 3])
        self.assertEqual(got["geometry_bbox_iou"], 1)

    def test_template_rotation_changes_projected_width_and_height(self):
        points = self.corners * [2, 1, 1]
        r90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        got = compute_geometry(points, r90, self.mask, self.depth, self.K)
        self.assertEqual(got["predicted_bbox_xyxy"], [1, 0, 3, 4])
        self.assertEqual(got["geometry_bbox_iou"], .5)

    def test_depth_translation_uses_all_masked_positive_depth(self):
        depth = np.zeros((5, 5), np.float32)
        mask = np.zeros((5, 5), bool)
        mask[1, 1] = mask[3, 3] = True
        depth[1, 1], depth[3, 3] = 1, 2
        got = compute_geometry(self.corners, np.eye(3), mask, depth, self.K)
        np.testing.assert_allclose(got["translation_camera_m"], [.05, .05, 1.5], atol=1e-7)

    def test_no_cad_recentering_or_surface_centroid_correction(self):
        got = compute_geometry(self.corners + [.1, 0, 0], np.eye(3),
                               self.mask, self.depth, self.K)
        self.assertEqual(got["predicted_bbox_xyxy"], [2, 1, 4, 3])
        self.assertAlmostEqual(got["geometry_bbox_iou"], 1 / 3, places=6)

    def test_truncates_before_clamping_and_keeps_full_image_bbox(self):
        points = np.array([[-.6, -.6, 0], [.27, .27, 0]], np.float32)
        got = compute_geometry(points, np.eye(3), self.mask, self.depth, self.K)
        self.assertEqual(got["predicted_bbox_xyxy"], [0, 0, 4, 4])
        self.assertEqual(got["geometry_bbox_iou"], .25)

    def test_zeros_negative_and_nonfinite_depth_excluded(self):
        self.depth[1, 1] = 0
        self.depth[1, 2] = -1
        self.depth[1, 3] = np.nan
        got = compute_geometry(self.corners, np.eye(3), self.mask, self.depth, self.K)
        self.assertEqual(got["observed_valid_depth_pixels"], 6)
        self.assertTrue(np.isfinite(got["translation_camera_m"]).all())
        self.assertEqual(got["observed_bbox_xyxy"], [1, 1, 3, 3])

    def test_unavailable_observations_never_receive_zero_score(self):
        for mask, depth, reason in [
            (np.zeros((5, 5), bool), self.depth, "empty_observed_mask"),
            (self.mask, np.zeros((5, 5)), "no_positive_finite_observed_depth"),
        ]:
            got = compute_geometry(self.corners, np.eye(3), mask, depth, self.K)
            self.assertEqual(got["status"], "unavailable")
            self.assertEqual(got["reason"], reason)
            self.assertIsNone(got["geometry_bbox_iou"])

    def test_nonpositive_projected_depth_is_explicitly_unavailable(self):
        got = compute_geometry(self.corners - [0, 0, 2], np.eye(3),
                               self.mask, self.depth, self.K)
        self.assertEqual(got["reason"], "nonpositive_or_nonfinite_projected_cad_depth")
        self.assertIsNone(got["geometry_bbox_iou"])

    def test_scale_and_reflections_are_not_rotations(self):
        for rotation in [np.eye(3) * 2, np.diag([-1, 1, 1])]:
            with self.assertRaises(ValueError):
                compute_geometry(self.corners, rotation, self.mask, self.depth, self.K)

    def test_iou_has_no_inclusive_pixel_area_and_no_overlap_is_zero(self):
        self.assertAlmostEqual(bbox_iou_xyxy([0, 0, 2, 2], [1, 0, 3, 2]), 1 / 3, places=6)
        self.assertEqual(bbox_iou_xyxy([0, 0, 2, 2], [2, 0, 4, 2]), 0)
        self.assertEqual(bbox_iou_xyxy([1, 1, 1, 1], [1, 1, 1, 1]), 0)

    @unittest.skipIf(torch is None, "Official-source tensor parity requires existing PyTorch.")
    def test_pinned_upstream_helper_projection_and_iou_parity(self):
        here = Path(__file__).resolve().parent
        repo = here.parents[1]
        audit = json.loads((here / "sam6d_source_audit.json").read_text(encoding="utf-8-sig"))
        source_root = repo / "outputs/cache/sam6d_matching_source"
        namespace = {"torch": torch}

        def load_function(source, name):
            tree = ast.parse(source)
            nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
            assert len(nodes) == 1
            module = ast.Module(body=nodes, type_ignores=[])
            exec(compile(ast.fix_missing_locations(module), "<pinned-official-SAM6D>", "exec"), namespace)
            return namespace[name]

        load_function(audit["remote_helper"]["raw_source"], "depth_image_to_pointcloud_translate_torch")
        detector = (source_root / "model/detector.py").read_text()
        project = load_function(detector, "project_template_to_image")
        translate = load_function(detector, "Calculate_the_query_translation")
        upstream_iou = load_function((source_root / "utils/bbox_utils.py").read_text(), "compute_iou")
        generator = np.random.default_rng(2306)
        points = generator.uniform(-.09, .09, (2048, 3)).astype(np.float32)
        angle = .73
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]], np.float32)
        rotations = np.stack([np.eye(3), rotation]).astype(np.float32)
        poses = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
        poses[:, :3, :3] = rotations
        k = np.array([[215, 0, 83], [0, 217, 71], [0, 0, 1]], np.float32)
        depth = generator.uniform(.6, .8, (151, 173)).astype(np.float32)
        masks = np.zeros((2, 151, 173), np.float32)
        masks[0, 31:107, 22:141] = 1
        masks[1, 43:129, 53:163] = 1
        fake = types.SimpleNamespace(ref_data={"poses": torch.from_numpy(poses),
                                               "pointcloud": torch.from_numpy(points[None])})
        fake.Calculate_the_query_translation = types.MethodType(translate, fake)
        batch = {"depth": torch.from_numpy(depth[None]), "cam_intrinsic": torch.from_numpy(k[None]),
                 "depth_scale": 1000.0}
        projected = project(fake, torch.tensor([0, 1]), torch.tensor([0, 0]), batch,
                            torch.from_numpy(masks.copy()))
        official_translation = fake.Calculate_the_query_translation(torch.from_numpy(masks.copy()),
                                                                    batch["depth"][0],
                                                                    batch["cam_intrinsic"][0], 1000.0)
        for index in range(2):
            got = compute_geometry(points, rotations[index], masks[index], depth, k)
            official_box = torch.cat([projected[index].amin(dim=0), projected[index].amax(dim=0)])
            self.assertEqual(got["predicted_bbox_xyxy"], official_box.tolist())
            np.testing.assert_allclose(got["translation_camera_m"], official_translation[index], atol=2e-7)
            score = upstream_iou(official_box[None], torch.tensor([got["observed_bbox_xyxy"]]))
            self.assertAlmostEqual(got["geometry_bbox_iou"], float(score), places=7)


if __name__ == "__main__":
    unittest.main()
