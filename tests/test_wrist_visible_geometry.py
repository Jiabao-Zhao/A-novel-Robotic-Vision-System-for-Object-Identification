import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import open3d as o3d

import cad_object_association as association
from CADPointCloudRegistration import CADPointCloudRegistration
import scripts.wrist_visible_geometry as visible
from scripts.run_wrist_geometry_ablation import collapse_views
from scripts.wrist_geometry_ablation_report import distribution


class VisibleSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / "offset_box.ply"
        mesh = o3d.geometry.TriangleMesh.create_box(.08, .04, .02).translate([.1, -.2, .3])
        o3d.io.write_triangle_mesh(str(self.path), mesh)
        self.color = [.85, .025, .025]

    def test_same_rgb_rays_return_only_visible_faces_in_original_metric_frame(self):
        surfaces = list(visible.visible_surfaces(self.path, self.color))
        self.assertEqual(len(surfaces), 14)
        np.testing.assert_array_equal([s["rgb"] for s in surfaces], association.render_cad_views(self.path, self.color))
        for index, axis, expected in ((0, 0, .18), (3, 0, .1), (2, 2, .32), (5, 2, .3)):
            points = surfaces[index]["surface_points_m"]
            np.testing.assert_allclose(points[:, axis], expected, atol=2e-8, rtol=0)
            self.assertTrue(np.all(points >= np.array([.1, -.2, .3]) - 2e-8))
            self.assertTrue(np.all(points <= np.array([.18, -.16, .32]) + 2e-8))
        expected_depth = 3 * np.linalg.norm([.04, .02, .01]) - .04
        depth = surfaces[0]["ray_depth_m"]
        np.testing.assert_allclose(depth[np.isfinite(depth)], expected_depth, atol=2e-8, rtol=0)
        np.testing.assert_allclose(surfaces[0]["center_m"], [.14, -.18, .31])

    def test_cache_preserves_the_actual_five_mm_cloud_and_fpfh_arrays(self):
        templates = association.render_cad_views(self.path, self.color)
        registrar = CADPointCloudRegistration()
        with patch.object(visible, "CACHE_ROOT", self.root / "cache"):
            fresh, metadata, _ = visible.prepare_visible_geometry(self.path, self.color, templates, registrar)
            cached, cached_metadata, _ = visible.prepare_visible_geometry(self.path, self.color, templates, registrar)
        self.assertTrue(all(not m["cache_hit"] for m in metadata))
        self.assertTrue(all(m["cache_hit"] for m in cached_metadata))
        self.assertEqual(registrar.config.voxel_size_m, .005)
        for (a, af), (b, bf) in zip(fresh, cached):
            np.testing.assert_array_equal(a.points, b.points)
            np.testing.assert_array_equal(af.data, bf.data)
            self.assertEqual(af.data.shape, (33, len(a.points)))


class CoverageTests(unittest.TestCase):
    def test_bidirectional_coverage_uses_actual_sets_and_explicit_transform(self):
        cad = o3d.geometry.PointCloud(o3d.utility.Vector3dVector([[0, 0, 0], [.01, 0, 0], [.02, 0, 0], [.03, 0, 0]]))
        observed = o3d.geometry.PointCloud(o3d.utility.Vector3dVector([[.1, 0, 1], [.11, 0, 1]]))
        transform = np.eye(4)
        transform[:3, 3] = [.1, 0, 1]
        raw = {"T_observed_from_cad": transform.tolist(), "max_correspondence_distance_m": .0075,
               "registration_fitness": 1., "correspondence_count": 2}
        result = visible.coverage_diagnostics(cad, observed, raw)
        self.assertEqual(result["C_obs"], 1.)
        self.assertEqual(result["C_cad"], .5)
        self.assertAlmostEqual(result["C_f1"], 2 / 3)
        self.assertEqual(result["cad_correspondence_count"], 2)
        np.testing.assert_array_equal(np.asarray(cad.points)[0], [0, 0, 0])

    def test_failed_alignment_has_no_fabricated_reverse_coverage(self):
        result = visible.coverage_diagnostics(None, None, {"T_observed_from_cad": None, "registration_fitness": 0.})
        self.assertEqual(result["C_obs"], 0.)
        self.assertIsNone(result["C_cad"])
        self.assertIsNone(result["C_f1"])


class ViewConsistentFusionTests(unittest.TestCase):
    def test_fusion_keeps_the_shared_view_instead_of_combining_independent_maxima(self):
        rows = []
        for index, (normalized, fitness) in enumerate(((.9, .1), (.5, .95), (.8, .9)), 1):
            rows.append({"cad_id": "cad", "object_id": "object", "view_index_1based": index,
                         "cls_cosine": 2 * normalized - 1, "normalized_cls": normalized,
                         "geometry_fitness": fitness, "fused_score": .5 * normalized + .5 * fitness,
                         "geometry_raw": {"registration_fitness": fitness}, "C_obs": fitness,
                         "C_cad": 1. if index == 1 else 0., "C_f1": 1. if index == 1 else 0., "coverage_status": "aligned"})
        result = collapse_views(rows)
        self.assertEqual(result["visual_winning_view_index_1based"], 1)
        self.assertEqual(result["geometry_winning_view_index_1based"], 2)
        self.assertEqual(result["winning_view_index_1based"], 3)
        self.assertAlmostEqual(result["fused_score"], .85)
        self.assertNotEqual(result["fused_score"], .5 * result["visual_score"] + .5 * result["geometry_score"])
        self.assertEqual(result["geometry_fitness_at_fused_view"], .9)
        for row in rows:
            row.update(C_cad=0., C_f1=0.)
        self.assertEqual(collapse_views(rows)["winning_view_index_1based"], 3)

    def test_coverage_distribution_counts_missing_alignment_without_imputation(self):
        result = distribution([0., 1., None])
        self.assertEqual((result["count"], result["missing"]), (2, 1))
        self.assertEqual(result["mean"], .5)
        self.assertIsNone(distribution([None])["mean"])


if __name__ == "__main__":
    unittest.main()
