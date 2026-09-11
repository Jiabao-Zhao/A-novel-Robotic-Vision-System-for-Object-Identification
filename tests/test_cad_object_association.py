import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
import open3d as o3d

import cad_object_association as association
import main
from CADPointCloudRegistration import CADPointCloudRegistration
from helper_function import CADModel


def scored(object_id, visual, geometry, status="matched"):
    return {"object_id": object_id, "visual_raw": 2 * visual - 1,
            "visual_score": visual, "geometry_score": geometry,
            "geometry_raw": {"status": status, "registration_fitness": geometry},
            "fused_score": association.fuse_scores(visual, geometry)}


class ScoreTests(unittest.TestCase):
    def test_cosine_uses_best_view_and_retains_negative_values(self):
        self.assertEqual(association.cosine_similarity([2, 0], [[0, 4], [3, 0]]), 1)
        self.assertEqual(association.cosine_similarity([2, 0], [[-4, 0]]), -1)
        with self.assertRaises(ValueError):
            association.cosine_similarity([0, 0], [[1, 0]])
        with self.assertRaises(ValueError):
            association.cosine_similarity([np.nan, 0], [[1, 0]])

    def test_visual_normalization(self):
        for raw, expected in [(-1, 0), (0, 0.5), (0.76, 0.88), (1, 1)]:
            self.assertAlmostEqual(association.normalize_visual_score(raw), expected)

    def test_geometry_normalization_preserves_bounded_fitness(self):
        for raw in (0, 0.35, 0.91, 1):
            self.assertEqual(association.normalize_geometry_score(raw), raw)
        for raw in (-0.1, 1.01, np.nan, np.inf):
            with self.assertRaises(ValueError):
                association.normalize_geometry_score(raw)

    def test_fixed_fusion_rules(self):
        self.assertAlmostEqual(association.fuse_scores(0.88, 0.91), 0.895)
        self.assertAlmostEqual(association.fuse_scores(0.81, 0.49, "geometric_mean"), 0.63)
        self.assertEqual(association.fuse_scores(1, 0, "geometric_mean"), 0)
        with self.assertRaises(ValueError):
            association.fuse_scores(1, 0, "learned")

    def test_ranking_and_automatic_match(self):
        ranking, selected, resolution, _ = association.rank_and_resolve([
            scored("object_001", 0.65, 0.5), scored("object_003", 0.88, 0.91),
        ])
        self.assertEqual([r["object_id"] for r in ranking], ["object_003", "object_001"])
        self.assertEqual((selected, resolution), ("object_003", "automatic_match"))

    def test_ties_are_deterministic_but_ambiguous(self):
        ranking, selected, resolution, reason = association.rank_and_resolve([
            scored("b", 0.9, 0.9), scored("a", 0.9, 0.9),
        ])
        self.assertEqual(ranking[0]["object_id"], "a")
        self.assertEqual((selected, resolution, reason), (None, "ambiguous", "small_fused_margin"))

    def test_strong_cross_candidate_disagreement(self):
        _, selected, resolution, reason = association.rank_and_resolve([
            scored("a", 0.95, 0.45), scored("b", 0.70, 0.99),
        ])
        self.assertEqual((selected, resolution), (None, "ambiguous"))
        self.assertEqual(reason, "modalities_prefer_different_objects")

    def test_single_candidate_disagreement_and_weak_evidence(self):
        for visual, geometry in [(0.95, 0.25), (0.61, 0.4)]:
            self.assertEqual(association.rank_and_resolve([scored("a", visual, geometry)])[2], "ambiguous")

    def test_all_registration_failures_are_ambiguous_not_absence(self):
        self.assertEqual(association.rank_and_resolve([
            scored("a", 0.2, 0, "failed"), scored("b", 0.1, 0, "failed"),
        ])[2:], ("ambiguous", "geometric_matching_failed_for_all"))

    def test_invalid_candidate_is_retained_and_prevents_forced_match(self):
        invalid = scored("b", 0.5, 0)
        invalid.update(geometry_score=None, fused_score=None)
        ranking, selected, resolution, _ = association.rank_and_resolve([scored("a", 0.95, 0.95), invalid])
        self.assertEqual(len(ranking), 2)
        self.assertEqual((selected, resolution), (None, "ambiguous"))

    def test_no_match_requires_weak_absolute_evidence(self):
        self.assertEqual(association.rank_and_resolve([
            scored("a", 0.3, 0.10), scored("b", 0.2, 0.05),
        ])[1:3], (None, "not_present"))
        self.assertEqual(association.rank_and_resolve([])[1:3], (None, "not_present"))


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.registrar = CADPointCloudRegistration()
        o3d.utility.random.seed(42)
        mesh = o3d.geometry.TriangleMesh.create_box(0.08, 0.04, 0.03)
        mesh += o3d.geometry.TriangleMesh.create_box(0.02, 0.02, 0.04).translate([0.06, 0.02, 0.03])
        mesh.compute_vertex_normals()
        self.cloud = mesh.sample_points_uniformly(3000)

    def test_fpfh_is_local_and_finite(self):
        down, feature = self.registrar.compute_fpfh(self.cloud)
        self.assertEqual(feature.data.shape, (33, len(down.points)))
        self.assertTrue(np.isfinite(feature.data).all())
        self.assertGreater(np.std(feature.data, axis=1).max(), 0)
        with self.assertRaises(ValueError):
            self.registrar.compute_fpfh(o3d.geometry.PointCloud())

    def test_real_ransac_icp_supports_partial_observation(self):
        cad, features = self.registrar.compute_fpfh(self.cloud)
        indices = np.flatnonzero(np.asarray(cad.points)[:, 0] > 0.03)
        partial = cad.select_by_index(indices)
        rotation = o3d.geometry.get_rotation_matrix_from_xyz([0.15, -0.1, 0.35])
        partial.rotate(rotation, center=(0, 0, 0)).translate([0.12, -0.06, 0.8])
        observed, observed_features = self.registrar.compute_fpfh(partial)
        result = self.registrar.match_fpfh(cad, features, observed, observed_features)
        self.assertEqual(result["status"], "matched")
        self.assertGreater(result["registration_fitness"], 0.8)
        self.assertLess(result["observed_to_cad_rmse_m"], 0.006)
        self.assertEqual(result["registration_fitness"], result["correspondence_count"] / len(observed.points))
        transform = np.asarray(result["T_observed_from_cad"])
        np.testing.assert_allclose(transform[3], [0, 0, 0, 1])

    def test_partial_surface_metric_does_not_penalize_unseen_cad(self):
        down, _ = self.registrar.compute_fpfh(self.cloud)
        partial = down.select_by_index(list(range(0, len(down.points), 3)))
        self.assertLess(self.registrar.observed_to_cad_rmse(down, partial, np.eye(4)), 0.004)
        shifted = np.eye(4)
        shifted[0, 3] = 0.2
        self.assertGreater(self.registrar.observed_to_cad_rmse(down, partial, shifted), 0.1)


class FeatureAndCacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        mesh = o3d.geometry.TriangleMesh.create_box(0.08, 0.04, 0.03)
        mesh.compute_vertex_normals()
        path = self.root / "box.ply"
        o3d.io.write_triangle_mesh(str(path), mesh)
        self.cad = CADModel("CAD_007", "box", str(path), "box", "component")
        self.registrar = CADPointCloudRegistration()
        self.rgb_path = self.root / "RGB.png"
        rgb = np.zeros((40, 80, 3), dtype=np.uint8)
        rgb[:, :40] = [20, 30, 220]  # BGR on disk; crop encoder must see RGB.
        cv2.imwrite(str(self.rgb_path), rgb)
        self.payload = {"objects": []}
        for index in range(2):
            cloud_path = self.root / f"object_{index}.ply"
            o3d.io.write_point_cloud(str(cloud_path), mesh.sample_points_uniformly(1000))
            self.payload["objects"].append({"object_id": f"object_{index:03d}",
                "roi": {"x1": index * 40, "y1": 0, "x2": index * 40 + 39, "y2": 39},
                "pointcloud_path": str(cloud_path)})

    def test_cpu_rendered_views(self):
        views = association.render_cad_views(self.cad.file_path)
        self.assertEqual(len(views), 14)
        self.assertTrue(all(v.shape == (224, 224, 3) and v.dtype == np.uint8 for v in views))
        self.assertTrue(all(np.any(v < 255) and np.any(v == 255) for v in views))

    def test_encoder_is_frozen_eval_and_uses_inference_mode_on_cpu(self):
        import torch

        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))

            def forward(self, batch):
                self.inference_mode = torch.is_inference_mode_enabled()
                return batch.mean(dim=(2, 3)) * self.weight

        encoder = Encoder()
        association.load_dino_encoder.cache_clear()
        self.addCleanup(association.load_dino_encoder.cache_clear)
        with patch("torch.hub.load", return_value=encoder) as load, patch.object(association, "DINO_DEVICE", "cpu"):
            features = association.extract_dino_features([np.full((20, 30, 3), 128, dtype=np.uint8)] * 2)
        self.assertFalse(encoder.training)
        self.assertTrue(encoder.inference_mode)
        self.assertFalse(encoder.weight.requires_grad)
        self.assertEqual(encoder.weight.device.type, "cpu")
        self.assertTrue(load.call_args.kwargs["pretrained"])
        np.testing.assert_array_equal(features[0], features[1])

    @patch.object(association, "extract_dino_features", side_effect=lambda images: np.ones((len(images), 384)))
    def test_cad_cache_hits_and_mesh_content_invalidation(self, encode):
        cache = self.root / "cache"
        first, initial = association.prepare_cad_features(self.cad, self.registrar, cache)
        with patch.object(self.registrar, "compute_fpfh", wraps=self.registrar.compute_fpfh) as fpfh:
            second, warm = association.prepare_cad_features(self.cad, self.registrar, cache)
            fpfh.assert_not_called()
        self.assertEqual(encode.call_count, 1)
        self.assertFalse(initial["cad_visual_cache_hit"])
        self.assertTrue(warm["cad_visual_cache_hit"] and warm["cad_geometry_cache_hit"])
        self.assertEqual(warm["cad_visual_preprocessing_time_s"], 0)
        self.assertEqual(warm["cad_geometry_preprocessing_time_s"], 0)
        np.testing.assert_array_equal(first["fpfh"].data, second["fpfh"].data)
        mesh = o3d.io.read_triangle_mesh(self.cad.file_path)
        mesh.scale(1.1, center=(0, 0, 0))
        o3d.io.write_triangle_mesh(self.cad.file_path, mesh)
        _, changed = association.prepare_cad_features(self.cad, self.registrar, cache)
        self.assertFalse(changed["cad_visual_cache_hit"] or changed["cad_geometry_cache_hit"])

    def test_all_candidates_both_modalities_and_scene_cache_reuse_across_targets(self):
        cache = {}
        seen_crops = []

        def encode(images):
            seen_crops.extend(images)
            if len(images) == 1 and not np.any(images[0]):
                return np.array([[-1.0, 0.0]])
            return np.tile([1.0, 0.0], (len(images), 1))

        with patch.object(association, "extract_dino_features", side_effect=encode), \
                patch.object(CADPointCloudRegistration, "match_fpfh", return_value={
                    "status": "matched", "registration_fitness": 0.8, "observed_to_cad_rmse_m": 0.002,
                }) as match:
            first = association.associate_cad_to_candidates(self.cad, self.payload, self.rgb_path,
                target_description="box", scene_cache=cache, cache_dir=self.root / "cache")
            other_cad = CADModel("CAD_008", "second", self.cad.file_path, "box", "component")
            second = association.associate_cad_to_candidates(other_cad, self.payload, self.rgb_path,
                target_description="second box", scene_cache=cache, cache_dir=self.root / "cache")
        self.assertEqual(match.call_count, 4)
        self.assertEqual(len(first["candidate_ranking"]), 2)
        self.assertEqual(first["candidate_ranking"][1]["visual_raw"], -1)
        self.assertEqual(first["candidate_ranking"][1]["geometry_score"], 0.8)
        self.assertEqual(len(seen_crops), 16)  # 14 CAD views + 2 original scene crops, once.
        np.testing.assert_array_equal(seen_crops[-2][0, 0], [220, 30, 20])
        self.assertEqual(second["runtime"]["scene_feature_cache_hits"], 2)
        self.assertEqual(second["runtime"]["workspace_dino_feature_time_s"], 0)
        self.assertEqual(second["runtime"]["workspace_fpfh_feature_time_s"], 0)
        self.assertEqual((second["target_description"], second["cad_id"]), ("second box", "CAD_008"))
        for key in ("cad_visual_preprocessing_time_s", "cad_geometry_preprocessing_time_s",
                    "workspace_dino_feature_time_s", "workspace_fpfh_feature_time_s",
                    "pairwise_matching_time_s", "fusion_time_s", "scene_association_time_s", "total_time_s"):
            self.assertGreaterEqual(first["runtime"][key], 0)
        self.assertEqual(len(first["runtime"]["per_candidate_matching_time_s"]), 2)
        self.assertGreaterEqual(first["runtime"]["total_time_s"], first["runtime"]["scene_association_time_s"])
        json.dumps(first, allow_nan=False)

    @patch.object(association, "extract_dino_features", return_value=np.ones((1, 384)))
    def test_invalid_crop_still_computes_geometry_and_changed_scene_invalidates(self, encode):
        cache = {}
        self.payload["objects"][0]["roi"] = {"x1": -20, "y1": 0, "x2": -5, "y2": 5}
        candidates, _ = association.prepare_scene_features(self.payload, self.rgb_path, self.registrar, cache)
        self.assertIsNone(candidates[0]["visual_feature"])
        self.assertIsNotNone(candidates[0]["fpfh"])
        self.assertEqual(encode.call_count, 1)
        cv2.imwrite(str(self.rgb_path), np.full((40, 80, 3), 255, dtype=np.uint8))
        _, timing = association.prepare_scene_features(self.payload, self.rgb_path, self.registrar, cache)
        self.assertEqual(timing["scene_feature_cache_hits"], 0)

    @patch.object(association, "extract_dino_features", side_effect=lambda images: np.ones((len(images), 384)))
    def test_insufficient_cloud_keeps_visual_evaluation_and_defers(self, encode):
        path = self.payload["objects"][1]["pointcloud_path"]
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector([[0, 0, 0], [0.01, 0, 0]]))
        o3d.io.write_point_cloud(path, cloud)
        with patch.object(CADPointCloudRegistration, "match_fpfh", return_value={
            "status": "matched", "registration_fitness": 0.95,
        }) as match:
            result = association.associate_cad_to_candidates(self.cad, self.payload, self.rgb_path,
                target_description="box", cache_dir=self.root / "cache")
        self.assertEqual(encode.call_count, 3)  # CAD views and both scene crops.
        self.assertEqual(match.call_count, 1)
        self.assertEqual(result["resolution"], "ambiguous")
        self.assertIsNone(result["selected_object_id"])
        self.assertEqual(len(result["candidate_ranking"]), 2)
        invalid = result["candidate_ranking"][1]
        self.assertEqual(invalid["visual_score"], 1)
        self.assertIsNone(invalid["geometry_score"])
        self.assertTrue(invalid["errors"])

    def test_raw_cloud_path_never_substitutes_downsampled_observation(self):
        self.assertEqual(association.raw_observed_cloud_path({"pointcloud_path": "a_downsampled.ply"}), Path("a.ply"))


class PipelineContractTests(unittest.TestCase):
    def test_retrieval_precedes_association_and_winner_uses_same_cad(self):
        events, scene_caches = [], []
        cad = CADModel("CAD_007", "round peg", "CAD/models/round_peg.stl", "", "")

        def retrieve(target):
            events.append(("retrieve", target))
            return cad, {"selected_cad_id": cad.cad_id, "selected_cad_name": cad.cad_name,
                         "selected_file_path": cad.file_path}

        def associate(model, payload, rgb, plane, **kwargs):
            events.append(("associate", kwargs["target_description"]))
            self.assertIs(model, cad)
            self.assertEqual(rgb, "original.png")
            scene_caches.append(kwargs["scene_cache"])
            return {"target_description": kwargs["target_description"], "cad_id": model.cad_id,
                    "cad_path": model.file_path, "selected_object_id": "object_003",
                    "final_object_id": "object_003", "resolution": "automatic_match"}

        payload = {"objects": [{"object_id": "object_003", "pointcloud_path": "partial_downsampled.ply"}]}
        registrar = Mock()
        registrar.run.return_value = {"aligned_cad_cloud_path": "aligned", "augmented_cloud_path": "augmented",
                                     "result_path": "registration.json"}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(main, "retrieve_cad_model", side_effect=retrieve) as retrieval, \
                patch.object(main, "associate_cad_to_candidates", side_effect=associate):
            results, retrieved, summary = main.associate_targets_with_cad(
                ["round peg", "peg"], payload, "original.png", output_dir=directory,
            )
            self.assertEqual(events[:2], [("retrieve", "round peg"), ("retrieve", "peg")])
            self.assertIs(scene_caches[0], scene_caches[1])
            records = main.register_resolved_associations(results[:1], payload, [0, 0, 1, 0],
                registrar=registrar, registration_root=Path(directory) / "registration", retrieved_cads=retrieved)
            self.assertEqual(retrieval.call_count, 2)
            self.assertEqual(registrar.run.call_args.kwargs["cad_path"], cad.file_path)
            self.assertEqual(registrar.run.call_args.kwargs["observed_cloud_path"], Path("partial.ply"))
            self.assertEqual(registrar.run.call_args.kwargs["cad_metadata"]["cad_id"], "CAD_007")
            self.assertEqual(records[0]["object_id"], "object_003")
            self.assertEqual(json.loads(summary.read_text())["associations"][0]["final_object_id"], "object_003")
            self.assertTrue((Path(directory) / "target_001.json").exists())

    def test_changed_cad_or_missing_fixed_handoff_cannot_trigger_registration(self):
        result = {"target_description": "peg", "cad_id": "correct", "cad_path": "correct.stl",
                  "final_object_id": "a", "selected_object_id": "a", "resolution": "automatic_match"}
        payload = {"objects": [{"object_id": "a", "pointcloud_path": "a.ply"}]}
        registrar = Mock()
        with tempfile.TemporaryDirectory() as directory:
            for retrieved in (None, {"peg": (SimpleNamespace(cad_id="wrong", file_path="wrong.stl"), {})}):
                with self.assertRaises(ValueError):
                    main.register_resolved_associations([result], payload, None, registrar=registrar,
                        registration_root=directory, retrieved_cads=retrieved)
        registrar.run.assert_not_called()

    def test_main_stops_downstream_for_ambiguous_or_absent_target(self):
        with tempfile.TemporaryDirectory() as directory:
            localization = Path(directory) / "localization.json"
            localization.write_text('{"objects": []}', encoding="utf-8")
            paths = dict.fromkeys(("rgb", "depth", "annotated_rgb", "workspace_cloud", "table_cloud", "segmented_cloud"), "unused")
            paths.update(localization=localization, object_count=0)
            for resolution in ("ambiguous", "not_present"):
                with patch.object(main, "run_pipeline", return_value=paths), \
                        patch.object(main, "associate_targets_with_cad", return_value=([{"resolution": resolution}], {}, "summary")), \
                        patch.object(main, "register_resolved_associations") as register, \
                        patch.object(main, "output_robot_base_pose") as robot, \
                        patch.object(main, "output_llm_plan") as planner:
                    with self.assertRaises(SystemExit):
                        main.main()
                    register.assert_not_called()
                    robot.assert_not_called()
                    planner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
