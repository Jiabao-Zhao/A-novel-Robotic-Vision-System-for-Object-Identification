"""Independent arithmetic and frozen-source checks of the supplied-pose inputs."""
import unittest

import numpy as np

from . import inputs


class InputProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = inputs.read(inputs.MANIFEST)
        cls.rows = cls.manifest["records"]

    def test_exact_eight_by_twelve_inventory(self):
        self.assertEqual(len(self.rows), 96)
        self.assertEqual({row["scene"] for row in self.rows}, {f"N{i:02d}" for i in range(8)})
        parts = set(self.manifest["parts"])
        self.assertEqual(len(parts), 12)
        for scene in self.manifest["scenes"]:
            rows = [row for row in self.rows if row["scene"] == scene]
            self.assertEqual({row["part"] for row in rows}, parts)
            self.assertEqual(len({row["candidate_id"] for row in rows}), 12)
        self.assertNotIn("Gear_Medium", parts)
        self.assertNotIn("Gear_Small", parts)
        self.assertTrue(all(row["associations_assumed"] for row in self.rows))
        self.assertTrue(all(row["input_status"] == "available" for row in self.rows))
        self.assertFalse(self.manifest["simulator_object_poses_read"])

    def test_world_poses_are_camera_compositions_of_observed_fits(self):
        for row in self.rows:
            camera = np.load(inputs.local(row["world_T_camera_source"]))
            fit = inputs.read(inputs.local(row["pose_source"]))
            arm = next(arm for arm in fit["geometry"]["arms"] if arm["arm_name"] == "contact_0mm")
            expected = camera @ np.asarray(arm["best_geometry"]["T_camera_from_cad"])
            actual = np.asarray(row["world_T_cad"])
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
            np.testing.assert_allclose(actual[:3, :3].T @ actual[:3, :3], np.eye(3), atol=1e-8)
            self.assertAlmostEqual(np.linalg.det(actual[:3, :3]), 1., places=8)
            self.assertFalse(row["simulator_object_poses_read"])

    def test_table_is_the_observed_plane_transformed_to_world(self):
        captures = inputs.read(inputs.STUDY / "data/capture_manifest.json")["states"]
        for row in self.rows:
            camera = np.load(inputs.local(row["world_T_camera_source"]))
            source = np.asarray(captures[row["scene"]]["plane_model_camera"])
            expected = np.linalg.inv(camera).T @ source
            expected /= np.linalg.norm(expected[:3])
            if expected[2] < 0:
                expected *= -1
            np.testing.assert_allclose(row["table_plane_world"], expected, atol=1e-12, rtol=0)
            self.assertAlmostEqual(row["table_z_m"], -expected[3] / expected[2], places=12)

    def test_frozen_sources_allow_only_recorded_path_migrations(self):
        migration = inputs.read(inputs.REPO / "outputs/cleanup_20261027/path_migration.json")
        updated = {row["file"]: row for row in migration["metadata_updates"]}
        moved_associations = "experiments/nist12_direct_cad_vlm_20261019/audit_labels.json"
        path_only_metadata = {
            "experiments/nist12_qwen_clip_reranking_20261023/data/capture_manifest.json",
            "experiments/nist12_qwen_clip_reranking_20261023/data/cad_inputs.json",
        }
        for path, digest in self.manifest["source_sha256"].items():
            source = inputs.ASSOCIATIONS if path == moved_associations else inputs.local(path)
            actual = inputs.sha256(source)
            if path in path_only_metadata:
                self.assertEqual(updated[path]["before_sha256"], digest, path)
                self.assertEqual(updated[path]["after_sha256"], actual, path)
            else:
                self.assertEqual(actual, digest, path)
        for row in self.rows:
            self.assertEqual(inputs.sha256(inputs.local(row["mesh_path"])), row["mesh_sha256"], row["part"])


if __name__ == "__main__":
    unittest.main()
