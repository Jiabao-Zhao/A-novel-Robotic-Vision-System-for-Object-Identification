import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d

from CADPointCloudRegistration import CADPointCloudRegistration
from simulation.libero_cad import (
    LIBERO_CAD_LIBRARY_PATH,
    register_libero_cad_to_observation,
    retrieve_libero_cad,
)


EMPTY_FILE_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def write_test_catalog(root):
    records = json.loads(LIBERO_CAD_LIBRARY_PATH.read_text(encoding="utf-8"))
    records[0]["asset_sha256"] = EMPTY_FILE_SHA256
    catalog_path = Path(root) / "libero_object_library.json"
    catalog_path.write_text(json.dumps(records), encoding="utf-8")
    return catalog_path


class FakeRegistrar:
    CAMERA_T_CAD = np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    def __init__(self, rmse_m=0.004, warnings=None):
        self.arguments = None
        self.rmse_m = rmse_m
        self.warnings = [] if warnings is None else warnings

    def run(self, **arguments):
        self.arguments = arguments
        output_dir = Path(arguments["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "T_observed_from_cad": self.CAMERA_T_CAD.tolist(),
            "cad_center_m": [1.0, 2.0, 3.0],
            "constrained_rmse_m": self.rmse_m,
            "warnings": self.warnings,
            "result_path": str(output_dir / "cad_registration_result.json"),
            "aligned_cad_cloud_path": str(output_dir / "cad_aligned_cloud.ply"),
            "augmented_cloud_path": str(output_dir / "augmented_object_cloud.ply"),
        }


class LiberoCADTests(unittest.TestCase):
    def test_retrieve_milk_cad_uses_declared_scale(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            assets_root = Path(temporary_directory)
            milk_path = assets_root / "stable_hope_objects" / "milk" / "textured.obj"
            milk_path.parent.mkdir(parents=True)
            milk_path.touch()
            catalog_path = write_test_catalog(assets_root)

            record = retrieve_libero_cad(
                "red milk carton",
                catalog_path=catalog_path,
                assets_root=assets_root,
            )

        self.assertEqual(record["cad_id"], "libero_object_milk")
        self.assertEqual(record["cad_path"], str(milk_path))
        self.assertEqual(record["scale_to_m"], 0.0075)
        self.assertEqual(record["cad_up_axis"], "Y")
        self.assertEqual(record["asset_repository"], "lerobot/libero-assets")
        self.assertTrue(record["exact_rendering_cad_prior_used"])

    def test_registration_composes_camera_and_world_transforms(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            assets_root = root / "assets"
            milk_path = assets_root / "stable_hope_objects" / "milk" / "textured.obj"
            milk_path.parent.mkdir(parents=True)
            milk_path.touch()
            observed_path = root / "object_003.ply"
            observed_path.touch()
            output_dir = root / "registration"
            catalog_path = write_test_catalog(root)
            localization = {
                "frame": "camera",
                "plane_model": [0.0, 0.0, 1.0, 0.0],
                "objects": [
                    {
                        "object_id": "object_003",
                        "pointcloud_path": str(observed_path),
                    }
                ],
            }
            world_T_camera = np.array(
                [
                    [1.0, 0.0, 0.0, 10.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            registrar = FakeRegistrar()

            result = register_libero_cad_to_observation(
                object_type="milk carton",
                object_id="object_003",
                localization=localization,
                world_T_camera=world_T_camera,
                output_dir=output_dir,
                catalog_path=catalog_path,
                assets_root=assets_root,
                registrar=registrar,
            )

        np.testing.assert_allclose(
            result["registered_center_camera_m"],
            [-2.0, 21.0, 3.0],
        )
        np.testing.assert_allclose(
            result["registered_center_world_m"],
            [8.0, -3.0, 21.0],
        )
        np.testing.assert_allclose(result["cad_center_cad_m"], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(
            result["world_T_cad"],
            world_T_camera @ FakeRegistrar.CAMERA_T_CAD,
        )
        np.testing.assert_allclose(
            np.asarray(result["world_T_cad"])[:3, 3],
            [10.0, 0.0, 20.0],
        )
        self.assertEqual(registrar.arguments["plane_model"], localization["plane_model"])
        self.assertEqual(registrar.arguments["cad_metadata"]["scale_to_m"], 0.0075)
        self.assertEqual(registrar.arguments["cad_metadata"]["cad_up_axis"], "Y")
        self.assertFalse(result["simulator_object_identity_or_pose_used"])
        self.assertTrue(result["registration_accepted"])

    def test_explicit_scale_overrides_automatic_unit_heuristic(self):
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]])

        CADPointCloudRegistration().normalize_cad_units(cloud, scale_to_m=0.0075)

        extent = np.asarray(cloud.get_axis_aligned_bounding_box().get_extent())
        np.testing.assert_allclose(extent, [0.015, 0.03, 0.045])

    def test_invalid_explicit_scale_is_rejected(self):
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector([[0.0, 0.0, 0.0]])

        for scale in (0.0, -0.5, np.nan, np.inf):
            with self.subTest(scale=scale):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    CADPointCloudRegistration().normalize_cad_units(
                        cloud,
                        scale_to_m=scale,
                    )

    def test_yaw_delta_rotates_about_object_pivot(self):
        registrar = CADPointCloudRegistration()
        pivot = np.array([0.4, -0.2, 1.1])
        rotation = registrar.rotation_about_axis([0.0, 0.0, 1.0], np.pi / 2.0)
        delta = registrar.rotation_about_point_transform(rotation, pivot)

        transformed_pivot = (delta @ np.append(pivot, 1.0))[:3]
        transformed_offset = (delta @ np.array([1.4, -0.2, 1.1, 1.0]))[:3]

        np.testing.assert_allclose(transformed_pivot, pivot, atol=1e-12)
        np.testing.assert_allclose(
            transformed_offset,
            [0.4, 0.8, 1.1],
            atol=1e-12,
        )

    def test_registration_requires_camera_frame_and_valid_plane(self):
        base = {
            "frame": "world",
            "plane_model": [0.0, 0.0, 1.0, 0.0],
            "objects": [],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaisesRegex(ValueError, "camera frame"):
                register_libero_cad_to_observation(
                    "milk",
                    "object_003",
                    base,
                    np.eye(4),
                    root,
                )
            base["frame"] = "camera"
            base["plane_model"] = [0.0, np.nan, 1.0, 0.0]
            with self.assertRaisesRegex(ValueError, "plane_model"):
                register_libero_cad_to_observation(
                    "milk",
                    "object_003",
                    base,
                    np.eye(4),
                    root,
                )

    def test_missing_localized_object_is_rejected_before_registration(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaisesRegex(LookupError, "object_003.*found 0"):
                register_libero_cad_to_observation(
                    object_type="milk",
                    object_id="object_003",
                    localization={
                        "frame": "camera",
                        "plane_model": [0.0, 0.0, 1.0, 0.0],
                        "objects": [],
                    },
                    world_T_camera=np.eye(4),
                    output_dir=root / "registration",
                    assets_root=root / "assets",
                    registrar=FakeRegistrar(),
                )

    def test_duplicate_localized_object_is_rejected(self):
        localization = {
            "frame": "camera",
            "plane_model": [0.0, 0.0, 1.0, 0.0],
            "objects": [
                {"object_id": "object_003"},
                {"object_id": "object_003"},
            ],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(LookupError, "object_003.*found 2"):
                register_libero_cad_to_observation(
                    object_type="milk",
                    object_id="object_003",
                    localization=localization,
                    world_T_camera=np.eye(4),
                    output_dir=Path(temporary_directory),
                )

    def test_registration_above_rmse_threshold_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            assets_root = root / "assets"
            milk_path = assets_root / "stable_hope_objects" / "milk" / "textured.obj"
            milk_path.parent.mkdir(parents=True)
            milk_path.touch()
            observed_path = root / "object_003.ply"
            observed_path.touch()
            catalog_path = write_test_catalog(root)
            localization = {
                "frame": "camera",
                "plane_model": [0.0, 0.0, 1.0, 0.0],
                "objects": [
                    {
                        "object_id": "object_003",
                        "pointcloud_path": str(observed_path),
                    }
                ],
            }

            with self.assertRaisesRegex(RuntimeError, "rejected.*RMSE"):
                register_libero_cad_to_observation(
                    object_type="milk",
                    object_id="object_003",
                    localization=localization,
                    world_T_camera=np.eye(4),
                    output_dir=root / "registration",
                    catalog_path=catalog_path,
                    assets_root=assets_root,
                    registrar=FakeRegistrar(rmse_m=0.02),
                )


if __name__ == "__main__":
    unittest.main()
