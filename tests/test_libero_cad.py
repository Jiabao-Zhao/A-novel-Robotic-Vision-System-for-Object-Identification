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
ASSET_REVISION = "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"
CATALOG_CASES = (
    (
        "alphabet soup can",
        "libero_object_alphabet_soup",
        "stable_hope_objects/alphabet_soup/textured.obj",
        0.01,
        "Y",
        "9f83394a8b6d243b2133be244b3de83e9318401e53a5768063031795bc6919a0",
    ),
    (
        "cream cheese box",
        "libero_object_cream_cheese",
        "stable_hope_objects/cream_cheese/cream_cheese.obj",
        0.008,
        "Z",
        "79ca603fd960a95643cf7532b702343c1b389af5cd86b84dfeb931a2cc23a4b3",
    ),
    (
        "salad dressing bottle",
        "libero_object_salad_dressing",
        "stable_hope_objects/salad_dressing/textured.obj",
        0.01,
        "Y",
        "0045f55b86068e26888bcff4ea8a0eaebe2f3ec791349cae0d593066d68b9ba7",
    ),
    (
        "barbecue sauce bottle",
        "libero_object_bbq_sauce",
        "stable_hope_objects/bbq_sauce/bbq_sauce.obj",
        0.0077,
        "Z",
        "69030e437ba6d8970e925f1b5344b80d2a67511658c2625dfbeb508f53e34417",
    ),
    (
        "red ketchup bottle",
        "libero_object_ketchup",
        "stable_hope_objects/ketchup/textured.obj",
        0.01,
        "Y",
        "dd07788b2fc0ece118cd97d544cea0fa6e87f56440f69d9bc13b876143490414",
    ),
    (
        "can of tomato sauce",
        "libero_object_tomato_sauce",
        "stable_hope_objects/tomato_sauce/textured.obj",
        0.01,
        "Y",
        "b75cd4063af13da2c3c95f4ed6cc2fbdc00f51730b49b704feb6f1ef362204bf",
    ),
    (
        "butter box",
        "libero_object_butter",
        "stable_hope_objects/butter/butter.obj",
        0.0075,
        "Z",
        "4d32a23384f059ee79cc6c9d4d18d2d12c171c13bd89aaa4829a332123a1bd4a",
    ),
    (
        "red milk carton",
        "libero_object_milk",
        "stable_hope_objects/milk/textured.obj",
        0.0075,
        "Y",
        "caff7624f6aa1183166344350f3491587c53ed42ca7a57058b6e52796cf42e1f",
    ),
    (
        "chocolate pudding package",
        "libero_object_chocolate_pudding",
        "stable_hope_objects/chocolate_pudding/textured.obj",
        0.01,
        "Z",
        "08ea9b9106f9595db3b8544f0c1c4b55e3b62303191f2cf8a75aa677abc8b237",
    ),
    (
        "carton of orange juice",
        "libero_object_orange_juice",
        "stable_hope_objects/orange_juice/textured.obj",
        0.0075,
        "Y",
        "8f2296913bd1a82160fa1291ddb72f4eeb3703437513481c8b840d1933c460f1",
    ),
)


def write_test_catalog(root):
    records = json.loads(LIBERO_CAD_LIBRARY_PATH.read_text(encoding="utf-8"))
    for record in records:
        record["asset_sha256"] = EMPTY_FILE_SHA256
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
    def test_catalog_declares_all_ten_verified_product_meshes(self):
        records = json.loads(LIBERO_CAD_LIBRARY_PATH.read_text(encoding="utf-8"))
        records_by_id = {record["cad_id"]: record for record in records}

        self.assertEqual(len(records), 10)
        self.assertEqual(len(records_by_id), 10)
        for _, cad_id, asset_path, scale_to_m, up_axis, asset_sha256 in CATALOG_CASES:
            with self.subTest(cad_id=cad_id):
                record = records_by_id[cad_id]
                self.assertEqual(record["asset_relative_path"], asset_path)
                self.assertEqual(record["scale_to_m"], scale_to_m)
                self.assertEqual(record["cad_up_axis"], up_axis)
                self.assertEqual(record["asset_revision"], ASSET_REVISION)
                self.assertEqual(record["asset_sha256"], asset_sha256)
                self.assertEqual(record["asset_repository"], "lerobot/libero-assets")
                self.assertEqual(record["license"], "CC BY-NC-SA 4.0")
                self.assertFalse(record["redistributed_in_this_repository"])
                self.assertTrue(record["exact_rendering_cad_prior_used"])

    def test_retrieve_all_product_cads_resolves_aliases_scales_and_axes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            assets_root = Path(temporary_directory)
            catalog_path = write_test_catalog(assets_root)
            for _, _, asset_path, _, _, _ in CATALOG_CASES:
                full_path = assets_root / asset_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.touch()

            for query, cad_id, asset_path, scale_to_m, up_axis, _ in CATALOG_CASES:
                with self.subTest(query=query):
                    record = retrieve_libero_cad(
                        query,
                        catalog_path=catalog_path,
                        assets_root=assets_root,
                    )
                    self.assertEqual(record["cad_id"], cad_id)
                    self.assertEqual(record["cad_path"], str(assets_root / asset_path))
                    self.assertEqual(record["scale_to_m"], scale_to_m)
                    self.assertEqual(record["cad_up_axis"], up_axis)
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
