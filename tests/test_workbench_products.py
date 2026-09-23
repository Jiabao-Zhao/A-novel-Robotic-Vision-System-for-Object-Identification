"""Checks for the revised inspection catalog, independent of simulation startup."""
import unittest

import numpy as np

from simulation.industrial_workbench import NIST_PARTS, PARTS, prepare_catalog, preview_placements


class WorkbenchProductsTest(unittest.TestCase):
    def test_membership(self):
        self.assertEqual(len(PARTS), 13)
        self.assertEqual(len(NIST_PARTS), 10)
        self.assertFalse(set(PARTS) & {"spacer", "bearing", "pulley", "cable_shark_device", "BNC_Male", "USB_Male"})
        self.assertEqual(set(preview_placements()), set(PARTS))

    def test_metric_assets_and_glue_policy(self):
        import open3d as o3d
        from simulation.benchmark_products import PRODUCTS

        catalog = prepare_catalog()
        for name in PRODUCTS:
            item = catalog[name]
            original = o3d.io.read_triangle_mesh(item["source_path"])
            converted = o3d.io.read_triangle_mesh(item["cad_path"])
            transform = np.asarray(item["cad_T_source_m"])
            expected = (np.asarray(original.vertices) * .001) @ transform[:3, :3].T + transform[:3, 3]
            np.testing.assert_allclose(np.ptp(np.asarray(converted.vertices), axis=0),
                                       np.ptp(expected, axis=0), atol=1e-7)
            self.assertTrue(o3d.io.read_triangle_mesh(item["visual_mesh_path"]).has_triangle_uvs())
        self.assertIn("bottom", catalog["lmo_glue"]["description_view_policy"]["excluded_views"])


if __name__ == "__main__":
    unittest.main()
