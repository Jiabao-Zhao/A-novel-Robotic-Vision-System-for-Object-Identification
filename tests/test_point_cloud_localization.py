import unittest

import numpy as np
import open3d as o3d

from point_cloud_localization import PointCloudLocalization


class PointCloudLocalizationTests(unittest.TestCase):
    def test_localized_cluster_contains_numeric_geometry(self):
        points = np.array(
            [
                [-0.1, -0.1, 1.0],
                [0.1, -0.1, 1.0],
                [-0.1, 0.1, 1.2],
                [0.1, 0.1, 1.2],
            ],
            dtype=float,
        )
        cluster = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        intrinsics = {
            "width": 256,
            "height": 256,
            "fx": 100.0,
            "fy": 100.0,
            "cx": 128.0,
            "cy": 128.0,
        }

        localized = PointCloudLocalization().localize_clusters([cluster], intrinsics)[0]

        self.assertEqual(localized.object_id, "object_001")
        self.assertEqual(localized.point_count, 4)
        self.assertEqual(localized.saved_point_count, 0)
        np.testing.assert_allclose(localized.centroid_3d_m, [0.0, 0.0, 1.1])
        np.testing.assert_allclose(localized.size_3d_m, [0.2, 0.2, 0.2])


if __name__ == "__main__":
    unittest.main()
