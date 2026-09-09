import unittest

import numpy as np
import open3d as o3d

from point_cloud_localization import PointCloudConfig, PointCloudLocalization


class PointCloudLocalizationTests(unittest.TestCase):
    def test_footprint_clustering_joins_height_fragments_without_merging_neighbor(self):
        from scipy.spatial.transform import Rotation
        patch = np.array(np.meshgrid(np.arange(4)*.002, np.arange(4)*.002,
                                     np.arange(2)*.002)).reshape(3, -1).T
        points = np.vstack((patch, patch + [0, 0, .020], patch + [.045, 0, 0]))
        rotation = Rotation.from_euler("x", 25, degrees=True).as_matrix()
        points = points @ rotation.T + [0, 0, 1.]
        normal = rotation[:, 2]
        plane = [*normal, -normal[2]]
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        intrinsics = {"width": 256, "height": 256, "fx": 300., "fy": 300., "cx": 128., "cy": 128.}
        config = PointCloudConfig(dbscan_eps_m=.006, dbscan_min_points=2, min_cluster_points=4,
                                 min_cluster_extent_m=.001, min_roi_width_px=1, min_roi_height_px=1)
        localizer = PointCloudLocalization(config)
        self.assertEqual(len(localizer.cluster_objects(cloud, intrinsics, plane)), 3)
        config.cluster_in_table_plane = True
        clusters = localizer.cluster_objects(cloud, intrinsics, plane)
        self.assertEqual(sorted(len(c.points) for c in clusters), [32, 64])
        self.assertGreater(np.ptp(np.asarray(clusters[0].points) @ normal), .020)
        # Raw-cloud cleanup must retain both faces as well, rather than undoing the fix.
        cleaned = localizer.clean_raw_cluster(clusters[0], plane)
        self.assertGreater(np.ptp(np.asarray(cleaned.points) @ normal), .019)

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
