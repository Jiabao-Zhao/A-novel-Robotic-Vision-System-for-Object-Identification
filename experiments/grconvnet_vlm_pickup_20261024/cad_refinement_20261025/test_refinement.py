"""Focused geometric invariants; no model weights, robot motion or physics."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from .refinement import AlignedCAD, MAX_SEED_SHIFT_M, protocol, rigid_target_at_initial_hand


def pose(rotation, translation):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("xyz", rotation).as_matrix()
    result[:3, 3] = translation
    return result


class RefinedGeometryTests(unittest.TestCase):
    def test_target_relocation_preserves_relative_pose(self):
        reference = pose([.4, -.6, .2], [.5, -.4, .2])
        proposed = pose([0., np.pi, -.9], [-.2, .05, .014])
        cad = pose([.01, -.005, .7], [-.21, .06, .009])
        moved = rigid_target_at_initial_hand(reference, proposed, cad)
        np.testing.assert_allclose(np.linalg.inv(reference) @ moved,
                                   np.linalg.inv(proposed) @ cad, atol=1e-14)

    def test_external_ray_contacts_keep_seed_line_and_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"box.stl"
            mesh = o3d.geometry.TriangleMesh.create_box(.04, .02, .016)
            mesh.translate([-.02, -.01, -.008])
            mesh.compute_vertex_normals()
            self.assertTrue(o3d.io.write_triangle_mesh(str(path), mesh))
            cad = AlignedCAD(path, np.eye(4))
            seed, axis = np.array([.001, .001, -.003]), np.array([1., 0., 0.])
            pairs, diagnostic = cad.local_pairs(seed, axis)
            self.assertEqual(diagnostic["CAD_lines_tested"], 15)
            self.assertEqual(len(pairs), 15)
            for pair in pairs:
                np.testing.assert_allclose(pair["contacts_world_m"][:, 0], [-.02, .02], atol=1e-8)
                delta = np.diff(pair["contacts_world_m"], axis=0)[0]
                np.testing.assert_allclose(delta/np.linalg.norm(delta), axis, atol=1e-8)
                self.assertLessEqual(pair["xy_shift_m"], MAX_SEED_SHIFT_M)
                self.assertEqual(pair["along_m"], 0.)
            far_pairs, far_diagnostic = cad.local_pairs(np.array([.03, .001, 0.]), axis)
            self.assertEqual(far_pairs, [])
            self.assertEqual(far_diagnostic["rejected_lines"]["seed_shift"], 15)

    def test_outer_rays_do_not_select_the_bore(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"ring.stl"
            mesh = o3d.geometry.TriangleMesh.create_torus(.02, .006, 80, 40)
            mesh.compute_vertex_normals()
            self.assertTrue(o3d.io.write_triangle_mesh(str(path), mesh))
            cad = AlignedCAD(path, np.eye(4))
            pairs, _ = cad.local_pairs(np.zeros(3), np.array([1., 0., 0.]))
            centerline = next(pair for pair in pairs if pair["fraction"] == .5 and pair["across_m"] == 0.)
            np.testing.assert_allclose(centerline["contacts_world_m"][:, 0], [-.026, .026], atol=1e-7)
            self.assertGreater(centerline["span_m"], .05)

    def test_frozen_search_budget(self):
        config = protocol()
        self.assertEqual(config["maximum_CAD_lines_per_seed"], 15)
        self.assertEqual(config["maximum_pose_opening_variants_per_seed"], 270)
        self.assertEqual(config["maximum_returned_poses_per_network_seed"], 1)
        self.assertEqual(config["finger_depths_m"], (.004, 0., -.004, -.008, .006, .008))


if __name__ == "__main__":
    unittest.main()
