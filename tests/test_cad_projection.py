import numpy as np
import open3d as o3d

from scripts.surface_verification import CadSurface, depth_points, translation_fit


def test_translation_only_fit_preserves_shape():
    points = np.array([[0., 0, 0], [.02, 0, 0], [0, .03, 0], [0, 0, .04]])
    t = np.array([.1, .2, .3])
    estimate, fitness = translation_fit(points, points + t)
    np.testing.assert_allclose(estimate, t)
    assert fitness == 1


def test_native_cad_ray_depth_and_backprojection():
    mesh = o3d.geometry.TriangleMesh.create_box(.04, .04, .04).translate([-.02, -.02, -.02])
    transform = np.eye(4)
    transform[2, 3] = .4
    k = np.array([[200., 0, 64], [0, 200., 64], [0, 0, 1.]])
    render = CadSurface(mesh).render(transform, k, (128, 128))
    assert not render['clipped']
    mask = np.isfinite(render['depth'])
    np.testing.assert_allclose(render['depth'][mask], .38, atol=1e-6)
    points = depth_points(render['depth'], mask, render['K'])
    assert abs(points[:, :2]).max() <= .02
    np.testing.assert_allclose(points[:, 2], .38, atol=1e-6)


def test_behind_camera_is_unavailable():
    mesh = o3d.geometry.TriangleMesh.create_box(.04, .04, .04)
    transform = np.eye(4)
    transform[2, 3] = -.4
    assert CadSurface(mesh).render(transform, np.eye(3), (128, 128)) is None
