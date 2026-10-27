"""Focused native-scale, orientation and partial-depth regressions; no labels."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
from scipy.spatial import cKDTree

SPEC = importlib.util.spec_from_file_location('depth_trial_geometry', Path(__file__).with_name('geometry.py'))
geometry = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = geometry
SPEC.loader.exec_module(geometry)


def box_vertices(dimensions=(.008, .007, .05)):
    return np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5)
                     for z in (-.5, .5)]) * dimensions


def test_cad_support_faces_preserve_native_width_thickness_and_length():
    vertices = box_vertices()
    faces = geometry.support_faces(vertices)
    assert len(faces) == 6
    normals = np.array([row['outward_normal'] for row in faces])
    np.testing.assert_allclose(np.sort(np.abs(normals).max(1)), np.ones(6))
    np.testing.assert_allclose(np.ptp(vertices, axis=0), [.008, .007, .05])


def test_tilted_table_contact_and_both_clearances_are_exact():
    vertices = box_vertices()
    observed = vertices + [0, 0, .4]
    plane = [.25, -.3, np.sqrt(1-.25**2-.3**2), -.3]
    normal, distance, basis = geometry.table_frame(plane, observed)
    for clearance in (0., .001):
        for outward in ([0, 0, -1], [1, 0, 0]):
            pose = geometry.contact_pose(vertices, observed, normal, distance, basis,
                                         outward, 37., clearance)
            points = vertices @ pose[:3, :3].T + pose[:3, 3]
            assert abs((points @ normal + distance).min() - clearance) < 1e-12
            np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-12)
            assert abs(np.linalg.det(pose[:3, :3]) - 1) < 1e-12


def test_continuous_yaw_partial_cloud_fit_keeps_height_and_scale():
    rng = np.random.default_rng(7)
    points = rng.uniform([-.025, -.01, 0], [.025, .01, .016], (700, 3))
    points[:, 2] -= points[:, 2].min()
    cad = SimpleNamespace(points=points, tree=cKDTree(points))
    truth = np.eye(4)
    truth[:3, :3] = geometry.axis_rotation([0, 0, 1], np.deg2rad(31.737))
    truth[:3, 3] = [.08, -.04, .001]
    observed = points[::2] @ truth[:3, :3].T + truth[:3, 3]
    initial = truth.copy()
    delta = geometry.axis_rotation([0, 0, 1], np.deg2rad(1.13))
    initial[:3, :3] = delta @ truth[:3, :3]
    initial[:2, 3] += [.0007, -.0004]
    pose, score = geometry.refine_pose(cad, observed, np.array([0., 0., 1.]), np.eye(3)[:, :2], initial)
    assert score['rmse_m'] < 1e-10
    assert np.all(np.diff(score['rmse_history_m']) <= 1e-12)
    np.testing.assert_allclose(pose, truth, atol=1e-9)


def test_pixel_zero_backprojection_keeps_global_roi_coordinates():
    K = np.array([[100., 0, 12.], [0, 200., 9.], [0, 0, 1.]])
    depth = np.full((2, 3), .5)
    mask = np.zeros((2, 3), bool)
    mask[1, 2] = True
    points = geometry.depth_points(depth, mask, K, roi_xy=(10, 8))
    np.testing.assert_allclose(points, [[0., 0., .5]])


def test_full_union_penalizes_visible_extra_cad_even_if_scene_depth_is_closer():
    observed = np.array([[True, True, False, False]])
    depth = np.array([[.4, .4, .3, .3]])
    rendered = np.array([[.4, .4, .4, .4]])
    score = geometry.score_projection(depth, observed, rendered)
    assert score['status'] == 'available'
    assert score['p_obs'] == 0 and score['p_cad'] == .5 and score['p_depth'] == 0
    assert score['geometry_score'] == 1 - .5 / 3
    assert score['projected_pixels'] == 4


def test_unobserved_hidden_cad_surface_is_not_a_depth_residual():
    # Only first-hit CAD depth enters this score. The hidden back face is absent.
    observed = np.array([[True, True, False]])
    depth = np.array([[.4, .4, .9]])
    rendered = np.array([[.4, .4, np.inf]])
    score = geometry.score_projection(depth, observed, rendered)
    assert score['geometry_score'] == 1


def test_no_overlap_and_missing_depth_are_explicitly_unavailable():
    observed = np.array([[True, False]])
    depth = np.array([[.4, .4]])
    rendered = np.array([[np.inf, .4]])
    assert geometry.score_projection(depth, observed, rendered)['geometry_score'] is None
    depth[0, 0] = np.nan
    assert geometry.score_projection(depth, observed, np.array([[.4, np.inf]]))['geometry_score'] is None


def test_five_mm_depth_error_scale_is_separate_from_one_mm_voxels():
    score = geometry.score_projection(np.array([[.401]]), np.array([[True]]), np.array([[.4]]))
    assert abs(score['p_depth'] - .2) < 1e-12
    assert geometry.VOXEL_M == .001


def test_exact_raycaster_distinguishes_eight_and_twelve_mm_native_widths():
    o3d = pytest.importorskip('open3d')
    from scripts.surface_verification import CadSurface
    K = np.array([[1000., 0, 96.], [0, 1000., 96.], [0, 0, 1.]])
    pose = np.eye(4)
    pose[2, 3] = .4
    surfaces = []
    for width in (.008, .012):
        mesh = o3d.geometry.TriangleMesh.create_box(width, .05, .008)
        mesh.translate([-width/2, -.025, -.004])
        surfaces.append(CadSurface(mesh))
    truth = surfaces[0].render(pose, K, (192, 192), pixel_offset=0.)
    observed, depth = np.zeros((192, 192), bool), np.full((192, 192), .6)
    x1, y1, x2, y2 = truth['roi']
    visible = np.isfinite(truth['depth'])
    observed[y1:y2, x1:x2] = visible
    depth[y1:y2, x1:x2][visible] = truth['depth'][visible]
    scores = [geometry.projection_score(SimpleNamespace(surface=surface), pose, K, depth, observed)
              for surface in surfaces]
    assert scores[0]['geometry_score'] == 1
    assert scores[1]['p_cad'] > 0
    assert scores[1]['geometry_score'] < scores[0]['geometry_score']
    # The wider CAD projects beyond the observed pin's rectangle. Its full
    # silhouette must remain, and analytic outside-observation counting must
    # equal the explicit full-frame union calculation.
    wrong = surfaces[1].render(pose, K, (192, 192), pixel_offset=0.)
    wrong_full = np.full((192, 192), np.inf)
    x1, y1, x2, y2 = wrong['roi']
    wrong_full[y1:y2, x1:x2] = wrong['depth']
    explicit = geometry.score_projection(depth, observed, wrong_full)
    for field in explicit:
        assert scores[1][field] == explicit[field]


def test_cad_support_normal_search_recovers_non_axis_aligned_box_faces():
    rotation = geometry.axis_rotation([.4, .1, .7], np.deg2rad(23))
    vertices = box_vertices() @ rotation.T
    faces = geometry.support_faces(vertices)
    assert len(faces) == 6
    normals = np.array([row['outward_normal'] for row in faces])
    for axis in rotation.T:
        assert np.max(normals @ axis) > 1 - 1e-10
        assert np.min(normals @ axis) < -1 + 1e-10


def test_axis_priority_retains_small_cylinder_ends_among_large_side_supports():
    theta = np.arange(64) * 2 * np.pi / 64
    ring = np.column_stack([.004 * np.cos(theta), .004 * np.sin(theta)])
    vertices = np.vstack([np.column_stack([ring, np.full(64, z)]) for z in (-.025, .025)])
    faces = geometry.support_faces(vertices)
    normals = np.array([row['outward_normal'] for row in faces])
    assert len(faces) <= 12
    assert np.max(normals[:, 2]) > 1 - 1e-10
    assert np.min(normals[:, 2]) < -1 + 1e-10
    assert np.max(np.abs(normals[:, 0])) > .99
    assert np.max(np.abs(normals[:, 1])) > .99


def test_analytic_outside_cad_observation_count_equals_full_union_randomly():
    rng = np.random.default_rng(20261023)
    for _ in range(30):
        observed = rng.random((31, 43)) > .8
        depth = rng.uniform(.3, .7, (31, 43))
        depth[rng.random(depth.shape) < .1] = 0
        rendered = np.full(depth.shape, np.inf)
        x1, y1, x2, y2 = 11, 7, 29, 23
        patch = rng.uniform(.3, .7, (y2-y1, x2-x1))
        patch[rng.random(patch.shape) < .3] = np.inf
        rendered[y1:y2, x1:x2] = patch
        full = geometry.score_projection(depth, observed, rendered)
        optimized = geometry.score_projection(depth[y1:y2, x1:x2], observed[y1:y2, x1:x2],
                                               patch, observed_total=int(observed.sum()))
        assert full == optimized
