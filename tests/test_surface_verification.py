import numpy as np
import open3d as o3d
from scripts.surface_verification import surface_scores, SIGMA_M, translation_fit, CadSurface, verify_pose


def test_depth_and_support_limits():
    K = np.eye(3)
    d = np.ones((1, 3))
    mask = np.array([[True, True, False]])
    r = np.array([[1., 1., np.inf]])
    assert surface_scores(d, mask, r, K, 0)["surface_score"] == 1
    r[0, 2] = 1
    assert surface_scores(d, mask, r, K, 0)["surface_score"] == 2 / 3
    r[0, 0] += SIGMA_M
    score = surface_scores(d, mask, r, K, 0)
    assert np.isclose(score["surface_score"], .5)
    assert score["surface_score"] < score["mask_iou"]


def test_external_occlusion_does_not_excuse_target_error():
    d = np.array([[1., 1.]])
    r = np.array([[1.02, 1.02]])
    scores = surface_scores(d, np.array([[True, False]]), r, np.eye(3), 0)
    assert scores["externally_occluded_pixels"] == 1
    assert 0 < scores["surface_score"] < .1


def test_unknown_depth_unassessable():
    result = surface_scores(np.zeros((2, 2)), np.ones((2, 2), bool), np.ones((2, 2)), np.eye(3))
    assert result["surface_score"] is None
    assert result["unknown_rendered_pixels"] == 4


def test_translation_only_fit_preserves_shape():
    points = np.array([[0., 0, 0], [.02, 0, 0], [0, .03, 0], [0, 0, .04]])
    t = np.array([.1, .2, .3])
    estimate, fitness = translation_fit(points, points + t)
    np.testing.assert_allclose(estimate, t)
    assert fitness == 1


def test_cropped_rays_match_independent_full_frame_render():
    from scripts.wrist_oracle_pose import perspective_render
    mesh = o3d.geometry.TriangleMesh.create_box(.025, .02, .04).translate([-.0125, -.01, -.02])
    T = np.eye(4)
    T[:3, 3] = [.025, -.015, .4]
    K = np.array([[200., 0, 64], [0, 200., 64], [0, 0, 1.]])
    expected = perspective_render(mesh, T, K, (128, 128), [1, 1, 1])
    actual = CadSurface(mesh).render(T, K, (128, 128))
    x1, y1, x2, y2 = actual["roi"]
    np.testing.assert_allclose(actual["depth"], expected["depth_m"][y1:y2, x1:x2], atol=1e-6)
    mask = expected["mask"]
    measured = np.where(mask, expected["depth_m"], .7)
    score = verify_pose(CadSurface(mesh), T, measured, mask, K)
    assert score["surface_score"] > .999999
    shifted = T.copy()
    shifted[0, 3] += .02
    assert verify_pose(CadSurface(mesh), shifted, measured, mask, K)["surface_score"] < .2


def test_invalid_pose_cannot_hide_below_table():
    mesh = o3d.geometry.TriangleMesh.create_box(.02, .02, .02)
    T = np.eye(4)
    T[2, 3] = .7
    score = verify_pose(CadSurface(mesh), T, np.full((64, 64), .6), np.ones((64, 64), bool),
                        np.array([[100., 0, 32], [0, 100., 32], [0, 0, 1]]), np.array([0., 0, -1, .6]))
    assert score["status"] == "table_penetration" and score["surface_score"] == 0


def test_invalid_pose_cannot_win_fusion_with_strong_appearance():
    from scripts.run_surface_verification import collapse_pair
    rows = [{"view_index_1based": 1, "status": "table_penetration", "global": 1., "patch": 1., "surface_score": 0.},
            {"view_index_1based": 2, "status": "assessable", "global": 1., "patch": .1, "surface_score": .1}]
    result = collapse_pair({"best_view_index_1based": 1}, rows)
    assert result["fused_view"] == 2 and result["frozen_view_fused"] is None
    assert rows[0]["view_consistent_fused"] is None
    assert collapse_pair({"best_view_index_1based": 1}, rows[:1])["geometry"] is None
