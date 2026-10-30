"""Offline numerical acceptance tests; no BOP ground truth or model downloads."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from freezev2_repro.registration import (
    Correspondences,
    RegistrationConfig,
    feature_ransac,
    joint_score,
    rank_candidates,
    refine_icp,
    rigid_fit,
    score_pose,
    topk_correspondences,
)


def _rotation():
    # Large rotations about three axes: explicitly outside a yaw-only search.
    return Rotation.from_euler("xyz", [117, -73, 158], degrees=True).as_matrix()


def _angular_error(actual, expected):
    return Rotation.from_matrix(actual @ expected.T).magnitude()


def test_rigid_fit_recovers_unrestricted_transform_without_scaling():
    rng = np.random.default_rng(12)
    cad = rng.normal(size=(40, 3)) * [0.025, 0.042, 0.013] + [0.1, 0.02, -0.01]
    rotation, translation = _rotation(), np.array([0.11, -0.09, 0.72])
    observed = cad @ rotation.T + translation
    actual_rotation, actual_translation = rigid_fit(cad, observed)
    assert _angular_error(actual_rotation, rotation) < 1e-12
    assert np.allclose(actual_translation, translation, atol=1e-12)
    assert np.isclose(np.linalg.det(actual_rotation), 1)
    assert np.allclose(cad @ actual_rotation.T + actual_translation, observed, atol=1e-12)
    # A reflected target cannot become a reflection in a proper rigid pose.
    reflected = observed.copy()
    reflected[:, 0] *= -1
    reflected_rotation, _ = rigid_fit(cad, reflected)
    assert np.isclose(np.linalg.det(reflected_rotation), 1)


def test_rigid_fit_rejects_unobservable_collinear_triplets():
    line = np.column_stack([np.arange(4), np.zeros((4, 2))])
    with pytest.raises(ValueError, match="Collinear"):
        rigid_fit(line, line)


def test_topk_is_cosine_stable_and_preserves_every_pair():
    cad = np.array([[2, 0], [4, 0], [0, 5], [-1, 0]], dtype=float)
    observed = np.array([[3, 0], [0, 7]], dtype=float)
    pairs = topk_correspondences(cad, observed, 3)
    assert pairs.cad_indices.tolist() == [[0, 1, 2], [2, 0, 1]]
    assert np.allclose(pairs.similarities, [[1, 1, 0], [1, 0, 0]])
    indices, similarities = pairs
    assert indices.shape == similarities.shape == (2, 3)
    with pytest.raises(ValueError, match="k must"):
        topk_correspondences(cad, observed, 5)
    with pytest.raises(ValueError, match="zero-length"):
        topk_correspondences(cad, [[0, 0]], 1)


def test_feature_score_uses_all_topk_inliers_over_sparse_count_not_pair_count():
    # Two CAD points per observed row lie within tau.  Their cosine sum is
    # 1.5 per observed row; correct Eq.5 score is 1.5, not 0.75 or <=1.
    cad = np.array([[0, 0, 0], [0.001, 0, 0], [0.03, 0.02, 0], [0.031, 0.02, 0]])
    observed = cad[[0, 2]]
    pairs = Correspondences(np.array([[0, 1], [2, 3]]), np.array([[1, 0.5], [1, 0.5]]))
    scores = score_pose(cad, observed, np.eye(3), np.zeros(3), 0.2, pairs, dense_xyz=observed)
    assert scores.feature_score == pytest.approx(1.5)
    assert scores.feature_inlier_pairs == 4
    assert scores.sparse_observed_count == 2
    assert scores.geometry_score == 1.0
    assert joint_score(1.5, 1.5, 1.0) == 2.25
    # Geometry is one-sided CAD coverage; a distant hidden CAD point contributes
    # to the denominator even though all sparse observed points are explained.
    more_cad = np.vstack([cad, [0.2, 0.2, 0.2]])
    scores = score_pose(more_cad, observed, np.eye(3), np.zeros(3), 0.2, pairs, dense_xyz=observed)
    assert scores.geometry_score == pytest.approx(4 / 5)
    assert scores.feature_score == pytest.approx(1.5)


def test_feature_ransac_recovers_large_rotation_with_partial_overlap_and_outliers():
    rng = np.random.default_rng(43)
    cad = rng.uniform(-0.05, 0.05, size=(100, 3))
    features = rng.normal(size=(100, 48))
    selected = rng.choice(len(cad), 34, replace=False)
    rotation, translation = _rotation(), np.array([0.04, -0.03, 0.58])
    observed_inliers = cad[selected] @ rotation.T + translation
    # 35% corrupted observed rows have arbitrary point and descriptor values.
    observed = np.vstack([observed_inliers, rng.uniform(-0.1, 0.1, size=(18, 3)) + translation])
    observed_features = np.vstack([features[selected], rng.normal(size=(18, 48))])
    # k=3 provides two distractor matches for each correct sparse observation.
    cfg = RegistrationConfig(k=3, iterations=4_000, seed=8)
    coarse = feature_ransac(cad, observed, features, observed_features, 0.18, cfg)
    assert coarse.status == "ok"
    assert _angular_error(coarse.R, rotation) < 1e-10
    assert np.linalg.norm(coarse.t - translation) < 1e-11
    assert coarse.diagnostics["valid_hypotheses"] > 0
    assert coarse.diagnostics["correspondence_pairs"] == len(observed) * 3
    independent = score_pose(cad, observed, coarse.R, coarse.t, 0.18, coarse.correspondences, config=cfg)
    assert coarse.score == pytest.approx(independent.feature_score, abs=1e-12)
    repeated = feature_ransac(cad, observed, features, observed_features, 0.18, cfg)
    assert np.array_equal(coarse.R, repeated.R)
    assert np.array_equal(coarse.t, repeated.t)
    assert coarse.diagnostics["best_hypothesis_index"] == repeated.diagnostics["best_hypothesis_index"]


def test_no_valid_ransac_hypothesis_returns_explicit_unavailable_pose():
    cad = np.column_stack([np.arange(12) / 100, np.zeros((12, 2))])
    descriptors = np.eye(12)
    result = feature_ransac(cad, cad + [0.01, 0.02, 0.6], descriptors, descriptors, 0.2, RegistrationConfig(k=1, iterations=32))
    assert result.status == "no_valid_hypothesis"
    assert result.R is result.t is result.score is None
    assert result.diagnostics["valid_hypotheses"] == 0
    assert result.diagnostics["degenerate_triplets"] == 32


def test_icp_recovers_small_residual_on_a_large_initial_rotation():
    rng = np.random.default_rng(123)
    cad = rng.uniform(-0.05, 0.05, size=(180, 3))
    true_rotation, true_translation = _rotation(), np.array([0.025, -0.03, 0.65])
    dense = cad @ true_rotation.T + true_translation
    initial_rotation = Rotation.from_euler("x", 0.7, degrees=True).as_matrix() @ true_rotation
    initial_translation = true_translation + [0.0005, -0.0004, 0.0006]
    cfg = RegistrationConfig(icp_max_iterations=30)
    pairs = Correspondences(np.arange(len(cad))[:, None], np.ones((len(cad), 1)))
    before = score_pose(cad, dense, initial_rotation, initial_translation, 0.18, pairs, dense, cfg)
    result = refine_icp(cad, dense, initial_rotation, initial_translation, 0.18, cfg)
    after = score_pose(cad, dense, result.R, result.t, 0.18, pairs, dense, cfg)
    assert result.status.startswith("converged")
    assert result.iterations > 0
    assert result.rmse_m < 1e-12
    assert _angular_error(result.R, true_rotation) < 1e-12
    assert np.linalg.norm(result.t - true_translation) < 1e-12
    assert after.geometry_score == result.geometry_score == 1.0
    assert after.feature_score >= before.feature_score


def test_icp_does_not_invent_pose_when_all_points_outside_gate():
    rng = np.random.default_rng(4)
    cad = rng.normal(size=(20, 3)) / 100
    R, t = _rotation(), np.array([0.0, 0.0, 0.7])
    result = refine_icp(cad, cad + [3, 3, 3], R, t, 0.1)
    assert result.status == "insufficient_inliers"
    assert result.inlier_count == 0
    assert result.rmse_m is None
    assert np.array_equal(result.R, R)
    assert np.array_equal(result.t, t)


def test_refined_score_reuses_c_and_rejects_wrong_point_rows():
    cad = np.array([[0, 0, 0], [0.02, 0, 0], [0, 0.03, 0]])
    observed = cad + [0, 0, 0.5]
    fixed = Correspondences(np.arange(3)[:, None], np.ones((3, 1)))
    good = score_pose(cad, observed, np.eye(3), [0, 0, 0.5], 0.1, fixed)
    wrong = score_pose(cad, observed, np.eye(3), [0, 0, 0.51], 0.1, fixed)
    assert good.feature_score == 1
    assert wrong.feature_score == 0
    assert fixed.cad_indices.tolist() == [[0], [1], [2]]
    with pytest.raises(ValueError, match="one nonempty row"):
        score_pose(cad, observed, np.eye(3), [0, 0, 0.5], 0.1, Correspondences(np.array([[0]]), np.ones((1, 1))))


def test_candidate_ranking_uses_joint_score_and_same_identity_translation_nms():
    def candidate(identifier, x, score, obj_id=1, confidence=0.5):
        return {"candidate_id": identifier, "obj_id": obj_id, "R": _rotation(), "t": np.array([x, 0, 0.7]), "score": score, "segmentation_confidence": confidence, "status": "ok"}
    candidates = [candidate("low_pose_high_mask", 0, 0.2, confidence=0.99), candidate("best", 0.001, 1.4, confidence=0.1), candidate("other_instance", 0.08, 0.9), candidate("other_CAD", 0.002, 0.8, obj_id=2)]
    candidates += [{"candidate_id": "failed", "status": "no_valid_hypothesis"}]
    ranked = rank_candidates(candidates, translation_nms_threshold_m=0.01)
    assert [item["candidate_id"] for item in ranked.selected] == ["best", "other_instance", "other_CAD"]
    assert ranked.suppressed[0]["candidate_id"] == "low_pose_high_mask"
    assert ranked.suppressed[0]["winner_candidate_id"] == "best"
    assert ranked.rejected[0]["candidate_id"] == "failed"
    limited = rank_candidates(candidates, translation_nms_threshold_m=0.01, max_instances=1)
    assert [item["candidate_id"] for item in limited.selected] == ["best"]
    assert any(item["reason"] == "instance_count_limit" for item in limited.suppressed)
    with pytest.raises(ValueError, match="explicit positive"):
        rank_candidates(candidates, translation_nms_threshold_m=0)


def test_candidate_tie_breaking_preserves_input_order():
    candidates = [{"candidate_id": i, "R": np.eye(3), "t": [i * 0.1, 0, 0.6], "score": 1.0} for i in [3, 1, 2]]
    result = rank_candidates(candidates, translation_nms_threshold_m=0.01)
    assert [item["candidate_id"] for item in result.selected] == [3, 1, 2]


def test_invalid_config_is_not_silently_ignored():
    cad, descriptors = np.eye(3), np.eye(3)
    with pytest.raises(ValueError, match="Unknown registration"):
        feature_ransac(cad, cad, descriptors, descriptors, 0.1, {"yaw_only": True})
    with pytest.raises(ValueError, match="iterations must be an integer"):
        feature_ransac(cad, cad, descriptors, descriptors, 0.1, {"iterations": 10.5})


def test_cuda_backend_is_explicit_or_matches_cpu_when_available():
    try:
        import torch
    except ImportError:
        torch = None
    rng = np.random.default_rng(87)
    cad, descriptors = rng.normal(size=(12, 3)) * 0.01, np.eye(12)
    observed = cad @ _rotation().T + [0.01, 0.02, 0.5]
    cfg = RegistrationConfig(k=1, iterations=64, seed=19, hypothesis_backend="torch_cuda", torch_dtype="float64")
    if torch is None or not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="torch_cuda"):
            feature_ransac(cad, observed, descriptors, descriptors, 0.1, cfg)
    else:
        cuda = feature_ransac(cad, observed, descriptors, descriptors, 0.1, cfg)
        cpu = feature_ransac(cad, observed, descriptors, descriptors, 0.1, replace(cfg, hypothesis_backend="numpy"))
        assert cuda.status == cpu.status == "ok"
        assert cuda.score == pytest.approx(cpu.score, abs=1e-12)
        assert _angular_error(cuda.R, _rotation()) < 1e-10
        assert np.allclose(cuda.t, cpu.t, atol=1e-10)
