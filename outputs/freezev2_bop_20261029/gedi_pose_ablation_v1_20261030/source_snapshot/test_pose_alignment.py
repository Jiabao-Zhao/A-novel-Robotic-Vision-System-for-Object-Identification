"""Focused CPU checks for row mapping and unchanged geometry denominators."""
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
SNAPSHOT = HERE.parents[1] / "outputs/freezev2_bop_20261029/classic_core_base_v2/source_snapshot"
sys.path.insert(0, str(SNAPSHOT))
sys.path.insert(0, str(HERE))

from pose_alignment import align_rows, paired_full_geometry_score
from freezev2_repro.registration import Correspondences, RegistrationConfig, score_pose


def _rows(n=5):
    return (np.arange(n * 3, dtype=np.float32).reshape(n, 3),
            np.arange(n * 4, dtype=np.float32).reshape(n, 4) + 1,
            np.arange(n * 2, dtype=np.float32).reshape(n, 2) + 1)


def test_retained_rows_map_both_CAD_and_observation_coordinates_and_visuals():
    points, visual, geo = _rows()
    # Noncontiguous removal at both ends reveals the common prefix-truncation bug.
    indices = np.array([1, 3], dtype=np.int64)
    before = [v.copy() for v in (points, visual, geo, indices)]
    for offset in (0.0, 100.0):
        result = align_rows(points + offset, visual + offset, geo[indices], indices)
        np.testing.assert_array_equal(result["points"], (points + offset)[indices])
        np.testing.assert_array_equal(result["visual"], (visual + offset)[indices])
        np.testing.assert_array_equal(result["geometric"], geo[indices])
        np.testing.assert_array_equal(result["original_indices"], indices)
    for actual, expected in zip((points, visual, geo, indices), before):
        np.testing.assert_array_equal(actual, expected)


def test_all_valid_control_is_identity_without_normalization_or_dtype_change():
    points, visual, geo = _rows()
    result = align_rows(points, visual, geo, np.arange(len(points)))
    for name, original in (("points", points), ("visual", visual), ("geometric", geo)):
        np.testing.assert_array_equal(result[name], original)
        assert result[name].dtype == original.dtype


@pytest.mark.parametrize("indices", [[1, 1], [2, 0], [-1, 2], [1, 5],
                                     [1.0, 2.0], [True, False], [[1, 2]], []])
def test_invalid_original_row_indices_fail_explicitly(indices):
    points, visual, geo = _rows()
    with pytest.raises(ValueError, match="original_indices"):
        align_rows(points, visual, geo[:2], indices)


def test_unsigned_descending_indices_do_not_wrap_past_order_validation():
    points, visual, geo = _rows()
    with pytest.raises(ValueError, match="strictly increasing"):
        align_rows(points, visual, geo[:2], np.array([3, 1], dtype=np.uint64))


@pytest.mark.parametrize("which", ["points", "visual", "geometric"])
def test_nonfinite_inputs_are_not_hidden_by_query_selection(which):
    points, visual, geo = _rows()
    arrays = {"points": points, "visual": visual, "geometric": geo[:2].copy()}
    arrays[which][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        align_rows(**arrays, original_indices=[1, 3])


@pytest.mark.parametrize("which", ["visual", "geometric"])
def test_zero_modality_rows_are_not_replaced_or_silently_normalized(which):
    points, visual, geo = _rows()
    arrays = {"points": points, "visual": visual, "geometric": geo[:2].copy()}
    arrays[which][0] = 0
    with pytest.raises(ValueError, match="zero-length"):
        align_rows(**arrays, original_indices=[1, 3])


def test_misaligned_shapes_and_non_numeric_arrays_fail():
    points, visual, geo = _rows()
    with pytest.raises(ValueError, match="same original row count"):
        align_rows(points, visual[:-1], geo[:2], [1, 3])
    with pytest.raises(ValueError, match="geometric rows"):
        align_rows(points, visual, geo[:3], [1, 3])
    with pytest.raises(ValueError, match="points"):
        align_rows(points[:, :2], visual, geo[:2], [1, 3])
    with pytest.raises(ValueError, match="real numeric"):
        align_rows(points, visual.astype(str), geo[:2], [1, 3])


def _score_fixture():
    full = np.array([[0., 0., 0.], [.02, 0., 0.], [0., .02, 0.], [1., 1., 1.]])
    feature = full[:3].copy()
    R = np.eye(3)
    t = np.array([0., 0., .7])
    dense = feature + t
    C = Correspondences(np.arange(3)[:, None], np.array([[1.], [.8], [.6]]))
    return full, feature, dense, R, t, C


def test_geometry_denominator_includes_feature_excluded_CAD_row():
    full, feature, dense, R, t, C = _score_fixture()
    cfg = RegistrationConfig(inlier_fraction=.03)
    before_full, before_dense = full.copy(), dense.copy()
    result = paired_full_geometry_score(feature, dense, R, t, .2, C, full, dense, cfg)
    frozen_subset = score_pose(feature, dense, R, t, .2, C, dense, cfg)
    assert frozen_subset.geometry_score == 1.0
    assert result["feature_score"] == frozen_subset.feature_score
    assert result["geometry_score"] == .75
    assert result["geometry_inlier_points"] == 3
    assert result["cad_count"] == 4 and result["feature_cad_count"] == 3
    assert result["sparse_observed_count"] == 3
    np.testing.assert_array_equal(full, before_full)
    np.testing.assert_array_equal(dense, before_dense)


def test_all_valid_scoring_matches_every_original_score_field_exactly():
    full, feature, dense, R, t, C = _score_fixture()
    cfg = RegistrationConfig()
    expected = asdict(score_pose(feature, dense, R, t, .2, C, dense, cfg))
    actual = paired_full_geometry_score(feature, dense, R, t, .2, C, feature, dense, cfg)
    assert {key: actual[key] for key in expected} == expected


def test_geometry_preserves_strict_gate_and_configured_threshold():
    _, feature, dense, R, t, C = _score_fixture()
    # The extra CAD point is exactly 1/8m from its nearest dense observation.
    # Using <= instead of the frozen strict < gate would incorrectly count it.
    full = np.vstack([feature, [0., 0., .125]])
    result = paired_full_geometry_score(feature, dense, R, t, 1., C, full, dense,
                                       {"inlier_fraction": .125})
    assert result["threshold_m"] == .125
    assert result["cad_count"] == 4
    assert result["geometry_inlier_points"] == 3
    assert result["geometry_score"] == .75
    expected = asdict(score_pose(full, dense, R, t, 1., C, dense_xyz=dense,
                                config={"inlier_fraction": .125}))
    assert {key: result[key] for key in expected} == expected


@pytest.mark.parametrize("which", ["full_cad", "dense"])
def test_geometry_only_clouds_must_be_finite_nonempty_three_dimensional(which):
    full, feature, dense, R, t, C = _score_fixture()
    kwargs = {"full_cad": full, "dense": dense}
    kwargs[which] = np.array([[np.nan, 0., 0.]])
    with pytest.raises(ValueError, match="finite"):
        paired_full_geometry_score(feature, dense, R, t, .2, C, **kwargs)
