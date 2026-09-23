import numpy as np

from scripts.surface_verification import equal_penalty_scores, SIGMA_M
from scripts.run_wrist_shortlist_geometry import top_views, intersect_views, intersection_or_union, winning_view, rank_pairs


def test_perfect_and_distinct_support_penalties():
    d = np.ones((1, 4))
    observed = np.array([[True, True, False, False]])
    rendered = np.array([[1., np.inf, 1., 1.]])
    result = equal_penalty_scores(d, observed, rendered)
    assert result["p_obs"] == .25
    assert result["p_cad"] == .5
    assert result["p_depth"] == 0
    assert result["geometry_score"] == .75
    assert equal_penalty_scores(d, np.ones_like(d, bool), d)["geometry_score"] == 1


def test_depth_clips_each_pixel_before_averaging():
    d = np.ones((1, 2))
    result = equal_penalty_scores(d, np.ones_like(d, bool), d + [[0, 4 * SIGMA_M]])
    assert result["p_depth"] == .5
    assert np.isclose(result["geometry_score"], 5 / 6)
    # No camera-ray norm: this is the screenshot's optical-axis depth residual.
    half = equal_penalty_scores(d, np.ones_like(d, bool), d + SIGMA_M / 2)
    assert np.isclose(half["p_depth"], .5)


def test_no_overlap_is_undefined_not_a_perfect_depth_match():
    result = equal_penalty_scores(np.ones((1, 2)), np.array([[True, False]]), np.array([[np.inf, 1.]]))
    assert result["status"] == "no_overlap"
    assert result["p_obs"] == result["p_cad"] == .5
    assert result["p_depth"] is None and result["geometry_score"] is None
    assert equal_penalty_scores(np.zeros((1, 2)), np.ones((1, 2), bool), np.ones((1, 2)))["geometry_score"] is None


def test_external_occlusion_preserves_target_depth_error():
    result = equal_penalty_scores(np.ones((1, 2)), np.array([[True, False]]), np.full((1, 2), 1.02))
    assert result["externally_occluded_pixels"] == 1
    assert result["p_depth"] == 1
    assert np.isclose(result["geometry_score"], 2 / 3)


def test_top5_intersection_and_empty_intersection_union():
    global_views = top_views([.9, .8, .7, .6, .5, .4, .3, .2])
    patch_views = top_views([.1, .2, .8, .7, .6, .9, 1., 0.])
    assert global_views == [1, 2, 3, 4, 5]
    assert intersect_views(global_views, patch_views) == [3, 4, 5]
    assert intersection_or_union(global_views, patch_views) == [3, 4, 5]
    assert intersect_views([1, 2], [3, 4]) == []
    assert intersection_or_union([1, 2], [3, 4]) == [1, 2, 3, 4]
    assert top_views([1., 1., 0.], 2) == [1, 2]


def test_unavailable_views_never_win_and_ties_are_reported():
    rows = [{"view_index_1based": 1, "geometry_score": None},
            {"view_index_1based": 2, "geometry_score": .7}]
    assert winning_view(rows, "geometry_score")["view_index_1based"] == 2
    assert winning_view(rows[:1], "geometry_score") is None
    pairs = [{"cad_id": "cad", "object_id": oid, "geometry_score": score}
             for oid, score in (("a", None), ("b", .7), ("c", .7))]
    result = rank_pairs(pairs, {"cad": "a"}, "cad_id", "object_id", "geometry_score")
    target = result["targets"][0]
    assert result["correct"] == 0 and target["margin"] is None
    assert target["tied_top_choices"] == ["b", "c"]
    empty = rank_pairs(pairs[:1], {"cad": "a"}, "cad_id", "object_id", "geometry_score")
    assert empty["targets"][0]["prediction"] is None
