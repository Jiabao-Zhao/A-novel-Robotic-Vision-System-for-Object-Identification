import numpy as np

from scripts.run_wrist_branch_order import combine, select_views, prune_candidates, pair_result, METHODS


def rows_for_selection():
    return [{"cad_id": "cad", "object_id": "object", "view_index_1based": i,
             "cls": 1. - i / 100, "G": .8, "P": 1. - i / 100,
             "H": None if i <= 5 else .9,
             "pose_status": "table_penetration" if i <= 5 else "assessable"}
            for i in range(1, 9)]


def test_table_before_shortlist_recovers_views_without_changing_pose():
    rows = rows_for_selection()
    raw = pair_result(rows, METHODS["global5_then_geometry"])
    filtered = pair_result(rows, METHODS["table_then_global5"])
    assert raw["selected_views"] == [1, 2, 3, 4, 5] and raw["score"] is None
    assert filtered["selected_views"] == [6, 7, 8] and filtered["score"] is not None
    assert rows[0]["pose_status"] == "table_penetration"


def test_same_hypotheses_make_fusion_invariant_to_evaluation_order():
    row = {"G": .8, "P": .4, "H": .6}
    assert np.isclose(combine(row, "GPH"), .6)
    assert np.isclose(combine(row, "GPH"), combine(row, "HPG"))
    assert np.isclose(combine(row, "PH"), (.4 + 2 * .6) / 3)
    assert combine({**row, "H": None}, "GPH") is None
    assert combine({**row, "H": None}, "GP") is not None


def test_geometry_selection_uses_geometry_not_cls_or_labels():
    rows = rows_for_selection()
    rows[6]["H"] = .99
    assert select_views(rows, "geometry5")[0]["view_index_1based"] == 7
    for row in rows:
        row["cad_id"] = "different_label"
    assert select_views(rows, "geometry5")[0]["view_index_1based"] == 7


def test_geometry_candidate_cutoff_preserves_color_ambiguity_ties():
    pairs = [{"cad_id": cad, "object_id": "object", "geometry_max": geometry, "score": score}
             for cad, geometry, score in (("a", .9, .2), ("red", .8, .7), ("blue", .8, .9), ("d", .1, .95))]
    pruned, retained = prune_candidates(pairs, "object_id", "cad_id", 2)
    assert retained["object"] == ["a", "blue", "red"]
    assert pruned[-1]["score"] is None
    assert pairs[-1]["score"] == .95


def test_empty_candidate_geometry_produces_no_invented_fallback():
    pruned, retained = prune_candidates([
        {"cad_id": "cad", "object_id": "object", "geometry_max": None, "score": .8}],
        "object_id", "cad_id", 3)
    assert retained["object"] == [] and pruned[0]["score"] is None
