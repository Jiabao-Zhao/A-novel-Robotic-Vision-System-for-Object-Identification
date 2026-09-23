import numpy as np

from scripts.run_bop_sam_depth_comparison import mask_nms, variant_ids


def detection(oid, category=1, score=.8):
    return {"object_id": oid, "category_id": category, "score": score}


def test_duplicate_sources_keep_higher_cad_score_without_source_preference():
    mask = np.ones((4, 4), dtype=bool)
    masks = {"sam_001": mask, "depth_001": mask}
    for winner, loser in (("sam_001", "depth_001"), ("depth_001", "sam_001")):
        kept, suppressed = mask_nms([detection(loser), detection(winner, score=.9)], masks)
        assert [r["object_id"] for r in kept] == [winner]
        assert suppressed == [{"object_id": loser, "retained_object_id": winner}]


def test_repeated_instances_and_different_cad_predictions_are_not_suppressed():
    left = np.array([[True, False], [True, False]])
    masks = {"sam_001": left, "sam_002": ~left, "depth_001": left}
    rows = [detection("sam_001"), detection("sam_002"), detection("depth_001", category=2)]
    kept, suppressed = mask_nms(rows, masks)
    assert len(kept) == 3
    assert suppressed == []


def test_exact_iou_boundary_is_retained_and_score_ties_are_deterministic():
    masks = {"sam_001": np.ones((1, 2), dtype=bool),
             "depth_001": np.array([[True, False]])}
    rows = [detection("sam_001"), detection("depth_001")]
    assert len(mask_nms(rows, masks)[0]) == 2  # IoU exactly .5, rule is > .5.
    masks["depth_001"] = masks["sam_001"]
    assert mask_nms(rows, masks) == mask_nms(rows[::-1], masks)


def test_empty_sam_frames_stay_empty_and_hybrid_contains_both_sources():
    masks = {"depth_001": np.ones((2, 2), dtype=bool)}
    assert variant_ids(masks, "sam_leading") == []
    assert mask_nms([], masks) == ([], [])
    masks["sam_001"] = masks["depth_001"]
    assert variant_ids(masks, "depth_leading") == ["depth_001"]
    assert variant_ids(masks, "sam_leading") == ["sam_001"]
    assert variant_ids(masks, "hybrid") == ["depth_001", "sam_001"]
