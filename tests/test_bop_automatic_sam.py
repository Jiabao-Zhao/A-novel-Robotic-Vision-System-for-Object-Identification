from scripts.run_bop_automatic_sam_comparison import align_patch_top_five
from scripts.run_bop_automatic_sam_masks import restore_proposal
from scripts.run_wrist_branch_order import select_views
import numpy as np
import pytest
import torch


@pytest.mark.parametrize("valid", [set(), {1, 8, 12}, set(range(2, 42))])
def test_lazy_alignment_matches_exhaustive_selection_with_ties_and_rejections(valid):
    old = [{"view_index_1based": i, "P": (i % 7) / 7,
            "pose_status": "assessable" if i in valid else "table_penetration"}
           for i in range(1, 43)]
    rows = [{**r, "pose_status": "not_visited_below_patch5"} for r in old[::-1]]
    calls = []

    def align(row):
        calls.append(row["view_index_1based"])
        return {"pose_status": "assessable" if row["view_index_1based"] in valid else "table_penetration"}

    chosen = align_patch_top_five(rows, align)
    assert [r["view_index_1based"] for r in chosen] == [r["view_index_1based"] for r in select_views(old, "table_patch5")]
    assert len(set(calls) & valid) == min(5, len(valid))
    if len(valid) < 5:
        assert len(calls) == 42


def test_sam6d_size_filter_uses_strict_relative_area_boundaries():
    mask = torch.zeros((1, 100, 100), dtype=torch.bool)
    mask[0, :2, :2] = True  # Four pixels; minimum is strictly greater than three.
    prediction = {"segmentation": mask[0].numpy(), "bbox": [0, 0, 5, 5]}
    assert not restore_proposal(prediction, (100, 100), (100, 100))[1]
    prediction["bbox"] = [0, 0, 6, 5]
    assert restore_proposal(prediction, (100, 100), (100, 100))[1]
    mask[0, 0, 0] = False
    prediction["segmentation"] = mask[0].numpy()
    assert not restore_proposal(prediction, (100, 100), (100, 100))[1]


def test_sam6d_binary_export_retains_positive_interpolated_edge_pixels():
    mask = torch.zeros((1, 4, 4), dtype=torch.bool)
    mask[0, 1:3, 1:3] = True
    prediction = {"segmentation": mask[0].numpy(), "bbox": [1, 1, 1, 1]}
    restored, _, _ = restore_proposal(prediction, (8, 8), (4, 4))
    assert restored[1, 1]  # Bilinear value 1/16: retained by upstream >0 export.
    assert not restored[0, 0]
    assert restored.dtype == np.bool_


def test_parallel_parity_does_not_claim_success_without_regions(tmp_path, monkeypatch):
    from scripts import run_bop_automatic_sam_comparison as comparison

    monkeypatch.setattr(comparison, "OUTPUT", tmp_path)
    comparison.verify_parallel_parity([], ({}, {}, [], {}), None)
    report = comparison.read_json(tmp_path / "parallel_parity.json")
    assert report["exact_numeric_parity"] is None
    assert report["status"] == "no_regions_to_check"
    assert report["regions"] == []
