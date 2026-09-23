from unittest.mock import Mock

import numpy as np
import pytest
import torch

from scripts import local_sam3 as sam3


def test_native_masks_exclusive_boxes_and_truthful_scores():
    mask = np.zeros((7, 11), dtype=bool)
    mask[2:5, 6:10] = True
    rows = sam3.proposal_records({"masks": [mask], "scores": [.9375]}, mask.shape, .95)
    assert len(rows) == 1
    assert np.array_equal(rows[0]["segmentation"], mask)
    assert rows[0]["bbox"] == [6, 2, 4, 3]
    assert rows[0]["predicted_iou"] == .9375
    assert rows[0]["stability_score"] is None
    assert rows[0]["stability_threshold"] == .95


def test_empty_masks_and_dimension_mismatch():
    assert sam3.proposal_records({"masks": [], "scores": []}, (7, 11), .85) == []
    assert sam3.proposal_records({"masks": [np.zeros((7, 11))], "scores": [.9]}, (7, 11), .85) == []
    with pytest.raises(ValueError, match="mask shape"):
        sam3.proposal_records({"masks": [np.ones((11, 7))], "scores": [.9]}, (7, 11), .85)


def test_generator_uses_text_free_grid_and_native_image():
    generator = object.__new__(sam3.Sam3AutomaticMasks)
    generator.pipeline = Mock(return_value={"masks": [], "scores": []})
    rgb = np.zeros((7, 11, 3), dtype=np.uint8)
    assert generator.generate(rgb, stability_score_thresh=.95) == []
    args, kwargs = generator.pipeline.call_args
    assert args[0].size == (11, 7)
    assert kwargs == dict(points_per_crop=32, points_per_batch=4, pred_iou_thresh=.88,
                          stability_score_thresh=.95, crops_nms_thresh=.7, crops_n_layers=0)
    with pytest.raises(ValueError, match="divide evenly"):
        generator.generate(rgb, points_per_batch=3)
    with pytest.raises(ValueError, match="uint8"):
        generator.generate(rgb.astype(float))


def test_loader_rejects_missing_weights_before_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    loader = Mock(return_value=(Mock(), {"missing_keys": {"mask_decoder.weight"}}))
    monkeypatch.setattr(sam3.Sam3TrackerModel, "from_pretrained", loader)
    with pytest.raises(RuntimeError, match="Incomplete SAM3 checkpoint"):
        sam3.Sam3AutomaticMasks()
    assert loader.call_args.kwargs["local_files_only"] is True
    assert loader.call_args.kwargs["key_mapping"] == sam3.TRACKER_KEYS


def test_nms_scores_are_fp32(monkeypatch):
    outputs = [{"iou_scores": torch.tensor([.9], dtype=torch.bfloat16)}]
    parent = Mock(return_value="result")
    monkeypatch.setattr(sam3.MaskGenerationPipeline, "postprocess", parent)
    pipeline = object.__new__(sam3._MaskPipeline)
    assert pipeline.postprocess(outputs, crops_nms_thresh=.7) == "result"
    assert outputs[0]["iou_scores"].dtype == torch.float32
    parent.assert_called_once_with(outputs, crops_nms_thresh=.7)
