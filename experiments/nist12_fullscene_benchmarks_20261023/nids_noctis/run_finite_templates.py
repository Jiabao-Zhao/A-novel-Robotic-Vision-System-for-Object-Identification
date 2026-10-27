"""Secondary CAD-only invalid-template guard; original native run is immutable."""

import gc
from pathlib import Path
import shutil
import sys
from time import perf_counter

import numpy as np
import torch

LOCAL = Path(__file__).resolve().parent
sys.path.insert(0, str(LOCAL.parent))
import common
from compat import import_native
import run_native as runner

METHOD = "nids_basic_finite_templates"


class ValidTemplateMetric(torch.nn.Module):
    def __init__(self, native_metric, valid):
        super().__init__()
        self.native_metric = native_metric
        self.valid = valid
        self.receipts = []

    def forward(self, query, reference):
        raw = self.native_metric(query, reference)
        valid = self.valid.to(raw.device)[None].expand_as(raw)
        assert torch.isfinite(raw[valid]).all()
        assert (~torch.isfinite(raw[~valid])).all()
        guarded = raw.masked_fill(~valid, -torch.inf)
        assert torch.equal(raw[valid], guarded[valid])
        self.receipts.append(dict(query_count=len(query), excluded_template_slots=int((~self.valid).sum()),
            excluded_score_entries=int((~valid).sum()), valid_scores_unchanged_exactly=True,
            valid_scores_finite=True, excluded_view_score="negative infinity before native max"))
        return guarded


def prepare(inputs):
    original = common.HERE / "nids_basic"
    out = common.HERE / METHOD
    out.mkdir(exist_ok=True)
    refs = torch.load(original / "native_reference_features.pt", map_location="cpu", weights_only=True)
    valid = torch.isfinite(refs["descriptors"]).all(-1)
    invalid = (~valid).nonzero().tolist()
    expected = [[5, slot] for slot in (16, 19, 20, 21, 24, 25)]
    assert invalid == expected
    assert valid.any(-1).all(), "Every CAD class must retain a valid template"
    paths = [Path(__file__), LOCAL / "test_finite_templates.py", LOCAL / "finite_templates_CPU_tests.json",
             LOCAL / "nids_zero_mask_diagnostic.json", common.HERE / "common.py",
             original / "configuration.json", original / "native_reference_features.pt", original / "reference_completion.json"]
    for scene in inputs["scenes"]:
        directory = original / scene["scene_id"]
        paths += [directory / name for name in ("raw_proposals.npz", "raw_proposals.json",
            "proposal_completion.json", "proposal_filter.json", "matching_scores.npz", "completion.json")]
    config = common.read(original / "configuration.json")
    config.update(method=METHOD, secondary_control=True,
        adaptation="Exclude only six CAD-only nonfinite FFA template scores before native max; preserve every finite score, all classes, thresholds, filters and NMS.",
        excluded_object_template_slots=invalid, scene_labels_read=False,
        query_reuse="Saved original FFA vectors reused exactly. Native appearance vectors recomputed from unchanged own proposals/images because original zero-selection run did not serialize them; recomputed FFA must equal saved FFA exactly.")
    config["source_sha256"].update({common.relative(p):common.digest(p) for p in paths})
    runner.write_once(out / "configuration.json", config)
    for name, expected_hash in config["source_sha256"].items():
        assert common.digest(common.path(name)) == expected_hash, name
    return out, original, refs, valid


def main():
    inputs = common.setup()
    directory, original, refs, valid = prepare(inputs)
    native = import_native("nids_basic", common)
    descriptor, dataset, _ = runner.descriptor_and_templates("nids_basic", native, inputs)
    matcher = runner.build_matcher("nids_basic", native, descriptor, dataset, directory)
    matcher.ref_data = refs
    metric = ValidTemplateMetric(matcher.matching_config.metric, valid)
    matcher.matching_config.metric = metric
    for scene in inputs["scenes"]:
        out = directory / scene["scene_id"]
        out.mkdir(exist_ok=True)
        if (out / "completion.json").exists():
            continue
        source = original / scene["scene_id"]
        for name in ("raw_proposals.npz", "raw_proposals.json", "proposal_completion.json"):
            if not (out / name).exists():
                shutil.copyfile(source / name, out / name)
            assert common.digest(out / name) == common.digest(source / name)
        raw = dict(np.load(source / "raw_proposals.npz"))
        raw_count = len(raw["boxes"])
        raw["proposal_ids"] = np.arange(raw_count)
        detections = native.utils.Detections({key:torch.from_numpy(value) for key,value in raw.items()})
        detections.remove_very_small_detections(matcher.post_processing_config.mask_post_processing)
        filtered_count = len(detections)
        filter_receipt = dict(retained_proposal_ids=detections.proposal_ids.tolist(), raw_proposals=raw_count,
                              retained_after_native_size_filter=filtered_count)
        assert common.read(source / "proposal_filter.json") == filter_receipt
        runner.write_once(out / "proposal_filter.json", filter_receipt)
        saved = dict(np.load(source / "matching_scores.npz"))
        np.testing.assert_array_equal(saved["proposal_ids"], detections.proposal_ids.numpy())
        ffa = torch.from_numpy(saved["FFA_features"])
        assert torch.isfinite(ffa).all()
        started = perf_counter()
        recomputed_ffa, appearance, _ = runner.query_features("nids_basic", native, descriptor, runner.native_rgb(scene), detections)
        assert torch.equal(ffa, recomputed_ffa), "Saved and recomputed FFA must match exactly"
        with torch.inference_mode():
            selected, identities, ffa_scores, best_views = matcher.find_matched_proposals(ffa)
            detail = dict(proposal_ids=detections.proposal_ids.numpy(), FFA_features=saved["FFA_features"],
                          appearance_features=appearance.numpy())
            detections.filter(selected)
            app_scores, _ = matcher.compute_appearance_score(best_views, identities, appearance[selected])
            scores = (ffa_scores + app_scores) / 2
            assert torch.isfinite(scores).all()
            detail.update(selected_proposal_ids=detections.proposal_ids.numpy(), ffa_score=ffa_scores.numpy(),
                appearance_score=app_scores.numpy(), selected_object_indices=identities.numpy(), best_template_indices=best_views.numpy())
            detections.add_attribute("object_ids", identities)
            detections.add_attribute("scores", scores)
            if len(detections):
                detections.apply_nms_per_object_id(nms_thresh=.25)
        runner.write_once(out / "finite_template_check.json", dict(
            **metric.receipts[-1], saved_original_FFA_reused=True, recomputed_FFA_exact_equal=True,
            appearance_features_finite=bool(torch.isfinite(appearance).all()), final_scores_finite=True,
            original_query_sha256=common.digest(source / "matching_scores.npz"), labels_read=False))
        runner.export_detections(METHOD, native, inputs, out, detections, raw_count, filtered_count,
                                 perf_counter()-started, detail)
        print(f'{METHOD} {scene["scene_id"]}: {raw_count} raw -> {filtered_count} filtered -> {len(detections)} retained', flush=True)
        del raw, detections, recomputed_ffa, appearance
        gc.collect()
        torch.cuda.empty_cache()
    config = common.read(directory / "configuration.json")
    for name, expected in config["source_sha256"].items():
        assert common.digest(common.path(name)) == expected, name


if __name__ == "__main__":
    main()
