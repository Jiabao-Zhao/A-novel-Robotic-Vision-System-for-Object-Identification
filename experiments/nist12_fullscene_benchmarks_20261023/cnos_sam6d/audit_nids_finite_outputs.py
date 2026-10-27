"""Replay frozen NIDS secondary outputs on CPU; never read audit identities."""
from pathlib import Path
import sys
from types import SimpleNamespace

LOCAL = Path(__file__).resolve().parent
sys.path[:0] = [str(LOCAL.parent), str(LOCAL.parent / 'nids_noctis')]
import common


def main():
    inputs = common.setup()
    from compat import import_native
    native = import_native('nids_basic', common)
    import numpy as np
    import torch
    from src.model.loss import PairwiseSimilarity
    from omegaconf import OmegaConf

    original = common.HERE / 'nids_basic'
    secondary = common.HERE / 'nids_basic_finite_templates'
    consumed = {}

    def checked(file, expected=None):
        actual = common.digest(file)
        if expected is not None:
            assert actual == expected, str(file)
        consumed[common.relative(file)] = actual
        return actual

    config = common.read(secondary / 'configuration.json')
    assert config['secondary_control'] and config['scene_labels_read'] is False
    checked(secondary / 'configuration.json')
    checked(original / 'configuration.json')
    checked(common.HERE / 'inputs.json')
    # Verify the full predeclared source/input chain, without mutating receipts.
    for filename, expected in config['source_sha256'].items():
        checked(common.path(filename), expected)
    reference_file = original / 'native_reference_features.pt'
    checked(reference_file)
    reference = torch.load(reference_file, map_location='cpu', weights_only=True)
    valid = torch.isfinite(reference['descriptors']).all(-1)
    assert valid.shape == (12, 42) and valid.any(-1).all()
    assert (~valid).nonzero().tolist() == config['excluded_object_template_slots']
    native_metric = PairwiseSimilarity(metric='cosine', chunk_size=16)
    metric_checks = []

    def guarded_metric(query, refs):
        unmodified = native_metric(query, refs)
        expanded = valid[None].expand_as(unmodified)
        assert torch.isfinite(unmodified[expanded]).all()
        assert (~torch.isfinite(unmodified[~expanded])).all()
        guarded = unmodified.masked_fill(~expanded, -torch.inf)
        assert torch.equal(guarded[expanded], unmodified[expanded])
        metric_checks.append(dict(queries=len(query), finite_scores_unchanged_exactly=True,
            excluded_slots=int((~valid).sum())))
        return guarded

    helper = SimpleNamespace(ref_data=reference,
        matching_config=SimpleNamespace(metric=guarded_metric, aggregation_function='max', confidence_thresh=.01))
    helper.best_template_pose = lambda scores, identities: native.detector.NIDSNET.best_template_pose(
        helper, scores, identities)
    filters = OmegaConf.create(dict(min_box_size=.05, min_mask_size=.0003))
    scenes, comparisons = [], 0

    def same(a, b):
        nonlocal comparisons
        aa = a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)
        bb = b.detach().cpu().numpy() if torch.is_tensor(b) else np.asarray(b)
        np.testing.assert_array_equal(aa, bb)
        comparisons += int(aa.size)

    for scene in inputs['scenes']:
        scene_id = scene['scene_id']
        source, output = original / scene_id, secondary / scene_id
        completion = common.read(output / 'completion.json')
        checked(output / 'completion.json')
        for name, key in [('raw_proposals.npz', 'raw_proposals_sha256'),
                          ('matching_scores.npz', 'matching_scores_sha256'),
                          ('proposal_filter.json', 'proposal_filter_sha256'),
                          ('detections.json', 'detections_sha256'), ('masks.npz', 'masks_sha256')]:
            checked(output / name, completion[key])
        for filename in ('raw_proposals.npz', 'raw_proposals.json', 'proposal_completion.json', 'proposal_filter.json'):
            assert checked(output / filename) == checked(source / filename)
        checked(source / 'matching_scores.npz')
        checked(source / 'detections.json')
        assert common.read(source / 'detections.json') == []
        check = common.read(output / 'finite_template_check.json')
        checked(output / 'finite_template_check.json')
        assert check['saved_original_FFA_reused'] and check['recomputed_FFA_exact_equal']
        assert check['valid_scores_unchanged_exactly'] and check['final_scores_finite']
        assert check['appearance_features_finite'] and not check['labels_read']
        with np.load(source / 'matching_scores.npz') as data:
            original_ffa = data['FFA_features']
            original_proposals = data['proposal_ids']
        with np.load(output / 'matching_scores.npz') as data:
            saved = dict(data)
        same(saved['FFA_features'], original_ffa)
        same(saved['proposal_ids'], original_proposals)
        ffa = torch.from_numpy(saved['FFA_features'])
        appearance = torch.from_numpy(saved['appearance_features'])
        assert torch.isfinite(ffa).all() and torch.isfinite(appearance).all()
        raw = dict(np.load(source / 'raw_proposals.npz'))
        raw_count = len(raw['boxes'])
        raw['proposal_ids'] = np.arange(raw_count)
        detections = native.utils.Detections({key: torch.from_numpy(value) for key, value in raw.items()})
        detections.remove_very_small_detections(filters)
        same(detections.proposal_ids, original_proposals)
        declared = common.read(output / 'proposal_filter.json')
        assert declared['retained_proposal_ids'] == detections.proposal_ids.tolist()
        assert completion['raw_proposals'] == raw_count
        assert completion['proposals_after_native_size_filter'] == len(detections)
        selected, identities, ffa_scores, views = native.detector.NIDSNET.find_matched_proposals(helper, ffa)
        detections.filter(selected)
        same(detections.proposal_ids, saved['selected_proposal_ids'])
        same(identities, saved['selected_object_indices'])
        same(views, saved['best_template_indices'])
        same(ffa_scores, saved['ffa_score'])
        app_scores, _ = native.detector.NIDSNET.compute_appearance_score(
            helper, views, identities, appearance[selected])
        same(app_scores, saved['appearance_score'])
        final_scores = (ffa_scores + app_scores) / 2
        assert torch.isfinite(final_scores).all()
        detections.add_attribute('object_ids', identities)
        detections.add_attribute('scores', final_scores)
        if len(detections):
            detections.apply_nms_per_object_id(nms_thresh=.25)
        rows = common.read(output / 'detections.json')
        masks = np.load(output / 'masks.npz')['masks']
        assert len(rows) == len(detections) == completion['retained_detections']
        assert masks.dtype == bool and masks.shape == tuple(detections.masks.shape)
        same(masks, detections.masks > 0)
        for i, row in enumerate(rows):
            assert row['mask_index'] == i
            assert row['proposal_id'] == int(detections.proposal_ids[i])
            assert row['object_id'] == int(detections.object_ids[i]) + 1
            assert row['score'] == float(detections.scores[i])
            same(row['bbox_xyxy'], detections.boxes[i])
        scenes.append(dict(scene_id=scene_id, raw_proposals=raw_count,
            native_size_filtered=len(original_proposals), pre_NMS_selected=len(selected),
            retained=len(rows), exact_cached_FFA_reuse=True, exact_native_semantic_scores=True,
            exact_native_appearance_scores=True, exact_native_class_and_view_indices=True,
            exact_native_NMS=True, exact_proposal_box_mask_mapping=True))
        del raw, detections, masks, saved, appearance
    for filename, expected in consumed.items():
        assert common.digest(common.path(filename)) == expected, f'Changed during replay: {filename}'
    result = dict(passed=True, no_ground_truth_read=True, GPU_used=False, model_calls=False,
        original_primary_unchanged=True, replay='Official CPU native FFA matching, appearance scoring, filters and per-class NMS',
        maximum_score_error=0.0, exact_array_values_checked=comparisons,
        invalid_reference_slots=(~valid).nonzero().tolist(), metric_checks=metric_checks,
        scenes=scenes, consumed_sha256=consumed, audit_source_sha256=common.digest(__file__))
    common.save(LOCAL / 'nids_finite_output_audit.json', result)
    print({k:v for k,v in result.items() if k not in ('consumed_sha256', 'metric_checks')}, flush=True)


if __name__ == '__main__':
    main()
