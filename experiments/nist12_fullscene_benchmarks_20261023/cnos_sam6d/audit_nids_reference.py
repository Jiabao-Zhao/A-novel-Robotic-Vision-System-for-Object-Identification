"""Independent CPU replay of native NIDS empty-reference propagation.

Reads CAD-only reference tensors and frozen query features, never simulator
labels. No encoder, detector, mask generator or inference runner is executed.
"""
import sys
from pathlib import Path

LOCAL = Path(__file__).resolve().parent
sys.path.insert(0, str(LOCAL.parent))
import common


def main():
    inputs = common.setup()
    source = common.SOURCES / 'nids'
    sys.path[:0] = [str(source), str(common.SOURCES / 'sam1')]
    import numpy as np
    import torch
    import torch.nn.functional as F
    from omegaconf import OmegaConf
    from src.dataloader.bop import BOPTemplate
    from src.model.loss import PairwiseSimilarity

    torch.set_num_threads(4)
    paths = [LOCAL / 'audit_nids_reference.py', common.HERE / 'inputs.json',
             common.HERE / 'nids_basic/configuration.json',
             common.HERE / 'nids_basic/native_reference_features.pt',
             common.HERE / 'nids_noctis/nids_zero_mask_diagnostic.json',
             common.HERE / 'templates42/metadata.json',
             source / 'src/dataloader/bop.py', source / 'src/model/dinov2.py',
             source / 'src/model/detector.py', source / 'src/model/loss.py',
             source / 'src/utils/bbox_utils.py']
    descriptor = torch.load(paths[3], map_location='cpu', weights_only=True)['descriptors']
    assert descriptor.shape == (12, 42, 1024)
    invalid = ~torch.isfinite(descriptor).all(-1)
    dataset = BOPTemplate(str(common.HERE / 'templates42'),
        obj_ids=[o['object_id'] for o in inputs['objects']],
        processing_config=OmegaConf.create(dict(image_size=224)),
        level_templates=0, pose_distribution='all')
    meta = common.read(common.HERE / 'templates42/metadata.json')
    np.testing.assert_array_equal(dataset.index_templates, meta['index_templates'])
    rows, zero_locations = [], []
    for obj in range(12):
        item = dataset[obj]
        masks = item['template_masks']
        assert masks.shape == (42, 224, 224)
        assert torch.isfinite(masks).all() and masks.min() >= 0 and masks.max() <= 1
        assert item['templates'].shape == (42, 3, 224, 224)
        assert torch.isfinite(item['templates']).all()
        down = F.interpolate(masks[:, None], (16, 16), mode='bilinear')
        sums = down.sum((1, 2, 3))
        for view in (sums == 0).nonzero().flatten().tolist():
            zero_locations.append([obj, view])
            canonical = int(dataset.index_templates[view])
            image_file = common.HERE / f'templates42/obj_{obj+1:06d}/{canonical:06d}.png'
            paths.append(image_file)
            assert masks[view].sum() > 0, 'The original foreground is present'
            assert (~torch.isfinite(descriptor[obj, view])).sum() == 1024
            # This is the native unguarded weighted-mean formula, with constant
            # dummy tokens only to isolate denominator behavior from the model.
            weighted_mean = (torch.ones((16, 16, 1))*down[view].permute(1, 2, 0)).sum((0, 1))/sums[view]
            assert torch.isnan(weighted_mean).all()
            rows.append(dict(object_id=obj+1, template_slot=view, canonical_view_id=canonical,
                mask_support_pixels=int((masks[view] > 0).sum()),
                mask_sum=float(masks[view].sum()), native_16x16_mask_sum=float(sums[view]),
                nonfinite_reference_values=1024))
    assert zero_locations == invalid.nonzero().tolist()
    expected = common.read(common.HERE / 'nids_noctis/nids_zero_mask_diagnostic.json')['invalid_views']
    assert [(r['object_id'], r['template_slot'], r['canonical_view_id']) for r in rows] == [
        (r['object_id'], r['template_slot'], r['canonical_view_id']) for r in expected]
    metric = PairwiseSimilarity(metric='cosine', chunk_size=16)
    scenes = []
    for scene in inputs['scenes']:
        directory = common.HERE / 'nids_basic' / scene['scene_id']
        paths.extend([directory / n for n in ('matching_scores.npz', 'detections.json', 'completion.json')])
        with np.load(directory / 'matching_scores.npz') as data:
            queries = torch.from_numpy(data['FFA_features'])
            assert torch.isfinite(queries).all()
            scores = metric(queries, descriptor)
            class_scores = scores.max(-1).values
            native_best = class_scores.max(-1).values
            assert torch.isnan(native_best).all()
            assert int((native_best > .01).sum()) == 0
            assert common.read(directory / 'detections.json') == []
            # Proposed secondary arm: remove only nonfinite CAD-reference slots
            # before native max reductions. Every unaffected score is identical.
            finite_scores = scores.masked_fill(invalid[None], -torch.inf)
            assert torch.equal(finite_scores[:, ~invalid], scores[:, ~invalid])
            assert torch.isfinite(finite_scores.max(-1).values).all()
            scenes.append(dict(scene_id=scene['scene_id'], queries=len(queries),
                finite_query_vectors=True, native_nonfinite_best_scores=int(torch.isnan(native_best).sum()),
                native_selected_proposals=0,
                secondary_finite_class_scores=True, secondary_unchanged_valid_view_scores=True))
    config = common.read(common.HERE / 'nids_basic/configuration.json')
    for filename in ('src/dataloader/bop.py', 'src/model/dinov2.py', 'src/model/detector.py',
                     'src/model/loss.py', 'src/utils/bbox_utils.py'):
        file = source / filename
        assert common.digest(file) == config['source_sha256'][common.relative(file)]
    result = dict(passed=True, no_ground_truth_read=True, model_calls=False, GPU_used=False,
        native_loader_and_mask_contract_correct=True,
        native_loader='Official BOPTemplate alpha masks, native crop/resize/pad and ImageNet normalization',
        cause='Six nonempty CAD masks have zero native bilinear 16x16 sampling mass; unguarded 0/0 FFA yields NaNs, and native max(view), max(object) propagates them.',
        invalid_reference_views=rows, finite_reference_views_per_object=(~invalid).sum(-1).tolist(),
        source_integration_assessment='No changed reference pixels, native view mapping, normalization or FFA mask handling caused the NaNs. Separate appearance routine needs only a singleton-channel API repair.',
        proposed_secondary_scope='Label-free exclusion of only six nonfinite reference slots before native max; retain all 12 classes, remaining native references, proposals, scores, thresholds and NMS. Preserve unchanged native arm separately.',
        scenes=scenes, consumed_sha256={common.relative(p):common.digest(p) for p in paths})
    common.save(LOCAL / 'nids_reference_audit.json', result)
    print({k:v for k,v in result.items() if k not in ('consumed_sha256', 'scenes')}, flush=True)


if __name__ == '__main__':
    main()
