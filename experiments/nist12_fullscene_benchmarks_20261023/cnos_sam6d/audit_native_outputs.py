"""Replay final CNOS/SAM-6D artifact invariants without observation labels."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import HERE as ROOT, SOURCES, digest, path, read, save


def bbox_iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    wh = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    intersection = float(np.prod(wh))
    return intersection / (float(np.prod(a[2:] - a[:2])) +
                           float(np.prod(b[2:] - b[:2])) - intersection)


def main():
    manifest = read(ROOT / 'inputs.json')
    hashes = {}

    def checked(filename, expected=None):
        filename = Path(filename)
        actual = digest(filename)
        if expected is not None:
            assert actual == expected, str(filename)
        hashes[str(filename)] = actual
        return actual

    for filename, expected in manifest['input_sha256'].items():
        checked(path(filename), expected)
    checked(ROOT / 'inputs.json')
    bank = read(ROOT / 'templates42/completion.json')
    for filename, expected in bank['sha256'].items():
        checked(path(filename), expected)
    report = dict(passed=True, no_ground_truth_read=True,
                  inference_or_model_calls=False, scenes=[], score_fusion_max_error=0.0)
    for native in ('cnos', 'sam6d'):
        protocol = read(ROOT / native / 'protocol.json')
        checked(ROOT / native / 'protocol.json')
        checked(HERE / 'run_native.py', protocol['runner_sha256'])
        checked(ROOT / 'inputs.json', protocol['inputs_sha256'])
        checked(ROOT / 'templates42/completion.json', protocol['reference_bank_sha256'])
        receipt = read(SOURCES / f'{native}_provenance.json')
        checked(SOURCES / f'{native}_provenance.json', protocol['native_source_receipt_sha256'])
        for filename, expected in receipt['source_sha256'].items():
            checked(SOURCES / native / filename, expected)
        outputs = ['cnos'] if native == 'cnos' else ['sam6d_rgb', 'sam6d_rgbd']
        for output in outputs:
            completion = read(ROOT / output / 'completion.json')
            assert completion['passed'] and completion['scenes'] == 8
            checked(ROOT / output / 'completion.json')
            checked(ROOT / native / 'reference_features.pt', completion['reference_features_sha256'])
        for scene in manifest['scenes']:
            scene_id = scene['scene_id']
            raw_dir = ROOT / native / 'raw_proposals' / scene_id
            evidence = read(raw_dir / 'completion.json')
            checked(raw_dir / 'completion.json')
            checked(raw_dir / 'soft_masks.npz', evidence['soft_masks_sha256'])
            checked(raw_dir / 'proposals.json', evidence['proposals_sha256'])
            checked(path(scene['rgb']), evidence['input_rgb_sha256'])
            raw_rows = read(raw_dir / 'proposals.json')
            raw_masks = np.load(raw_dir / 'soft_masks.npz')['masks']
            assert raw_masks.dtype == np.float16 and raw_masks.shape[0] == len(raw_rows)
            assert list(raw_masks.shape[1:]) == evidence['native_resolution']
            assert len(raw_rows) == evidence['proposals']
            h, w = raw_masks.shape[1:]
            boxes = np.asarray([r['bbox_xyxy'] for r in raw_rows], dtype=np.int64)
            areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1) / (h * w)
            mask_areas = np.asarray([m.astype(np.float32).sum(dtype=np.float32) / (h*w)
                                     for m in raw_masks])
            kept = np.flatnonzero((areas > .05**2) & (mask_areas > .0003)).tolist()
            filters = read(ROOT / native / scene_id / 'proposal_filter.json')
            checked(ROOT / native / scene_id / 'proposal_filter.json')
            assert kept == filters['retained_proposal_ids']
            assert len(kept) == filters['retained_after_native_size_filter']
            assert len(raw_rows) == filters['raw_proposals']
            status = dict(method=native, scene_id=scene_id, raw_proposals=len(raw_rows),
                          after_native_size_filter=len(kept), retained={})
            for output in outputs:
                directory = ROOT / output / scene_id
                completion = read(directory / 'completion.json')
                checked(directory / 'completion.json')
                checked(directory / 'masks.npz', completion['masks_sha256'])
                checked(directory / 'detections.json', completion['detections_sha256'])
                rows = read(directory / 'detections.json')
                masks = np.load(directory / 'masks.npz')['masks']
                assert masks.dtype == bool and masks.shape == (len(rows), h, w)
                assert len(rows) == completion['retained_detections']
                assert len({r['proposal_id'] for r in rows}) == len(rows)
                assert completion['no_ground_truth_read'] and completion['own_proposals']
                for i, row in enumerate(rows):
                    pid = row['proposal_id']
                    assert pid in kept and row['mask_index'] == i
                    assert 1 <= row['object_id'] <= 12 and np.isfinite(row['score'])
                    np.testing.assert_array_equal(masks[i], raw_masks[pid] > 0)
                    np.testing.assert_array_equal(row['bbox_xyxy'], boxes[pid])
                    c = row['components']
                    if output == 'cnos':
                        values = np.asarray(c['class_scores'])
                        assert values.shape == (12,) and np.isfinite(values).all()
                        assert row['object_id'] == int(values.argmax()) + 1
                        assert row['score'] == float(values.max()) and row['score'] > .15
                    else:
                        s, a, g, r = [np.float32(c[key]) for key in
                            ('semantic', 'appearance', 'geometry_bbox_iou', 'patch_coverage')]
                        assert np.isfinite([s, a, g, r]).all()
                        assert s > .2 and 0 <= g <= 1 and 0 <= r <= 1
                        assert 0 <= c['best_template_ordinal'] < 42
                        expected = (s+a)/np.float32(2) if output == 'sam6d_rgb' else (
                            s+a+g*r)/(np.float32(2)+r)
                        error = abs(float(expected)-row['score'])
                        report['score_fusion_max_error'] = max(report['score_fusion_max_error'], error)
                        assert error < 2e-7
                for i, a in enumerate(rows):
                    for b in rows[i+1:]:
                        if a['object_id'] == b['object_id']:
                            assert bbox_iou(a['bbox_xyxy'], b['bbox_xyxy']) <= .2500001
                status['retained'][output] = len(rows)
                del masks
            report['scenes'].append(status)
            del raw_masks
    for filename, expected in hashes.items():
        assert digest(filename) == expected, f'Changed during audit: {filename}'
    report['consumed_sha256'] = hashes
    report['audit_source_sha256'] = digest(__file__)
    save(HERE / 'native_output_audit.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('consumed_sha256', 'scenes')}, indent=2))


if __name__ == '__main__':
    main()
