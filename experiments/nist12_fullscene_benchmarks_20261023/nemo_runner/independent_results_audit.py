"""Post-lock independent result replay. Run only after root authorizes GT audit."""
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import read, save, digest, path, relative

METHODS = ('ours', 'cnos', 'sam6d_rgb', 'sam6d_rgbd', 'nids_basic',
           'nids_basic_finite_templates', 'noctis', 'nemo')
INCLUSIVE = {'ours', 'cnos', 'sam6d_rgb', 'sam6d_rgbd', 'audit'}


def edges(box, method):
    value = np.asarray(box, dtype=np.float64).copy()
    if method in INCLUSIVE:
        value[2:] += 1
    return value


def intersection_over_union(first, second):
    low = np.maximum(first[:2], second[:2])
    high = np.minimum(first[2:], second[2:])
    intersect = np.maximum(high - low, 0).prod()
    areas = np.maximum(first[2:] - first[:2], 0).prod() + np.maximum(second[2:] - second[:2], 0).prod()
    return float(intersect / (areas - intersect)) if areas > intersect else 0.0


def independent_selection(rows, object_id):
    candidates = [row for row in rows if row['object_id'] == object_id]
    if not candidates:
        return None, 'no_retained_detection'
    scores = np.asarray([row['score'] for row in candidates], dtype=np.float64)
    index = int(scores.argmax())
    if np.count_nonzero(np.abs(scores - scores[index]) <= 1e-12) > 1:
        return None, 'tie'
    return candidates[index], 'ok'


def main():
    benchmark = HERE.parent
    completion = read(benchmark / 'analysis/completion.json')
    prediction_lock = read(benchmark / 'predictions_lock.json')
    assert completion['predictions_locked_before_audit']
    assert digest(benchmark / 'predictions_lock.json') == completion['prediction_lock_sha256']
    for filename, expected in prediction_lock['sha256'].items():
        assert digest(path(filename)) == expected, filename
    rows = read(benchmark / 'analysis/per_task.json')
    index = {(row['scene_id'], row['target'], row['method']): row for row in rows}
    assert len(index) == len(rows) == 8 * 12 * len(METHODS)
    manifest = read(benchmark / 'inputs.json')
    objects = manifest['objects']
    capture_root = benchmark.parents[1] / 'outputs/datasets/nist_task_board/shared_inputs/direct_cad_captures'
    results, audit_hashes = [], {}
    for scene in manifest['scenes']:
        scene_id = scene['scene_id']
        capture = capture_root / 'scenes' / scene_id / 'capture_audit.json'
        # First opening of GT in this verifier occurs only after the complete prediction-lock checks.
        truths = read(capture)['audit_only_objects']
        audit_hashes[relative(capture)] = digest(capture)
        for method in METHODS:
            folder = benchmark / method / scene_id
            detections = read(folder / 'detections.json')
            decisions = ({row['object_id']: row['status'] for row in read(folder / 'decisions.json')}
                         if method == 'ours' else {})
            for item in objects:
                observed = index[scene_id, item['target'], method]
                selected, status = independent_selection(detections, item['object_id'])
                if method == 'ours' and decisions[item['object_id']] != 'ok':
                    assert selected is None
                    status = decisions[item['object_id']]
                if method == 'nids_basic':
                    assert not detections and selected is None
                    status = 'unavailable_reference_features'
                iou = (0.0 if selected is None else intersection_over_union(
                    edges(selected['bbox_xyxy'], method), edges(truths[item['target']]['bbox'], 'audit')))
                correct = status == 'ok' and iou >= 0.5
                selected_id = None if selected is None else selected['proposal_id']
                assert observed['correct'] == correct
                assert observed['status'] == status
                assert observed['proposal_id'] == selected_id
                assert abs(observed['target_bbox_IoU'] - iou) < 1e-12
                results.append({'scene_id': scene_id, 'target': item['target'], 'method': method,
                                'correct': correct, 'status': status, 'bbox_IoU': iou})
    lookup = {(value['scene_id'], value['target'], value['method']): value for value in results}
    summary = read(benchmark / 'analysis/summary.json')
    per_object = read(benchmark / 'analysis/per_object.json')
    paired = read(benchmark / 'analysis/paired_changes.json')
    for method in METHODS:
        selected = [value for value in results if value['method'] == method]
        assert summary[method]['correct'] == sum(value['correct'] for value in selected)
        assert summary[method]['available'] == sum(value['status'] == 'ok' for value in selected)
        assert summary[method]['unavailable'] == sum(value['status'].startswith('unavailable') for value in selected)
        for value in per_object:
            if method == 'nids_basic':
                assert value[method] is None
            else:
                assert value[method] == sum(row['correct'] for row in selected if row['target'] == value['target'])
        if method == 'ours':
            continue
        changes = next(value for value in paired if value['method'] == method)
        if method == 'nids_basic':
            assert changes['comparison_available'] is False
            assert 'fixed' not in changes and 'introduced' not in changes
            assert 'fixed_vs_ours' not in summary[method]
            assert 'introduced_vs_ours' not in summary[method]
            continue
        fixed, introduced = set(), set()
        for value in selected:
            key = value['scene_id'], value['target']
            ours = lookup[key[0], key[1], 'ours']['correct']
            if value['correct'] and not ours:
                fixed.add(key)
            if ours and not value['correct']:
                introduced.add(key)
        assert summary[method]['fixed_vs_ours'] == len(fixed)
        assert summary[method]['introduced_vs_ours'] == len(introduced)
        assert changes['comparison_available'] is True
        assert {(value['scene_id'], value['target']) for value in changes['fixed']} == fixed
        assert {(value['scene_id'], value['target']) for value in changes['introduced']} == introduced
    save(benchmark / 'analysis/independent_results_verification.json', {
        'passed': True, 'requests_independently_replayed': len(results),
        'primary_selection_and_box_iou_exact': True,
        'per_object_totals_summary_and_paired_changes_match': True,
        'native_invalid_NIDS_remains_unavailable': True,
        'prediction_lock_sha256': digest(benchmark / 'predictions_lock.json'),
        'analysis_completion_sha256': digest(benchmark / 'analysis/completion.json'),
        'audit_file_sha256': audit_hashes, 'records': results,
    })
    print(f'Independent primary selection/box-IoU/per-object/paired replay passed for {len(results)} requests.')


if __name__ == '__main__':
    main()
