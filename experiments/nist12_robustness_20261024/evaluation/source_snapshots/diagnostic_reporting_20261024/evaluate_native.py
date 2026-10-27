"""Native own-mask audit, callable only after whole-study lock and parent GO.

Imports and synthetic tests never read audit annotations. Native scores are
preserved. The primary criterion matches the main study: requested CAD identity
with observed target bbox IoU >= 0.5. Physical mask identity is diagnostic only.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import sys
import numpy as np
import common

TRIAL = common.HERE.parent
OUT = TRIAL / 'evaluation/native_baselines'
LOCK = TRIAL / 'evaluation/prediction_lock.json'
METHODS = ('cnos', 'sam6d_rgb', 'sam6d_rgbd', 'nids_basic_finite_templates', 'noctis', 'nemo')
UNAVAILABLE = 'nids_basic_native_unguarded'
BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 246810


def edges(box, method):
    if box is None:
        return None
    result = list(map(float, box))
    if method in ('audit', 'cnos', 'sam6d_rgb', 'sam6d_rgbd'):
        result[2] += 1.
        result[3] += 1.
    return result


def iou(a, b):
    if a is None or b is None:
        return 0.
    intersection = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    area = lambda r: max(0., r[2]-r[0])*max(0., r[3]-r[1])
    union = area(a)+area(b)-intersection
    return intersection/union if union else 0.


def select(rows, object_id):
    candidates = sorted((r for r in rows if r['object_id'] == object_id), key=lambda r: -r['score'])
    if not candidates:
        return None, 'no_retained_detection'
    assert all(np.isfinite(r['score']) for r in candidates)
    if len(candidates) > 1 and abs(candidates[0]['score']-candidates[1]['score']) <= 1e-12:
        return None, 'unavailable_top_score_tie'
    return candidates[0], 'ok'


def stage_boxes(method, sid):
    if method in ('cnos', 'sam6d_rgb', 'sam6d_rgbd'):
        source = 'cnos' if method == 'cnos' else 'sam6d'
        rows = common.read(common.HERE/source/'raw_proposals'/sid/'proposals.json')
        folder = common.HERE/source/sid
    elif method in ('nids_basic_finite_templates', 'noctis'):
        folder = common.HERE/method/sid
        rows = common.read(folder/'raw_proposals.json')
    else:
        return None, None
    raw = {int(r['proposal_id']): edges(r['bbox_xyxy'], method) for r in rows}
    retained = common.read(folder/'proposal_filter.json')['retained_proposal_ids']
    return list(raw.values()), [raw[int(pid)] for pid in retained]


def conditional_boxes(folder, object_id):
    """Reconstruct released NeMO connected-component size filtering exactly."""
    import cv2
    with np.load(folder/f'raw_obj_{object_id:06d}.npz') as bank:
        amodal = bank['amodal_mask'].astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(amodal, connectivity=8)
    before, after = [], []
    for label in range(1, count):
        x, y, width, height, area = [int(x) for x in stats[label]]
        box = [x, y, x+width, y+height]
        before.append(box)
        if area >= 1024:
            after.append(box)
    return before, after


def recall(boxes, target_box):
    return None if boxes is None else any(iou(box, target_box) >= .5 for box in boxes)


def physical_identity(mask, labels, instances, label_counts=None):
    area = int(mask.sum())
    maximum_label = max((int(r['label_id']) for r in instances), default=0)
    if label_counts is None:
        label_counts = np.bincount(labels.ravel(), minlength=maximum_label+1)
    intersections = np.bincount(labels[mask], minlength=maximum_label+1)
    overlaps = []
    for item in instances:
        index = int(item['label_id'])
        intersection = int(intersections[index])
        union = area+int(label_counts[index])-intersection
        overlaps.append((intersection/union if union else 0., item['target']))
    overlaps.sort(reverse=True)
    maximum, target = overlaps[0] if overlaps else (0., None)
    return (target if maximum >= .1 else None), maximum, dict((name, value) for value, name in overlaps)


def failure(correct, status, raw_recall, filtered_recall, selected, target,
            conditional_before=None, conditional_after=None):
    if correct:
        return 'correct'
    if status == 'unavailable_reference_features':
        return 'method_reference_features_invalid'
    if status == 'unavailable_top_score_tie':
        return 'top_score_tie'
    if raw_recall is False:
        return 'own_proposal_localization_miss'
    if filtered_recall is False:
        return 'native_size_filter_miss'
    if conditional_before and not conditional_after:
        return 'conditional_component_size_filter_miss'
    if status == 'no_retained_detection':
        return 'target_identity_missing_after_native_matching' if raw_recall is not None else 'no_target_conditioned_detection'
    if selected == target:
        return 'target_selected_but_bbox_IoU_below_0.5'
    if selected is None:
        return 'selected_region_unassigned_or_poor_localization'
    return 'wrong_physical_instance_selected'


def family(target):
    return 'round' if str(target).startswith('RGOCG') else 'rectangular' if str(target).startswith('KET') else None


def pin_confusion(correct, target, selected):
    if correct or not family(target) or not family(selected):
        return None
    if family(target) != family(selected):
        return 'round_rectangular_confusion'
    return 'same_shape_wrong_size' if target != selected else None


def cluster_ci(rows, value='correct', denominator='eligible', repeats=BOOTSTRAPS):
    """Match the main evaluator's paired base-scene percentile bootstrap."""
    clusters = defaultdict(lambda: [0., 0.])
    for row in rows:
        clusters[row['base_scene_id']][0] += float(row[value]) if row[denominator] else 0.
        clusters[row['base_scene_id']][1] += float(row[denominator])
    array = np.asarray(list(clusters.values()), dtype=float)
    if not len(array) or not array[:, 1].sum():
        return None
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(array), (repeats, len(array)))
    totals = array[draws].sum(axis=1)
    ratios = totals[totals[:, 1] > 0, 0]/totals[totals[:, 1] > 0, 1]
    return [float(x) for x in np.quantile(ratios, [.025, .975])]


def verify_prediction_lock():
    frozen = common.read(LOCK)
    assert frozen['labels_read'] is False
    assert frozen['scenes'] == 380 and frozen['queries'] == 5000
    for filename, expected in frozen['files'].items():
        assert common.digest(common.path(filename)) == expected, filename
    for method in METHODS:
        for scene in common.setup()['scenes']:
            for name in ('detections.json', 'masks.npz', 'completion.json'):
                filename = common.HERE/method/scene['scene_id']/name
                assert common.relative(filename) in frozen['files'], filename
    assert common.relative(Path(__file__)) in frozen['files']
    # These are evaluation outputs, read only after the same whole-study lock.
    completion = common.read(TRIAL/'evaluation/results/completion.json')
    assert completion['prediction_lock_sha256'] == common.digest(LOCK)
    return frozen


def csv_write(filename, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows)


def evaluate():
    verify_prediction_lock()  # No annotation access may precede this check.
    manifest = common.setup()
    ours = {(r['scene_id'], r['object_id']): r for r in common.read(TRIAL/'evaluation/results/per_query.json')
            if r['method'] == 'clip_top5_depth' and r['cohort'] == 'simulation' and r['condition'] == 'nominal'}
    assert len(ours) == 240
    records, audit_hashes = [], {}
    for scene in manifest['scenes']:
        sid = scene['scene_id']
        folder = TRIAL/'audit'/sid
        for name in ('truth.json', 'instance_labels.npz'):
            audit_hashes[common.relative(folder/name)] = common.digest(folder/name)
        instances = common.read(folder/'truth.json')['instances']
        with np.load(folder/'instance_labels.npz') as bank:
            labels = bank['labels']
        label_counts = np.bincount(labels.ravel())
        for method in (*METHODS, UNAVAILABLE):
            unavailable = method == UNAVAILABLE
            directory = common.HERE/method/sid
            detections = [] if unavailable else common.read(directory/'detections.json')
            masks = None if unavailable else np.load(directory/'masks.npz')['masks']
            raw, filtered = (None, None) if unavailable else stage_boxes(method, sid)
            for item in manifest['objects']:
                oid, target = int(item['object_id']), item['target']
                truths = [r for r in instances if int(r['object_id']) == oid]
                assert len(truths) == 1 and truths[0]['visible_pixels'] > 0, 'Nominal cohort contract changed'
                truth = truths[0]
                target_box = edges(truth['bbox_xyxy'], 'audit')
                chosen, status = (None, 'unavailable_reference_features') if unavailable else select(detections, oid)
                chosen_iou = 0. if chosen is None else iou(edges(chosen['bbox_xyxy'], method), target_box)
                correct = None if unavailable else status == 'ok' and chosen_iou >= .5
                selected, maximum, overlaps = (None, None, {}) if chosen is None else physical_identity(
                    masks[int(chosen['mask_index'])].astype(bool), labels, instances, label_counts)
                raw_recall, filtered_recall = recall(raw, target_box), recall(filtered, target_box)
                conditional_before = conditional_after = None
                if method == 'nemo':
                    before, after = conditional_boxes(directory, oid)
                    exported = [edges(r['bbox_xyxy'], method) for r in detections if r['object_id'] == oid]
                    assert after == exported, 'NeMO native 1024-pixel component reconstruction differs'
                    conditional_before, conditional_after = recall(before, target_box), recall(after, target_box)
                error = failure(correct, status, raw_recall, filtered_recall, selected, target,
                                conditional_before, conditional_after)
                retained_any = None if unavailable else recall([edges(r['bbox_xyxy'], method) for r in detections], target_box)
                retained_identity = None if unavailable else recall([edges(r['bbox_xyxy'], method) for r in detections if r['object_id'] == oid], target_box)
                if status == 'ok' and not correct and retained_identity:
                    error = 'lower_scored_correct_target_detection'
                records.append(dict(scene_id=sid, base_scene_id=sid, object_id=oid, target=target, method=method,
                    run_status='inherited_native_reference_failure_not_rerun' if unavailable else 'completed',
                    eligible=not unavailable,
                    correct=correct, status=status, error=error, score=None if chosen is None else chosen['score'],
                    proposal_id=None if chosen is None else chosen['proposal_id'],
                    selected_bbox_xyxy=None if chosen is None else chosen['bbox_xyxy'],
                    target_bbox_xyxy=truth['bbox_xyxy'],
                    selected_physical_target=selected, selected_physical_mask_iou=maximum,
                    target_bbox_iou=chosen_iou, target_mask_iou=overlaps.get(target),
                    own_raw_candidate_recall=raw_recall, after_native_size_filter_recall=filtered_recall,
                    retained_box_recall_any_identity=retained_any, retained_box_recall_requested_identity=retained_identity,
                    conditional_component_recall_before_area_filter=conditional_before,
                    conditional_component_recall_after_area_filter=conditional_after,
                    retained_detections=None if unavailable else len(detections),
                    pin_error=pin_confusion(correct, target, selected),
                    ours_correct=ours[sid, oid]['correct'], ours_status=ours[sid, oid]['status'],
                    ours_proposal_id=ours[sid, oid]['proposal_id'], ours_crop=ours[sid, oid]['crop'],
                    delta_vs_ours=None if unavailable else int(correct)-int(ours[sid, oid]['correct']),
                    change_vs_ours=None if unavailable else 'fixed' if correct and not ours[sid, oid]['correct'] else
                                   'introduced' if not correct and ours[sid, oid]['correct'] else 'unchanged', rgb=scene['rgb']))
            del masks
        print(f'Native baseline audit complete: {sid}', flush=True)
    summary = []
    for method in (*METHODS, UNAVAILABLE):
        rows = [r for r in records if r['method'] == method]
        unavailable = method == UNAVAILABLE
        summary.append(dict(method=method, queries=len(rows), available=not unavailable,
            correct=None if unavailable else sum(r['correct'] for r in rows),
            accuracy=None if unavailable else sum(r['correct'] for r in rows)/len(rows),
            accuracy_cluster_ci95=cluster_ci(rows),
            own_raw_candidate_recalled=sum(r['own_raw_candidate_recall'] is True for r in rows),
            own_raw_candidate_queries=sum(r['own_raw_candidate_recall'] is not None for r in rows),
            after_native_size_filter_recalled=sum(r['after_native_size_filter_recall'] is True for r in rows),
            conditional_before_area_recalled=sum(r['conditional_component_recall_before_area_filter'] is True for r in rows),
            conditional_after_area_recalled=sum(r['conditional_component_recall_after_area_filter'] is True for r in rows),
            errors=dict(Counter(r['error'] for r in rows if not r['correct'])),
            fixed_vs_ours=None if unavailable else sum(r['change_vs_ours'] == 'fixed' for r in rows),
            introduced_vs_ours=None if unavailable else sum(r['change_vs_ours'] == 'introduced' for r in rows),
            accuracy_difference_vs_ours=None if unavailable else sum(r['delta_vs_ours'] for r in rows)/len(rows),
            paired_difference_cluster_ci95=cluster_ci(rows, value='delta_vs_ours')))
    objects = []
    for item in manifest['objects']:
        for method in (*METHODS, UNAVAILABLE):
            rows = [r for r in records if r['target'] == item['target'] and r['method'] == method]
            objects.append(dict(target=item['target'], method=method, queries=20,
                correct=None if method == UNAVAILABLE else sum(r['correct'] for r in rows),
                accuracy_cluster_ci95=cluster_ci(rows),
                accuracy_difference_vs_ours=None if method == UNAVAILABLE else sum(r['delta_vs_ours'] for r in rows)/len(rows),
                paired_difference_cluster_ci95=cluster_ci(rows, value='delta_vs_ours'),
                own_raw_candidate_recalled=sum(r['own_raw_candidate_recall'] is True for r in rows),
                own_raw_candidate_queries=sum(r['own_raw_candidate_recall'] is not None for r in rows),
                after_native_size_filter_recalled=sum(r['after_native_size_filter_recall'] is True for r in rows),
                retained_requested_identity_recalled=sum(r['retained_box_recall_requested_identity'] is True for r in rows),
                errors=dict(Counter(r['error'] for r in rows if not r['correct']))))
    paired = [dict(method=row['method'], reference='clip_top5_depth', queries=row['queries'],
                   comparison_available=row['available'], fixed=row['fixed_vs_ours'], introduced=row['introduced_vs_ours'],
                   accuracy_difference=row['accuracy_difference_vs_ours'],
                   paired_cluster_ci95=row['paired_difference_cluster_ci95']) for row in summary]
    for name, rows in [('per_query', records), ('summary', summary), ('per_object', objects), ('paired_summary', paired),
                       ('paired_changes', [r for r in records if r['change_vs_ours'] in ('fixed', 'introduced')])]:
        common.save(OUT/(name+'.json'), rows)
        csv_write(OUT/(name+'.csv'), rows)
    common.save(OUT/'completion.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        nominal_scenes=20, queries_per_method=240, methods=list(METHODS),
        unavailable_row=UNAVAILABLE, prediction_lock_sha256=common.digest(LOCK), audit_sha256=audit_hashes,
        source_sha256=common.digest(__file__), native_scores_preserved=True,
        intervals=f'{BOOTSTRAPS} paired base-scene cluster percentile bootstrap draws; seed {BOOTSTRAP_SEED}; native unavailable row excluded.',
        physical_identity='Diagnostic only: highest observed-mask IoU >= 0.1; primary correctness uses bbox IoU >= 0.5.',
        NeMO_attribution='Own CAD-conditioned component masks; no class-agnostic proposal recall is claimed.'))


if __name__ == '__main__':
    assert sys.argv[1:] == ['evaluate', '--read-audit-after-go'], 'Explicit whole-study lock and parent GO required.'
    evaluate()
