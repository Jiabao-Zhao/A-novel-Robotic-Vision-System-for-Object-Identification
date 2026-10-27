"""Four-arm audit. `lock` is label-free; `evaluate` requires an explicit GO flag.

This script opens no audit annotation at import or during prediction locking.
Earlier simulator capture QA used labels for input checks, without classification
scoring or selection feedback. Classification-accuracy evaluation waits for the
verified prediction lock and GO. Scores are uncalibrated: a missing prediction
never certifies that a requested CAD is absent.
"""
from collections import defaultdict
from datetime import datetime, timezone
import csv
import json
import math
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import HERE, OLD, REPO, digest, path, read, relative, save

OUT = HERE/'evaluation'
LOCK = OUT/'prediction_lock.json'
ARMS = ('clip_only', 'clip_top5_depth', 'depth_all', 'size_only')
EPS = 1e-12
IOU_GATE = .5
BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 246810


def now():
    return datetime.now(timezone.utc).isoformat()


def cohorts():
    sources = [('simulation', HERE/'capture_manifest.json', HERE/'catalog.json', 200),
               ('depth_stress', HERE/'depth_stress_manifest.json', HERE/'catalog.json', 140),
               ('new_cads', HERE/'new_cads/inputs.json', HERE/'new_cads/catalog.json', 20),
               ('real_data', HERE/'real_data/inputs.json', HERE/'real_data/catalog.json', 20)]
    result = []
    for name, manifest_file, catalog_file, expected in sources:
        scenes = read(manifest_file)['scenes']
        objects = read(catalog_file)['objects']
        assert len(scenes) == expected, (name, len(scenes), expected)
        result.append(dict(name=name, scenes=scenes, objects=objects,
                           manifest_file=manifest_file, catalog_file=catalog_file))
    assert len({s['scene_id'] for c in result for s in c['scenes']}) == 380
    return result


def bbox_half_open(box, convention='inclusive'):
    if box is None:
        return None
    x1, y1, a, b = [float(x) for x in box]
    if convention == 'inclusive':
        return [x1, y1, a+1., b+1.]
    if convention == 'bop_xywh':
        return [x1, y1, x1+a, y1+b]
    raise ValueError(convention)


def box_iou(a, b):
    if a is None or b is None:
        return 0.
    overlap = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    area_a = max(0., a[2]-a[0])*max(0., a[3]-a[1])
    area_b = max(0., b[2]-b[0])*max(0., b[3]-b[1])
    union = area_a+area_b-overlap
    return overlap/union if union else 0.


def rank_available(values, unavailable_status='unavailable_no_finite_scores'):
    """Keep unavailable values explicit and do not break a tied maximum by ID."""
    values = [(int(pid), float(score)) for pid, score in values
              if score is not None and math.isfinite(float(score))]
    values.sort(key=lambda x: (-x[1], x[0]))
    if not values:
        return dict(status=unavailable_status, proposal_id=None, score=None,
                    tied_ids=[], ranked=[])
    tied = [pid for pid, score in values if abs(score-values[0][1]) <= EPS]
    if len(tied) > 1:
        return dict(status='unavailable_top_score_tie', proposal_id=None,
                    score=values[0][1], tied_ids=tied, ranked=values)
    return dict(status='ok', proposal_id=values[0][0], score=values[0][1],
                tied_ids=[], ranked=values)


def arm_decisions(task, geometry_pairs):
    geometry = [(r['proposal_id'], r.get('geometry_score')) for r in geometry_pairs]
    size = [(r['proposal_id'], r.get('size_score')) for r in geometry_pairs]
    clip = [(pid, score) for score, pid in task.get('scores', [])]
    selected = set(task.get('candidate_ids', []))
    result = dict(clip_only=rank_available(clip), depth_all=rank_available(geometry),
                  size_only=rank_available(size))
    result['clip_top5_depth'] = rank_available(
        [(pid, score) for pid, score in geometry if pid in selected],
        unavailable_status=task['status'] if task['status'] != 'ok' else 'unavailable_no_finite_depth_in_top5')
    return result


def visual_tasks():
    tasks, files = {}, sorted((HERE/'visual').glob('*/shortlists.json'))
    assert files, 'No final CLIP shortlists exist.'
    for file in files:
        for row in read(file):
            key = row['scene_id'], int(row['object_id'])
            assert key not in tasks, ('Duplicate CLIP query', key)
            tasks[key] = row
    return tasks, files


def hash_prediction_inventory(files, expected_banks):
    """Deduplicate payload paths, then verify every referenced hypothesis bank."""
    unique = {Path(file).resolve() for file in files}
    expected = {Path(file).resolve(): value for file, value in expected_banks.items()}
    assert set(expected).issubset(unique), 'A hypothesis bank is absent from the prediction inventory'
    hashes = {}
    for file in sorted(unique):
        actual = digest(file)
        if file in expected:
            assert actual == expected[file], ('Hypothesis bank SHA256 mismatch', str(file))
        hashes[relative(file)] = actual
    return hashes


def prediction_files():
    """Validate all 5,000 queries and enumerate final, label-free evidence."""
    groups, tasks_and_files = cohorts(), visual_tasks()
    tasks, shortlist_files = tasks_and_files
    files = set(shortlist_files)
    expected_banks = {}
    expected_tasks = set()
    for file in (HERE/'protocol.json', HERE/'new_cads/protocol.json', HERE/'real_data/protocol.json'):
        files.add(file)
    for group in groups:
        files.update((group['manifest_file'], group['catalog_file']))
        objects = {int(r['object_id']): r for r in group['objects']}
        for item in objects.values():
            files.add(path(item['mesh']))
        for scene in group['scenes']:
            sid = scene['scene_id']
            files.update(path(scene[key]) for key in ('rgb', 'depth', 'intrinsics'))
            loc_file = HERE/'geometry/localization'/sid/'localization.json'
            masks_file = loc_file.with_name('masks.npz')
            scores_file = HERE/'geometry/scores'/(sid+'.json')
            assert all(f.exists() for f in (loc_file, masks_file, scores_file)), ('Incomplete prediction scene', sid)
            files.update((loc_file, masks_file, scores_file))
            localization = read(loc_file)
            assert localization['input_sha256'] == {key: digest(path(scene[key]))
                for key in ('rgb', 'depth', 'intrinsics')}, ('Changed localization inputs', sid)
            proposals = localization['proposals']
            proposal_ids = {int(r['proposal_id']) for r in proposals}
            assert len(proposal_ids) == len(proposals)
            for proposal in proposals:
                files.add(path(proposal['crop']))
                assert digest(path(proposal['crop'])) == proposal['crop_sha256']
            score_rows = read(scores_file)['pairs']
            assert all(r['target'] == objects[int(r['object_id'])]['target'] for r in score_rows), ('CAD identity mismatch', sid)
            keys = {(int(r['object_id']), int(r['proposal_id'])) for r in score_rows}
            assert len(keys) == len(score_rows), ('Duplicate geometry pairs', sid)
            assert keys == {(oid, pid) for oid in objects for pid in proposal_ids}, ('Incomplete geometry pairs', sid)
            for score in score_rows:
                fit_file = path(score['fit_file'])
                files.add(fit_file)
                fit = read(fit_file)
                bank = path(fit['bank_file']).resolve()
                expected = fit['bank_sha256']
                assert bank.is_file() and len(expected) == 64
                assert bank not in expected_banks or expected_banks[bank] == expected, ('Conflicting bank hashes', str(bank))
                expected_banks[bank] = expected
                files.add(bank)
            for oid, item in objects.items():
                key = sid, oid
                expected_tasks.add(key)
                assert key in tasks and tasks[key]['target'] == item['target'], ('Missing CLIP query', key)
                task = tasks[key]
                available = {int(pid) for _, pid in task.get('scores', [])}
                unavailable = {int(pid) for pid in task.get('unavailable_ids', [])}
                assert not available & unavailable and available | unavailable == proposal_ids, key
                assert set(task.get('candidate_ids', [])).issubset(available)
    assert set(tasks) == expected_tasks and len(tasks) == 5000, (len(tasks), len(expected_tasks))
    for name in ('inputs.json', 'descriptions.json', 'features.npz', 'mapping.json',
                 'encoding.json', 'protocol.json', 'description_completion.json'):
        files.add(HERE/'visual'/name)
    visual_inputs = read(HERE/'visual/inputs.json')
    mapping = read(HERE/'visual/mapping.json')
    assert [r['input_id'] for r in mapping['images']] == [r['input_id'] for r in visual_inputs], 'Feature/input row ordering changed'
    assert [r['sha256'] for r in mapping['images']] == [r['sha256'] for r in visual_inputs], 'Feature/input image mapping changed'
    descriptions = read(HERE/'visual/descriptions.json')
    description_keys = [(r['input_id'], r['mode']) for r in descriptions]
    assert len(description_keys) == len(set(description_keys))
    assert set(description_keys) == {(r['input_id'], mode) for r in visual_inputs for mode in ('keywords', 'phrases')}
    for item in visual_inputs:
        image_file = path(item['file'])
        assert digest(image_file) == item['sha256'], ('Visual input changed', item['input_id'])
        files.add(image_file)
    # All immutable model-output records and native baseline outputs present at lock.
    for base in (HERE/'visual', HERE/'baselines'):
        for file in base.rglob('*') if base.exists() else []:
            if file.is_file() and file.suffix in ('.json', '.npz', '.png', '.npy') and 'progress' not in file.name:
                files.add(file)
    # Scripts are locked independently of the archived execution hashes.
    files.update(HERE.rglob('*.py'))
    for name in ('run_clip.py', 'run_clip_tokenizer_fix.py', 'run_qwen.py'):
        files.add(OLD/name)
    for mode in ('keywords', 'phrases'):
        files.add(OLD/'prompts'/(mode+'.txt'))
    assert all('audit' not in f.relative_to(HERE).parts for f in files if f.is_relative_to(HERE)), 'Audit file in prediction inventory'
    assert all(f.is_file() for f in files), 'A required final prediction artifact is missing.'
    return hash_prediction_inventory(files, expected_banks), groups, tasks


def lock_predictions(create):
    hashes, groups, tasks = prediction_files()
    if LOCK.exists():
        previous = read(LOCK)
        assert previous['files'] == hashes, 'Prediction/source inventory changed after lock. Preserve old lock; investigate.'
        return previous, groups, tasks
    assert create, 'Run the label-free lock command after every prediction is complete.'
    lock = dict(created_utc=now(), labels_read=False, scenes=380, queries=5000,
                arms=list(ARMS), files=hashes,
                hypothesis_archives='Every unique hypothesis bank payload is SHA256-verified against its fit summary and included in the locked file inventory; repeated references are deduplicated before hashing.',
                absence='No arm implements calibrated absence or duplicate-instance rejection.')
    save(LOCK, lock)
    return lock, groups, tasks


def read_truth(group, scene):
    """Called only after the complete prediction lock was verified by main()."""
    sid = scene['scene_id']
    if group['name'] == 'real_data':
        from PIL import Image
        folder = HERE/'real_data/audit'/sid
        key = str(scene['source_image_id'])
        objects, info = read(folder/'scene_gt.json')[key], read(folder/'scene_gt_info.json')[key]
        assert len(objects) == len(info)
        instances = []
        for index, (obj, details) in enumerate(zip(objects, info)):
            mask_file = folder/'mask_visib'/f'{scene["source_image_id"]:06d}_{index:06d}.png'
            mask = np.asarray(Image.open(mask_file)) > 0 if mask_file.exists() else None
            visible = int(details['px_count_visib'])
            instances.append(dict(instance_id=f'{sid}__{index:06d}', object_id=int(obj['obj_id']),
                target=f'TLESS_{obj["obj_id"]:06d}', visible_pixels=visible,
                box=bbox_half_open(details['bbox_visib'], 'bop_xywh') if visible else None,
                mask=mask, label_id=None))
        return instances, None
    folder = (HERE/'new_cads/audit'/sid if group['name'] == 'new_cads' else
              HERE/'audit'/(scene['base_scene_id'] if group['name'] == 'depth_stress' else sid))
    truth = read(folder/'truth.json')
    labels = np.load(folder/'instance_labels.npz')['labels']
    instances = [dict(instance_id=r['instance_id'], object_id=int(r['object_id']),
                      target=r['target'], visible_pixels=int(r['visible_pixels']),
                      box=bbox_half_open(r['bbox_xyxy']), mask=None, label_id=int(r['label_id']))
                 for r in truth['instances']]
    return instances, labels


def proposal_audit(proposals, masks, instances, labels):
    result = {}
    # A fully hidden final instance may have no pixels and a label above max().
    label_count = max((r['label_id'] for r in instances), default=0)+1 if labels is not None else 0
    label_sizes = np.bincount(labels.ravel(), minlength=label_count) if labels is not None else None
    for proposal in proposals:
        pid = int(proposal['proposal_id'])
        mask = masks[int(proposal['mask_index'])].astype(bool)
        area = int(mask.sum())
        intersects = np.bincount(labels[mask], minlength=len(label_sizes)) if labels is not None else None
        matches = []
        for instance in instances:
            if labels is not None:
                index = instance['label_id']
                intersection, gt_area = int(intersects[index]), int(label_sizes[index])
            elif instance['mask'] is not None:
                intersection = int(np.count_nonzero(mask & instance['mask']))
                gt_area = int(instance['mask'].sum())
            else:
                intersection = gt_area = None
            union = None if gt_area is None else area+gt_area-intersection
            matches.append(dict(instance_id=instance['instance_id'], object_id=instance['object_id'],
                target=instance['target'], box_iou=box_iou(bbox_half_open(proposal['bbox_xyxy']), instance['box']),
                observed_mask_intersection_pixels=intersection,
                mask_iou=None if union is None else intersection/union if union else 0.))
        result[pid] = matches
    return result


def query_state(instances, object_id):
    physical = [r for r in instances if r['object_id'] == object_id]
    visible = [r for r in physical if r['visible_pixels'] > 0]
    return ('absent' if not physical else 'present_visible' if visible else 'present_unobservable',
            len(physical), len(visible))


def decision_correct(decision, matches, object_id):
    return decision['proposal_id'] is not None and any(
        r['object_id'] == object_id and r['box_iou'] >= IOU_GATE for r in matches)


def diagnostic_identity(matches):
    valid = [r for r in matches if r['mask_iou'] is not None and math.isfinite(r['mask_iou'])]
    if not valid:
        return dict(match=None, status='unavailable_no_mask_overlap_values', max_mask_iou=None, tied_identities=[])
    maximum = max(r['mask_iou'] for r in valid)
    tied = [r for r in valid if abs(r['mask_iou']-maximum) <= EPS]
    identities = [dict(instance_id=r['instance_id'], object_id=r['object_id'],
                       target=r['target'], mask_iou=r['mask_iou']) for r in tied] if len(tied) > 1 else []
    status = ('unassigned_low_mask_overlap' if maximum < .1 else
              'unresolved_max_mask_iou_tie' if len(tied) > 1 else 'resolved')
    return dict(match=tied[0] if status == 'resolved' else None, status=status,
                max_mask_iou=maximum, tied_identities=identities)


def pin_family(target):
    return 'round' if target.startswith('RGOCG') else 'rectangular' if target.startswith('KET') else None


def cluster_ci(rows, value='correct', denominator='present', repeats=BOOTSTRAPS):
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


def summarize(rows):
    present = [r for r in rows if r['present']]
    visible = [r for r in rows if r['state'] == 'present_visible']
    absent = [r for r in rows if r['state'] == 'absent']
    duplicates = [r for r in rows if r['visible_instances'] >= 2]
    localized = [r for r in visible if r['candidate_recall']]
    retrieved = [r for r in visible if r['clip_top5_recall']]
    return dict(queries=len(rows), physically_present=len(present), observable_present=len(visible),
        unobservable_present=len(present)-len(visible), correct=sum(r['correct'] for r in present),
        present_accuracy=sum(r['correct'] for r in present)/len(present) if present else None,
        present_accuracy_cluster_ci95=cluster_ci(rows),
        observable_accuracy=sum(r['correct'] for r in visible)/len(visible) if visible else None,
        unavailable_present=sum(r['proposal_id'] is None for r in present),
        absent_queries=len(absent), absent_false_selections=sum(r['proposal_id'] is not None for r in absent),
        absent_unavailable=sum(r['proposal_id'] is None for r in absent), certified_absence=0,
        duplicate_visible_queries=len(duplicates), duplicate_class_correct=sum(r['correct'] for r in duplicates),
        duplicate_requests_need_clarification=len(duplicates), certified_duplicate_detection=0,
        candidate_bbox_recall=sum(r['candidate_recall'] for r in visible)/len(visible) if visible else None,
        clip_top5_bbox_recall=sum(r['clip_top5_recall'] for r in visible)/len(visible) if visible else None,
        candidate_recalled_queries=len(localized),
        correct_given_candidate_recall=sum(r['correct'] for r in localized),
        accuracy_given_candidate_recall=sum(r['correct'] for r in localized)/len(localized) if localized else None,
        clip_top5_recalled_queries=len(retrieved),
        correct_given_clip_top5_recall=sum(r['correct'] for r in retrieved),
        accuracy_given_clip_top5_recall=sum(r['correct'] for r in retrieved)/len(retrieved) if retrieved else None,
        lost_before_candidate_verification=len(visible)-len(localized),
        lost_by_CLIP_top5_after_candidate_recall=sum(not r['clip_top5_recall'] for r in localized),
        no_valid_target_depth_among_recalled_candidates=sum(not r['target_depth_available_any'] for r in localized),
        no_valid_target_depth_in_recalled_top5=sum(not r['target_depth_available_top5'] for r in retrieved),
        pin_wrong_size=sum(r['pin_error'] == 'same_shape_wrong_size' for r in present),
        pin_wrong_shape=sum(r['pin_error'] == 'round_rectangular_confusion' for r in present))


def grouped_summaries(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    return [dict(zip(keys, key), **summarize(values)) for key, values in sorted(groups.items())]


def paired_changes(rows):
    indexed = {(r['scene_id'], r['object_id'], r['method']): r for r in rows}
    result = []
    for row in rows:
        for reference in ('clip_only', 'depth_all'):
            if row['method'] == reference:
                continue
            old = indexed[row['scene_id'], row['object_id'], reference]
            if row['present'] and bool(row['correct']) != bool(old['correct']):
                result.append(dict(scene_id=row['scene_id'], base_scene_id=row['base_scene_id'],
                    cohort=row['cohort'], condition=row['condition'], target=row['target'], object_id=row['object_id'],
                    method=row['method'], reference=reference, change='fixed' if row['correct'] else 'introduced',
                    old_proposal_id=old['proposal_id'], new_proposal_id=row['proposal_id'],
                    old_status=old['status'], new_status=row['status'], rgb=row['rgb'],
                    old_crop=old['crop'], new_crop=row['crop']))
    return result


def paired_summary(rows):
    indexed = {(r['scene_id'], r['object_id'], r['method']): r for r in rows}
    result = []
    for method in ARMS:
        for reference in ('clip_only', 'depth_all'):
            if method == reference:
                continue
            for cohort in sorted({r['cohort'] for r in rows}):
                selected = [r for r in rows if r['cohort'] == cohort and r['method'] == method]
                deltas = [dict(r, delta=int(r['correct'])-int(indexed[r['scene_id'], r['object_id'], reference]['correct'])) for r in selected]
                n = sum(r['present'] for r in deltas)
                result.append(dict(cohort=cohort, method=method, reference=reference,
                    physically_present=n, fixed=sum(r['present'] and r['delta'] == 1 for r in deltas),
                    introduced=sum(r['present'] and r['delta'] == -1 for r in deltas),
                    accuracy_difference=sum(r['delta'] for r in deltas if r['present'])/n if n else None,
                    paired_cluster_ci95=cluster_ci(deltas, value='delta')))
    return result


def paired_condition_effects(rows):
    """A query is paired to its same CAD, method, and nominal base layout."""
    nominal = {(r['scene_id'], r['object_id'], r['method']): r for r in rows
               if r['cohort'] == 'simulation' and r['condition'] == 'nominal'}
    effects = []
    for row in rows:
        if row['cohort'] not in ('simulation', 'depth_stress') or row['condition'] == 'nominal':
            continue
        before = nominal[row['base_scene_id'], row['object_id'], row['method']]
        paired_present = bool(row['present'] and before['present'])
        delta = int(row['correct'])-int(before['correct']) if paired_present else None
        effects.append(dict(cohort=row['cohort'], condition=row['condition'], method=row['method'],
            scene_id=row['scene_id'], base_scene_id=row['base_scene_id'], object_id=row['object_id'],
            target=row['target'], present=paired_present, comparison_eligible=paired_present, delta=delta,
            nominal_correct=before['correct'], condition_correct=row['correct'],
            nominal_proposal_id=before['proposal_id'], condition_proposal_id=row['proposal_id'],
            change='not_comparable_not_present' if not paired_present else
                   'fixed' if delta == 1 else 'introduced' if delta == -1 else 'unchanged',
            condition_state=row['state'], nominal_state=before['state'],
            target_pixel_fraction_relative_nominal=row['target_pixel_fraction_relative_nominal']))
    groups = defaultdict(list)
    for row in effects:
        groups[row['cohort'], row['condition'], row['method']].append(row)
    summaries = []
    for (cohort, condition, method), values in sorted(groups.items()):
        present = [r for r in values if r['present']]
        n = len(present)
        summaries.append(dict(cohort=cohort, condition=condition, method=method,
            total_paired_queries=len(values), paired_physically_present_queries=n,
            excluded_absent_pairs=len(values)-n,
            nominal_correct=sum(r['nominal_correct'] for r in present),
            condition_correct=sum(r['condition_correct'] for r in present),
            fixed=sum(r['delta'] == 1 for r in present),
            introduced=sum(r['delta'] == -1 for r in present),
            accuracy_difference=sum(r['delta'] for r in present)/n if n else None,
            paired_cluster_ci95=cluster_ci(values, value='delta')))
    return effects, summaries


def relative_nominal_pixel_group(value, visible_pixels):
    if visible_pixels == 0:
        return 'fully_hidden'
    if value is None:
        return 'nominal_pixels_unavailable'
    if value <= .25:
        return '(0,0.25]'
    if value <= .5:
        return '(0.25,0.5]'
    if value <= .75:
        return '(0.5,0.75]'
    if value <= 1.:
        return '(0.75,1]'
    return '>1'


def threshold_curves(rows):
    """Fixed diagnostic grids; do not select or report a fitted 'best' threshold."""
    result = []
    groups = defaultdict(list)
    for row in rows:
        groups[row['cohort'], row['method']].append(row)
    for (cohort, method), values in sorted(groups.items()):
        grid = np.linspace(-1., 1., 101) if method == 'clip_only' else np.linspace(0., 1., 101)
        for threshold in grid:
            counts = defaultdict(int, {name: 0 for name in (
                'present_queries', 'present_emitted', 'present_withheld', 'present_correct', 'present_wrong',
                'absent_queries', 'absent_emitted', 'absent_withheld', 'duplicate_visible_queries',
                'duplicate_candidate_count_flagged', 'single_visible_queries', 'single_candidate_count_flagged')})
            for row in values:
                emitted = row['proposal_id'] is not None and row['score'] >= threshold
                state = 'absent' if row['state'] == 'absent' else 'present'
                counts[state+'_queries'] += 1
                counts[state+('_emitted' if emitted else '_withheld')] += 1
                if state == 'present' and emitted:
                    counts['present_correct' if row['correct'] else 'present_wrong'] += 1
                ambiguous = sum(score >= threshold for _, score in row['ranked']) >= 2
                if row['visible_instances'] >= 2:
                    counts['duplicate_visible_queries'] += 1
                    counts['duplicate_candidate_count_flagged'] += int(ambiguous)
                elif row['visible_instances'] == 1:
                    counts['single_visible_queries'] += 1
                    counts['single_candidate_count_flagged'] += int(ambiguous)
            result.append(dict(cohort=cohort, method=method, threshold=float(threshold), **counts))
    return result


def csv_write(file, rows):
    file.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with file.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, separators=(',', ':')) if isinstance(v, (dict, list, tuple)) else v
                             for k, v in row.items()})


def evaluate(groups, tasks):
    save(OUT/'audit_access.json', dict(first_classification_accuracy_audit_read_not_before_utc=now(),
        prediction_lock_sha256=digest(LOCK), command='evaluate --read-audit-after-go',
        message='Predictions and evaluation source were verified before classification-accuracy annotation access.',
        earlier_capture_QA='capture/audit_capture.py used simulator truth for counts, projection and paired visibility input checks; no classification scoring or selection feedback.'))
    # Classification evaluation first reads these counts after lock verification.
    nominal_pixels = {}
    simulation = next(g for g in groups if g['name'] == 'simulation')
    for scene in simulation['scenes']:
        if scene['condition'] == 'nominal':
            truth = read(HERE/'audit'/scene['scene_id']/'truth.json')
            counts = defaultdict(int)
            for instance in truth['instances']:
                counts[int(instance['object_id'])] += int(instance['visible_pixels'])
            for oid, pixels in counts.items():
                nominal_pixels[scene['scene_id'], oid] = pixels
    rows = []
    for group in groups:
        for scene in group['scenes']:
            sid = scene['scene_id']
            instances, labels = read_truth(group, scene)
            loc_file = HERE/'geometry/localization'/sid/'localization.json'
            proposals = read(loc_file)['proposals']
            by_pid = {int(r['proposal_id']): r for r in proposals}
            masks = np.load(loc_file.with_name('masks.npz'))['masks']
            audits = proposal_audit(proposals, masks, instances, labels)
            score_pairs = read(HERE/'geometry/scores'/(sid+'.json'))['pairs']
            for item in group['objects']:
                oid, target = int(item['object_id']), item['target']
                state, physical_count, visible_count = query_state(instances, oid)
                task = tasks[sid, oid]
                relevant = [r for r in score_pairs if int(r['object_id']) == oid]
                decisions = arm_decisions(task, relevant)
                target_ious = {pid: max((r['box_iou'] for r in values if r['object_id'] == oid), default=0.)
                               for pid, values in audits.items()}
                candidate_recall = any(v >= IOU_GATE for v in target_ious.values())
                top5_recall = any(target_ious.get(int(pid), 0.) >= IOU_GATE for pid in task['candidate_ids'])
                target_pixels = sum(r['visible_pixels'] for r in instances if r['object_id'] == oid)
                nominal_count = nominal_pixels.get((scene['base_scene_id'], oid))
                pixel_fraction = target_pixels/nominal_count if nominal_count else None
                available_depth = {int(r['proposal_id']) for r in relevant
                                   if r.get('geometry_score') is not None and math.isfinite(r['geometry_score'])}
                target_depth_any = any(target_ious.get(pid, 0.) >= IOU_GATE for pid in available_depth)
                target_depth_top5 = any(target_ious.get(pid, 0.) >= IOU_GATE
                                       for pid in available_depth & set(task['candidate_ids']))
                for method, decision in decisions.items():
                    pid = decision['proposal_id']
                    matches = audits.get(pid, [])
                    correct = decision_correct(decision, matches, oid)
                    valid_masks = [r for r in matches if r['mask_iou'] is not None]
                    identity = diagnostic_identity(matches)
                    physical_match = identity['match']
                    pin_error = None
                    if not correct and physical_match and pin_family(target) and pin_family(physical_match['target']):
                        pin_error = ('same_shape_wrong_size' if pin_family(target) == pin_family(physical_match['target'])
                                     and target != physical_match['target'] else
                                     'round_rectangular_confusion' if pin_family(target) != pin_family(physical_match['target']) else None)
                    rows.append(dict(cohort=group['name'], scene_id=sid, base_scene_id=scene['base_scene_id'],
                        condition=scene['condition'], object_id=oid, target=target,
                        new_query_primary=group['name'] == 'new_cads' and target.endswith('_Female'),
                        method=method, state=state, present=state != 'absent', physical_instances=physical_count,
                        visible_instances=visible_count, needs_clarification=visible_count >= 2,
                        correct=bool(correct), status=decision['status'], proposal_id=pid, score=decision['score'],
                        tied_ids=decision['tied_ids'], ranked=decision['ranked'],
                        target_bbox_iou=target_ious.get(pid, 0.),
                        target_mask_iou=max((r['mask_iou'] for r in valid_masks if r['object_id'] == oid), default=None),
                        selected_physical_target=None if physical_match is None else physical_match['target'],
                        selected_physical_mask_iou=identity['max_mask_iou'],
                        selected_physical_identity_status=identity['status'],
                        selected_physical_tied_identities=identity['tied_identities'],
                        physical_match_rule='Diagnostic only: unique largest observed-mask IoU >= 0.1; maxima tied within 1e-12 remain unresolved with all tied instances retained. Primary correctness uses bbox IoU>=0.5.',
                        pin_error=pin_error, candidate_recall=candidate_recall, clip_top5_recall=top5_recall,
                        candidate_target_bbox_iou={str(pid): value for pid, value in target_ious.items()},
                        selected_mask_instance_pixels=[dict(instance_id=r['instance_id'], target=r['target'],
                            pixels=r['observed_mask_intersection_pixels']) for r in matches
                            if r['observed_mask_intersection_pixels'] is not None and r['observed_mask_intersection_pixels'] > 0],
                        target_depth_available_any=target_depth_any, target_depth_available_top5=target_depth_top5,
                        target_visible_pixels=target_pixels, nominal_target_visible_pixels=nominal_count,
                        target_pixel_fraction_relative_nominal=pixel_fraction,
                        relative_nominal_pixel_group=relative_nominal_pixel_group(pixel_fraction, target_pixels),
                        clip_status=task['status'], candidate_count=len(proposals),
                        rgb=scene['rgb'], crop=None if pid is None else by_pid[pid]['crop']))
            print(f'Audited {sid}: {len(group["objects"])} CAD queries', flush=True)
    destination = OUT/'results'
    save(destination/'per_query.json', rows)
    csv_write(destination/'per_query.csv', rows)
    for name, keys in [('per_cohort', ['cohort', 'method']),
                       ('per_condition', ['cohort', 'condition', 'method']),
                       ('per_object', ['cohort', 'target', 'method'])]:
        values = grouped_summaries(rows, keys)
        save(destination/(name+'.json'), values)
        csv_write(destination/(name+'.csv'), values)
    for name, values in [('paired_changes', paired_changes(rows)), ('paired_summary', paired_summary(rows)),
                         ('threshold_curves', threshold_curves(rows))]:
        save(destination/(name+'.json'), values)
        csv_write(destination/(name+'.csv'), values)
    effects, effect_summary = paired_condition_effects(rows)
    for name, values in [('paired_condition_queries', effects), ('paired_condition_effects', effect_summary)]:
        save(destination/(name+'.json'), values)
        csv_write(destination/(name+'.csv'), values)
    occluded = [r for r in rows if r['cohort'] == 'simulation'
                and r['condition'] in ('occlusion_light', 'occlusion_heavy')]
    occlusion_groups = grouped_summaries(occluded,
        ['condition', 'relative_nominal_pixel_group', 'method'])
    save(destination/'occlusion_by_relative_nominal_pixels.json', occlusion_groups)
    csv_write(destination/'occlusion_by_relative_nominal_pixels.csv', occlusion_groups)
    females = [r for r in rows if r['new_query_primary']]
    controls = [r for r in rows if r['cohort'] == 'new_cads' and not r['new_query_primary']]
    save(destination/'new_cad_subsets.json', dict(
        newly_queried_females=grouped_summaries(females, ['method']),
        twelve_known_controls=grouped_summaries(controls, ['method']),
        scope='Female CADs are new classification queries; they previously appeared as fixed fixtures.'))
    save(destination/'completion.json', dict(completed_utc=now(), rows=len(rows),
        queries=len(rows)//len(ARMS), prediction_lock_sha256=digest(LOCK),
        primary='Requested CAD identity with bounding-box IoU >= 0.5 against any visible matching instance.',
        absence='Unavailable is not certified absence. Threshold grids are diagnostic, without selected operating point.',
        ambiguity='Class correctness and duplicate-instance clarification are separate.',
        relative_pixels='Summed target visible pixels divided by same-base nominal target visible pixels; not amodal or physical visibility. Fully hidden targets remain separate.',
        intervals=f'{BOOTSTRAPS} paired base-scene cluster bootstrap draws; seed {BOOTSTRAP_SEED}.',
        limitations='Twenty cached T-LESS frames are a real pilot, not a full BOP benchmark. No held-out threshold tuning.'))


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else ''
    assert command in ('lock', 'evaluate'), 'Use lock, or evaluate --read-audit-after-go after parent GO.'
    if command == 'evaluate':
        assert sys.argv[2:] == ['--read-audit-after-go'], 'Audit access requires explicit parent GO.'
    lock, groups, tasks = lock_predictions(create=command == 'lock')
    print(f'Prediction lock verified: {len(lock["files"])} artifacts, {len(tasks)} CAD queries', flush=True)
    if command == 'evaluate':
        evaluate(groups, tasks)


if __name__ == '__main__':
    main()
