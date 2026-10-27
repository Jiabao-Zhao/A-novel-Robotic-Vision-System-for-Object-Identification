"""Development-fixed proposal plausibility; primary rankings stay unchanged."""
import math
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import HERE, digest, path, read, relative, save

DIRECTORY = HERE/'evaluation/plausibility'
CONFIRMATORY_COHORTS = ('simulation',)
METHOD = 'clip_top5_depth'


def finite(value):
    return value is not None and math.isfinite(float(value))


def development_threshold(rows):
    positives = [r for r in rows if r['bbox_target']]
    negatives = [r for r in rows if not r['bbox_target']]
    positive_finite = [r for r in positives if finite(r['score'])]
    negative_finite = [r for r in negatives if finite(r['score'])]
    maximum = max((float(r['score']) for r in negative_finite), default=None)
    tau = None if maximum is None else float(np.nextafter(np.float64(maximum), np.float64(np.inf)))
    assert tau is None or math.isfinite(tau)
    passing = lambda values: sum(tau is not None and finite(r['score']) and r['score'] >= tau for r in values)
    positive_passes, negative_passes = passing(positives), passing(negatives)
    assert negative_passes == 0
    ratio = lambda numerator, denominator: numerator/denominator if denominator else None
    return dict(status='available' if tau is not None else 'unavailable_no_finite_non_target_development_score',
        tau=tau, tau_float64_hex=None if tau is None else tau.hex(), maximum_finite_non_target_score=maximum,
        pairs=len(rows), bbox_target_pairs=len(positives), bbox_non_target_pairs=len(negatives),
        finite_target_pairs=len(positive_finite), finite_non_target_pairs=len(negative_finite),
        unavailable_target_pairs=len(positives)-len(positive_finite),
        unavailable_non_target_pairs=len(negatives)-len(negative_finite),
        target_pairs_passing=positive_passes, non_target_pairs_passing=negative_passes,
        target_coverage_all_pairs=ratio(positive_passes, len(positives)),
        target_coverage_finite_pairs=ratio(positive_passes, len(positive_finite)),
        non_target_pass_rate_all_pairs=ratio(negative_passes, len(negatives)),
        non_target_pass_rate_finite_pairs=ratio(negative_passes, len(negative_finite)))


def in_scope(cohort, method, target, calibration):
    return cohort in CONFIRMATORY_COHORTS and method == METHOD and target in calibration['known_targets']


def overlay(task, pairs, calibration):
    ids = [int(pid) for pid in task.get('candidate_ids', [])]
    by_id = {int(r['proposal_id']): r for r in pairs}
    assert len(by_id) == len(pairs)
    assert set(ids).issubset(by_id)
    available = [pid for pid in ids if finite(by_id[pid].get('geometry_score'))]
    unavailable = [dict(proposal_id=pid, status=by_id[pid].get('status', 'unavailable_depth_score'))
                   for pid in ids if pid not in available]
    tau = calibration['tau']
    passing = [pid for pid in available if tau is not None and by_id[pid]['geometry_score'] >= tau]
    reason = (task['status'] if task['status'] != 'ok' else
              calibration['status'] if tau is None else
              'unavailable_no_finite_top5_depth_scores' if not available else None)
    flag = ('unavailable' if reason else 'zero_passing_proposals' if not passing else
            'one_passing_proposal' if len(passing) == 1 else 'multiple_passing_proposals')
    return dict(flag=flag, unavailable_reason=reason, tau=tau, passing_proposal_ids=passing,
        passing_proposals=len(passing), available_proposal_ids=available,
        unavailable_candidates=unavailable, shortlist_proposals=len(ids),
        all_shortlisted_scores_available=len(available) == len(ids),
        certifies_absence=False, certifies_duplicates=False, primary_ranking_modified=False)


def verified_calibration():
    lock_file = DIRECTORY/'calibration_lock.json'
    lock = read(lock_file)
    assert lock['historical_development_accuracy_labels_read'] is True
    assert lock['new_robustness_accuracy_labels_read'] is False
    files = set()
    historical_sources = {
        HERE/name: DIRECTORY/'frozen_sources'/Path(name).name
        for name in ('evaluation/plausibility.py', 'evaluation/calibrate_plausibility.py',
                     'evaluation/test_plausibility.py', 'evaluation/evaluate.py',
                     'evaluation/test_evaluate.py', 'evaluation/render_report.py',
                     'baselines/evaluate_native.py')
    }
    for filename, expected in lock['files'].items():
        # The calibration predates retirement; verify its original source snapshots.
        # Current retained-scope sources join the new whole-study lock separately.
        file = historical_sources.get(path(filename), path(filename))
        assert digest(file) == expected, ('Frozen plausibility calibration changed', filename)
        files.add(file)
    files.add(lock_file)
    calibration = read(DIRECTORY/'calibration.json')
    assert calibration['protocol_sha256'] == digest(DIRECTORY/'protocol.json')
    return calibration, files


def freeze_predictions(rows, calibration):
    assert len(rows) == 2400 and len({(r['scene_id'], r['object_id']) for r in rows}) == 2400
    value = dict(queries=2400, calibration_sha256=digest(DIRECTORY/'calibration.json'),
        historical_development_accuracy_labels_read=True, new_robustness_accuracy_labels_read=False,
        primary_rankings_modified=False, predictions=rows)
    destination = DIRECTORY/'predictions.json'
    if destination.exists():
        assert read(destination) == value, 'Frozen prospective plausibility flags changed'
    else:
        save(destination, value)
    return destination


def summarize_overlay(rows):
    assert rows and all(r['plausibility_eligible'] for r in rows)
    flags = [r['plausibility']['flag'] for r in rows]
    present = [r for r in rows if r['present']]
    absent = [r for r in rows if not r['present']]
    return dict(queries=len(rows), unavailable=flags.count('unavailable'),
        zero_passing_proposals=flags.count('zero_passing_proposals'),
        one_passing_proposal=flags.count('one_passing_proposal'),
        multiple_passing_proposals=flags.count('multiple_passing_proposals'),
        physically_present_queries=len(present), absent_queries=len(absent),
        present_primary_correct=sum(r['correct'] for r in present),
        present_primary_winner_passing=sum(r['plausibility_winner_passes'] for r in present),
        present_correct_winner_passing=sum(r['correct'] and r['plausibility_winner_passes'] for r in present),
        absent_primary_winner_passing=sum(r['plausibility_winner_passes'] for r in absent),
        certified_absence=0, certified_duplicate_detection=0)
