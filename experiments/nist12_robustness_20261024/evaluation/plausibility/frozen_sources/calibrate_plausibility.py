"""One prospective calibration using historical development artifacts only."""
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import HERE, PRIOR, REPO, digest, path, read, relative, save
from plausibility import DIRECTORY, development_threshold, finite, overlay
from evaluate import bbox_half_open, box_iou

HISTORICAL_AUDIT = REPO/'experiments/nist12_direct_cad_vlm_20261019'


def require_prospective():
    assert not (HERE/'evaluation/prediction_lock.json').exists(), 'Whole-study predictions already locked'
    assert not (HERE/'evaluation/audit_access.json').exists(), 'New robustness accuracy annotation access already recorded'
    assert not (HERE/'evaluation/results/completion.json').exists(), 'New robustness evaluation already completed'


def frozen_protocol():
    protocol = read(DIRECTORY/'protocol.json')
    for filename, expected in protocol['input_and_source_sha256'].items():
        assert digest(path(filename)) == expected, filename
    return protocol


def prepare():
    require_prospective()
    if (DIRECTORY/'protocol.json').exists():
        frozen_protocol()
        print('Existing prospective calibration protocol verified.', flush=True)
        return
    inputs = read(PRIOR/'inputs.json')
    shortlists = read(PRIOR/'ours/shortlists.json')
    assert len(inputs['scenes']) == 8 and len(inputs['objects']) == 12 and len(shortlists) == 96
    assert [s['scene_id'] for s in inputs['scenes']] == [f'N{i:02d}' for i in range(8)]
    known = read(HERE/'catalog.json')['objects']
    assert {r['target'] for r in known} == {r['target'] for r in inputs['objects']}
    files = {PRIOR/'inputs.json', PRIOR/'ours/shortlists.json', PRIOR/'ours/geometry_protocol.json',
             HERE/'protocol.json', HERE/'catalog.json'}
    source_names = ['evaluation/plausibility.py', 'evaluation/calibrate_plausibility.py',
                    'evaluation/test_plausibility.py', 'evaluation/evaluate.py',
                    'evaluation/test_evaluate.py', 'evaluation/render_report.py',
                    'baselines/evaluate_native.py']
    files.update(HERE/name for name in source_names)
    snapshot = DIRECTORY/'frozen_sources'
    snapshot.mkdir(parents=True, exist_ok=True)
    for name in source_names:
        destination = snapshot/Path(name).name
        assert not destination.exists(), 'Prospective source snapshot already exists'
        shutil.copyfile(HERE/name, destination)
        files.add(destination)
    geometry = REPO/'experiments/nist12_qwen_clip_reranking_20261023/depth_verification_20261023/geometry.py'
    files.add(geometry)
    audit_paths = []
    for scene in inputs['scenes']:
        sid = scene['scene_id']
        files.add(PRIOR/'ours'/sid/'localization.json')
        label_file = HISTORICAL_AUDIT/'scenes'/sid/'capture_audit.json'
        assert label_file.is_file()
        audit_paths.append(relative(label_file))
    fits = set()
    for task in shortlists:
        assert task['status'] == 'ok' and len(task['candidate_ids']) == 5
        for pid in task['candidate_ids']:
            fits.add(PRIOR/'ours/fits'/f'{task["scene_id"]}__{task["target"]}__p{pid:03d}.json')
    assert len(fits) == 480 and all(f.is_file() for f in fits)
    files.update(fits)
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        frozen_before_historical_calibration_label_parsing=True,
        historical_development_accuracy_labels_read=False, new_robustness_accuracy_labels_read=False,
        historical_scenes=[f'N{i:02d}' for i in range(8)], queries=96, candidate_pairs=480,
        known_targets=[r['target'] for r in known], historical_audit_files=audit_paths,
        labels='Historical requested-CAD bbox IoU >=0.5 is target; lower IoU is non-target. Inclusive bbox coordinates converted to half-open edges. Require visible development target and valid bbox.',
        finite_scores='Use only saved contact_0mm best-geometry scores. Missing/nonfinite scores remain unavailable, never zero-filled.',
        tau='float64 nextafter(max finite bbox-non-target development score,+infinity); pass iff finite score>=tau. No clipping, coverage tuning or held-out threshold selection.',
        missing_negative_case='If no finite non-target development score exists, calibration is unavailable and tau=null; no fallback threshold.',
        historical_negatives='Retain every shortlisted negative, including old gripper regions; no semantic or outcome-based exclusion.',
        flags='With usable top-five scores: zero, one or multiple passing proposals. Otherwise unavailable, with original retrieval/fit statuses retained. Proposal counts are not physical instance counts.',
        confirmatory_scope='Only known12 CADs, simulation/depth_stress cohorts, CLIP-top-five-depth arm; 4080 queries. No U/T or other-arm extension.',
        primary='No primary ranking, abstention rule, input, model, fitting or prediction is changed.',
        claims='Zero observed development non-target pair passes is an empirical development property only; positive coverage may be zero. No certified absence, duplicate detection or generalization guarantee.',
        coverage='Report target/non-target counts, finite/unavailable counts, passing counts and rates using all-pair and finite-pair denominators, even when zero.',
        geometry_source_sha256=digest(geometry),
        input_and_source_sha256={relative(f):digest(f) for f in sorted(files)})
    save(DIRECTORY/'protocol.json', protocol)
    print('Prospective supplementary protocol frozen before historical audit label parsing.', flush=True)


def calibrate():
    require_prospective()
    assert not (DIRECTORY/'calibration_lock.json').exists(), 'Calibration already frozen; do not overwrite'
    protocol = frozen_protocol()
    inputs = read(PRIOR/'inputs.json')
    shortlists = read(PRIOR/'ours/shortlists.json')
    truths, proposals, annotations = {}, {}, {}
    # First semantic label access occurs only after the frozen protocol is verified.
    save(DIRECTORY/'development_label_access.json', dict(first_historical_label_parsing_not_before_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=digest(DIRECTORY/'protocol.json'), historical_development_accuracy_labels_read=True,
        new_robustness_accuracy_labels_read=False, source_sha256=digest(__file__)))
    for scene in inputs['scenes']:
        sid = scene['scene_id']
        label_file = HISTORICAL_AUDIT/'scenes'/sid/'capture_audit.json'
        annotations[relative(label_file)] = digest(label_file)
        truths[sid] = read(label_file)['audit_only_objects']
        proposals[sid] = {int(r['proposal_id']):r for r in read(PRIOR/'ours'/sid/'localization.json')['proposals']}
    pairs, query_rows = [], []
    for task in shortlists:
        sid, target = task['scene_id'], task['target']
        truth = truths[sid][target]
        assert truth['visible_pixels'] > 0 and len(truth['bbox']) == 4
        target_box = bbox_half_open(truth['bbox'])
        for pid in task['candidate_ids']:
            fit_file = PRIOR/'ours/fits'/f'{sid}__{target}__p{pid:03d}.json'
            fit = read(fit_file)
            assert fit['geometry_source_sha256'] == protocol['geometry_source_sha256']
            assert (fit['scene_id'], fit['target'], fit['proposal_id']) == (sid, target, pid)
            assert len(fit['geometry']['arms']) == 1
            arm = fit['geometry']['arms'][0]
            assert arm['clearance_m'] == 0.
            best = arm.get('best_geometry')
            raw_score = None if best is None else best['verification']['geometry_score']
            score = float(raw_score) if finite(raw_score) else None
            overlap = box_iou(bbox_half_open(proposals[sid][pid]['bbox_xyxy']), target_box)
            pairs.append(dict(scene_id=sid, target=target, proposal_id=pid, score=score,
                bbox_iou=overlap, bbox_target=overlap >= .5,
                status=arm['status'] if raw_score is None or finite(raw_score) else 'unavailable_nonfinite_saved_score',
                unavailable_raw_score_repr=repr(raw_score) if raw_score is not None and not finite(raw_score) else None,
                fit_file=relative(fit_file)))
    assert len(pairs) == 480
    calibration = development_threshold(pairs)
    calibration.update(known_targets=protocol['known_targets'], protocol_sha256=digest(DIRECTORY/'protocol.json'),
        historical_development_accuracy_labels_read=True, new_robustness_accuracy_labels_read=False,
        model_calls=0, new_fits=0, heldout_coverage_used_for_tuning=False,
        original_main_protocol_sha256=digest(HERE/'protocol.json'), development_annotation_sha256=annotations)
    for row in pairs:
        score = row['score']
        row['passes'] = calibration['tau'] is not None and score is not None and score >= calibration['tau']
    for task in shortlists:
        relevant = [dict(r, geometry_score=r['score']) for r in pairs if r['scene_id'] == task['scene_id'] and r['target'] == task['target']]
        flag = overlay(task, relevant, calibration)
        query_rows.append(dict(scene_id=task['scene_id'], target=task['target'], **flag))
    save(DIRECTORY/'development_pairs.json', pairs)
    save(DIRECTORY/'development_queries.json', query_rows)
    save(DIRECTORY/'calibration.json', calibration)
    files = dict(protocol['input_and_source_sha256'])
    files.update(annotations)
    for name in ('protocol.json', 'development_label_access.json', 'development_pairs.json', 'development_queries.json', 'calibration.json'):
        files[relative(DIRECTORY/name)] = digest(DIRECTORY/name)
    save(DIRECTORY/'calibration_lock.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        historical_development_accuracy_labels_read=True, new_robustness_accuracy_labels_read=False,
        primary_predictions_modified=False, files=files))
    print('Historical development-only calibration frozen; no NEW robustness accuracy annotation read.', flush=True)
    print({key:calibration[key] for key in ('tau','pairs','target_pairs_passing','non_target_pairs_passing',
                                           'target_coverage_all_pairs','non_target_pass_rate_all_pairs')}, flush=True)


if __name__ == '__main__':
    assert sys.argv[1:] in (['prepare'], ['calibrate'])
    {'prepare':prepare, 'calibrate':calibrate}[sys.argv[1]]()
