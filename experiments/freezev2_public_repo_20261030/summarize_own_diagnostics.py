"""Supplemental CPU-only snapshots of recorded own-inference diagnostics.

Reads small JSON receipts and original candidate reports, never descriptor arrays,
models, RGB-D or GT. Writes only new timestamped files under own_output/diagnostics.
A completed attempt with zero selected poses is not a saved prediction. Recorded
finite/alignment checks are identified as such; this is not a new heavy audit or
an official evaluator, and it never replaces the live ledger/CSV/score summary.
"""
from __future__ import annotations
import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

CLASSIFICATIONS = ('complete_with_saved_poses', 'complete_zero_selected', 'failed', 'blocked',
                   'pending', 'missing', 'unreadable_receipt', 'identity_mismatch', 'invalid_receipt')
TERMINAL = frozenset(CLASSIFICATIONS[:4])
STAGES = ('geometry', 'visual', 'gedi', 'fuse')


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_json(path):
    path = Path(path)
    data = path.read_bytes()
    return json.loads(data.decode('utf-8-sig')), {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest()}


def number(value):
    return float(value) if not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 else None


def key(row):
    fields = ('scene_id', 'im_id', 'obj_id', 'inst_count')
    if not isinstance(row, dict) or any(isinstance(row.get(k), bool) or not isinstance(row.get(k), int) for k in fields):
        raise ValueError('Official targets must contain integer identities and instance counts')
    if row['scene_id'] < 0 or row['im_id'] < 0 or row['obj_id'] < 1 or row['inst_count'] < 0:
        raise ValueError('Invalid official target identity/count')
    return tuple(row[k] for k in fields[:3])


def folder_name(row):
    return f's{row["scene_id"]:06d}_i{row["im_id"]:06d}_o{row["obj_id"]:06d}'


def inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Original own-result path escapes this own-inference output')
    return path


def candidate_report(case, receipt, own_root):
    """Read an original report only when its bytes match the case receipt."""
    value = receipt.get('original_result')
    if not value:
        case['original_result'] = {'status': 'not_recorded'}
        return
    try:
        path = inside(value, own_root)
        data = path.read_bytes()
        identity = {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest()}
        wanted = receipt.get('outputs', {}).get(str(path))
        case['original_result'] = dict(identity, matches_receipt_hash=identity['sha256'] == wanted)
        if identity['sha256'] != wanted:
            case['issues'].append('Original candidate report hash differs or is absent from case receipt')
            return
        report = json.loads(data.decode('utf-8-sig'))
        if not isinstance(report, dict):
            raise ValueError('Original candidate report must contain an object')
        if any(report.get(k) != case['target'][k] for k in ('scene_id', 'im_id', 'obj_id')):
            case['issues'].append('Original candidate report target differs')
            return
        if report.get('instance_count') != case['requested_instances']:
            case['issues'].append('Original candidate report requested instances differ')
            return
        candidates = report.get('candidates')
        if not isinstance(candidates, list) or any(not isinstance(c, dict) for c in candidates):
            case['issues'].append('Original candidate report does not contain a valid candidates list')
            return
        selected = report.get('selected')
        if case['recorded_status'] == 'complete' and (not isinstance(selected, list) or report.get('selected_count') != len(selected) or len(selected) != case['selected_instances_recorded']):
            case['issues'].append('Original candidate selection count differs from case receipt')
            return
        case['original_result']['status'] = 'hash_verified_small_JSON'
        case['candidate_counts'] = {'reports_verified': 1, 'valid': 0, 'invalid': 0, 'unknown': 0}
        for candidate in candidates:
            status = candidate.get('status')
            case['candidate_counts'][status if status in ('valid', 'invalid') else 'unknown'] += 1
            if status == 'invalid':
                case['invalid_candidates'].append({
                    'candidate_index': candidate.get('candidate_index'),
                    'error': candidate.get('error') if isinstance(candidate.get('error'), str) else '(original error unavailable)',
                    'mask_path': candidate.get('mask_path'),
                    'target_cache_path': candidate.get('target_cache_path')})
        # Coarse/refined scores and pose arrays are intentionally not recomputed.
    except (OSError, ValueError, TypeError) as exc:
        case.setdefault('original_result', {'path': str(value)})
        case['original_result']['status'] = 'unreadable_or_outside_own_output'
        case['issues'].append(type(exc).__name__ + ': ' + str(exc))


def target_case(row, receipt_path, exists, pins, own_root):
    case = {'target': dict(row), 'classification': 'missing', 'recorded_status': None,
            'requested_instances': row['inst_count'], 'selected_instances_recorded': 0,
            'receipt': {'path': str(receipt_path), 'sha256': None}, 'identity_matches': None,
            'guarded_runtime_s': None, 'original_localization_time_s': None,
            'candidate_counts': {'reports_verified': 0, 'valid': 0, 'invalid': 0, 'unknown': 0},
            'invalid_candidates': [], 'issues': []}
    if not exists:
        return case
    try:
        data = Path(receipt_path).read_bytes()
        case['receipt'] = {'path': str(receipt_path), 'sha256': hashlib.sha256(data).hexdigest()}
        receipt = json.loads(data.decode('utf-8-sig'))
        if not isinstance(receipt, dict):
            raise ValueError('Receipt must contain an object')
    except (OSError, ValueError, TypeError) as exc:
        case['classification'] = 'unreadable_receipt'
        case['issues'].append(type(exc).__name__ + ': ' + str(exc))
        return case
    case['recorded_status'] = receipt.get('status')
    case['guarded_runtime_s'] = number(receipt.get('runtime_s'))
    case['reason_recorded'] = receipt.get('reason')
    case['completed_utc_recorded'] = receipt.get('completed_utc')
    case['identity_matches'] = receipt.get('source_pins') == pins
    if not case['identity_matches']:
        case['classification'] = 'identity_mismatch'
        return case
    status = receipt.get('status')
    case['selected_count_in_receipt'] = receipt.get('selected_count')
    case['pose_entries_in_receipt'] = len(receipt['poses']) if isinstance(receipt.get('poses'), list) else None
    if receipt.get('target') != row and not (status in ('running', 'pending') and receipt.get('target') is None):
        case['classification'] = 'invalid_receipt'
        case['issues'].append('Receipt target/count differs from the exact official row')
        return case
    status = receipt.get('status')
    if status == 'complete':
        poses, count = receipt.get('poses'), receipt.get('selected_count')
        if not isinstance(poses, list) or isinstance(count, bool) or not isinstance(count, int) or count != len(poses) or not 0 <= count <= row['inst_count']:
            case['classification'] = 'invalid_receipt'
            case['issues'].append('Completed receipt selected count differs from saved poses or requested count')
            return case
        case['classification'] = 'complete_with_saved_poses' if count else 'complete_zero_selected'
        case['selected_instances_recorded'] = count
        times = [number(pose.get('time')) if isinstance(pose, dict) else None for pose in poses]
        if times and None not in times and all(t == times[0] for t in times):
            case['original_localization_time_s'] = times[0]
    elif status in ('failed', 'blocked'):
        case['classification'] = status
    elif status in ('running', 'pending'):
        case['classification'] = 'pending'
    else:
        case['classification'] = 'invalid_receipt'
        case['issues'].append('Unknown receipt status')
    if case['classification'] in TERMINAL:
        candidate_report(case, receipt, own_root)
    return case


def cad_record(output, dataset, obj, pins):
    result = {'obj_id': obj, 'stages': {}, 'retained_points': None, 'descriptor_shapes': {},
              'geometry_original_runtime_s': None, 'geometry_guarded_or_adopted_runtime_s': None}
    complete = []
    retained = []
    for stage in STAGES:
        path = output / 'stages' / dataset / f'obj_{obj:06d}' / stage / 'receipt.json'
        record = {'status': 'missing', 'receipt': {'path': str(path), 'sha256': None},
                  'identity_matches': None, 'runtime_guarded_or_adopted_s': None,
                  'timing_scope': 'unavailable', 'recorded_diagnostics': {}, 'recorded_output_hashes': {}}
        result['stages'][stage] = record
        if not path.is_file():
            continue
        try:
            receipt, identity = read_json(path)
            if not isinstance(receipt, dict):
                raise ValueError('Stage receipt must contain an object')
            record.update(status=receipt.get('status'), receipt=identity,
                          identity_matches=receipt.get('source_pins') == pins)
            if not record['identity_matches']:
                record['status'] = 'identity_mismatch'
                continue
            diagnostics = receipt.get('diagnostics') or {}
            if not isinstance(diagnostics, dict):
                raise ValueError('Stage diagnostics must contain an object')
            record['recorded_diagnostics'] = diagnostics
            record['recorded_output_hashes'] = receipt.get('outputs', {})
            record['runtime_guarded_or_adopted_s'] = number(receipt.get('runtime_s'))
            adopted = receipt.get('adopted_original_geometry') or receipt.get('adopted_original_probe')
            record['timing_scope'] = 'adopted_original_probe_runtime' if adopted else 'guarded_stage_attempt_runtime'
            if stage == 'geometry':
                result['geometry_original_runtime_s'] = number(diagnostics.get('geometry_runtime_s'))
                result['geometry_guarded_or_adopted_runtime_s'] = record['runtime_guarded_or_adopted_s']
            elif record['status'] == 'complete':
                complete.append(stage)
                count = diagnostics.get('retained_points')
                result['descriptor_shapes'][stage] = diagnostics.get('feature_shape')
                if isinstance(count, int) and not isinstance(count, bool) and count > 0:
                    retained.append(count)
                    result['retained_points'] = count
        except (OSError, ValueError, TypeError) as exc:
            record['status'] = 'unreadable_receipt'
            record['error'] = type(exc).__name__ + ': ' + str(exc)
    result['finite_alignment_validation'] = {
        'evidence': 'recorded_complete_sealed_stage_validation', 'freshly_checked': False,
        'recorded_complete_stages': complete,
        'all_descriptor_stages_recorded_complete': complete == ['visual', 'gedi', 'fuse'],
        'retained_counts_agree': len(set(retained)) == 1 if len(retained) == 3 else None,
        'scope_note': 'Sealed stage completion recorded finite descriptors and point alignment; this helper reads only JSON metadata, not arrays or models.'}
    return result


def build_snapshot(config_path, observed_utc=None):
    """Observe current JSON receipts without modifying any inference/evaluation input."""
    config_path = Path(config_path).resolve()
    config, config_identity = read_json(config_path)
    output = Path(config['paths']['output']).resolve()
    manifest, manifest_identity = read_json(output / 'source_snapshot/source_manifest.json')
    run, run_identity = read_json(output / 'run_manifest.json')
    pins = run['source_pins']
    config_hash = canonical(config)
    if pins.get('configuration_sha256') != config_hash or manifest.get('configuration_sha256') != config_hash:
        raise ValueError('Current configuration differs from sealed run identity')
    if pins.get('orchestration_source_manifest_sha256') != manifest_identity['sha256']:
        raise ValueError('Sealed source manifest differs from run identity')
    if pins.get('public_revision') != config.get('public_revision'):
        raise ValueError('Public revision differs between configuration and run identity')
    assets = None
    assets_identity = None
    global_issues = []
    if config['paths'].get('assets_output'):
        assets, assets_identity = read_json(Path(config['paths']['assets_output']) / 'assets_manifest.json')
        if pins.get('assets_manifest_sha256') and pins['assets_manifest_sha256'] != assets_identity['sha256']:
            global_issues.append('Assets manifest metadata SHA differs from the run pin')
    helper_identity = {'path': str(Path(__file__).resolve()), 'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    snapshot = {'schema': 'supplemental-own-diagnostics-snapshot-v1',
        'observed_utc': observed_utc or dt.datetime.now(dt.timezone.utc).isoformat(),
        'identity': {'own_output_root': str(output), 'configuration': config_identity,
                     'configuration_canonical_sha256': config_hash, 'source_manifest': manifest_identity,
                     'run_manifest': run_identity, 'assets_manifest_metadata': assets_identity, 'recorded_source_pins': pins, 'diagnostic_helper': helper_identity},
        'GT_access': False, 'GPU_or_model_execution': False, 'official_outputs_modified': False,
        'fresh_check_scope': 'JSON metadata source/config/asset-target/case receipts and original candidate report SHA checks only; no model, CAD, descriptor or RGB-D re-audit.',
        'selected_count_scope': 'Counts selected poses from source-matched completed case receipts only; non-complete receipt counts remain separately recorded, never counted as correct predictions.',
        'candidate_status_scope': 'Only original status invalid counts as an invalid candidate; an ICP cap/nonconvergence flag alone is not classified as invalid.',
        'issues': global_issues,
        'snapshot_consistency': 'Live atomic receipts may advance during this observation; this is a timestamped partial snapshot, not a lock or proof of worker liveness.',
        'claim_boundary': 'Saved pose counts are not correct-pose counts, pose AR, mask AP or CAD classification accuracy.',
        'datasets': {}}
    all_cases = []
    for dataset in config['datasets']:
        target_path = Path(config['paths']['bop_root']) / dataset / 'test_targets_bop19.json'
        targets, target_identity = read_json(target_path)
        if not isinstance(targets, list):
            raise ValueError('Official target list must contain a list')
        dataset_issues = []
        expected_sha = ((assets or {}).get('datasets', {}).get(dataset, {}).get('target_sha256'))
        target_identity['expected_sha256'] = expected_sha
        target_identity['matches_pinned_SHA'] = target_identity['sha256'] == expected_sha if expected_sha else None
        if assets is not None and not target_identity['matches_pinned_SHA']:
            dataset_issues.append('Official target-list SHA differs from pinned assets metadata or target pin is absent')
        seen = set()
        for row in targets:
            k = key(row)
            if k in seen:
                raise ValueError('Duplicate official target key')
            seen.add(k)
        target_root = output / 'datasets' / dataset / 'targets'
        existing = {p.name: p for p in target_root.iterdir() if p.is_dir()} if target_root.is_dir() else {}
        expected_names = {folder_name(row) for row in targets}
        unexpected = sorted(set(existing) - expected_names)
        extra_receipts = []
        if unexpected:
            dataset_issues.append('Unexpected target directories exist outside the official target keys')
        for name in unexpected:
            receipt_path = existing[name] / 'receipt.json'
            extra = {'directory': str(existing[name]), 'receipt': {'path': str(receipt_path), 'sha256': None}}
            try:
                value, identity = read_json(receipt_path)
                extra.update(receipt=identity, recorded_target=value.get('target'), recorded_status=value.get('status'), identity_matches=value.get('source_pins') == pins)
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                extra['error'] = type(exc).__name__ + ': ' + str(exc)
            extra_receipts.append(extra)
        cases = []
        for row in targets:
            name = folder_name(row)
            receipt = target_root / name / 'receipt.json'
            cases.append(target_case(row, receipt, name in existing and receipt.is_file(), pins, output))
        counts = Counter(c['classification'] for c in cases)
        counts = {name: counts[name] for name in CLASSIFICATIONS}
        candidates = Counter()
        causes = Counter()
        for case in cases:
            candidates.update(case['candidate_counts'])
            causes.update(candidate['error'] for candidate in case['invalid_candidates'])
        zero_breakdown = Counter({'no_candidates': 0, 'all_candidates_invalid': 0, 'no_selected_despite_valid': 0, 'cause_unavailable': 0})
        for case in cases:
            if case['classification'] != 'complete_zero_selected':
                continue
            cc = case['candidate_counts']
            if not cc['reports_verified'] or cc['unknown']:
                zero_breakdown['cause_unavailable'] += 1
            elif not (cc['valid'] + cc['invalid']):
                zero_breakdown['no_candidates'] += 1
            elif not cc['valid']:
                zero_breakdown['all_candidates_invalid'] += 1
            else:
                zero_breakdown['no_selected_despite_valid'] += 1
        cads = [cad_record(output, dataset, obj, pins) for obj in sorted({t['obj_id'] for t in targets})]
        guarded = [c['guarded_runtime_s'] for c in cases if c['classification'] in TERMINAL and c['guarded_runtime_s'] is not None]
        original = [c['original_localization_time_s'] for c in cases if c['original_localization_time_s'] is not None]
        requested = sum(t['inst_count'] for t in targets)
        selected = sum(c['selected_instances_recorded'] for c in cases)
        snapshot['datasets'][dataset] = {'expected_targets': len(targets), 'requested_instances': requested,
            'selected_instances_recorded': selected, 'unselected_requested_instances_including_pending': requested-selected,
            'counts': counts, 'official_targets': target_identity, 'cases': cases,
            'unexpected_target_directories': unexpected, 'unexpected_target_receipts': extra_receipts, 'issues': dataset_issues,
            'candidate_totals': dict(candidates), 'invalid_candidate_causes': dict(sorted(causes.items())),
            'zero_selected_breakdown': dict(zero_breakdown),
            'targets_missing_requested_instances_after_complete': sum(c['classification'].startswith('complete_') and c['selected_instances_recorded'] < c['requested_instances'] for c in cases),
            'cad_onboarding': cads,
            'timing': {'target_guarded_runtime_s_sum': sum(guarded), 'target_guarded_runtime_count': len(guarded),
                'original_localization_time_s_sum': sum(original), 'original_localization_time_targets': len(original),
                'targets_without_original_localization_time': sum(counts[n] for n in TERMINAL)-len(original),
                'onboarding_original_geometry_runtime_s_sum': sum(c['geometry_original_runtime_s'] or 0 for c in cads),
                'onboarding_guarded_or_adopted_stage_runtime_s_sum': sum(s['runtime_guarded_or_adopted_s'] or 0 for c in cads for s in c['stages'].values()),
                'scope_note': 'Original localization times come once per completed target with saved poses, not once per instance. Zero-pose/failed attempts have guarded time only. Geometry inner-render time and guarded/adopted stage time are separate; missing values remain null and sums cover observed values only.'}}
        all_cases.extend(cases)
    snapshot['totals'] = {'expected_targets': len(all_cases),
        'requested_instances': sum(d['requested_instances'] for d in snapshot['datasets'].values()),
        'selected_instances_recorded': sum(c['selected_instances_recorded'] for c in all_cases),
        'counts': {name: sum(d['counts'][name] for d in snapshot['datasets'].values()) for name in CLASSIFICATIONS}}
    snapshot['coverage_complete'] = all(c['classification'] in TERMINAL for c in all_cases) and not global_issues and not any(d['issues'] for d in snapshot['datasets'].values())
    snapshot['partial'] = not snapshot['coverage_complete'] or any(c['issues'] for c in all_cases)
    snapshot['status'] = 'partial' if snapshot['partial'] else 'all_targets_terminal_diagnostic_snapshot'
    return snapshot


def markdown(snapshot):
    lines = ['# Supplemental own-inference diagnostics', '',
             '**Status: '+snapshot['status']+'**. Observed '+snapshot['observed_utc']+'.', '',
             'This is a receipt snapshot. Saved poses are not verified correct poses. No GT, GPU, model or official evaluator was used; official ledgers, CSVs and score summaries are unchanged.', '',
             '| Dataset | Official targets | With saved poses | Complete, zero poses | Failed | Blocked | Pending | Missing | Unreadable/identity/invalid | Instances requested / selected |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, data in snapshot['datasets'].items():
        c = data['counts']; bad=sum(c[k] for k in ('unreadable_receipt','identity_mismatch','invalid_receipt'))
        values = [name,data['expected_targets'],c['complete_with_saved_poses'],c['complete_zero_selected'],c['failed'],c['blocked'],c['pending'],c['missing'],bad,f'{data["requested_instances"]} / {data["selected_instances_recorded"]}']
        lines.append('| '+' | '.join(map(str,values))+' |')
    if snapshot['issues']:
        lines += ['', 'Snapshot identity issues: '+json.dumps(snapshot['issues'])]
    for name, data in snapshot['datasets'].items():
        if data['issues']:
            lines += ['', name.upper()+' accounting issues: '+json.dumps(data['issues']), 'Unexpected target directories: '+json.dumps(data['unexpected_target_directories'])]
    lines += ['', 'The unselected count includes work not attempted yet and is not an accuracy measurement. Every official target remains in the snapshot denominator.', '']
    for name, data in snapshot['datasets'].items():
        lines += ['## '+name.upper(), '', '| CAD | Recorded retained points | Recorded finite/alignment stage checks | Geometry original / guarded-or-adopted seconds |', '|---|---:|---|---|']
        for cad in data['cad_onboarding']:
            check=cad['finite_alignment_validation']; stages=', '.join(check['recorded_complete_stages']) or 'not recorded'
            lines.append(f'| {cad["obj_id"]} | {cad["retained_points"]} | {stages}; freshly checked: false | {cad["geometry_original_runtime_s"]} / {cad["geometry_guarded_or_adopted_runtime_s"]} |')
        lines += ['', 'Descriptor arrays were not reopened. Counts and finite/alignment evidence above come from source-pinned stage receipts.', '', 'Original invalid-candidate errors (exact text and case receipt identities are retained in the JSON):', '']
        lines += ['Zero-selected breakdown: '+json.dumps(data['zero_selected_breakdown'],sort_keys=True), '']
        if data['invalid_candidate_causes']:
            for error, count in data['invalid_candidate_causes'].items():
                lines.append('- '+str(count)+' × '+error.replace('\n',' ').replace('|','\\|'))
        else:
            lines.append('No invalid-candidate causes recorded in the hash-verified reports observed so far.')
        lines += ['', 'Timing: '+json.dumps(data['timing'],sort_keys=True), '']
    return '\n'.join(lines)+'\n'


def write_snapshot(snapshot, output_root):
    output_root = Path(output_root).resolve()
    if output_root != Path(snapshot['identity']['own_output_root']).resolve():
        raise ValueError('Snapshot output root differs from its recorded own-output identity')
    folder = output_root / 'diagnostics'
    if folder.exists() and not folder.resolve().is_relative_to(output_root):
        raise ValueError('Diagnostics directory escapes own-inference output')
    folder.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    stem = 'snapshot_'+stamp+'_'+uuid.uuid4().hex[:8]
    json_path, md_path = folder/(stem+'.json'), folder/(stem+'.md')
    for path, text in ((json_path,json.dumps(snapshot,indent=2,sort_keys=True,allow_nan=False)+'\n'),(md_path,markdown(snapshot))):
        temporary = folder/(path.name+'.tmp')
        with temporary.open('x',encoding='utf-8') as stream:
            stream.write(text)
        os.replace(temporary,path)
    return json_path, md_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    result = build_snapshot(args.config)
    json_path, md_path = write_snapshot(result, result['identity']['own_output_root'])
    print(json.dumps({'status':result['status'],'partial':result['partial'],'totals':result['totals'],
                      'JSON':str(json_path),'Markdown':str(md_path)},indent=2))
