"""CPU-only exact comparison of a recorded own target and a runtime trial.

No GT, CUDA, model or inference imports. The only allowed numerical difference is
CSV runtime. Compressed NPZ bytes are not compared because ZIP metadata may vary.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

FIELDS = ('scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time')
SCHEMA = 'public-dino-model-reuse-trial-v1'


def local(value):
    value = str(value)
    if os.name != 'nt' and len(value) > 2 and value[1] == ':':
        value = '/mnt/' + value[0].lower() + '/' + value[3:].replace('\\', '/')
    elif os.name == 'nt' and value.startswith('/mnt/') and len(value) > 7:
        value = value[5].upper() + ':/' + value[7:]
    return Path(value).expanduser().resolve()


def read(path):
    return json.loads(local(path).read_text(encoding='utf-8-sig'))


def sha256(path):
    h = hashlib.sha256()
    with local(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def flags(command):
    if not isinstance(command, list) or len(command) < 4 or command[1] != '-B' or command[3] != 'run-localization':
        raise ValueError('Expected an original run-localization command')
    tokens = command[4:]
    if len(tokens) % 2:
        raise ValueError('Malformed original command flags')
    result = {}
    for key, value in zip(tokens[::2], tokens[1::2]):
        if not isinstance(key, str) or not key.startswith('--') or key in result or not isinstance(value, str):
            raise ValueError('Malformed or duplicated original command flag')
        result[key] = value
    if '--work-dir' not in result or '--output' not in result:
        raise ValueError('Original output/work-dir flags are required')
    return result


def _normalizer(before, after):
    pairs = sorted(((after[k], before[k]) for k in ('--work-dir', '--output')), key=lambda p: -len(p[0]))
    def normalize(value):
        if isinstance(value, str):
            for old, new in pairs:
                value = value.replace(old, new)
            return value
        if isinstance(value, list):
            return [normalize(v) for v in value]
        if isinstance(value, dict):
            return {k: normalize(v) for k, v in value.items()}
        return value
    return normalize


def _array(array, normalize):
    # Path metadata has naturally different Unicode/bytes widths in separate
    # trial folders. Normalize the strings before comparing their inferred dtype.
    if array.dtype.kind == 'U':
        return np.asarray([normalize(str(v)) for v in array.ravel()]).reshape(array.shape)
    if array.dtype.kind == 'S':
        return np.asarray([normalize(v.decode('utf-8')).encode('utf-8') for v in array.ravel()]).reshape(array.shape)
    return array


def compare_trial(baseline_receipt, trial_receipt, report_path=None):
    baseline_path, trial_path = local(baseline_receipt), local(trial_receipt)
    baseline, trial = read(baseline_path), read(trial_path)
    if baseline.get('status') != 'complete' or trial.get('status') != 'complete':
        raise ValueError('Both recorded attempts must be complete')
    if trial.get('schema') != SCHEMA:
        raise ValueError('Unknown runtime trial schema')
    if trial.get('baseline_receipt_sha256') != sha256(baseline_path):
        raise ValueError('Trial does not bind this exact baseline receipt')
    if not baseline.get('source_pins') or trial.get('source_pins') != baseline['source_pins']:
        raise ValueError('Source/configuration/model pins differ')
    before, after = flags(baseline['command']), flags(trial['command'])
    if baseline['command'][:4] != trial['command'][:4] or {k: v for k, v in before.items() if k not in ('--work-dir', '--output')} != {k: v for k, v in after.items() if k not in ('--work-dir', '--output')}:
        raise ValueError('Only output/work-dir paths may differ between commands')
    base_work, trial_work = local(before['--work-dir']), local(after['--work-dir'])
    base_csv, trial_csv = local(before['--output']), local(after['--output'])
    if base_work == trial_work or base_csv == trial_csv or base_work in trial_work.parents or trial_work in base_work.parents:
        raise ValueError('Trial and baseline payloads must be separate folders')
    if local(baseline.get('original_csv', base_csv)) != base_csv:
        raise ValueError('Baseline CSV differs from its original command')
    outputs = baseline.get('outputs')
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError('Baseline has no hash-bound payloads')
    for name, wanted in outputs.items():
        if sha256(name) != wanted:
            raise ValueError('A hash-bound baseline payload changed: ' + name)
    target = baseline.get('target', {})
    try:
        suffix = 'scene_{scene_id:06d}_im_{im_id:06d}_obj_{obj_id:06d}/result.json'.format(**target)
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError('Baseline target identity is incomplete') from exc
    base_result, trial_result = base_work / suffix, trial_work / suffix
    if local(baseline.get('original_result', base_result)) != base_result:
        raise ValueError('Baseline result differs from the command target')
    required_bound = {str(local(name)) for name in outputs}
    if str(base_csv) not in required_bound or str(base_result) not in required_bound:
        raise ValueError('Baseline CSV/result are not hash-bound')
    normalize = _normalizer(before, after)
    a_result, b_result = read(base_result), read(trial_result)
    issues, arrays, npz_files, masks = [], [], [], []
    if a_result != normalize(b_result):
        issues.append('Original candidate/selected result JSON differs after output-path normalization')
    for result, work, label in ((a_result, base_work, 'baseline'), (b_result, trial_work, 'trial')):
        for key in ('scene_id', 'im_id', 'obj_id'):
            if result.get(key) != target[key]:
                raise ValueError(label + ' result target differs')
        if result.get('instance_count') != target.get('inst_count'):
            raise ValueError(label + ' requested instance count differs')
        candidates = result.get('candidates')
        if not isinstance(candidates, list) or len(candidates) != result.get('candidate_count'):
            raise ValueError(label + ' candidate accounting is malformed')
        for candidate in candidates:
            if candidate.get('status') not in ('valid', 'invalid'):
                raise ValueError(label + ' has an unknown candidate status')
            for field in ('target_cache_path', 'coarse_cache_path', 'fine_cache_path', 'mask_path'):
                if field not in candidate:
                    raise ValueError(label + ' candidate is missing ' + field)
                p = local(candidate[field])
                if work not in p.parents:
                    raise ValueError(label + ' candidate payload escapes its work folder')
                if candidate['status'] == 'valid' and not p.is_file():
                    issues.append(label + ' valid candidate payload is missing: ' + field)
    a_files = {p.relative_to(base_work).as_posix(): p for p in base_work.rglob('*') if p.is_file()}
    b_files = {p.relative_to(trial_work).as_posix(): p for p in trial_work.rglob('*') if p.is_file()}
    if set(a_files) != set(b_files):
        issues.append('Saved payload sets differ: baseline_only=' + repr(sorted(set(a_files) - set(b_files))) + '; trial_only=' + repr(sorted(set(b_files) - set(a_files))))
    for name in sorted(set(a_files) & set(b_files)):
        p, q = a_files[name], b_files[name]
        if p.suffix == '.npz':
            npz_files.append(name)
            try:
                with np.load(p, allow_pickle=False) as av, np.load(q, allow_pickle=False) as bv:
                    if set(av.files) != set(bv.files):
                        issues.append(name + ': array keys differ')
                    for key in sorted(set(av.files) & set(bv.files)):
                        aa, bb = _array(av[key], lambda x: x), _array(bv[key], normalize)
                        ok = aa.shape == bb.shape and aa.dtype == bb.dtype
                        if ok:
                            ok = np.array_equal(aa, bb, equal_nan=True) if aa.dtype.kind in 'fc' else np.array_equal(aa, bb)
                        arrays.append({'file': name, 'key': key, 'shape': list(aa.shape), 'dtype': str(aa.dtype), 'equal': bool(ok)})
                        if not ok:
                            issues.append(name + ':' + key + ': array values, dtype, or shape differ')
            except (OSError, ValueError, TypeError, KeyError) as exc:
                issues.append(name + ': NPZ comparison failed: ' + str(exc))
        elif p.suffix.lower() == '.png':
            equal = sha256(p) == sha256(q)
            masks.append({'file': name, 'equal': equal})
            if not equal:
                issues.append(name + ': saved mask bytes differ')
        elif p == base_result:
            pass
        else:
            issues.append('Unknown saved payload: ' + name)
    def csv_rows(path):
        with path.open(newline='', encoding='utf-8-sig') as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ValueError('Original CSV field names differ')
            rows = list(reader)
        for row in rows:
            if not math.isfinite(float(row['time'])) or float(row['time']) < 0:
                raise ValueError('Original CSV time is invalid')
        return rows
    a_rows, b_rows = csv_rows(base_csv), csv_rows(trial_csv)
    a_poses = [{k: v for k, v in row.items() if k != 'time'} for row in a_rows]
    b_poses = [{k: v for k, v in row.items() if k != 'time'} for row in b_rows]
    if a_poses != b_poses:
        issues.append('Original CSV pose fields differ; only time may differ')
    report = {'schema': 'public-runtime-trial-exact-comparison-v1',
              'status': 'exact_parity' if not issues else 'different', 'exact_parity': not issues,
              'baseline_receipt': str(baseline_path), 'baseline_receipt_sha256': sha256(baseline_path),
              'trial_receipt': str(trial_path), 'trial_receipt_sha256': sha256(trial_path),
              'source_pins': baseline['source_pins'], 'target': target,
              'candidate_count': a_result['candidate_count'], 'selected_count': len(a_result.get('selected', [])),
              'NPZ_files_compared': npz_files, 'arrays': arrays, 'masks': masks, 'issues': issues,
              'allowed_differences': ['Trial output/work-directory paths', 'CSV time field and process wall time', 'NPZ ZIP metadata; arrays compared exactly with equal NaNs'],
              'timing': {'baseline_wall_s': baseline.get('runtime_s'), 'trial_wall_s': trial.get('runtime_s'),
                         'baseline_csv_times': [float(row['time']) for row in a_rows],
                         'trial_csv_times': [float(row['time']) for row in b_rows]},
              'GT_access': False, 'GPU_or_model_inference': False, 'benchmark_outputs_changed': False,
              'helper_sha256': sha256(__file__)}
    if report_path is not None:
        path = local(report_path)
        if path in (baseline_path, trial_path) or base_work in path.parents or trial_work in path.parents:
            raise ValueError('Comparison report must be outside both payload folders and receipts')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-receipt', type=Path, required=True)
    parser.add_argument('--trial-receipt', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args(argv)
    result = compare_trial(args.baseline_receipt, args.trial_receipt, args.report)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['exact_parity'] else 1


if __name__ == '__main__':
    raise SystemExit(main())