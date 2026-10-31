import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_runtime_trial as m


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def csv_file(path, *, time='10', score='.3'):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=m.FIELDS)
        writer.writeheader()
        writer.writerow({'scene_id': '1', 'im_id': '2', 'obj_id': '3', 'score': score,
                         'R': '1 0 0 0 1 0 0 0 1', 't': '0 0 100', 'time': time})


@pytest.fixture
def case(tmp_path):
    baseline_dir = tmp_path / 'baseline'
    trial_dir = tmp_path / 'trial_output_different_path_length'
    baseline_dir.mkdir()
    trial_dir.mkdir()
    commands = []
    results = []
    payloads = []
    for folder, runtime in ((baseline_dir, '10'), (trial_dir, '3')):
        work = folder / 'original'
        stage = work / 'scene_000001_im_000002_obj_000003'
        stage.mkdir(parents=True)
        command = ['python', '-B', '/public/run_bop.py', 'run-localization', '--dataset', 'lmo',
                   '--work-dir', str(work), '--output', str(folder / 'original.csv')]
        commands.append(command)
        candidates = []
        for i, status in ((0, 'valid'), (1, 'invalid')):
            prefix = stage / f'candidate_{i:03d}'
            row = {'candidate_index': i, 'status': status, 'mask_path': str(prefix) + '_mask.png',
                   'target_cache_path': str(prefix) + '_target.npz', 'coarse_cache_path': str(prefix) + '_coarse.npz',
                   'fine_cache_path': str(prefix) + '_fine.npz', 'nms_selected': i == 0,
                   'nms_status': 'selected' if i == 0 else 'invalid'}
            Path(row['mask_path']).write_bytes(b'fixture-mask')
            if i == 0:
                row.update(R=np.eye(3).tolist(), t_mm=[0, 0, 100], final_score=.3)
                for key in ('target_cache_path', 'coarse_cache_path', 'fine_cache_path'):
                    np.savez_compressed(row[key], values=np.array([1., np.nan, 2.]),
                                        input_path=np.array(str(prefix) + '_target.npz'), count=np.int32(3))
            else:
                row['error'] = 'RuntimeError: GeDi output contains non-finite values at scale 0.30'
            candidates.append(row)
        result = {'dataset': 'lmo', 'scene_id': 1, 'im_id': 2, 'obj_id': 3, 'instance_count': 1,
                  'candidate_count': 2, 'candidates': candidates,
                  'selected': [{k: candidates[0][k] for k in ('candidate_index', 'R', 't_mm', 'final_score')}],
                  'output': str(stage / 'result.json'), 'work_dir': str(stage)}
        result_path = stage / 'result.json'
        dump(result_path, result)
        csv_file(folder / 'original.csv', time=runtime)
        results.append(result_path)
        payloads.append({str(p): m.sha256(p) for p in work.rglob('*') if p.is_file()})
        payloads[-1][str(folder / 'original.csv')] = m.sha256(folder / 'original.csv')
    baseline_path = baseline_dir / 'receipt.json'
    baseline = {'status': 'complete', 'source_pins': {'public_revision': 'test', 'configuration_sha256': 'fixed'},
                'command': commands[0], 'target': {'scene_id': 1, 'im_id': 2, 'obj_id': 3, 'inst_count': 1},
                'original_csv': str(baseline_dir / 'original.csv'), 'original_result': str(results[0]),
                'outputs': payloads[0], 'runtime_s': 12}
    dump(baseline_path, baseline)
    trial_path = trial_dir / 'trial_report.json'
    trial = {'schema': m.SCHEMA, 'status': 'complete', 'source_pins': baseline['source_pins'],
             'baseline_receipt_sha256': m.sha256(baseline_path), 'command': commands[1], 'runtime_s': 5}
    dump(trial_path, trial)
    return {'baseline': baseline_path, 'trial': trial_path, 'results': results,
            'commands': commands, 'tmp': tmp_path}


def compare(case, **kwargs):
    return m.compare_trial(case['baseline'], case['trial'], **kwargs)


def change_json(path, function):
    value = m.read(path)
    function(value)
    dump(path, value)


def change_npz(path, function):
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    function(arrays)
    np.savez_compressed(path, **arrays)


def trial_stage(case):
    return case['results'][1].parent


def test_exact_arrays_path_metadata_and_nan_with_time_difference(case):
    result = compare(case)
    assert result['exact_parity'] and result['status'] == 'exact_parity'
    assert len(result['arrays']) == 9 and len(result['NPZ_files_compared']) == 3
    assert result['timing']['baseline_csv_times'] == [10.]
    assert result['timing']['trial_csv_times'] == [3.]
    assert not result['GT_access'] and not result['GPU_or_model_inference']


@pytest.mark.parametrize('mutation', ['value', 'shape', 'dtype', 'missing_key', 'extra_key', 'object'])
def test_changed_or_unknown_npz_arrays_fail(case, mutation):
    p = trial_stage(case) / 'candidate_000_target.npz'
    def modify(a):
        if mutation == 'value': a['values'][0] += 1e-15
        elif mutation == 'shape': a['values'] = a['values'].reshape(1, 3)
        elif mutation == 'dtype': a['values'] = a['values'].astype(np.float32)
        elif mutation == 'missing_key': a.pop('count')
        elif mutation == 'extra_key': a['unknown'] = np.int32(3)
        else: a['unknown'] = np.array([{'bad': 1}], dtype=object)
    change_npz(p, modify)
    assert not compare(case)['exact_parity']


@pytest.mark.parametrize('mutation', ['score', 'error', 'nms', 'rotation', 'extra_key'])
def test_original_result_is_exact_including_errors_nms_and_unknown_keys(case, mutation):
    def modify(v):
        if mutation == 'score': v['selected'][0]['final_score'] += 1e-15
        elif mutation == 'error': v['candidates'][1]['error'] = 'different failure'
        elif mutation == 'nms': v['candidates'][0]['nms_selected'] = False
        elif mutation == 'rotation': v['selected'][0]['R'][0][0] += 1e-15
        else: v['extra'] = 1
    change_json(case['results'][1], modify)
    assert not compare(case)['exact_parity']


def test_invalid_candidate_missing_in_both_is_normal_but_one_extra_fails(case):
    assert compare(case)['exact_parity']
    np.savez_compressed(trial_stage(case) / 'candidate_001_target.npz', values=np.array([1.]))
    assert not compare(case)['exact_parity']


def test_missing_valid_candidate_cache_fails(case):
    (trial_stage(case) / 'candidate_000_fine.npz').unlink()
    assert not compare(case)['exact_parity']


def test_changed_mask_fails(case):
    (trial_stage(case) / 'candidate_000_mask.png').write_bytes(b'different')
    assert not compare(case)['exact_parity']


def test_changed_csv_pose_fails(case):
    csv_file(case['trial'].parent / 'original.csv', score='.30000001')
    assert not compare(case)['exact_parity']


def test_baseline_hash_mutation_rejected(case):
    (case['results'][0].parent / 'candidate_000_mask.png').write_bytes(b'changed')
    with pytest.raises(ValueError, match='baseline payload changed'):
        compare(case)


@pytest.mark.parametrize('mutation', ['pins', 'command', 'receipt_hash', 'state', 'schema'])
def test_trial_identity_and_finished_state_required(case, mutation):
    def modify(v):
        if mutation == 'pins': v['source_pins'] = {'configuration_sha256': 'different'}
        elif mutation == 'command': v['command'][5] = 'different_dataset'
        elif mutation == 'receipt_hash': v['baseline_receipt_sha256'] = 'wrong'
        elif mutation == 'state': v['status'] = 'running'
        else: v['schema'] = 'unknown'
    change_json(case['trial'], modify)
    with pytest.raises(ValueError): compare(case)


def test_optional_report_is_new_file_only(case):
    report = case['tmp'] / 'comparison.json'
    compare(case, report_path=report)
    assert m.read(report)['exact_parity']
    with pytest.raises(FileExistsError): compare(case, report_path=report)
    with pytest.raises(ValueError): compare(case, report_path=trial_stage(case) / 'comparison.json')