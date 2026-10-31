"""Small offline fixtures for supplemental diagnostics; no inference/evaluation."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('summarize_own_diagnostics_under_test', HERE / 'summarize_own_diagnostics.py')
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def target_path(out, row):
    return out / 'datasets/lmo/targets' / f's{row["scene_id"]:06d}_i{row["im_id"]:06d}_o{row["obj_id"]:06d}' / 'receipt.json'


def pose():
    return {'score': 0.5, 'R': [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 't_mm': [0, 0, 500], 'time': 2.0}


@pytest.fixture
def case(tmp_path):
    out = tmp_path / 'own'
    view = tmp_path / 'GT_free_view'
    config = {'method_id': 'fixture-original-public-base', 'public_revision': 'public-source-pin',
              'datasets': ['lmo'], 'paths': {'output': str(out), 'bop_root': str(view),
              'geometry_root': str(tmp_path / 'geometry'), 'features_dir': str(out / 'features'),
              'assets_output': str(tmp_path / 'assets')},
              'settings': {'raw_points': 5000}}
    config_path = dump(out / 'configuration.json', config)
    manifest = dump(out / 'source_snapshot/source_manifest.json',
                    {'configuration_sha256': canonical(config), 'files': {'suite_runner.py': 'frozen-source-pin'}})
    pins = {'configuration_sha256': canonical(config), 'public_revision': 'public-source-pin',
            'orchestration_source_manifest_sha256': sha(manifest)}
    dump(out / 'run_manifest.json', {'configuration': config, 'source_pins': pins})
    rows = [{'scene_id': 2, 'im_id': i, 'obj_id': 1, 'inst_count': n}
            for i, n in zip(range(1, 7), (2, 1, 2, 1, 1, 2))]
    targets = dump(view / 'lmo/test_targets_bop19.json', rows)
    assets_manifest = dump(tmp_path / 'assets/assets_manifest.json',
                           {'datasets': {'lmo': {'target_sha256': sha(targets)}}})
    for row, status, poses in zip(rows[:5], ('complete', 'complete', 'failed', 'blocked', 'running'), ([pose()], [], [], [], [])):
        value = {'status': status, 'source_pins': dict(pins), 'target': row,
                 'selected_count': len(poses), 'poses': poses, 'runtime_s': 3.0,
                 'original_result': None, 'outputs': {}}
        if status in ('failed', 'blocked'):
            value['reason'] = f'fixture {status} reason'
        dump(target_path(out, row), value)
    ledger = dump(out / 'datasets/lmo/target_ledger.json', {'status': 'in_progress', 'sentinel': 'unchanged'})
    csv = out / 'datasets/lmo/freezev2public_lmo-test.csv'
    csv.write_text('scene_id,im_id,obj_id,score,R,t,time\n', encoding='utf-8')
    return {'out': out, 'view': view, 'config': config, 'config_path': config_path,
            'pins': pins, 'rows': rows, 'targets': targets, 'ledger': ledger, 'csv': csv,
            'assets_manifest': assets_manifest}


def snapshot(case):
    return d.build_snapshot(case['config_path'], observed_utc='2026-10-30T18:30:00+00:00')


def test_partial_accounting_keeps_every_official_target_and_requested_instance(case):
    result = snapshot(case)
    ds = result['datasets']['lmo']
    assert result['partial'] is True
    assert ds['expected_targets'] == 6
    assert ds['requested_instances'] == 9
    assert ds['selected_instances_recorded'] == 1
    assert ds['targets_missing_requested_instances_after_complete'] == 2
    assert ds['zero_selected_breakdown']['cause_unavailable'] == 1
    assert ds['counts']['complete_with_saved_poses'] == 1
    assert ds['counts']['complete_zero_selected'] == 1
    assert ds['counts']['failed'] == 1
    assert ds['counts']['blocked'] == 1
    assert ds['counts']['pending'] == 1
    assert ds['counts']['missing'] == 1
    assert len(ds['cases']) == 6
    assert sum(ds['counts'].values()) == 6


def test_zero_selected_complete_target_is_not_a_saved_prediction(case):
    result = snapshot(case)
    ds = result['datasets']['lmo']
    assert ds['counts']['complete_zero_selected'] == 1
    assert ds['cases'][1]['classification'] == 'complete_zero_selected'
    assert ds['cases'][1]['selected_instances_recorded'] == 0
    assert 'metrics' not in result


def test_unreadable_live_receipt_remains_partial_instead_of_disappearing(case):
    target_path(case['out'], case['rows'][0]).write_text('{interrupted', encoding='utf-8')
    ds = snapshot(case)['datasets']['lmo']
    assert ds['counts']['unreadable_receipt'] == 1
    assert ds['counts']['complete_with_saved_poses'] == 0
    assert ds['expected_targets'] == 6
    assert ds['selected_instances_recorded'] == 0


def test_wrong_source_configuration_cannot_count_as_current_saved_poses(case):
    p = target_path(case['out'], case['rows'][0])
    receipt = json.loads(p.read_text())
    receipt['source_pins']['configuration_sha256'] = 'different configuration'
    dump(p, receipt)
    result = snapshot(case)
    ds = result['datasets']['lmo']
    assert result['partial'] is True
    assert ds['counts']['identity_mismatch'] == 1
    assert ds['counts']['complete_with_saved_poses'] == 0
    assert ds['selected_instances_recorded'] == 0


def test_inconsistent_complete_selected_count_is_an_invalid_receipt(case):
    p = target_path(case['out'], case['rows'][0])
    receipt = json.loads(p.read_text())
    receipt['selected_count'] = 0
    dump(p, receipt)
    ds = snapshot(case)['datasets']['lmo']
    assert ds['counts']['invalid_receipt'] == 1
    assert ds['selected_instances_recorded'] == 0


@pytest.mark.parametrize('kind', ['missing', 'duplicate'])
def test_official_denominator_must_exist_and_have_unique_targets(case, kind):
    if kind == 'missing':
        case['targets'].unlink()
    else:
        dump(case['targets'], [case['rows'][0], copy.deepcopy(case['rows'][0])])
    with pytest.raises((ValueError, FileNotFoundError)):
        snapshot(case)


def test_writer_creates_only_supplemental_diagnostics_and_preserves_live_outputs(case):
    before = {p: p.read_bytes() for p in (case['ledger'], case['csv'])}
    result = snapshot(case)
    json_path, markdown_path = d.write_snapshot(result, case['out'])
    for p in (Path(json_path), Path(markdown_path)):
        assert p.is_file()
        assert p.parent == case['out'] / 'diagnostics'
    assert json.loads(Path(json_path).read_text())['partial'] is True
    for p, original in before.items():
        assert p.read_bytes() == original



def test_all_terminal_snapshot_is_complete_accounting_without_accuracy_claim(case):
    for row in case['rows'][4:]:
        dump(target_path(case['out'], row), {'status': 'blocked', 'source_pins': case['pins'],
             'target': row, 'selected_count': 0, 'poses': [], 'runtime_s': 0.0,
             'reason': 'unavailable CAD descriptor', 'original_result': None, 'outputs': {}})
    result = snapshot(case)
    assert result['partial'] is False
    assert result['status'] == 'all_targets_terminal_diagnostic_snapshot'
    assert result['datasets']['lmo']['selected_instances_recorded'] == 1
    assert 'metrics' not in result


def add_invalid_report(case):
    row = case['rows'][1]
    receipt_path = target_path(case['out'], row)
    result_path = receipt_path.parent / 'original/result.json'
    errors = ['RuntimeError: GeDi output contains non-finite values at scale 0.30',
              'ValueError: mask contains no valid depth pixels']
    report = {'dataset': 'lmo', 'scene_id': row['scene_id'], 'im_id': row['im_id'],
              'obj_id': row['obj_id'], 'instance_count': row['inst_count'],
              'candidate_count': 2, 'valid_candidate_count': 0, 'selected_count': 0, 'selected': [],
              'candidates': [{'candidate_index': i, 'status': 'invalid', 'error': error}
                             for i, error in enumerate(errors)]}
    dump(result_path, report)
    receipt = json.loads(receipt_path.read_text())
    receipt.update(original_result=str(result_path), outputs={str(result_path): sha(result_path)})
    dump(receipt_path, receipt)
    return row, result_path, errors


def test_invalid_candidate_causes_are_exact_verified_original_messages(case):
    row, path, errors = add_invalid_report(case)
    ds = snapshot(case)['datasets']['lmo']
    assert ds['invalid_candidate_causes'] == {error: 1 for error in errors}
    assert ds['candidate_totals']['reports_verified'] == 1
    assert ds['candidate_totals']['invalid'] == 2
    assert ds['candidate_totals']['valid'] == 0
    assert ds['zero_selected_breakdown']['all_candidates_invalid'] == 1
    assert ds['zero_selected_breakdown']['cause_unavailable'] == 0
    saved = ds['cases'][1]
    assert saved['classification'] == 'complete_zero_selected'
    assert saved['original_result']['sha256'] == sha(path)
    assert saved['original_result']['matches_receipt_hash'] is True
    assert [entry['error'] for entry in saved['invalid_candidates']] == errors
    assert saved['receipt']['sha256'] == sha(target_path(case['out'], row))


def test_changed_original_report_is_not_relabelled_as_a_candidate_failure(case):
    _, path, _ = add_invalid_report(case)
    changed = json.loads(path.read_text())
    changed['candidates'][0]['error'] = 'different untrusted error'
    dump(path, changed)
    ds = snapshot(case)['datasets']['lmo']
    assert ds['invalid_candidate_causes'] == {}
    assert ds['candidate_totals']['reports_verified'] == 0
    saved = ds['cases'][1]
    assert saved['original_result']['matches_receipt_hash'] is False
    assert saved['issues']


def test_recorded_CAD_metadata_and_timing_do_not_load_descriptor_or_model_arrays(case, monkeypatch):
    out = case['out']
    for stage, shape, seconds in [('geometry', None, 24.0), ('visual', [4730, 1536], 110.0),
                                  ('gedi', [4730, 64], 8.0), ('fuse', [4730, 128], 4.0)]:
        diagnostics = ({'raw_points': 5000, 'templates': 162, 'geometry_runtime_s': 13.0}
                       if stage == 'geometry' else {'retained_points': 4730, 'feature_shape': shape})
        dump(out / 'stages/lmo/obj_000001' / stage / 'receipt.json',
             {'status': 'complete', 'source_pins': case['pins'], 'diagnostics': diagnostics,
              'runtime_s': seconds, 'outputs': {str(out / 'features/dummy.npz'): 'recorded-hash-only'}})
    file_open = Path.open
    def metadata_only(path, *args, **kwargs):
        if path.suffix.lower() in ('.npz', '.pth', '.ply'):
            raise AssertionError('Supplemental snapshots must not read descriptor/model payloads')
        return file_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', metadata_only)
    cad = snapshot(case)['datasets']['lmo']['cad_onboarding'][0]
    assert cad['obj_id'] == 1
    assert cad['retained_points'] == 4730
    assert cad['descriptor_shapes'] == {'visual': [4730, 1536], 'gedi': [4730, 64], 'fuse': [4730, 128]}
    validation = cad['finite_alignment_validation']
    assert validation['freshly_checked'] is False
    assert validation['evidence'] == 'recorded_complete_sealed_stage_validation'
    assert validation['all_descriptor_stages_recorded_complete'] is True
    assert validation['retained_counts_agree'] is True
    assert cad['geometry_original_runtime_s'] == 13.0
    assert cad['geometry_guarded_or_adopted_runtime_s'] == 24.0



def test_generic_running_receipt_without_final_target_or_pose_fields_is_pending(case):
    p = target_path(case['out'], case['rows'][4])
    receipt = json.loads(p.read_text())
    for field in ('target', 'selected_count', 'poses'):
        receipt.pop(field)
    dump(p, receipt)
    ds = snapshot(case)['datasets']['lmo']
    assert ds['counts']['pending'] == 1
    assert ds['counts']['invalid_receipt'] == 0
    assert ds['cases'][4]['classification'] == 'pending'
    assert ds['cases'][4]['selected_instances_recorded'] == 0


def test_valid_original_candidate_is_not_invalidated_by_ICP_iteration_cap(case):
    row = case['rows'][0]
    receipt_path = target_path(case['out'], row)
    report_path = receipt_path.parent / 'original/result.json'
    selected = {'candidate_index': 0, 'final_score': 0.5, 'R': pose()['R'], 't_mm': pose()['t_mm']}
    dump(report_path, {'dataset': 'lmo', 'scene_id': row['scene_id'], 'im_id': row['im_id'],
         'obj_id': row['obj_id'], 'instance_count': row['inst_count'], 'candidate_count': 1,
         'valid_candidate_count': 1, 'selected_count': 1, 'selected': [selected],
         'candidates': [{'candidate_index': 0, 'status': 'valid', 'icp_converged': False}]})
    receipt = json.loads(receipt_path.read_text())
    receipt.update(original_result=str(report_path), outputs={str(report_path): sha(report_path)})
    dump(receipt_path, receipt)
    ds = snapshot(case)['datasets']['lmo']
    assert ds['candidate_totals']['valid'] == 1
    assert ds['candidate_totals']['invalid'] == 0
    assert ds['invalid_candidate_causes'] == {}
    assert ds['cases'][0]['classification'] == 'complete_with_saved_poses'



def test_changed_official_target_bytes_are_reported_against_the_pinned_asset_hash(case):
    # Same target rows, different serialization: this tests byte identity alone.
    case['targets'].write_text(json.dumps(case['rows']), encoding='utf-8')
    result = snapshot(case)
    ds = result['datasets']['lmo']
    assert result['partial'] is True
    assert result['coverage_complete'] is False
    assert ds['official_targets']['matches_pinned_SHA'] is False
    assert ds['official_targets']['expected_sha256'] == json.loads(case['assets_manifest'].read_text())['datasets']['lmo']['target_sha256']


def test_unexpected_target_directory_cannot_be_hidden_by_official_row_iteration(case):
    extra = {'scene_id': 999999, 'im_id': 999999, 'obj_id': 999999, 'inst_count': 1}
    extra_path = target_path(case['out'], extra)
    dump(extra_path, {'status': 'blocked', 'source_pins': case['pins'], 'target': extra,
                     'selected_count': 0, 'poses': [], 'reason': 'unexpected receipt'})
    result = snapshot(case)
    ds = result['datasets']['lmo']
    assert result['partial'] is True
    assert result['coverage_complete'] is False
    assert extra_path.parent.name in ds['unexpected_target_directories']
    assert ds['expected_targets'] == 6
    assert len(ds['cases']) == 6
