"""Controller rules using explicit fixture ledgers/receipts, never model inference.

Mock metric numbers below test aggregation and denominator bookkeeping only;
they are not measured BOP scores and are never written into real run folders.
"""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pytest

from freezev2_repro import case_inference, suite
from freezev2_repro.data import sha256
from freezev2_repro.durable import atomic_json, read_json
from freezev2_repro.evaluation import PosePrediction, write_bop_csv
from freezev2_repro.mask_index import MaskIndex
from freezev2_repro.masks import LocalizationTarget, encode_rle


def _controller_fixture(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    originals = []
    assets = {'datasets': {}}
    for name in suite.DATASETS:
        camera = data / name / 'test' / '000001' / 'scene_camera.json'
        camera.parent.mkdir(parents=True)
        camera.write_text(json.dumps({str(image): {'cam_K': [100, 0, 2, 0, 100, 1, 0, 0, 1], 'depth_scale': 1}
                                      for image in (1, 2)}))
        for image in (1, 2):
            originals.append({'dataset': name, 'scene_id': 1, 'im_id': image,
                              'camera_path': camera.relative_to(data).as_posix(),
                              'sha256': 'fixture-RGB', 'depth_sha256': 'fixture-depth'})
        target = tmp_path / f'{name}_targets.json'
        target.write_text(json.dumps([{'scene_id': 1, 'im_id': 1, 'obj_id': 1, 'inst_count': 1},
                                     {'scene_id': 1, 'im_id': 2, 'obj_id': 2, 'inst_count': 1}]))
        assets['datasets'][name] = {'official_target_images': 2, 'input_images': 2, 'target_records': 2,
            'requested_instances': 2, 'cad_objects': 2, 'object_ids': [1, 2],
            'masks': {'source_kind': 'predicted', 'segmentor': 'test_only_public_fixture', 'sha256': 'mask-fixture-' + name},
            'targets': {'path': str(target), 'sha256': sha256(target)}}
    (data / 'full_manifest.json').write_text(json.dumps({'images': originals}))
    assets_path = tmp_path / 'assets.json'
    assets_path.write_text(json.dumps(assets))
    protocol = {'method': 'FreeZeV2-base-paper-v1-independent',
        'mask_configuration': 'CNOS-FastSAM-public-BOP23', 'datasets': list(suite.DATASETS),
        'execution_order': list(suite.DATASETS),
        'base': {'seed': 0, 'registration': {'k': 10, 'iterations': 10000},
                 'candidate_ranking': {'translation_nms_diameter_fraction': 0.1}},
        'paths': {'suite_output_root': str(tmp_path / 'run'), 'assets_manifest': str(assets_path),
                  'dataset_root': str(data), 'inference_python': sys.executable},
        'resources': {'minimum_free_disk_gb': 0, 'cpu_workers': 1, 'export_interval_images': 50},
        'validation': {'tless_fixed_cohort': [{'scene_id': 1, 'im_id': 1}], 'other_dataset_target_rows': 1}}
    return protocol, assets


def _pose(image, object_id):
    return {'scene_id': 1, 'image_id': image, 'object_id': object_id, 'score': 0.5,
            'R_m2c': np.eye(3).tolist(), 't_m': [0, 0, 0.7], 'image_time_s': 0.1,
            'candidate_id': 'synthetic-unit-test-only'}


def _ledger(root, name, image, object_id, status, *, predictions=None, partial=False):
    path = root / 'datasets' / name / 'images' / f'000001_{image:06d}'
    value = {'identity': {'run': {'test_only': True}, 'scene_id': 1, 'image_id': image, 'target_rows': [[object_id, 1]]},
             'status': 'blocked' if partial else 'complete_with_failures', 'runtime_s': 0.1,
             'targets': [{'object_id': object_id, 'requested_instances': 1, 'status': status,
                         'valid_predictions': len(predictions or []), 'candidates': [],
                         'reason': 'no_predicted_masks' if status == 'failed' else None}],
             'predictions': predictions or [], 'GT_pose_access': False}
    atomic_json(path / ('partial.json' if partial else 'complete.json'), value)
    return value


def _mock_evaluated_dataset(root, name, assets, *, metric=0.5):
    _ledger(root, name, 1, 1, 'completed', predictions=[_pose(1, 1)])
    _ledger(root, name, 2, 2, 'failed')  # A missing-mask failure stays in the denominator.
    folder = root / 'datasets' / name
    csv = folder / 'predictions' / f'freezev2base_{name}-test.csv'
    write_bop_csv([PosePrediction(1, 1, 1, 0.5, np.eye(3), np.array([0, 0, 0.7]), 0.1)], csv)
    score_file = folder / 'official_evaluation' / csv.stem / 'scores_bop19.json'
    metrics = {key: metric for key in ('bop19_average_recall', 'bop19_average_recall_vsd',
                                     'bop19_average_recall_mssd', 'bop19_average_recall_mspd')}
    atomic_json(score_file, metrics)
    atomic_json(folder / 'evaluation_receipt.json', {'test_only': True, 'status': 'complete', 'official_pose_metrics': metrics,
        'csv_sha256': sha256(csv), 'target_sha256': assets['datasets'][name]['targets']['sha256'],
        'score_file': str(score_file), 'score_sha256': sha256(score_file)})
    for object_id in (1, 2):
        for stage in ('geometry', 'features'):
            atomic_json(folder / 'onboarding' / f'obj_{object_id:06d}' / stage / 'complete.json',
                        {'test_only': True, 'runtime_s': 0.1})
    return csv


def test_summary_keeps_full_official_denominator_including_missing_masks_and_pending(tmp_path):
    protocol, assets = _controller_fixture(tmp_path)
    root = Path(protocol['paths']['suite_output_root'])
    _ledger(root, 'lmo', 1, 1, 'failed')
    # Only one of two official target rows has been attempted.
    result = suite.summarize(root, assets, {}, protocol)
    row = result['datasets']['lmo']
    assert row['expected_targets'] == row['expected_instances'] == 2
    assert row['completed_targets'] == 0 and row['failed_targets'] == 1
    assert row['pending_targets'] == 1 and row['valid_pose_predictions'] == 0
    assert result['expected_targets'] == 14
    assert not result['suite_complete_and_evaluated'] and result['all_seven_aggregate'] is None


def test_all_seven_aggregate_requires_seven_current_metric_receipts_not_success_only(tmp_path):
    protocol, assets = _controller_fixture(tmp_path)
    root = Path(protocol['paths']['suite_output_root'])
    states = {name: {'status': 'complete'} for name in suite.DATASETS}
    for index, name in enumerate(suite.DATASETS[:-1]):
        _mock_evaluated_dataset(root, name, assets, metric=0.1 * (index + 1))
    result = suite.summarize(root, assets, states, protocol)
    assert result['all_seven_aggregate'] is None
    _mock_evaluated_dataset(root, suite.DATASETS[-1], assets, metric=0.7)
    result = suite.summarize(root, assets, states, protocol)
    assert result['suite_complete_and_evaluated']
    assert result['all_seven_aggregate']['bop19_average_recall'] == pytest.approx(0.4)
    assert all(row['failed_targets'] == 1 and row['expected_targets'] == 2 for row in result['datasets'].values())


def test_stale_csv_or_metric_bytes_cannot_enable_an_all_seven_aggregate(tmp_path):
    protocol, assets = _controller_fixture(tmp_path)
    root = Path(protocol['paths']['suite_output_root'])
    for name in suite.DATASETS:
        _mock_evaluated_dataset(root, name, assets)
    states = {name: {'status': 'complete'} for name in suite.DATASETS}
    assert suite.summarize(root, assets, states, protocol)['suite_complete_and_evaluated']
    csv = root / 'datasets/lmo/predictions/freezev2base_lmo-test.csv'
    csv.write_text(csv.read_text() + '\n')
    result = suite.summarize(root, assets, states, protocol)
    assert not result['suite_complete_and_evaluated'] and result['all_seven_aggregate'] is None
    _mock_receipt = read_json(root / 'datasets/lmo/evaluation_receipt.json')
    _mock_receipt['csv_sha256'] = sha256(csv)
    atomic_json(root / 'datasets/lmo/evaluation_receipt.json', _mock_receipt)
    score = Path(_mock_receipt['score_file'])
    score.write_text(score.read_text() + '\n')
    result = suite.summarize(root, assets, states, protocol)
    assert not result['suite_complete_and_evaluated'] and result['all_seven_aggregate'] is None


def test_partial_dataset_list_never_claims_all_seven_macro(tmp_path):
    protocol, assets = _controller_fixture(tmp_path)
    root = Path(protocol['paths']['suite_output_root'])
    for name in suite.DATASETS:
        _mock_evaluated_dataset(root, name, assets)
    protocol['datasets'] = list(suite.DATASETS[:6])
    try:
        result = suite.summarize(root, assets, {}, protocol)
    except ValueError:
        return  # Explicitly rejecting an incomplete all-seven protocol is valid.
    assert not result['suite_complete_and_evaluated'] and result['all_seven_aggregate'] is None


def test_controller_identity_is_immutable_and_official_targets_are_not_filtered(tmp_path, monkeypatch):
    protocol, assets = _controller_fixture(tmp_path)
    monkeypatch.setattr(suite, 'implementation_identity', lambda: {'test_only_code': 'immutable-A'})
    runner = suite.SuiteRunner(protocol)
    targets = runner.targets('lmo')
    assert [asdict(target) for target in targets] == [asdict(LocalizationTarget(1, 1, 1, 1)), asdict(LocalizationTarget(1, 2, 2, 1))]
    resumed = suite.SuiteRunner(protocol)
    assert resumed.method_digest == runner.method_digest
    changed = deepcopy(protocol)
    changed['base']['seed'] = 1
    with pytest.raises(ValueError, match='identity'):
        suite.SuiteRunner(changed)


def test_partial_export_keeps_denominator_but_cannot_finish_dataset_inference(tmp_path, monkeypatch):
    """Simulate one lost image receipt, without running any feature backend."""
    protocol, assets = _controller_fixture(tmp_path)
    monkeypatch.setattr(suite, 'implementation_identity', lambda: {'test_only_code': 'immutable-A'})
    runner = suite.SuiteRunner(protocol)
    runner.onboard = lambda *args: {'test_only': 'mock_CAD_bundle'}
    runner.get_backends = lambda: ('explicit_unit_mock_DINO', 'explicit_unit_mock_GeDi')
    runner.dataset = lambda *args: None
    runner.heartbeat = lambda **kwargs: None
    runner.check_csv = lambda path: {'valid': True, 'test_only': True}

    class IndexFixture:
        def close(self):
            pass

    runner.index = lambda *args: IndexFixture()

    def one_missing_ledger(dataset, scene, image, targets, bundles, index, recipe,
                           backends, folder, identity, resources, heartbeat):
        if image == 1:
            return _ledger(runner.root, 'lmo', 1, 1, 'completed', predictions=[_pose(1, 1)])
        # An interrupted/deleted second receipt must not be inferred complete
        # from the absence of any recorded blocked target.
        return {'test_only': True, 'targets': [], 'predictions': []}

    monkeypatch.setattr(suite, 'infer_image', one_missing_ledger)
    csv, complete = runner.infer_dataset('lmo')
    receipt = read_json(csv.parent / 'export_receipt.json')
    assert receipt['expected_targets'] == 2 and receipt['ledger_targets'] == 1
    assert receipt['missing_targets_remain_in_official_denominator']
    assert not complete


def test_partial_blocked_image_can_resume_into_an_explicit_missing_mask_failure(tmp_path):
    class SensorFixture:
        def frame(self, scene, image):
            return {'rgb': np.zeros((3, 4, 3), np.uint8), 'depth_m': np.ones((3, 4)),
                    'K': np.eye(3), 'source': {'test_only': 'synthetic_sensor_without_GT'}}

    class EmptyPublicMaskIndex:
        def candidates(self, *args):
            return []

    target = LocalizationTarget(1, 2, 3, 1)
    folder = tmp_path / 'image'
    resources = {'minimum_free_disk_gb': 0}
    recipe = {'candidate_ranking': {'translation_nms_diameter_fraction': 0.1}}
    initial = case_inference.infer_image(SensorFixture(), 1, 2, [target], {}, EmptyPublicMaskIndex(),
        recipe, None, folder, {'test_only': 'fixed_recipe'}, resources)
    assert initial['targets'][0]['status'] == 'blocked'
    assert (folder / 'partial.json').is_file() and not (folder / 'complete.json').exists()
    # The CAD is now available. There are truly no external masks for this
    # official target, so no model/pose fallback may manufacture a prediction.
    resumed = case_inference.infer_image(SensorFixture(), 1, 2, [target], {3: {'model': {'diameter_m': 0.1}}},
        EmptyPublicMaskIndex(), recipe, None, folder, {'test_only': 'fixed_recipe'}, resources)
    assert resumed['targets'][0]['status'] == 'failed'
    assert resumed['targets'][0]['reason'] == 'no_predicted_masks'
    assert resumed['predictions'] == [] and (folder / 'complete.json').is_file()


def _index_sources(tmp_path):
    path = tmp_path / 'public.json'
    path.write_text(json.dumps([{'scene_id': 1, 'image_id': 2, 'category_id': 3, 'score': 0.5,
                                 'segmentation': encode_rle(np.ones((2, 3), np.uint8))}]))
    return [{'path': str(path), 'sha256': sha256(path), 'segmentor': 'mock_public_format',
             'source_id': 'integrity-test-only', 'source_kind': 'predicted'}]


def test_sqlite_seal_is_stable_read_only_and_detects_external_changes_before_query(tmp_path):
    sources, path = _index_sources(tmp_path), tmp_path / 'index.sqlite'
    index = MaskIndex(path, sources)
    seal = read_json(path.with_name(path.name + '.integrity.json'))
    assert seal['database_sha256'] == sha256(path) and seal['rows'] == 1
    before = sha256(path)
    assert len(index.candidates(1, 2, 3, 1)) == 1
    with pytest.raises(sqlite3.OperationalError, match='readonly'):
        index.db.execute('UPDATE masks SET score=1')
    assert sha256(path) == before
    connection = sqlite3.connect(path)
    connection.execute('UPDATE masks SET score=0.9')
    connection.commit()
    connection.close()
    with pytest.raises(ValueError, match='checksum changed'):
        index.candidates(1, 2, 3, 1)
    index.close()
    with pytest.raises(ValueError, match='checksum changed'):
        MaskIndex(path, sources)


def test_commit_before_sidecar_interruption_recovers_only_after_row_verification(tmp_path, monkeypatch):
    sources, path = _index_sources(tmp_path), tmp_path / 'index.sqlite'
    original = MaskIndex._write_seal

    def crash_before_sidecar(self, count, *, recovered):
        raise RuntimeError('simulated interruption after committed SQLite rows')

    monkeypatch.setattr(MaskIndex, '_write_seal', crash_before_sidecar)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        MaskIndex(path, sources)
    assert path.is_file() and not path.with_name(path.name + '.integrity.json').exists()
    monkeypatch.setattr(MaskIndex, '_write_seal', original)
    recovered = MaskIndex(path, sources)
    assert recovered.seal['recovered_after_row_verification']
    recovered.close()
    path.with_name(path.name + '.integrity.json').unlink()
    connection = sqlite3.connect(path)
    connection.execute('UPDATE masks SET score=0.9')
    connection.commit()
    connection.close()
    with pytest.raises(ValueError, match='rows differ'):
        MaskIndex(path, sources)
