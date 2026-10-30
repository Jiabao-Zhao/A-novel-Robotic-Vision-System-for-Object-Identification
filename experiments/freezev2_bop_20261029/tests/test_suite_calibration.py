"""Calibration sealing tests use tiny fixture files, never trained inference."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from freezev2_repro import suite
from freezev2_repro.data import sha256
from freezev2_repro.durable import atomic_json, digest, read_json
from test_suite_controller import _controller_fixture


def test_all_scene_camera_files_are_sealed_once_and_bound_to_cache_identity(tmp_path, monkeypatch):
    protocol, _ = _controller_fixture(tmp_path)
    monkeypatch.setattr(suite, 'implementation_identity', lambda: {'test_only': 'calibration-A'})
    runner = suite.SuiteRunner(protocol)
    manifest_path = runner.root / 'run_manifest.json'
    original_receipt = manifest_path.read_bytes()
    receipt = read_json(manifest_path)
    cameras = receipt['camera_sha256_by_relative_path']
    # The fixture has two frames sharing each scene, including datasets not
    # reached by a small first cohort. All seven camera files are sealed.
    assert len(cameras) == 7 and len(runner.originals) == 14
    dataset_root = Path(protocol['paths']['dataset_root'])
    assert cameras == {path: sha256(dataset_root / path) for path in cameras}
    assert receipt['camera_snapshot_digest'] == digest(cameras)
    assert all(row['camera_sha256'] == cameras[row['camera_path']] for row in runner.originals.values())
    assert runner.identity('lmo')['camera_snapshot_digest'] == digest(cameras)
    resumed = suite.SuiteRunner(protocol)
    assert resumed.camera_snapshot_digest == runner.camera_snapshot_digest
    assert manifest_path.read_bytes() == original_receipt


def test_resume_rejects_camera_edits_even_when_retained_manifest_is_unchanged(tmp_path, monkeypatch):
    protocol, _ = _controller_fixture(tmp_path)
    monkeypatch.setattr(suite, 'implementation_identity', lambda: {'test_only': 'calibration-A'})
    runner = suite.SuiteRunner(protocol)
    root = Path(protocol['paths']['dataset_root'])
    manifest_before = (root / 'full_manifest.json').read_bytes()
    camera = root / next(path for path in runner.camera_sha256_by_relative_path if path.startswith('ycbv/'))
    contents = read_json(camera)
    contents['1']['cam_K'][0] = 123
    camera.write_text(json.dumps(contents))
    assert (root / 'full_manifest.json').read_bytes() == manifest_before
    with pytest.raises(ValueError, match='Camera calibration changed'):
        suite.SuiteRunner(protocol)


def test_resume_does_not_silently_seal_an_old_unpinned_run(tmp_path, monkeypatch):
    protocol, _ = _controller_fixture(tmp_path)
    monkeypatch.setattr(suite, 'implementation_identity', lambda: {'test_only': 'calibration-A'})
    runner = suite.SuiteRunner(protocol)
    path = runner.root / 'run_manifest.json'
    receipt = read_json(path)
    receipt.pop('camera_sha256_by_relative_path')
    receipt.pop('camera_snapshot_digest')
    atomic_json(path, receipt)
    with pytest.raises(ValueError, match='snapshot is absent'):
        suite.SuiteRunner(protocol)


def test_calibration_path_escape_is_rejected_before_inference(tmp_path, monkeypatch):
    protocol, _ = _controller_fixture(tmp_path)
    monkeypatch.setattr(suite, 'implementation_identity', lambda: {'test_only': 'calibration-A'})
    path = Path(protocol['paths']['dataset_root']) / 'full_manifest.json'
    manifest = read_json(path)
    manifest['images'][0]['camera_path'] = '../outside/scene_camera.json'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='inside the dataset root'):
        suite.SuiteRunner(protocol)


def test_frame_read_checks_calibration_snapshot_and_keeps_extrinsics_private(tmp_path):
    model = tmp_path / 'lmo' / 'models'
    model.mkdir(parents=True)
    (model / 'models_info.json').write_text('{"1":{"diameter":100}}')
    scene = tmp_path / 'lmo' / 'test' / '000001'
    (scene / 'rgb').mkdir(parents=True)
    (scene / 'depth').mkdir()
    rgb, depth = scene / 'rgb' / '000001.png', scene / 'depth' / '000001.png'
    Image.fromarray(np.full((3, 4, 3), 100, np.uint8)).save(rgb)
    Image.fromarray(np.full((3, 4), 1000, np.uint16)).save(depth)
    camera = scene / 'scene_camera.json'
    metadata = {'1': {'cam_K': [100, 0, 2, 0, 100, 1, 0, 0, 1], 'depth_scale': 1,
                      'cam_R_w2c': 'forbidden', 'cam_t_w2c': 'forbidden'}}
    camera.write_text(json.dumps(metadata))
    # A malformed pose-GT file verifies that calibration checks never access it.
    (scene / 'scene_gt.json').write_text('DO NOT READ POSE GT')
    originals = {('lmo', 1, 1): {'camera_path': camera.relative_to(tmp_path).as_posix(),
        'camera_sha256': sha256(camera), 'sha256': sha256(rgb), 'depth_sha256': sha256(depth)}}
    reader = suite.CheckedDataset(tmp_path, 'lmo', originals)
    frame = reader.frame(1, 1)
    assert frame['depth_m'][0, 0] == pytest.approx(1)
    assert not {'cam_R_w2c', 'cam_t_w2c'} & (frame.keys() | frame['source'].keys())
    metadata['1']['depth_scale'] = .1
    camera.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='Camera calibration changed'):
        reader.frame(1, 1)
