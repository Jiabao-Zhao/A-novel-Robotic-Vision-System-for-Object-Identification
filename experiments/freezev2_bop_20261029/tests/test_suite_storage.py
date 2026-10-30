"""Durability and ranking acceptance tests; mock maps are test-only, never models."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from freezev2_repro import case_inference, onboarding
from freezev2_repro.data import BOPDataset, sha256
from freezev2_repro.durable import atomic_json, atomic_npz, ensure_identity, read_json, worker_lock
from freezev2_repro.mask_index import MaskIndex
from freezev2_repro.masks import LocalizationTarget, encode_rle, load_mask_candidates, select_localization_candidates
from freezev2_repro.rendering import RaycastRenderer, camera_poses, sample_texture


def test_atomic_checkpoints_keep_prior_value_and_reject_cache_identity_changes(tmp_path):
    path = tmp_path / 'checkpoint.json'
    atomic_json(path, {'phase': 1, 'array': np.array([1, 2])})
    assert read_json(path) == {'phase': 1, 'array': [1, 2]}
    before = sha256(path)
    with pytest.raises(ValueError):
        atomic_json(path, {'invalid': float('nan')})
    assert sha256(path) == before
    archive = tmp_path / 'array.npz'
    atomic_npz(archive, sample=np.arange(12).reshape(4, 3))
    with np.load(archive, allow_pickle=False) as data:
        np.testing.assert_array_equal(data['sample'], np.arange(12).reshape(4, 3))
    folder = tmp_path / 'cache'
    ensure_identity(folder, {'recipe': 'frozen-A'})
    ensure_identity(folder, {'recipe': 'frozen-A'})
    with pytest.raises(ValueError, match='Incompatible cache identity'):
        ensure_identity(folder, {'recipe': 'different-B'})
    partial = tmp_path / 'unidentified'
    partial.mkdir()
    (partial / 'orphan').write_text('partial')
    with pytest.raises(ValueError, match='Unidentified partial'):
        ensure_identity(partial, {'recipe': 'A'})


def test_actual_os_lock_blocks_another_process_and_stale_pid_text_is_not_liveness(tmp_path):
    path = tmp_path / 'worker.lock'
    # The current PID really is live, but no handle owns this file yet.
    path.write_text(json.dumps({'pid': os.getpid(), 'status': 'running'}))
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(Path(__file__).parents[1]) + os.pathsep + environment.get('PYTHONPATH', '')
    code = '''import sys
from freezev2_repro.durable import worker_lock
try:
    with worker_lock(sys.argv[1]):
        print('acquired')
except RuntimeError:
    print('blocked_by_handle')
    sys.exit(23)
'''
    with worker_lock(path):
        blocked = subprocess.run([sys.executable, '-c', code, str(path)], env=environment,
                                 capture_output=True, text=True, timeout=15)
        assert blocked.returncode == 23 and 'blocked_by_handle' in blocked.stdout
    released = subprocess.run([sys.executable, '-c', code, str(path)], env=environment,
                              capture_output=True, text=True, timeout=15)
    assert released.returncode == 0 and 'acquired' in released.stdout


def _sources(tmp_path):
    masks = [np.zeros((3, 4), np.uint8)]
    for column in range(4):
        mask = np.zeros((3, 4), np.uint8)
        mask[:, column] = 1
        masks.append(mask)
    sources = []
    for number, segmentor in enumerate(('cnos-sam', 'cnos-fastsam', 'cnos-sam')):
        rows = [{'scene_id': 1, 'image_id': 2, 'category_id': 3, 'score': score,
                 'segmentation': encode_rle(mask)} for score, mask in zip((0.99, 0.9, 0.9, 0.4, 0.1), masks)]
        # A different image/category cannot affect the localization shortlist.
        rows += [{'scene_id': 9, 'image_id': 2, 'category_id': 3, 'score': 1,
                  'segmentation': encode_rle(masks[1])}]
        path = tmp_path / f'source{number}.json'
        path.write_text(json.dumps(rows))
        sources.append({'path': str(path), 'sha256': sha256(path), 'segmentor': segmentor,
                        'source_id': f'public-fixture-{number}', 'source_kind': 'predicted'})
    return sources


def test_sql_mask_shortlist_matches_existing_base_n_plus_one_and_resumes(tmp_path):
    sources = _sources(tmp_path)
    combined = []
    for source in sources:
        combined += load_mask_candidates(source['path'], segmentor=source['segmentor'], source_id=source['source_id'])
    expected = select_localization_candidates(combined, scene_id=1, image_id=2, object_id=3, instance_count=1)
    path = tmp_path / 'index.sqlite'
    index = MaskIndex(path, sources)
    actual = index.candidates(1, 2, 3, 1)
    assert [(value.segmentor, value.source_id, value.record_index, value.score) for value in actual] == [(value.segmentor, value.source_id, value.record_index, value.score) for value in expected]
    for left, right in zip(actual, expected):
        np.testing.assert_array_equal(left.mask, right.mask)
        assert left.mask.any()  # Highest-confidence empty mask did not consume a slot.
    assert len(actual) == 4  # N+1 for each of two segmentors, not each source file.
    index.close()
    resumed = MaskIndex(path, sources)
    assert [value.candidate_id for value in resumed.candidates(1, 2, 3, 1)] == [value.candidate_id for value in actual]
    resumed.close()
    changed = deepcopy(sources)
    changed[0]['source_id'] = 'different-provenance'
    with pytest.raises(ValueError, match='identity differs'):
        MaskIndex(path, changed)
    Path(sources[0]['path']).write_text('[]')
    with pytest.raises(ValueError, match='checksum changed'):
        MaskIndex(path, sources)


def _onboarding_fixture(tmp_path, monkeypatch):
    points = np.array([[-0.02, -0.02, 0], [0.02, -0.02, 0], [0.02, 0.02, 0], [-0.02, 0.02, 0]])
    model = {'vertices': points, 'triangles': np.array([[0, 1, 2], [0, 2, 3]]), 'colors': np.ones((4, 3)),
             'diameter_m': 0.2, 'sha256': 'synthetic-storage-fixture', 'appearance': 'test_only', 'texture': None}
    # Only the sampling dependency is mocked. All durable writes, hash/identity
    # checks, view loops, visibility and source-to-camera projection are real.
    fake_o3d = SimpleNamespace(
        utility=SimpleNamespace(random=SimpleNamespace(seed=lambda seed: None), Vector3dVector=np.asarray, Vector3iVector=np.asarray),
        geometry=SimpleNamespace(TriangleMesh=lambda *args: SimpleNamespace(sample_points_poisson_disk=lambda **kwargs: SimpleNamespace(points=points))))
    monkeypatch.setitem(sys.modules, 'open3d', fake_o3d)
    K = np.array([[20., 0, 3.5], [0, 20., 3.5], [0, 0, 1.]])
    monkeypatch.setattr(onboarding, 'camera_poses', lambda *args, **kwargs: (K, [(np.eye(3), np.array([0, 0, 1.]))] * 162))
    calls = []

    class ToyRenderer:
        def __init__(self, *args, **kwargs):
            pass

        def render(self, *args, **kwargs):
            calls.append(len(calls))
            return {'rgb': np.full((8, 8, 3), 100, np.uint8), 'depth_m': np.ones((8, 8))}

    monkeypatch.setattr(onboarding, 'RaycastRenderer', ToyRenderer)
    cnos = tmp_path / 'canonical.npy'
    np.save(cnos, np.tile(np.eye(4), (162, 1, 1)))
    recipe = {'seed': 0, 'render': {'resolution': 8, 'occupancy': 0.5, 'visibility_tolerance_fraction': 0.005},
              'cad': {'initial_poisson_points': 4, 'points': 3, 'min_visible_views': 18},
              'features': {'pca_components': 64}}
    return model, recipe, cnos, calls


def test_interrupted_geometry_resumes_completed_views_and_rejects_mismatches(tmp_path, monkeypatch):
    model, recipe, cnos, calls = _onboarding_fixture(tmp_path, monkeypatch)
    folder = tmp_path / 'geometry'

    def interrupt(**status):
        if status['view'] == 3:
            raise RuntimeError('test interruption before view3')

    with pytest.raises(RuntimeError, match='test interruption'):
        onboarding.onboard_geometry(model, folder, recipe, cnos, 'test-implementation', minimum_free_gb=0, heartbeat=interrupt)
    assert len(calls) == 3
    assert not (folder / 'complete.json').exists()
    previous = {path.name: sha256(path) for path in (folder / 'views').glob('*.npz')}
    completed = onboarding.onboard_geometry(model, folder, recipe, cnos, 'test-implementation', minimum_free_gb=0)
    assert len(calls) == 162  # Reused3 + rendered159, rather than rerendering162.
    assert completed['views'] == 162 and completed['retained_points'] == 3
    assert all(sha256(folder / 'views' / name) == checksum for name, checksum in previous.items())
    cached = onboarding.onboard_geometry(model, folder, recipe, cnos, 'test-implementation', minimum_free_gb=0)
    assert cached == completed and len(calls) == 162
    changed = deepcopy(recipe)
    changed['seed'] = 1
    with pytest.raises(ValueError, match='Incompatible cache identity'):
        onboarding.onboard_geometry(model, folder, changed, cnos, 'test-implementation', minimum_free_gb=0)
    with pytest.raises(ValueError, match='Incompatible cache identity'):
        onboarding.onboard_geometry(model, folder, recipe, cnos, 'different-implementation', minimum_free_gb=0)
    # A valid but modified view NPZ is still corruption of the completed cache.
    file = folder / 'views' / '000.npz'
    with np.load(file, allow_pickle=False) as stored:
        arrays = {name: np.array(stored[name]) for name in stored.files}
    arrays['rgb'][0, 0, 0] = 17
    atomic_npz(file, **arrays)
    with pytest.raises(ValueError, match='corrupt|Corrupt|hash|checksum'):
        onboarding.onboard_geometry(model, folder, recipe, cnos, 'test-implementation', minimum_free_gb=0)


def test_interrupted_feature_onboarding_reuses_saved_real_map_slots(tmp_path, monkeypatch):
    model, recipe, cnos, _ = _onboarding_fixture(tmp_path, monkeypatch)
    geometry = tmp_path / 'geometry'
    onboarding.onboard_geometry(model, geometry, recipe, cnos, 'test-implementation', minimum_free_gb=0)
    extraction_calls = []

    class MockDINO:
        provenance = {'test_only': True, 'backend': 'mock-map-storage-test'}

        def extract(self, image):
            extraction_calls.append(1)
            return np.ones((16, 16, 1536), np.float32)

    class NeverGeDi:
        provenance = {'test_only': True}

        def compute(self, *args):
            raise AssertionError('This storage test stops before geometric extraction')

    def first_interrupt(**status):
        if status.get('view') == 3:
            raise RuntimeError('first controlled interruption')

    features = tmp_path / 'features'
    with pytest.raises(RuntimeError, match='controlled interruption'):
        onboarding.onboard_features(model, geometry, features, recipe, (MockDINO(), NeverGeDi()), 'test-implementation', minimum_free_gb=0, heartbeat=first_interrupt)
    assert len(extraction_calls) == 3
    checksums = {path.name: sha256(path) for path in (features / 'visual_maps').glob('*.npz')}

    def second_interrupt(**status):
        if status.get('view') == 4:
            raise RuntimeError('second controlled interruption')

    with pytest.raises(RuntimeError, match='controlled interruption'):
        onboarding.onboard_features(model, geometry, features, recipe, (MockDINO(), NeverGeDi()), 'test-implementation', minimum_free_gb=0, heartbeat=second_interrupt)
    assert len(extraction_calls) == 4  # Only view3 was newly extracted.
    assert all(sha256(features / 'visual_maps' / name) == checksum for name, checksum in checksums.items())


def test_image_resume_round_trips_json_identity_and_partial_mismatch_is_rejected(tmp_path):
    class FailedInput:
        calls = 0

        def frame(self, scene_id, image_id):
            self.calls += 1
            raise ValueError('explicit synthetic sensor-input failure')

    dataset = FailedInput()
    targets = [LocalizationTarget(1, 2, 3, 1)]
    identity = {'run_version': 'fixed-test-only'}
    resources = {'minimum_free_disk_gb': 0}
    folder = tmp_path / 'image'
    value = case_inference.infer_image(dataset, 1, 2, targets, {}, None, {}, None, folder, identity, resources)
    assert dataset.calls == 1 and value['status'] == 'complete_with_failures'
    resumed = case_inference.infer_image(dataset, 1, 2, targets, {}, None, {}, None, folder, identity, resources)
    assert dataset.calls == 1 and resumed['status'] == value['status']
    partial = tmp_path / 'other-image'
    atomic_json(partial / 'identity.json', {'run': {'different': True}, 'scene_id': 1, 'image_id': 2, 'target_rows': [[3, 1]]})
    with pytest.raises(ValueError, match='identity|Identity'):
        case_inference.infer_image(dataset, 1, 2, targets, {}, None, {}, None, partial, identity, resources)


def test_texture_bottom_left_bilinear_sampling():
    texture = {'image_rgb': np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 0]]], np.uint8),
               'uv_origin': 'bottom_left'}
    color = sample_texture(texture, np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]]))
    np.testing.assert_allclose(color[:4], [[0, 0, 1], [1, 1, 0], [1, 0, 0], [0, 1, 0]])
    np.testing.assert_allclose(color[4], [0.5, 0.5, 0.25])


def test_real_ycbv_texture_survives_native_cad_render_when_assets_configured():
    root = os.environ.get('FREEZEV2_BOP_ROOT')
    cnos = os.environ.get('FREEZEV2_CNOS_POSES')
    if not root or not cnos:
        pytest.skip('Configure existing BOP root/CNOS array for the optional actual-asset CPU render')
    pytest.importorskip('open3d')
    model = BOPDataset(root, 'ycbv').model(1)
    assert model['appearance'] == 'texture' and model['texture']['vertex_order_verified']
    renderer = RaycastRenderer(model['vertices'], model['triangles'], model['colors'], texture=model['texture'])
    K, poses = camera_poses(model['vertices'], resolution=96, camera_to_model=np.load(cnos, allow_pickle=False))
    rendered = renderer.render(*poses[80], K, height=96, width=96)
    foreground = rendered['depth_m'] > 0
    assert foreground.any()
    spread = np.ptp(rendered['rgb'][foreground].astype(int), axis=1)
    assert np.count_nonzero(spread > 10) > 10  # Texture is not a gray replacement.
