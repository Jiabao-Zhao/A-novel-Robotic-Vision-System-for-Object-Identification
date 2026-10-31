from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import onboard_original as m


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class Durable:
    atomic_json = staticmethod(dump)
    def __init__(self): self.lock_active = False; self.lock_path = None
    def ensure_identity(self, folder, identity):
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / 'identity.json'
        if target.exists():
            if json.loads(target.read_text()) != identity: raise ValueError('Incompatible cache identity')
        else:
            if any(folder.iterdir()): raise ValueError('Unidentified partial cache')
            dump(target, identity)
    def require_disk(self, path, minimum): assert self.lock_active and minimum == 15
    @contextmanager
    def worker_lock(self, path):
        self.lock_path = path; self.lock_active = True
        try: yield object()
        finally: self.lock_active = False


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'VIEWS', 3); monkeypatch.setattr(m, 'SIZE', 4)
    monkeypatch.setattr(m, 'verify_inputs', lambda ctx: None)
    def camera(z):
        return SimpleNamespace(R=np.eye(3), t=np.array([0., 0., z]), K=np.diag([4., 4., 1.]),
                               direction=np.array([0., 0., 1.]), size=4)
    class API:
        def __init__(self): self.calls = []; self.saved = None; self.templates = []
        def load_mesh(self, path): self.calls.append(('load', path)); return object()
        def sample_query_points(self, mesh, n, seed):
            self.calls.append(('sample', n, seed)); return np.arange(n * 3).reshape(n, 3).astype(float)
        def make_template_cameras(self, n, size):
            self.calls.append(('cameras', n, size)); return [camera(1) for _ in range(n)]
        def render_templates(self, mesh, cameras, size, target_fill):
            self.calls.append(('render', mesh, size, target_fill)); assert all(c.t[2] == 1 for c in cameras)
            self.templates = [SimpleNamespace(rgb=np.full((4, 4, 3), i, dtype=np.uint8),
                depth=np.full((4, 4), 10 + i, dtype=np.float32), mask=np.ones((4, 4), dtype=bool),
                camera=camera(10 + i)) for i in range(3)]
            return self.templates
        def save_onboarding_cache(self, path, query, cameras, **extra):
            self.saved = list(cameras)
            np.savez_compressed(path, query_points=np.asarray(query, dtype=np.float32),
                camera_R=np.array([c.R for c in cameras], dtype=np.float32),
                camera_t=np.array([c.t for c in cameras], dtype=np.float32),
                camera_K=np.array([c.K for c in cameras], dtype=np.float32),
                camera_direction=np.array([c.direction for c in cameras], dtype=np.float32),
                camera_size=np.array([c.size for c in cameras], dtype=np.int32), **extra)
    ctx = {'folder': tmp_path / 'obj', 'model': tmp_path / 'original.ply',
           'pins': {'protocol': {'raw_query_points': 7}, 'model_sha256': 'fixed'}}
    return ctx, Durable(), API()


def test_exact_original_api_contract_and_fitted_cameras(case):
    ctx, durable, api = case
    receipt = m.render_stage(ctx, durable, api)
    assert receipt['status'] == 'complete' and receipt['raw_query_points'] == 7
    assert receipt['fitted_returned_cameras_saved']
    assert api.calls == [('load', ctx['model']), ('sample', 7, 0), ('cameras', 3, 4), ('render', ctx['model'], 4, .5)]
    assert all(c.t[2] >= 10 for c in api.saved)
    payload = ctx['folder'] / 'geometry'
    with np.load(payload / 'onboarding.npz') as data:
        np.testing.assert_array_equal(data['camera_t'][:, 2], [10, 11, 12])
        np.testing.assert_array_equal(data['template_depth'][:, 0, 0], [10, 11, 12])
    assert len(list((payload / 'rgb').glob('*.png'))) == 3
    assert Path(receipt['contact_sheet']['path']).parent == ctx['folder']


def test_complete_cache_reuses_only_matching_hashes(case):
    ctx, durable, api = case
    first = m.render_stage(ctx, durable, api); calls = list(api.calls)
    assert m.render_stage(ctx, durable, api) == first and api.calls == calls
    (ctx['folder'] / 'geometry/rgb/001.png').write_bytes(b'changed')
    with pytest.raises(Exception): m.render_stage(ctx, durable, api)


def test_identity_change_refuses_reuse(case):
    ctx, durable, api = case
    m.render_stage(ctx, durable, api)
    ctx['pins']['protocol']['raw_query_points'] = 8
    with pytest.raises(ValueError, match='identity'): m.render_stage(ctx, durable, api)


def test_partial_cache_never_counts_complete(case):
    ctx, durable, api = case
    durable.ensure_identity(ctx['folder'], ctx['pins'])
    (ctx['folder'] / 'geometry').mkdir()
    with pytest.raises(ValueError, match='Unreceipted'): m.render_stage(ctx, durable, api)
    assert not (ctx['folder'] / 'geometry_receipt.json').exists()


def test_renderer_failure_retains_partial_diagnostic_and_failed_receipt(case):
    ctx, durable, api = case
    def fail(*args, **kwargs): raise RuntimeError('renderer failed')
    api.render_templates = fail
    with pytest.raises(RuntimeError): m.render_stage(ctx, durable, api)
    receipt = json.loads((ctx['folder'] / 'geometry_receipt.json').read_text())
    assert receipt['status'] == 'failed' and receipt['reason'] == 'renderer failed'
    assert Path(receipt['partial_directory']).is_dir() and not (ctx['folder'] / 'geometry').exists()


@pytest.mark.parametrize('defect', ['depth_shape', 'rgb_shape', 'mask', 'camera', 'nonfinite', 'blank'])
def test_misaligned_renderer_outputs_are_rejected(case, defect):
    ctx, durable, api = case
    original = api.render_templates
    def broken(*args, **kwargs):
        output = original(*args, **kwargs)
        item = output[1]
        if defect == 'depth_shape': item.depth = np.ones((3, 4))
        if defect == 'rgb_shape': item.rgb = np.zeros((3, 4, 3), dtype=np.uint8)
        if defect == 'mask': item.mask = np.zeros((4, 4), dtype=bool)
        if defect == 'camera': item.camera.t = np.ones((3, 1))
        if defect == 'nonfinite': item.depth[0, 0] = np.nan
        if defect == 'blank': item.depth[:] = 0
        return output
    api.render_templates = broken
    with pytest.raises(ValueError): m.render_stage(ctx, durable, api)
    assert json.loads((ctx['folder'] / 'geometry_receipt.json').read_text())['status'] == 'failed'


def test_source_change_prevents_complete_receipt(case, monkeypatch):
    ctx, durable, api = case
    def changed(ctx): raise ValueError('source changed')
    monkeypatch.setattr(m, 'verify_inputs', changed)
    with pytest.raises(ValueError, match='source changed'): m.render_stage(ctx, durable, api)
    assert not (ctx['folder'] / 'geometry').exists()
    assert json.loads((ctx['folder'] / 'geometry_receipt.json').read_text())['status'] == 'failed'


def test_missing_contact_sheet_refuses_complete_cache(case):
    ctx, durable, api = case
    result = m.render_stage(ctx, durable, api)
    Path(result['contact_sheet']['path']).unlink()
    with pytest.raises(ValueError, match='contact sheet'): m.render_stage(ctx, durable, api)


def test_all_real_render_work_is_inside_shared_compute_lock(case, monkeypatch, tmp_path):
    ctx, durable, api = case
    monkeypatch.setattr(m.ref, 'pinned_guards', lambda root: (durable, object()))
    object_folder = tmp_path / 'outputs/freezev2_public_repo_20261030/base_public_v1/geometry_smoke/object'
    def context(root, suite, dataset, obj_id, raw_points, object_folder):
        assert durable.lock_active and object_folder is not None
        ctx['folder'] = object_folder
        return ctx
    def original(ctx): assert durable.lock_active; return api
    monkeypatch.setattr(m, 'context', context); monkeypatch.setattr(m, 'original_api', original)
    m.run(tmp_path, object_folder, 'tless', 1, 7)
    assert durable.lock_path == tmp_path / 'outputs/freezev2_public_repo_20261030/base_public_v1/compute.lock'
    assert not durable.lock_active


def test_cli_requires_explicit_raw_count_and_labels_output(case, monkeypatch, capsys):
    with pytest.raises(SystemExit): m.main(['--dataset', 'tless', '--obj-id', '1'])
    captured = []
    def run(*args):
        captured.append(args); return {'status': 'complete', 'payload_directory': 'geometry', 'raw_query_points': 5000, 'templates': 162}
    monkeypatch.setattr(m, 'run', run)
    m.main(['--root', str(HERE.parents[1]), '--dataset', 'tless', '--obj-id', '1', '--raw-points', '5000', '--output', str(HERE / 'object')])
    assert captured[0][-1] == 5000 and captured[0][2:4] == ('tless', 1)
    assert 'final retained' in m.__doc__


def test_package_alias_refuses_conflicting_owner(tmp_path):
    source, other, alias = tmp_path / 'source', tmp_path / 'other', tmp_path / 'alias'
    source.mkdir(); other.mkdir()
    m.package_alias(source, alias)
    assert alias.resolve() == source.resolve()
    with pytest.raises(ValueError, match='different owner'): m.package_alias(other, alias)
    plain = tmp_path / 'plain'; plain.mkdir()
    with pytest.raises(ValueError, match='non-symlink'): m.package_alias(source, plain)


def test_isolated_aliases_include_only_cached_trimesh(tmp_path):
    base = tmp_path / 'outputs/cache/fabrica_native_20261027/python'
    package = base / 'trimesh'; package.mkdir(parents=True); (package / '__init__.py').write_text('version marker')
    metadata = base / 'trimesh-4.4.1.dist-info'; metadata.mkdir()
    (metadata / 'METADATA').write_text('Metadata-Version: 2.1\nName: trimesh\nVersion: 4.4.1\n')
    (metadata / 'WHEEL').write_text('Root-Is-Purelib: true\nTag: py3-none-any\n')
    alias, pins = m.prepare_packages(tmp_path, tmp_path / 'newoutput')
    assert {p.name for p in alias.iterdir()} == {'trimesh', 'trimesh-4.4.1.dist-info'}
    assert pins['version'] == '4.4.1' and pins['package']['files']['__init__.py']
    (alias / 'unexpected').mkdir()
    with pytest.raises(ValueError, match='Unexpected package'): m.prepare_packages(tmp_path, tmp_path / 'newoutput')


def test_venv_executable_symlink_is_not_resolved(tmp_path, monkeypatch):
    prefix = tmp_path / 'venv'; (prefix / 'bin').mkdir(parents=True)
    executable = prefix / 'bin/python'; executable.symlink_to(sys.executable)
    monkeypatch.setattr(m.ref.sys, 'prefix', str(prefix))
    assert m.ref.evaluation_python({'python': str(executable)}) == executable
    m.ref.require_runtime({'python': str(executable)})
@pytest.fixture
def context_case(tmp_path, monkeypatch):
    suite = tmp_path / 'outputs/freezev2_public_repo_20261030/base_public_v1'
    suite.mkdir(parents=True)
    old = tmp_path / 'outputs/freezev2_bop_20261029'
    runtime = {'python': str(Path(sys.prefix) / 'bin/python'), 'toolkit_root': str(tmp_path / 'toolkit'),
               'toolkit_revision': m.ref.TOOLKIT_REVISION, 'renderer_type': 'vispy'}
    dump(old / 'evaluation_runtime.json', runtime)
    source_models = tmp_path / 'physical/models_cad'; source_models.mkdir(parents=True)
    model = source_models / 'obj_000001.ply'; model.write_bytes(b'unchanged native mm CAD')
    info = source_models / 'models_info.json'; dump(info, {'1': {'diameter': 20.}})
    alias = suite / 'bop_view/tless/models'; alias.parent.mkdir(parents=True); alias.symlink_to(source_models, target_is_directory=True)
    assets = {'status': 'verified_inference_assets_only_not_pose_performance', 'GT_pose_access': False,
              'datasets': {'tless': {'native_CAD_units': 'millimetres', 'required_object_ids': [1],
                'models_source': str(source_models), 'model_files': [{'path': str(model), 'sha256': m.ref.sha256(model)}],
                'models_info_sha256': m.ref.sha256(info)}}}
    dump(suite / 'assets_manifest.json', assets)
    monkeypatch.setattr(m.ref, 'git_identity', lambda path, rev: {'revision': rev, 'files': 'unchanged'})
    monkeypatch.setattr(m, 'prepare_packages', lambda root, out: (out / 'runtime/aliases', {'version': '4.4.1'}))
    return tmp_path, suite, assets, model, runtime


def test_context_pins_native_model_protocol_and_source(context_case):
    root, suite, assets, model, runtime = context_case
    ctx = m.context(root, suite, 'tless', 1, 5000, suite / 'geometry_smoke/object')
    pins = ctx['pins']; protocol = pins['protocol']
    assert protocol['raw_query_points'] == 5000 and protocol['CAD_units'] == 'millimetres'
    assert protocol['template_count'] == 162 and protocol['render_size'] == 480
    assert protocol['GT_access'] is False and protocol['learned_feature_extraction'] is False
    assert protocol['target_fill'] == .5 and protocol['seed'] == 0
    assert pins['model_sha256'] == m.ref.sha256(model)
    assert pins['public_source']['revision'] == m.ref.PUBLIC_REVISION
    assert pins['helper_sha256'] == m.ref.sha256(m.__file__)
    assert ctx['folder'] == suite / 'geometry_smoke/object'


@pytest.mark.parametrize('defect', ['incomplete_assets', 'GT_allowed', 'model_changed', 'wrong_units', 'unknown_object', 'outside_output'])
def test_context_rejects_unverified_inputs(context_case, defect):
    root, suite, assets, model, runtime = context_case
    obj, folder = 1, suite / 'geometry_smoke/object'
    if defect == 'incomplete_assets': assets['status'] = 'running'
    if defect == 'GT_allowed': assets['GT_pose_access'] = True
    if defect == 'model_changed': model.write_bytes(b'changed')
    if defect == 'wrong_units': assets['datasets']['tless']['native_CAD_units'] = 'metres'
    if defect == 'unknown_object': obj = 2
    if defect == 'outside_output': folder = root / 'existing_results'
    dump(suite / 'assets_manifest.json', assets)
    with pytest.raises((ValueError, FileNotFoundError)):
        m.context(root, suite, 'tless', obj, 5000, folder)


@pytest.mark.parametrize('raw_points', [0, -1, 3.5, True])
def test_raw_geometry_count_must_be_explicit_positive_integer(context_case, raw_points):
    root, suite, _, _, _ = context_case
    with pytest.raises(ValueError, match='positive integer'):
        m.context(root, suite, 'tless', 1, raw_points)