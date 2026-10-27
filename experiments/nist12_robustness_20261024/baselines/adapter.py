"""Retarget pinned full-scene runners without editing prior evidence or methods."""
from pathlib import Path
import shutil
import sys
import types
import common

ROOT = common.HERE
PRIOR = common.PRIOR
METHODS = ['cnos', 'sam6d_rgb', 'sam6d_rgbd', 'nids_basic_finite_templates', 'noctis', 'nemo']


def load_runner(relative, replacements=()):
    filename = PRIOR / relative
    source = filename.read_text(encoding='utf-8')
    for before, after in replacements:
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    module = types.ModuleType('retargeted_native_runner')
    module.__file__ = str(filename)
    sys.path.insert(0, str(filename.parent))
    exec(compile(source, str(filename), 'exec'), module.__dict__)
    return module


def prepare():
    capture = common.read(ROOT.parent / 'capture_manifest.json')
    scenes = [{key: row[key] for key in ('scene_id', 'rgb', 'depth', 'intrinsics')}
              for row in capture['scenes'] if row['condition'] == 'nominal']
    objects = common.read(ROOT.parent / 'catalog.json')['objects']
    assert len(scenes) == 20 and len(objects) == 12
    previous = common.read(PRIOR / 'inputs.json')['objects']
    assert objects == previous, 'Native catalogue must match the frozen reference caches exactly'
    hashes = {value: common.digest(common.path(value))
              for scene in scenes for key, value in scene.items() if key != 'scene_id'}
    hashes.update({obj['mesh']: common.digest(common.path(obj['mesh'])) for obj in objects})
    inputs = dict(scenes=scenes, objects=objects, depth_units='meters',
                  coordinate_frame='camera: x right, y down, z forward', input_sha256=hashes)
    common.save(ROOT / 'inputs.json', inputs)
    if not (ROOT / 'templates42').exists():
        shutil.copytree(PRIOR / 'templates42', ROOT / 'templates42')
    # References depend only on unchanged CADs, not on any scene or audit label.
    references = {}
    for method, name in [('cnos', 'reference_features.pt'), ('sam6d', 'reference_features.pt'),
                         ('nids_basic', 'native_reference_features.pt'), ('noctis', 'native_reference_features.pt')]:
        src = PRIOR / method / name
        references[common.relative(src)] = common.digest(src)
        directory = ROOT / ('nids_basic_finite_templates' if method == 'nids_basic' else method)
        directory.mkdir(exist_ok=True)
        dest = directory / name
        if not dest.exists():
            # Hard links preserve exact bytes and avoid duplicating multi-GB frozen caches.
            import os
            os.link(src, dest)
        assert common.digest(dest) == references[common.relative(src)]
    for source in sorted((PRIOR / 'nemo/memories').glob('*')):
        references[common.relative(source)] = common.digest(source)
    for source in [PRIOR / 'nemo_runner/primary_references.json',
                   PRIOR / 'nemo_runner/weights/checkpoint.pth']:
        references[common.relative(source)] = common.digest(source)
    paths = [Path(__file__), ROOT / 'common.py', ROOT / 'run_all.py',
             PRIOR / 'cnos_sam6d/run_native.py', PRIOR / 'nids_noctis/run_native.py',
             PRIOR / 'nids_noctis/compat.py', PRIOR / 'nids_noctis/run_finite_templates.py',
             PRIOR / 'nemo_runner/run.py', ROOT.parent / 'protocol.json']
    protocol = dict(frozen_before_new_model_inference=True, nominal_scenes_only=True,
        scene_ids=[s['scene_id'] for s in scenes], requests=240, methods=METHODS,
        input_sha256=common.digest(ROOT / 'inputs.json'), no_audit_data_or_scene_proposals=True,
        native_components='Pinned official sources, preprocessing, native template bank, thresholds, per-proposal class assignment and NMS from the immutable 20261023 full-scene benchmark.',
        execution_changes=['Retarget label-free input/output paths to this new trial.',
          'Replace only prior experiment-specific eight-scene count checks/receipts with twenty.',
          'Reuse exact frozen CAD-only reference descriptor tensors and corrected NeMO 32-view memories.',
          'NIDS practical control excludes the same six nonfinite USB CAD FFA slots before native max; finite scores remain bitwise unchanged. No adapter training.',
          'Do not rerun the known invalid unguarded NIDS reference route; disclose its inherited reference failure separately.'],
        source_sha256={common.relative(p):common.digest(p) for p in paths}, reference_sha256=references,
        excluded_NIDS_zero_based_object_template_slots=[[5, x] for x in (16,19,20,21,24,25)],
        NeMO_reference_route='Corrected custom-CAD native Pyrender 42-view bank, prospectively fixed 32-view subset; released NeMO encoder and detector. Not exact BOP PBR onboarding.')
    destination = ROOT / 'protocol.json'
    if destination.exists():
        assert common.read(destination) == protocol, 'Frozen baseline protocol differs'
    else:
        common.save(destination, protocol)
    print('Frozen label-free baseline addon: 20 scenes, 12 CADs, 240 queries per method.', flush=True)


def verify():
    receipt = common.read(ROOT / 'protocol.json')
    assert common.digest(ROOT / 'inputs.json') == receipt['input_sha256']
    for key in ('source_sha256', 'reference_sha256'):
        for filename, expected in receipt[key].items():
            assert common.digest(common.path(filename)) == expected, filename
    inputs = common.setup()
    for filename, expected in inputs['input_sha256'].items():
        assert common.digest(common.path(filename)) == expected, filename
    return inputs


def cnos_sam(method, phase):
    runner = load_runner('cnos_sam6d/run_native.py', [
        ('dict(passed=True, scenes=8,', 'dict(passed=True, scenes=len(manifest["scenes"]),'),
        ('dict(passed=True,scenes=8,method=output,', 'dict(passed=True,scenes=len(manifest["scenes"]),method=output,')])
    def cached_reference(method, native, detector, manifest):
        import torch
        detector.ref_data = {key: tensor.cuda() for key, tensor in torch.load(
            ROOT / method / 'reference_features.pt', map_location='cpu', weights_only=True).items()}
        return common.read(ROOT / 'templates42/metadata.json')
    runner.reference_features = cached_reference
    {'propose': runner.propose, 'match': runner.match}[phase](method)


def nids_noctis(method, phase):
    sys.path.insert(0, str(PRIOR / 'nids_noctis/hf_compat'))
    runner = load_runner('nids_noctis/run_native.py', [
        ('len(config["scenes"]) == 8', 'len(config["scenes"]) == 20')])
    internal = 'nids_basic' if method == 'nids_basic_finite_templates' else method
    def directory(value):
        result = ROOT / ('nids_basic_finite_templates' if value == 'nids_basic' else value)
        result.mkdir(exist_ok=True)
        return result
    runner.method_directory = directory
    configure = runner.configuration
    def configuration(value, inputs):
        config = configure(value, inputs)
        config.update(output_method=method, adapter_protocol_sha256=common.digest(ROOT / 'protocol.json'),
                      CAD_only_finite_guard=value == 'nids_basic')
        config['source_sha256'].update({common.relative(p):common.digest(p)
            for p in (Path(__file__), ROOT / 'common.py')})
        return config
    runner.configuration = configuration
    if internal == 'nids_basic':
        import torch
        import run_finite_templates
        build = runner.build_matcher
        def matcher(*args, **kwargs):
            model = build(*args, **kwargs)
            refs = torch.load(directory(internal) / 'native_reference_features.pt', map_location='cpu', weights_only=True)
            valid = torch.isfinite(refs['descriptors']).all(-1)
            assert (~valid).nonzero().tolist() == [[5, slot] for slot in (16,19,20,21,24,25)]
            assert valid.any(-1).all()
            model.matching_config.metric = run_finite_templates.ValidTemplateMetric(model.matching_config.metric, valid)
            return model
        runner.build_matcher = matcher
    inputs = common.setup()
    if phase == 'prepare':
        runner.prepare(internal, inputs)
    elif phase == 'propose':
        runner.proposals(internal, inputs)
    elif phase == 'match':
        runner.score(internal, inputs)


def nemo():
    runner = load_runner('nemo_runner/run.py')
    runner.OUTPUT = ROOT / 'nemo'
    def memories(model, manifest, references, device):
        result = {}
        for item in manifest['objects']:
            object_id = item['object_id']
            filename = PRIOR / 'nemo/memories' / f'obj_{object_id:06d}.pt'
            receipt = common.read(filename.with_suffix('.json'))
            assert receipt['reference_sha256'] == references['objects'][str(object_id)]['sha256']
            assert receipt['checkpoint_sha256'] == common.digest(runner.CHECKPOINT)
            result[object_id] = filename
        return result
    runner.build_memories = memories
    sys.argv = [str(Path(__file__))]
    runner.main()


if __name__ == '__main__':
    if sys.argv[1] == 'prepare':
        prepare()
    else:
        verify()
        method, phase = sys.argv[1:3]
        if method in ('cnos', 'sam6d'):
            cnos_sam(method, phase)
        elif method in ('nids_basic_finite_templates', 'noctis'):
            nids_noctis(method, phase)
        elif method == 'nemo':
            nemo()
