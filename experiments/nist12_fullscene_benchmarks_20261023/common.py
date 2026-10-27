"""Paths and label-free input manifest for the full-scene comparison."""
from pathlib import Path
import hashlib
import json
import sys
import shutil

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCES = HERE / 'sources'
WEIGHTS = HERE / 'weights'
DEPS = HERE / 'deps'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def path(value):
    text = str(value).replace('\\', '/')
    retired = 'C:/Users/Jiabao Zhao/Documents/Codex/2026-06-02/can-you-add-this-skill-to'
    for prefix in (retired, '/mnt/c' + retired[2:]):
        if text == prefix or text.startswith(prefix + '/'):
            suffix = text[len(prefix):].lstrip('/')
            if '..' in suffix.split('/') or ':' in suffix:
                raise ValueError('Unsafe retired-workspace path suffix')
            return REPO / 'outputs/datasets/nist_task_board/legacy_workspace_inputs' / suffix
    if len(text) > 1 and text[1] == ':' and sys.platform != 'win32':
        text = '/mnt/' + text[0].lower() + text[2:]
    p = Path(text)
    return p if p.is_absolute() else REPO / p


def relative(value):
    return Path(value).resolve().relative_to(REPO).as_posix()


def setup():
    sys.path.insert(0, str(DEPS))
    import os
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    os.environ.setdefault('YOLO_CONFIG_DIR', str(HERE / 'ultralytics_config'))
    return read(HERE / 'inputs.json')


def prepare():
    original = HERE.parent / 'nist12_qwen_clip_reranking_20261023'
    capture = read(original / 'data/capture_manifest.json')['states']
    catalog = read(original / 'data/cad_inputs.json')
    scenes, objects, hashes = [], [], {}
    for scene_id, item in capture.items():
        row = {'scene_id': scene_id, 'rgb': item['rgb_file'],
               'depth': item['depth_file'], 'intrinsics': item['intrinsics_file']}
        for key in ('rgb', 'depth', 'intrinsics'):
            hashes[row[key]] = digest(path(row[key]))
        scenes.append(row)
    for index, (target, item) in enumerate(catalog.items(), 1):
        mesh = path(item['mesh_file'])
        assert digest(mesh) == item['mesh_sha256']
        snapshot = HERE / 'cad_assets' / (target + mesh.suffix)
        snapshot.parent.mkdir(exist_ok=True)
        if not snapshot.exists():
            shutil.copyfile(mesh, snapshot)
        assert digest(snapshot) == item['mesh_sha256']
        objects.append({'object_id': index, 'target': target, 'mesh': relative(snapshot),
                        'color_rgb': item['color_rgb'], 'native_extent_m': item['native_extent_m']})
        hashes[relative(snapshot)] = item['mesh_sha256']
    assert len(scenes) == 8 and len(objects) == 12
    value = {'scenes': scenes, 'objects': objects, 'depth_units': 'meters',
             'coordinate_frame': 'camera: x right, y down, z forward', 'input_sha256': hashes}
    destination = HERE / 'inputs.json'
    if destination.exists():
        assert read(destination) == value
    else:
        save(destination, value)
    save(HERE / 'protocol.json', {
        'task': 'Supplied CAD identity -> highest-scored retained detection of that identity in full scene',
        'scenes': [r['scene_id'] for r in scenes], 'object_count': 12, 'requests': 96,
        'methods': ['cnos', 'sam6d_rgb', 'sam6d_rgbd', 'nids_basic', 'noctis', 'nemo', 'ours'],
        'shared_inputs': 'Full balanced-light RGB, single-camera metric depth, intrinsics, native-scale CAD catalogue',
        'proposal_policy': 'Every method generates its own proposals; no supplied scene crops, masks, poses or proposal count.',
        'native_settings': 'Official method preprocessing, template bank and postprocessing. Record source commits and any compatibility or batch-size adjustments.',
        'ours': 'Fresh camera-frame depth localization, unchanged CLIP image/text top-five retrieval and zero-clearance table-constrained geometry selection.',
        'ours_localization': 'Historical localizer configuration with no world-coordinate workspace crop or camera extrinsics; no forced object count.',
        'evaluation': 'Freeze label-free outputs first. Audit labels are used only for per-object evaluation; no detection is a miss, exact ties unresolved.',
        'evaluation_primary': 'Correct CAD identity and highest-scored retained target box IoU >= 0.5 with target audit box.',
        'evaluation_secondary': 'Mask IoU, own proposal recall, wrong physical object, localization miss and classification miss separately.',
        'training': 'No target training; NIDS adapter disabled; all weights frozen.',
        'hardware': 'RTX 3060 Ti 8GB; sequential GPU model loads; cache outputs.',
        'protected_prompt_sha256': digest(REPO / 'docs/cad_view_prompt.txt'),
        'inputs_sha256': digest(destination), 'frozen_before_new_model_inference': True,
    })
    print(f'Prepared {len(scenes)} raw scenes and {len(objects)} CADs; no scene proposals in manifest.')


if __name__ == '__main__':
    prepare()
