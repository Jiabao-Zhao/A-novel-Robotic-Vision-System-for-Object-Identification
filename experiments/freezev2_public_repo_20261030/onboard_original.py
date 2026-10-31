"""Cache geometry/RGB/depth from the pinned, unmodified public onboarding API.

--raw-points is mandatory: it is a pre-visibility sample count, not the paper's
final retained point count. This stage extracts no learned descriptors or poses.
"""
from __future__ import annotations
import argparse
import datetime as dt
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import uuid
import numpy as np

import reference_gate as ref

DATASETS = ('lmo', 'tless', 'tudl', 'icbin', 'itodd', 'hb', 'ycbv')
VIEWS, SIZE, SEED, TARGET_FILL = 162, 480, 0, 0.5
TRIMESH_VERSION = '4.4.1'


def tree_identity(folder):
    folder = Path(folder).resolve()
    records = {}
    for path in sorted(folder.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
            if not path.resolve().is_relative_to(folder):
                raise ValueError('Dependency file escapes its pinned package')
            records[str(path.relative_to(folder))] = ref.sha256(path)
    if not records:
        raise ValueError('Empty cached dependency package')
    return {'files': records, 'tree_sha256': ref.digest(records)}


def package_alias(source, alias):
    source, alias = Path(source).resolve(), Path(alias)
    if not source.is_dir():
        raise FileNotFoundError(source)
    alias.parent.mkdir(parents=True, exist_ok=True)
    if alias.is_symlink():
        if alias.resolve() != source:
            raise ValueError('Cached dependency alias has a different owner')
    elif alias.exists():
        raise ValueError('Refusing to overwrite a non-symlink dependency path')
    else:
        alias.symlink_to(source, target_is_directory=True)
    return alias


def prepare_packages(root, output):
    if os.name != 'posix':
        raise RuntimeError('Use the recorded WSL evaluation Python for onboarding')
    source = Path(root) / 'outputs/cache/fabrica_native_20261027/python'
    package, metadata = source / 'trimesh', source / f'trimesh-{TRIMESH_VERSION}.dist-info'
    if importlib.metadata.PathDistribution(metadata).version != TRIMESH_VERSION:
        raise ValueError('Unexpected cached Trimesh version')
    wheel = (metadata / 'WHEEL').read_text(encoding='utf-8')
    if 'Root-Is-Purelib: true' not in wheel or 'Tag: py3-none-any' not in wheel:
        raise ValueError('Cached Trimesh must be the recorded pure-Python distribution')
    pins = {'version': TRIMESH_VERSION, 'source': str(package.resolve()),
            'package': tree_identity(package), 'metadata': tree_identity(metadata)}
    aliases = Path(output) / 'runtime/onboarding_packages'
    if aliases.exists() and {p.name for p in aliases.iterdir()} - {'trimesh', metadata.name}:
        raise ValueError('Unexpected package in isolated onboarding aliases')
    package_alias(package, aliases / 'trimesh')
    package_alias(metadata, aliases / metadata.name)
    return aliases, pins


def context(root, output, dataset, obj_id, raw_points, object_folder=None):
    if dataset not in DATASETS or isinstance(obj_id, bool) or obj_id < 1:
        raise ValueError('Unsupported dataset or object ID')
    if isinstance(raw_points, bool) or not isinstance(raw_points, int) or raw_points < 1:
        raise ValueError('Specify a positive integer raw CAD sample count')
    root, output = ref.local(root), ref.local(output)
    old = root / 'outputs/freezev2_bop_20261029'
    runtime_path = old / 'evaluation_runtime.json'
    runtime = ref.read(runtime_path)
    # Check the venv prefix, preserving the executable symlink path.
    ref.require_runtime(runtime)
    source = old / 'external/FreezeV2-Re'
    source_identity = ref.git_identity(source, ref.PUBLIC_REVISION)
    toolkit = ref.local(runtime['toolkit_root'])
    toolkit_identity = ref.git_identity(toolkit, ref.TOOLKIT_REVISION)
    if runtime.get('toolkit_revision') != ref.TOOLKIT_REVISION or runtime.get('renderer_type') != 'vispy':
        raise ValueError('Unexpected pinned BOP renderer runtime')
    assets_path = output / 'assets_manifest.json'
    assets = ref.read(assets_path)
    if assets.get('status') != 'verified_inference_assets_only_not_pose_performance' or assets.get('GT_pose_access') is not False:
        raise ValueError('Complete inference-only asset preparation is required')
    asset = assets['datasets'][dataset]
    if asset.get('native_CAD_units') != 'millimetres' or obj_id not in asset['required_object_ids']:
        raise ValueError('CAD units or official target object identity differs')
    model = output / 'bop_view' / dataset / 'models' / f'obj_{obj_id:06d}.ply'
    info = model.parent / 'models_info.json'
    source_model = ref.local(asset['models_source']) / model.name
    expected = [row for row in asset['model_files'] if ref.local(row['path']) == source_model]
    model_sha = ref.sha256(model)
    if len(expected) != 1 or model.resolve() != source_model.resolve() or model_sha != expected[0]['sha256']:
        raise ValueError('Prepared CAD alias or physical model checksum differs')
    if ref.sha256(info) != asset['models_info_sha256']:
        raise ValueError('Native CAD metadata checksum differs')
    aliases, trimesh = prepare_packages(root, output)
    protocol = {'dataset': dataset, 'obj_id': obj_id, 'raw_query_points': raw_points,
                'seed': SEED, 'template_count': VIEWS, 'render_size': SIZE, 'target_fill': TARGET_FILL,
                'CAD_units': 'millimetres', 'depth_units': 'millimetres',
                'camera_frame': 'OpenCV model-to-camera: +x right,+y down,+z forward',
                'query_count_role': 'raw pre-visibility samples; not final retained descriptor points',
                'GT_access': False, 'learned_feature_extraction': False}
    pins = {'protocol': protocol, 'helper_sha256': ref.sha256(__file__),
            'public_source': source_identity, 'toolkit_source': toolkit_identity,
            'evaluation_runtime_sha256': ref.sha256(runtime_path), 'Trimesh': trimesh,
            'model_source': str(source_model), 'model_sha256': model_sha,
            'models_info_sha256': ref.sha256(info), 'assets_manifest_sha256': ref.sha256(assets_path),
            'frozen_guard_manifest_sha256': ref.FROZEN_MANIFEST_SHA256}
    folder = ref.local(object_folder) if object_folder is not None else output / 'onboarding' / dataset / f'obj_{obj_id:06d}' / f'raw_{raw_points:08d}'
    if not folder.is_relative_to(output) or folder == output:
        raise ValueError('Per-object output must stay inside this isolated study output')
    return {'root': root, 'output': output, 'folder': folder, 'runtime': runtime,
            'source': source, 'toolkit': toolkit, 'aliases': aliases, 'model': model, 'pins': pins}


def original_api(ctx):
    for name, module in list(sys.modules.items()):
        if name == 'freezev2' or name.startswith('freezev2.'):
            path = getattr(module, '__file__', None)
            if path and not Path(path).resolve().is_relative_to(ctx['source'].resolve()):
                raise RuntimeError('Use a fresh process for original public onboarding')
    sys.path[:0] = [str(ctx['source']), str(ctx['toolkit']), str(ctx['aliases'])]
    import trimesh
    if trimesh.__version__ != TRIMESH_VERSION or not Path(trimesh.__file__).resolve().is_relative_to(Path(ctx['pins']['Trimesh']['source'])):
        raise ValueError('Loaded Trimesh differs from the recorded package')
    # Environment changes affect this renderer process only.
    os.environ.update(ctx['runtime'].get('environment', {}))
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '2'
    return importlib.import_module('freezev2.onboard')


def validate_templates(templates, raw_points, query):
    query = np.asarray(query)
    if query.shape != (raw_points, 3) or not np.isfinite(query).all():
        raise ValueError('Original raw query samples are nonfinite or misaligned')
    if len(templates) != VIEWS:
        raise ValueError('Original renderer returned an incomplete template set')
    for item in templates:
        rgb, depth, camera = np.asarray(item.rgb), np.asarray(item.depth), item.camera
        if rgb.shape != (SIZE, SIZE, 3) or rgb.dtype != np.uint8:
            raise ValueError('RGB template dimensions/type differ')
        if depth.shape != (SIZE, SIZE) or not np.isfinite(depth).all() or np.min(depth) < 0 or not np.any(depth > 0):
            raise ValueError('Rendered depth dimensions, finiteness, or foreground differ')
        if np.asarray(item.mask).shape != depth.shape or not np.array_equal(item.mask, depth > 0):
            raise ValueError('Template mask is not aligned with rendered depth')
        for value, shape in ((camera.R, (3, 3)), (camera.t, (3,)), (camera.K, (3, 3)), (camera.direction, (3,))):
            if np.asarray(value).shape != shape or not np.isfinite(value).all():
                raise ValueError('Fitted returned camera is invalid')
        if camera.size != SIZE or not np.allclose(camera.R @ camera.R.T, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(camera.R), 1, atol=1e-5):
            raise ValueError('Fitted camera frame or dimensions differ')
        if camera.K[0, 0] <= 0 or camera.K[1, 1] <= 0:
            raise ValueError('Invalid fitted focal length')


def validate_saved(folder, raw_points):
    from PIL import Image
    folder = Path(folder)
    expected = {'onboarding.npz'} | {f'rgb/{i:03d}.png' for i in range(VIEWS)}
    actual = {str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()}
    if actual != expected:
        raise ValueError('Saved geometry payload is incomplete or has unexpected files')
    with np.load(folder / 'onboarding.npz', allow_pickle=False) as arrays:
        shapes = {'query_points': (raw_points, 3), 'camera_R': (VIEWS, 3, 3), 'camera_t': (VIEWS, 3),
                  'camera_K': (VIEWS, 3, 3), 'camera_direction': (VIEWS, 3), 'camera_size': (VIEWS,),
                  'template_depth': (VIEWS, SIZE, SIZE)}
        if set(arrays.files) != set(shapes):
            raise ValueError('Saved original onboarding cache arrays differ')
        for key, shape in shapes.items():
            if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
                raise ValueError('Saved cache arrays are nonfinite or misaligned: ' + key)
        if not np.all(arrays['camera_size'] == SIZE) or np.any(arrays['template_depth'] < 0):
            raise ValueError('Saved cache camera/depth conventions differ')
        if not np.all(np.any(arrays['template_depth'] > 0, axis=(1, 2))):
            raise ValueError('Saved template lacks depth foreground')
    for i in range(VIEWS):
        with Image.open(folder / 'rgb' / f'{i:03d}.png') as image:
            if image.mode != 'RGB' or image.size != (SIZE, SIZE):
                raise ValueError('Saved RGB template and depth dimensions differ')
    return {name: ref.sha256(folder / name) for name in sorted(expected)}


def contact_sheet(templates, folder):
    from PIL import Image, ImageDraw
    indices = [i for i in (0, 20, 40, 60, 80, 100, 120, 140, 161) if i < len(templates)]
    image = Image.new('RGB', (720, 780), (35, 35, 35))
    draw = ImageDraw.Draw(image)
    for cell, idx in enumerate(indices):
        x, y = (cell % 3) * 240, (cell // 3) * 260
        tile = Image.fromarray(templates[idx].rgb).resize((240, 240))
        image.paste(tile, (x, y))
        draw.text((x + 6, y + 242), f'Original template {idx:03d}', fill='white')
    target = Path(folder) / 'template_contact_sheet.png'
    temporary = target.with_name(target.name + '.' + uuid.uuid4().hex + '.tmp')
    image.save(temporary, format='PNG'); os.replace(temporary, target)
    return {'path': str(target), 'sha256': ref.sha256(target), 'template_indices': indices,
            'content': 'RGB renderer output only; resized for display, no GT overlays'}


def verify_inputs(ctx):
    if ref.git_identity(ctx['source'], ref.PUBLIC_REVISION) != ctx['pins']['public_source'] or ref.git_identity(ctx['toolkit'], ref.TOOLKIT_REVISION) != ctx['pins']['toolkit_source']:
        raise ValueError('Original source changed during onboarding')
    if ref.sha256(ctx['model']) != ctx['pins']['model_sha256'] or ref.sha256(__file__) != ctx['pins']['helper_sha256']:
        raise ValueError('CAD or onboarding helper changed during rendering')
    if tree_identity(ctx['pins']['Trimesh']['source']) != ctx['pins']['Trimesh']['package']:
        raise ValueError('Cached Trimesh source changed during onboarding')


def render_stage(ctx, durable, api):
    from PIL import Image
    folder, pins = ctx['folder'], ctx['pins']
    durable.ensure_identity(folder, pins)
    receipt_path, payload = folder / 'geometry_receipt.json', folder / 'geometry'
    count = pins['protocol']['raw_query_points']
    if receipt_path.exists():
        receipt = ref.read(receipt_path)
        if receipt.get('source_pins') != pins:
            raise ValueError('Existing onboarding receipt identity differs')
        if receipt.get('status') == 'complete':
            if validate_saved(payload, count) != receipt.get('payload_sha256'):
                raise ValueError('Completed onboarding payload checksum differs')
            sheet = receipt.get('contact_sheet', {})
            if not Path(sheet.get('path', '')).is_file() or ref.sha256(sheet['path']) != sheet.get('sha256'):
                raise ValueError('Completed contact sheet is absent or changed')
            return receipt
    if payload.exists():
        raise ValueError('Unreceipted or incomplete geometry payload; preserve it and use a new output')
    staging = folder / ('partial_' + uuid.uuid4().hex)
    staging.mkdir(); (staging / 'rgb').mkdir()
    start = time.perf_counter()
    receipt = {'schema': 'unchanged-public-geometry-onboarding-v1', 'status': 'running',
               'source_pins': pins, 'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
               'partial_directory': str(staging), 'claim_scope': 'Raw geometry/templates only; no features or pose performance'}
    durable.atomic_json(receipt_path, receipt)
    try:
        mesh = api.load_mesh(ctx['model'])
        query = api.sample_query_points(mesh, n=count, seed=SEED)
        initial_cameras = api.make_template_cameras(n=VIEWS, size=SIZE)
        templates = api.render_templates(ctx['model'], initial_cameras, size=SIZE, target_fill=TARGET_FILL)
        validate_templates(templates, count, query)
        # render_templates fits each camera distance. Persist its returned camera,
        # never the unfitted unit-distance camera supplied to the renderer.
        cameras = [item.camera for item in templates]
        api.save_onboarding_cache(staging / 'onboarding.npz', query, cameras,
                                  template_depth=np.stack([item.depth for item in templates]).astype(np.float32))
        for i, item in enumerate(templates):
            Image.fromarray(item.rgb).save(staging / 'rgb' / f'{i:03d}.png')
        hashes = validate_saved(staging, count)
        # Exact alignment check against the returned fitted cameras/depth.
        with np.load(staging / 'onboarding.npz', allow_pickle=False) as saved:
            for key, values in (('camera_R', [c.R for c in cameras]), ('camera_t', [c.t for c in cameras]),
                                ('camera_K', [c.K for c in cameras]), ('camera_direction', [c.direction for c in cameras]),
                                ('template_depth', [item.depth for item in templates])):
                if not np.array_equal(saved[key], np.asarray(values, dtype=np.float32)):
                    raise ValueError('Saved original API arrays changed camera/depth alignment: ' + key)
        verify_inputs(ctx)
        sheet = contact_sheet(templates, folder)
        os.replace(staging, payload)
        receipt.update(status='complete', payload_directory=str(payload), payload_sha256=hashes,
                       completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(), runtime_s=time.perf_counter() - start,
                       raw_query_points=count, templates=VIEWS, fitted_returned_cameras_saved=True, contact_sheet=sheet)
        durable.atomic_json(receipt_path, receipt)
        return receipt
    except Exception as exc:
        receipt.update(status='failed', exception_type=type(exc).__name__, reason=str(exc), runtime_s=time.perf_counter() - start)
        durable.atomic_json(receipt_path, receipt)
        raise


def run(root, output, dataset, obj_id, raw_points):
    root = ref.local(root)
    suite = root / 'outputs/freezev2_public_repo_20261030/base_public_v1'
    durable, _owned = ref.pinned_guards(root)
    # VisPy rendering is synchronous in this process; this OS handle stays open
    # throughout rendering and cache verification. No second renderer is launched.
    with durable.worker_lock(suite / 'compute.lock'):
        durable.require_disk(suite, 15)
        ctx = context(root, suite, dataset, obj_id, raw_points, object_folder=output)
        return render_stage(ctx, durable, original_api(ctx))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--dataset', required=True, choices=DATASETS)
    parser.add_argument('--obj-id', required=True, type=int)
    parser.add_argument('--raw-points', required=True, type=int, help='Explicit raw geometry sample count; not final retained descriptor points')
    parser.add_argument('--output', type=Path, help='Per-object geometry folder inside the fixed study output; shared compute.lock is unchanged')
    args = parser.parse_args(argv)
    root = ref.local(args.root)
    output = ref.local(args.output) if args.output is not None else None
    result = run(root, output, args.dataset, args.obj_id, args.raw_points)
    print(json.dumps({'status': result['status'], 'output': result['payload_directory'],
                      'raw_query_points': result['raw_query_points'], 'templates': result['templates']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())