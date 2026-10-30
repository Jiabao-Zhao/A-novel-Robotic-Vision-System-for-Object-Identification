"""Bounded CPU-only input/texture/mapping diagnostics for the seven BOP sets.

These are data and interface checks, not learned inference, mask accuracy or
pose evaluation.  No scene_gt, annotation masks, GT poses, or trained model is
read.  Public predicted mask JSON is streamed with a bounded record buffer.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw

from .data import BOPDataset, DATASETS, SPLITS, read_ply_header, sha256, write_json
from .geometry import backproject, observed_clouds, project, surface_sample, transform_points, visible_surface
from .masks import MaskCandidate, decode_rle, load_localization_targets


def local_path(value):
    value = str(value)
    if os.name != 'nt' and re.match(r'^[A-Za-z]:[\\/]', value):
        value = '/mnt/' + value[0].lower() + '/' + value[3:].replace('\\', '/')
    return Path(value).expanduser().resolve()


def iter_json_array(path, *, chunk_chars=1024 * 1024, max_record_chars=8 * 1024 * 1024):
    """Yield records from one JSON array without materialising the whole file.

    The official public predictions use a top-level array. Wrappers, trailing
    data, malformed records and oversized records fail explicitly. Bounds are
    in decoded characters; dense masks are decoded only after cohort selection.
    Supports a gzip-compressed array as well. No on-disk extraction is made.
    """
    if not isinstance(chunk_chars, int) or not 1 <= chunk_chars <= max_record_chars:
        raise ValueError('JSON chunk size must be positive and at most the record bound')
    if not isinstance(max_record_chars, int) or max_record_chars > 64 * 1024 * 1024:
        raise ValueError('JSON record bound must be a positive integer at most64Mi characters')
    opener = gzip.open if Path(path).suffix == '.gz' else open
    decoder = json.JSONDecoder()
    with opener(path, 'rt', encoding='utf-8-sig') as stream:
        buffer, pos, eof = '', 0, False

        def refill():
            nonlocal buffer, pos, eof
            remaining = buffer[pos:]
            if len(remaining) > max_record_chars:
                raise ValueError('JSON record exceeds the configured diagnostic memory bound')
            addition = stream.read(chunk_chars)
            buffer, pos = remaining + addition, 0
            eof = not addition

        def whitespace():
            nonlocal pos
            while True:
                while pos < len(buffer) and buffer[pos].isspace():
                    pos += 1
                if pos < len(buffer) or eof:
                    return
                refill()

        refill()
        whitespace()
        if pos >= len(buffer) or buffer[pos] != '[':
            raise ValueError('Bounded public-mask reader requires a top-level JSON array')
        pos += 1
        first = True
        while True:
            whitespace()
            if pos < len(buffer) and buffer[pos] == ']':
                pos += 1
                break
            if not first:
                if pos >= len(buffer) or buffer[pos] != ',':
                    raise ValueError('Malformed JSON array separator or missing closing bracket')
                pos += 1
                whitespace()
                if pos >= len(buffer) or buffer[pos] == ']':
                    raise ValueError('Trailing comma or missing JSON array record')
            while True:
                try:
                    value, end = decoder.raw_decode(buffer, pos)
                    if end == len(buffer) and not eof:
                        # A number can be complete at the chunk boundary but
                        # continue in the next chunk (e.g. '1' then '2,').
                        refill()
                        continue
                    if end < len(buffer) and buffer[end] not in ',] \t\r\n':
                        if eof:
                            raise ValueError('Invalid token after a JSON array value')
                        refill()
                        continue
                    if end - pos > max_record_chars:
                        raise ValueError('JSON record exceeds the configured diagnostic memory bound')
                    pos = end
                    break
                except json.JSONDecodeError as exc:
                    if eof:
                        raise ValueError(f'Incomplete/malformed JSON array record: {exc}') from exc
                    refill()
            yield value
            first = False
        whitespace()
        if pos != len(buffer):
            raise ValueError('Unexpected data after the JSON prediction array')


def scan_texture_assets(root, datasets=DATASETS):
    """Inspect every CAD header and referenced texture, one file at a time."""
    root, models, missing = local_path(root), [], []
    for dataset in datasets:
        directory = root / dataset / ('models_cad' if dataset == 'tless' else 'models')
        for model in sorted(directory.glob('obj_*.ply')):
            header = read_ply_header(model)
            row = {'dataset': dataset, 'object_id': int(model.stem[4:]), 'model_path': str(model),
                   'ply_format': header['format'], 'vertices': header['vertex_count'], 'faces': header['face_count'],
                   'has_vertex_rgb': {'red', 'green', 'blue'} <= {p[1] for p in header['vertex_properties']},
                   'texture_path': None}
            if header['texture_filename']:
                texture = model.parent / header['texture_filename']
                row['texture_path'] = str(texture)
                if not texture.is_file():
                    missing.append(str(texture))
                else:
                    with Image.open(texture) as image:
                        row.update(texture_size_wh=list(image.size), texture_mode=image.mode)
                    row['texture_sha256'] = sha256(texture)
            models.append(row)
    return {'models': models, 'model_count': len(models),
            'textured_models': sum(row['texture_path'] is not None for row in models),
            'missing_textures': missing}


def _thumbnail(rgb, label):
    image = Image.fromarray(np.asarray(rgb, np.uint8)).convert('RGB')
    image.thumbnail((256, 192))
    result = Image.new('RGB', (256, 216), 'white')
    result.paste(image, ((256 - image.width) // 2, 0))
    ImageDraw.Draw(result).text((4, 196), label, fill='black')
    return result


def _select_masks(asset, keys, max_per_key=2):
    """Keep only bounded top-confidence raw records for diagnostic target keys."""
    path = local_path(asset['path'])
    if asset.get('source_kind') != 'predicted':
        raise ValueError('Dataset diagnostics accept declared public predicted masks only')
    if asset.get('sha256') and sha256(path) != asset['sha256']:
        raise ValueError('Predicted-mask file differs from its acquired asset manifest')
    selected, seen = {key: [] for key in keys}, 0
    for record_index, row in enumerate(iter_json_array(path)):
        seen += 1
        if not isinstance(row, dict):
            raise ValueError('Public mask records must be JSON objects')
        if any(name in row for name in ('gt_id', 'visib_fract', 'cam_R_m2c', 'cam_t_m2c')):
            raise ValueError('Ground-truth annotation fields are forbidden in predicted mask diagnostics')
        key = (int(row['scene_id']), int(row.get('image_id', row.get('im_id'))), int(row['category_id']))
        if key not in selected:
            continue
        score = float(row['score'])
        if not math.isfinite(score):
            raise ValueError('Nonfinite external segmentation confidence')
        selected[key].append((score, record_index, row))
        selected[key].sort(key=lambda entry: (-entry[0], entry[1]))
        del selected[key][max_per_key:]
    return selected, seen


def _sensor_projection_check(frame):
    depth = frame['depth_m']
    valid = np.isfinite(depth) & (depth > 0)
    yy, xx = np.nonzero(valid)
    if not len(xx):
        raise ValueError('Image has no finite positive sensor depth')
    sample = np.linspace(0, len(xx) - 1, min(128, len(xx)), dtype=int)
    uv = np.column_stack([xx[sample], yy[sample]])
    xyz = backproject(uv, depth[yy[sample], xx[sample]], frame['K'])
    projected, visible = project(xyz, frame['K'])
    residual = float(np.max(np.abs(projected - uv)))
    if not visible.all() or residual > 1e-7:
        raise ValueError('Calibrated sensor project/backproject round trip failed')
    return {'valid_depth_pixels': len(xx), 'invalid_depth_pixels': int(depth.size - len(xx)),
            'positive_depth_min_m': float(depth[valid].min()), 'positive_depth_max_m': float(depth[valid].max()),
            'sampled_sensor_pixels': len(sample), 'projection_round_trip_max_px': residual,
            'scope': 'sensor pixels only; no mask or pose prediction generated'}


def run_dataset_checks(root, output, *, datasets=DATASETS, samples_per_dataset=2,
                       asset_manifest=None, cnos_poses=None, render_size=96, render_cads=True):
    """Check bounded actual assets and write an audit/gallery to a new folder.

    Per dataset, sample the first/last pinned images without inspecting GT.
    Render one CAD in two canonical views at<=256px. At most two public masks
    per selected image/official target object are decoded. This diagnostic cap
    does not change the inference candidate selection policy.
    """
    if not isinstance(samples_per_dataset, int) or not 1 <= samples_per_dataset <= 8:
        raise ValueError('samples_per_dataset must be in [1,8] for bounded diagnostics')
    if not isinstance(render_size, int) or not 16 <= render_size <= 256:
        raise ValueError('render_size must be in [16,256] for bounded CPU diagnostics')
    if not datasets or any(name not in DATASETS for name in datasets):
        raise ValueError('Choose one or more supported BOP datasets')
    root, output = local_path(root), local_path(output)
    if output.exists():
        raise ValueError('Diagnostic output already exists; preserve it and select a new directory')
    output.mkdir(parents=True)
    started = time.perf_counter()
    manifest_path = root / 'full_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assets = json.loads(local_path(asset_manifest).read_text()) if asset_manifest else None
    cameras = np.load(local_path(cnos_poses), allow_pickle=False) if cnos_poses else None
    if cameras is not None and cameras.shape != (162, 4, 4):
        raise ValueError('Canonical camera array must be162x4x4')
    report = {'schema_version': 1, 'utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'bounded CPU input/texture/RGB-D/predicted-mask mapping checks; no learned inference or accuracy metrics',
              'GT_pose_access': False, 'dataset_root': str(root), 'manifest_sha256': sha256(manifest_path),
              'limits': {'images_per_dataset': samples_per_dataset, 'masks_per_image_target': 2,
                         'render_size': render_size, 'CADs_rendered_per_dataset': 1,
                         'JSON_chunk_chars': 1024 * 1024, 'JSON_max_record_chars': 8 * 1024 * 1024},
              'disk_free_bytes_before': shutil.disk_usage(output).free,
              'texture_assets': scan_texture_assets(root, datasets), 'datasets': {}}
    thumbnails = []
    for name in datasets:
        rows = sorted((row for row in manifest['images'] if row['dataset'] == name), key=lambda row: (row['scene_id'], row['im_id']))
        if not rows:
            raise ValueError(f'No existing pinned images for {name}')
        selected = [rows[int(i)] for i in np.unique(np.linspace(0, len(rows) - 1, min(samples_per_dataset, len(rows)), dtype=int))]
        dataset = BOPDataset(root, name)
        missing = [str(root / row[key]) for row in rows for key in ('rgb_path', 'depth_path', 'camera_path') if not (root / row[key]).is_file()]
        result = {'split': dataset.split, 'pinned_images': len(rows), 'missing_input_paths': missing,
                  'CAD_objects': len(dataset.models_info), 'object_ids': sorted(map(int, dataset.models_info)), 'frames': []}
        per_dataset_assets = assets['datasets'].get(name) if assets else None
        chosen_targets = {}
        if per_dataset_assets:
            targets = load_localization_targets(local_path(per_dataset_assets['targets']['path']))
            for row in selected:
                choices = [target for target in targets if (target.scene_id, target.image_id) == (row['scene_id'], row['im_id'])]
                if choices:
                    chosen_targets[(row['scene_id'], row['im_id'])] = min(choices, key=lambda target: target.object_id)
            target_keys = {(t.scene_id, t.image_id, t.object_id) for t in chosen_targets.values()}
            selected_masks, records = _select_masks(per_dataset_assets['masks'], target_keys)
            result['predicted_masks'] = {'path': str(local_path(per_dataset_assets['masks']['path'])),
                                         'sha256': per_dataset_assets['masks']['sha256'], 'streamed_records': records}
        else:
            selected_masks = {}
            result['predicted_masks'] = {'status': 'not_checked', 'reason': 'No acquired public-mask manifest supplied'}
        folder = output / name
        folder.mkdir()
        for row in selected:
            frame = dataset.frame(row['scene_id'], row['im_id'])
            if frame['source']['rgb_sha256'] != row['sha256'] or frame['source']['depth_sha256'] != row['depth_sha256']:
                raise ValueError('Selected source image/depth differs from the existing pinned input manifest')
            record = {'scene_id': row['scene_id'], 'im_id': row['im_id'], 'source': frame['source'],
                      'K': frame['K'].tolist(), 'depth_scale_mm': frame['depth_scale_mm'],
                      'sensor_geometry': _sensor_projection_check(frame), 'masks': []}
            label = f'{name} {row["scene_id"]}/{row["im_id"]}'
            thumbnails.append(_thumbnail(frame['rgb'], label + ' source'))
            target = chosen_targets.get((row['scene_id'], row['im_id']))
            if target:
                record['localization_target'] = {'obj_id': target.object_id, 'inst_count': target.instance_count}
                key = (target.scene_id, target.image_id, target.object_id)
                for confidence, index, detection in selected_masks[key]:
                    candidate = MaskCandidate(target.scene_id, target.image_id, target.object_id, confidence,
                                              decode_rle(detection['segmentation']), per_dataset_assets['masks']['segmentor'],
                                              'public_predicted:' + per_dataset_assets['masks']['sha256'], index,
                                              source_file_sha256=per_dataset_assets['masks']['sha256'])
                    entry = {'candidate_id': candidate.candidate_id, 'confidence': confidence,
                             'record_index': index, 'mask_pixels': int(candidate.mask.sum()), 'shape': list(candidate.mask.shape)}
                    if candidate.mask.shape != frame['depth_m'].shape:
                        raise ValueError('Public predicted-mask dimensions disagree with actual RGB-D')
                    try:
                        cloud = observed_clouds(frame['depth_m'], candidate.mask, frame['K'], seed=0)
                        projected, valid = project(cloud['sparse'], frame['K'])
                        error = float(np.abs(projected - cloud['pixel_uv']).max())
                        if not valid.all() or error > 1e-7:
                            raise ValueError('Masked sparse-point projection mapping failed')
                        entry.update(status='mapping_checked', sparse_points=len(cloud['sparse']), dense_points=len(cloud['dense']),
                                     sparse_projection_max_px=error, crop=cloud['crop'].__dict__)
                    except ValueError as exc:
                        entry.update(status='unusable_sensor_mask', reason=str(exc))
                    record['masks'].append(entry)
                    overlay = frame['rgb'].copy()
                    overlay[candidate.mask] = (0.55 * overlay[candidate.mask] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
                    thumbnails.append(_thumbnail(overlay, label + f' predicted obj{target.object_id}'))
            result['frames'].append(record)
        if render_cads and cameras is not None:
            from .rendering import RaycastRenderer, camera_poses
            object_id = min(map(int, dataset.models_info))
            model = dataset.model(object_id)
            renderer = RaycastRenderer(model['vertices'], model['triangles'], model['colors'], texture=model['texture'])
            K, poses = camera_poses(model['vertices'], resolution=render_size, occupancy=0.5, camera_to_model=cameras)
            points, _, _ = surface_sample(model['vertices'], model['triangles'], 256, 0)
            rendered_views = []
            for view in (0, 80):
                rotation, translation = poses[view]
                rendered = renderer.render(rotation, translation, K, height=render_size, width=render_size)
                uv, visible = visible_surface(points, None, rotation, translation, K, rendered['depth_m'], 0.005 * model['diameter_m'])
                pixels = np.floor(uv[visible] + 0.5).astype(int)
                depth = transform_points(points[visible], rotation, translation)[:, 2]
                residual = np.abs(depth - rendered['depth_m'][pixels[:, 1], pixels[:, 0]])
                rgb_file = folder / f'obj_{object_id:06d}_view_{view:03d}.png'
                Image.fromarray(rendered['rgb']).save(rgb_file)
                np.savez_compressed(rgb_file.with_suffix('.npz'), depth_m=rendered['depth_m'], R_m2c=rotation, t_m=translation, K=K)
                rendered_views.append({'view': view, 'foreground_pixels': int((rendered['depth_m'] > 0).sum()),
                                       'visible_diagnostic_points': int(visible.sum()),
                                       'visible_depth_residual_max_m': float(residual.max()) if len(residual) else None,
                                       'rgb_path': str(rgb_file), 'rgb_sha256': sha256(rgb_file)})
                thumbnails.append(_thumbnail(rendered['rgb'], f'{name} CAD{object_id} CNOS{view}'))
            result['CAD_render'] = {'object_id': object_id, 'native_units': model['native_units'], 'internal_units': model['internal_units'],
                                    'diameter_m': model['diameter_m'], 'bounds_m': [model['vertices'].min(0).tolist(), model['vertices'].max(0).tolist()],
                                    'appearance': model['appearance'], 'model_sha256': model['sha256'],
                                    'texture_sha256': model['texture']['sha256'] if model['texture'] else None,
                                    'views': rendered_views, 'scope': 'two low-resolution CPU diagnostic renders; not full CAD onboarding'}
        else:
            result['CAD_render'] = {'status': 'not_run', 'reason': 'CPU render disabled or canonical camera asset not supplied'}
        report['datasets'][name] = result
        write_json(output / 'progress.json', report)
    if thumbnails:
        sheet = Image.new('RGB', (256 * 4, 216 * math.ceil(len(thumbnails) / 4)), '#dddddd')
        for index, image in enumerate(thumbnails):
            sheet.paste(image, ((index % 4) * 256, (index // 4) * 216))
        sheet.save(output / 'input_mapping_contact_sheet.png')
        report['contact_sheet'] = str(output / 'input_mapping_contact_sheet.png')
    report.update(runtime_s=time.perf_counter() - started, disk_free_bytes_after=shutil.disk_usage(output).free,
                  input_integrity_passed=not report['texture_assets']['missing_textures'] and all(not value['missing_input_paths'] for value in report['datasets'].values()),
                  status='checks_completed', learned_inference_run=False, accuracy_metrics_computed=False)
    write_json(output / 'dataset_checks.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--assets-manifest')
    parser.add_argument('--cnos-poses')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--samples-per-dataset', type=int, default=2)
    parser.add_argument('--render-size', type=int, default=96)
    parser.add_argument('--no-render', action='store_true')
    args = parser.parse_args()
    report = run_dataset_checks(args.dataset_root, args.output, datasets=args.datasets,
                                samples_per_dataset=args.samples_per_dataset, asset_manifest=args.assets_manifest,
                                cnos_poses=args.cnos_poses, render_size=args.render_size, render_cads=not args.no_render)
    print(json.dumps({'status': report['status'], 'datasets': list(report['datasets']),
                      'report': str(local_path(args.output) / 'dataset_checks.json'),
                      'learned_inference_run': False, 'accuracy_metrics_computed': False}))


if __name__ == '__main__':
    main()
