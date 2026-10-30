"""Isolated FreeZeV2 base runner. No evaluation annotations enter this module."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import platform
import re
import time

import numpy as np
from PIL import Image, ImageDraw

from .data import BOPDataset, sha256, write_json
from .features import FrozenDINOv2, FrozenGeDi, PCATransform, fuse_features, sample_feature_map
from .geometry import observed_clouds, project, transform_points, visible_surface
from .masks import load_localization_targets, load_mask_candidates, select_localization_candidates
from .registration import feature_ransac, refine_icp, score_pose, joint_score, rank_candidates
from .evaluation import PosePrediction, write_bop_csv
from .rendering import camera_poses, render_mesh, RaycastRenderer


def local_path(value):
    """One recorded Windows dataset/cache location, usable from the existing WSL."""
    text = str(value)
    if os.name != 'nt' and re.match(r'^[A-Za-z]:[\\/]', text):
        text = '/mnt/' + text[0].lower() + '/' + text[3:].replace('\\', '/')
    return Path(text).expanduser().resolve()


def load_config(path):
    config = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if config['method'] != 'FreeZeV2-base-paper-v1-independent':
        raise ValueError('This runner implements one explicitly named base configuration')
    # No accidental Accurate/SAR/learned pose estimator or training switches.
    if config['registration']['k'] != 10 or config['registration']['iterations'] != 10000:
        raise ValueError('Base configuration requires top10 and10000 hypotheses')
    if config['render']['views'] != 162 or config['render']['resolution'] != 480:
        raise ValueError('Base requires162 views at480x480')
    return config


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source_hashes():
    return {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def doctor(config):
    paths = {name: str(local_path(value)) for name, value in config['paths'].items()}
    issues = []
    for name in ('dataset_root', 'dino_source', 'dino_checkpoint', 'gedi_source', 'gedi_checkpoint', 'targets', 'cnos_poses'):
        if not Path(paths[name]).exists():
            issues.append({'component': name, 'reason': 'missing', 'path': paths[name]})
    for mask in config['masks']:
        if not local_path(mask['path']).is_file():
            issues.append({'component': 'predicted_masks', 'reason': 'missing', 'path': str(local_path(mask['path']))})
    modules = {m: importlib.util.find_spec(m) is not None for m in ('torch', 'open3d', 'pointnet2_ops', 'torchgeometry', 'cv2')}
    for name in ('torch', 'open3d', 'pointnet2_ops', 'torchgeometry', 'cv2'):
        if not modules[name]:
            issues.append({'component': name, 'reason': 'not importable in this interpreter'})
    cuda = None
    if modules['torch']:
        import torch
        cuda = {'available': torch.cuda.is_available(), 'torch': torch.__version__, 'cuda': torch.version.cuda}
        if cuda['available']:
            cuda['device'] = torch.cuda.get_device_name(0)
        else:
            issues.append({'component': 'cuda', 'reason': 'CUDA unavailable'})
    if not (local_path(config['paths']['dataset_root']) / config['dataset'] / 'models_eval').is_dir():
        evaluator_issues = ['Official models_eval is absent; inference models_cad is not silently aliased to evaluation models.']
    else:
        evaluator_issues = []
    return {'scope': 'dependency inventory, not inference or benchmark completion', 'python': platform.python_version(),
            'paths': paths, 'modules': modules, 'cuda': cuda, 'inference_blockers': issues,
            'evaluation_blockers': evaluator_issues, 'config_sha256': config_hash(config), 'source_sha256': source_hashes()}


def _object_dir(config):
    return local_path(config['paths']['output_root']) / 'onboarding' / config['dataset'] / f"obj_{config['object_id']:06d}"


def _geometry_signature(config, model):
    return {'config_sha256': config_hash(config), 'source_sha256': source_hashes(), 'model_sha256': model['sha256'],
            'camera_array_sha256': sha256(local_path(config['paths']['cnos_poses']))}


def _poisson_surface(model, count, seed):
    import open3d as o3d
    o3d.utility.random.seed(seed)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(model['vertices']), o3d.utility.Vector3iVector(model['triangles']))
    cloud = mesh.sample_points_poisson_disk(number_of_points=count, init_factor=5)
    # Rendering visibility is determined by projected depth; normals are not an extra gate.
    return np.asarray(cloud.points).copy()


def geometry_onboard(config):
    """Native-scale CAD rendering/visibility, independently executable without weights."""
    dataset = BOPDataset(local_path(config['paths']['dataset_root']), config['dataset'])
    model = dataset.model(config['object_id'])
    output = _object_dir(config)
    signature = _geometry_signature(config, model)
    manifest = output / 'geometry_manifest.json'
    if manifest.is_file():
        old = json.loads(manifest.read_text())
        if old['signature'] != signature:
            raise ValueError('Onboarding cache identity changed; select a separate versioned output_root')
        for relative, checksum in old['files'].items():
            if sha256(output / relative) != checksum:
                raise ValueError(f'Corrupt geometry cache: {relative}')
        return old
    if output.exists() and any(output.iterdir()):
        raise ValueError('Incomplete onboarding directory; preserve evidence and choose a new output_root')
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    points = _poisson_surface(model, config['cad']['initial_poisson_points'], config['seed'])
    cams = np.load(local_path(config['paths']['cnos_poses']), allow_pickle=False)
    if cams.shape != (162, 4, 4):
        raise ValueError('Verified CNOS camera-to-model array must have162 4x4 matrices')
    K, poses = camera_poses(model['vertices'], resolution=config['render']['resolution'], occupancy=config['render']['occupancy'], camera_to_model=cams)
    visibility = np.zeros((len(poses), len(points)), bool)
    renderer = RaycastRenderer(model['vertices'], model['triangles'], model['colors'],texture=model.get('texture'))
    files, view_audits, thumbs = {}, [], []
    for view, (R, t) in enumerate(poses):
        rendered = renderer.render(R, t, K, height=config['render']['resolution'], width=config['render']['resolution'])
        uv, visible = visible_surface(points, None, R, t, K, rendered['depth_m'],
                                       config['render']['visibility_tolerance_fraction'] * model['diameter_m'])
        height, width = rendered['depth_m'].shape
        visible &= (uv[:, 0] >= 0) & (uv[:, 0] <= width - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1)
        visibility[view] = visible
        rgb_path = output / 'views' / f'{view:03d}.png'
        rgb_path.parent.mkdir(exist_ok=True)
        Image.fromarray(rendered['rgb']).save(rgb_path)
        depth_path = rgb_path.with_suffix('.npz')
        np.savez_compressed(depth_path, depth_m=rendered['depth_m'].astype(np.float32), R_m2c=R, t_m=t, K=K)
        for p in (rgb_path, depth_path):
            files[str(p.relative_to(output))] = sha256(p)
        # Pixel-depth residual is a mapping check, not pose accuracy against GT.
        pixels = np.floor(uv[visible] + .5).astype(int)
        z = transform_points(points[visible], R, t)[:, 2]
        residuals = np.abs(z - rendered['depth_m'][pixels[:, 1], pixels[:, 0]])
        view_audits.append({'view': view, 'visible_points': int(visible.sum()),
                            'depth_residual_max_m': float(residuals.max()) if len(residuals) else None,
                            'foreground_pixels': int((rendered['depth_m'] > 0).sum())})
        if view in (0, 20, 40, 60, 80, 100, 120, 140, 161):
            im = Image.fromarray(rendered['rgb']).resize((240, 240))
            ImageDraw.Draw(im).text((5, 5), f'CNOS view {view}', fill='red')
            thumbs.append(im)
    eligible = np.flatnonzero(visibility.sum(axis=0) >= config['cad']['min_visible_views'])
    if len(eligible) < config['cad']['points']:
        raise ValueError(f'Only{len(eligible)} CAD points pass visibility, need{config["cad"]["points"]}; no threshold was relaxed')
    selected = np.random.default_rng(config['seed']).choice(eligible, config['cad']['points'], replace=False)
    np.savez_compressed(output / 'geometry.npz', points_m=points[selected], support_points_m=points,
                        visibility=visibility[:, selected], K=K, R=np.array([p[0] for p in poses]),
                        t_m=np.array([p[1] for p in poses]), selected_indices=selected)
    files['geometry.npz'] = sha256(output / 'geometry.npz')
    sheet = Image.new('RGB', (720, 720), 'white')
    for i, im in enumerate(thumbs):
        sheet.paste(im, ((i % 3) * 240, (i // 3) * 240))
    sheet.save(output / 'render_contact_sheet.png')
    files['render_contact_sheet.png'] = sha256(output / 'render_contact_sheet.png')
    result = {'status': 'geometry_and_visibility_complete', 'signature': signature, 'files': files,
              'CAD_frame': 'original native CAD axes, converted mm to metres; never recentered in cache',
              'pose_direction': 'CAD_to_OpenCV_camera', 'sampling': 'Open3D Poisson disk, initial oversampling then visible-point seeded selection',
              'native_vertices': len(model['vertices']), 'triangles': len(model['triangles']), 'initial_samples': len(points),
              'eligible_at_least18_views': len(eligible), 'retained_points': len(selected), 'views': view_audits,
              'renderer': 'Open3D native raycasting, float32 intersections; pinhole/photometry tested against software rasterizer',
              'runtime_s': time.perf_counter() - start, 'claim': 'Rendering/mapping validation only; no trained features or pose predictions.'}
    write_json(manifest, result)
    return result


def load_feature_backends(config):
    cfg, paths = config['features'], config['paths']
    # Check both weights before allocating giant model GPU memory.
    for key in ('dino_checkpoint', 'gedi_checkpoint'):
        if not local_path(paths[key]).is_file():
            raise FileNotFoundError(f'Required trained weights missing: {local_path(paths[key])}; no descriptor substitution')
    dino = FrozenDINOv2(local_path(paths['dino_source']), local_path(paths['dino_checkpoint']),
                        layer_index=cfg['dino_layer_index'], model_name='dinov2_vitg14', device='cuda',
                        expected_source_hashes=cfg['dino_expected_source_hashes'], source_revision_hint=cfg['dino_source_revision_hint'],
                        checkpoint_sha256=cfg.get('dino_checkpoint_sha256'))
    gedi = FrozenGeDi(local_path(paths['gedi_source']), local_path(paths['gedi_checkpoint']),
                      expected_source_sha256=cfg['gedi_source_sha256'], expected_revision=cfg.get('gedi_source_revision'),
                      checkpoint_sha256=cfg.get('gedi_checkpoint_sha256'), seed=config['seed'],
                      samples_per_batch=cfg.get('gedi_sampling_batch_size',500),
                      samples_per_patch_lrf=cfg.get('gedi_patch_lrf_points',4000),samples_per_patch_out=cfg.get('gedi_patch_network_points',512),
                      forward_microbatch_size=config.get('resources',{}).get('gedi_forward_microbatch_size',32))
    return dino, gedi


def feature_onboard(config):
    """Stream visible-view mean, CAD-only PCA and two-radius GeDi; never train."""
    dino, gedi = load_feature_backends(config)
    geometry = geometry_onboard(config)
    output = _object_dir(config)
    manifest = output / 'feature_manifest.json'
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old['geometry_signature'] != geometry['signature']:
            raise ValueError('Feature cache belongs to a different geometry configuration')
        if old['backends'] != {'dino': dino.provenance, 'gedi': gedi.provenance}:
            raise ValueError('Feature backend identity differs; use separate output_root')
        for name, checksum in old['files'].items():
            if sha256(output / name) != checksum:
                raise ValueError(f'Feature cache corrupt: {name}')
        return old
    data = np.load(output / 'geometry.npz', allow_pickle=False)
    points = data['points_m']
    total = None
    counts = np.zeros(len(points), np.int32)
    for view, visible in enumerate(data['visibility']):
        rgb = np.asarray(Image.open(output / 'views' / f'{view:03d}.png').convert('RGB'))
        fmap = dino.extract(rgb)
        if total is None:
            total = np.zeros((len(points), fmap.shape[-1]), np.float64)
        uv, _ = project(transform_points(points, data['R'][view], data['t_m'][view]), data['K'])
        total[visible] += sample_feature_map(fmap, uv[visible], image_size=rgb.shape[:2])
        counts[visible] += 1
    if np.any(counts < config['cad']['min_visible_views']):
        raise ValueError('Missing CAD feature views')
    visual = (total / counts[:, None]).astype(np.float32)
    pca = PCATransform(config['features']['pca_components']).fit(visual)
    geometric = gedi.compute(points, data['support_points_m'], BOPDataset(local_path(config['paths']['dataset_root']), config['dataset']).model(config['object_id'])['diameter_m'])
    fused = fuse_features(pca.transform(visual), geometric)
    pca.save(output / 'pca.npz')
    np.savez_compressed(output / 'features.npz', fused=fused, geometric=geometric, visual_mean=visual, view_counts=counts)
    result = {'status': 'frozen_features_complete', 'geometry_signature': geometry['signature'],
              'backends': {'dino': dino.provenance, 'gedi': gedi.provenance},
              'files': {name: sha256(output / name) for name in ('features.npz', 'pca.npz')},
              'PCA_fit_scope': 'this CAD only, no observed/test data', 'feature_shape': list(fused.shape)}
    write_json(manifest, result)
    return result


def _pose_dict(R, t):
    return {'R_m2c': np.asarray(R).tolist(), 't_m': np.asarray(t).tolist()}


def _draw_pose(rgb, points, K, coarse, refined, output):
    image = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(image)
    for pose, color in ((coarse, 'orange'), (refined, 'lime')):
        camera = transform_points(points[::max(1, len(points) // 500)], pose.R, pose.t)
        uv, _ = project(camera, K)
        for (u, v), z in zip(uv, camera[:, 2]):
            if z > 0 and np.isfinite([u, v]).all() and 0 <= u < image.width and 0 <= v < image.height:
                draw.ellipse((u - 1, v - 1, u + 1, v + 1), fill=color)
    draw.text((8, 8), 'orange: coarse; green: ICP; no GT', fill='red')
    image.save(output)


def infer(config, *, single_mask=False, allow_oracle=False):
    """Fixed-cohort predicted-mask localization. GT poses are never loaded."""
    scope = 'oracle_mask_diagnostic' if allow_oracle else ('single_mask_diagnostic' if single_mask else 'predicted_mask_localization')
    output = local_path(config['paths']['output_root']) / scope
    if output.exists():
        raise ValueError('Run outputs already exist; preserve them and select a separate versioned output_root')
    targets = load_localization_targets(local_path(config['paths']['targets']))
    keys = {(row['scene_id'], row['im_id'], config['object_id']) for row in config['cohort']}
    targets = [t for t in targets if (t.scene_id, t.image_id, t.object_id) in keys]
    if len(targets) != len(keys):
        raise ValueError('Fixed cohort does not exactly match official localization targets')
    candidates = []
    for entry in config['masks']:
        candidates.extend(load_mask_candidates(local_path(entry['path']), segmentor=entry['segmentor'], source_id=entry['source_id'],
                                              source_kind=entry.get('source_kind', 'predicted'), allow_oracle=allow_oracle, target_keys=keys))
    # The only oracle entry point is explicitly labeled and separately stored.
    if allow_oracle and (not candidates or not all(c.source_kind == 'oracle' for c in candidates)):
        raise ValueError('Oracle and predicted scopes may not be mixed or mislabeled')
    feature_onboard(config)
    dino, gedi = load_feature_backends(config)
    dataset = BOPDataset(local_path(config['paths']['dataset_root']), config['dataset'])
    model = dataset.model(config['object_id'])
    object_output = _object_dir(config)
    cad = np.load(object_output / 'geometry.npz', allow_pickle=False)['points_m']
    cad_features = np.load(object_output / 'features.npz', allow_pickle=False)['fused']
    pca = PCATransform.load(object_output / 'pca.npz')
    output.mkdir(parents=True)
    predictions, all_diagnostics = [], []
    for target in targets:
        start = time.perf_counter()
        frame = dataset.frame(target.scene_id, target.image_id)
        pool = select_localization_candidates(candidates, scene_id=target.scene_id, image_id=target.image_id,
                                               object_id=target.object_id, instance_count=target.instance_count, allow_oracle=allow_oracle)
        if single_mask:
            pool = pool[:1]
        poses, diagnostics = [], []
        for candidate in pool:
            candidate_start = time.perf_counter()
            folder = output / f'{target.scene_id:06d}_{target.image_id:06d}' / f'candidate_{candidate.record_index:06d}_{candidate.segmentor}'
            folder.mkdir(parents=True)
            Image.fromarray(candidate.mask.astype(np.uint8) * 255).save(folder / 'mask.png')
            trace = {'candidate': candidate.provenance(), 'source': frame['source'], 'status': 'started'}
            try:
                if candidate.mask.shape != frame['depth_m'].shape:
                    raise ValueError('Mask/depth dimensions differ')
                clouds = observed_clouds(frame['depth_m'], candidate.mask, frame['K'], grid=config['observed']['grid'],
                                          dense_count=config['observed']['dense_points'], seed=config['seed'])
                crop = clouds['crop'].extract_rgb(frame['rgb'])
                Image.fromarray(crop).save(folder / 'crop.png')
                visual = sample_feature_map(dino.extract(crop), clouds['feature_uv'], image_size=crop.shape[:2])
                geometric = gedi.compute(clouds['sparse'], clouds['dense'], model['diameter_m'])
                features = fuse_features(pca.transform(visual), geometric)
                coarse = feature_ransac(cad, clouds['sparse'], cad_features, features, model['diameter_m'], config['registration'])
                trace['registration'] = coarse.diagnostics
                if coarse.status != 'ok':
                    raise ValueError(f'Global registration unavailable: {coarse.status}')
                refined = refine_icp(cad, clouds['dense'], coarse.R, coarse.t, model['diameter_m'], config['registration'])
                scores = score_pose(cad, clouds['sparse'], refined.R, refined.t, model['diameter_m'], coarse.correspondences,
                                    dense_xyz=clouds['dense'], config=config['registration'])
                score = joint_score(coarse.score, scores.feature_score, scores.geometry_score)
                pose = dict(candidate_id=candidate.candidate_id, obj_id=target.object_id, R=refined.R, t=refined.t, score=score, status='ok')
                trace.update(status='ok', coarse_pose=_pose_dict(coarse.R, coarse.t), refined_pose=_pose_dict(refined.R, refined.t),
                             coarse_feature_score=coarse.score, refined_feature_score=scores.feature_score, geometry_score=scores.geometry_score,
                             joint_score=score, icp=refined.diagnostics, icp_status=refined.status,
                             crop=asdict(clouds['crop']), sparse_count=len(clouds['sparse']), dense_count=len(clouds['dense']))
                np.savez_compressed(folder / 'observed.npz', sparse_m=clouds['sparse'], dense_m=clouds['dense'], fused=features,
                                    correspondence_indices=coarse.correspondences.cad_indices, correspondence_cosines=coarse.correspondences.similarities)
                _draw_pose(frame['rgb'], cad, frame['K'], coarse, refined, folder / 'pose_overlay.png')
                poses.append(pose)
            except (ValueError, RuntimeError) as exc:
                trace.update(status='failed', failure_type=type(exc).__name__, reason=str(exc))
            trace['runtime_s'] = time.perf_counter() - candidate_start
            write_json(folder / 'diagnostic.json', trace)
            diagnostics.append(trace)
        ranking = rank_candidates(poses, translation_nms_threshold_m=config['candidate_ranking']['translation_nms_diameter_fraction'] * model['diameter_m'], max_instances=target.instance_count)
        elapsed = time.perf_counter() - start
        for pose in ranking.selected:
            predictions.append(PosePrediction(target.scene_id, target.image_id, target.object_id, pose['score'], pose['R'], pose['t'], elapsed))
        all_diagnostics.append({'target': asdict(target), 'candidate_count': len(pool), 'selected': [p['candidate_id'] for p in ranking.selected],
                                'suppressed': ranking.suppressed, 'rejected': ranking.rejected, 'runtime_s': elapsed, 'candidates': diagnostics,
                                'status': 'ok' if ranking.selected else 'no_valid_pose'})
        write_json(output / 'progress.json', all_diagnostics)
    result = {'scope': scope, 'method': config['method'], 'config_sha256': config_hash(config), 'source_sha256': source_hashes(),
              'target_sha256': sha256(local_path(config['paths']['targets'])), 'GT_pose_access': False,
              'predictions': len(predictions), 'targets': len(targets), 'images': all_diagnostics,
              'claim': 'Own inference validity only; published performance not established.'}
    if predictions:
        csv_path = output / f'freezev2base_{config["dataset"]}-test.csv'
        write_bop_csv(predictions, csv_path)
        result['bop_csv'] = str(csv_path)
    else:
        result['status'] = 'no_valid_predictions'
    write_json(output / 'run_manifest.json', result)
    return result
