"""CPU-only, no-GT visualization of an already selected original-source pose.

This helper projects every saved CAD query point. It does not estimate, score,
refine, select or verify poses, and does not load a neural network.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def font(size):
    p = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    return ImageFont.truetype(str(p), size) if p.is_file() else ImageFont.load_default()


def visualize(config_path, pose_path=None, output=None):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    paths = config['paths']
    own_root = Path(paths['output'])
    pose_path = Path(pose_path) if pose_path else own_root / 'first_execution/first_own_pose.json'
    output = Path(output) if output else pose_path.parent / 'first_pose_overlay.png'
    result = json.loads(pose_path.read_text())
    if result['selected_count'] != 1 or len(result['selected']) != 1:
        raise ValueError('This first-pose visualization expects one already selected pose')
    chosen = result['selected'][0]
    candidates = [c for c in result['candidates'] if c['candidate_index'] == chosen['candidate_index']]
    if len(candidates) != 1 or candidates[0]['status'] != 'valid':
        raise ValueError('Selected candidate provenance is inconsistent')
    candidate = candidates[0]
    ds, scene, image, obj = result['dataset'], int(result['scene_id']), int(result['im_id']), int(result['obj_id'])
    scene_root = Path(paths['bop_root']) / ds / 'test' / f'{scene:06d}'
    rgb_path, camera_path = scene_root / 'rgb' / f'{image:06d}.png', scene_root / 'scene_camera.json'
    mask_path = Path(candidate['mask_path'])
    query_path = Path(paths['features_dir']) / f'{ds}_obj_{obj:06d}_query.npz'
    geometry_path = Path(paths['public_source']) / 'freezev2/geometry.py'
    inputs = {'configuration': config_path, 'selected_pose_report': pose_path,
              'RGB': rgb_path, 'camera_metadata': camera_path, 'selected_predicted_mask': mask_path,
              'public_query_cache': query_path, 'original_projection_source': geometry_path}
    for p in inputs.values():
        if not p.is_file():
            raise FileNotFoundError(p)
    camera = json.loads(camera_path.read_text())
    info = camera.get(str(image), camera.get(f'{image:06d}'))
    K = np.asarray(info['cam_K'], dtype=np.float64).reshape(3, 3)
    with np.load(query_path, allow_pickle=False) as cached:
        points = np.array(cached['query_points'], dtype=np.float64, copy=True)
    R = np.asarray(chosen['R'], dtype=np.float64)
    t = np.asarray(chosen['t_mm'], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Saved query points must be finite Nx3')
    spec = importlib.util.spec_from_file_location('_original_public_projection', geometry_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    uv, z = module.project_points(points, K, R, t)
    rgb = Image.open(rgb_path).convert('RGB')
    mask = np.asarray(Image.open(mask_path).convert('L')) != 0
    width, height = rgb.size
    if mask.shape != (height, width):
        raise ValueError('Predicted mask and RGB dimensions differ')
    finite = np.isfinite(uv).all(axis=1) & np.isfinite(z) & (z > 0)
    in_frame = finite & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    eroded = np.asarray(Image.fromarray(mask.astype('uint8') * 255).filter(ImageFilter.MinFilter(3))) != 0
    boundary = mask & ~eroded
    outline = Image.fromarray(boundary.astype('uint8') * 255).filter(ImageFilter.MaxFilter(3))
    overlay = rgb.convert('RGBA')
    cyan = Image.new('RGBA', rgb.size, (0, 230, 255, 0))
    cyan.putalpha(outline.point(lambda v: int(v * 0.9)))
    overlay = Image.alpha_composite(overlay, cyan)
    dots = Image.new('RGBA', rgb.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(dots)
    for x, y in uv[in_frame]:
        draw.ellipse((float(x)-0.8, float(y)-0.8, float(x)+0.8, float(y)+0.8), fill=(255, 65, 45, 145))
    overlay = Image.alpha_composite(overlay, dots).convert('RGB')
    # Display-only ROI: union of selected mask and in-frame projected points.
    yy, xx = np.nonzero(mask)
    display_xy = np.vstack((np.column_stack((xx, yy)), uv[in_frame]))
    if len(display_xy) == 0:
        raise ValueError('No displayable mask or projected points')
    lower = np.floor(display_xy.min(axis=0) - 24).astype(int)
    upper = np.ceil(display_xy.max(axis=0) + 25).astype(int)
    x0, y0 = max(0, lower[0]), max(0, lower[1])
    x1, y1 = min(width, upper[0]), min(height, upper[1])
    crop = overlay.crop((x0, y0, x1, y1))
    crop.thumbnail((width, height), Image.Resampling.NEAREST)
    zoom = Image.new('RGB', (width, height), '#20242b')
    # Upscale only for inspection; no geometry is recomputed in zoom coordinates.
    scale = min(width/(x1-x0), height/(y1-y0))
    crop = overlay.crop((x0, y0, x1, y1)).resize((round((x1-x0)*scale), round((y1-y0)*scale)), Image.Resampling.NEAREST)
    zoom.paste(crop, ((width-crop.width)//2, (height-crop.height)//2))
    top, bottom, gap = 72, 100, 10
    canvas = Image.new('RGB', (3*width+2*gap, top+height+bottom), '#14181f')
    panel_x = [0, width+gap, 2*(width+gap)]
    for x, panel in zip(panel_x, (rgb, overlay, zoom)):
        canvas.paste(panel, (x, top))
    d = ImageDraw.Draw(canvas)
    d.text((12, 8), f'Own FreeZeV2-Re pose | {ds.upper()} scene {scene} / image {image} / object {obj}', font=font(22), fill='white')
    for x, title in zip(panel_x, ('Actual RGB input', 'Selected mask + projected CAD query points', f'Display zoom: x=[{x0},{x1}), y=[{y0},{y1})')):
        d.text((x+10, 44), title, font=font(16), fill='#dce1e8')
    y = top+height+12
    d.text((12, y), f'Cyan: predicted-mask outline   Red: all {len(points):,} CAD query points with the saved selected R,t; {int(in_frame.sum()):,} project inside the image.', font=font(18), fill='#dce1e8')
    d.text((12, y+27), f'Model -> OpenCV camera; millimetres. Candidate {chosen["candidate_index"]}; saved pose score {float(chosen["final_score"]):.6g}.', font=font(18), fill='#dce1e8')
    d.text((12, y+54), 'No GT was read. Projection includes hidden/backside CAD points; this is a visual diagnostic, not an accuracy measurement.', font=font(18), fill='#f4d37c')
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    code_copy = output.with_suffix('.py')
    if Path(__file__).resolve() != code_copy.resolve():
        shutil.copyfile(__file__, code_copy)
    receipt = {'schema': 'no-GT-own-selected-pose-projection-v1',
        'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'dataset': ds, 'scene_id': scene, 'im_id': image, 'obj_id': obj,
        'public_revision': config['public_revision'], 'method_id': config['method_id'],
        'prediction_origin': 'our original-public-source inference; not author prediction CSV',
        'GT_access': False, 'model_or_GPU_inference': False, 'pose_or_candidate_changes': False,
        'coordinate_frame': 'native BOP CAD millimetres -> OpenCV camera (+x right,+y down,+z forward)',
        'transform_convention': 'camera_points = CAD_points @ R.T + t_mm',
        'projection': 'unchanged public freezev2.geometry.project_points(points,K,R,t)',
        'K_native_pixels': K.tolist(), 'R_selected': R.tolist(), 't_selected_mm': t.tolist(),
        'selected_candidate_index': int(chosen['candidate_index']),
        'saved_final_score': float(chosen['final_score']),
        'query_count': len(points), 'positive_depth_finite_count': int(finite.sum()),
        'in_frame_projected_count': int(in_frame.sum()), 'mask_pixels': int(mask.sum()),
        'native_RGB_size_wh': [width, height], 'figure_size_wh': list(canvas.size),
        'display_zoom_xyxy': [int(v) for v in (x0,y0,x1,y1)],
        'display_zoom_scale_only': scale,
        'display_visibility': 'positive camera depth and image bounds only; no hidden-surface or measured-depth visibility culling',
        'claim_boundary': 'No GT comparison; rendered overlay does not establish pose accuracy.',
        'inputs': {k: {'path': str(p), 'sha256': sha256(p)} for k,p in inputs.items()},
        'visualization_code': {'path': str(code_copy), 'sha256': sha256(code_copy)},
        'output': {'path': str(output), 'sha256': sha256(output)},
        'visual_review': {'status': 'pending human/model image inspection'}}
    receipt_path = output.with_suffix('.json')
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'output': str(output), 'receipt': str(receipt_path), 'query_count': len(points),
                      'in_frame_projected_count': int(in_frame.sum()), 'GT_access': False}, indent=2))
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--pose-result')
    p.add_argument('--output')
    args = p.parse_args()
    visualize(args.config, args.pose_result, args.output)
