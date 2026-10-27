"""Visualize frozen depth selections; never fit poses or rerank candidates."""
import hashlib
import html
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
REPO = STUDY.parents[1]
OUT = HERE / 'figures_v2'
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
_SCENES = {}
_SURFACES = {}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def local(value):
    value = str(value).replace('\\', '/')
    if len(value) > 1 and value[1] == ':' and sys.platform != 'win32':
        value = '/mnt/' + value[0].lower() + value[2:]
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def save_new(path, value):
    if Path(path).exists():
        assert read(path) == value, f'Refusing to overwrite figure evidence: {path}'
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def pretty(name):
    if name.startswith('RGOCG'):
        return 'Round pin ' + name.rsplit('_', 1)[-1]
    if name.startswith('KET'):
        return 'Rectangular pin ' + name.rsplit('_', 1)[-1]
    return name.replace('_Male', '').replace('_', ' ')


def validate_frozen():
    # This verifies every selection and fit before any post-hoc identity is read.
    from analyze_results import verify_before_labels
    # All raw-pose arithmetic was already checked by the completed analyzer;
    # hashes and candidate/ranking arithmetic are still checked afresh here.
    selections, complete = verify_before_labels(replay_raw=False)
    audit = read(HERE / 'analysis/completion.json')
    assert audit['selected_before_evaluation']
    assert audit['selections_sha256'] == complete['selections_sha256']
    for name, expected in audit['artifact_sha256'].items():
        assert sha(HERE / 'analysis' / name) == expected, name
    return selections, complete


def choose_cases(changes):
    fusion = [r for r in changes if r['method'] == 'contact_0mm_fusion']
    picked = [r for r in fusion if r['change'] == 'fixed'][:3]
    picked += [r for r in fusion if r['change'] in ('introduced', 'lost_to_tie_or_unavailable')][:3]
    picked += [r for r in changes if r['method'] == 'contact_0mm_geometry'
               and r['change'] == 'introduced'][:3]
    for key in [('N00', 'RGOCG8-50_8mm'), ('N04', 'KET8_Square_8mm')]:
        if any((r['state'], r['target']) == key for r in picked):
            continue
        candidates = [r for r in changes if (r['state'], r['target']) == key
                      and r['method'] == 'contact_0mm_geometry']
        if candidates:
            picked.append(candidates[0])
    omitted_truth = [r for r in changes if r['method'] == 'contact_0mm_geometry'
                     and not r['truth_in_top5']]
    picked += omitted_truth[:1]
    if not picked:
        picked = changes[:3]
    return picked


def scene(state, capture):
    if state not in _SCENES:
        source = capture['states'][state]
        _SCENES[state] = dict(rgb=np.asarray(Image.open(local(source['rgb_file'])).convert('RGB')),
            depth=np.load(local(source['depth_file'])), K=np.load(local(source['intrinsics_file'])),
            plane=np.asarray(source['plane_model_camera'], float),
            crops={c['proposal_id']: c for c in source['crops']})
    return _SCENES[state]


def observation(data, proposal_id):
    roi = data['crops'][proposal_id]['roi']
    x1, y1, x2, y2 = [roi[k] for k in ('x1', 'y1', 'x2', 'y2')]
    yy, xx = np.mgrid[y1:y2+1, x1:x2+1]
    z, K, plane = data['depth'][y1:y2+1, x1:x2+1], data['K'], data['plane']
    xyz = np.stack(((xx-K[0, 2])*z/K[0, 0], (yy-K[1, 2])*z/K[1, 1], z), axis=-1)
    distance = np.abs(xyz @ plane[:3] + plane[3]) / np.linalg.norm(plane[:3])
    valid = np.isfinite(z) & (z > 0) & (distance > .002)
    mask = np.zeros(data['depth'].shape, bool)
    mask[y1:y2+1, x1:x2+1] = valid
    return mask, xyz[valid]


def surface(target, catalog):
    if target not in _SURFACES:
        import open3d as o3d
        from scripts.surface_verification import CadSurface
        mesh_path = local(catalog[target]['mesh_file'])
        assert sha(mesh_path) == catalog[target]['mesh_sha256']
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        _SURFACES[target] = (CadSurface(mesh), mesh)
    return _SURFACES[target]


def frozen_pose(row, pid):
    candidate, = [c for c in row['candidate_scores'] if c['proposal_id'] == pid]
    if candidate['status'] != 'available' or 'pose_index' not in candidate:
        return candidate, None
    fitted = read(HERE / candidate['fit_file'])
    arm, = [a for a in fitted['geometry']['arms'] if a['arm_name'] == candidate['arm']]
    return candidate, arm['pose_bank'][candidate['pose_index']]


def case_figure(row, capture, catalog, identities):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import TwoSlopeNorm
    data = scene(row['state'], capture)
    raycaster, mesh = surface(row['target'], catalog)
    baseline = row['baseline_selected_ids']
    selected = row['selected_ids']
    ids = [('CLIP + DINO selected', baseline[0] if len(baseline) == 1 else None),
           ('Depth condition selected', selected[0] if len(selected) == 1 else None)]
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    fig.subplots_adjust(left=.04, right=.88, bottom=.20, top=.82, hspace=.65, wspace=.30)
    metadata, new_cloud = [], None
    color_image = None
    for index, (label, pid) in enumerate(ids):
        if pid is None:
            for axis in axes[index]:
                axis.axis('off')
            axes[index, 0].text(.5, .5, 'No unique available selection', ha='center', va='center')
            continue
        score, pose = frozen_pose(row, pid)
        mask, points = observation(data, pid)
        product = identities[row['state'], pid]
        item = {'role': label, 'proposal_id': pid, 'posthoc_identity': product,
                'candidate_score': score, 'frozen_pose': pose}
        metadata.append(item)
        if pose is None:
            for axis in axes[index]:
                axis.axis('off')
            axes[index, 0].text(.5, .5, 'No available projected pose', ha='center', va='center')
            continue
        T = np.asarray(pose['T_camera_from_cad'], float)
        yy, xx = np.nonzero(mask)
        bounds = int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1
        rendered = raycaster.render(T, data['K'], data['depth'].shape, bounds, pixel_offset=0.)
        assert rendered is not None and not rendered['clipped']
        x1, y1, x2, y2 = rendered['roi']
        rgb = data['rgb'][y1:y2, x1:x2]
        observed = mask[y1:y2, x1:x2]
        cad_depth = rendered['depth']
        cad_mask = np.isfinite(cad_depth) & (cad_depth > 0)
        depth = data['depth'][y1:y2, x1:x2]
        valid = observed & cad_mask & np.isfinite(depth) & (depth > 0)
        delta = np.full(depth.shape, np.nan)
        delta[valid] = (cad_depth[valid] - depth[valid]) * 1000
        axes[index, 0].imshow(rgb)
        short_label = 'CLIP + DINO' if index == 0 else 'Depth condition'
        axes[index, 0].set_title(f'{short_label}: crop {pid}\n{pretty(product)}', loc='left', fontsize=11)
        axes[index, 1].imshow(rgb)
        x_grid, y_grid = np.arange(-1, observed.shape[1]+1), np.arange(-1, observed.shape[0]+1)
        axes[index, 1].contour(x_grid, y_grid, np.pad(observed, 1), [.5],
                               colors=['#16ad8b'], linewidths=1.8)
        axes[index, 1].contour(x_grid, y_grid, np.pad(cad_mask, 1), [.5],
                               colors=['#366fd1'], linewidths=1.6)
        axes[index, 1].set_title(f'Observed / fitted CAD mask\nIoU {score.get("mask_iou", 0):.4f}', loc='left', fontsize=11)
        axes[index, 2].set_facecolor('#e7ecef')
        color_image = axes[index, 2].imshow(delta, cmap='coolwarm', norm=TwoSlopeNorm(vcenter=0, vmin=-5, vmax=5))
        axes[index, 2].set_title('Camera-Z residual (mm)\nCAD − observed on valid overlap', loc='left', fontsize=11)
        from geometry import table_frame
        normal, plane_offset, _ = table_frame(data['plane'], points)
        observed_heights = 1000 * (points @ normal + plane_offset)
        cad_heights = 1000 * ((raycaster.vertices @ T[:3, :3].T + T[:3, 3]) @ normal + plane_offset)
        p95, maximum = float(np.percentile(observed_heights, 95)), float(observed_heights.max())
        metrics = (f'DINO {score.get("dino_score", 0):.4f}   G {score.get("geometry_score", 0):.4f}\n'
                   f'Camera-Z MAE {1000*score.get("depth_mae_m", 0):.3f} mm\n'
                   f'Observed table height: p95 {p95:.2f}, max {maximum:.2f} mm')
        axes[index, 0].text(0, -.10, metrics, transform=axes[index, 0].transAxes, fontsize=9, va='top')
        tilt = float(np.rad2deg(np.arccos(np.clip(T[:3, 2] @ normal, -1, 1))))
        axes[index, 1].text(0, -.10, f'Fitted CAD max table height: {cad_heights.max():.2f} mm\nCAD +Z / table normal: {tilt:.2f}°\nSupport {pose["support_index"]}; yaw seed {pose["initial_yaw_deg"]}°; clearance {1000*pose["clearance_m"]:g} mm',
                            transform=axes[index, 1].transAxes, fontsize=9, va='top')
        item['CAD_z_to_table_normal_deg'] = tilt
        item['observed_table_height_p95_mm'] = p95
        item['observed_table_height_max_mm'] = maximum
        item['CAD_table_height_min_mm'] = float(cad_heights.min())
        item['CAD_table_height_max_mm'] = float(cad_heights.max())
        item['rendered_union_roi_xyxy'] = list(rendered['roi'])
        if index == 1:
            new_cloud = (points, T, data['plane'], mesh, product)
        for axis in axes[index]:
            axis.set_xticks([])
            axis.set_yticks([])
    method = 'DINO + depth fusion' if row['method'].endswith('fusion') else 'Geometry only'
    fig.suptitle(f'{row["state"]} · queried CAD: {pretty(row["target"])}\n{method} · {row["change"].replace("_", " ")}',
                 fontsize=19, x=.04, y=.96, ha='left')
    fig.legend(handles=[Line2D([0], [0], color='#16ad8b', lw=2, label='Observed foreground'),
                        Line2D([0], [0], color='#366fd1', lw=2, label='Projected CAD silhouette')],
               loc='lower left', bbox_to_anchor=(.04, .04), frameon=False, ncols=2, fontsize=10)
    fig.text(.04, .105, 'Both rows fit the same queried CAD to the indicated crop, using frozen poses.\nThe baseline crop selection and all experiment scores remain unchanged.', fontsize=10, color='#52626c')
    if color_image is not None:
        colorbar_axis = fig.add_axes([.91, .27, .013, .47])
        colorbar = fig.colorbar(color_image, cax=colorbar_axis)
        colorbar.set_label('Signed camera-Z error (mm); display clipped at ±5', fontsize=10)
    name = f'{row["state"]}__{row["target"]}__{row["method"]}'
    path = OUT / (name + '.png')
    fig.savefig(path, dpi=130, facecolor='white')
    plt.close(fig)
    return {'file': path.name, 'state': row['state'], 'target': row['target'],
            'method': row['method'], 'change': row['change'], 'panels': metadata}, new_cloud


def cloud_figure(cloud, target):
    import matplotlib.pyplot as plt
    from geometry import sample_mesh, table_frame
    points, T, plane, mesh, product = cloud
    normal, offset, basis = table_frame(plane, points)
    center = points.mean(0)
    origin = center - (center @ normal + offset) * normal
    transform = lambda p: np.column_stack(((p-origin) @ basis, p @ normal + offset)) * 1000
    observed = transform(points[np.linspace(0, len(points)-1, min(len(points), 3500), dtype=int)])
    cad = sample_mesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), count=6500)
    aligned = transform(cad @ T[:3, :3].T + T[:3, 3])
    from matplotlib.ticker import MaxNLocator
    fig = plt.figure(figsize=(15, 9))
    for i, (azimuth, elevation) in enumerate([(35, 25), (125, 18)]):
        axis = fig.add_axes([.025 + i*.50, .23, .45, .57], projection='3d')
        axis.scatter(*aligned.T, s=1.5, color='#4279d0', alpha=.25, label='Complete CAD at frozen pose')
        axis.scatter(*observed.T, s=2.5, color='#c44f31', alpha=.8, label='Single-camera observed surface')
        lo, hi = np.minimum(aligned.min(0), observed.min(0)), np.maximum(aligned.max(0), observed.max(0))
        xx, yy = np.meshgrid([lo[0]-3, hi[0]+3], [lo[1]-3, hi[1]+3])
        axis.plot_surface(xx, yy, np.zeros_like(xx), color='#aab8c0', alpha=.18, shade=False)
        axis.set_xlabel('Table X (mm)', fontsize=10, labelpad=9)
        axis.set_ylabel('Table Y (mm)', fontsize=10, labelpad=9)
        axis.set_zlabel('Height above table (mm)', fontsize=10, labelpad=7)
        axis.xaxis.set_major_locator(MaxNLocator(3))
        axis.yaxis.set_major_locator(MaxNLocator(3))
        axis.zaxis.set_major_locator(MaxNLocator(5))
        axis.tick_params(axis='both', labelsize=9, pad=2)
        axis.set_box_aspect(np.maximum(hi-lo, 3))
        axis.view_init(elevation, azimuth)
        if i == 0:
            handles, labels = axis.get_legend_handles_labels()
    fig.suptitle(f'Partial depth observation aligned with one supplied CAD\nQuery: {pretty(target)} · selected crop: {pretty(product)}', fontsize=19, x=.04, y=.96, ha='left')
    fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(.04, .105), frameon=False, ncols=2, fontsize=11)
    fig.text(.04, .045, 'Coordinates use the observed table plane, not simulator world poses. CAD dimensions remain unchanged.\nHidden CAD surfaces are shown for context; only the visible observation is measured.', fontsize=10, color='#52626c')
    path = OUT / 'partial_cloud_alignment.png'
    fig.savefig(path, dpi=140, facecolor='white')
    plt.close(fig)
    return path.name


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    _, complete = validate_frozen()
    # Labels are read only after the verification gate above.
    outcomes = read(HERE / 'analysis/per_task.json')
    identities = {(r['state'], r['truth_proposal_id']): r['target'] for r in outcomes
                  if r['method'] == 'clip_dino_baseline'}
    cases = choose_cases(read(HERE / 'analysis/paired_changes.json'))
    capture = read(STUDY / 'data/capture_manifest.json')
    catalog = read(STUDY / 'data/cad_inputs.json')
    OUT.mkdir(exist_ok=True)
    save_new(OUT / 'source_lock.json', {'selections_sha256': complete['selections_sha256'],
        'source_sha256': {p.name: sha(p) for p in [Path(__file__), HERE / 'analysis/paired_changes.json', HERE / 'analysis/per_task.json']},
        'scope': 'Render frozen poses only; no registration, new score, selection or model inference.'})
    rendered, cloud = [], None
    for row in cases:
        metadata, available_cloud = case_figure(row, capture, catalog, identities)
        rendered.append(metadata)
        if cloud is None and available_cloud is not None:
            cloud = available_cloud
            cloud_target = row['target']
    cloud_image = cloud_figure(cloud, cloud_target) if cloud is not None else None
    save_new(OUT / 'cases.json', {'cases': rendered, 'partial_cloud_image': cloud_image,
        'no_refitting_or_reranking': True})
    blocks = ['<!doctype html><html><head><meta charset="utf-8"><title>CAD-depth verification figures</title><style>body{font:16px/1.5 Segoe UI,Arial,sans-serif;color:#22323c;background:#f5f7f8}main{max-width:1500px;margin:auto;padding:30px}img{width:100%;background:white;margin:12px 0 28px}h1{font-size:32px}a{color:#236d8b}</style></head><body><main><h1>CAD alignment with single-camera observations</h1><p>All poses and selections were frozen before these figures were rendered. Green outlines are observed foreground; blue outlines are the camera-visible CAD silhouette. Depth errors are camera-Z differences in millimeters.</p><p><a href="../analysis/index.html">All methods, every changed selection and exact scores</a></p>']
    if cloud_image:
        blocks.append(f'<h2>Partial observed cloud and complete CAD</h2><img src="{cloud_image}" alt="Partial point cloud and aligned native CAD">')
    for row in rendered:
        blocks.append(f'<h2>{html.escape(row["state"] + " · " + pretty(row["target"]))}</h2><p>{html.escape(row["method"])} · {html.escape(row["change"])}</p><img src="{html.escape(row["file"])}" alt="Observed crop, CAD projection and depth residual">')
    blocks.append('<p>Raw transforms and exact component scores are in <a href="cases.json">cases.json</a>. Table contact is a modeling assumption; this is development simulation evidence.</p></main></body></html>')
    (OUT / 'index.html').write_text('\n'.join(blocks), encoding='utf-8')
    save_new(OUT / 'completion.json', {'cases': len(rendered), 'selection_sha256': complete['selections_sha256'],
        'no_refitting_or_reranking': True, 'artifact_sha256': {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'completion.json'}})
    print(json.dumps({'rendered_cases': len(rendered), 'partial_cloud_figure': cloud_image}), flush=True)


if __name__ == '__main__':
    main()
