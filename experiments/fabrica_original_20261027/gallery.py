"""Top-down CAD contact sheets for saved original Fabrica candidates.

This module only renders saved data. It does not generate, filter, rank or
execute grasps. One line is drawn per saved contact pair, not per hand angle.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from experiments.cad_grasp_geometry_20261026.gallery import local_path, read_stl


ROOT = Path(__file__).resolve().parents[2]
CAD_COLOR = '#cbd2d8'
SCENES = [f'N{i:02d}' for i in range(8)]
IMAGE_DPI = 120
PAIRS_PER_PAGE = 12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pair_array(record):
    """Keep the producer's complete contact-pair list, including its order."""
    pairs = np.asarray(record['contact_pairs_world_m'], dtype=float)
    if not pairs.size:
        return np.empty((0, 2, 3), dtype=float)
    if pairs.ndim != 3 or pairs.shape[1:] != (2, 3) or not np.isfinite(pairs).all():
        raise ValueError(f"Invalid saved contact pairs: {record['scene']} / {record['part']}")
    return pairs


def pose_count(record):
    return len(record.get('retained_poses', record.get('grasps', [])))


def validate_records(records):
    if not isinstance(records, list) or len(records) != 96:
        raise ValueError('Expected one result for each of 96 saved placements')
    scenes = sorted({str(r['scene']) for r in records})
    parts = sorted({str(r['part']) for r in records})
    if scenes != SCENES or len(parts) != 12:
        raise ValueError('Expected scenes N00-N07 and twelve CAD parts')
    keys = {(r['scene'], r['part']) for r in records}
    if len(keys) != 96:
        raise ValueError('Repeated or missing scene/part record')
    for record in records:
        pose = np.asarray(record['world_T_cad'], dtype=float)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError('Expected finite world_T_cad in metres')
        pairs = pair_array(record)
        if not isinstance(record.get('retained_poses', record.get('grasps', [])), list):
            raise ValueError('Saved grasps must be a list, including when unavailable')
        if record.get('retained_pair_count', len(pairs)) != len(pairs):
            raise ValueError('Saved retained-pair count does not match the plotted pairs')
        if record.get('retained_pose_count', pose_count(record)) != pose_count(record):
            raise ValueError('Saved retained-pose count does not match the pose records')
    return scenes, parts


def outcome_label(record):
    """Expose a failed/incomplete source status instead of calling it no solution."""
    status = str(record.get('status', 'status_not_recorded'))
    if 'timeout' in status.lower() or 'timed_out' in status.lower():
        return 'Timed out; not a finding of no grasps'
    if 'error' in status.lower() or 'unavailable' in status.lower():
        return status.replace('_', ' ')
    if not len(pair_array(record)):
        return 'No retained contact pairs'
    return ''


def render_sheets(records, meshes, scenes, parts, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': 'white',
                         'savefig.bbox': None})
    lookup = {(r['scene'], r['part']): r for r in records}
    images = []
    for scene in scenes:
        fig, axes = plt.subplots(3, 4, figsize=(16, 11), dpi=IMAGE_DPI,
                                 constrained_layout=True)
        tiles, overlay_data, pair_artists = [], [], []
        for ax, part in zip(axes.flat, parts):
            record = lookup[(scene, part)]
            vertices, faces = meshes[part]
            transform = np.asarray(record['world_T_cad'], dtype=float)
            world = vertices @ transform[:3, :3].T + transform[:3, 3]
            center = (world.min(axis=0) + world.max(axis=0)) / 2
            points_mm = (world - center) * 1000
            ax.add_collection(PolyCollection(points_mm[faces][:, :, :2],
                facecolors=CAD_COLOR, edgecolors='none', antialiaseds=False, zorder=1))

            pairs = pair_array(record)
            # Separate neighboring page colors; color does not encode quality.
            hues = (.07 + np.arange(max(len(pairs), 1)) * .6180339887498949) % 1.
            colors = plt.get_cmap('turbo')(hues)
            tile_pairs = []
            for pair, color in zip(pairs, colors):
                pair_mm = (pair - center) * 1000
                pair_artists.extend(ax.plot(pair_mm[:, 0], pair_mm[:, 1], color=color,
                                            linewidth=.8, zorder=4))
                pair_artists.append(ax.scatter(pair_mm[:, 0], pair_mm[:, 1], color=[color],
                                               s=8, edgecolors='none', zorder=5))
                tile_pairs.append((pair_mm[:, :2], color[:3]))
            radius = max(float(np.ptp(points_mm[:, :2], axis=0).max()) * .64, 2.)
            ax.set_xlim(-radius, radius)
            ax.set_ylim(-radius, radius)
            ax.set_aspect('equal')
            ax.set_axis_off()
            count = len(pairs)
            poses = pose_count(record)
            label = (part.replace('_', ' ') + '\n'
                     + f'{count} retained contact ' + ('pair' if count == 1 else 'pairs')
                     + f'; {poses} gripper ' + ('pose' if poses == 1 else 'poses'))
            outcome = outcome_label(record)
            if outcome:
                label += '\n' + outcome
            ax.set_title(label, fontsize=9.5, pad=8,
                         color='#b33d32' if not count or outcome else '#263846')
            tiles.append(dict(part=part, contact_pairs=count, gripper_poses=poses,
                              status=record.get('status'), note=outcome))
            overlay_data.append((ax, tile_pairs))
        fig.suptitle(scene + '\n'
            'Lines join retained surface-contact pairs; each pair can have several gripper orientations.\n'
            'Height and orientation are not shown. VLM selection and execution have not run.',
            fontsize=12)
        # Freeze the solved layout before saving either image or measuring data
        # coordinates. SVG uses the same pixel frame as these two saved PNGs.
        fig.canvas.draw()
        fig.set_layout_engine('none')
        pixel_width, pixel_height = fig.canvas.get_width_height()
        for tile, (ax, pairs) in zip(tiles, overlay_data):
            tile['pair_overlays'] = []
            for index, (xy, rgb) in enumerate(pairs, 1):
                pixels = ax.transData.transform(xy)
                pixels[:, 1] = pixel_height - pixels[:, 1]
                tile['pair_overlays'].append(dict(pair_number=index,
                    endpoints_px=np.round(pixels, 4).tolist(),
                    color_rgb=np.rint(np.asarray(rgb) * 255).astype(int).tolist()))
        path = output / f'{scene}.png'
        fig.savefig(path, dpi=IMAGE_DPI, facecolor='white')
        for artist in pair_artists:
            artist.set_visible(False)
        base_path = output / f'{scene}_base.png'
        fig.savefig(base_path, dpi=IMAGE_DPI, facecolor='white')
        plt.close(fig)
        images.append(dict(scene=scene, image=path.name, base_image=base_path.name,
            pixel_width=pixel_width, pixel_height=pixel_height, tiles=tiles,
            contact_pairs=sum(t['contact_pairs'] for t in tiles),
            gripper_poses=sum(t['gripper_poses'] for t in tiles)))
        print('CAD contact sheet: ' + scene, flush=True)
    return images


HTML = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fabrica CAD contact pairs</title><style>
*{box-sizing:border-box}body{margin:0;background:#fff;color:#263846;
font:16px system-ui,sans-serif}main{max-width:1700px;margin:auto;padding:20px}
h1{font-size:25px;margin:0 0 12px}p{line-height:1.5;margin:9px 0}
label{display:flex;align-items:center;gap:12px;font-weight:600;margin:12px 0}
.controls{display:flex;gap:28px;flex-wrap:wrap}
select{font:inherit;padding:7px 14px;background:white;border:1px solid #aab8c2;
border-radius:4px}.sheet{position:relative;margin:12px 0}
img{display:block;width:100%;height:auto;margin:0}
.sheet svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.note{font-size:14px;color:#526674}a{color:#17678c}
details{font-size:14px;margin:18px 0}summary{cursor:pointer}table{border-collapse:collapse;
width:100%;margin-top:12px}th,td{text-align:left;padding:8px;border-bottom:1px solid #dce3e8}
</style><main><h1>Fabrica CAD contact pairs</h1>
<p>Each colored line joins one retained surface-contact pair on the CAD model.
All retained pairs remain available. The page control shows twelve pairs per part
at a time in their saved order, or all pairs together. A pair can support several
gripper orientations; the two full counts are shown separately on each CAD tile.</p>
<p class="note">These are views from above. Finger height and orientation are not visible.
No VLM selection, pickup or assembly execution has run. The gallery does not rank,
shortlist or remove candidates. Twelve is a display page size, not a selected grasp count.</p>
<div class="controls"><label>Scene <select id="scene"></select></label>
<label>Contact pairs <select id="page"></select></label></div>
<p id="totals" class="note"></p><p id="range" class="note"></p>
<a id="full" href="#">Open archived image with all contact pairs</a>
<div class="sheet"><img id="sheet" alt="Twelve CAD silhouettes with colored surface-contact pairs">
<svg id="overlay" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"></svg></div>
<details><summary>Saved status for each part</summary><table>
<thead><tr><th>Part</th><th>Contact pairs</th><th>Visible pair numbers</th><th>Gripper poses</th><th>Status</th></tr></thead>
<tbody id="status"></tbody></table></details>
</main><script>const D=__DATA__;
const el=id=>document.getElementById(id),select=el('scene'),page=el('page');
const NS='http://www.w3.org/2000/svg';
for(const item of D.overviews){const option=document.createElement('option');
option.value=item.scene;option.textContent=item.scene;select.appendChild(option)}
function current(){return D.overviews.find(x=>x.scene===select.value)}
function shape(name,attrs){const item=document.createElementNS(NS,name);
for(const [key,value] of Object.entries(attrs)){item.setAttribute(key,String(value))}return item}
function showPage(){const r=current(),all=page.value==='all',start=all?0:Number(page.value)*D.pairs_per_page;
const stop=all?Infinity:start+D.pairs_per_page,svg=el('overlay');svg.replaceChildren();
el('range').textContent=all?'All retained pairs shown; no ranking or candidate filtering.':
'Display page '+(Number(page.value)+1)+': pair numbers '+(start+1)+'–'+stop+
' per part where available, in saved order. This is pagination only; all candidates remain available.';
el('status').replaceChildren();for(const item of r.tiles){const visible=item.pair_overlays.slice(start,stop);
for(const pair of visible){const [a,b]=pair.endpoints_px,color='rgb('+pair.color_rgb.join(',')+')';
svg.appendChild(shape('line',{x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke:color,'stroke-width':1.25}));
for(const p of [a,b]){svg.appendChild(shape('circle',{cx:p[0],cy:p[1],r:2.3,fill:color}))}}
const visibleRange=visible.length?visible[0].pair_number+'–'+visible[visible.length-1].pair_number:'None on this page';
const tr=document.createElement('tr');for(const value of [item.part.replaceAll('_',' '),item.contact_pairs,
visibleRange,item.gripper_poses,item.note||item.status||'Status not recorded']){
const td=document.createElement('td');td.textContent=value;tr.appendChild(td)}el('status').appendChild(tr)}}
function showScene(){const r=current(),image=el('sheet');image.src=r.base_image;
image.width=r.pixel_width;image.height=r.pixel_height;
image.alt=r.scene+': twelve CAD parts, with contact-pair lines on the selected display page';
el('overlay').setAttribute('viewBox','0 0 '+r.pixel_width+' '+r.pixel_height);
el('full').href=r.image;el('totals').textContent=r.contact_pairs+' contact pairs; '+
r.gripper_poses+' gripper poses in this scene. Counts do not measure pickup or assembly success.';
page.replaceChildren();const count=Math.max(1,...r.tiles.map(x=>Math.ceil(x.contact_pairs/D.pairs_per_page)));
for(let i=0;i<count;i++){const option=document.createElement('option');option.value=String(i);
option.textContent='Page '+(i+1)+' ('+(i*D.pairs_per_page+1)+'–'+((i+1)*D.pairs_per_page)+' per part)';
page.appendChild(option)}const option=document.createElement('option');option.value='all';
option.textContent='All contact pairs';page.appendChild(option);showPage()}
select.onchange=showScene;page.onchange=showPage;showScene();</script></html>'''


def main(run_dir):
    run_dir = local_path(run_dir)
    source = run_dir / 'results.json'
    before = sha(source)
    records = json.loads(source.read_text(encoding='utf-8'))
    scenes, parts = validate_records(records)
    output = run_dir / 'gallery'
    output.mkdir(parents=True, exist_ok=True)
    meshes, mesh_sources = {}, {}
    for part in parts:
        record = next(r for r in records if r['part'] == part)
        path = local_path(record['mesh_path'])
        meshes[part] = read_stl(path)
        mesh_sources[part] = dict(path=str(path), sha256=sha(path))
    overviews = render_sheets(records, meshes, scenes, parts, output)
    manifest = dict(placements=len(records), scenes=scenes, parts=parts,
        overviews=overviews, source_results=str(source), source_results_sha256=before,
        renderer_source_sha256=sha(__file__), native_meshes=mesh_sources,
        input_units='metres', plot_units='millimetres, centered per part for display only',
        contact_pairs=sum(len(pair_array(r)) for r in records),
        gripper_poses=sum(pose_count(r) for r in records),
        pairs_per_page=PAIRS_PER_PAGE, pagination_only=True, original_pair_order_preserved=True,
        every_contact_pair_reachable=True, exact_pixel_overlay=True,
        no_contact_pairs_removed=True, no_repeated_line_per_hand_angle=True,
        no_CAD_simplification=True, no_candidate_generation_or_ranking=True,
        no_VLM_calls=True, no_robot_execution=True, no_assembly_views=True)
    if sha(source) != before:
        raise ValueError('Saved results changed during rendering')
    data = json.dumps(manifest, ensure_ascii=True).replace('<', '\\u003c')
    (output / 'index.html').write_text(HTML.replace('__DATA__', data), encoding='utf-8')
    (output / 'gallery_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n',
                                                 encoding='utf-8')
    print('GALLERY=' + str(output / 'index.html'), flush=True)
    return output


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python -m experiments.fabrica_original_20261027.gallery RUN_DIRECTORY')
    main(sys.argv[1])
