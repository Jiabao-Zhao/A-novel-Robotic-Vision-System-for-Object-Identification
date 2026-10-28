"""Render saved CAD grasp candidates without generating or selecting any grasps.

Usage: python -m experiments.fabrica_candidates_20261027.gallery RUN_DIRECTORY
The only inputs are results.json, native CAD meshes, and the saved gripper meshes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import ConvexHull

from experiments.cad_grasp_geometry_20261026.gallery import local_path, read_stl


ROOT = Path(__file__).resolve().parents[2]
HAND_PATH = ROOT / 'outputs/cad_grasp_geometry_20261026/hand_model.json'
ROWS_PER_DETAIL_PAGE = 4
CAD_COLOR = '#cbd2d8'
FINGER_COLOR = '#477d9e'
PAD_COLOR = '#e6a247'
SAMPLE_COLOR = '#bc467b'
CONTACT_COLOR = '#147d55'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def points_array(value, name, shape):
    points = np.asarray(value, dtype=float)
    if points.shape != shape or not np.isfinite(points).all():
        raise ValueError(f'{name} must contain finite values with shape {shape}')
    return points


def actual_points(grasp):
    """Use modeled finite-pad witnesses, rather than relabeling sampled pairs."""
    if 'contact_points_world_m' in grasp:
        return points_array(grasp['contact_points_world_m'], 'contact_points_world_m', (2, 3))
    contacts = grasp.get('actual_contacts', [])
    if len(contacts) == 2 and all(c.get('point_world_m') is not None for c in contacts):
        return points_array([c['point_world_m'] for c in contacts], 'actual_contacts', (2, 3))
    return None


def closed_opening(grasp):
    """Return the modeled first-contact command, with its provenance for labeling."""
    if grasp.get('contact_opening_m') is not None:
        return float(grasp['contact_opening_m']), 'modeled first contact'
    openings = grasp.get('first_contact_openings_m', [])
    if isinstance(openings, dict):
        openings = list(openings.values())
    finite = [float(v) for v in openings if v is not None and np.isfinite(v)]
    if finite:
        return float(np.max(finite)), 'earlier modeled finger contact'
    contacts = grasp.get('actual_contacts', [])
    finite = [float(c['opening_m']) for c in contacts
              if c.get('opening_m') is not None and np.isfinite(c['opening_m'])]
    if finite:
        return float(np.max(finite)), 'earlier modeled finger contact'
    return float(grasp['opening_m']), 'saved open command; first-contact command unavailable'


def load_hand():
    """Read the exported finite collision components; no simulator is loaded."""
    hand = json.loads(HAND_PATH.read_text(encoding='utf-8'))
    components = []
    for saved in hand['components']:
        component = dict(saved)
        vertices = np.asarray(component['vertices_zero_m'], dtype=float)
        hull = ConvexHull(vertices)
        used = np.unique(hull.simplices)
        mapping = np.full(len(vertices), -1, dtype=int)
        mapping[used] = np.arange(len(used))
        component['vertices_zero_m'] = vertices[used]
        component['triangles'] = mapping[hull.simplices]
        component['translation_per_opening'] = np.asarray(component['translation_per_opening'])
        components.append(component)
    return components


def validate_records(records):
    if not isinstance(records, list) or len(records) != 96:
        raise ValueError('Expected a results.json list containing all 96 saved placements')
    scenes = sorted({str(r['scene']) for r in records})
    parts = sorted({str(r['part']) for r in records})
    if scenes != [f'N{i:02d}' for i in range(8)] or len(parts) != 12:
        raise ValueError('Expected scenes N00-N07 and twelve CAD parts')
    if len({(r['scene'], r['part']) for r in records}) != 96:
        raise ValueError('A scene/part placement is missing or repeated')
    for record in records:
        points_array(record['world_T_cad'], 'world_T_cad', (4, 4))
        for grasp in record['grasps']:
            if not grasp.get('valid', True):
                raise ValueError('A rejected candidate cannot be displayed as retained')
            points_array(grasp['world_T_grip'], 'world_T_grip', (4, 4))
            points_array(grasp['contacts_world_m'], 'contacts_world_m', (2, 3))
            actual_points(grasp)
            opening, _ = closed_opening(grasp)
            if not np.isfinite(opening):
                raise ValueError('The displayed gripper opening is nonfinite')
    return scenes, parts


def world_vertices(record, mesh):
    vertices, _ = mesh
    pose = np.asarray(record['world_T_cad'], dtype=float)
    return vertices @ pose[:3, :3].T + pose[:3, 3]


def colors_for(count, plt):
    return plt.get_cmap('turbo')(np.linspace(.07, .93, max(count, 1)))


def draw_cad(ax, vertices, faces, horizontal, vertical, PolyCollection):
    ax.add_collection(PolyCollection(vertices[faces][:, :, [horizontal, vertical]],
        facecolors=CAD_COLOR, edgecolors='none', antialiaseds=False, zorder=1))


def render_overviews(records, meshes, scenes, parts, output, plt, PolyCollection):
    lookup = {(r['scene'], r['part']): r for r in records}
    images = []
    for scene in scenes:
        fig, axes = plt.subplots(3, 4, figsize=(16, 11), constrained_layout=True)
        for ax, part in zip(axes.flat, parts):
            record = lookup[(scene, part)]
            vertices = world_vertices(record, meshes[part])
            faces = meshes[part][1]
            center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
            points = (vertices - center) * 1000
            draw_cad(ax, points, faces, 0, 1, PolyCollection)
            grasps = record['grasps']
            colors = colors_for(len(grasps), plt)
            for index, (grasp, color) in enumerate(zip(grasps, colors), start=1):
                pair = (np.asarray(grasp['contacts_world_m']) - center) * 1000
                ax.plot(pair[:, 0], pair[:, 1], color=color, lw=.85, zorder=4)
                ax.scatter(pair[:, 0], pair[:, 1], color=[color], s=9,
                           edgecolors='none', zorder=5)
                middle = pair.mean(axis=0)
                ax.annotate(str(index), middle[:2], xytext=(3, 3), textcoords='offset points',
                            fontsize=7, color=color, zorder=6,
                            bbox=dict(facecolor='white', edgecolor='none', alpha=.72, pad=.35))
            radius = max(float(np.ptp(points[:, :2], axis=0).max()) * .64, 2.)
            ax.set_xlim(-radius, radius)
            ax.set_ylim(-radius, radius)
            ax.set_aspect('equal')
            ax.set_axis_off()
            count = len(grasps)
            ax.set_title(part.replace('_', ' ') + f'\n{count} retained '
                         + ('group' if count == 1 else 'groups'), fontsize=10,
                         color='#b33d32' if not grasps else '#263846', pad=8)
        fig.suptitle(scene + ': retained CAD grasp groups, viewed from above\n'
                     'Each line joins a sampled contact pair. Numbers match the detailed poses; '
                     'finger depth is shown below.', fontsize=12)
        path = output / f'{scene}.png'
        fig.savefig(path, dpi=125, facecolor='white')
        plt.close(fig)
        images.append(dict(scene=scene, image=path.name, tiles=12,
                           retained_groups=sum(len(lookup[(scene, p)]['grasps']) for p in parts)))
    return images


def project_hull(ax, points, horizontal, vertical, color, filled, PolyCollection):
    projection = points[:, [horizontal, vertical]]
    if np.linalg.matrix_rank(projection - projection.mean(axis=0)) < 2:
        return
    hull = ConvexHull(projection)
    polygon = projection[hull.vertices]
    ax.add_collection(PolyCollection([polygon], facecolors=color if filled else 'none',
        edgecolors=color, linewidths=.8 if filled else .7, alpha=.85, zorder=3 if filled else 2))


def render_details(record, mesh, components, output, case_number, plt, PolyCollection):
    """Show every representative; page breaks are layout choices, never filters."""
    grasps = record['grasps']
    cad_world = world_vertices(record, mesh)
    faces = mesh[1]
    # The fixed palm is deliberately outside these close-ups. The two complete
    # finger bodies and their finite pads are the relevant saved hand components.
    fingers = [c for c in components if c['is_pad'] or 'hand_collision' not in c['name']]
    pages = []
    for start in range(0, len(grasps), ROWS_PER_DETAIL_PAGE):
        page_grasps = grasps[start:start + ROWS_PER_DETAIL_PAGE]
        fig, axes = plt.subplots(len(page_grasps), 2, squeeze=False,
                                 figsize=(12, 3.35 * len(page_grasps)), constrained_layout=True)
        page_items = []
        for offset, grasp in enumerate(page_grasps):
            index = start + offset + 1
            pose = np.asarray(grasp['world_T_grip'], dtype=float)
            # All detail coordinates are in the saved gripper frame; positive
            # local z is downward, so the side plot uses -z as height.
            local_cad = (cad_world - pose[:3, 3]) @ pose[:3, :3]
            local_cad[:, 2] *= -1
            local_cad *= 1000
            sampled = (np.asarray(grasp['contacts_world_m']) - pose[:3, 3]) @ pose[:3, :3]
            sampled[:, 2] *= -1
            sampled *= 1000
            witnesses = actual_points(grasp)
            if witnesses is not None:
                witnesses = (witnesses - pose[:3, 3]) @ pose[:3, :3]
                witnesses[:, 2] *= -1
                witnesses *= 1000
            opening, opening_label = closed_opening(grasp)
            hand_points = []
            for component in fingers:
                points = (component['vertices_zero_m']
                          + opening * component['translation_per_opening']) * 1000
                points = points.copy()
                points[:, 2] *= -1
                hand_points.append((component, points))
            bounds = np.vstack([sampled] + [p for _, p in hand_points])
            members = grasp.get('group_members', grasp.get('grouped_candidate_ids', [grasp['id']]))
            member_count = len(members) if isinstance(members, list) else int(members)
            depth = grasp.get('pad_depth_m')
            depth_label = f'{1000 * float(depth):.2f} mm' if depth is not None else 'unavailable'
            caption = (f"{index}. {grasp['id']}  |  {member_count} candidate(s) in group\n"
                       f'Gripper-height offset from sampled center: {depth_label}')
            for column, vertical in enumerate((1, 2)):
                ax = axes[offset, column]
                draw_cad(ax, local_cad, faces, 0, vertical, PolyCollection)
                for component, points in hand_points:
                    project_hull(ax, points, 0, vertical,
                                 PAD_COLOR if component['is_pad'] else FINGER_COLOR,
                                 bool(component['is_pad']), PolyCollection)
                ax.plot(sampled[:, 0], sampled[:, vertical], '--', color=SAMPLE_COLOR,
                        lw=1., zorder=5)
                ax.scatter(sampled[:, 0], sampled[:, vertical], s=14, facecolors='none',
                           edgecolors=SAMPLE_COLOR, linewidths=.8, zorder=6)
                if witnesses is not None:
                    ax.scatter(witnesses[:, 0], witnesses[:, vertical], color=CONTACT_COLOR,
                               s=20, edgecolors='white', linewidths=.3, zorder=7)
                lower = bounds[:, [0, vertical]].min(axis=0) - 6
                upper = bounds[:, [0, vertical]].max(axis=0) + 6
                # Keep a useful close-up even for a very narrow pin.
                span = np.maximum(upper - lower, [32., 28.])
                center = (lower + upper) / 2
                ax.set_xlim(center[0] - span[0] / 2, center[0] + span[0] / 2)
                ax.set_ylim(center[1] - span[1] / 2, center[1] + span[1] / 2)
                ax.set_aspect('equal', adjustable='box')
                ax.set_facecolor('white')
                ax.grid(color='#e9edf0', lw=.5, zorder=0)
                ax.tick_params(labelsize=7)
                ax.set_xlabel('Across fingers (mm)', fontsize=8)
                ax.set_ylabel('Front / back (mm)' if column == 0 else 'Height (mm)', fontsize=8)
                view = 'Grasp area from above' if column == 0 else 'Grasp area from the side'
                ax.set_title(view + '\n' + caption, fontsize=8.4)
            page_items.append(dict(number=index, id=grasp['id'],
                                   family_id=grasp.get('family_id'), members=member_count,
                                   pad_depth_m=depth, displayed_opening_m=opening,
                                   displayed_opening_source=opening_label))
        fig.suptitle(record['scene'] + ' / ' + record['part'].replace('_', ' ')
                     + '\nCAD rendering: blue finger bodies, orange pads, pink sampled pair, '
                     'green modeled contact points. No pickup execution.', fontsize=10)
        path = output / f'detail_{case_number:03d}_{len(pages) + 1:02d}.png'
        fig.savefig(path, dpi=125, facecolor='white')
        plt.close(fig)
        pages.append(dict(image=path.name, first=start + 1, last=start + len(page_grasps),
                          candidates=page_items))
    return pages


HTML = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fabrica candidate CAD views</title><style>
:root{font:15px system-ui;color:#263846;background:white}body{margin:0}
main{max-width:1600px;margin:20px auto;padding:0 20px}h1{font-size:23px;margin:0 0 8px}
p{line-height:1.5}.controls{display:flex;gap:22px;flex-wrap:wrap;margin:18px 0}
label{display:flex;align-items:center;gap:8px;font-weight:600}select{font:inherit;
padding:7px 12px;border:1px solid #aebbc5;border-radius:4px;background:white}
img{display:block;width:100%;height:auto;margin:10px 0 24px}h2{font-size:19px;margin:28px 0 7px}
.note{color:#526674;font-size:14px}.empty{padding:20px;background:#f3f5f6;border-radius:4px}
</style><main><h1>Retained CAD grasp groups</h1>
<p>Each scene sheet shows all retained representative contact pairs on the CAD models.
The detailed CAD views show every representative with its finger bodies and pads.
These are geometric candidates; VLM selection, pickup, and assembly have not run.</p>
<div class="controls"><label>Scene <select id="scene"></select></label></div>
<img id="sheet" alt="All twelve CAD parts and their retained contact pairs">
<h2>Finger placement for each part</h2><div class="controls">
<label>Part <select id="part"></select></label><label id="pageLabel">Poses <select id="page"></select></label>
</div><p id="status" class="note"></p><p id="empty" class="empty" hidden>No retained candidate for this placement.</p>
<img id="detail" alt="Top and side CAD views of the retained gripper poses">
<p class="note">The close-ups use the saved gripper frame, in millimeters. The fixed palm is omitted;
the complete finger bodies and pads are shown. Long CAD features can extend outside a close-up;
the scene sheet shows the whole part. Lines join the original sampled pair. Green dots show
independently modeled contact points when available. Page breaks do not remove poses.</p>
<p class="note">The height offset is the saved gripper reference height minus the sampled
contact-pair center height. It describes finger placement, not measured contact depth.</p>
</main><script>const D=__DATA__;
const el=id=>document.getElementById(id),scene=el('scene'),part=el('part'),page=el('page');
function options(node,items){node.replaceChildren();for(const [value,label] of items){
const o=document.createElement('option');o.value=value;o.textContent=label;node.appendChild(o)}}
options(scene,D.scenes.map(x=>[x,x]));options(part,D.parts.map(x=>[x,x.replaceAll('_',' ')]));
function record(){return D.cases.find(x=>x.scene===scene.value&&x.part===part.value)}
function showDetail(){const r=record(),p=r.pages[Number(page.value)||0],image=el('detail');
image.hidden=!p;el('empty').hidden=Boolean(p);if(p){image.src=p.image;
image.alt=r.scene+' '+r.part+': poses '+p.first+' to '+p.last+', top and side CAD renders'}
el('status').textContent=r.retained_groups+' retained group(s); every representative is shown. '+
'The numbers on the scene sheet match the detailed pose numbers.';}
function showPart(){const r=record();options(page,r.pages.map((p,i)=>[String(i),p.first+'–'+p.last]));
el('pageLabel').hidden=r.pages.length<2;showDetail()}
function showScene(){el('sheet').src=D.overviews.find(x=>x.scene===scene.value).image;
el('sheet').alt=scene.value+': all twelve CAD parts with retained sampled contact pairs';showPart()}
scene.onchange=showScene;part.onchange=showPart;page.onchange=showDetail;showScene();
</script></html>'''


def main(run_dir):
    run_dir = local_path(run_dir)
    source = run_dir / 'results.json'
    source_sha = sha(source)
    records = json.loads(source.read_text(encoding='utf-8'))
    scenes, parts = validate_records(records)
    output = run_dir / 'gallery'
    output.mkdir(parents=True, exist_ok=True)
    meshes = {}
    mesh_hashes = {}
    for part in parts:
        record = next(r for r in records if r['part'] == part)
        path = local_path(record['mesh_path'])
        meshes[part] = read_stl(path)
        mesh_hashes[part] = dict(path=str(path), sha256=sha(path))
    components = load_hand()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': 'white'})
    overviews = render_overviews(records, meshes, scenes, parts, output, plt, PolyCollection)
    cases = []
    for number, record in enumerate(sorted(records, key=lambda r: (r['scene'], r['part']))):
        pages = render_details(record, meshes[record['part']], components, output,
                               number, plt, PolyCollection)
        displayed = sum(len(p['candidates']) for p in pages)
        if displayed != len(record['grasps']):
            raise ValueError('The detailed gallery omitted a retained representative')
        cases.append(dict(scene=record['scene'], part=record['part'],
                          retained_groups=len(record['grasps']),
                          all_valid_candidates=len(record.get('all_valid_grasps', [])), pages=pages))
        print(f"CAD views {number + 1}/{len(records)}: {record['scene']} / {record['part']}", flush=True)
    manifest = dict(scenes=scenes, parts=parts, placements=len(records),
                    retained_groups=sum(len(r['grasps']) for r in records),
                    overviews=overviews, cases=cases, source_results=str(source),
                    source_results_sha256=source_sha, source_sha256=sha(__file__),
                    hand_mesh_sha256=sha(HAND_PATH), native_meshes=mesh_hashes,
                    native_CAD_scale='meters; all display coordinates are millimeters',
                    no_CAD_simplification=True, no_representatives_omitted=True,
                    no_candidate_generation=True, no_VLM_calls=True,
                    no_pickup_or_assembly_execution=True, no_simulator_renderer=True)
    data = json.dumps(manifest, ensure_ascii=True).replace('<', '\\u003c')
    (output / 'index.html').write_text(HTML.replace('__DATA__', data), encoding='utf-8')
    if sha(source) != source_sha:
        raise ValueError('Input results changed while the gallery was rendered')
    (output / 'gallery_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print('GALLERY=' + str(output / 'index.html'), flush=True)
    return output


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python -m experiments.fabrica_candidates_20261027.gallery RUN_DIRECTORY')
    main(sys.argv[1])
