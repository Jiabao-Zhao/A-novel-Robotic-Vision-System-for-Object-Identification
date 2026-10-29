"""Render saved selected grasps on native CAD, with optional pickup side views.

This module consumes results.json. It never generates, filters, scores or
executes a grasp, and never renders a final assembly configuration.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
import textwrap

import numpy as np

from experiments.cad_grasp_selection_20261028 import gallery as old
from experiments.fabrica_original_20261027 import gallery as source


ROOT = source.ROOT
DEFAULT_HAND = ROOT / 'outputs/cad_grasp_geometry_20261026/hand_model.json'
CAD_COLOR = source.CAD_COLOR
CONTACT_COLOR = '#285bc2'
CAPTION = 'Geometry-selected grasps. VLM selection is excluded; pickup and assembly remain paused.'
selected_pose = old.selected_pose
validate_records = old.validate_records


def transform_points(points, transform):
    transform = np.asarray(transform, dtype=float)
    return np.asarray(points, dtype=float) @ transform[:3, :3].T + transform[:3, 3]


def failure_reason(record):
    """Translate the most frequent rejection into ordinary language for a tile."""
    counts = record.get('rejection_counts', record.get('failure_counts', {}))
    reason = max(counts, key=lambda name: (counts[name], name)) if counts else ''
    translations = (
        ('Gripper solid intersects target', 'A finger or the palm would enter the part'),
        ('Complete gripper intersects receiver', 'The gripper would touch the receiving part'),
        ('Complete gripper violates table', 'A finger or the palm would touch the table'),
        ('Intended pad-center surface normal', 'The finger pads do not face the part surfaces squarely'),
        ('Surrounding contact geometry', 'The part surface does not sit squarely against a pad'),
        ('No aligned inner-pad support', 'A pad has no suitably aligned contact surface'),
        ('No modeled inner-pad contact', 'A pad has no modeled contact with the part'),
        ('Part contact occurs outside', 'Contact would occur outside the usable pad face'),
        ('Intended pad center violates', 'The contact is too close to a pad edge'),
        ('Intended pad center does not contact', 'A contact point misses the CAD surface'),
        ('Inner-pad contact occupies', 'A pad would cover a surface needed for assembly'),
        ('Protected mating-surface information', 'The surfaces needed for assembly are not defined'),
        ('Holding or pre-opening outside', 'The required finger opening is outside the gripper limits'),
    )
    for prefix, plain in translations:
        if reason.startswith(prefix):
            return plain
    message = record.get('selection', {}).get('message', '')
    return str(message or reason or 'No saved valid grasp is available').replace('_', ' ')


def outcome(record):
    selection = record['selection']
    if selection.get('status') == 'selected':
        return 'Selected grasp ' + str(selection['selected_grasp']['candidate_id'])
    status = str(selection.get('status', 'unavailable'))
    if 'timeout' in status or 'timed_out' in status:
        return 'The search timed out'
    if 'unavailable' in status:
        return 'Required input unavailable'
    return 'No grasp passed the geometry checks'


def tile_metadata(record, pair_world=None):
    selection = record['selection']
    pose = selected_pose(record)
    result = dict(part=record['part'], status=selection.get('status'),
        outcome=outcome(record), reason='' if pose is not None else failure_reason(record),
        selected_count=int(pose is not None),
        proposed_count=selection.get('proposed_count', 0),
        valid_count=selection.get('valid_count', 0),
        rejection_counts=record.get('rejection_counts', record.get('failure_counts', {})),
        wall_seconds=record.get('wall_seconds'))
    if pose is not None:
        result.update(candidate_id=pose['candidate_id'], criteria=pose.get('criteria', {}),
            holding_opening_m=pose['holding_opening_m'],
            contacts_cad_m=pose['contacts_cad_m'], contacts_world_m=pair_world.tolist(),
            cad_T_hand=pose['cad_T_hand'], cad_T_grip_site=pose.get('cad_T_grip_site'),
            world_T_hand_pickup=pose['world_T_hand_pickup'])
    return result


def _title(ax, record):
    selected = selected_pose(record)
    label = record['part'].replace('_', ' ') + '\n' + outcome(record)
    if selected is None:
        label += '\n' + '\n'.join(textwrap.wrap(failure_reason(record), width=38)[:3])
    ax.set_title(label, fontsize=8.8, pad=8,
                 color='#263846' if selected is not None else '#b33d32')
    ax.set_aspect('equal')
    ax.set_axis_off()


def render_sheets(records, meshes, scenes, parts, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    output = Path(output)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': 'white',
                         'savefig.bbox': None})
    lookup = {(row['scene'], row['part']): row for row in records}
    sheets = []
    for scene in scenes:
        fig, axes = plt.subplots(3, 4, figsize=(16, 11), dpi=source.IMAGE_DPI,
                                 constrained_layout=True)
        tiles, overlays = [], []
        for ax, part in zip(axes.flat, parts):
            row = lookup[(scene, part)]
            pose = selected_pose(row)
            pair_world, pair_xy = None, None
            if row.get('world_T_cad') is not None:
                vertices, faces = meshes[part]
                world = transform_points(vertices, row['world_T_cad'])
                center = (world.min(axis=0) + world.max(axis=0)) / 2
                points_mm = (world-center)*1000
                ax.add_collection(PolyCollection(points_mm[faces, :2],
                    facecolors=CAD_COLOR, edgecolors='none', antialiaseds=False, zorder=1))
                if pose is not None:
                    pair_world = transform_points(pose['contacts_cad_m'], row['world_T_cad'])
                    pair_xy = ((pair_world-center)*1000)[:, :2]
                    ax.plot(pair_xy[:, 0], pair_xy[:, 1], color=CONTACT_COLOR,
                            linewidth=1.25, zorder=4)
                    ax.scatter(pair_xy[:, 0], pair_xy[:, 1], color=CONTACT_COLOR,
                               s=14, edgecolors='none', zorder=5)
                radius = max(float(np.ptp(points_mm[:, :2], axis=0).max())*.64, 2.)
                ax.set_xlim(-radius, radius)
                ax.set_ylim(-radius, radius)
            else:
                ax.text(.5, .5, 'Observed CAD pose unavailable', transform=ax.transAxes,
                        ha='center', va='center', fontsize=9, color='#b33d32')
            _title(ax, row)
            tiles.append(tile_metadata(row, pair_world))
            overlays.append((ax, pair_xy))
        fig.suptitle(scene + '\n' + CAPTION + '\n'
            'View from above: each line joins the two selected pad-center contacts. Contact height is not visible in this view.',
            fontsize=11)
        fig.canvas.draw()
        fig.set_layout_engine('none')
        width, height = fig.canvas.get_width_height()
        for tile, (ax, xy) in zip(tiles, overlays):
            tile['selected_contact_overlay_px'] = None
            if xy is not None:
                pixels = ax.transData.transform(xy)
                pixels[:, 1] = height-pixels[:, 1]
                tile['selected_contact_overlay_px'] = np.round(pixels, 4).tolist()
        path = output / f'{scene}.png'
        fig.savefig(path, dpi=source.IMAGE_DPI, facecolor='white')
        if scene == scenes[0]:
            fig.savefig(output/'N00_preview.png', dpi=source.IMAGE_DPI, facecolor='white')
        plt.close(fig)
        sheets.append(dict(scene=scene, image=path.name, tiles=tiles,
            pixel_width=width, pixel_height=height,
            selected_count=sum(tile['selected_count'] for tile in tiles)))
        print('Selected CAD contact sheet: ' + scene, flush=True)
    return sheets


def pickup_geometry(record, vertices, hand):
    """Place every saved collision component using the grip-site frame only.

    cad_T_hand may refer to the fingertip in this study. Treating it as the
    original exported grip site would shift all geometry, so no fallback to it
    is permitted. The vertical coordinate is signed distance above the table.
    """
    from scipy.spatial import ConvexHull

    pose = selected_pose(record)
    if pose is None:
        return None
    grip_site = np.asarray(pose.get('cad_T_grip_site'), dtype=float)
    if grip_site.shape != (4, 4) or not np.isfinite(grip_site).all():
        raise ValueError('Pickup side view requires the saved original cad_T_grip_site')
    table = np.asarray(record.get('table_plane_world'), dtype=float)
    if table.shape != (4,) or not np.isfinite(table).all() or np.linalg.norm(table[:3]) <= 0:
        raise ValueError('Pickup side view requires the saved table plane')
    table = table/np.linalg.norm(table[:3])
    pickup = np.asarray(record['world_T_cad'], dtype=float)
    world_T_grip_site = pickup @ grip_site
    part_world = transform_points(vertices, pickup)
    center = (part_world.min(axis=0)+part_world.max(axis=0))/2
    horizontal = world_T_grip_site[:3, 0].copy()
    horizontal -= table[:3]*float(horizontal@table[:3])
    if np.linalg.norm(horizontal) < 1e-9:
        reference = np.eye(3)[np.argmin(np.abs(table[:3]))]
        horizontal = reference-table[:3]*float(reference@table[:3])
    horizontal /= np.linalg.norm(horizontal)

    def project(points):
        points = np.asarray(points, dtype=float)
        return np.column_stack(((points-center)@horizontal,
                                points@table[:3]+table[3]))*1000

    components = []
    for component in hand['components']:
        local = (np.asarray(component['vertices_zero_m'], dtype=float)
                 + float(pose['holding_opening_m'])*np.asarray(component['translation_per_opening']))
        world = transform_points(local, world_T_grip_site)
        components.append(dict(name=component['name'], is_pad=component['is_pad'],
            vertices_world_m=world, projected_mm=project(world),
            faces=ConvexHull(local).simplices))
    contacts_world = transform_points(pose['contacts_cad_m'], pickup)
    fingertip_world = (pickup@np.asarray(pose['cad_T_hand'], dtype=float))[:3, 3]
    return dict(part_world_m=part_world, part_projected_mm=project(part_world),
        components=components, contacts_projected_mm=project(contacts_world),
        fingertip_projected_mm=project(fingertip_world[None, :])[0],
        horizontal_world=horizontal, vertical_world=table[:3],
        world_T_grip_site=world_T_grip_site,
        table_minimum_clearance_m=min(float(np.min(c['vertices_world_m']@table[:3]+table[3]))
                                      for c in components))


def render_side_sheets(records, meshes, scenes, parts, output, hand_path=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    output = Path(output)
    lookup = {(row['scene'], row['part']): row for row in records}
    hands, hand_sources, sheets = {}, {}, []
    for scene in scenes:
        fig, axes = plt.subplots(3, 4, figsize=(16, 11), dpi=source.IMAGE_DPI,
                                 constrained_layout=True)
        tiles = []
        for ax, part in zip(axes.flat, parts):
            row, tile = lookup[(scene, part)], dict(part=part, rendered=False)
            pose = selected_pose(row)
            if pose is not None:
                path = source.local_path(row.get('hand_path') or hand_path or DEFAULT_HAND)
                if path not in hands:
                    hands[path] = json.loads(path.read_text(encoding='utf-8'))
                    hand_sources[str(path)] = source.sha(path)
                vertices, faces = meshes[part]
                geom = pickup_geometry(row, vertices, hands[path])
                for component in geom['components']:
                    xy = component['projected_mm']
                    ax.add_collection(PolyCollection(xy[component['faces']],
                        facecolors='#e1a23c' if component['is_pad'] else '#82a6c2',
                        edgecolors='none', alpha=.7, antialiaseds=False, zorder=1))
                xy = geom['part_projected_mm']
                ax.add_collection(PolyCollection(xy[faces], facecolors=CAD_COLOR,
                    edgecolors='none', antialiaseds=False, zorder=2))
                contacts = geom['contacts_projected_mm']
                ax.scatter(contacts[:, 0], contacts[:, 1], s=12,
                           c=CONTACT_COLOR, zorder=4, edgecolors='none')
                tip = geom['fingertip_projected_mm']
                ax.scatter([tip[0]], [tip[1]], marker='+', s=32,
                           c='#b54724', linewidths=1, zorder=5)
                all_xy = np.concatenate([xy]+[c['projected_mm'] for c in geom['components']])
                low, high = all_xy.min(axis=0), all_xy.max(axis=0)
                span = np.maximum(high-low, 5.)
                ax.set_xlim(low[0]-.12*span[0], high[0]+.12*span[0])
                ax.set_ylim(min(-2., low[1]-.05*span[1]), high[1]+.1*span[1])
                ax.axhline(0., color='#827365', linewidth=1., zorder=0)
                ax.text(.02, .015, 'Table', transform=ax.transAxes, fontsize=7, color='#827365')
                tile.update(rendered=True, hand_source=str(path),
                    component_names=[c['name'] for c in geom['components']],
                    components_rendered=len(geom['components']),
                    world_T_grip_site=geom['world_T_grip_site'].tolist(),
                    horizontal_world=geom['horizontal_world'].tolist(),
                    vertical_world=geom['vertical_world'].tolist(),
                    table_minimum_clearance_m=geom['table_minimum_clearance_m'],
                    fingertip_projected_mm=tip.tolist())
            else:
                ax.text(.5, .5, 'No selected pickup pose', transform=ax.transAxes,
                        ha='center', va='center', fontsize=9, color='#b33d32')
            _title(ax, row)
            tiles.append(tile)
        fig.suptitle(scene + '\nPickup side views: palm and fingers in blue, pads in gold, CAD in gray.\n'
            'The red cross marks the fingertip reference. Geometry is shown at the selected holding opening; the line is the table.',
            fontsize=11)
        fig.canvas.draw()
        fig.set_layout_engine('none')
        path = output/f'{scene}_pickup_side.png'
        fig.savefig(path, dpi=source.IMAGE_DPI, facecolor='white')
        plt.close(fig)
        sheets.append(dict(scene=scene, image=path.name, tiles=tiles,
                           rendered_count=sum(tile['rendered'] for tile in tiles)))
        print('Selected pickup side sheet: '+scene, flush=True)
    return sheets, hand_sources


HTML = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Selected NIST CAD grasps</title><style>
body{margin:0;background:white;color:#263846;font:15px system-ui,sans-serif}
main{max-width:1600px;margin:auto;padding:22px}h1{font-size:23px;margin:0 0 10px}
p{line-height:1.45;max-width:1100px}.note{color:#596875}select{font:inherit;padding:5px 12px}
img{display:block;width:100%;height:auto;margin:16px 0}a{color:#285bc2}
.controls{display:flex;gap:20px;align-items:center;flex-wrap:wrap}details{margin:18px 0}
summary{cursor:pointer}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;vertical-align:top;padding:8px;border-bottom:1px solid #dde3e8}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}.failed{color:#b33d32}
</style><main><h1>Selected grasp on each CAD model</h1>
<p>Each blue line joins the two selected finger-pad contact points on the gray CAD model.
The part retains its saved observed pickup orientation. A red label explains why no grasp was selected.</p>
<p class="note">The pipeline chooses the grasp using geometry. VLM selection is excluded.
Pickup and assembly execution remain paused. These pictures do not establish pickup or assembly success.</p>
<div class="controls"><label>Scene <select id="scene"></select></label>
<label>View <select id="view"><option value="top">Contacts from above</option></select></label>
<a id="full" href="#">Open this image</a><span id="totals"></span></div>
<img id="sheet" alt="Selected contacts on twelve NIST CAD models">
<p id="viewnote" class="note"></p>
<details><summary>Saved selection details and reasons</summary><table><thead>
<tr><th>Part</th><th>Result</th><th>Valid / proposed</th><th>Details</th></tr>
</thead><tbody id="rows"></tbody></table></details></main><script>
const D=__DATA__,el=id=>document.getElementById(id);
for(const item of D.overviews){const o=document.createElement('option');o.value=item.scene;o.textContent=item.scene;el('scene').appendChild(o)}
if(D.side_overviews.length){const o=document.createElement('option');o.value='side';o.textContent='Pickup side view';el('view').appendChild(o)}
function cell(value){const e=document.createElement('td');e.textContent=value;return e}
function show(){const item=D.overviews.find(x=>x.scene===el('scene').value),side=el('view').value==='side';
const view=side?D.side_overviews.find(x=>x.scene===item.scene):item;el('sheet').src=view.image;el('full').href=view.image;
el('sheet').alt=item.scene+(side?': complete selected gripper geometry at pickup':': selected contact pairs on native CAD');
el('totals').textContent=item.selected_count+' / '+item.tiles.length+' parts have a selected grasp.';
el('viewnote').textContent=side?'Pickup side view: blue palm and fingers, gold pads, gray part, brown table line. The red cross is the fingertip reference. These are the complete collision solids used by the geometry checks, at the holding finger opening.':'View from above: height and full gripper geometry are shown in the pickup side view when available.';
el('rows').replaceChildren();for(const tile of item.tiles){const row=document.createElement('tr');if(!tile.selected_count)row.className='failed';
for(const value of [tile.part.replaceAll('_',' '),tile.selected_count?tile.outcome:tile.outcome+': '+tile.reason,tile.valid_count+' / '+tile.proposed_count])row.appendChild(cell(value));
const td=cell(''),details=document.createElement('details'),summary=document.createElement('summary'),pre=document.createElement('pre');
summary.textContent='Saved measurements';pre.textContent=JSON.stringify(tile,null,2);details.append(summary,pre);td.appendChild(details);row.appendChild(td);el('rows').appendChild(row)}}
el('scene').onchange=show;el('view').onchange=show;show();</script></html>'''


def render(run_dir, *, side_views=True, hand_path=None):
    run_dir = source.local_path(run_dir)
    results_path = run_dir/'results.json'
    before = source.sha(results_path)
    records = json.loads(results_path.read_text(encoding='utf-8'))
    scenes, parts = validate_records(records)
    output = run_dir/'gallery'
    output.mkdir(parents=True, exist_ok=True)
    meshes, mesh_sources = {}, {}
    for part in parts:
        path = source.local_path(next(row['mesh_path'] for row in records if row['part'] == part))
        meshes[part] = source.read_stl(path)
        mesh_sources[part] = dict(path=str(path), sha256=source.sha(path))
    overviews = render_sheets(records, meshes, scenes, parts, output)
    side_overviews, hand_sources = (render_side_sheets(records, meshes, scenes, parts, output, hand_path)
                                    if side_views else ([], {}))
    selected_count = sum(row['selection']['status'] == 'selected' for row in records)
    plotted_count = sum(sheet['selected_count'] for sheet in overviews)
    if selected_count != plotted_count:
        raise ValueError('Selected count differs from the plotted contact-pair count')
    if side_views and sum(sheet['rendered_count'] for sheet in side_overviews) != selected_count:
        raise ValueError('Selected count differs from rendered complete pickup hands')
    if source.sha(results_path) != before:
        raise ValueError('Saved selections changed during rendering')
    manifest = dict(placements=len(records), scenes=scenes, parts=parts,
        overviews=overviews, side_overviews=side_overviews,
        selected_grasps=selected_count, source_results=str(results_path),
        source_results_sha256=before, renderer_source_sha256=source.sha(__file__),
        native_meshes=mesh_sources, hand_sources_sha256=hand_sources,
        status_counts=dict(Counter(row['selection']['status'] for row in records)),
        input_units='metres', display_units='millimetres', selected_only=True,
        top_projection='world XY, centered per CAD for display',
        side_projection='table normal vertical; projected jaw closing axis horizontal',
        side_hand_frame='saved cad_T_grip_site; cad_T_hand is fingertip reference only',
        side_hand_geometry='all saved convex collision components at holding opening',
        no_candidate_generation_or_ranking=True, no_mesh_simplification=True,
        VLM_selection_excluded=True, no_robot_execution=True,
        no_final_assembly_views=True, source_results_unchanged=True,
        browser_interaction_tested=False)
    packed = json.dumps(manifest, ensure_ascii=True).replace('<', '\\u003c')
    (output/'index.html').write_text(HTML.replace('__DATA__', packed), encoding='utf-8')
    (output/'gallery_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print('GALLERY='+str(output/'index.html'), flush=True)
    return output


main = render


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python -m experiments.cad_grasp_refinement_20261029.gallery RUN_DIRECTORY')
    render(sys.argv[1])
