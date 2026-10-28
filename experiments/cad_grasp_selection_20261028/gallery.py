"""Selected-only CAD contact atlas for the eight saved NIST scenes.

Rendering consumes saved selections. It never samples, validates, ranks, calls
a VLM, executes motion or renders the final assembly configuration.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

from experiments.fabrica_original_20261027 import gallery as previous


ROOT = previous.ROOT
CAD_COLOR = previous.CAD_COLOR
CONTACT_COLOR = '#285bc2'
CAPTION = 'One geometrically selected grasp per available part; VLM and execution paused.'


def selected_pose(record):
    selection = record['selection']
    pose = selection.get('selected_grasp')
    if selection.get('status') != 'selected':
        if pose is not None:
            raise ValueError('A failed selection must not contain a fallback grasp')
        return None
    if not isinstance(pose, dict) or not pose.get('valid'):
        raise ValueError('Selected result must contain a valid complete grasp')
    if pose.get('validity', {}).get('status') != 'valid':
        raise ValueError('Selected grasp must have passed geometry validation')
    contacts = np.asarray(pose.get('contacts_cad_m'), dtype=float)
    hand = np.asarray(pose.get('cad_T_hand'), dtype=float)
    if contacts.shape != (2, 3) or not np.isfinite(contacts).all():
        raise ValueError('Selected grasp needs two finite CAD-frame contacts')
    if hand.shape != (4, 4) or not np.isfinite(hand).all():
        raise ValueError('Selected grasp needs a finite CAD-relative hand transform')
    return pose


def validate_records(records):
    if not isinstance(records, list) or len(records) != 96:
        raise ValueError('Expected all 96 placements: twelve parts in eight scenes')
    scenes = sorted({str(record['scene']) for record in records})
    parts = sorted({str(record['part']) for record in records})
    if scenes != previous.SCENES or len(parts) != 12:
        raise ValueError('Expected N00-N07 and twelve parts')
    if len({(record['scene'], record['part']) for record in records}) != 96:
        raise ValueError('Repeated or missing scene/part selection')
    for record in records:
        pose = selected_pose(record)
        if record.get('world_T_cad') is None:
            if pose is not None:
                raise ValueError('Cannot render a selected grasp without its estimated CAD pose')
            continue
        transform = np.asarray(record['world_T_cad'], dtype=float)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError('Expected a finite estimated world_T_cad transform')
        if pose is not None:
            expected = transform @ np.asarray(pose['cad_T_hand'])
            if not np.allclose(expected, pose.get('world_T_hand_pickup'), rtol=0., atol=1e-9):
                raise ValueError('Saved pickup hand transform does not match the rendered CAD frame')
    return scenes, parts


def outcome(record):
    selection = record['selection']
    status = selection.get('status', 'status_not_recorded')
    if status == 'selected':
        return 'Selected grasp ' + str(selection['selected_grasp']['candidate_id'])
    if status == 'no_valid_grasp_within_budget':
        return f"No valid grasp in {selection.get('proposed_count', 0)} proposals"
    if 'unavailable' in status:
        return 'Required input unavailable'
    if 'timeout' in status or 'timed_out' in status:
        return 'Selection timed out'
    return str(status).replace('_', ' ')


def tile_metadata(record, contacts_world=None):
    selection = record['selection']
    selected = selected_pose(record)
    tile = dict(part=record['part'], status=selection.get('status'),
                outcome=outcome(record), selected_count=int(selected is not None),
                proposed_count=selection.get('proposed_count', 0),
                valid_count=selection.get('valid_count', 0),
                message=selection.get('message', ''),
                missing_or_invalid_inputs=selection.get('missing_or_invalid_inputs', []),
                rejection_counts=record.get('rejection_counts', record.get('failure_counts', {})),
                wall_seconds=record.get('wall_seconds'))
    if selected is not None:
        tile.update(candidate_id=selected['candidate_id'],
                    criteria=selected['criteria'],
                    holding_opening_m=selected['holding_opening_m'],
                    pre_opening_m=selected['pre_opening_m'],
                    contacts_cad_m=selected['contacts_cad_m'],
                    contacts_world_m=contacts_world.tolist(),
                    approach_world=selected['approach_world'],
                    world_T_hand_pickup=selected['world_T_hand_pickup'])
    return tile


def render_sheets(records, meshes, scenes, parts, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': 'white',
                         'savefig.bbox': None})
    lookup = {(record['scene'], record['part']): record for record in records}
    overviews = []
    for scene in scenes:
        fig, axes = plt.subplots(3, 4, figsize=(16, 11), dpi=previous.IMAGE_DPI,
                                 constrained_layout=True)
        tiles, overlay_coordinates = [], []
        for ax, part in zip(axes.flat, parts):
            record = lookup[(scene, part)]
            selected = selected_pose(record)
            pair_world = None
            if record.get('world_T_cad') is not None:
                vertices, faces = meshes[part]
                transform = np.asarray(record['world_T_cad'], dtype=float)
                world = vertices @ transform[:3, :3].T + transform[:3, 3]
                center = (world.min(axis=0) + world.max(axis=0)) / 2
                points_mm = (world - center) * 1000
                ax.add_collection(PolyCollection(points_mm[faces][:, :, :2],
                    facecolors=CAD_COLOR, edgecolors='none', antialiaseds=False, zorder=1))
                if selected is not None:
                    pair = np.asarray(selected['contacts_cad_m'], dtype=float)
                    pair_world = pair @ transform[:3, :3].T + transform[:3, 3]
                    pair_mm = (pair_world - center) * 1000
                    ax.plot(pair_mm[:, 0], pair_mm[:, 1], color=CONTACT_COLOR,
                            linewidth=1.25, zorder=4)
                    ax.scatter(pair_mm[:, 0], pair_mm[:, 1], color=CONTACT_COLOR,
                               s=14, edgecolors='none', zorder=5)
                    overlay_coordinates.append((ax, pair_mm[:, :2]))
                else:
                    overlay_coordinates.append((ax, None))
                radius = max(float(np.ptp(points_mm[:, :2], axis=0).max()) * .64, 2.)
                ax.set_xlim(-radius, radius)
                ax.set_ylim(-radius, radius)
            else:
                ax.text(.5, .5, 'Observed CAD pose\nunavailable', transform=ax.transAxes,
                        ha='center', va='center', fontsize=10, color='#b33d32')
                overlay_coordinates.append((ax, None))
            ax.set_aspect('equal')
            ax.set_axis_off()
            label = part.replace('_', ' ') + '\n' + outcome(record)
            if selected is not None:
                label += (f"; {record['selection'].get('valid_count', 0)} valid / "
                          f"{record['selection'].get('proposed_count', 0)} proposed")
            ax.set_title(label, fontsize=9., pad=8,
                         color='#263846' if selected is not None else '#b33d32')
            tiles.append(tile_metadata(record, pair_world))
        fig.suptitle(scene + '\n' + CAPTION + '\n'
            'Lines join the two selected pad-center contacts, viewed from above. Height and hand orientation are not shown.',
            fontsize=11.5)
        fig.canvas.draw()
        fig.set_layout_engine('none')
        width, height = fig.canvas.get_width_height()
        for tile, (ax, pair_xy) in zip(tiles, overlay_coordinates):
            tile['selected_contact_overlay_px'] = None
            if pair_xy is not None:
                pixels = ax.transData.transform(pair_xy)
                pixels[:, 1] = height - pixels[:, 1]
                tile['selected_contact_overlay_px'] = np.round(pixels, 4).tolist()
                tile['projected_contact_span_mm'] = float(np.linalg.norm(pair_xy[1]-pair_xy[0]))
        path = output / f'{scene}.png'
        fig.savefig(path, dpi=previous.IMAGE_DPI, facecolor='white')
        if scene == scenes[0]:
            # Stable preview name for inline/local-image review.
            fig.savefig(output / 'N00_preview.png', dpi=previous.IMAGE_DPI, facecolor='white')
        plt.close(fig)
        overviews.append(dict(scene=scene, image=path.name, pixel_width=width,
                              pixel_height=height, tiles=tiles,
                              selected_count=sum(tile['selected_count'] for tile in tiles)))
        print('Selected CAD contact sheet: ' + scene, flush=True)
    return overviews


HTML = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Selected CAD grasps — eight NIST scenes</title><style>
body{margin:0;background:#fff;color:#263846;font:15px system-ui,sans-serif}main{max-width:1600px;margin:auto;padding:22px}
h1{font-size:23px;margin:0 0 10px}p{line-height:1.45;max-width:1100px}.note{color:#596875}
select{font:inherit;padding:5px 12px;margin-left:7px}img{display:block;width:100%;height:auto;margin:16px 0}
a{color:#285bc2}details{margin:20px 0}summary{cursor:pointer}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;vertical-align:top;padding:8px;border-bottom:1px solid #dde3e8}th{font-weight:600}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}.failed{color:#b33d32}.controls{display:flex;gap:20px;align-items:center;flex-wrap:wrap}
</style><main><h1>Selected grasp on each CAD model</h1>
<p>Each blue line joins the two intended pad-center contacts of the one selected grasp.
The gray CAD retains its saved observed pickup orientation. A red label means no grasp
was selected; no invalid fallback is shown.</p>
<p class="note">Geometric selection only: assembly moment arm first, CAD surface centrality second,
downward approach preference third. This view from above does not show hand height or orientation;
the selected pickup pose is available in the details. VLM selection, pickup and assembly remain paused.
No final assembly views are included.</p>
<div class="controls"><label>Scene<select id="scene"></select></label>
<a id="full" href="#">Open this CAD sheet</a><span id="totals"></span></div>
<img id="sheet" alt="Selected grasp contacts on twelve CAD models">
<details><summary>Selected grasp measurements and missing-selection reasons</summary>
<p class="note">Moment arm and surface distance are geometric criteria, not measured torque or assembly precision.</p>
<table><thead><tr><th>Part</th><th>Result</th><th>Valid / proposed</th><th>Moment arm</th><th>Surface distance</th><th>Downward angle</th><th>Details</th></tr></thead><tbody id="rows"></tbody></table></details>
</main><script>const D=__DATA__;const el=id=>document.getElementById(id);
for(const overview of D.overviews){const option=document.createElement('option');option.value=overview.scene;option.textContent=overview.scene;el('scene').appendChild(option)}
function td(value){const cell=document.createElement('td');cell.textContent=value;return cell}
function number(value,digits=3){return Number.isFinite(value)?value.toFixed(digits):'Unavailable'}
function show(){const item=D.overviews.find(x=>x.scene===el('scene').value);el('sheet').src=item.image;
el('sheet').alt=item.scene+': one selected contact pair per available part';el('full').href=item.image;
el('totals').textContent=item.selected_count+' / '+item.tiles.length+' parts have a selected grasp.';el('rows').replaceChildren();
for(const part of item.tiles){const row=document.createElement('tr'),c=part.criteria||{};
if(!part.selected_count)row.className='failed';for(const value of [part.part.replaceAll('_',' '),part.outcome,
part.valid_count+' / '+part.proposed_count,part.selected_count?number(c.q_tau_m*1000)+' mm':'—',
part.selected_count?number(c.q_surface_m*1000)+' mm':'—',part.selected_count?number(c.theta_down_rad*180/Math.PI,1)+'°':'—'])row.appendChild(td(value));
const cell=document.createElement('td'),details=document.createElement('details'),summary=document.createElement('summary'),pre=document.createElement('pre');
summary.textContent=part.selected_count?'Pickup pose and criteria':'Reason';pre.textContent=JSON.stringify(part,null,2);
details.append(summary,pre);cell.appendChild(details);row.appendChild(cell);el('rows').appendChild(row)}}
el('scene').onchange=show;show();</script></html>'''


def main(run_dir):
    run_dir = previous.local_path(run_dir)
    source = run_dir / 'results.json'
    before = previous.sha(source)
    records = json.loads(source.read_text(encoding='utf-8'))
    scenes, parts = validate_records(records)
    output = run_dir / 'gallery'
    output.mkdir(parents=True, exist_ok=True)
    meshes, sources = {}, {}
    for part in parts:
        record = next(record for record in records if record['part'] == part)
        path = previous.local_path(record['mesh_path'])
        meshes[part] = previous.read_stl(path)
        sources[part] = dict(path=str(path), sha256=previous.sha(path),
                             native_triangles=len(meshes[part][1]))
    overviews = render_sheets(records, meshes, scenes, parts, output)
    expected = sum(record['selection'].get('status') == 'selected' for record in records)
    plotted = sum(overview['selected_count'] for overview in overviews)
    if plotted != expected:
        raise ValueError('Selected-pose count and plotted contact-pair count differ')
    if previous.sha(source) != before:
        raise ValueError('Saved selection results changed during rendering')
    manifest = dict(placements=len(records), scenes=scenes, parts=parts,
        overviews=overviews, selected_grasps=plotted,
        status_counts=dict(Counter(record['selection'].get('status') for record in records)),
        source_results=str(source), source_results_sha256=before,
        renderer_source_sha256=previous.sha(__file__), native_meshes=sources,
        projection='world XY at observed pickup pose; centered per tile for display only',
        contact_frame='native CAD; transformed with saved estimated pickup pose',
        input_units='metres', display_units='millimetres', selected_only=True,
        no_candidate_generation_or_ranking=True, no_mesh_simplification=True,
        no_vlm_calls=True, no_robot_execution=True, no_final_assembly_views=True,
        static_preview='N00_preview.png', browser_interaction_tested=False)
    data = json.dumps(manifest, ensure_ascii=True).replace('<', '\\u003c')
    (output / 'index.html').write_text(HTML.replace('__DATA__', data), encoding='utf-8')
    (output / 'gallery_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n',
                                                 encoding='utf-8')
    (run_dir / 'gallery_verification.json').write_text(json.dumps(dict(
        placements=len(records), scenes=len(scenes), selected_saved=expected,
        selected_plotted=plotted, all_sheet_images_present=all(
            (output / overview['image']).is_file() for overview in overviews),
        source_results_unchanged=True, no_final_assembly_views=True,
        browser_interaction_tested=False), indent=2) + '\n', encoding='utf-8')
    print('GALLERY=' + str(output / 'index.html'), flush=True)
    return output


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python -m experiments.cad_grasp_selection_20261028.gallery RUN_DIRECTORY')
    main(sys.argv[1])
