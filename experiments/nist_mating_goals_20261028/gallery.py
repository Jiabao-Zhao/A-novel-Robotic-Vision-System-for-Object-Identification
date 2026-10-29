"""Render CAD-only side sections of old and reviewed mating references.

The renderer never calls grasp generation, a VLM, a simulator or a robot.
Blue/red areas are actual mesh sections, rather than component convex hulls.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import html
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.path import Path as PlotPath
from matplotlib.patches import PathPatch
import numpy as np
from shapely.geometry import box
from shapely.geometry.polygon import orient

from .cad_tools import (read_native_mesh, transform_points, mesh_component_labels,
                        section_segments, section_geometry)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT/'outputs/nist_mating_goals_20261028'
ORDER = ['BNC_Male','DSUB_Male','Gear_Large','KET12_Square_12mm',
         'KET16_Square_16mm','KET8_Square_8mm','M16_Hex_Nut',
         'RGOCG12-50_12mm','RGOCG16-50_16mm','RGOCG8-50_8mm',
         'USB_Male','Waterproof_Male']
BLUE = '#177cb8'
OLD = '#c05c3c'
GRAY = '#848e99'


@lru_cache(maxsize=48)
def mesh(path):
    path = Path(path)
    if not path.is_absolute(): path = ROOT/path
    v, f = read_native_mesh(path)
    return v, f, mesh_component_labels(f)


def patch(ax, geometry, color, alpha):
    """Draw polygon holes as holes, preserving section geometry."""
    if geometry.is_empty: return
    if geometry.geom_type != 'Polygon':
        for item in getattr(geometry, 'geoms', []): patch(ax, item, color, alpha)
        return
    polygon = orient(geometry, sign=1)
    vertices, codes = [], []
    for ring in [polygon.exterior, *polygon.interiors]:
        points = np.asarray(ring.coords)
        vertices.extend(points)
        codes.extend([PlotPath.MOVETO] + [PlotPath.LINETO]*(len(points)-2) + [PlotPath.CLOSEPOLY])
    ax.add_patch(PathPatch(PlotPath(vertices, codes), facecolor=color,
                          edgecolor='none', alpha=alpha, zorder=2))


def _local_geometry(geometry, transverse, center):
    from shapely.affinity import affine_transform
    # Section coordinates follow ascending transverse axes; display in mm.
    return affine_transform(geometry, [1000, 0, 0, 1000,
                            -1000*center[transverse[0]], -1000*center[transverse[1]]])


def draw(ax, row, *, old=False, cut_axis=1, compact=False):
    transform = np.asarray(row['old_board_T_part'] if old else row['review_board_T_part'])
    v, faces, labels = mesh(row['part_mesh'])
    world = transform_points(v, transform)
    receiver_rows = row.get('old_receivers', row['receivers']) if old else row['receivers']
    center = np.asarray(row['receiver_mating_frame_board'])[:3, 3].copy()
    # A tiny declared plane offset avoids vertex/coplanar ambiguities; no CAD
    # vertices are moved. The two views are near-central sections, not surfaces.
    offset = 3e-6
    level = (transform[cut_axis,3] if old else center[cut_axis]) + offset
    transverse = [i for i in range(3) if i != cut_axis]
    color = OLD if old else BLUE
    geometries = []
    limits_points = [world]
    for item in receiver_rows:
        rv, rf, rl = mesh(item['mesh_path'])
        rw = transform_points(rv, item['board_T_mesh'])
        if item['name'] != 'board': limits_points.append(rw)
        geom, report = section_geometry(rw, rf, cut_axis, level,
            component_labels=rl, return_report=True)
        local = _local_geometry(geom, transverse, center)
        if report['complete']: patch(ax, local, GRAY, .28)
        segments = section_segments(rw, rf, cut_axis, level)
        segments = (segments-center[transverse])*1000
        ax.add_collection(LineCollection(segments, colors=GRAY, linewidths=.85, zorder=3))
        geometries.append((item['name'], geom, report))
    geom, report = section_geometry(world, faces, cut_axis, level,
        component_labels=labels, return_report=True)
    local = _local_geometry(geom, transverse, center)
    if report['complete']: patch(ax, local, color, .26)
    segments = section_segments(world, faces, cut_axis, level)
    segments = (segments-center[transverse])*1000
    ax.add_collection(LineCollection(segments, colors=color, linewidths=1.1, zorder=5))
    checks = []
    for name, other, rreport in geometries:
        valid = report['complete'] and rreport['complete']
        intersection = geom.intersection(other) if valid else None
        if intersection is not None and not intersection.is_empty:
            patch(ax, _local_geometry(intersection, transverse, center), '#d72e2e', .8)
        checks.append(dict(receiver=name, both_sections_complete=valid,
                           intersection_area_mm2=(intersection.area*1e6 if valid else None)))
    receiver_pairs=[]
    if row['part']=='M16_Hex_Nut':
        for i,(name,a,ar) in enumerate(geometries):
            for other,b,br in geometries[i+1:]:
                complete=ar['complete'] and br['complete']
                overlap=a.intersection(b) if complete else None
                if overlap is not None:
                    patch(ax,_local_geometry(overlap,transverse,center),'#d72e2e',.7)
                receiver_pairs.append(dict(receivers=[name,other],both_sections_complete=complete,
                    intersection_area_mm2=overlap.area*1e6 if complete else None))
    points = np.concatenate(limits_points)
    projected = (points[:, transverse]-center[transverse])*1000
    lo, hi = projected.min(0), projected.max(0)
    # Focus on the USB connector/receiver interface; the frozen cable can extend
    # outside this window and is explicitly labeled in its detail caption.
    if row['part'] == 'USB_Male':
        lo = np.maximum(lo, [-42., -35.]); hi = np.minimum(hi, [42., 90.])
    if row['part'] == 'Gear_Large':
        lo[0] = max(lo[0],-42.); hi[0] = min(hi[0],42.)
    span = np.maximum(hi-lo, [18., 20.])
    lo -= span*.11; hi += span*.11
    # Include the local board band near the interface, without zooming out to
    # the entire 384-mm plate.
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_aspect('equal', adjustable='box', anchor='N')
    direction = transform[:3, :3] @ np.asarray(row['axis_direction_cad'])
    middle = ((world.min(0)+world.max(0))/2-center)[transverse]*1000
    middle = np.minimum(np.maximum(middle, lo+.18*(hi-lo)), hi-.18*(hi-lo))
    projected_direction = direction[transverse]
    length = float(np.linalg.norm(projected_direction))
    if length > .05:
        tip = middle + projected_direction/length*min(13., .23*span[1])
        ax.annotate('', xy=tip, xytext=middle,
                    arrowprops=dict(arrowstyle='-|>', color='#dd8500', lw=2.0), zorder=8)
    else:
        ax.text(.02,.96,'Insertion direction perpendicular to this cut', transform=ax.transAxes,
                fontsize=7, va='top', color='#9b5b00')
    if not old and row['part'].startswith(('KET','RGOCG')):
        arrow_x = hi[0]-.18*(hi[0]-lo[0])
        ax.annotate('',xy=(arrow_x,-12.),xytext=(arrow_x,0.),
                    arrowprops=dict(arrowstyle='<->',color='#273a50',lw=1.1),zorder=9)
        ax.text(arrow_x+.5,-6.,'12 mm\ntarget',fontsize=7,va='center',zorder=10,
                bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=1))
    ax.axvline(0, color='#bbc2ca', lw=.5, ls=':', zorder=0)
    ax.axhline(0, color='#bbc2ca', lw=.5, ls=':', zorder=0)
    letters = 'XYZ'
    ax.set_xlabel(f'{letters[transverse[0]]} offset from receiver (mm)', fontsize=8)
    ax.set_ylabel('Z offset from receiver (mm)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines[['top','right']].set_visible(False)
    if not compact:
        ax.set_title(('Previous goal' if old else 'Reviewed reference') +
                     f' · {letters[transverse[0]]}–Z cut', fontsize=10,
                     color=OLD if old else BLUE)
    return dict(cut_axis='XYZ'[cut_axis], level_board_m=float(level),
                near_center_plane_offset_m=offset,
                part_section_complete=report['complete'],
                receiver_intersections=checks,
                receiver_pair_intersections=receiver_pairs,
                scope='one two-dimensional cut; not a 3D fit certificate')


def status_text(row):
    part = row['part']
    if part.startswith(('KET','RGOCG')):
        return 'Chamfer down; chosen tip depth 12 mm'
    return {
        'Gear_Large':'Gear sits on plate; shaft aligned',
        'M16_Hex_Nut':'Screw lowered 8.99 mm; thread fit unresolved',
        'BNC_Male':'10.5 mm axial overlap; CAD solids still intersect',
        'USB_Male':'12 mm plug overlap; sampled CAD cuts clear',
        'DSUB_Male':'Connector turned downward; seated stop unresolved',
        'Waterproof_Male':'Connector turned sideways; seated stop unresolved',
    }.get(part, row['status'].replace('_',' '))


def preferred_axis(row):
    if row['part'] in ('Waterproof_Male','DSUB_Male','USB_Male'): return 0
    v, _, _ = mesh(row['part_mesh'])
    v = transform_points(v, row['review_board_T_part'])
    return 1 if np.ptp(v[:,0]) >= np.ptp(v[:,1]) else 0


def render():
    output = OUTPUT/'gallery'
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ['pins_review.json','connectors_review.json','gears_nut_review.json']:
        rows.extend(json.loads((OUTPUT/name).read_text())['parts'])
    rows.sort(key=lambda r:ORDER.index(r['part']))
    if len(rows) != 12 or len({r['part'] for r in rows}) != 12:
        raise ValueError('Exactly twelve unique part reviews required')
    manifests = []
    for row in rows:
        fig, axes = plt.subplots(2,2,figsize=(11.8,10.5))
        fig.suptitle(row['part'].replace('_',' ')+'\n'+textwrap.fill(status_text(row),65), fontsize=14, y=.985)
        cuts = []
        for old, line in [(True,axes[0]),(False,axes[1])]:
            for axis, ax in zip((1,0),line): cuts.append(dict(old_goal=old, **draw(ax,row,old=old,cut_axis=axis)))
        caption = ('Gray: receiver/board   Orange arrow: insertion direction   '
                   'Red overlap: intersecting solid sections when both cuts close.\n'
                   'Cuts are 0.003 mm off each view center; exact levels are saved. They do not certify full 3D fit.')
        if row['part'].startswith(('KET','RGOCG')):
            caption += '\n12 mm below the board top is the user-chosen task depth; the through-pin CAD has no axial stop.'
        if row['part']=='USB_Male': caption += '\nConnector detail: portions of the cable outside the displayed window are cropped.'
        if row['part']=='M16_Hex_Nut': caption += '\nRed also highlights the screw-head/board overlap in the previous mounting.'
        fig.text(.5,.012,caption,ha='center',va='bottom',fontsize=8,color='#495463')
        fig.subplots_adjust(left=.08,right=.97,top=.88,bottom=.13,hspace=.50,wspace=.30)
        file = output/f"{row['part']}.png"
        fig.savefig(file,dpi=150,facecolor='white'); plt.close(fig)
        manifests.append(dict(part=row['part'],file=file.name,status=row['status'],
                              caption=status_text(row),sections=cuts))
    fig, axes = plt.subplots(3,4,figsize=(18,13.5))
    for row, ax in zip(rows,axes.ravel()):
        draw(ax,row,cut_axis=preferred_axis(row),compact=True)
        ax.set_title(row['part'].replace('_',' ')+'\n'+textwrap.fill(status_text(row),38),fontsize=9)
    fig.suptitle('NIST CAD mating review — side sections of reviewed references\n'
                 'Gray receiver · blue part · orange insertion direction · red section overlap',fontsize=15,y=.995)
    fig.text(.5,.006,'Pin depth is the chosen 12 mm task target. '
             'These are CAD reference poses, not executed or fully verified assemblies.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.025,1,.95])
    fig.savefig(output/'overview.png',dpi=145,facecolor='white'); plt.close(fig)
    data=dict(parts=manifests,overview='overview.png',
              no_grasp_generation=True,no_VLM=True,no_robot_or_physics=True,
              display_units='millimetres',input_units='metres',
              no_full_3D_fit_certificate=True)
    (output/'gallery_manifest.json').write_text(json.dumps(data,indent=2)+'\n')
    cards=[]
    for row, entry in zip(rows,manifests):
        warnings=''.join('<li>'+html.escape(w)+'</li>' for w in row['warnings'])
        changes=''.join('<li>'+html.escape(w)+'</li>' for w in row['changes'])
        detail=json.dumps({k:row[k] for k in ['status','old_board_T_part',
            'review_board_T_part','proposed_board_T_part','axis_direction_cad',
            'axis_direction_board']},indent=2)
        cards.append(f'''<article id="{row['part']}"><h2>{html.escape(row['part'].replace('_',' '))}</h2>
<p class="status">{html.escape(status_text(row))}</p>
<a href="{entry['file']}"><img src="{entry['file']}" loading="lazy" alt="Previous and reviewed side sections of {row['part']}"></a>
<details><summary>What changed and what remains unresolved</summary><ul>{changes}</ul><ul>{warnings}</ul>
<pre>{html.escape(detail)}</pre></details></article>''')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NIST mating-goal CAD review</title><style>
body{margin:0;background:#f5f7f9;color:#213040;font:16px system-ui,sans-serif}main{max-width:1480px;margin:auto;padding:24px}
h1{font-size:27px}p{line-height:1.5}.notice{padding:16px;background:#fff2d7;border-left:4px solid #bb7d10}
article{background:white;border:1px solid #dbe1e8;border-radius:8px;padding:18px;min-width:0}h2{font-size:20px;margin:0}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.status{color:#76551a;font-weight:600}img{width:100%;height:auto}
summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}li{margin:8px 0}a{color:#126aa2}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
</style><main><h1>NIST mating-goal CAD review</h1>
<p class="notice"><strong>Superseded mounting/destination assumptions — October 29:</strong>
The official NIST design requires vertical waterproof mating, exchanges our USB/DSUB receiver locations, and corrects M4/M12 bolts assigned to pin holes.
The round-12 alternate hole used below belongs to the M12 bolt. These drawings preserve the earlier mesh diagnostics; they are not the corrected official assembly layout.
See the <a href="../../nist_official_reference_20261029/gallery/index.html">official source gallery</a> and
<a href="../../../docs/nist_task_board_1_reference.md">reference report</a>. User-selected pin depth remains 12 mm.</p>
<p>Twelve parts compared with their actual receiver meshes. Old goals appear in the upper row; reviewed references in the lower row.
These are mesh sections: gray receiver, blue reviewed part, orange insertion arrow and red overlapping solid areas where both sections close.</p>
<p class="notice">Grasp generation, pickup and assembly remain paused. The pipeline makes the final grasp decision using geometric criteria and deterministic priorities; VLM selection is not part of the pipeline. A reference shown here is not automatically a verified seated goal.
Straight pins use the user-chosen 12 mm tip depth below the board top. Incomplete section outlines cannot establish clearance.</p>
<p><a href="../review.json">Complete measurements and transforms</a> · <a href="../report.md">Review report</a> ·
<a href="../assembly_goals.json">Corrected goal registry and unresolved parameters</a></p>
<a href="overview.png"><img src="overview.png" alt="All twelve reviewed mating side sections"></a>
<div class="grid">'''+''.join(cards)+'</div></main></html>'
    (output/'index.html').write_text(page,encoding='utf-8')
    return rows,data


if __name__=='__main__': render()
