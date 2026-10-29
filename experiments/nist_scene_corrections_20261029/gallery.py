"""Corrected static CAD side views; no candidate sampling or robot execution."""
from __future__ import annotations

from functools import lru_cache
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from experiments.nist_mating_goals_20261028.cad_tools import read_native_mesh, section_segments, transform_points

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'outputs/nist_scene_corrections_20261029'


@lru_cache(maxsize=64)
def mesh(path):
    path = Path(path)
    return read_native_mesh(path if path.is_absolute() else ROOT / path)


def _draw(ax, row, registry, cut_axis):
    pose = np.asarray(row['board_T_part'], float)
    vertices, faces = mesh(row['part_mesh'])
    points = transform_points(vertices, pose)
    axis_point = np.asarray(row['axis_point_cad_m'], float)
    center = transform_points(axis_point[None], pose)[0]
    level = float(center[cut_axis] + 3e-6)
    transverse = [i for i in range(3) if i != cut_axis]
    reports = []
    receiver_points = []
    for receiver in row['receivers']:
        v, f = mesh(receiver['mesh_path'])
        world = transform_points(v, receiver['board_T_mesh'])
        segments = section_segments(world, f, cut_axis, level)
        display = (segments - center[transverse]) * 1000
        ax.add_collection(LineCollection(display, colors='#87939d', linewidths=.85, zorder=2))
        if receiver['name'].lower() not in ('board', 'task_board'):
            receiver_points.append(world)
        reports.append(dict(receiver=receiver['name'], segment_count=len(segments)))
    segments = section_segments(points, faces, cut_axis, level)
    display = (segments - center[transverse]) * 1000
    ax.add_collection(LineCollection(display, colors='#137db4', linewidths=1.1, zorder=3))
    # Insertion axis point is a mating datum, not a cable bounding-box center.
    direction = pose[:3, :3] @ np.asarray(row['axis_direction_cad'], float)
    projected = direction[transverse]
    if np.linalg.norm(projected) > .1:
        ax.annotate('', xy=projected * 7., xytext=-projected * 5.,
                    arrowprops=dict(arrowstyle='->', color='#df8b00', lw=1.6), zorder=5)
    top = float(registry.get('board_top_board_m', .0089916 / 2))
    support = float(registry.get('support_z_board_m', -.0089916 / 2 - .020))
    ax.axhline((top-center[2])*1000, color='#6f7780', ls='--', lw=.7)
    ax.axhline((support-center[2])*1000, color='#353535', lw=1.1)
    if row['part'].startswith(('KET', 'RGOCG')):
        width = max(12., float(np.ptp(points[:, transverse[0]]) * 500 + 7))
        zlo, zhi = support - .003, float(points[:, 2].max() + .005)
    elif row['part'] == 'USB_Male':
        width, zlo, zhi = 27., top-.015, top+.065
    else:
        all_points = np.concatenate([points, *receiver_points]) if receiver_points else points
        width = max(16., float(np.max(np.abs(all_points[:, transverse[0]]-center[transverse[0]]))*1000+5))
        width = min(width, 65.)
        zlo, zhi = min(support-.003, float(all_points[:, 2].min()-.003)), float(all_points[:, 2].max()+.006)
    ax.set_xlim(-width, width)
    ax.set_ylim((zlo-center[2])*1000, (zhi-center[2])*1000)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(('X' if cut_axis == 1 else 'Y')+' from mating axis (mm)', fontsize=8)
    ax.set_ylabel('Z from mating datum (mm)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=.16)
    ax.set_title('X–Z side cut' if cut_axis == 1 else 'Y–Z side cut', fontsize=10)
    return dict(cut_axis=cut_axis, offset_m=3e-6, target_segments=len(segments), receivers=reports)


def render():
    registry = json.loads((OUTPUT / 'assembly_goals.json').read_text(encoding='utf-8'))
    rows = registry['parts']
    gallery = OUTPUT / 'gallery'
    gallery.mkdir(parents=True, exist_ok=True)
    cards, records = [], []
    fig, axes = plt.subplots(3, 4, figsize=(15, 13), layout='constrained')
    for row, overview_ax in zip(rows, axes.flat):
        part = row['part']
        if row.get('board_T_part') is None:
            overview_ax.text(.5,.5,part+'\nUnavailable reference',ha='center',va='center',transform=overview_ax.transAxes)
            continue
        detail, detail_axes = plt.subplots(1, 2, figsize=(10, 6))
        detail.subplots_adjust(left=.09, right=.98, bottom=.16, top=.85, wspace=.32)
        cuts = [_draw(ax, row, registry, cut_axis) for ax, cut_axis in zip(detail_axes, (1, 0))]
        title = part.replace('_', ' ')
        detail.suptitle(title+' — corrected CAD goal', fontsize=13)
        footer = 'Blue: part. Gray: receiving CAD. Orange: insertion direction. Black: support surface. Static CAD reference only.'
        if part.startswith(('KET', 'RGOCG')):
            footer += '\nPin tip: 20 mm below board top; not bottomed against the support.'
        detail.text(.5,.008,footer,ha='center',va='bottom',fontsize=8)
        filename = part+'.png'
        detail.savefig(gallery / filename, dpi=150)
        plt.close(detail)
        _draw(overview_ax, row, registry, 1)
        overview_ax.set_title(title, fontsize=10)
        overview_ax.set_xlabel('')
        overview_ax.set_ylabel('')
        records.append(dict(part=part, file=filename, cuts=cuts))
        status = str(row.get('goal_status', row.get('status', 'CAD reference')))
        cards.append(f'<article><h2>{html.escape(title)}</h2><p>{html.escape(status)}</p><a href="{filename}"><img src="{filename}" alt="Corrected side views of {html.escape(title)}"></a></article>')
    fig.suptitle('Corrected NIST assembly references — 20 mm pin-tip target\nStatic CAD side cuts; no grasp generation or robot execution', fontsize=15)
    fig.savefig(gallery / 'overview.png', dpi=150)
    plt.close(fig)
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Corrected NIST assembly references</title><style>body{font:16px system-ui;margin:0;background:#f4f6f8;color:#203040}main{max-width:1450px;margin:auto;padding:24px}p{line-height:1.5}.notice{background:#fff0d2;padding:16px;border-left:4px solid #b37900}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}article{background:white;padding:16px;border:1px solid #d7dce0;border-radius:8px}img{width:100%;height:auto}h2{font-size:20px}a{color:#13689c}@media(max-width:850px){.grid{grid-template-columns:1fr}}</style>
<main><h1>Corrected NIST assembly references</h1><p class="notice">Current scene and goal corrections, October 29. Pin-tip depth: <strong>20 mm below the board top</strong>.
Grasp generation, pickup and assembly remain paused. VLM selection is excluded. These are corrected static CAD references, not measured assembly-success results.</p>
<p>Blue: target part. Gray: receiving CAD. Orange: insertion direction. Black line: supporting surface. Dashed line: board top.
Electrical contacts and threads may have modeled contact/interference; source placement is not a physical fit certificate.</p>
<p><a href="../assembly_goals.json">Goal transforms and mating regions</a> · <a href="../verification.json">Offline verification</a></p>
<a href="overview.png"><img src="overview.png" alt="All twelve corrected CAD assembly references"></a><div class="grid">''' + ''.join(cards) + '</div></main></html>'
    (gallery / 'index.html').write_text(page, encoding='utf-8')
    (gallery / 'manifest.json').write_text(json.dumps(dict(parts=records, no_grasp_or_motion_execution=True), indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(side_view_sheets=len(records), gallery=str(gallery / 'index.html'))))


if __name__ == '__main__':
    render()
