"""Inspect and draw CAD sampling labels without generating grasp candidates."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
from PIL import Image

from .sampling_annotations import CONFIG
from .task_inputs import REGISTRY, ROOT, build_request

OUTPUT = ROOT / 'outputs/cad_grasp_selection_20261029/region_guidance'


def _projection(points, origin, direction):
    # Insertion direction points down in this CAD-only inspection view.
    up = -np.asarray(direction, float)
    seed = np.eye(3)[int(np.argmin(np.abs(up)))]
    right = np.cross(seed, up)
    right /= np.linalg.norm(right)
    depth = np.cross(up, right)
    points = np.asarray(points)-origin
    return np.stack([points@right+.25*(points@depth),
                     points@up+.18*(points@depth)], axis=-1)*1000


def audit():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    registry = json.loads(REGISTRY.read_text(encoding='utf-8'))
    annotations = json.loads(CONFIG.read_text(encoding='utf-8'))['sampling_annotations']
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    rows = []
    for row, ax in zip(registry['parts'], axes.flat):
        name = row['part']
        # CAD-local region compilation is independent of the pickup transform.
        result = build_request(name, np.eye(4), [0, 0, 1, 0], registry=registry)
        if result['status'] != 'available':
            raise ValueError((name, result.get('message')))
        request = result['request']
        region = request['sampling_contact_region']
        patches = region['sampling_triangles_cad_m']
        source_ids = region['source_face_ids']
        assert region['eligible_area_m2'] > 0
        assert region['endpoint_contains'](patches.mean(axis=1), source_ids).all()
        vertices, faces = request['vertices_cad_m'], request['faces']
        origin = np.asarray(row['axis_point_cad_m'])
        direction = np.asarray(row['axis_direction_cad'])
        full = _projection(vertices[faces], origin, direction)
        permitted = _projection(patches, origin, direction)
        ax.add_collection(PolyCollection(full, facecolors='#d7dce0', edgecolors='none'))
        ax.add_collection(PolyCollection(permitted, facecolors='#078c72', edgecolors='none'))
        # Show the connector rather than shrinking it to the cable's full length.
        # The complete cable is still in request['vertices_cad_m'] and all rays.
        if name == 'USB_Male':
            body = vertices[vertices[:, 0] < .036]
            extents = _projection(body, origin, direction)
            detail = 'Connector detail; complete cable retained in CAD'
        else:
            extents = full.reshape(-1, 2)
            detail = annotations[name]['label']
        low, high = extents.min(axis=0), extents.max(axis=0)
        center = (low+high)/2
        radius = max(float(np.max(high-low))*.56, 1.)
        ax.set_xlim(center[0]-radius, center[0]+radius)
        ax.set_ylim(center[1]-radius, center[1]+radius)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(name.replace('_', ' '), fontsize=11)
        area = region['eligible_area_m2']*1e6
        ax.text(.5, -.025, f'{area:,.1f} mm² eligible surface',
                transform=ax.transAxes, ha='center', fontsize=10, color='#075e50')
        ax.text(.5, -.085, detail, transform=ax.transAxes, ha='center', fontsize=8)
        provenance = region['provenance']
        rows.append(dict(part=name, label=annotations[name]['label'],
            eligible_area_mm2=area,
            full_surface_area_mm2=provenance['original_surface_area_m2']*1e6,
            eligible_area_fraction=provenance['eligible_area_fraction'],
            full_mesh_triangle_count=len(faces),
            eligible_original_face_count=provenance['eligible_source_face_count'],
            clipped_sampling_patch_count=len(patches),
            mesh_binding={k: region[k] for k in ('mesh_vertices_sha256', 'mesh_faces_sha256')},
            source_CAD_sha256=row['part_mesh_sha256'],
            boundary_margin_m=provenance['boundary_margin_m'],
            reviewed_geometry_basis=annotations[name]['reviewed_geometry_basis'],
            complete_CAD_retained=True, functional_graspability_validated=False))
    assert len(rows) == 12
    fig.suptitle('Permitted CAD contact regions — no grasp candidates generated', fontsize=17, y=.985)
    fig.text(.5, .947, 'Green: permitted sampling surface. Gray: complete CAD. '
             'Insertion direction points down in each view.', ha='center', fontsize=11)
    fig.text(.5, .017, 'Oblique CAD projections show labels from both sides, not camera visibility. '
             'Finite finger contact and collision checks still follow sampling.',
             ha='center', fontsize=10)
    fig.subplots_adjust(left=.025, right=.975, top=.90, bottom=.07, hspace=.40, wspace=.18)
    fig.savefig(OUTPUT/'regions.png', dpi=145)
    plt.close(fig)
    with Image.open(OUTPUT/'regions.png') as image:
        image.verify()
    sources = [*Path(__file__).parent.glob('*.py'), CONFIG, REGISTRY]
    receipt = dict(scope='CAD region compilation and annotation inspection only',
        part_types=len(rows), candidate_generation=False, selection=False,
        VLM_selection=False, pickup=False, assembly=False, physics=False,
        full_CAD_retained_for_rays_and_collisions=True,
        measured_grasp_quality_or_runtime_improvement=False,
        pin_tip_depth_below_board_top_m=.020, parts=rows,
        source_sha256={path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sorted(sources)},
        source_inventory_timing='Recorded after implementation and offline checks; not a historical pre-run freeze')
    (OUTPUT/'audit.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    lines = ['# CAD proposal-region inspection', '',
        'This checks labels on all 12 CAD types. It generates no grasp candidates and makes no pickup or assembly claim.', '',
        'Both contact points must belong to the permitted region before the orientation pool and 100-pose cap. '
        'The complete CAD remains in ray and collision checks. The pin goal remains 20 mm below the board top, '
        'leaving 30 mm of the 50 mm shaft exposed.', '',
        '![Permitted CAD regions](regions.png)', '',
        '| Part | Permitted proposal region | Eligible surface (mm²) | Fraction of full surface |',
        '|---|---|---:|---:|']
    for row in rows:
        lines.append(f"| {row['part']} | {row['label']} | {row['eligible_area_mm2']:.1f} | {row['eligible_area_fraction']:.1%} |")
    lines += ['', 'These initial geometric annotations identify proposal locations. They do not certify material strength, '
              'finger contact area, holding stability, insertion clearance, threading or assembly precision. '
              'The later complete-hand checks remain necessary. No runtime or quality improvement has been measured.', '',
              'The paper motivates processing parts of an object separately. The functional NIST labels and '
              'region-guided ray sampling are our adaptation; this is not a reproduction of Instant-Sem or its neural reconstruction.', '',
              '[Machine-readable inspection](audit.json) · [Paper-source record](paper_reference.json)', '']
    (OUTPUT/'README.md').write_text('\n'.join(lines), encoding='utf-8')
    table = ''.join('<tr><td>'+html.escape(row['part'])+'</td><td>'+html.escape(row['label'])+
        f"</td><td>{row['eligible_area_mm2']:.1f}</td></tr>" for row in rows)
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><title>CAD proposal regions</title>' + \
        '<style>body{font:17px system-ui;margin:28px;color:#23303a}img{width:100%;height:auto}' + \
        'table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}</style>' + \
        '<h1>CAD proposal regions</h1><p>Green surfaces supply contact proposals. The complete gray CAD remains in ray and collision checks. ' + \
        'These are region labels, not generated grasps. VLM, pickup and assembly remain paused.</p>' + \
        '<img src="regions.png" alt="Permitted proposal surfaces for twelve NIST CAD parts">' + \
        '<table><tr><th>Part</th><th>Proposal region</th><th>Eligible surface (mm²)</th></tr>' + table + '</table>' + \
        '<p>No physical holding or assembly validation. No grasp-quality or runtime improvement measured. ' + \
        'USB view shows connector detail; its complete cable remains in the CAD.</p>' + \
        '<p><a href="README.md">Detailed report</a> · <a href="audit.json">Inspection data</a></p></html>'
    (OUTPUT/'index.html').write_text(page, encoding='utf-8')
    print(json.dumps(dict(part_types=len(rows), nonempty_regions=len(rows), candidate_generation=False,
                         output=str(OUTPUT))))
    return receipt


if __name__ == '__main__':
    audit()
