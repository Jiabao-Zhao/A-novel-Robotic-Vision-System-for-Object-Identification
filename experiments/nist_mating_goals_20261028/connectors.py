"""Read-only connector/receiver CAD review; no grasp or motion generation.

Recover useful source-export relations, check the frozen USB frame mapping,
and retain unresolved seating/interference as unresolved. Sparse mesh sections
are measured evidence, not a continuous three-dimensional collision proof.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
from shapely.ops import unary_union

from experiments.fabrica_goal_geometry_20261027.task_inputs import geometry_spec
from experiments.cad_grasp_geometry_20261026.geometry_kernel import MeshCollider
from .cad_tools import (mesh_component_labels, read_native_mesh,
                        section_geometry, transform_points)


ROOT = Path(__file__).resolve().parents[2]
MESHES = ROOT / 'outputs/simulation/nist_task_board_1/assets/meshes'
OUTPUT = ROOT / 'outputs/nist_mating_goals_20261028/connectors_review.json'
USB_FROZEN = (ROOT / 'outputs/datasets/nist_task_board/legacy_workspace_inputs'
              '/usb-cable-workspace/usb_connector_with_cable.stl')
SOURCE_FLIP = np.diag([1., -1., -1.])


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def _pose(rotation=None, translation=None):
    result = np.eye(4)
    if rotation is not None:
        result[:3, :3] = rotation
    if translation is not None:
        result[:3, 3] = translation
    return result


def _frame(origin, direction, x_hint=(1., 0., 0.)):
    z = np.asarray(direction, float)
    z /= np.linalg.norm(z)
    x = np.asarray(x_hint, float)
    x -= z * np.dot(x, z)
    x /= np.linalg.norm(x)
    return _pose(np.column_stack((x, np.cross(z, x), z)), origin)


def source_export_relation(base):
    """Same source coordinate system is evidence only for BNC and catalog USB.

    Both source files in these two pairs received the same Rx(pi) conversion.
    Their original bbox centers recover the converted female-to-male relation.
    DSUB and Waterproof originals are not assumed to describe a seated assembly.
    """
    if base not in {'BNC', 'USB'}:
        raise ValueError('Only the inspected BNC and USB source pairs are used')
    centers = {}
    records = []
    for sex in ('Male', 'Female'):
        path = MESHES / f'{base}_{sex}.STL'
        vertices, _ = read_native_mesh(path)
        centers[sex] = (vertices.min(0) + vertices.max(0)) / 2
        records.append(dict(path=_relative(path), sha256=_sha(path),
                            source_bbox_center_mm=centers[sex].tolist()))
    delta = SOURCE_FLIP @ (centers['Male']-centers['Female']) * .001
    return _pose(translation=delta), records


def frozen_usb_mapping():
    """Prove catalog-to-frozen rigid mapping on all catalog vertices.

    The frozen file adds one cable component. Its five connector components
    have the same triangle counts as the catalog; no scaling or fitting occurs.
    """
    catalog = MESHES / 'USB_Male_m.stl'
    original, faces = read_native_mesh(catalog)
    frozen, frozen_faces = read_native_mesh(USB_FROZEN)
    mapping = _pose(np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]),
                    [0., 0., .00425])
    mapped = transform_points(original, mapping)
    distances = cKDTree(frozen).query(mapped)[0]
    if distances.max() > 1e-8:
        raise ValueError('Frozen USB no longer matches the inspected catalog mapping')
    return mapping, dict(
        catalog_to_frozen=mapping.tolist(), units='metres',
        catalog_vertices_checked=len(original),
        maximum_catalog_vertex_to_frozen_distance_m=float(distances.max()),
        catalog_triangle_count=len(faces), frozen_triangle_count=len(frozen_faces),
        catalog_triangle_component_counts=np.bincount(mesh_component_labels(faces)).tolist(),
        frozen_triangle_component_counts=np.bincount(mesh_component_labels(frozen_faces)).tolist(),
        matched_scope='all catalog connector vertices; extra frozen cable is not in catalog',
        no_scale_change=True)


def _section_audit(male_vertices, male_faces, female_vertices, female_faces,
                   female_T_male, axis, levels, *, classify_contact_disks=False):
    male = transform_points(male_vertices, female_T_male)
    male_labels = mesh_component_labels(male_faces)
    female_labels = mesh_component_labels(female_faces)
    sections = []
    for level in levels:
        a, ar = section_geometry(male, male_faces, axis, float(level),
                                component_labels=male_labels, return_report=True)
        b, br = section_geometry(female_vertices, female_faces, axis, float(level),
                                component_labels=female_labels, return_report=True)
        complete = bool(ar['complete'] and br['complete'])
        record = dict(level_female_m=float(level), complete=complete,
            overlap_area_m2=float(a.intersection(b).area) if complete else None,
            male_section_area_m2=float(a.area), female_section_area_m2=float(b.area))
        if classify_contact_disks and complete:
            polys = [a] if a.geom_type == 'Polygon' else getattr(a, 'geoms', [])
            contact_disks = unary_union([p for p in polys if p.geom_type == 'Polygon'
                                        and 1e-8 < p.area < 1e-6])
            record['overlap_outside_small_contact_disks_area_m2'] = float(
                a.intersection(b).difference(contact_disks).area)
        sections.append(record)
    areas = [s['overlap_area_m2'] for s in sections if s['complete']]
    witness = None
    if areas and max(areas) > 1e-12:
        row = max((s for s in sections if s['complete']), key=lambda s:s['overlap_area_m2'])
        a = section_geometry(male, male_faces, axis, row['level_female_m'],
                             component_labels=male_labels)
        b = section_geometry(female_vertices, female_faces, axis, row['level_female_m'],
                             component_labels=female_labels)
        point = a.intersection(b).representative_point()
        xyz = np.zeros(3)
        xyz[axis] = row['level_female_m']
        xyz[[i for i in range(3) if i != axis]] = [point.x, point.y]
        native = transform_points(xyz[None], np.linalg.inv(female_T_male))
        male_collider = MeshCollider(male_vertices, male_faces)
        female_collider = MeshCollider(female_vertices, female_faces)
        witness = dict(point_female_m=xyz.tolist(),
            inside_closed_male_mesh=bool(male_collider.contains_points(native)[0]),
            inside_closed_female_mesh=bool(female_collider.contains_points(xyz[None])[0]),
            method='independent three-direction 3D ray parity on native closed mesh components',
            significance='one interior point in both rigid modeled solids proves overlap at this review pose; not a physical contact/compliance failure')
    return dict(section_axis_female=axis, sections=sections,
                complete_sections=sum(s['complete'] for s in sections),
                positive_overlap_sections=sum(a > 1e-12 for a in areas),
                maximum_sampled_overlap_area_m2=max(areas, default=None),
                positive_overlap_3D_witness=witness,
                area_reporting_threshold_m2=1e-12,
                no_mesh_changes=True,
                scope='finite 2D section samples, not a continuous 3D collision certificate')


def _component_measurements(vertices, faces):
    labels = mesh_component_labels(faces)
    result = []
    for component in np.unique(labels):
        tri = vertices[faces[labels == component]]
        volume = np.einsum('ij,ij->i', tri[:, 0],
                           np.cross(tri[:, 1], tri[:, 2])).sum() / 6
        result.append(dict(component=int(component), triangles=len(tri),
                           signed_volume_m3=float(volume),
                           bounds_m=[tri.reshape(-1, 3).min(0).tolist(),
                                     tri.reshape(-1, 3).max(0).tolist()]))
    return result


def _dsub_contact_sizes(mv, mf, fv, ff):
    """Measure the contact disks/socket holes instead of declaring a pose error."""
    male = section_geometry(mv, mf, 2, .02066125,
                            component_labels=mesh_component_labels(mf))
    female = section_geometry(fv, ff, 2, .020123,
                              component_labels=mesh_component_labels(ff))
    m_polys = [male] if male.geom_type == 'Polygon' else list(male.geoms)
    f_polys = [female] if female.geom_type == 'Polygon' else list(female.geoms)
    pins = [g for g in m_polys if g.geom_type == 'Polygon' and 5e-7 < g.area < 1e-6]
    holes = [Polygon(ring) for g in f_polys if g.geom_type == 'Polygon'
             for ring in g.interiors if 5e-7 < Polygon(ring).area < 1e-6]
    return dict(male_section_z_m=.02066125, female_section_z_m=.020123,
        male_contact_disk_count=len(pins), female_socket_hole_count=len(holes),
        mean_male_contact_area_m2=float(np.mean([p.area for p in pins])),
        mean_female_socket_area_m2=float(np.mean([p.area for p in holes])),
        male_area_equivalent_diameter_m=float(np.mean([2*np.sqrt(p.area/np.pi) for p in pins])),
        female_area_equivalent_diameter_m=float(np.mean([2*np.sqrt(p.area/np.pi) for p in holes])),
        interpretation='rigid modeled contact disks are larger than socket voids; intended electrical-contact/compliance is not represented or verified')


def review_connectors():
    records = []
    for part in ('BNC_Male', 'DSUB_Male', 'Waterproof_Male', 'USB_Male'):
        base = part.split('_')[0]
        spec = geometry_spec(part)
        male_path = USB_FROZEN if base == 'USB' else MESHES / f'{part}_m.stl'
        female_path = MESHES / f'{base}_Female_m.stl'
        mv, mf = read_native_mesh(male_path)
        fv, ff = read_native_mesh(female_path)
        receiver = next(r for r in spec['receivers']
                        if Path(r['mesh_path']).name == female_path.name)
        board_T_female = np.asarray(receiver['board_T_mesh'])
        measurements = dict(male_components=_component_measurements(mv, mf),
                            female_components=_component_measurements(fv, ff))
        warnings = [
            'Receiver mounting is the existing CAD scene convention, not measured physical board calibration.',
            'Static section checks do not verify latch, spring, electrical-contact or compliant engagement.'
        ]
        evidence = [dict(path=_relative(male_path), sha256=_sha(male_path)),
                    dict(path=_relative(female_path), sha256=_sha(female_path))]
        if base in {'BNC', 'USB'}:
            catalog_relation, source_records = source_export_relation(base)
            evidence += source_records
            if base == 'USB':
                frozen_T_catalog, mapping = frozen_usb_mapping()
                relation = catalog_relation @ np.linalg.inv(frozen_T_catalog)
                direction = [-1., 0., 0.]
                tip = [float(mv[:, 0].min()), 0., .00425]
                part_frame = _frame(tip, direction, (0., 1., 0.))
                measurements['catalog_to_frozen_mapping'] = mapping
                status = 'CAD_supported_source_pose_with_verified_USB_frame_mapping'
                warnings += ['25 complete axial sections show no overlap; this is not an exhaustive 3D clearance proof.']
                changes = ['Replace the inherited identity rotation with the verified frozen-to-catalog frame mapping.',
                           'Recover the catalog male/female source-export relationship instead of bounding-plane contact.']
            else:
                relation = catalog_relation
                direction = [0., 0., -1.]
                tip = [0., 0., float(mv[:, 2].min())]
                part_frame = _frame(tip, direction)
                status = 'source_relationship_recovered_but_rigid_CAD_interference_unresolved'
                warnings += ['Recovered source-export placement has positive solid section overlap; do not treat it as a validated seated goal.',
                             'Both meshes have one positive-volume edge-connected component; overlap is not from unioning a separate inward cavity shell.']
                changes = ['Keep the observed downward mating direction.',
                           'Show the recovered 10.5 mm source-export axial overlap for review; do not promote it to an executable seated goal.']
            transformed = transform_points(mv, relation)
            low = max(float(transformed[:, 2].min()), float(fv[:, 2].min()))
            high = min(float(transformed[:, 2].max()), float(fv[:, 2].max()))
            levels = np.linspace(low+.00001, high-.00001, 25)
            axis = 2
            receiver_frame = _frame([0., 0., float(fv[:, 2].max())], [0., 0., 1.])
            measurements['source_female_T_catalog_male'] = catalog_relation.tolist()
            measurements['axial_bbox_overlap_m'] = high-low
        elif base == 'DSUB':
            # The key outline matches after Ry(pi); it must not be flipped in Y.
            # The female front insert is offset +0.18 mm in X in its bbox frame.
            # Male insert shoulder z=19.2 meeting female front z=21.9 is a
            # review datum, not an independently established seating depth.
            mb = next(c for c in measurements['male_components'] if c['bounds_m'][0][2] < 0)
            front = next(c for c in measurements['female_components'] if c['bounds_m'][0][2] > 0)
            front_center_x = np.asarray(front['bounds_m']).mean(0)[0]
            male_shoulder = mb['bounds_m'][1][2]
            female_front = float(fv[:, 2].max())
            relation = _pose(np.diag([-1., 1., -1.]),
                             [front_center_x, 0., male_shoulder+female_front])
            direction = [0., 0., 1.]
            part_frame = _frame([0., 0., float(mv[:, 2].max())], direction)
            receiver_frame = _frame([front_center_x, 0., female_front], [0., 0., 1.])
            axis = 2
            levels = np.linspace(.01931, .02189, 17)
            status = 'direction_and_key_outline_corrected_seating_interference_unresolved'
            changes = ['Turn male +Z toward receiver -Z with Ry(pi); preserve the D-shaped key orientation.',
                       'Account for the receiver front insert X offset; remove the old lower-plane touching target.']
            measurements['review_alignment'] = dict(
                female_insert_center_x_m=float(front_center_x),
                female_insert_center_method='bbox center of the separate front insert component',
                male_insert_shoulder_z_m=float(male_shoulder),
                male_shoulder_method='maximum Z of the separate rear/body component',
                female_front_z_m=female_front,
                axial_bbox_overlap_m=.0026,
                axial_position_is_review_datum_not_verified_stop=True)
            warnings += ['The two source exports have unrelated offsets and are not accepted as an assembled pair.',
                         'The shown insert-face contact position still has section interference; exact pin/socket fit and seated stop remain unresolved.',
                         'Contact interference can require an explicit compliant-contact interpretation; do not infer that the physical connector cannot mate.']
            measurements['contact_size_comparison'] = _dsub_contact_sizes(mv, mf, fv, ff)
        else:
            # Female sockets open along +Y, while male front geometry is +Z.
            # Additional roll aligns the asymmetric key outline better than
            # Rx(pi/2) alone. This does not solve the observed end interference.
            rotation = np.array([[-1., 0., 0.], [0., 0., -1.], [0., -1., 0.]])
            relation = _pose(rotation, [.0001, .0215, -.00025])
            direction = [0., 0., 1.]
            part_frame = _frame([.0001, -.00025, float(mv[:, 2].max())], direction)
            receiver_frame = _frame([0., float(fv[:, 1].max()), 0.], [0., 1., 0.])
            axis = 1
            levels = np.array([.00381, .004123, .005123, .008123, .012123, .015123, .01674])
            status = 'receiver_axis_and_key_orientation_corrected_seating_interference_unresolved'
            changes = ['Use the receiver +Y socket opening; the previous assumption of a vertical +Z receiver was wrong.',
                       'Map male +Z toward receiver -Y and align the asymmetric key pattern.',
                       'Show a shoulder-to-mouth review position; do not invent a verified seated stop.']
            measurements['review_alignment'] = dict(
                female_socket_outward_axis=[0., 1., 0.],
                female_front_y_m=float(fv[:, 1].max()),
                male_shoulder_z_m=.00465,
                transverse_contact_array_center_offset_m=[.0001, -.00025],
                axial_position_is_review_datum_not_verified_stop=True)
            warnings += ['Male and female originals are independent local exports, not a recoverable assembled source pose.',
                         'Axial seating and interfering modeled front/key features remain unresolved under rigid CAD.',
                         'The current receiver mounting makes the mating approach horizontal; a downward-only assumption would require correcting the receiver mounting too.']
        audit = _section_audit(mv, mf, fv, ff, relation, axis, levels,
                               classify_contact_disks=base == 'DSUB')
        measurements['section_audit'] = audit
        if base == 'DSUB':
            measurements['sampled_contact_overlap_classification'] = dict(
                maximum_overlap_outside_small_contact_disks_area_m2=max(
                    s['overlap_outside_small_contact_disks_area_m2']
                    for s in audit['sections'] if s['complete']),
                disk_area_range_m2=[1e-8, 1e-6],
                interpretation='sampled interference is confined to the contact disks, not the keyed shell/body; seating-stop semantics and physical contact compliance remain unverified')
        board_T_part = board_T_female @ relation
        board_direction = board_T_part[:3, :3] @ np.asarray(direction)
        board_frame = board_T_female @ receiver_frame
        supported = base == 'USB' and audit['complete_sections'] == len(levels) and audit['positive_overlap_sections'] == 0
        posed = transform_points(mv, board_T_part)
        measurements['minimum_part_z_minus_board_top_m'] = float(posed[:, 2].min()-spec['board_top_m'])
        if base == 'Waterproof':
            warnings += ['At this unresolved review pose the male extends 0.75 mm below the board-top plane; existing receiver mounting and body/board clearance need review.']
        records.append(dict(
            part=part, part_mesh=_relative(male_path),
            receivers=[dict(name=r['name'], mesh_path=_relative(r['mesh_path']),
                            board_T_mesh=np.asarray(r['board_T_mesh']).tolist())
                       for r in spec['receivers']],
            old_board_T_part=np.asarray(spec['goal_T_cad_board']).tolist(),
            review_board_T_part=board_T_part.tolist(),
            proposed_board_T_part=board_T_part.tolist() if supported else None,
            axis_direction_cad=direction, axis_direction_board=board_direction.tolist(),
            part_mating_frame_cad=part_frame.tolist(),
            receiver_mating_frame_board=board_frame.tolist(),
            review_receiver_T_part=relation.tolist(),
            review_receiver_mating_T_part_mating=(np.linalg.inv(receiver_frame) @ relation @ part_frame).tolist(),
            status=status, CAD_supported_review_goal=supported,
            verified_physical_seated_goal=False,
            review_pose_is_not_validated_goal=not supported,
            measurements=measurements, changes=changes, warnings=warnings,
            evidence=evidence))
    return dict(units='metres', frame='board-centred XY; original world Z',
                no_grasp_generation=True, no_VLM_selection=True,
                no_robot_or_physics_execution=True,
                original_results_modified=False, parts=records)


def main():
    result = review_connectors()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    for record in result['parts']:
        audit = record['measurements']['section_audit']
        print(record['part'], record['status'],
              'max_sampled_overlap_mm2=', audit['maximum_sampled_overlap_area_m2']*1e6)


if __name__ == '__main__':
    main()
