"""Static full-hand height audit of saved observed 8 mm pin alignments.

No candidate bank, selector, simulator state, motion or execution is used. The
declared pose uses ideal cylinder equator contacts, the existing near-tip pad
anchors, and a straight-down hand. Its optimistic axial-location bound is not
a claim that a protected/contact/receiver-valid pose exists there.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.cad_grasp_selection_20261028.geometry import (
    component_vertices, pad_calibration, rigid_transform, transform_points,
)
from experiments.cad_grasp_selection_20261029.hand_frame import fingertip_calibration
from experiments.cad_grasp_selection_20261029.sampler import pose_from_pair

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT/'outputs/cad_grasp_geometry_20261026/inputs.json'
HAND = ROOT/'outputs/cad_grasp_geometry_20261026/hand_model.json'
REGISTRY = ROOT/'outputs/nist_scene_corrections_20261029/assembly_goals.json'
PART = 'RGOCG8-50_8mm'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_case(row, hand, *, axis_point=(0., 0., 0.), axis_direction=(0., 0., 1.),
               radius_m=.004, exposed_axial_interval_m=(-.025, .005),
               table_clearance_m=.0001):
    """Measure every unchanged hand vertex for one declared ideal side grip.

    Translating the contact midpoint by lambda along the pin axis translates
    every hand vertex by that same vector. Therefore table clearance has the
    exact affine form h(lambda)=h(0)+lambda*(up dot axis_world). The endpoint
    maximizing it over the declared exposed interval is an optimistic bound;
    it ignores pad footprint, chamfers and all receiver/contact constraints.
    """
    pickup = rigid_transform(row['world_T_cad'], 'observed pickup alignment')
    plane = np.asarray(row['table_plane_world'], float)
    axis_point, axis = np.asarray(axis_point, float), np.asarray(axis_direction, float)
    interval = np.asarray(exposed_axial_interval_m, float)
    if (plane.shape != (4,) or not np.isfinite(plane).all()
            or not np.isclose(np.linalg.norm(plane[:3]), 1., atol=1e-8)
            or axis_point.shape != (3,) or axis.shape != (3,)
            or not np.isfinite(axis_point).all() or not np.isfinite(axis).all()
            or not np.isclose(np.linalg.norm(axis), 1., atol=1e-8)
            or interval.shape != (2,) or not np.isfinite(interval).all()
            or interval[1] < interval[0] or not np.isfinite(radius_m) or radius_m <= 0):
        raise ValueError('Static audit needs finite unit table/axis data and a metric exposed interval')
    up, axis_world = plane[:3], pickup[:3, :3]@axis
    closing_world = np.cross(up, axis_world)
    if np.linalg.norm(closing_world) <= 1e-12:
        reference = np.eye(3)[np.argmin(np.abs(up))]
        closing_world = reference-(reference@up)*up
    closing_world /= np.linalg.norm(closing_world)
    closing_cad = pickup[:3, :3].T@closing_world
    approach_cad = pickup[:3, :3].T@(-up)
    contacts = axis_point+np.array([-1., 1.])[:, None]*radius_m*closing_cad
    declared_pose = pose_from_pair(contacts, approach_cad, hand)
    grip_pose = np.asarray(declared_pose['cad_T_grip_site'])
    calibration, tip = pad_calibration(hand), fingertip_calibration(hand)
    held = declared_pose['holding_opening_m']
    # This is the current pre-opening convention: 2 mm additional full pad gap.
    opened = min(float(calibration['opening_limits_m'][1]),
                 held+.002/calibration['gap_per_opening'])
    component_rows = []
    for component in hand['components']:
        vertices = np.concatenate([component_vertices(component, q) for q in (held, opened)])
        world = transform_points(transform_points(vertices, grip_pose), pickup)
        height = world@up+plane[3]
        component_rows.append(dict(component=component['name'], is_pad=component['is_pad'],
            minimum_table_clearance_m=float(height.min()), vertex_count=len(component['vertices_zero_m']),
            vertices_evaluated=len(vertices)))
    minimum = min(component_rows, key=lambda item:item['minimum_table_clearance_m'])
    contact_height = transform_points(contacts, pickup)@up+plane[3]
    slope = float(axis_world@up)
    best_lambda = float(interval[1] if slope >= 0 else interval[0])
    optimistic = minimum['minimum_table_clearance_m']+best_lambda*slope
    return dict(scene=row.get('scene'), part=row.get('part', PART),
        pose_source=row.get('pose_source', 'caller-supplied observed CAD alignment'),
        simulator_object_poses_read=False, candidate_generation_run=False,
        observed_axis_world=axis_world.tolist(), axis_vertical_component=slope,
        approach_world=(-up).tolist(), closing_world=closing_world.tolist(),
        declared_axial_contact_location_m=0., ideal_radius_m=float(radius_m),
        ideal_contact_heights_above_table_m=contact_height.tolist(),
        holding_opening_m=float(held), pre_opening_m=float(opened),
        contact_inset_from_pad_tip_m=tip['contact_inset_m'],
        full_hand_distal_extent_beyond_pad_tip_m=tip['full_hand_distal_extent_beyond_pad_tip_m'],
        full_hand_distal_extent_beyond_contact_anchor_m=float(
            tip['full_hand_distal_extent_beyond_pad_tip_m']+tip['contact_inset_m']),
        declared_pose_complete_hand_minimum_clearance_m=minimum['minimum_table_clearance_m'],
        limiting_component=minimum['component'], components=component_rows,
        table_clearance_required_m=float(table_clearance_m),
        declared_pose_table_clear=bool(minimum['minimum_table_clearance_m'] >= table_clearance_m),
        optimistic_exposed_axis_interval_m=interval.tolist(),
        optimistic_best_axial_location_m=best_lambda,
        optimistic_best_complete_hand_clearance_m=float(optimistic),
        optimistic_bound_can_clear_table=bool(optimistic >= table_clearance_m),
        bound_scope='ideal equator side grip with straight-down approach; whole original hand; axial translation only',
        bound_does_not_establish=['source-mesh contact', 'complete-pad protected clearance',
            'target closing clearance', 'receiver clearance', 'grasp validity', 'pickup', 'assembly'])


def audit_saved_hand():
    inputs, hand, registry = [json.loads(path.read_text(encoding='utf-8'))
                              for path in (INPUTS, HAND, REGISTRY)]
    rows = [row for row in inputs['records'] if row['part'] == PART]
    if len(rows) != 8 or {row['scene'] for row in rows} != {f'N{i:02d}' for i in range(8)}:
        raise ValueError('Static audit expects exactly the eight saved 8 mm pin observed alignments')
    part = next(row for row in registry['parts'] if row['part'] == PART)
    # Native pin is 50 mm long, with its positive-Z end inserted by the declared
    # 20 mm target. The raw part CAD is read only to measure these axis bounds.
    from experiments.cad_grasp_geometry_20261026.gallery import read_stl
    mesh_path = ROOT/part['part_mesh']
    if _sha(mesh_path) != part['part_mesh_sha256']:
        raise ValueError('Static audit pin CAD does not match the corrected task registry')
    vertices, _ = read_stl(mesh_path)
    axis, origin = np.asarray(part['axis_direction_cad'], float), np.asarray(part['axis_point_cad_m'], float)
    axial = (vertices-origin)@axis
    protected = part['protected_contact_region']
    if (protected['type'] != 'halfspace' or
            not np.allclose(protected['normal_cad'], axis, atol=1e-12, rtol=0)):
        raise ValueError('Static audit expects the reviewed positive-axis inserted-end label')
    exposed = [float(axial.min()), float(-protected['offset_m']-origin@axis)]
    records = [audit_case(row, hand, axis_point=origin, axis_direction=axis,
                         exposed_axial_interval_m=exposed) for row in rows]
    tip = fingertip_calibration(hand)
    return dict(study='static full-hand height audit of eight saved 8 mm pin CAD alignments',
        method='one ideal equator pose per observed alignment, plus exact affine axial-translation upper bound',
        candidate_generation_run=False, simulator_started=False, simulator_object_poses_read=False,
        pickup_assembly_vlm_run=False, geometry_modified=False, independent_depth_offsets_m=[],
        source_sha256={str(path.relative_to(ROOT)).replace('\\', '/'):_sha(path)
                       for path in (INPUTS, HAND, REGISTRY, mesh_path)},
        usable_pad_tip_grip_site_m=tip['usable_pad_tip_origin_grip_site_m'].tolist(),
        contact_anchors_grip_site_m=tip['contact_anchors_zero_grip_site_m'].tolist(),
        physical_fingertip_origin_grip_site_m=tip['fingertip_origin_grip_site_m'].tolist(),
        full_hand_distal_extent_beyond_pad_tip_m=tip['full_hand_distal_extent_beyond_pad_tip_m'],
        nominal_8mm_pin_equator_height_m=.004,
        nominal_lying_pin_clearance_with_current_anchors_m=.004-(
            tip['full_hand_distal_extent_beyond_pad_tip_m']+tip['contact_inset_m']),
        pin_tip_depth_target_m=part['pin_tip_depth_below_board_top_m'],
        records=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    text = json.dumps(audit_saved_hand(), indent=2, allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
