"""Review six native pin CADs against the actual NIST board apertures.

This computes CAD measurements and user-specified 12 mm depth goals. It does not
generate grasps, invoke a VLM, step physics, or execute insertion. Straight
unshouldered through-pins have no CAD-defined axial seating stop; the depth is
the user's explicit task requirement, not a recovered mechanical stop.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon

from experiments.runtime_antipodal_assembly_20261023.extract_goal_audit import (
    native_mesh, source_constants, top_boundaries,
)
from experiments.fabrica_goal_geometry_20261027.task_inputs import geometry_spec


ROOT = Path(__file__).resolve().parents[2]
MESHES = ROOT / 'outputs/simulation/nist_task_board_1/assets/meshes'
OUTPUT = ROOT / 'outputs/nist_mating_goals_20261028/pins_review.json'
PARTS = ('RGOCG8-50_8mm', 'RGOCG12-50_12mm', 'RGOCG16-50_16mm',
         'KET8_Square_8mm', 'KET12_Square_12mm', 'KET16_Square_16mm')
NUMERICAL_TOL_M = 1e-8
TASK_DEPTH_M = .012
DEPTH_SOURCE = 'user_explicit_instruction_this_turn: use 12 mm then'


def _relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _polygon(boundary):
    polygon = Polygon(boundary['top_boundary_xy_m'])
    if not polygon.is_valid or polygon.area <= 0:
        raise ValueError('Expected a simple positive-area board opening')
    return polygon


def _pin_cross_section(vertices):
    # Every chamfer narrows toward the +Z tip; its full XY projection is the
    # full shank polygon. Preserve the faceted circle rather than fitting one.
    xy = vertices[:, :2]
    return xy[ConvexHull(xy).vertices]


def _fit(pin_xy, opening, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, s], [s, -c]])  # Rz(yaw) @ Rx(pi), XY block.
    center = np.asarray(opening['center_board_xy_m'])
    polygon = Polygon(pin_xy @ rotation.T + center)
    aperture = _polygon(opening)
    outside = float(polygon.difference(aperture).area)
    contained = bool(aperture.buffer(NUMERICAL_TOL_M).covers(polygon))
    clearance = float(aperture.boundary.distance(polygon.boundary)) if contained else None
    return dict(yaw_rad=float(yaw), full_shank_inside_aperture=contained,
                outside_area_m2=outside, minimum_polygon_side_clearance_m=clearance)


def _opening_yaws(opening, round_pin):
    if round_pin:
        return [0.]
    # The board's KET8/KET16 openings are slightly rotated in their source CAD.
    # Measure the longest side of the aperture's minimum-area oriented rectangle;
    # then verify containment against the actual boundary, not that rectangle.
    xy = np.asarray(_polygon(opening).minimum_rotated_rectangle.exterior.coords)[:-1]
    edges = np.roll(xy,-1,axis=0)-xy
    edge = edges[int(np.argmax(np.linalg.norm(edges,axis=1)))]
    primary = float(np.arctan2(edge[1],edge[0]))
    primary = (primary+np.pi/2) % np.pi - np.pi/2
    return [primary, primary+np.pi/2]


def _terminal_measurements(vertices):
    levels = np.unique(vertices[:, 2])
    cross_extents = []
    for level in levels:
        points = vertices[np.abs(vertices[:, 2]-level) < 1e-9]
        cross_extents.append(np.ptp(points[:, :2], axis=0))
    if (len(levels) != 3 or not np.all(cross_extents[2] < cross_extents[0])
            or not np.allclose(cross_extents[0], cross_extents[1], atol=1e-9, rtol=0)):
        raise ValueError('Pin does not have the expected single +Z chamfer')
    return dict(chamfered_end_cad='+Z', insertion_direction_cad=[0., 0., 1.],
                axial_levels_cad_m=levels.tolist(),
                cross_section_extents_xy_m=[x.tolist() for x in cross_extents],
                chamfer_length_m=float(levels[2]-levels[1]),
                axial_length_m=float(levels[-1]-levels[0]),
                shoulder_or_axial_stop_present=False)


def _wall_measurements(board_v, board_f, opening, bottom_openings):
    center = np.asarray(opening['center_board_xy_m'])
    match = min(bottom_openings, key=lambda h:np.linalg.norm(
        np.asarray(h['center_board_xy_m'])-center))
    upper, lower = _polygon(opening), _polygon(match)
    loop_difference = float(upper.symmetric_difference(lower).area)
    boundary_difference = float(upper.boundary.hausdorff_distance(lower.boundary))
    triangles = board_v[board_f]
    cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    unit = cross/np.linalg.norm(cross, axis=1)[:, None]
    xy = triangles[:, :, :2]
    lo, hi = np.array(upper.bounds[:2]), np.array(upper.bounds[2:])
    inside_box = np.all((xy >= lo-1e-8) & (xy <= hi+1e-8), axis=(1,2))
    side = inside_box & (np.abs(unit[:, 2]) < 1e-7)
    return dict(top_and_bottom_aperture_symmetric_difference_m2=loop_difference,
                top_and_bottom_aperture_boundary_hausdorff_distance_m=boundary_difference,
                local_vertical_wall_triangles=int(side.sum()),
                through_aperture_same_top_and_bottom=boundary_difference <= NUMERICAL_TOL_M,
                board_only_axial_stop_present=False)


def review():
    board_path = MESHES/'GMC_Laser_Plate_Virtual_m.stl'
    board_v, board_f = native_mesh(board_path)
    circles, other = top_boundaries(board_v, board_f)
    flipped = board_v.copy(); flipped[:, 2] *= -1
    bc, bo = top_boundaries(flipped, board_f)
    fixtures = source_constants()['FIXTURES']
    for opening in circles:
        center = np.asarray(opening['center_board_xy_m'])
        occupied = []
        for i, fixture in enumerate(fixtures):
            xy = (np.asarray(fixture[1:3])-192.)*.001
            if np.linalg.norm(xy-center) < .0002:
                occupied.append(dict(fixture_index=i, name=fixture[0]))
        opening['existing_fixed_fixtures_at_center'] = occupied
    parts = []
    for part in PARTS:
        path = MESHES/f'{part}_m.stl'
        vertices, faces = native_mesh(path)
        terminal = _terminal_measurements(vertices)
        section = _pin_cross_section(vertices)
        spec = geometry_spec(part)
        old = np.asarray(spec['goal_T_cad_board'])
        is_round = part.startswith('RGOCG')
        candidates = []
        opening_set = circles if is_round else [h for h in other if _polygon(h).area < .001]
        for index, opening in enumerate(opening_set):
            fits = [_fit(section, opening, yaw) for yaw in _opening_yaws(opening,is_round)]
            fits = [f for f in fits if f['full_shank_inside_aperture']]
            if fits:
                chosen = max(fits, key=lambda f:f['minimum_polygon_side_clearance_m'])
                candidates.append(dict(opening=opening,
                    opening_id=opening.get('audit_hole_id',f'noncircular_{index:02d}'),
                    aperture_area_m2=_polygon(opening).area, **chosen))
        free = [c for c in candidates if not c['opening'].get('existing_fixed_fixtures_at_center')]
        if not free:
            raise ValueError(f'No unobstructed compatible aperture for {part}')
        selected = min(free, key=lambda c:c['aperture_area_m2'])
        opening = selected['opening']
        center = np.asarray(opening['center_board_xy_m'])
        yaw = selected['yaw_rad']
        old_yaw = float(np.arctan2(old[1,0],old[0,0]))
        # A rectangular section is unchanged by a half-turn. Choose the
        # equivalent key orientation nearest the inherited yaw for readability.
        if not is_round:
            yaw += np.rint((old_yaw-yaw)/np.pi)*np.pi
        c,s=np.cos(yaw),np.sin(yaw)
        rotation = np.array([[c,s,0.],[s,-c,0.],[0.,0.,-1.]])
        board_top = float(spec['board_top_m'])
        task_depth = TASK_DEPTH_M  # Explicit user-defined endpoint, not a CAD stop.
        display = np.eye(4); display[:3,:3] = rotation
        display[:3,3] = [*center, board_top-task_depth+vertices[:,2].max()]
        part_frame = np.eye(4); part_frame[:3,3] = [0.,0.,vertices[:,2].max()]
        receiver_frame = np.eye(4); receiver_frame[:3,:3] = rotation
        receiver_frame[:3,3] = [*center,board_top]
        local_seated = np.eye(4); local_seated[2,3] = task_depth
        np.testing.assert_allclose(receiver_frame@local_seated@np.linalg.inv(part_frame),display,atol=1e-12)
        wall = _wall_measurements(board_v,board_f,opening,bc+bo)
        if not wall['through_aperture_same_top_and_bottom']:
            raise ValueError('Top/bottom aperture shapes differ')
        original = MESHES/f'{part}.STL'; original_v,_ = native_mesh(original)
        original_board_v,_ = native_mesh(MESHES/'GMC_Laser_Plate_Virtual.STL')
        original_evidence = dict(original_pin_bounds_mm=[original_v.min(0).tolist(),original_v.max(0).tolist()],
            original_board_bounds_mm=[original_board_v.min(0).tolist(),original_board_v.max(0).tolist()],
            original_pin_is_individually_origin_based=True,
            original_export_does_not_supply_an_assembled_pin_to_board_transform=True)
        all_candidates = []
        for candidate in candidates:
            value = {k:v for k,v in candidate.items() if k != 'opening'}
            value.update(center_board_xy_m=candidate['opening']['center_board_xy_m'],
                existing_fixed_fixtures_at_center=candidate['opening'].get('existing_fixed_fixtures_at_center',[]))
            all_candidates.append(value)
        warnings = [
            'No shoulder or bottom stop defines a unique seated depth in the board CAD.',
            'The 12 mm tip depth is explicitly chosen by the user as the task endpoint; it is not a CAD-derived hard stop.',
            'CAD clearance is checked; physical manufacturing tolerances and successful insertion remain unverified.',
            'Hole assignment uses free compatible CAD geometry; official task destination semantics are not independently supplied.',
            'Board/world Z placement and fixed-fixture mounting remain the existing nominal scene conventions.',
            'A support surface below the board could limit travel, but its placement is not an axial stop encoded in the receiver CAD.'
        ]
        if part=='RGOCG12-50_12mm':
            warnings.append('A tighter 12.199668 mm circular opening exists, but its center is occupied by the fixed M12 screw; retain the free 12.991029 mm aperture as a loose-clearance review destination.')
        measurements=dict(**terminal,**wall,**original_evidence,
            selected_opening_id=selected['opening_id'],
            selected_opening_center_board_xy_m=center.tolist(),
            selected_opening_extent_xy_m=opening['extent_xy_m'],
            selected_aperture_aligned_yaw_rad=yaw,
            selected_aperture_aligned_yaw_deg=float(np.rad2deg(yaw)),
            aperture_yaw_method='long side of minimum-area oriented rectangle; containment checked against actual CAD boundary',
            selected_opening_boundary_xy_m=opening['top_boundary_xy_m'],
            selected_minimum_polygon_side_clearance_m=selected['minimum_polygon_side_clearance_m'],
            full_shank_projection_outside_selected_aperture_m2=selected['outside_area_m2'],
            compatible_apertures=all_candidates,
            board_thickness_m=float(np.ptp(board_v[:,2])),
            chosen_task_depth_m=task_depth,
            depth_source=DEPTH_SOURCE,
            tip_protrusion_below_board_underside_m=float(task_depth-np.ptp(board_v[:,2])),
            display_engagement_depth_m=task_depth,
            display_depth_status='user_chosen_task_endpoint_12mm_below_board_top',
            depth_parameter='D = positive distance from board top to chamfered tip along world -Z',
            full_shank_spans_entire_board_depth_interval_m=[float(np.ptp(board_v[:,2])+terminal['chamfer_length_m']),terminal['axial_length_m']],
            interval_note='Board-only geometry: chamfer entirely below bottom, rear end at/above top; not a commanded endpoint or required assembly depth.',
            contact_seating_equality='chamfered_tip_world_z = board_top_world_z - 0.012 m',
            seating_equality_source='user task-depth requirement, not contact with an axial stop')
        parts.append(dict(part=part,part_mesh=_relative(path),
            receivers=[{**r,'mesh_path':_relative(r['mesh_path']),
                        'board_T_mesh':np.asarray(r['board_T_mesh']).tolist()} for r in spec['receivers']],
            old_board_T_part=old.tolist(),review_board_T_part=display.tolist(),
            proposed_board_T_part=display.tolist(),axis_direction_cad=[0.,0.,1.],axis_direction_board=[0.,0.,-1.],
            part_mating_frame_cad=part_frame.tolist(),receiver_mating_frame_board=receiver_frame.tolist(),
            mating_transform_parameterization=dict(equation='board_T_part(D) = board_T_receiver_mating @ translate(0,0,D) @ inverse(part_T_part_mating)',
                parameter='D',chosen_parameter_value_m=task_depth,depth_source=DEPTH_SOURCE,
                units='metres',rotation_and_transverse_alignment_CAD_supported=True,
                axial_seated_position_defined=True,axial_stop_CAD_defined=False),
            status='CAD_direction_clearance_checked_user_depth_12mm',measurements=measurements,
            changes=['Reverse the pin so the modeled +Z chamfer faces downward into the aperture.',
                     'Align rectangular pin cross-section axes with the measured aperture orientation, including the small KET8/KET16 CAD rotations.',
                     'Set the final chamfered tip to 12 mm below the board top as explicitly required by the user; preserve that this depth is not derived from a CAD stop.'],
            warnings=warnings,evidence=[dict(path=_relative(p),sha256=_hash(p)) for p in
                [path,original,board_path,MESHES/'GMC_Laser_Plate_Virtual.STL',ROOT/'simulation/nist_task_board_1.py']]))
    result=dict(scope='CAD mating-direction and receiver compatibility review only',
        frame='board-centered XY; original nominal scene world Z; metres',
        grasp_generation=False,VLM_selection=False,pickup_execution=False,assembly_execution=False,
        simulator_object_state_read=False,parts=parts)
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    result=review()
    for row in result['parts']:
        m=row['measurements']
        print(row['part'],m['selected_opening_id'],
              f"clearance={m['selected_minimum_polygon_side_clearance_m']*1000:.6f} mm",
              'direction=+CAD Z to world -Z; user task depth=12 mm')
