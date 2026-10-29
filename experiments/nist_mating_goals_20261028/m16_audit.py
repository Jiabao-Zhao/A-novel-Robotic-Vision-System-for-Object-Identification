"""Reproduce the saved M16 mounting and sampled thread-phase CAD inspection.

This records the numerical inspection that produced ``m16_thread_audit.json``.
It reads existing CAD and nominal fixture transforms; it does not update the
simulation layout, generate grasps, call a VLM, or run robot/physics execution.
Sparse complete sections can witness represented-solid interference, but cannot
certify a fully seated threaded fit. The local yaw search is not a global fit.

Run from the repository root with the existing NumPy/SciPy/Shapely environment:

    python -B -m experiments.nist_mating_goals_20261028.m16_audit

The script was saved after the original inspection for reproducibility; its
creation does not retroactively make it a pre-run source lock.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from shapely import affinity

from experiments.fabrica_goal_geometry_20261027.task_inputs import geometry_spec
from experiments.nist_mating_goals_20261028.cad_tools import (
    mesh_component_labels, read_native_mesh, section_geometry, transform_points,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / 'outputs/nist_mating_goals_20261028/m16_thread_audit.json'


def measure():
    """Return the saved audit schema, in metres, square metres and degrees."""
    base = ROOT / 'outputs/simulation/nist_task_board_1/assets/meshes'
    meshes = {}
    for name in ['M16_Hex_Nut', 'M16_Screw', 'GMC_Laser_Plate_Virtual']:
        vertices, faces = read_native_mesh(base / f'{name}_m.stl')
        meshes[name] = (vertices, faces, mesh_component_labels(faces))

    spec = geometry_spec('M16_Hex_Nut')
    receivers = {item['name']: item for item in spec['receivers']}
    board_T = np.array(receivers['board']['board_T_mesh'])
    screw_T = np.array(receivers['fixture_7_M16_Screw']['board_T_mesh'])
    old_nut_T = np.array(spec['goal_T_cad_board'])
    board_top = spec['board_top_m']
    thickness = float(np.ptp(meshes['GMC_Laser_Plate_Virtual'][0][:, 2]))
    board_bottom = board_top-thickness
    corrected_screw_T = screw_T.copy()
    corrected_screw_T[2, 3] -= thickness

    original_nut, _ = read_native_mesh(base / 'M16_Hex_Nut.STL')
    original_screw, _ = read_native_mesh(base / 'M16_Screw.STL')
    nut_center = (original_nut.min(0)+original_nut.max(0))/2
    screw_center = (original_screw.min(0)+original_screw.max(0))/2
    source_delta = (nut_center-screw_center)*[1, -1, -1]*.001
    # Exact planes measured in the original mm exports. The unchanged board STL
    # has the same 8.991600036621094 mm thickness as the gap between these planes.
    original_board_thickness = 8.991600036621094*.001
    source_shoulder_z = 93.70751953125
    source_nut_seat_z = float(original_nut[:, 2].max())

    board_vertices = transform_points(meshes['GMC_Laser_Plate_Virtual'][0], board_T)
    mount_sections = []
    for z in board_bottom+thickness*np.array([.1, .25, .5, .75, .9]):
        board_geometry, board_report = section_geometry(
            board_vertices, meshes['GMC_Laser_Plate_Virtual'][1], 2, float(z),
            component_labels=meshes['GMC_Laser_Plate_Virtual'][2], return_report=True)
        row = {'board_z_m': float(z), 'board_section_complete': board_report['complete']}
        for name, transform in [('old', screw_T), ('shoulder_at_board_bottom', corrected_screw_T)]:
            screw_vertices = transform_points(meshes['M16_Screw'][0], transform)
            screw_geometry, screw_report = section_geometry(
                screw_vertices, meshes['M16_Screw'][1], 2, float(z),
                component_labels=meshes['M16_Screw'][2], return_report=True)
            row[name] = {
                'intersection_area_m2': board_geometry.intersection(screw_geometry).area,
                'screw_section_complete': screw_report['complete'],
                'screw_section_area_m2': screw_geometry.area,
            }
        mount_sections.append(row)

    levels = np.linspace(-.0057, .0057, 9)
    sections, completeness = [], []
    for z in levels:
        nut_geometry, nut_report = section_geometry(
            *meshes['M16_Hex_Nut'][:2], 2, float(z),
            component_labels=meshes['M16_Hex_Nut'][2], return_report=True)
        screw_geometry, screw_report = section_geometry(
            *meshes['M16_Screw'][:2], 2, float(z+source_delta[2]),
            component_labels=meshes['M16_Screw'][2], return_report=True)
        untranslated_screw_area = screw_geometry.area
        screw_geometry = affinity.translate(
            screw_geometry, xoff=-source_delta[0], yoff=-source_delta[1])
        sections.append((nut_geometry, screw_geometry,
                         nut_report['complete'] and screw_report['complete']))
        # Translation does not change the screw section's area or completeness.
        completeness.append({
            'nut_local_z_m': float(z),
            'M16_Hex_Nut': {
                'complete': nut_report['complete'], 'area_m2': nut_geometry.area,
                'components': [item for item in nut_report['components'] if not item['complete']],
            },
            'M16_Screw': {
                'complete': screw_report['complete'], 'area_m2': untranslated_screw_area,
                'components': [item for item in screw_report['components'] if not item['complete']],
            },
        })

    def maximum_overlap(degrees):
        # This is the original objective. The incomplete central slice has empty
        # area and contributes zero; the maximum is determined by complete slices.
        return max(affinity.rotate(nut, float(degrees), origin=(0, 0))
                   .intersection(screw).area for nut, screw, _ in sections)

    optimum = minimize_scalar(maximum_overlap, bounds=(210, 220), method='bounded',
                              options={'xatol': 1e-5})
    thread_sections = []
    for angle in [0., 215., float(optimum.x)]:
        thread_sections.append({
            'nut_yaw_deg': angle,
            'maximum_section_intersection_area_m2': maximum_overlap(angle),
            'section_intersection_area_m2': [
                affinity.rotate(nut, angle, origin=(0, 0)).intersection(screw).area
                if complete else None for nut, screw, complete in sections],
        })

    record = {
        'scope': 'read-only CAD mounting and sampled thread-phase inspection; no grasp, VLM, robot or physics execution',
        'units': 'metres, square metres and degrees',
        'original_source': {
            'nut_seating_plane_z_mm': source_nut_seat_z,
            'screw_head_shoulder_z_mm': source_shoulder_z,
            'gap_mm': source_shoulder_z-source_nut_seat_z,
            'board_thickness_mm': original_board_thickness*1000,
            'gap_matches_board_thickness_abs_m': abs(
                (source_shoulder_z-source_nut_seat_z)*.001-original_board_thickness),
            'native_screw_T_nut_from_shared_source_centers': source_delta.tolist(),
            'shared_source_frame_status': 'consistent pair placement and exact board-thickness gap; CAD-derived interpretation, not physical calibration',
        },
        'current_mount': {
            'board_top_m': board_top, 'board_bottom_m': board_bottom,
            'board_T_screw': screw_T.tolist(), 'board_T_nut_old': old_nut_T.tolist(),
            'native_screw_head_shoulder_z_m': -.0045,
            'shoulder_world_z_m': float(screw_T[2, 3]-.0045),
            'error': 'screw head shoulder is at board top, so head overlaps plate volume around a shaft-size opening',
        },
        'corrected_mount_candidate': {
            'board_T_screw': corrected_screw_T.tolist(),
            'translation_change_m': [0., 0., -thickness],
            'screw_head_shoulder_world_z_m': float(corrected_screw_T[2, 3]-.0045),
            'source_relative_nut_center_world_m': (corrected_screw_T[:3, 3]+source_delta).tolist(),
            'nut_bottom_world_m': float(corrected_screw_T[2, 3]+source_delta[2]-.0065),
            'note': 'Restores screw-head shoulder to board underside; source pair then puts nut underside at board top. Thread phase and CAD interference remain separate unresolved checks.',
        },
        'board_screw_sections': mount_sections,
        'source_height_thread_phase_inspection': {
            'nut_local_z_sections_m': levels.tolist(),
            'all_sections_complete': all(complete for _, _, complete in sections),
            'angles': thread_sections,
            'continuous_local_minimization_bounds_deg': [210, 220],
            'minimum_local_angle_deg': float(optimum.x),
            'minimum_local_max_intersection_m2': float(optimum.fun),
            'scope': 'Nine fixed horizontal levels, eight complete usable sections. Central nut plane has incomplete linework and is excluded (null area). Local yaw search is not a global continuous fit certificate. Positive overlap in a complete section is a represented-solid intersection witness under the helper union-components convention.',
            'section_completeness': completeness,
            'complete_section_count': sum(complete for _, _, complete in sections),
            'incomplete_sections_excluded_from_fit_evidence': [
                index for index, (_, _, complete) in enumerate(sections) if not complete],
        },
        'conclusion': {
            'screw_mount_correction_supported_by_CAD': True,
            'nut_seating_height_supported_by_board_face': True,
            'verified_fully_seated_threaded_pose': False,
            'reason': 'The source pair establishes a plausible seat datum but its thread surfaces overlap in complete inspected sections. A local yaw adjustment reduces maximum sampled overlap to about0.0695mm² without removing it. This could reflect thread phase and mesh approximation; full meshing fit, tightening endpoint and physical tolerance remain unverified.',
        },
    }
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(measure(), indent=2)+'\n', encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
