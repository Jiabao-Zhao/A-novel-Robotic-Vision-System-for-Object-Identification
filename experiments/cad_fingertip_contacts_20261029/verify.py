"""Post-launch saved-result audit; never generates a pose or reruns collision."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import ConvexHull

from experiments.cad_grasp_selection_20261028.criteria import evaluate_criteria, select_priority
from experiments.cad_grasp_selection_20261028.geometry import component_vertices, transform_points, rigid_transform
from experiments.cad_grasp_selection_20261029.hand_frame import fingertip_calibration
from .run_nist import ROOT, previous


def read(path):return json.loads(Path(path).read_text())


def verify(folder):
    folder=Path(folder).resolve()
    manifest=read(folder/'launch_manifest.json')
    for section in ('source_sha256','protected_sha256'):
        for name,expected in manifest[section].items():
            if previous.sha(ROOT/name)!=expected:raise ValueError('Launch hash changed: '+name)
    hand=read(previous.HAND);tip=fingertip_calibration(hand)
    calibration=read(folder/'contact_calibration.json')
    rows=read(folder/'results.json');summary=read(folder/'summary.json')
    if len(rows)!=96 or len({(r['scene'],r['part']) for r in rows})!=96:raise ValueError('Incomplete trial')
    audits=[]
    watched={}
    for row in rows:
        path=folder/'cases'/row['scene']/row['part']/'candidate_diagnostics.json'
        watched[path]=previous.sha(path)
        candidates=read(path)
        if len(candidates)!=row['selection']['proposed_count'] or len(candidates)>100:raise ValueError('Candidate count mismatch')
        valid=[];counts={};fitted=0;component_values=0
        for c in candidates:
            validity=c['validity'];diagnostics=validity['diagnostics']
            for name,e in diagnostics.items():counts.setdefault(name,Counter())[e['status']]+=1
            if (validity['status']=='valid')!=all(e['status']=='pass' for e in diagnostics.values()):
                raise ValueError('Validity disagrees with independent check statuses')
            if c['first_contact_fit']['status']!='fitted':
                if validity['status']=='valid':raise ValueError('Unfitted fallback selected')
                continue
            fitted+=1
            grip=rigid_transform(c['cad_T_grip_site']);pose=rigid_transform(c['cad_T_hand'])
            if not np.allclose(pose@tip['fingertip_T_grip_site'],grip,atol=1e-10,rtol=0):raise ValueError('Frame mismatch')
            if c['contact_calibration_sha256']!=calibration['calibration_sha256']:raise ValueError('Contact calibration mismatch')
            actual=np.asarray(c['actual_contacts_hand_m']);cad=np.asarray(c['contacts_cad_m'])
            if not np.allclose(transform_points(actual,grip),cad,atol=1e-10,rtol=0):raise ValueError('Contact frame mismatch')
            saved={e['component']:e['table_minimum_clearance_m'] for e in validity['checks']['component_geometry']}
            minima=[]
            for comp in hand['components']:
                points=np.concatenate([component_vertices(comp,q) for q in (c['holding_opening_m'],c['pre_opening_m'])])
                world=transform_points(transform_points(points,grip),row['world_T_cad'])
                plane=np.asarray(row['table_plane_world'])
                measured=float(np.min(world@plane[:3]+plane[3]));minima.append(measured)
                if abs(measured-saved[comp['name']])>1e-10:raise ValueError('Component table clearance mismatch')
                component_values+=1
            if abs(min(minima)-validity['checks']['table_minimum_clearance_m'])>1e-10:raise ValueError('Global table clearance mismatch')
            if validity['status']=='valid':
                # Independently require the chosen points to lie on inward
                # original finger support planes beyond the old pad edge.
                for i,name in enumerate(calibration['finger_components']):
                    comp=next(e for e in hand['components'] if e['name']==name)
                    point=actual[i]-c['holding_opening_m']*np.asarray(comp['translation_per_opening'])
                    hull=ConvexHull(np.asarray(comp['vertices_zero_m']))
                    signed=point@hull.equations[:,:3].T+hull.equations[:,3]
                    axis=np.array([1.,0,0]) if i==0 else np.array([-1.,0,0])
                    allowed=hull.equations[:,:3]@axis>=np.cos(calibration['normal_angle_rad'])
                    if (point[2]<calibration['minimum_contact_z_grip_site_m']-1.02e-8 or
                        not np.any(allowed&(np.abs(signed)<=1.02e-8)) or np.any(signed>1.02e-8)):
                        raise ValueError('Selected contact is not on permitted complete-finger support')
                criteria=evaluate_criteria(cad,row['selection']['reference'],
                    np.asarray(row['world_T_cad'])[:3,:3]@np.asarray(c['approach_cad']),
                    np.asarray(row['table_plane_world'])[:3])
                for name,value in criteria.items():
                    if not np.allclose(value,c['criteria'][name],atol=1e-10,rtol=0):raise ValueError('Criterion mismatch: '+name)
                valid.append(c)
        if len(valid)!=row['selection']['valid_count']:raise ValueError('Valid count mismatch')
        decision=select_priority(valid,row['selection'].get('tie_tolerances'))
        expected=decision['selected_candidate'];actual=row['selection']['selected_grasp']
        if (expected is None)!=(actual is None) or (expected is not None and expected['candidate_id']!=actual['candidate_id']):
            raise ValueError('Selected priority is not best among valid samples')
        audits.append(dict(scene=row['scene'],part=row['part'],proposed=len(candidates),fitted=fitted,
            valid=len(valid),component_table_values_checked=component_values,
            independent_check_counts={name:dict(values) for name,values in counts.items()}))
    if any(previous.sha(path)!=expected for path,expected in watched.items()):raise ValueError('Evidence changed during audit')
    total=dict(proposed=sum(e['proposed'] for e in audits),fitted=sum(e['fitted'] for e in audits),
        valid=sum(e['valid'] for e in audits),selected=sum(e['valid']>0 for e in audits))
    if total['valid']!=summary['valid_poses'] or total['selected']!=summary['selected_placements']:raise ValueError('Trial totals mismatch')
    output=dict(totals=total,cases=audits,launch_hashes_match=True,
        original_hand_geometry_unchanged=True,actual_contact_frames_checked=True,
        all_fitted_complete_hand_table_distances_checked=True,selected_real_support_planes_checked=True,
        all_valid_criteria_and_selection_priorities_recomputed=True,
        source_collision_engine_rerun=False,no_physical_success_measured=True,
        audit_source_sha256=previous.sha(__file__),post_launch_audit_helper=True)
    previous.dump(folder/'independent_verification.json',output)
    print(json.dumps(total),flush=True)
    return output


if __name__=='__main__':verify(sys.argv[1])
