"""Surface-normal/cylinder CAD candidates, fixed down; no VLM or execution."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import sys
import time
import traceback
import json
import numpy as np

from experiments.cad_grasp_geometry_20261026 import run_trial as geometry
from experiments.cad_grasp_geometry_20261026.geometry_kernel import MeshCollider
from experiments.cad_grasp_geometry_20261026.inputs import load_inputs,MANIFEST
from experiments.cad_grasp_regions_20261026 import run as compact
from experiments.cad_grasp_regions_20261026.regions import contact_regions
from .directions import surface_directions,CONFIG as DIRECTION_CONFIG
from .contacts import inspect_contacts
from .centering import center_from_first_contacts

ROOT=geometry.ROOT
BASE=ROOT/'outputs/cad_surface_grasps_20261026'
PREVIOUS=ROOT/'outputs/cad_grasp_regions_20261026/runs/20261026T184853Z'
CONFIG=dict({k:v for k,v in compact.CONFIG.items() if k!='axis_horizontal_tolerance_deg'},
            flat_contact_normal_deg=1.,curved_mesh_normal_deg=15.,
            cylinder_transverse_tolerance_deg=1.,cylinder_tangent_base_tolerance_deg=1.,
            cylinder_tangent_mesh_allowance='actual_contact_triangle_radial_angular_span',
            near_contact_band_m=.000025,finite_pad_center_projection=True)


def contact_model_checks(candidate,collider,pads,direction_analysis):
    ids=[p['first_contact']['triangle_index'] for p in candidate['actual_contacts']]
    fits=[h for h in direction_analysis['cylinder_hypotheses']
          if all(i in h['supporting_triangle_ids'] for i in ids)]
    # A cylinder proposal can touch a broad flat instead, e.g. outside a nut
    # whose bore supplied the direction. Classify the actual touched geometry.
    if not fits:
        reason,evidence=inspect_contacts(candidate,collider,pads,CONFIG['flat_contact_normal_deg'])
        return reason,dict(evidence,contact_geometry='opposing_flat_surfaces')
    checked=[]
    for fit in sorted(fits,key=lambda h:h['circle_vertex_residual95_m']):
        reason,evidence=inspect_contacts(candidate,collider,pads,CONFIG['curved_mesh_normal_deg'],fit)
        checked.append(dict(cylinder_id=fit['id'],reason=reason))
        if reason is None:
            return None,dict(evidence,contact_geometry='fitted_round_side',cylinder_model=fit,
                             checked_cylinder_models=checked)
    return reason,dict(evidence,contact_geometry='fitted_round_side',checked_cylinder_models=checked)


def run_case(row,output):
    started=time.perf_counter()
    record=dict(row,status='started',generated_count=0,valid_count=0,grouped_count=0,
                grasps=[],contact_families=[])
    folder=Path(output)/'cases'/row['scene']/row['part']
    try:
        components,pads,gap_offset=geometry.hand_geometry()
        vertices,faces,info=geometry.load_target(row)
        collider=MeshCollider(vertices,faces)
        if collider.topology['closed_component_count']!=collider.topology['component_count']:
            raise ValueError('Native target has an open connected component')
        analysis=surface_directions(vertices,faces)
        for i,direction in enumerate(analysis['directions'],1):
            direction['id']=f'D{i:03d}'
        record.update(mesh_info=info,target_solid_topology=collider.topology,
                      direction_analysis=analysis,directions=analysis['directions'])
        plane=np.asarray(row['table_plane_world'])
        checks=[]
        seen=[]
        raw_regions=0
        grid_lines=0
        for direction in record['directions']:
            x=np.asarray(direction['closing_axis_world'])
            # Cylindrical tangency is checked on the fitted surface afterwards;
            # the mesh screen retains its declared tessellation tolerance.
            angle=CONFIG['curved_mesh_normal_deg'] if direction['source']=='cylinder_radial' else CONFIG['flat_contact_normal_deg']
            result=contact_regions(vertices,faces,x,
                max_width_m=geometry.HAND['opening_limits_m'][1]-gap_offset-geometry.CONFIG['extra_opening_m'],
                grid_spacing_m=CONFIG['grid_spacing_m'],normal_angle_deg=angle,
                neighbor_width_jump_m=CONFIG['neighbor_width_jump_m'],
                neighbor_midpoint_jump_m=CONFIG['neighbor_midpoint_jump_m'],
                neighbor_normal_angle_deg=2*angle)
            direction['region_sampling']=result['metadata']
            raw_regions+=len(result['regions'])
            grid_lines+=result['metadata']['horizontal_lines_cast']
            for region in compact.merge_regions(result,pads):
                family=dict((k,v) for k,v in region.items() if k!='coordinates_m')
                family_id=direction['id']+'_'+region['region_id']
                family.update(family_id=family_id,direction_id=direction['id'],axis_label=direction['label'],prepared=False)
                placement,reason=compact.prepare_family(region,result,x,components,pads,gap_offset,plane,collider)
                if reason:
                    family['preparation_rejection']=reason
                    record['contact_families'].append(family)
                    continue
                pair=np.asarray(placement['pair']['contacts_world_m'])
                position=pair.mean(axis=0)
                position[2]=placement['height_m']
                opening=placement['pair']['width_m']+gap_offset+geometry.CONFIG['extra_opening_m']
                signature=dict(position=position,axis=x,opening=opening)
                duplicate=next((p for p in seen if compact.same_placement(p,signature)),None)
                if duplicate is not None:
                    family.update(preparation_rejection='duplicate_physical_hand_placement',duplicate_of=duplicate['family_id'])
                    record['contact_families'].append(family)
                    continue
                seen.append(dict(signature,family_id=family_id))
                positive_pose=np.eye(4)
                positive_pose[:3,:3]=compact.hand_rotation(x)
                positive_pose[:3,3]=position
                projected,projection=center_from_first_contacts(positive_pose,opening,collider,pads,
                    geometry.CONFIG['contact_search_fraction_tolerance'])
                position=projected[:3,3]
                family.update(prepared=True,height_m=placement['height_m'],pad_depth_m=placement['pad_depth_m'],
                              placement_rule=placement['placement_rule'],height_geometry_queries=placement['height_geometry_queries'],
                              center_projection=projection)
                aliases=[]
                for sign in (1.,-1.):
                    pose=np.eye(4)
                    pose[:3,:3]=compact.hand_rotation(sign*x)
                    pose[:3,3]=position
                    candidate=dict(id=f'S{len(checks)+1:04d}',family_id=family_id,
                        direction_id=direction['id'],direction_label=direction['label'],direction_source=direction['source'],
                        axis_label=direction['label'],jaw_sign=sign,surface_projection_error_deg=direction['projection_error_deg'],
                        world_T_grip=pose.tolist(),contacts_world_m=pair.tolist(),width_m=placement['pair']['width_m'],
                        opening_m=opening,pad_depth_m=placement['pad_depth_m'],center_projection=projection)
                    reason,evidence=geometry.validate(candidate,collider,components,pads,plane)
                    candidate.update(evidence)
                    candidate['interference_check_passed']=reason is None
                    if reason is None:
                        reason,details=contact_model_checks(candidate,collider,pads,analysis)
                        candidate.update(details)
                    candidate.update(valid=reason is None,rejection_reason=reason)
                    checks.append(candidate)
                    aliases.append(candidate)
                family.update(candidate_ids=[c['id'] for c in aliases],valid_alias_ids=[c['id'] for c in aliases if c['valid']])
                representative=compact.canonical_family(aliases)
                if representative:
                    record['grasps'].append(representative)
                record['contact_families'].append(family)
        record['grasps']=compact.collapse_equivalent_families(record['grasps'])
        record.update(raw_contact_regions=raw_regions,horizontal_grid_lines=grid_lines,
            merged_contact_regions=len(record['contact_families']),
            prepared_contact_families=sum(f['prepared'] for f in record['contact_families']),
            center_projection_geometry_queries=sum(f.get('center_projection',{}).get('geometry_queries',0) for f in record['contact_families']),
            generated_count=len(checks),valid_count=sum(c['valid'] for c in checks),
            grouped_count=len(record['grasps']),
            rejection_counts=dict(Counter(c['rejection_reason'] for c in checks if not c['valid'])),
            preparation_rejection_counts=dict(Counter(f['preparation_rejection'] for f in record['contact_families'] if not f['prepared'])),
            status='feasible_grasps_available' if record['grasps'] else 'no_feasible_surface_grasp')
        geometry.dump(folder/'candidate_checks.json',checks)
        record['all_candidate_checks']=str((folder/'candidate_checks.json').relative_to(ROOT))
    except Exception as error:
        record.update(status='experiment_error',error=str(error),traceback=traceback.format_exc())
    record['timing_seconds']=time.perf_counter()-started
    geometry.dump(folder/'result.json',record)
    return record


def main():
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output=BASE/'runs'/stamp
    output.mkdir(parents=True,exist_ok=False)
    rows=load_inputs()
    if len(sys.argv)>1:
        rows=rows[:int(sys.argv[1])]
    protected=[MANIFEST,geometry.BASE/'hand_model.json',PREVIOUS/'manifest.json',PREVIOUS/'results.json',
               ROOT/'docs/cad_view_prompt.txt',Path(geometry.__file__),Path(geometry.__file__).with_name('geometry_kernel.py'),
               Path(compact.__file__),Path(compact.__file__).with_name('regions.py')]
    manifest=dict(created_utc=stamp,cases=len(rows),config=CONFIG,direction_config=DIRECTION_CONFIG,
        validation_config=geometry.CONFIG,baseline_run=str(PREVIOUS.relative_to(ROOT)),input_manifest=str(MANIFEST.relative_to(ROOT)),
        no_vlm=True,no_robot_execution=True,no_physics_steps=True,no_candidate_cap=True,no_score_ranking=True,
        no_native_CAD_axis_constraint=True,no_other_part_or_robot_self_checks=True,
        pose_assumption='same saved observed-depth CAD fits; correct CAD/crop correspondence supplied',
        quality_definition='surface-aligned finite pad and interference feasibility; not force/slip or assembly precision',
        source_sha256={str(p.relative_to(ROOT)):geometry.sha(p) for p in Path(__file__).parent.glob('*.py')},
        protected_sha256={str(p.relative_to(ROOT)):geometry.sha(p) for p in protected},
        planner_reference='one antipodal surface-normal/raycast principles; no upstream framework installed',
        planner_source_manifest=json.loads(Path(__file__).with_name('source_manifest.json').read_text()))
    geometry.dump(output/'manifest.json',manifest)
    print('OUTPUT='+str(output),flush=True)
    records=[]
    with ProcessPoolExecutor(max_workers=CONFIG['workers'],initializer=geometry.initialize) as pool:
        for job in as_completed([pool.submit(run_case,row,str(output)) for row in rows]):
            record=job.result()
            records.append(record)
            records.sort(key=lambda r:(r['scene'],r['part']))
            geometry.dump(output/'results.json',records)
            print(json.dumps({k:record.get(k) for k in ('scene','part','status','generated_count','valid_count','grouped_count','timing_seconds','error')}),flush=True)
    after={str(p.relative_to(ROOT)):geometry.sha(p) for p in protected}
    geometry.dump(output/'preservation.json',dict(unchanged=after==manifest['protected_sha256'],sha256_after=after))
    print('COMPLETE='+str(output),flush=True)


if __name__=='__main__':
    main()
