"""Generate one hand placement per opposing CAD contact region, before VLM."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_key,'1')
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from experiments.cad_grasp_geometry_20261026 import run_trial as old
from experiments.cad_grasp_geometry_20261026.geometry_kernel import MeshCollider
from experiments.cad_grasp_geometry_20261026.inputs import load_inputs, MANIFEST, local
from experiments.cad_grasp_geometry_20261026.hand_model import vertices as hand_vertices
from .regions import contact_regions, recast_pair

ROOT = old.ROOT
BASE = ROOT/'outputs/cad_grasp_regions_20261026'
REFERENCE = ROOT/'outputs/cad_grasp_geometry_20261026/runs/20261026T172654Z_surface_coverage'
CONFIG = dict(axis_horizontal_tolerance_deg=1., grid_spacing_m=.001,
              neighbor_width_jump_m=.002, neighbor_midpoint_jump_m=.001,
              neighbor_normal_angle_deg=30., region_width_merge_m=.002,
              region_midpoint_merge_m=.001, table_numerical_reserve_m=.00002,
              height_search_resolution_m=.00001,
              physical_duplicate_position_m=.000001,physical_duplicate_angle_rad=.000001,
              contact_depths_per_region=1, jaw_aliases_per_region=2, workers=4)


def cad_axes(row):
    rotation = np.asarray(row['world_T_cad'])[:3,:3]
    result = []
    for index in range(3):
        axis = rotation[:,index]
        error = float(np.degrees(np.arcsin(np.clip(abs(axis[2]),0,1))))
        usable = error <= CONFIG['axis_horizontal_tolerance_deg']
        horizontal = axis.copy()
        horizontal[2] = 0.
        if np.linalg.norm(horizontal)>1e-10:
            horizontal /= np.linalg.norm(horizontal)
        result.append(dict(label='CAD_'+'XYZ'[index],cad_index=index,
                      cad_axis_world=axis.tolist(),closing_axis_world=horizontal.tolist(),
                      projection_error_deg=error,usable=usable,
                      reason=None if usable else 'CAD_axis_not_approximately_horizontal'))
    return result


def merge_regions(result, pads):
    """Nearby fragments within one pad footprint become one contact region."""
    regions = result['regions']
    if not regions:
        return []
    pad_y_half = min(np.ptp(np.asarray(p['vertices_zero_m'])[:,1])/2 for p in pads)
    pad_z_half = min(np.ptp(np.asarray(p['vertices_zero_m'])[:,2])/2 for p in pads)
    x = np.asarray(result['frame']['closing_axis_world'])
    mid_x = [float(np.median(np.mean(r['contact_pairs_world_m'],axis=1)@x)) for r in regions]
    edges = []
    for i,a in enumerate(regions):
        for j in range(i):
            b = regions[j]
            y_gap = max(0.,a['y_min']-b['y_max'],b['y_min']-a['y_max'])
            z_gap = max(0.,a['z_min']-b['z_max'],b['z_min']-a['z_max'])
            if (y_gap <= pad_y_half and z_gap <= pad_z_half
                and abs(a['median_width_m']-b['median_width_m'])<=CONFIG['region_width_merge_m']
                and abs(mid_x[i]-mid_x[j])<=CONFIG['region_midpoint_merge_m']):
                edges.append((i,j))
    graph = coo_matrix((np.ones(len(edges)),([i for i,j in edges],[j for i,j in edges])),shape=(len(regions),len(regions)))
    count, labels = connected_components(graph,directed=False)
    merged = []
    for label in range(count):
        members = [r for r,l in zip(regions,labels) if l==label]
        coordinates = np.concatenate([r['coordinates_m'] for r in members])
        pairs = np.concatenate([r['contact_pairs_world_m'] for r in members])
        merged.append(dict(region_id=f'R{label+1:03d}', member_region_ids=[r['id'] for r in members],
                  cells=sum(r['cells'] for r in members),
                  y_min=min(r['y_min'] for r in members),y_max=max(r['y_max'] for r in members),
                  z_min=min(r['z_min'] for r in members),z_max=max(r['z_max'] for r in members),
                  center_y=float(coordinates[:,0].mean()),center_z=float(coordinates[:,1].mean()),
                  median_width_m=float(np.median(np.linalg.norm(pairs[:,1]-pairs[:,0],axis=1))),
                  coordinates_m=coordinates))
    return merged


def hand_rotation(x):
    x = np.asarray(x,dtype=float)
    a = np.array([0.,0.,-1.])
    return np.column_stack((x,np.cross(a,x),a))


def minimum_table_height(midpoint, x, opening, components, plane):
    """Solve the same full-plane constraint for both jaw aliases, not a grid."""
    required = -np.inf
    xy = np.array([midpoint[0],midpoint[1],0.])
    for sign in (1.,-1.):
        rotation = hand_rotation(sign*x)
        for c in components:
            for width in (0.,opening):
                points = hand_vertices(c,width)@rotation.T+xy
                lowest = float((points@plane[:3]+plane[3]).min())
                required = max(required,(old.CONFIG['table_clearance_m']-lowest)/plane[2])
    return float(required+CONFIG['table_numerical_reserve_m'])


def target_clear_height(midpoint,x,opening,desired,upper,components,collider):
    """Project height onto continuous open-hand approach clearance, not six depths."""
    queries = 0
    def clear(height):
        nonlocal queries
        queries += 1
        for sign in (1.,-1.):
            rotation = hand_rotation(sign*x)
            translation = np.array([midpoint[0],midpoint[1],height])
            alias_clear = True
            for c in components:
                points = hand_vertices(c,opening)@rotation.T+translation
                if collider.swept_intersects(points,[0.,0.,old.CONFIG['straight_approach_m']],old.CONFIG['open_clearance_m']):
                    alias_clear = False
                    break
            if alias_clear:
                return True
        return False
    if clear(desired):
        return desired,queries
    if upper<desired or not clear(upper):
        return None,queries
    low,high = desired,upper
    while high-low>CONFIG['height_search_resolution_m']:
        middle = (low+high)/2
        if clear(middle):
            high = middle
        else:
            low = middle
    return high,queries


def prepare_family(region, region_result, x, components, pads, gap_offset, plane,collider):
    """One geometry-centered height, adjusted analytically for the table."""
    y,z = region['center_y'],region['center_z']
    pair = recast_pair(region_result,y,z)
    if pair is None:
        coords = region['coordinates_m']
        chosen = coords[np.argmin(np.sum((coords-[y,z])**2,axis=1))]
        y,z = chosen
        pair = recast_pair(region_result,y,z)
    if pair is None:
        return None,'no_qualified_pair_at_region_center'
    pad_low = max(float(np.asarray(p['vertices_zero_m'])[:,2].min()) for p in pads)
    pad_high = min(float(np.asarray(p['vertices_zero_m'])[:,2].max()) for p in pads)
    pad_center = (pad_low+pad_high)/2
    preferred_height = float(z+pad_center)
    height = preferred_height
    height_queries = 0
    for _ in range(3):
        points = np.asarray(pair['contacts_world_m'])
        opening = pair['width_m']+gap_offset+old.CONFIG['extra_opening_m']
        height = max(preferred_height,minimum_table_height(points.mean(axis=0),x,opening,components,plane))
        height,queries = target_clear_height(points.mean(axis=0),x,opening,height,
                          region['z_max']+pad_high-old.CONFIG['pad_edge_margin_m'],components,collider)
        height_queries += queries
        if height is None:
            return None,'open_hand_cannot_clear_target_within_contact_region_height'
        # Work in the intersection of the sampled contact region and the finite
        # inner pads. A new height is never paired with an old untested contact.
        low = max(region['z_min'],height-pad_high+old.CONFIG['pad_edge_margin_m'])
        high = min(region['z_max'],height-pad_low-old.CONFIG['pad_edge_margin_m'])
        if high < low:
            return None,'contact_region_cannot_overlap_inner_pads_above_table'
        desired_z = (low+high)/2
        new_pair = recast_pair(region_result,y,desired_z)
        if new_pair is None:
            coords = region['coordinates_m']
            eligible = coords[(coords[:,1]>=low)&(coords[:,1]<=high)]
            if not len(eligible):
                return None,'no_qualified_contact_in_pad_height_overlap'
            chosen = eligible[np.argmin(np.sum((eligible-[region['center_y'],desired_z])**2,axis=1))]
            y,desired_z = chosen
            new_pair = recast_pair(region_result,y,desired_z)
        if new_pair is None:
            return None,'representative_ray_has_no_qualified_contacts'
        pair = new_pair
        z = desired_z
    depth = float(height-z)
    return dict(pair=pair,height_m=float(height),pad_depth_m=depth,
                preferred_pad_center_height_m=preferred_height,
                height_geometry_queries=height_queries,
                placement_rule='pad_center_projected_to_table_and_target_height_constraints'),None


def canonical_family(aliases):
    good = [g for g in aliases if g['valid']]
    if not good:
        return None
    chosen = min(good,key=lambda g:(0 if g['jaw_sign']>0 else 1,g['opening_m'],g['id']))
    return dict(chosen,grouped_candidate_ids=[g['id'] for g in good],
                jaw_aliases=good,source_family_ids=sorted({g['family_id'] for g in good}),
                representative_rule='canonical_jaw_then_smallest_checked_approach_opening')


def same_placement(first,second):
    """Sampling/contact-depth labels do not distinguish physical hand poses."""
    return (np.linalg.norm(first['position']-second['position'])<=CONFIG['physical_duplicate_position_m']
            and float(first['axis']@second['axis'])>=np.cos(CONFIG['physical_duplicate_angle_rad'])
            and abs(first['opening']-second['opening'])<=CONFIG['physical_duplicate_position_m'])


def same_final_grip(first,second):
    a,b = np.asarray(first['world_T_grip']),np.asarray(second['world_T_grip'])
    angle_cos = float((np.trace(a[:3,:3]@b[:3,:3].T)-1)/2)
    return (np.linalg.norm(a[:3,3]-b[:3,3])<=CONFIG['physical_duplicate_position_m']
            and angle_cos>=np.cos(CONFIG['physical_duplicate_angle_rad'])
            and abs(first['contact_opening_m']-second['contact_opening_m'])<=CONFIG['physical_duplicate_position_m'])


def collapse_equivalent_families(grasps):
    """Different safe approach openings need not become repeated final grips."""
    groups = []
    for grasp in grasps:
        match = next((i for i,g in enumerate(groups) if same_final_grip(g,grasp)),None)
        if match is None:
            groups.append(grasp)
        else:
            groups[match] = canonical_family(groups[match]['jaw_aliases']+grasp['jaw_aliases'])
    return groups


def run_case(row,output):
    started = time.perf_counter()
    record = dict(row,axes=cad_axes(row),status='started',contact_families=[],
                  generated_count=0,valid_count=0,grouped_count=0,grasps=[])
    folder = Path(output)/'cases'/row['scene']/row['part']
    try:
        components,pads,gap_offset = old.hand_geometry()
        vertices,faces,mesh_info = old.load_target(row)
        collider = MeshCollider(vertices,faces)
        record.update(mesh_info=mesh_info,target_solid_topology=collider.topology)
        if collider.topology['component_count']!=collider.topology['closed_component_count']:
            raise ValueError('Native target has an open component')
        plane = np.asarray(row['table_plane_world'])
        checks,rejections = [],Counter()
        prepared_placements = []
        raw_regions = 0
        grid_lines = 0
        for axis in record['axes']:
            if not axis['usable']:
                continue
            x = np.asarray(axis['closing_axis_world'])
            result = contact_regions(vertices,faces,x,
                        max_width_m=old.HAND['opening_limits_m'][1]-gap_offset-old.CONFIG['extra_opening_m'],
                        grid_spacing_m=CONFIG['grid_spacing_m'],normal_angle_deg=old.CONFIG['normal_angle_deg'],
                        neighbor_width_jump_m=CONFIG['neighbor_width_jump_m'],
                        neighbor_midpoint_jump_m=CONFIG['neighbor_midpoint_jump_m'],
                        neighbor_normal_angle_deg=CONFIG['neighbor_normal_angle_deg'])
            axis['region_sampling'] = result['metadata']
            raw_regions += len(result['regions'])
            grid_lines += result['metadata']['horizontal_lines_cast']
            merged = merge_regions(result,pads)
            axis['merged_region_count'] = len(merged)
            for region in merged:
                family_id = axis['label']+'_'+region['region_id']
                family = {k:v for k,v in region.items() if k!='coordinates_m'}
                family.update(family_id=family_id,axis_label=axis['label'],prepared=False)
                placement,reason = prepare_family(region,result,x,components,pads,gap_offset,plane,collider)
                if reason:
                    family['preparation_rejection'] = reason
                    record['contact_families'].append(family)
                    continue
                seed_pair = np.asarray(placement['pair']['contacts_world_m'])
                position = seed_pair.mean(axis=0)
                position[2] = placement['height_m']
                signature = dict(position=position,axis=x,
                      opening=placement['pair']['width_m']+gap_offset+old.CONFIG['extra_opening_m'])
                duplicate = next((p for p in prepared_placements if same_placement(p,signature)),None)
                if duplicate is not None:
                    family.update(preparation_rejection='duplicate_physical_hand_placement',
                                  duplicate_of=duplicate['family_id'])
                    record['contact_families'].append(family)
                    continue
                prepared_placements.append(dict(signature,family_id=family_id))
                family.update(prepared=True,height_m=placement['height_m'],
                    pad_depth_m=placement['pad_depth_m'],placement_rule=placement['placement_rule'])
                family['height_geometry_queries'] = placement['height_geometry_queries']
                aliases = []
                for sign in (1.,-1.):
                    pair = np.asarray(placement['pair']['contacts_world_m'])
                    pose = np.eye(4)
                    pose[:3,:3] = hand_rotation(sign*x)
                    pose[:3,3] = pair.mean(axis=0)
                    pose[2,3] = placement['height_m']
                    width = float(np.linalg.norm(pair[1]-pair[0]))
                    candidate = dict(id=f'C{len(checks)+1:04d}',family_id=family_id,axis_label=axis['label'],
                        cad_axis_index=axis['cad_index'],jaw_sign=sign,
                        cad_axis_projection_error_deg=axis['projection_error_deg'],
                        world_T_grip=pose.tolist(),contacts_world_m=pair.tolist(),width_m=width,
                        opening_m=width+gap_offset+old.CONFIG['extra_opening_m'],pad_depth_m=placement['pad_depth_m'])
                    reason,evidence = old.validate(candidate,collider,components,pads,plane)
                    candidate.update(valid=reason is None,rejection_reason=reason,**evidence)
                    checks.append(candidate)
                    aliases.append(candidate)
                    if reason:
                        rejections[reason] += 1
                representative = canonical_family(aliases)
                family['candidate_ids'] = [g['id'] for g in aliases]
                family['valid_alias_ids'] = [g['id'] for g in aliases if g['valid']]
                if representative:
                    record['grasps'].append(representative)
                record['contact_families'].append(family)
        record['grasps'] = collapse_equivalent_families(record['grasps'])
        record.update(raw_contact_regions=raw_regions,horizontal_grid_lines=grid_lines,
                      merged_contact_regions=len(record['contact_families']),
                      prepared_contact_families=sum(f['prepared'] for f in record['contact_families']),
                      generated_count=len(checks),valid_count=sum(c['valid'] for c in checks),
                      grouped_count=len(record['grasps']),rejection_counts=dict(rejections),
                      preparation_rejection_counts=dict(Counter(f['preparation_rejection'] for f in record['contact_families'] if not f['prepared'])),
                      status='feasible_grasps_available' if record['grasps'] else 'no_feasible_centered_CAD_axis_grasp')
        old.dump(folder/'candidate_checks.json',checks)
        record['all_candidate_checks'] = str((folder/'candidate_checks.json').relative_to(ROOT))
    except Exception as error:
        record.update(status='experiment_error',error=str(error),traceback=traceback.format_exc())
    record['timing_seconds'] = time.perf_counter()-started
    old.dump(folder/'result.json',record)
    return record


def main():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = BASE/'runs'/stamp
    output.mkdir(parents=True,exist_ok=False)
    rows = load_inputs()
    if len(sys.argv)>1:
        rows = rows[:int(sys.argv[1])]
    protected = [MANIFEST,old.BASE/'hand_model.json',REFERENCE/'results.json',REFERENCE/'manifest.json',
                 ROOT/'docs/cad_view_prompt.txt',Path(old.__file__),Path(old.__file__).with_name('geometry_kernel.py')]
    manifest = dict(created_utc=stamp,cases=len(rows),config=CONFIG,validation_config=old.CONFIG,
               baseline_run=str(REFERENCE.relative_to(ROOT)),
               input_manifest=str(MANIFEST.relative_to(ROOT)),no_vlm=True,no_robot_execution=True,
               no_physics_steps=True,no_candidate_cap=True,no_score_ranking=True,
               pose_assumption='same saved observed-depth CAD fits; correct CAD/crop correspondence supplied',
               quality_definition='region-centered, axis-constrained geometric feasibility; not holding or assembly success',
               source_sha256={str(p.relative_to(ROOT)):old.sha(p) for p in Path(__file__).parent.glob('*.py')},
               protected_sha256={str(p.relative_to(ROOT)):old.sha(p) for p in protected})
    old.dump(output/'manifest.json',manifest)
    print('OUTPUT='+str(output),flush=True)
    records = []
    with ProcessPoolExecutor(max_workers=CONFIG['workers'],initializer=old.initialize) as pool:
        jobs = [pool.submit(run_case,row,str(output)) for row in rows]
        for job in as_completed(jobs):
            r = job.result()
            records.append(r)
            records.sort(key=lambda row:(row['scene'],row['part']))
            old.dump(output/'results.json',records)
            print(json.dumps({k:r.get(k) for k in ('scene','part','status','raw_contact_regions','merged_contact_regions','generated_count','valid_count','grouped_count','timing_seconds','error')}),flush=True)
    after = {str(p.relative_to(ROOT)):old.sha(p) for p in protected}
    old.dump(output/'preservation.json',dict(unchanged=after==manifest['protected_sha256'],sha256_after=after))
    print('COMPLETE='+str(output),flush=True)


if __name__=='__main__':
    main()
