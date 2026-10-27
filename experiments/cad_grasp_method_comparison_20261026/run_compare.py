"""Upstream contact samplers with declared common downward Panda adapters.

Runs no learned model, robot trajectory or physics step. Original sampler output
is retained in an intrinsic CAD cache. These are not complete framework runs.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import json
import sys
import time
import traceback
import numpy as np

from experiments.cad_grasp_geometry_20261026 import run_trial as core
from experiments.cad_grasp_geometry_20261026.geometry_kernel import MeshCollider
from experiments.cad_grasp_geometry_20261026.inputs import load_inputs,MANIFEST,local
from experiments.cad_grasp_regions_20261026 import run as placement
from experiments.cad_surface_grasps_20261026.centering import center_from_first_contacts
from .pair_reduction import original_pair_sparsity,canonical_pairs,SOURCE as SPARSITY_SOURCE

ROOT=core.ROOT
BASE=ROOT/'outputs/cad_grasp_method_comparison_20261026'
REFERENCE=ROOT/'outputs/cad_surface_grasps_20261026/runs/20261026T200202Z'
CONFIG=dict(pair_sparsity_m=.005,maximum_closing_projection_deg=15.,strict_projection_subset_deg=1.,
    adapters=['midpoint','panda_table'],both_jaw_assignments=True,
    maximum_contact_width_m=.067,extra_opening_m=.012,workers=4,
    primary_protocol='panda_table_proj15')
PROTOCOLS=['midpoint_proj1','midpoint_proj15','panda_table_proj1','panda_table_proj15']


def project_pair(pair):
    difference=pair[1]-pair[0]
    width=float(np.linalg.norm(difference))
    tilt=float(np.degrees(np.arcsin(np.clip(abs(difference[2])/width,0,1))))
    horizontal=difference.copy()
    horizontal[2]=0.
    horizontal/=np.linalg.norm(horizontal)
    return horizontal,width,tilt


def run_case(row,intrinsic,output):
    started=time.perf_counter()
    record=dict(row,method=intrinsic['method'],status='started',grasps=[],generated_count=0,valid_count=0,grouped_count=0)
    folder=Path(output)/'cases'/intrinsic['method']/row['scene']/row['part']
    try:
        components,pads,gap_offset=core.hand_geometry()
        vertices,faces,mesh_info=core.load_target(row)
        collider=MeshCollider(vertices,faces)
        if collider.topology['component_count']!=collider.topology['closed_component_count']:
            raise ValueError('Native mesh component is open')
        record.update(mesh_info=mesh_info,target_solid_topology=collider.topology)
        raw=np.load(local(intrinsic['npz_path']))['pairs_m']
        pose=np.asarray(row['world_T_cad'])
        pairs=raw@pose[:3,:3].T+pose[:3,3]
        pairs=canonical_pairs(pairs)
        widths=np.linalg.norm(pairs[:,1]-pairs[:,0],axis=1)
        finite=np.isfinite(pairs).all(axis=(1,2))
        good=finite&(widths>1e-6)&(widths<=core.HAND['opening_limits_m'][1]-gap_offset-CONFIG['extra_opening_m'])
        with_width=int(good.sum())
        differences=pairs[:,1]-pairs[:,0]
        tilts=np.degrees(np.arcsin(np.clip(np.abs(differences[:,2])/np.maximum(widths,1e-30),0,1)))
        good&=tilts<=CONFIG['maximum_closing_projection_deg']
        eligible=pairs[good]
        sparse=original_pair_sparsity(eligible,CONFIG['pair_sparsity_m'])
        record.update(intrinsic_record=intrinsic,native_pairs=len(raw),width_qualified_pairs=with_width,
            horizontal_projection_qualified_pairs=len(eligible),sparse_pairs=len(sparse),
            candidate_conversion='closing direction projected horizontally, fixed-down frame; original pairs remain native CAD samples',
            input_frame_of_cached_pairs='native_CAD_meters',table_plane_source='same saved observed-depth plane')
        record['runtime_sampling_seconds']=time.perf_counter()-started
        pad_low=max(float(np.asarray(p['vertices_zero_m'])[:,2].min()) for p in pads)
        pad_high=min(float(np.asarray(p['vertices_zero_m'])[:,2].max()) for p in pads)
        pad_center=(pad_low+pad_high)/2
        plane=np.asarray(row['table_plane_world'])
        checks=[]
        adapter_seconds=Counter()
        center_queries=0
        for index,pair in enumerate(sparse,1):
            x,width,tilt=project_pair(pair)
            opening=width+gap_offset+CONFIG['extra_opening_m']
            midpoint=pair.mean(axis=0)
            direct_height=float(midpoint[2]+pad_center)
            table_height=max(direct_height,placement.minimum_table_height(midpoint,x,opening,components,plane))
            for adapter,height in (('midpoint',direct_height),('panda_table',table_height)):
                adapter_started=time.perf_counter()
                positive=np.eye(4)
                positive[:3,:3]=placement.hand_rotation(x)
                positive[:3,3]=midpoint
                positive[2,3]=height
                projection=None
                if adapter=='panda_table':
                    positive,projection=center_from_first_contacts(positive,opening,collider,pads,
                        core.CONFIG['contact_search_fraction_tolerance'])
                    center_queries+=projection['geometry_queries']
                protocols=[adapter+'_proj15']
                if tilt<=CONFIG['strict_projection_subset_deg']:
                    protocols.append(adapter+'_proj1')
                for sign in (1.,-1.):
                    hand=positive.copy()
                    hand[:3,:3]=placement.hand_rotation(sign*x)
                    candidate=dict(id=f'U{len(checks)+1:05d}',family_id=f'PAIR{index:04d}',
                        adapter=adapter,protocols=protocols,pair_index=index,native_pair_projection_deg=tilt,jaw_sign=sign,
                        world_T_grip=hand.tolist(),contacts_world_m=pair.tolist(),width_m=width,
                        opening_m=opening,pad_depth_m=float(height-midpoint[2]),center_projection=projection)
                    reason,evidence=core.validate(candidate,collider,components,pads,plane)
                    candidate.update(valid=reason is None,rejection_reason=reason,**evidence)
                    checks.append(candidate)
                adapter_seconds[adapter]+=time.perf_counter()-adapter_started
        primary=[c for c in checks if CONFIG['primary_protocol'] in c['protocols']]
        primary_valid=[c for c in primary if c['valid']]
        families=[]
        for index in range(1,len(sparse)+1):
            aliases=[c for c in primary_valid if c['pair_index']==index]
            representative=placement.canonical_family(aliases)
            if representative:
                families.append(representative)
        record['grasps']=placement.collapse_equivalent_families(families)
        summaries={}
        for protocol in PROTOCOLS:
            trials=[c for c in checks if protocol in c['protocols']]
            passing=[c for c in trials if c['valid']]
            summaries[protocol]=dict(checked=len(trials),valid=len(passing),available=bool(passing),
                passing_ids=[c['id'] for c in passing],families=len({c['family_id'] for c in passing}),
                rejection_counts=dict(Counter(c['rejection_reason'] for c in trials if not c['valid'])))
        record.update(generated_count=len(checks),primary_generated_count=len(primary),valid_count=len(primary_valid),
            all_adapter_valid_count=sum(c['valid'] for c in checks),grouped_count=len(record['grasps']),
            protocol_summary=summaries,adapter_validation_seconds=dict(adapter_seconds),
            center_projection_geometry_queries=center_queries,
            rejection_counts=summaries[CONFIG['primary_protocol']]['rejection_counts'],
            status='feasible_grasps_available' if primary_valid else 'no_feasible_upstream_pair_in_common_adapter')
        core.dump(folder/'candidate_checks.json',checks)
        record['all_candidate_checks']=str((folder/'candidate_checks.json').relative_to(ROOT))
        core.dump(folder/'sparse_world_pairs.json',sparse.tolist())
    except Exception as error:
        record.update(status='experiment_error',error=str(error),traceback=traceback.format_exc())
    record['timing_seconds']=time.perf_counter()-started
    core.dump(folder/'result.json',record)
    return record


def main(cache_path,limit=None):
    cache=local(cache_path)
    cache_records=json.loads((cache/'records.json').read_text())
    if any(r['status']!='sampled' for r in cache_records):
        raise ValueError('Intrinsic cache contains unavailable/error records')
    inputs=load_inputs()
    if limit is not None:
        inputs=inputs[:limit]
    cache_lookup={(r['part'],r['method']):r for r in cache_records}
    methods=sorted({r['method'] for r in cache_records})
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output=BASE/'comparison'/stamp
    output.mkdir(parents=True,exist_ok=False)
    protected=[MANIFEST,core.BASE/'hand_model.json',ROOT/'docs/cad_view_prompt.txt',
        REFERENCE/'manifest.json',REFERENCE/'results.json',Path(core.__file__),
        Path(core.__file__).with_name('geometry_kernel.py'),Path(placement.__file__),
        cache/'manifest.json',cache/'records.json']
    protected.extend(local(record[key]) for record in cache_records
                     for key in ('npz_path','metadata_path'))
    sources=[Path(__file__),Path(__file__).with_name('upstream.py'),Path(__file__).with_name('pair_reduction.py'),
             SPARSITY_SOURCE,Path(center_from_first_contacts.__code__.co_filename)]
    manifest=dict(created_utc=stamp,cases_per_method=len(inputs),methods=methods,config=CONFIG,protocols=PROTOCOLS,
        intrinsic_cache=str(cache.relative_to(ROOT)),reference_run=str(REFERENCE.relative_to(ROOT)),
        input_manifest=str(MANIFEST.relative_to(ROOT)),validation_config=core.CONFIG,
        source_sha256={str(p.relative_to(ROOT)):core.sha(p) for p in sources},
        protected_sha256={str(p.relative_to(ROOT)):core.sha(p) for p in protected},
        executed='original upstream contact generation and Fabrica pair sparsity, plus declared common Panda adapters',
        not_executed='complete one/EmbodiChain/Fabrica framework, learned ranking, robot execution or physics',
        no_vlm=True,no_robot_execution=True,no_physics_steps=True,no_our_shared_region_guard=True,
        no_our_cylinder_classification=True,no_our_flat_one_degree_filter=True,
        timing_note='Intrinsic CAD generation occurs once per part; runtime includes common adapters and both conversion controls, not one selected deployment protocol. No input pose-estimation or gallery time.')
    core.dump(output/'manifest.json',manifest)
    print('OUTPUT='+str(output),flush=True)
    records=[]
    with ProcessPoolExecutor(max_workers=CONFIG['workers'],initializer=core.initialize) as pool:
        jobs=[pool.submit(run_case,row,cache_lookup[row['part'],method],str(output)) for row in inputs for method in methods]
        for job in as_completed(jobs):
            r=job.result()
            records.append(r)
            records.sort(key=lambda a:(a['method'],a['scene'],a['part']))
            core.dump(output/'results.json',records)
            print(json.dumps({k:r.get(k) for k in ('method','scene','part','native_pairs','sparse_pairs','status','valid_count','grouped_count','timing_seconds','error')}),flush=True)
    after={str(p.relative_to(ROOT)):core.sha(p) for p in protected}
    core.dump(output/'preservation.json',dict(unchanged=after==manifest['protected_sha256'],sha256_after=after))
    print('COMPLETE='+str(output),flush=True)


if __name__=='__main__':
    main(sys.argv[1],int(sys.argv[2]) if len(sys.argv)>2 else None)
