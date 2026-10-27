"""Cache actual upstream contact samples once per native CAD, before pose fitting.

No grasp validity is implied. Saved observed CAD poses transform these intrinsic
samples into each scene later; this stage never reads simulator object state.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import traceback
import numpy as np

from experiments.cad_grasp_geometry_20261026 import run_trial as geometry
from experiments.cad_grasp_geometry_20261026.inputs import load_inputs,MANIFEST
from .upstream import generate_pairs,source_manifest,verify_ray_chunk_equivalence

ROOT=geometry.ROOT
BASE=ROOT/'outputs/cad_grasp_method_comparison_20261026/intrinsics'
METHODS=('one_default','one_budget512','graspkit_512')
CONFIG=dict(seed=23,n_sample=512,workers=2,methods=list(METHODS),
    input_and_output_frame='native_CAD',units='meters',no_pair_NMS=True,
    no_pose_construction=True,no_hand_collision_checks=True,no_table_checks=True)


def generate_case(row,method,output):
    started=time.perf_counter()
    record=dict(part=row['part'],method=method,mesh_path=row['mesh_path'],mesh_sha256=row['mesh_sha256'],
                status='started',frame='native_CAD',units='meters')
    try:
        local_row=dict(row,world_T_cad=np.eye(4).tolist())
        vertices,faces,info=geometry.load_target(local_row)
        sampling_started=time.perf_counter()
        sampled=generate_pairs(vertices,faces,method,seed=CONFIG['seed'],n_sample=CONFIG['n_sample'])
        sourcegen_seconds=time.perf_counter()-sampling_started
        directory=Path(output)/'pairs'/row['part']
        directory.mkdir(parents=True,exist_ok=True)
        npz=directory/(method+'.npz')
        metadata_path=directory/(method+'.json')
        np.savez_compressed(npz,pairs_m=sampled['pairs_world_m'],normals=sampled['normals_world'])
        metadata=dict(sampled['metadata'],input_and_output_frame='native_CAD',units='meters',
                      sourcegen_seconds=sourcegen_seconds,part=row['part'],method=method,
                      mesh_path=row['mesh_path'],mesh_sha256=row['mesh_sha256'],local_mesh_info=info)
        geometry.dump(metadata_path,metadata)
        record.update(status='sampled',raw_pairs=len(sampled['pairs_world_m']),
            npz_path=str(npz.relative_to(ROOT)),metadata_path=str(metadata_path.relative_to(ROOT)),
            sourcegen_seconds=sourcegen_seconds,local_mesh_info=info)
    except Exception as error:
        record.update(status='error',error=str(error),traceback=traceback.format_exc())
    record['total_task_seconds']=time.perf_counter()-started
    return record


def main():
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output=BASE/stamp
    output.mkdir(parents=True,exist_ok=False)
    rows=load_inputs()
    unique={}
    for row in rows:
        previous=unique.get(row['part'])
        if previous and (previous['mesh_path'],previous['mesh_sha256'])!=(row['mesh_path'],row['mesh_sha256']):
            raise ValueError('A CAD identity maps to multiple meshes: '+row['part'])
        unique.setdefault(row['part'],row)
    rows=[unique[name] for name in sorted(unique)]
    protected=[MANIFEST,Path(geometry.__file__),Path(geometry.__file__).with_name('geometry_kernel.py'),
               ROOT/'docs/cad_view_prompt.txt']
    graspkit=ROOT/'experiments/runtime_antipodal_assembly_20261023/vendor'
    protected += [graspkit/name for name in ('antipodal_sampler.py','_compat.py','provenance.json','LICENSE.txt')]
    source=Path(__file__).parent
    source_files=[Path(__file__),source/'upstream.py']
    source_files += [p for p in (source/'vendor/one').iterdir() if p.is_file()]
    manifest=dict(created_utc=stamp,config=CONFIG,unique_CADs=len(rows),sampling_tasks=len(rows)*len(METHODS),
        input_manifest=str(MANIFEST.relative_to(ROOT)),sampler_sources=source_manifest(),
        ray_chunk_equivalence=verify_ray_chunk_equivalence(),
        no_simulator_state=True,no_VLM=True,no_robot_execution=True,no_model_weights=True,
        source_sha256={str(p.relative_to(ROOT)):geometry.sha(p) for p in source_files},
        protected_sha256={str(p.relative_to(ROOT)):geometry.sha(p) for p in protected},
        timing_note='Source generation is cached CAD-intrinsic work; it is not per-scene runtime. sourcegen_seconds includes backend first-load in that process; upstream sampling_seconds excludes backend import. Four/all-native geometry is unchanged; no NMS or hand checks are performed.')
    geometry.dump(output/'manifest.json',manifest)
    print('OUTPUT='+str(output),flush=True)
    records=[]
    with ProcessPoolExecutor(max_workers=CONFIG['workers']) as pool:
        jobs=[pool.submit(generate_case,row,method,str(output)) for row in rows for method in METHODS]
        for job in as_completed(jobs):
            record=job.result()
            records.append(record)
            records.sort(key=lambda r:(r['part'],r['method']))
            geometry.dump(output/'records.json',records)
            print(json.dumps({k:record.get(k) for k in ('part','method','status','raw_pairs','sourcegen_seconds','error')}),flush=True)
    after={str(p.relative_to(ROOT)):geometry.sha(p) for p in protected}
    geometry.dump(output/'preservation.json',dict(unchanged=after==manifest['protected_sha256'],sha256_after=after))
    print('COMPLETE='+str(output),flush=True)


if __name__=='__main__':
    main()
