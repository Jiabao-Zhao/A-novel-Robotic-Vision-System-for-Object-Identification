"""Resumable orchestration of the pinned public FreeZeV2-Re commands.

This file supplies process isolation, checkpoints and target accounting only.
All descriptors, pose estimation, candidate selection and scoring are original.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import csv
import io
import json
import math
import os
from pathlib import Path
import runpy
import sys
import time
import uuid

import reference_gate as ref

DATASETS=('lmo','tless','tudl','icbin','itodd','hb','ycbv')
FIELDS=('scene_id','im_id','obj_id','score','R','t','time')


class IdentityError(RuntimeError):
    """Shared source/configuration/cache corruption: never a scored case failure."""

FIXED={'dino_model':'dinov2_vitg14','dino_layer':30,'depth_tolerance_mm':1.0,
       'nms_translation_threshold_mm':10.0,'raw_points':5000,'min_views':18,
       'depth_sampling':'inverse_bilinear','pca_dim':64,'grid_size':16,'dense_size':3000,
       'top_k':10,'iterations':10000,'icp_max_iterations':30,'edge_similarity_threshold':.9,
       'seed':0,'precision':'FP32','TF32':False}


def utc():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_text(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('x',encoding='utf-8',newline='') as stream:
        stream.write(text);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def path_of(config,name):return ref.local(config['paths'][name],resolve=False)


def validate_configuration(config):
    if config.get('public_revision')!=ref.PUBLIC_REVISION:raise ValueError('Public revision differs')
    if config.get('datasets')!=list(DATASETS):raise ValueError('All seven datasets in fixed order are required')
    if any(config.get('settings',{}).get(k)!=v for k,v in FIXED.items()):raise ValueError('Fixed public inference settings differ')
    resources=config.get('resources',{})
    if resources.get('inference_workers')!=1 or resources.get('cpu_threads')!=4 or resources.get('disk_reserve_GiB')!=15:
        raise ValueError('Single GPU, CPU4 and 15GiB reserve are required')
    if config.get('no_test_GT_tuning') is not True or config.get('no_author_CSV_inference_inputs') is not True:
        raise ValueError('GT/author-prediction inference boundary is required')
    return config


def target_key(row):return tuple(int(row[k]) for k in ('scene_id','im_id','obj_id'))


def exact_targets(path):
    ref.target_counts(path)
    return [{k:int(row[k]) for k in ('scene_id','im_id','obj_id','inst_count')} for row in ref.read(path)]


def check_snapshot(output,config):
    file=Path(output)/'source_snapshot/source_manifest.json'
    if not file.is_file():raise ValueError('Seal source with launch_inference.py --prepare-only first')
    manifest=ref.read(file)
    if manifest['configuration_sha256']!=ref.digest(config):raise ValueError('Sealed configuration differs')
    for name,sha in manifest['files'].items():
        if ref.sha256(file.parent/name)!=sha:raise ValueError('Sealed orchestration source changed: '+name)
    if Path(__file__).resolve()!=file.parent/'suite_runner.py':raise ValueError('Execute the sealed suite_runner.py')
    return manifest


def context(config_path):
    config=validate_configuration(ref.read(config_path));output=path_of(config,'output')
    sealed=check_snapshot(output,config)
    root=Path(__file__).resolve().parents[4]
    # Snapshot: ROOT/outputs/<study>/<run>/source_snapshot/suite_runner.py
    if not (root/'AGENTS.md').exists():raise ValueError('Cannot resolve research repository from sealed snapshot')
    assets_file=path_of(config,'assets_output')/'assets_manifest.json';assets=ref.read(assets_file)
    if assets.get('GT_pose_access') is not False or assets.get('GT_annotation_links')!=0:
        raise ValueError('A complete GT-free inference view is required')
    if assets.get('status')!='verified_inference_assets_only_not_pose_performance':raise ValueError('Inference assets are not ready')
    source=ref.git_identity(path_of(config,'public_source'),ref.PUBLIC_REVISION)
    for checkpoint,label in ((assets['DINO_checkpoint'],'DINO'),(assets['GeDi_checkpoint'],'GeDi')):
        if ref.sha256(ref.local(checkpoint['path']))!=checkpoint['sha256']:raise ValueError(label+' checkpoint changed')
    for value in assets['DINO_source']['selected_files'].values():
        if ref.sha256(ref.local(value['path']))!=value['sha256']:raise ValueError('DINO source changed')
    item=assets['GeDi_source']['gedi_py']
    if ref.sha256(ref.local(item['path']))!=item['sha256']:raise ValueError('GeDi source changed')
    if path_of(config,'bop_root')!=ref.local(assets['bop_root'],resolve=False):raise ValueError('Inference asset view differs')
    if path_of(config,'compute_lock')!=path_of(config,'assets_output')/'compute.lock':raise ValueError('Shared compute lock differs')
    if Path(sys.prefix).resolve()!=path_of(config,'inference_python').parent.parent.resolve():raise ValueError('Use the recorded inference venv')
    durable,owned=ref.pinned_guards(root)
    pins={'configuration_sha256':ref.digest(config),'public_revision':ref.PUBLIC_REVISION,
          'public_tracked_tree_sha256':source['tracked_tree_sha256'],'assets_manifest_sha256':ref.sha256(assets_file),
          'orchestration_source_manifest_sha256':ref.sha256(output/'source_snapshot/source_manifest.json'),
          'DINO_checkpoint_sha256':assets['DINO_checkpoint']['sha256'],'GeDi_checkpoint_sha256':assets['GeDi_checkpoint']['sha256'],
          'frozen_guard_manifest_sha256':ref.FROZEN_MANIFEST_SHA256}
    manifest={'schema':'unchanged-public-own-inference-run-v1','source_pins':pins,'configuration':config,
              'public_source_identity':source,'sealed_orchestration':sealed,
              'inference_GT_access':False,'author_predictions_inference_inputs':False,
              'geometry_runtime_separate':True,'process_boundary':'one original localization call per official object-image target',
              'timing_policy':'onboarding separate; observed original target runtimes summed per image; process startup tracked separately'}
    old=output/'run_manifest.json'
    if old.exists() and ref.read(old)!=manifest:raise ValueError('Run identity differs; use a versioned output')
    durable.atomic_json(old,manifest)
    return {'root':root,'config':config,'output':output,'assets':assets,'pins':pins,'durable':durable,'owned':owned,'manifest':manifest}


def environment(ctx):
    config=ctx['config'];env=os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1',TORCH_HOME=str(path_of(config,'torch_home')),
               OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4',LP_NUM_THREADS='4',
               CUDA_VISIBLE_DEVICES='0',NVIDIA_TF32_OVERRIDE='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
               PYOPENGL_PLATFORM='egl',EGL_PLATFORM='surfaceless')
    for name in ('PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF'):env.pop(name,None)
    return env


@contextmanager
def wait_compute(ctx,lock_path=None):
    durable=ctx['durable'];path=Path(lock_path or path_of(ctx['config'],'compute_lock'))
    while True:
        manager=durable.worker_lock(path)
        try:stream=manager.__enter__()
        except RuntimeError as exc:
            if 'OS lock' not in str(exc) and 'live suite' not in str(exc):raise
            durable.atomic_json(ctx['output']/'status.json',{'status':'waiting_for_compute_lock','utc':utc(),'lock':str(path),
                                   'pid':os.getpid(),'start_ticks':ctx['owned'].start_ticks(os.getpid())})
            time.sleep(15);continue
        break
    try:yield stream
    finally:manager.__exit__(None,None,None)


def original_argv(ctx,command):
    config=ctx['config']
    return [str(path_of(config,'inference_python')),'-B',str(path_of(config,'public_source')/'run_bop.py'),*map(str,command)]


def feature_paths(ctx,dataset,obj):
    stem=f'{dataset}_obj_{obj:06d}'
    return {name:path_of(ctx['config'],'features_dir')/(stem+'_'+name+'.npz') for name in ('visual','gedi','query')}


def geometry_folder(ctx,dataset,obj):
    return path_of(ctx['config'],'geometry_root')/f'{dataset}_obj_{obj:06d}_raw{FIXED["raw_points"]}'


def feature_command(ctx,dataset,obj,stage):
    config=ctx['config'];s=config['settings'];files=feature_paths(ctx,dataset,obj)
    common=['--dataset',dataset,'--obj-id',str(obj)]
    geometry=geometry_folder(ctx,dataset,obj)/'geometry'
    if stage=='visual':
        return original_argv(ctx,['extract-query-visual',*common,'--layer',s['dino_layer'],'--depth-tolerance',s['depth_tolerance_mm'],
            '--device','cuda','--dino-model',s['dino_model'],'--facet','token','--min-views',s['min_views'],'--depth-sampling',s['depth_sampling'],
            '--onboarding-cache',geometry/'onboarding.npz','--rgb-dir',geometry/'rgb','--dinov2-root',path_of(config,'dinov2_root'),'--output',files['visual']])
    if stage=='gedi':
        return original_argv(ctx,['extract-query-gedi',*common,'--bop-root',path_of(config,'bop_root'),
            '--onboarding-cache',files['visual'],'--gedi-root',path_of(config,'gedi_root'),'--checkpoint',path_of(config,'gedi_checkpoint'),
            '--seed',s['seed'],'--output',files['gedi']])
    if stage=='fuse':
        return original_argv(ctx,['fuse-query-features',*common,'--visual',files['visual'],'--geometric',files['gedi'],
                                 '--pca-dim',s['pca_dim'],'--output',files['query']])
    raise ValueError('Unknown feature stage')


def validate_npz(ctx,dataset,obj,stage):
    import numpy as np
    files=feature_paths(ctx,dataset,obj);name={'visual':'visual','gedi':'gedi','fuse':'query'}[stage];path=files[name]
    with np.load(path,allow_pickle=False) as data:
        points=np.asarray(data['query_points'])
        feature=np.asarray(data[{'visual':'visual_features','gedi':'geometric_features','fuse':'fused_features'}[stage]])
        if points.ndim!=2 or points.shape[1]!=3 or not len(points) or not np.isfinite(points).all():raise ValueError('Invalid retained CAD points')
        if feature.ndim!=2 or len(feature)!=len(points) or not np.isfinite(feature).all():raise ValueError('Invalid aligned CAD descriptors')
        if stage=='visual':
            if int(data['dino_layer'])!=30 or str(data['dino_model'])!='dinov2_vitg14' or float(data['depth_tolerance'])!=1.0 or int(data['min_views'])!=18:
                raise IdentityError('CAD visual metadata differs from fixed settings')
        if stage!='visual':
            with np.load(files['visual'],allow_pickle=False) as visual:
                if not np.array_equal(points,visual['query_points']):raise IdentityError('Visual/GeDi point alignment differs')
        if stage=='gedi' and feature.shape[1]!=64:raise IdentityError('Original GeDi descriptor dimension differs')
        if stage=='fuse' and feature.shape[1]!=128:raise IdentityError('Original fused descriptor dimension differs')
    return {str(path):ref.sha256(path)}, {'retained_points':len(points),'feature_shape':list(feature.shape)}


def stage_folder(ctx,dataset,obj,stage):return ctx['output']/'stages'/dataset/f'obj_{obj:06d}'/stage


def receipt_reusable(receipt,pins,command):
    if receipt.get('source_pins')!=pins or receipt.get('command')!=command:raise IdentityError('Stage identity differs')
    if receipt.get('status') not in ('complete','failed','blocked'):return False
    for path,sha in receipt.get('outputs',{}).items():
        if not ref.local(path).is_file() or ref.sha256(ref.local(path))!=sha:raise IdentityError('Receipted payload changed')
    return receipt.get('status')=='complete'


def owned_stage(ctx,folder,command,lock,outputs_check,*,original_command=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);receipt_file=folder/'receipt.json';pins=ctx['pins']
    recorded=original_command or command
    if receipt_file.exists():
        receipt=ref.read(receipt_file)
        if receipt_reusable(receipt,pins,recorded):return receipt
        if receipt.get('status') in ('failed','blocked'):raise RuntimeError('Preserved prior stage failure: '+receipt.get('reason','unknown'))
        # An interrupted child has released its inherited flock before a resume
        # can enter this stage. Preserve any partial products for diagnosis.
        for output in receipt.get('expected_outputs',[]):
            path=Path(output)
            if path.exists():os.replace(path,folder/(path.name+'.interrupted.'+uuid.uuid4().hex))
    ctx['durable'].require_disk(ctx['output'],15)
    expected_outputs=[]
    if '--output' in recorded:
        candidate=recorded[recorded.index('--output')+1]
        if Path(candidate).suffix in ('.npz','.csv'):expected_outputs=[candidate]
    receipt={'status':'running','source_pins':pins,'command':recorded,'owned_command':command,'started_utc':utc(),'expected_outputs':expected_outputs}
    ctx['durable'].atomic_json(receipt_file,receipt);started=time.perf_counter()
    def heartbeat(pid):
        ctx['durable'].atomic_json(ctx['output']/'status.json',{'status':'running','utc':utc(),'stage':str(folder),
            'pid':os.getpid(),'start_ticks':ctx['owned'].start_ticks(os.getpid()),'guard_pid':pid,'guard_start_ticks':ctx['owned'].start_ticks(pid)})
    try:
        code=ctx['owned'].run_owned(command,environment(ctx),folder,lock,heartbeat)
        if code:raise RuntimeError('Original command returned '+str(code)+'; see '+str(folder/'execution.log'))
        outputs,diagnostics=outputs_check()
        if ref.git_identity(path_of(ctx['config'],'public_source'),ref.PUBLIC_REVISION)!=ctx['manifest']['public_source_identity']:
            raise IdentityError('Original public source changed during a stage')
        if 'target' in diagnostics:receipt.update(diagnostics)
        receipt.update(status='complete',outputs=outputs,diagnostics=diagnostics,runtime_s=time.perf_counter()-started,completed_utc=utc())
        ctx['durable'].atomic_json(receipt_file,receipt);return receipt
    except Exception as exc:
        receipt.update(status='failed',reason=type(exc).__name__+': '+str(exc),runtime_s=time.perf_counter()-started,completed_utc=utc())
        ctx['durable'].atomic_json(receipt_file,receipt);raise


def wrapped_original(ctx,folder,command):
    request=Path(folder)/'original_command.json';request.parent.mkdir(parents=True,exist_ok=True)
    ctx['durable'].atomic_json(request,{'command':command,'public_revision':ref.PUBLIC_REVISION,'source':str(path_of(ctx['config'],'public_source'))})
    return [str(path_of(ctx['config'],'inference_python')),'-B',str(Path(__file__).resolve()),'--internal-original',str(request)]


def geometry_stage(ctx,dataset,obj,suite_lock):
    # The unchanged helper CLI owns compute.lock itself, so the outer guard holds
    # suite.lock here. This avoids nested flock acquisition while preserving its
    # original source and the shared GPU lock for the entire renderer lifetime.
    folder=geometry_folder(ctx,dataset,obj);helper=Path(__file__).with_name('onboard_original.py')
    command=[str(path_of(ctx['config'],'evaluation_python')),'-B',str(helper),'--root',str(ctx['root']),
             '--dataset',dataset,'--obj-id',str(obj),'--raw-points','5000','--output',str(folder)]
    stage=stage_folder(ctx,dataset,obj,'geometry')
    def check():
        import onboard_original as onboard
        record=ref.read(folder/'geometry_receipt.json')
        if record.get('status')!='complete':raise ValueError('Geometry receipt incomplete')
        expected={'dataset':dataset,'obj_id':obj,'raw_query_points':5000,'seed':0,'template_count':162,'render_size':480,'target_fill':.5}
        if any(record['source_pins']['protocol'].get(k)!=v for k,v in expected.items()):raise IdentityError('Geometry protocol differs')
        if record['source_pins']['helper_sha256']!=ref.sha256(helper):raise IdentityError('Original geometry helper differs')
        model=path_of(ctx['config'],'bop_root')/dataset/'models'/f'obj_{obj:06d}.ply'
        if ref.sha256(model)!=record['source_pins']['model_sha256']:raise IdentityError('CAD model changed after geometry')
        if record['source_pins']['public_source']['tracked_tree_sha256']!=ctx['manifest']['public_source_identity']['tracked_tree_sha256']:
            raise IdentityError('Geometry public source differs')
        try:payload=onboard.validate_saved(folder/'geometry',5000)
        except (ValueError,FileNotFoundError) as exc:raise IdentityError('Geometry payload is invalid') from exc
        if payload!=record['payload_sha256']:raise IdentityError('Geometry bytes differ')
        outputs={str(folder/'geometry_receipt.json'):ref.sha256(folder/'geometry_receipt.json')}
        outputs.update({str(folder/'geometry'/p):sha for p,sha in record['payload_sha256'].items()})
        return outputs,{'original_geometry_receipt':str(folder/'geometry_receipt.json'),'raw_points':5000,'templates':162,'geometry_runtime_s':record['runtime_s']}
    # Existing original geometry produced by the live parent probe is adopted only
    # after its real shared compute flock has ended and every payload is verified.
    with wait_compute(ctx):
        if (folder/'geometry_receipt.json').is_file() and ref.read(folder/'geometry_receipt.json').get('status')=='complete' and not (stage/'receipt.json').exists():
            outputs,diagnostics=check();receipt={'status':'complete','source_pins':ctx['pins'],'command':command,'outputs':outputs,
                'diagnostics':diagnostics,'runtime_s':diagnostics['geometry_runtime_s'],'adopted_original_geometry':True}
            ctx['durable'].atomic_json(stage/'receipt.json',receipt);return receipt
    # Another short-lived observer can race for compute.lock; unchanged helper
    # fails closed rather than run a second renderer. No live lock is removed.
    return owned_stage(ctx,stage,command,suite_lock,check)


def feature_stage(ctx,dataset,obj,stage,compute):
    folder=stage_folder(ctx,dataset,obj,stage);command=feature_command(ctx,dataset,obj,stage)
    output=feature_paths(ctx,dataset,obj)[{'visual':'visual','gedi':'gedi','fuse':'query'}[stage]]
    if output.exists() and not (folder/'receipt.json').exists():raise IdentityError('Unreceipted feature exists; use explicit --adopt-stage with recorded command')
    return owned_stage(ctx,folder,wrapped_original(ctx,folder,command),compute,
                       lambda:validate_npz(ctx,dataset,obj,stage),original_command=command)


def target_folder(ctx,dataset,target):
    s,i,o=target_key(target)
    return ctx['output']/'datasets'/dataset/'targets'/f's{s:06d}_i{i:06d}_o{o:06d}'


def localization_command(ctx,dataset,target,folder):
    c=ctx['config'];s=c['settings'];single=folder/'target.json'
    ctx['durable'].atomic_json(single,[target])
    command=['run-localization','--dataset',dataset,'--bop-root',path_of(c,'bop_root'),'--split','test',
        '--query-cache-dir',path_of(c,'features_dir'),'--detection-json',ref.local(ctx['assets']['datasets'][dataset]['masks']['path']),
        '--targets-json',single,'--nms-translation-threshold-mm',s['nms_translation_threshold_mm'],
        '--dinov2-root',path_of(c,'dinov2_root'),'--gedi-root',path_of(c,'gedi_root'),'--checkpoint',path_of(c,'gedi_checkpoint'),
        '--device','cuda','--grid-size',s['grid_size'],'--dense-size',s['dense_size'],'--top-k',s['top_k'],'--iterations',s['iterations'],
        '--seed',s['seed'],'--edge-similarity-threshold',s['edge_similarity_threshold'],'--icp-max-iterations',s['icp_max_iterations'],
        '--work-dir',folder/'original','--output',folder/'original.csv']
    return original_argv(ctx,command)


def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as stream:
        reader=csv.DictReader(stream)
        if tuple(reader.fieldnames or ())!=FIELDS:raise ValueError('Original BOP CSV header differs')
        rows=list(reader)
    for row in rows:
        r=list(map(float,row['R'].split()));t=list(map(float,row['t'].split()))
        if len(r)!=9 or len(t)!=3 or not all(math.isfinite(v) for v in [*r,*t,float(row['score']),float(row['time'])]):
            raise ValueError('Original BOP CSV pose is invalid')
    return rows


def validate_target(folder,target):
    s,i,o=target_key(target);folder=Path(folder);csv_path=folder/'original.csv'
    result=folder/'original'/f'scene_{s:06d}_im_{i:06d}_obj_{o:06d}'/'result.json'
    record=ref.read(result);rows=read_csv(csv_path);selected=record['selected']
    if any(int(record[k])!=target[k] for k in ('scene_id','im_id','obj_id')) or record['instance_count']!=target['inst_count']:
        raise IdentityError('Original selected-pose target differs')
    if len(rows)!=len(selected) or len(selected)>target['inst_count']:raise IdentityError('Original selection count differs')
    poses=[]
    for row,pose in zip(rows,selected):
        if target_key(row)!=target_key(target):raise IdentityError('Original CSV contains another target')
        values=[pose['final_score'],*(v for line in pose['R'] for v in line),*pose['t_mm']]
        serialized=[float(row['score']),*map(float,row['R'].split()),*map(float,row['t'].split())]
        if any(float(format(a,'.9g'))!=b for a,b in zip(values,serialized)):raise IdentityError('CSV pose differs from original selected result')
        poses.append({'score':float(pose['final_score']),'R':pose['R'],'t_mm':pose['t_mm'],'time':float(row['time'])})
    outputs={str(csv_path):ref.sha256(csv_path),str(result):ref.sha256(result)}
    # Candidate masks and all coarse/refined/observed caches remain inside this
    # target directory. Hash them in its completion receipt for failure tracing.
    for file in sorted((folder/'original').rglob('*')):
        if file.is_file():outputs[str(file)]=ref.sha256(file)
    return outputs,{'target':target,'poses':poses,'selected_count':len(poses),'original_csv':str(csv_path),
                    'original_csv_sha256':outputs[str(csv_path)],'original_result':str(result),
                    'candidate_count':record['candidate_count'],'valid_candidate_count':record['valid_candidate_count']}


def target_stage(ctx,dataset,target,compute,blocked=None):
    folder=target_folder(ctx,dataset,target);file=folder/'receipt.json';folder.mkdir(parents=True,exist_ok=True)
    if blocked:
        if file.exists():
            old=ref.read(file)
            if old.get('source_pins')!=ctx['pins'] or old.get('target')!=target:raise IdentityError('Blocked receipt identity differs')
            return old
        row={'status':'blocked','target':target,'source_pins':ctx['pins'],'reason':blocked,'selected_count':0,'poses':[],
             'original_csv':None,'original_result':None,'runtime_s':0,'completed_utc':utc()}
        ctx['durable'].atomic_json(file,row);return row
    command=localization_command(ctx,dataset,target,folder)
    if file.exists():
        old=ref.read(file)
        if old.get('target')!=target:raise IdentityError('Existing target identity differs')
        if old.get('status') in ('complete','failed','blocked'):
            receipt_reusable(old,ctx['pins'],command)
            return old
    try:
        record=owned_stage(ctx,folder,wrapped_original(ctx,folder,command),compute,
                           lambda:validate_target(folder,target),original_command=command)
        details=record.pop('diagnostics');record.update(details,target=target)
        ctx['durable'].atomic_json(file,record);return record
    except IdentityError:raise
    except Exception as exc:
        record=ref.read(file);record.update(target=target,selected_count=0,poses=[],original_csv=None,original_result=None,reason=str(exc))
        ctx['durable'].atomic_json(file,record);return record


def export_dataset(ctx,dataset,targets,records):
    directory=ctx['output']/'datasets'/dataset;directory.mkdir(parents=True,exist_ok=True)
    indexed={target_key(r['target']):r for r in records}
    if len(indexed)!=len(records) or set(indexed)-set(map(target_key,targets)):raise IdentityError('Unknown or duplicate target receipts')
    image_times={}
    for record in records:
        key=target_key(record['target'])[:2]
        # Original localization time excludes process startup; no-pose failures
        # have no CSV time, so their guarded attempt runtime is kept explicitly.
        timing=record['poses'][0]['time'] if record.get('poses') else record.get('runtime_s',0)
        image_times[key]=image_times.get(key,0)+max(0,float(timing))
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=FIELDS);writer.writeheader();ledger=[]
    for target in targets:
        record=indexed.get(target_key(target))
        if record is None:continue
        folder=target_folder(ctx,dataset,target);poses=[]
        for pose in record.get('poses',[]):
            pose=dict(pose);pose['time']=image_times[target_key(target)[:2]];poses.append(pose)
        for pose in poses:
            # Same write_bop_csv/_fmt formatting as the original public source.
            # The independent evaluator reopens the original CSV/results once;
            # incremental export does no quadratic filesystem rereading.
            writer.writerow({'scene_id':target['scene_id'],'im_id':target['im_id'],'obj_id':target['obj_id'],
                'score':format(float(pose['score']),'.9g'),
                'R':' '.join(format(float(v),'.9g') for line in pose['R'] for v in line),
                't':' '.join(format(float(v),'.9g') for v in pose['t_mm']),
                'time':format(float(pose['time']),'.9g')})
        ledger.append({**target,'status':record['status'],'reason':record.get('reason'),
            'selected_count':len(poses),'poses':poses,'original_csv':record.get('original_csv'),
            'original_csv_sha256':record.get('original_csv_sha256'),'original_result':record.get('original_result'),
            'target_receipt':str(folder/'receipt.json'),'target_receipt_sha256':record.get('_receipt_sha256') or ref.sha256(folder/'receipt.json')})
    csv_path=directory/f'freezev2public_{dataset}-test.csv';atomic_text(csv_path,stream.getvalue())
    final=len(ledger)==len(targets)
    value={'schema':'unchanged-public-full-target-ledger-v1','dataset':dataset,'source_pins':ctx['pins'],
        'status':'all_targets_recorded' if final else 'in_progress','expected_object_image_targets':len(targets),
        'requested_instances':sum(t['inst_count'] for t in targets),'recorded_object_image_targets':len(ledger),
        'targets':ledger,'csv':str(csv_path),'csv_sha256':ref.sha256(csv_path),
        'timing_policy':'sum original per-target localization times per image; guarded attempt runtime used for zero-pose targets; startup/runtime limitations recorded',
        'counts':{status:sum(r['status']==status for r in records) for status in ('complete','failed','blocked')},
        'pose_count':sum(len(r.get('poses',[])) for r in records),'updated_utc':utc()}
    ctx['durable'].atomic_json(directory/'target_ledger.json',value)
    return value


def run_dataset(ctx,dataset,suite_lock):
    c=ctx['config'];asset=ctx['assets']['datasets'][dataset]
    target_path=path_of(c,'bop_root')/dataset/'test_targets_bop19.json'
    if ref.sha256(target_path)!=asset['target_sha256']:raise IdentityError('Official target list changed')
    masks=asset.get('masks')
    if not masks or ref.sha256(ref.local(masks['path']))!=masks['actual_sha256']:raise IdentityError('Recorded predicted masks missing/changed')
    targets=exact_targets(target_path);blocked={};onboarding=[]
    for obj in asset['required_object_ids']:
        try:
            geometry=geometry_stage(ctx,dataset,obj,suite_lock)
            stages=[]
            with wait_compute(ctx) as compute:
                for stage in ('visual','gedi','fuse'):stages.append(feature_stage(ctx,dataset,obj,stage,compute))
            onboarding.append({'obj_id':obj,'status':'complete','geometry':geometry['runtime_s'],'feature_stages':sum(r['runtime_s'] for r in stages)})
        except IdentityError:raise
        except Exception as exc:
            blocked[obj]=type(exc).__name__+': '+str(exc);onboarding.append({'obj_id':obj,'status':'failed','reason':blocked[obj]})
        ctx['durable'].atomic_json(ctx['output']/'datasets'/dataset/'onboarding_summary.json',{'objects':onboarding,'blocked_objects':blocked,'source_pins':ctx['pins']})
    records=[]
    for target in targets:
        with wait_compute(ctx) as compute:
            record=target_stage(ctx,dataset,target,compute,blocked.get(target['obj_id']))
            record['_receipt_sha256']=ref.sha256(target_folder(ctx,dataset,target)/'receipt.json')
            records.append(record)
        # Incremental official-format CSV and full target accounting survive exit.
        export_dataset(ctx,dataset,targets,records)
    ledger=export_dataset(ctx,dataset,targets,records)
    from evaluate_own import evaluate_dataset
    try:
        with wait_compute(ctx) as compute:
            evaluation=evaluate_dataset(ctx['root'],ctx['output'],dataset,Path(ledger['csv']),c,
                ctx['output']/'datasets'/dataset/'target_ledger.json',compute,durable=ctx['durable'],owned=ctx['owned'])
    except ValueError as exc:raise IdentityError('Evaluation artifact/source identity failed: '+str(exc)) from exc
    except Exception as exc:evaluation={'status':'failed_evaluation','reason':type(exc).__name__+': '+str(exc)}
    result={'dataset':dataset,'status':'all_targets_recorded','counts':ledger['counts'],'pose_count':ledger['pose_count'],
            'expected_object_image_targets':len(targets),'requested_instances':ledger['requested_instances'],
            'evaluation':evaluation,'onboarding':onboarding,'source_pins':ctx['pins']}
    ctx['durable'].atomic_json(ctx['output']/'datasets'/dataset/'summary.json',result);return result


def adopt_stage(ctx,dataset,obj,stage,command_file,runtime_s):
    expected=feature_command(ctx,dataset,obj,stage);actual=ref.read(command_file)
    if isinstance(actual,dict):actual=actual['command']
    # Argument ordering and an omitted explicit default do not alter the
    # original argparse semantics. No additional or duplicated flags are allowed.
    def payload(command):
        script=str(path_of(ctx['config'],'public_source')/'run_bop.py')
        if command[:2]!=expected[:2] or command[2]!=script:raise ValueError('Probe executable differs')
        result={'subcommand':command[3]};tokens=command[4:]
        if len(tokens)%2:raise ValueError('Probe flags are malformed')
        for flag,value in zip(tokens[::2],tokens[1::2]):
            if not flag.startswith('--') or flag in result:raise ValueError('Probe flags are duplicated/malformed')
            result[flag]=value
        if stage=='visual':result.setdefault('--facet','token')
        return result
    if payload(actual)!=payload(expected):raise ValueError('Adopted command differs from fixed original CLI')
    probe_file=Path(command_file).parent/'receipt.json'
    probe=ref.read(probe_file)
    if probe.get('status')!='complete' or probe.get('returncode')!=0 or probe.get('command')!=actual:
        raise ValueError('Original probe receipt is not a completed command')
    probe_pins=probe.get('source_pins',{})
    if probe_pins.get('configuration')!=ctx['config'] or probe_pins.get('public_source')!=ctx['manifest']['public_source_identity']:
        raise ValueError('Original probe source/configuration identity differs')
    if probe_pins.get('assets_manifest_sha256')!=ctx['pins']['assets_manifest_sha256']:
        raise ValueError('Original probe asset identity differs')
    if runtime_s and float(runtime_s)!=float(probe['runtime_s']):raise ValueError('Probe runtime differs from recorded receipt')
    runtime_s=float(probe['runtime_s'])
    folder=stage_folder(ctx,dataset,obj,stage)
    if (folder/'receipt.json').exists():raise ValueError('A stage receipt already exists')
    with wait_compute(ctx):
        outputs,diagnostics=validate_npz(ctx,dataset,obj,stage)
        if outputs!=probe.get('outputs'):raise ValueError('Probe output checksum differs')
        receipt={'status':'complete','source_pins':ctx['pins'],'command':expected,'recorded_probe_command':actual,
            'outputs':outputs,'diagnostics':diagnostics,'runtime_s':float(runtime_s),'adopted_original_probe':True,
            'probe_command_sha256':ref.sha256(command_file),'completed_utc':utc()}
        ctx['durable'].atomic_json(folder/'receipt.json',receipt)
    return receipt


def internal_original(request):
    value=ref.read(request);source=Path(value['source']);command=value['command']
    if value['public_revision']!=ref.PUBLIC_REVISION:raise ValueError('Internal original revision differs')
    if command[2]!=str(source/'run_bop.py'):raise ValueError('Unexpected original executable')
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_default_dtype(torch.float32)
    sys.path.insert(0,str(source));sys.argv=[str(source/'run_bop.py'),*command[3:]]
    runpy.run_path(str(source/'run_bop.py'),run_name='__main__')


def run(config):
    ctx=context(config);out=ctx['output'];results=[]
    with ctx['durable'].worker_lock(out/'suite.lock') as suite_lock:
        ctx['durable'].atomic_json(out/'linux_process.json',{'pid':os.getpid(),'start_ticks':ctx['owned'].start_ticks(os.getpid()),
            'actual_cmdline':Path('/proc/self/cmdline').read_bytes().decode().rstrip('\0').split('\0'),'started_utc':utc()})
        for dataset in DATASETS:
            try:result=run_dataset(ctx,dataset,suite_lock)
            except IdentityError:raise
            except Exception as exc:
                result={'dataset':dataset,'status':'blocked_dataset','reason':type(exc).__name__+': '+str(exc),'metrics':None}
                ctx['durable'].atomic_json(out/'datasets'/dataset/'summary.json',result)
            results.append(result)
            from evaluate_own import write_summary
            write_summary(ctx['root'],out,durable=ctx['durable'])
            ctx['durable'].atomic_json(out/'results_summary.json',{'status':'running','source_pins':ctx['pins'],'results':results,
                'seven_dataset_aggregate':None,'GT_private':['itodd','hb'],'updated_utc':utc()})
        final={'status':'suite_pass_finished','source_pins':ctx['pins'],'results':results,'seven_dataset_aggregate':None,
               'GT_private':['itodd','hb'],'paper_performance_reproduced':False,'completed_utc':utc()}
        write_summary(ctx['root'],out,durable=ctx['durable'])
        ctx['durable'].atomic_json(out/'results_summary.json',final);ctx['durable'].atomic_json(out/'status.json',final)
        return final


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--internal-original',type=Path)
    parser.add_argument('--adopt-stage',choices=('visual','gedi','fuse'))
    parser.add_argument('--dataset',choices=DATASETS);parser.add_argument('--obj-id',type=int)
    parser.add_argument('--command-json',type=Path);parser.add_argument('--runtime-s',type=float,default=0)
    args=parser.parse_args(argv)
    if args.internal_original:internal_original(args.internal_original);return 0
    if args.config is None:parser.error('--config is required')
    if args.adopt_stage:
        if args.dataset is None or args.obj_id is None or args.command_json is None:parser.error('Adoption requires dataset, object and recorded command')
        ctx=context(args.config)
        with ctx['durable'].worker_lock(ctx['output']/'suite.lock'):
            result=adopt_stage(ctx,args.dataset,args.obj_id,args.adopt_stage,args.command_json,args.runtime_s)
    else:result=run(args.config)
    print(json.dumps(result,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
