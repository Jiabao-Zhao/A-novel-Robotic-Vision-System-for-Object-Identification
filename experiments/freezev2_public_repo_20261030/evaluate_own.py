"""Evaluate only this study's generated poses with the unchanged public CLI.

All official localization targets, including blocked/failed rows with no pose,
remain in the evaluator denominator. Author submission CSVs are never inputs.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import datetime as dt
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
import reference_gate as ref

DATASETS = ('lmo','tless','tudl','icbin','itodd','hb','ycbv')
TERMINAL = ('complete','failed','blocked')
METRICS = tuple(ref.METRICS.values())


def own_path(value, output, label):
    path = ref.local(value)
    if not path.is_relative_to(Path(output).resolve()):
        raise ValueError(label + ' must remain inside this generated-inference output')
    if any(part.lower() in ('author_predictions','reference') for part in path.parts) or path.name.startswith('freezev21_'):
        raise ValueError('Author/reference predictions cannot count as generated inference')
    return path


def key(row):
    names = ('scene_id','im_id','obj_id')
    if any(isinstance(row.get(k),bool) or not isinstance(row.get(k),int) for k in names):
        raise ValueError('Target IDs must be integers')
    result = tuple(row[k] for k in names)
    if min(result) < 0 or result[2] < 1: raise ValueError('Invalid target ID')
    return result


def pose_signature(target, score, rotation, translation, seconds):
    R = np.asarray(rotation, dtype=np.float64).reshape(3,3)
    t = np.asarray(translation, dtype=np.float64).reshape(3)
    numbers = [float(score),float(seconds),*R.reshape(-1),*t]
    if not all(math.isfinite(x) for x in numbers): raise ValueError('Nonfinite saved/CSV pose')
    if float(seconds) < 0 and float(seconds) != -1: raise ValueError('Invalid BOP runtime marker')
    if not np.allclose(R @ R.T,np.eye(3),atol=1e-4,rtol=0) or not np.isclose(np.linalg.det(R),1,atol=1e-4,rtol=0):
        raise ValueError('BOP rotation is not a proper rigid rotation')
    # Exact public write_bop_csv/_fmt precision, not a fitted pose tolerance.
    return (*target,*(format(x,'.9g') for x in numbers))


def geometry_signature(target, pose):
    return pose_signature(target,pose['score'],pose['R'],pose['t_mm'],0)


def checked_original_rows(source, csv_path):
    for name,module in list(sys.modules.items()):
        if name == 'freezev2' or name.startswith('freezev2.'):
            path = getattr(module,'__file__',None)
            if path and not Path(path).resolve().is_relative_to(Path(source).resolve()):
                raise RuntimeError('Use a fresh process for pinned original BOP validation')
    if str(source) not in sys.path: sys.path.insert(0,str(source))
    original = importlib.import_module('freezev2.bop')
    with Path(csv_path).open(newline='',encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['scene_id','im_id','obj_id','score','R','t','time']:
            raise ValueError('Unexpected generated BOP CSV header')
    return original.read_bop_csv(csv_path)


def audit_ledger(output,dataset,csv_path,ledger_path,targets_path,pins,source):
    output = Path(output).resolve()
    csv_path = own_path(csv_path,output,'CSV'); ledger_path = own_path(ledger_path,output,'Target ledger')
    if csv_path.name != f'freezev2public_{dataset}-test.csv':
        raise ValueError('Generated CSV must have its own method/dataset BOP filename')
    ledger = ref.read(ledger_path)
    if ledger.get('dataset') != dataset or ledger.get('status') != 'all_targets_recorded':
        raise ValueError('Inference ledger has not recorded every official target')
    required_pins = {name:pins[name] for name in ('configuration_sha256','public_revision','public_tracked_tree_sha256')}
    if any(ledger.get('source_pins',{}).get(name) != value for name,value in required_pins.items()):
        raise ValueError('Target ledger source/configuration identity differs')
    if own_path(ledger.get('csv',''),output,'Ledger CSV') != csv_path or ledger.get('csv_sha256') != ref.sha256(csv_path):
        raise ValueError('Ledger generated CSV identity differs')
    official = ref.read(targets_path); ref.target_counts(targets_path)
    wanted = {key(row):int(row['inst_count']) for row in official}
    if ledger.get('expected_object_image_targets') != len(wanted) or ledger.get('requested_instances') != sum(wanted.values()):
        raise ValueError('Ledger denominator differs from official target/instance counts')
    seen = set(); expected = []; receipt_hashes = {}; original_csv_hashes = {}
    counts = Counter(); with_poses = 0; image_times = {}; target_times_unavailable = 0; ledger_pose_times = []
    for row in ledger.get('targets',[]):
        target = key(row)
        if target in seen or target not in wanted: raise ValueError('Duplicate or unknown target in ledger')
        seen.add(target)
        if isinstance(row.get('inst_count'),bool) or row.get('inst_count') != wanted[target]:
            raise ValueError('Recorded requested instances differ from official target')
        status = row.get('status')
        if status not in TERMINAL: raise ValueError('Pending/missing target receipt cannot count complete')
        counts[status] += 1
        poses = row.get('poses')
        if not isinstance(poses,list) or row.get('selected_count') != len(poses) or len(poses) > wanted[target]:
            raise ValueError('Selected pose count differs from saved target ledger')
        with_poses += bool(poses)
        receipt = own_path(row.get('target_receipt',''),output,'Target receipt')
        receipt_sha = ref.sha256(receipt)
        if receipt_sha != row.get('target_receipt_sha256'): raise ValueError('Target receipt checksum differs')
        receipt_hashes[str(receipt)] = receipt_sha
        saved = ref.read(receipt)
        if saved.get('status') != status or saved.get('target') != {name:row[name] for name in ('scene_id','im_id','obj_id','inst_count')}:
            raise ValueError('Target receipt target/status differs from exact ledger')
        if any(saved.get('source_pins',{}).get(name) != value for name,value in required_pins.items()):
            raise ValueError('Target receipt source/configuration identity differs')
        saved_poses = saved.get('poses',[])
        if saved.get('selected_count') != len(poses) or Counter(geometry_signature(target,p) for p in saved_poses) != Counter(geometry_signature(target,p) for p in poses):
            raise ValueError('Target receipt poses differ from ledger')
        timing = float(saved_poses[0]['time']) if saved_poses else float(saved.get('runtime_s',0))
        if not math.isfinite(timing): raise ValueError('Nonfinite target runtime')
        if timing < 0: target_times_unavailable += 1
        image_times[target[:2]] = image_times.get(target[:2],0) + max(0,timing)
        ledger_pose_times.extend((target[:2],float(p['time'])) for p in poses)
        for name in ('original_csv','original_csv_sha256','original_result'):
            if saved.get(name) != row.get(name): raise ValueError('Target receipt artifact identity differs')
        expected.extend(pose_signature(target,p['score'],p['R'],p['t_mm'],p['time']) for p in poses)
        original_csv = row.get('original_csv')
        if original_csv is None:
            if poses: raise ValueError('Saved poses have no original public generated CSV')
        else:
            original_csv = own_path(original_csv,output,'Original generated CSV')
            actual_sha = ref.sha256(original_csv)
            if actual_sha != row.get('original_csv_sha256'): raise ValueError('Original target CSV changed')
            original_csv_hashes[str(original_csv)] = actual_sha
            rows = checked_original_rows(source,original_csv)
            signatures = [pose_signature((int(r['scene_id']),int(r['im_id']),int(r['obj_id'])),float(r['score']),
                [float(x) for x in r['R'].split()],[float(x) for x in r['t'].split()],float(r['time'])) for r in rows]
            selected = [pose_signature(target,p['score'],p['R'],p['t_mm'],p['time']) for p in saved_poses]
            if Counter(signatures) != Counter(selected): raise ValueError('Ledger poses differ from original public target CSV')
        original_result = row.get('original_result')
        if original_result is not None:
            result = own_path(original_result,output,'Original public result')
            receipt_hashes[str(result)] = ref.sha256(result)
            selected = ref.read(result).get('selected')
            if not isinstance(selected,list): raise ValueError('Original multi-mask selected pose list is missing')
            original_geometry = [pose_signature(target,p['final_score'],p['R'],p['t_mm'],0) for p in selected]
            if Counter(original_geometry) != Counter(geometry_signature(target,p) for p in saved_poses):
                raise ValueError('Saved poses differ from original multi-mask selected geometry')
    for image,seconds in ledger_pose_times:
        if format(seconds,'.9g') != format(image_times[image],'.9g'):
            raise ValueError('Aggregate image runtime differs from recorded original/failed target attempts')
    if seen != set(wanted): raise ValueError('Missing official targets; failed rows must remain recorded')
    actual = []
    for row in checked_original_rows(source,csv_path):
        target = tuple(int(row[k]) for k in ('scene_id','im_id','obj_id'))
        if target not in wanted: raise ValueError('CSV contains an unknown official target')
        actual.append(pose_signature(target,float(row['score']),[float(x) for x in row['R'].split()],
            [float(x) for x in row['t'].split()],float(row['time'])))
    if Counter(actual) != Counter(expected): raise ValueError('Aggregate CSV poses differ from exact saved target poses')
    return {'official_object_image_targets':len(wanted),'requested_instances':sum(wanted.values()),
        'recorded_targets':len(seen),'missing_targets':0,'terminal_status_counts':dict(counts),
        'targets_with_saved_poses':with_poses,'targets_without_saved_poses':len(seen)-with_poses,
        'saved_poses':len(actual),'failed_and_blocked_remain_in_denominator':True,
        'target_receipts_sha256':receipt_hashes,'original_generated_CSVs_sha256':original_csv_hashes,
        'original_public_BOP_CSV_validation':True,
        'inference_runtime':ledger.get('runtime_summary'),
        'image_runtime_s_by_scene_image':{f'{k[0]:06d}/{k[1]:06d}':v for k,v in sorted(image_times.items())},
        'target_runtime_unavailable_markers':target_times_unavailable,
        'CSV_time_policy':'Per image: sum original per-target CSV runtime when poses exist; otherwise guarded target attempt runtime, clipped below at zero. Process startup excluded when original CSV time exists; blocked targets contribute zero.',
        'runtime_note':'Pose CSV times are preserved; missing CAD/inference timing is not inferred from evaluator wall time.'}


def configuration_pins(root,configuration):
    if not isinstance(configuration,dict): raise ValueError('Pass the canonical inference configuration dictionary')
    path = Path(root)/'experiments/freezev2_public_repo_20261030/public_inference_v1.json'
    if ref.read(path) != configuration: raise ValueError('Passed configuration differs from canonical experiment manifest')
    return {'configuration_sha256':ref.digest(configuration),'configuration_file':str(path),
            'configuration_file_sha256':ref.sha256(path)}


def context(root,output,dataset,configuration):
    root,output = ref.local(root),ref.local(output)
    if dataset not in DATASETS: raise ValueError('Unknown Classic-Core dataset')
    old = root/'outputs/freezev2_bop_20261029'
    runtime_path = old/'evaluation_runtime.json'; runtime = ref.read(runtime_path)
    source = old/'external/FreezeV2-Re'; original = ref.git_identity(source,ref.PUBLIC_REVISION)
    toolkit = ref.local(runtime['toolkit_root']); toolkit_identity = ref.git_identity(toolkit,ref.TOOLKIT_REVISION)
    if runtime.get('toolkit_revision') != ref.TOOLKIT_REVISION or runtime.get('renderer_type') != 'vispy' or runtime.get('protocol',{}).get('visib_gt_min') != -1:
        raise ValueError('Evaluator must use pinned BOP19 localization protocol')
    targets = ref.local(runtime['dataset_view'])/dataset/'test_targets_bop19.json'
    pins = {**configuration_pins(root,configuration),'public_revision':ref.PUBLIC_REVISION,
        'public_tracked_tree_sha256':original['tracked_tree_sha256'],'toolkit_revision':ref.TOOLKIT_REVISION,
        'toolkit_tracked_tree_sha256':toolkit_identity['tracked_tree_sha256'],
        'official_targets_sha256':ref.sha256(targets),'evaluation_runtime_sha256':ref.sha256(runtime_path),
        'evaluation_helper_sha256':ref.sha256(__file__),'frozen_guards_manifest_sha256':ref.FROZEN_MANIFEST_SHA256}
    return {'root':root,'output':output,'dataset':dataset,'runtime':runtime,'source':source,'toolkit':toolkit,
        'source_identity':original,'toolkit_identity':toolkit_identity,'targets':targets,'pins':pins}


def command_environment(ctx,csv_path,folder):
    runtime = ctx['runtime']; python = str(ref.evaluation_python(runtime))
    command = [python,str(ctx['source']/'run_bop.py'),'evaluate-submission','--dataset',ctx['dataset'],
        '--result-csv',str(csv_path),'--bop-root',str(ref.local(runtime['dataset_view'])),
        '--bop-toolkit',str(ctx['toolkit']),'--eval-root',str(folder/'official_evaluation'),'--num-workers','1']
    env = os.environ.copy(); env.update(runtime.get('environment',{}))
    env.update(BOP_PATH=str(ref.local(runtime['dataset_view'])),PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=str(ctx['toolkit'])+os.pathsep+str(ctx['source']),
               PATH=str(Path(python).parent)+os.pathsep+env.get('PATH',''))
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): env[name]='4'
    return command,env


def pose_metrics(scores):
    result = {name:float(scores[name]) for name in METRICS}
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in result.values()):
        raise ValueError('Official pose recalls are nonfinite or outside[0,1]')
    return result


def evaluate_dataset(root,output,dataset,csv_path,configuration,ledger_path,lock_stream,durable=None,owned=None):
    """Called by suite_runner while it owns base_public_v1/compute.lock."""
    ctx = context(root,output,dataset,configuration)
    if durable is None or owned is None: durable,owned = ref.pinned_guards(ctx['root'])
    if lock_stream is None or lock_stream.closed: raise ValueError('A live inherited shared compute lock is required')
    os.fstat(lock_stream.fileno())
    required_lock = ctx['root']/'outputs/freezev2_public_repo_20261030/base_public_v1/compute.lock'
    if Path(lock_stream.name).resolve() != required_lock.resolve(): raise ValueError('Evaluator received a different compute lock')
    csv_path = own_path(csv_path,ctx['output'],'Generated CSV'); ledger_path = own_path(ledger_path,ctx['output'],'Ledger')
    audit = audit_ledger(ctx['output'],dataset,csv_path,ledger_path,ctx['targets'],ctx['pins'],ctx['source'])
    pins = {**ctx['pins'],'CSV_sha256':ref.sha256(csv_path),'target_ledger_sha256':ref.sha256(ledger_path),
            'ledger_artifacts_sha256':ref.digest({'receipts':audit['target_receipts_sha256'],'original_CSVs':audit['original_generated_CSVs_sha256']})}
    folder = ctx['output']/'datasets'/dataset; folder.mkdir(parents=True,exist_ok=True)
    receipt_path = folder/'own_evaluation_receipt.json'
    score_path = folder/'official_evaluation'/csv_path.stem/'scores_bop19.json'
    old = ref.read(receipt_path) if receipt_path.exists() else {}
    if old.get('source_pins') and old['source_pins'] != pins:
        raise ValueError('Saved evaluator source/configuration/CSV identity differs; use a separate output version')
    if old.get('status') == 'complete' and score_path.is_file() and ref.sha256(score_path) == old.get('score_sha256') and pose_metrics(ref.read(score_path)) == old.get('metrics'):
        return old
    entry = {'schema':'unchanged-public-own-prediction-evaluation-v1','dataset':dataset,'source_pins':pins,
        'prediction_origin':'our inference generated by the pinned unmodified public FreeZeV2-Re code',
        'authors_predictions_used':False,'CSV':str(csv_path),'ledger':str(ledger_path),'coverage':audit,
        'protocol':{'task':'BOP19 model-based 6D localization','split':'test','targets_filename':'test_targets_bop19.json',
            'visib_gt_min':-1,'instance_policy':'official inst_count most-visible GT; all recorded failures stay in denominator'},
        'GT_role':'evaluation only; never inference initialization/refinement/ranking/tuning',
        'claim_scope':'Generated own predictions; no claim of reproducing published performance'}
    state = ctx['runtime']['datasets'].get(dataset,{})
    if dataset in ref.PRIVATE_DATASETS or state.get('GT_withheld'):
        entry.update(status='metrics_unavailable_private_GT',metrics=None,evaluation_wall_time_s=None,
            reason='Official HB/ITODD test GT is private; complete generated CSV/coverage retained; no remote submission')
        durable.atomic_json(receipt_path,entry); return entry
    if not state.get('ready'): raise RuntimeError('Public official evaluation assets are not ready: '+dataset)
    durable.require_disk(folder,15)
    command,env = command_environment(ctx,csv_path,folder)
    entry.update(status='running',command=command,started_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    durable.atomic_json(receipt_path,entry)
    def heartbeat(guard_pid):
        durable.atomic_json(folder/'own_evaluation_status.json',{'status':'official_own_evaluation','dataset':dataset,
            'pid':os.getpid(),'start_ticks':owned.start_ticks(os.getpid()),'guard_pid':guard_pid,'epoch':time.time()})
    start = time.perf_counter()
    try:
        code = owned.run_owned(command,env,folder,lock_stream,heartbeat)
    except Exception as exc:
        entry.update(status='failed_owned_evaluator',metrics=None,exception_type=type(exc).__name__,reason=str(exc))
        durable.atomic_json(receipt_path,entry)
        raise
    entry.update(evaluator_returncode=code,evaluation_wall_time_s=time.perf_counter()-start)
    if code != 0 or not score_path.is_file():
        entry.update(status='failed_original_evaluator',metrics=None,log=str(folder/'execution.log'))
        durable.atomic_json(receipt_path,entry); raise RuntimeError('Original own-submission evaluator failed: '+str(folder/'execution.log'))
    try:
        if ref.git_identity(ctx['source'],ref.PUBLIC_REVISION) != ctx['source_identity'] or ref.git_identity(ctx['toolkit'],ref.TOOLKIT_REVISION) != ctx['toolkit_identity']:
            raise ValueError('Original source changed during evaluation')
        if ref.sha256(csv_path) != pins['CSV_sha256'] or ref.sha256(ledger_path) != pins['target_ledger_sha256']:
            raise ValueError('Generated predictions/target ledger changed during evaluation')
        if audit_ledger(ctx['output'],dataset,csv_path,ledger_path,ctx['targets'],ctx['pins'],ctx['source']) != audit:
            raise ValueError('Saved inference artifacts changed during evaluation')
        scores = ref.read(score_path)
        entry.update(status='complete',metrics=pose_metrics(scores),score_file=str(score_path),score_sha256=ref.sha256(score_path),
            completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            CSV_mean_time_s=scores.get('bop19_average_time_per_image'),
            timing_note='Original generated CSV timing retained; evaluator wall time is not inference or onboarding runtime')
    except Exception as exc:
        entry.update(status='failed_evaluation_validation',metrics=None,exception_type=type(exc).__name__,reason=str(exc))
        durable.atomic_json(receipt_path,entry)
        raise
    durable.atomic_json(receipt_path,entry)
    return entry


def write_summary(root,output,durable=None):
    root,output = ref.local(root),ref.local(output)
    if durable is None: durable,_ = ref.pinned_guards(root)
    datasets = {}
    for dataset in DATASETS:
        path = output/'datasets'/dataset/'own_evaluation_receipt.json'
        if not path.is_file(): datasets[dataset]={'status':'incomplete','metrics':None}; continue
        receipt = ref.read(path)
        datasets[dataset] = {'status':receipt['status'],'metrics':receipt.get('metrics'),'coverage':receipt.get('coverage'),
            'evaluation_wall_time_s':receipt.get('evaluation_wall_time_s'),'receipt':str(path),'receipt_sha256':ref.sha256(path)}
    public_complete = all(datasets[d]['status']=='complete' for d in ref.PUBLIC_DATASETS)
    private_recorded = all(datasets[d]['status']=='metrics_unavailable_private_GT' for d in ref.PRIVATE_DATASETS)
    summary = {'schema':'generated-public-code-own-results-summary-v1','datasets':datasets,
        'status':'all_feasible_public_metrics_complete_private_GT_unavailable' if public_complete and private_recorded else 'incomplete',
        'prediction_origin':'our generated inference using pinned unmodified third-party public code; not authors CSVs',
        'seven_dataset_aggregate':None,'aggregate_reason':'HB/ITODD test GT is private; no compatible complete seven-dataset score',
        'published_performance_reproduced':False}
    durable.atomic_json(output/'own_results_summary.json',summary)
    lines = ['# Generated public-code inference results','',summary['prediction_origin']+'.','',
        'Suite status: **'+summary['status']+'**. Seven-dataset aggregate: unavailable because HB/ITODD test GT is private.','',
        '| Dataset | Status | Targets recorded/official | Failed | Blocked | Saved poses | Pose AR | VSD | MSSD | MSPD |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for dataset,row in datasets.items():
        coverage = row.get('coverage') or {}; counts = coverage.get('terminal_status_counts',{}); metrics = row.get('metrics') or {}
        scores = ['â€”' if name not in metrics else f'{metrics[name]:.6f}' for name in METRICS]
        lines.append('| '+' | '.join([dataset,row['status'],str(coverage.get('recorded_targets','â€”'))+'/'+str(coverage.get('official_object_image_targets','â€”')),
            str(counts.get('failed','â€”')),str(counts.get('blocked','â€”')),str(coverage.get('saved_poses','â€”')),*scores])+' |')
    lines += ['', 'All failed/blocked targets remain in the original official target denominator. Ground-truth poses are evaluation-only.',
        'Pose AR is the official BOP19 localization score; these are not segmentation AP results. Published-performance parity is not established.',
        'Evaluator wall time is separate from CAD onboarding and inference; unavailable stage timing is left unavailable.','']
    (output/'OWN_INFERENCE_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    return summary