"""Run the pinned public repository's author-prediction reference gate unchanged.

This evaluates the authors' FreeZeV2.1/Accurate CSVs. It does not evaluate our own
inference and does not establish that the public base inference reproduces them.
"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

PUBLIC_REVISION = '12f82a2f8afd435466a87cf2dafce404087aeb3f'
TOOLKIT_REVISION = 'cea62d651c7e395b2e1962b9749e4e89693c6ac4'
FROZEN_MANIFEST_SHA256 = '237f10948fddc3a02fd42176e5acca3a9db1b7598e5b7a34d6d6766190e85c50'
PUBLIC_DATASETS = ('lmo','tless','tudl','icbin','ycbv')
PRIVATE_DATASETS = ('itodd','hb')
METRICS = {'ar':'bop19_average_recall','vsd':'bop19_average_recall_vsd',
           'mssd':'bop19_average_recall_mssd','mspd':'bop19_average_recall_mspd'}
TOLERANCE = 5e-4


def fs(path):
    path = Path(path)
    if os.name == 'nt' and path.is_absolute() and not str(path).startswith('\\\\?\\'):
        return Path('\\\\?\\'+str(path))
    return path


def local(value, *, resolve=True):
    value = str(value)
    if os.name != 'nt' and re.match(r'^[A-Za-z]:[\\/]',value):
        value = '/mnt/'+value[0].lower()+'/'+value[3:].replace('\\','/')
    elif os.name == 'nt' and re.match(r'^/mnt/[a-z]/',value):
        value = value[5].upper()+':/'+value[7:]
    path = Path(value).expanduser()
    return path.resolve() if resolve else path


def read(path): return json.loads(fs(path).read_text(encoding='utf-8-sig'))


def sha256(path):
    h = hashlib.sha256()
    with fs(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def git_read(path,*arguments,input_bytes=None,text=False):
    # A native Windows checkout uses autocrlf=true. Keep this interpretation
    # process-scoped so WSL sees the same tracked source without editing it or
    # changing any local/global Git configuration or index stat cache.
    env=os.environ.copy();env['GIT_OPTIONAL_LOCKS']='0'
    command=['git','-c','core.autocrlf=true','-C',str(path),*arguments]
    return subprocess.check_output(command,input=input_bytes,text=text,env=env)


def verify_blob_content(physical,canonical):
    if physical==canonical:return 'byte_identical'
    if (b'\0' not in canonical and b'\0' not in physical and b'\r\n' not in canonical
            and physical.replace(b'\r\n',b'\n')==canonical):
        return 'LF_git_blob_to_CRLF_checkout_only'
    raise ValueError('Working source differs from its pinned Git blob beyond checkout line endings')


def git_identity(path, expected_revision):
    path=Path(path)
    revision=git_read(path,'rev-parse','HEAD',text=True).strip()
    if revision!=expected_revision:raise ValueError('Unexpected source revision: '+str(path))
    staged=git_read(path,'diff','--cached','--name-only','HEAD',text=True)
    if staged.strip():raise ValueError('Tracked public/evaluator index differs from pinned HEAD: '+str(path))
    # Worktree status can report a racy false modification after CRLF conversion.
    # Complete byte-for-byte pinned Git blob checks below replace that weaker test.
    names=git_read(path,'ls-tree','-r','--name-only','HEAD',text=True).splitlines()
    requests=(''.join('HEAD:'+name+'\n' for name in names)).encode('utf-8')
    batch=git_read(path,'cat-file','--batch',input_bytes=requests)
    physical_hashes={};canonical_hashes={};line_endings=[];offset=0
    for name in names:
        target=(path/name).resolve()
        if not target.is_relative_to(path.resolve()) or not target.is_file():
            raise ValueError('Source file escaped checkout or is absent: '+name)
        end=batch.find(b'\n',offset)
        if end<0:raise ValueError('Incomplete pinned Git blob response')
        header=batch[offset:end].split()
        if len(header)!=3 or header[1]!=b'blob':raise ValueError('Tracked source is not a Git blob: '+name)
        size=int(header[2]);offset=end+1;canonical=batch[offset:offset+size];offset+=size
        if len(canonical)!=size or batch[offset:offset+1]!=b'\n':raise ValueError('Malformed pinned Git blob content')
        offset+=1;physical=fs(target).read_bytes()
        conversion=verify_blob_content(physical,canonical)
        if conversion!='byte_identical':line_endings.append(name)
        physical_hashes[name]=hashlib.sha256(physical).hexdigest()
        canonical_hashes[name]=hashlib.sha256(canonical).hexdigest()
    if offset!=len(batch):raise ValueError('Unexpected extra pinned Git blob response')
    return {'revision':revision,'tracked_files':physical_hashes,
            'tracked_tree_sha256':digest(physical_hashes),
            'canonical_git_blob_files':canonical_hashes,'canonical_git_tree_sha256':digest(canonical_hashes),
            'checkout_line_ending_only_files':line_endings,
            'line_ending_policy':'Exact Git blob bytes or verified LF-to-CRLF checkout conversion only; public Python source unchanged',
            'git_process_config':{'core.autocrlf':'true','GIT_OPTIONAL_LOCKS':'0'}}


def original_specs(source):
    """Read literal original metadata without modifying/importing public code."""
    tree = ast.parse(fs(Path(source)/'freezev2/bop.py').read_text(encoding='utf-8'))
    values = {}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ('REFERENCE_SUBMISSIONS','BOP_TOOLKIT_COMMIT'):
                    values[target.id] = ast.literal_eval(node.value)
    if values.get('BOP_TOOLKIT_COMMIT') != TOOLKIT_REVISION:
        raise ValueError('Original public evaluator pins a different toolkit')
    specs = values.get('REFERENCE_SUBMISSIONS')
    if not isinstance(specs,dict) or set(specs) != set(PUBLIC_DATASETS+PRIVATE_DATASETS):
        raise ValueError('Unexpected original reference dataset metadata')
    return specs


def metric_comparison(scores, expected, tolerance=TOLERANCE):
    if not math.isfinite(tolerance) or tolerance < 0: raise ValueError('Invalid metric tolerance')
    result = {}
    for short,key in METRICS.items():
        actual,wanted = float(scores[key]),float(expected[short])
        if not math.isfinite(actual) or not 0 <= actual <= 1 or not math.isfinite(wanted) or not 0 <= wanted <= 1:
            raise ValueError('Nonfinite or out-of-range reference recall')
        difference = actual-wanted
        result[key] = {'actual':actual,'expected':wanted,'difference':difference,
                       'absolute_tolerance':tolerance,'passed':abs(difference) <= tolerance+1e-15}
    return result


def target_counts(path):
    rows = read(path); seen = set(); instances = 0
    if not isinstance(rows,list) or not rows: raise ValueError('Missing official full target list')
    for row in rows:
        key = tuple(row.get(k) for k in ('scene_id','im_id','obj_id'))
        values = (*key,row.get('inst_count'))
        if any(isinstance(v,bool) or not isinstance(v,int) for v in values): raise ValueError('Invalid target integers')
        if key in seen or min(key)<0 or key[2]<1 or row['inst_count']<1: raise ValueError('Duplicate/invalid official target')
        seen.add(key);instances += row['inst_count']
    return {'object_image_targets':len(rows),'requested_instances':instances,
            'images':len({key[:2] for key in seen})}


def author_csv_info(path):
    rows = 0; images = set()
    with fs(path).open(newline='',encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['scene_id','im_id','obj_id','score','R','t','time']:
            raise ValueError('Unexpected untouched author CSV header')
        for row in reader:
            if None in row or any(v is None for v in row.values()): raise ValueError('Malformed author CSV row')
            key = int(row['scene_id']),int(row['im_id']); obj=int(row['obj_id'])
            rotation = [float(x) for x in row['R'].split()]; translation=[float(x) for x in row['t'].split()]
            numbers = rotation+translation+[float(row['score']),float(row['time'])]
            if min(key)<0 or obj<1 or len(rotation)!=9 or len(translation)!=3 or not all(math.isfinite(x) for x in numbers):
                raise ValueError('Invalid author CSV values')
            rows += 1;images.add(key)
    if not rows: raise ValueError('Author reference CSV is empty')
    return {'saved_author_predictions':rows,'CSV_images':len(images),'CSV_bytes':fs(path).stat().st_size,
            'prediction_origin':'untouched authors public FreeZeV2.1/Accurate submission',
            'timing_role':'Authors saved CSV timing is preserved; evaluator wall time is measured separately'}


def evaluation_python(runtime):
    # Resolving the executable follows a venv symlink to system Python and loses
    # its venv. Preserve the executable path; resolve its parent only for identity.
    return local(runtime['python'], resolve=False)


def command_and_environment(runtime, source, dataset, csv_path, folder):
    python = str(evaluation_python(runtime))
    toolkit,view = local(runtime['toolkit_root']),local(runtime['dataset_view'])
    if runtime.get('toolkit_revision') != TOOLKIT_REVISION or runtime.get('renderer_type') != 'vispy':
        raise ValueError('Unexpected recorded evaluator/toolkit configuration')
    if runtime.get('protocol',{}).get('visib_gt_min') != -1:
        raise ValueError('Reference gate requires original BOP19 localization semantics')
    command = [python,str(Path(source)/'run_bop.py'),'evaluate-reference','--dataset',dataset,
               '--bop-root',str(view),'--result-csv',str(csv_path),'--bop-toolkit',str(toolkit),
               '--eval-root',str(Path(folder)/'official_eval'),'--num-workers','1']
    env = os.environ.copy();env.update(runtime.get('environment',{}))
    env['BOP_PATH']=str(view)
    env['PYTHONPATH']=str(toolkit)+os.pathsep+env.get('PYTHONPATH','')
    env['PATH']=str(Path(python).parent)+os.pathsep+env.get('PATH','')
    env['PYTHONDONTWRITEBYTECODE']='1'
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): env[key]='4'
    return command,env


def pinned_guards(root):
    snapshot = Path(root)/'outputs/freezev2_bop_20261029/classic_core_base_v2/source_snapshot'
    manifest_path = snapshot/'source_manifest.json'
    if sha256(manifest_path) != FROZEN_MANIFEST_SHA256: raise ValueError('Frozen guard manifest changed')
    for name,wanted in read(manifest_path)['files'].items():
        path = (snapshot/name).resolve()
        if not path.is_relative_to(snapshot.resolve()) or sha256(path) != wanted:
            raise ValueError('Frozen process guard source changed: '+name)
    for name,module in list(sys.modules.items()):
        if name=='freezev2_repro' or name.startswith('freezev2_repro.'):
            path=getattr(module,'__file__',None)
            if path and not Path(path).resolve().is_relative_to(snapshot.resolve()):
                raise RuntimeError('Use a fresh process for verified frozen ownership helpers')
    sys.path.insert(0,str(snapshot))
    return importlib.import_module('freezev2_repro.durable'),importlib.import_module('freezev2_repro.owned_process')


def context(root, output):
    root,output = local(root),local(output)
    runtime_path = root/'outputs/freezev2_bop_20261029/evaluation_runtime.json'
    runtime=read(runtime_path)
    source=root/'outputs/freezev2_bop_20261029/external/FreezeV2-Re'
    original=git_identity(source,PUBLIC_REVISION)
    toolkit=git_identity(local(runtime['toolkit_root']),TOOLKIT_REVISION)
    specs=original_specs(source)
    supplied=output/'reference_specs.json'
    if supplied.exists() and read(supplied)!=specs: raise ValueError('Downloaded CSV specification differs from original source')
    return {'root':root,'output':output,'runtime':runtime,'runtime_path':runtime_path,'source':source,
            'source_identity':original,'toolkit_identity':toolkit,'specs':specs}


def dataset_pins(ctx,dataset):
    spec=ctx['specs'][dataset]
    csv_path=ctx['output']/'author_predictions'/spec['filename']
    targets=local(ctx['runtime']['dataset_view'])/dataset/'test_targets_bop19.json'
    counts=target_counts(targets);info=author_csv_info(csv_path)
    pins={'public_revision':PUBLIC_REVISION,'public_tracked_tree_sha256':ctx['source_identity']['tracked_tree_sha256'],
          'public_run_bop_sha256':sha256(ctx['source']/'run_bop.py'),
          'toolkit_revision':TOOLKIT_REVISION,'toolkit_tracked_tree_sha256':ctx['toolkit_identity']['tracked_tree_sha256'],
          'toolkit_eval_script_sha256':sha256(local(ctx['runtime']['toolkit_root'])/'scripts/eval_bop19_pose.py'),
          'CSV_sha256':sha256(csv_path),'official_targets_sha256':sha256(targets),
          'runtime_sha256':sha256(ctx['runtime_path']),'reference_gate_sha256':sha256(Path(__file__)),
          'frozen_guards_manifest_sha256':FROZEN_MANIFEST_SHA256,
          'expected_scores':spec['expected'],'absolute_tolerance':TOLERANCE}
    return csv_path,targets,pins,counts,info


def reusable(receipt,pins,score_path,expected):
    if receipt.get('status')!='complete' or receipt.get('source_pins')!=pins or not fs(score_path).is_file():return False
    if sha256(score_path)!=receipt.get('score_sha256'):return False
    scores=read(score_path)
    try: comparison=metric_comparison(scores,expected)
    except (KeyError,TypeError,ValueError):return False
    actual={key:float(scores[key]) for key in METRICS.values()}
    return all(row['passed'] for row in comparison.values()) and receipt.get('metrics')==actual


def require_runtime(runtime):
    if os.name!='posix' or Path(sys.prefix).resolve()!=evaluation_python(runtime).parent.parent.resolve():
        raise RuntimeError('Execute reference_gate.py using the recorded WSL evaluation Python')

def run_dataset(ctx,dataset,durable,owned,lock):
    folder=ctx['output']/'reference'/dataset;folder.mkdir(parents=True,exist_ok=True)
    receipt_path=folder/'metric_receipt.json'
    state=ctx['runtime']['datasets'].get(dataset,{})
    if dataset in PRIVATE_DATASETS or state.get('GT_withheld'):
        receipt={'dataset':dataset,'status':'skipped_private_GT','metrics':None,
                 'reason':'Official test GT is withheld; no local reference score or automatic remote submission',
                 'reference_prediction_configuration':'authors FreeZeV2.1/Accurate; not our generated base inference'}
        durable.atomic_json(receipt_path,receipt);return receipt
    if not state.get('ready'):raise RuntimeError('Existing public evaluation assets are not ready: '+dataset)
    csv_path,targets,pins,counts,info=dataset_pins(ctx,dataset)
    score_path=folder/'official_eval'/csv_path.stem/'scores_bop19.json'
    if receipt_path.exists():
        old=read(receipt_path)
        if old.get('source_pins') and old['source_pins']!=pins:
            raise ValueError('Existing reference run has incompatible source/CSV/runtime; use a new output version')
        if reusable(old,pins,score_path,ctx['specs'][dataset]['expected']):return old
    command,env=command_and_environment(ctx['runtime'],ctx['source'],dataset,csv_path,folder)
    require_runtime(ctx['runtime'])
    durable.require_disk(folder,15)
    entry={'schema':'unchanged-public-reference-score-gate-v1','dataset':dataset,'status':'running',
           'source_pins':pins,'command':command,'author_CSV':str(csv_path),'CSV_info':info,
           'official_targets':str(targets),'target_counts':counts,'reference_prediction_configuration':'authors FreeZeV2.1/Accurate',
           'claim_scope':'Evaluating authors untouched predictions; does not establish our own or public base inference performance',
           'original_public_source':str(ctx['source']),'evaluation_environment_python':str(local(ctx['runtime']['python'])),
           'GT_role':'official evaluation only','original_public_command_used':True}
    durable.atomic_json(receipt_path,entry)
    def heartbeat(guard_pid):
        durable.atomic_json(folder/'status.json',{'status':'official_reference_evaluation','pid':os.getpid(),
            'start_ticks':owned.start_ticks(os.getpid()),'guard_pid':guard_pid,'epoch':time.time(),'dataset':dataset})
    start=time.perf_counter()
    code=owned.run_owned(command,env,folder,lock,heartbeat)
    entry.update(evaluator_returncode=code,evaluation_wall_time_s=time.perf_counter()-start)
    if code!=0 or not fs(score_path).is_file():
        entry.update(status='failed_original_evaluator',log=str(folder/'execution.log'))
        durable.atomic_json(receipt_path,entry);raise RuntimeError('Original reference command failed: '+str(folder/'execution.log'))
    # Detect any changed source/input during the owned evaluator execution.
    if git_identity(ctx['source'],PUBLIC_REVISION)!=ctx['source_identity'] or git_identity(local(ctx['runtime']['toolkit_root']),TOOLKIT_REVISION)!=ctx['toolkit_identity']:
        raise ValueError('Public or toolkit source changed during evaluation')
    if dataset_pins(ctx,dataset)[2]!=pins:raise ValueError('Reference input/runtime changed during evaluation')
    scores=read(score_path);comparison=metric_comparison(scores,ctx['specs'][dataset]['expected'])
    passed=all(row['passed'] for row in comparison.values())
    entry.update(status='complete' if passed else 'reference_score_mismatch',metrics={key:float(scores[key]) for key in METRICS.values()},
        comparison=comparison,score_file=str(score_path),score_sha256=sha256(score_path),
        authors_CSV_mean_time_s=scores.get('bop19_average_time_per_image'),timing_tolerance_applied=False,
        timing_note='Saved authors prediction timing is preserved; local evaluator wall time is not inference runtime')
    durable.atomic_json(receipt_path,entry)
    if not passed:raise RuntimeError('Reference score differs from original published target; inspect '+str(receipt_path))
    return entry


def run(root,output,datasets):
    ctx=context(root,output);durable,owned=pinned_guards(ctx['root'])
    ctx['output'].mkdir(parents=True,exist_ok=True)
    results=[];failures=[]
    with durable.worker_lock(ctx['output']/'compute.lock') as lock:
        for dataset in datasets:
            try:results.append(run_dataset(ctx,dataset,durable,owned,lock))
            except Exception as exc:
                failures.append({'dataset':dataset,'type':type(exc).__name__,'reason':str(exc)})
                if len(datasets)==1:raise
        durable.atomic_json(ctx['output']/'reference_gate_summary.json',{
            'results':results,'failures':failures,'source_identity':ctx['source_identity'],
            'claim_scope':'Author-prediction score reproduction only; generated inference is a separate gate',
            'private_GT_datasets':{'itodd':'unavailable locally','hb':'unavailable locally'}})
    return {'results':results,'failures':failures}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--dataset',required=True,choices=(*PUBLIC_DATASETS,*PRIVATE_DATASETS,'all-public'))
    args=parser.parse_args();root=local(args.root)
    output=local(args.output or root/'outputs/freezev2_public_repo_20261030/base_public_v1')
    datasets=PUBLIC_DATASETS if args.dataset=='all-public' else (args.dataset,)
    result=run(root,output,datasets)
    print(json.dumps({'completed':[{ 'dataset':row['dataset'],'status':row['status'],'metrics':row.get('metrics')} for row in result['results']],
                      'failures':result['failures']},indent=2))
    return 1 if result['failures'] else 0

if __name__=='__main__':raise SystemExit(main())