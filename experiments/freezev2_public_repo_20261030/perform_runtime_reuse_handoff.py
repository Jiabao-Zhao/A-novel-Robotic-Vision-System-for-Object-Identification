"""One-time, recorded idle-boundary handoff to a validated runtime revision.

Staging/importing this helper does nothing. --execute is required. It never
clears locks or kills a GPU task: the old suite must first be idle behind an
exclusive real compute flock and have no children. Failures stop the handoff.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import runtime_reuse_driver as revision


CONTROLLER = r'''
import fcntl,hashlib,json,os,pathlib,signal,sys,time,uuid
request=json.loads(sys.argv[1]);folder=pathlib.Path(request['folder']);out=pathlib.Path(request['output'])
compute=pathlib.Path(request['compute_lock']);old=request['linux_process'];deadline=time.monotonic()+240
helper_sha=request['handoff_helper_sha256'];actions=[]
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(pathlib.Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
 path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
 with tmp.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def record(action,**details):
 actions.append({'action':action,'epoch':time.time(),**details});write(folder/'controller_actions.json',actions)
def identity(pid):
 p=pathlib.Path('/proc')/str(pid)
 try:
  line=(p/'stat').read_text();tail=line.rsplit(')',1)[1].split()
  return {'pid':pid,'start_ticks':tail[19],'state':tail[0],
          'actual_cmdline':(p/'cmdline').read_bytes().decode().rstrip('\0').split('\0')}
 except (FileNotFoundError,ProcessLookupError):return None
def check_idle():
 actual=identity(int(old['pid']))
 if actual is None or actual['start_ticks']!=str(old['start_ticks']) or actual['actual_cmdline']!=old['actual_cmdline'] or actual['state']=='Z':
  raise RuntimeError('Old Linux suite identity no longer matches; do not terminate another process')
 status=read(out/'status.json')
 if status.get('status')!='waiting_for_compute_lock' or int(status.get('pid',-1))!=old['pid'] or str(status.get('start_ticks'))!=str(old['start_ticks']):
  return False
 children=pathlib.Path('/proc')/str(old['pid'])/'task'/str(old['pid'])/'children'
 if children.read_text().split():return False
 if pathlib.Path(status.get('lock','')).resolve()!=compute.resolve():raise RuntimeError('Old suite is waiting for an unexpected lock')
 return True
def legacy_receipts():
 receipts={}
 for p in sorted((out/'datasets').glob('*/targets/*/receipt.json')):
  if not p.resolve().is_relative_to(out.resolve()):raise RuntimeError('Target receipt escaped own output')
  value=read(p)
  if value.get('status') not in ('complete','failed','blocked'):raise RuntimeError('Idle handoff found a pending target receipt')
  receipts[p.relative_to(out).as_posix()]=sha(p)
 return receipts
stream=None;suite_stream=None
try:
 stream=compute.open('a+b')
 while True:
  if time.monotonic()>=deadline:raise TimeoutError('No idle compute boundary within240 seconds; original run remains active')
  try:fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);break
  except BlockingIOError:time.sleep(.05)
 record('acquired_real_compute_flock')
 while not check_idle():
  if time.monotonic()>=deadline:raise TimeoutError('Old suite did not enter a verified idle wait')
  time.sleep(.1)
 legacy=legacy_receipts()
 ready={'status':'ready_for_native_supervisor_stop','linux_process':identity(int(old['pid'])),
        'legacy_terminal_count':len(legacy),'handoff_helper_sha256':helper_sha,'compute_flock_held':True,'actions':actions}
 write(folder/'ready.json',ready);record('verified_idle_suite_without_children',pid=old['pid'])
 ack=folder/'native_supervisor_stopped.json'
 while not ack.exists():
  if time.monotonic()>=deadline:raise TimeoutError('Native stop acknowledgment not received; no Linux termination performed')
  if not check_idle():raise RuntimeError('Old suite stopped being idle before native acknowledgment')
  time.sleep(.1)
 value=read(ack)
 if value.get('status')!='exact_native_supervisor_stopped' or value.get('handoff_helper_sha256')!=helper_sha:
  raise RuntimeError('Invalid native stop acknowledgment')
 if not check_idle():raise RuntimeError('Old suite is no longer idle after native supervisor stop')
 os.kill(int(old['pid']),signal.SIGTERM);record('terminated_exact_idle_linux_suite',pid=old['pid'],start_ticks=old['start_ticks'])
 while identity(int(old['pid'])) is not None:
  actual=identity(int(old['pid']))
  if actual and actual['start_ticks']!=str(old['start_ticks']):break
  if time.monotonic()>=deadline:raise TimeoutError('Terminated old suite has not disappeared; do not launch replacement')
  time.sleep(.1)
 suite_stream=(out/'suite.lock').open('a+b')
 fcntl.flock(suite_stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 record('verified_real_suite_lock_free_and_old_suite_gone')
 # Recheck the complete cutover boundary, after the old writer is gone.
 if legacy_receipts()!=legacy:raise RuntimeError('Legacy receipts changed during the supposedly idle handoff')
 transition=pathlib.Path(request['transition_path'])
 if transition.exists():raise FileExistsError('Transition manifest already exists; preserve it')
 payload={'schema':'runtime-reuse-transition-v1','status':'ready_for_handoff',
          'base_configuration_sha256':request['configuration_sha256'],
          'base_source_manifest_sha256':request['base_source_manifest_sha256'],
          'execution_revision_sha256':request['execution_revision_sha256'],
          'legacy_receipts':legacy,'old_linux_process':old,'old_windows_bridge':request['windows_bridge'],
          'old_wsl_bridge':request['wsl_bridge'],'handoff_helper_sha256':helper_sha,'actions':actions,
          'reason':'User-authorized validated execution speedup at an idle target boundary',
          'completed_epoch':time.time()}
 write(transition,payload)
 completed={'status':'old_owners_reconciled_transition_written','transition_path':str(transition),
            'transition_sha256':sha(transition),'legacy_terminal_count':len(legacy),
            'handoff_helper_sha256':helper_sha,'old_linux_process':old,'actions':actions}
 write(folder/'controller_completed.json',completed)
 print(json.dumps(completed),flush=True)
except BaseException as exc:
 failure={'status':'handoff_failed_closed','error':type(exc).__name__+': '+str(exc),
          'handoff_helper_sha256':helper_sha,'actions':actions,'epoch':time.time()}
 write(folder/'controller_failure.json',failure);print(json.dumps(failure),flush=True);raise
finally:
 if suite_stream is not None:suite_stream.close()
 if stream is not None:stream.close()
'''


def native_identity(record,psutil):
    p=psutil.Process(int(record['pid']))
    if abs(p.create_time()-float(record['create_time']))>=.01 or p.cmdline()!=record['actual_cmdline']:
        raise RuntimeError('Recorded native process identity differs; no process was terminated')
    return p


def same_identity_live(record,psutil):
    try:
        p=psutil.Process(int(record['pid']))
        return abs(p.create_time()-float(record['create_time']))<.01 and p.cmdline()==record['actual_cmdline']
    except (psutil.NoSuchProcess,psutil.ZombieProcess):return False


def wait_file(path,controller,deadline):
    while not path.exists():
        code=controller.poll()
        if code is not None:raise RuntimeError('WSL handoff controller exited before readiness: '+str(code))
        if time.monotonic()>=deadline:raise TimeoutError('WSL handoff readiness timed out')
        time.sleep(.1)
    return revision.read(path)


def execute(config_path,manifest_path,validation_path,folder):
    if os.name!='nt':raise RuntimeError('Use the recorded native Windows Python')
    if os.path.normcase(str(Path(sys.executable).resolve()))!=os.path.normcase(str(Path('D:/Python/python.exe').resolve())):
        raise RuntimeError('Use D:\\Python\\python.exe for native process reconciliation')
    config,out,saved=revision.configuration(config_path)
    revision.verify_revision(saved,manifest_path,require_sealed_self=False)
    revision.require_validation(saved,manifest_path,validation_path)
    if revision.sha256(Path(revision.__file__))!=revision.read(manifest_path)['files']['runtime_reuse_driver.py']:
        raise ValueError('Handoff validation module differs from the sealed driver')
    folder=revision.local(folder)
    if not folder.is_relative_to(out/'auxiliary_diagnostics'):raise ValueError('Handoff evidence must remain in own auxiliary diagnostics')
    folder.mkdir(parents=True,exist_ok=False)
    transition=out/'execution_revisions/transition_runtime_reuse_v1.json'
    if transition.exists():raise FileExistsError('A handoff transition already exists; do not overwrite it')
    helper_sha=revision.sha256(__file__)
    import psutil
    windows=revision.read(out/'windows_bridge.json');wsl=revision.read(out/'wsl_bridge.json');linux=revision.read(out/'linux_process.json')
    native_identity(windows,psutil);native_identity(wsl,psutil)
    if not linux.get('actual_cmdline') or not linux.get('start_ticks'):raise ValueError('Old Linux identity is incomplete')
    def mount(value):
        value=str(Path(value).resolve()).replace('\\','/')
        return '/mnt/'+value[0].lower()+value[2:] if len(value)>2 and value[1]==':' else value
    request={'folder':mount(folder),'output':config['paths']['output'],'compute_lock':config['paths']['compute_lock'],
             'transition_path':mount(transition),'linux_process':linux,'windows_bridge':windows,'wsl_bridge':wsl,
             'configuration_sha256':revision.digest(config),'base_source_manifest_sha256':revision.BASE_MANIFEST_SHA,
             'execution_revision_sha256':revision.sha256(manifest_path),'validation_receipt_sha256':revision.sha256(validation_path),
             'handoff_helper_sha256':helper_sha}
    revision.atomic_json(folder/'handoff_request.json',request)
    flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
    command=['wsl.exe','--exec',config['paths']['inference_python'],'-B','-',json.dumps(request,separators=(',',':'))]
    with (folder/'controller.log').open('ab',buffering=0) as log:
        controller=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,creationflags=flags)
        controller.stdin.write(CONTROLLER.encode());controller.stdin.close()
        deadline=time.monotonic()+245
        try:
            ready=wait_file(folder/'ready.json',controller,deadline)
            if ready.get('status')!='ready_for_native_supervisor_stop' or ready.get('handoff_helper_sha256')!=helper_sha or ready.get('compute_flock_held') is not True:
                raise RuntimeError('Unexpected handoff readiness evidence')
            # Revalidate both native records immediately before stopping only the
            # supervisor. The Linux controller has the real exclusive GPU lock.
            supervisor=native_identity(windows,psutil);bridge=native_identity(wsl,psutil)
            supervisor.terminate();supervisor.wait(timeout=10)
            if same_identity_live(windows,psutil):raise RuntimeError('Old native supervisor remains live')
            revision.atomic_json(folder/'native_supervisor_stopped.json',{'status':'exact_native_supervisor_stopped',
                'old_windows_bridge':windows,'old_wsl_bridge':wsl,'handoff_helper_sha256':helper_sha,'epoch':time.time()})
            code=controller.wait(timeout=max(1,deadline-time.monotonic()))
            if code:raise RuntimeError('WSL controller failed closed; inspect controller_failure.json')
            completed=revision.read(folder/'controller_completed.json')
            if completed.get('status')!='old_owners_reconciled_transition_written' or completed.get('handoff_helper_sha256')!=helper_sha:
                raise RuntimeError('Old-owner reconciliation receipt differs')
            # The original WSL bridge must exit; never terminate a potentially
            # surviving GPU/evaluator owner to force this check through.
            bridge.wait(timeout=15)
            if same_identity_live(wsl,psutil) or same_identity_live(windows,psutil):raise RuntimeError('An exact old native owner remains live')
            if revision.sha256(transition)!=completed['transition_sha256']:raise RuntimeError('Transition receipt changed')
            sealed_launcher=Path(manifest_path).resolve().parent/'launch_runtime_reuse.py'
            import importlib.util
            spec=importlib.util.spec_from_file_location('sealed_runtime_reuse_handoff_launcher',sealed_launcher)
            launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
            launcher.probe_locks(config,out)
            revision.verified_transition(saved,manifest_path,transition,require_sealed_self=False)
            revision.require_validation(saved,manifest_path,validation_path)
            command=[sys.executable,'-B',str(sealed_launcher),'--config',str(saved),'--transition-manifest',str(transition),
                     '--validation-receipt',str(Path(validation_path).resolve())]
            result=subprocess.run(command,capture_output=True,text=True,creationflags=flags)
            revision.atomic_json(folder/'launch_result.json',{'command':command,'returncode':result.returncode,'stdout':result.stdout,
                'stderr':result.stderr,'handoff_helper_sha256':helper_sha,'epoch':time.time()})
            if result.returncode:raise RuntimeError('Replacement launch failed closed: '+result.stderr)
            final={'status':'validated_runtime_reuse_launched','transition_manifest':str(transition),
                   'transition_sha256':revision.sha256(transition),'handoff_helper_sha256':helper_sha,
                   'legacy_terminal_count':completed['legacy_terminal_count'],'epoch':time.time()}
            revision.atomic_json(folder/'handoff_completed.json',final)
            return final
        except BaseException as exc:
            revision.atomic_json(folder/'native_failure.json',{'status':'handoff_failed_closed','error':type(exc).__name__+': '+str(exc),
                'handoff_helper_sha256':helper_sha,'controller_pid':controller.pid,'epoch':time.time()})
            # No force kill: the bounded controller releases its flock itself.
            raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path)
    p.add_argument('--revision-manifest',required=True,type=Path);p.add_argument('--validation-receipt',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--execute',action='store_true')
    args=p.parse_args(argv)
    if not args.execute:p.error('--execute is required for this one-time planned handoff')
    print(json.dumps(execute(args.config,args.revision_manifest,args.validation_receipt,args.output),indent=2))

if __name__=='__main__':main()