"""Stage or launch the separately sealed, validated DINO reuse execution revision."""
from __future__ import annotations
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import runtime_reuse_driver as revision


def mount(path):
    value=str(Path(path).resolve()).replace('\\','/')
    return '/mnt/'+value[0].lower()+value[2:] if len(value)>2 and value[1]==':' else value


def exact_live(record,psutil):
    try:
        p=psutil.Process(int(record['pid']))
        return abs(p.create_time()-float(record['create_time']))<.01 and p.cmdline()==record['actual_cmdline']
    except (psutil.NoSuchProcess,psutil.ZombieProcess):return False


def probe_locks(config,out):
    code=('import fcntl,sys; handles=[]; '
          '\nfor path in sys.argv[1:]:\n f=open(path,"a+b"); '
          'fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB); handles.append(f)')
    command=['wsl.exe','--exec',config['paths']['inference_python'],'-B','-c',code,
             mount(out/'suite.lock'),config['paths']['compute_lock']]
    result=subprocess.run(command,capture_output=True,text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if result.returncode:raise RuntimeError('A surviving suite/GPU/evaluator owns a real OS lock, or probing failed: '+result.stdout+result.stderr)


def seal_and_check(config_path,*,prepare_only=False):
    manifest=revision.seal_revision(config_path)
    config,out,saved=revision.configuration(config_path)
    return manifest,config,out,saved


def launch(config_path,transition_path,validation_path,*,prepare_only=False):
    if os.name!='nt':raise RuntimeError('Use native Windows Python for this launcher')
    manifest,config,out,saved=seal_and_check(config_path)
    if prepare_only:return {'status':'execution_revision_sealed_no_inference_launched','execution_manifest':str(manifest),'execution_revision_sha256':revision.sha256(manifest)}
    revision.require_validation(saved,manifest,validation_path)
    revision.verified_transition(saved,manifest,transition_path,require_sealed_self=False)
    # Verification must run through the sealed driver, not mutable study code.
    sealed_driver=manifest.parent/'runtime_reuse_driver.py'
    base=revision.load_base(out)
    original_launcher_path=out/'source_snapshot/launch_inference.py'
    import importlib.util
    spec=importlib.util.spec_from_file_location('original_reuse_launcher_guards',original_launcher_path)
    original_launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(original_launcher)
    import psutil
    with original_launcher.launcher_lock(out):
        for name in ('windows_bridge.json','wsl_bridge.json'):
            p=out/name
            if p.exists() and exact_live(revision.read(p),psutil):raise RuntimeError('The exact prior native process is live; no duplicate launched')
        probe_locks(config,out)
        command=[sys.executable,'-B',str(manifest.parent/'launch_runtime_reuse.py'),'--supervise','--config',str(saved),
                 '--revision-manifest',str(manifest),'--transition-manifest',str(revision.local(transition_path)),
                 '--validation-receipt',str(revision.local(validation_path))]
        flags=subprocess.CREATE_NO_WINDOW|subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP
        with (out/'runtime_reuse_bridge.log').open('ab',buffering=0) as log:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,creationflags=flags)
        handle=psutil.Process(child.pid)
        record={'pid':child.pid,'create_time':handle.create_time(),'actual_cmdline':handle.cmdline(),'command':command,
                'launched_epoch':time.time(),'source_manifest_sha256':revision.BASE_MANIFEST_SHA,
                'execution_revision_sha256':revision.sha256(manifest),'transition_manifest_sha256':revision.sha256(transition_path)}
        revision.atomic_json(out/'windows_bridge.json',record)
    return {'status':'launched_validated_runtime_reuse','bridge':record,'output':str(out)}


def supervise(config_path,manifest,transition_path,validation_path):
    if os.name!='nt':raise RuntimeError('Supervisor requires native Windows Python')
    config,out,_=revision.verify_revision(config_path,manifest,require_sealed_self=False)
    revision.require_validation(config_path,manifest,validation_path)
    if Path(__file__).resolve()!=Path(manifest).resolve().parent/'launch_runtime_reuse.py':raise ValueError('Execute the sealed native launcher')
    import psutil
    kernel=ctypes.windll.kernel32;kernel.SetThreadExecutionState(0x80000000|0x00000001)
    command=['wsl.exe','--exec','env','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4',
             config['paths']['inference_python'],'-B',mount(Path(manifest).parent/'runtime_reuse_driver.py'),
             '--config',mount(config_path),'--revision-manifest',mount(manifest),
             '--transition-manifest',mount(transition_path),'--validation-receipt',mount(validation_path)]
    try:
        for attempt in range(int(config['resources'].get('auto_restart_limit',2))+1):
            while True:
                try:probe_locks(config,out);break
                except RuntimeError:
                    revision.atomic_json(out/'supervisor_status.json',{'status':'waiting_for_surviving_linux_owner','attempt':attempt,'epoch':time.time(),'execution_revision_sha256':revision.sha256(manifest)})
                    time.sleep(15)
            with (out/'runtime_reuse_worker.log').open('ab',buffering=0) as log:
                child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                p=psutil.Process(child.pid)
                revision.atomic_json(out/'wsl_bridge.json',{'pid':p.pid,'create_time':p.create_time(),'actual_cmdline':p.cmdline(),'command':command,
                                                          'execution_revision_sha256':revision.sha256(manifest)})
                code=child.wait()
            status=revision.read(out/'status.json') if (out/'status.json').exists() else {}
            if code==0 and status.get('status')=='suite_pass_finished':
                revision.atomic_json(out/'supervisor_status.json',{'status':'suite_pass_finished','exit_code':code,'attempt':attempt,'epoch':time.time(),'execution_revision_sha256':revision.sha256(manifest)})
                return 0
            revision.atomic_json(out/'supervisor_status.json',{'status':'retrying' if attempt<int(config['resources'].get('auto_restart_limit',2)) else 'failed',
                                                             'exit_code':code,'attempt':attempt,'epoch':time.time(),'execution_revision_sha256':revision.sha256(manifest)})
            if attempt<int(config['resources'].get('auto_restart_limit',2)):time.sleep(15)
        return 1
    finally:kernel.SetThreadExecutionState(0x80000000)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--revision-manifest',type=Path);p.add_argument('--transition-manifest',type=Path)
    p.add_argument('--validation-receipt',type=Path);p.add_argument('--supervise',action='store_true')
    args=p.parse_args(argv)
    if not args.prepare_only and (not args.transition_manifest or not args.validation_receipt):p.error('Transition and passed validation are required')
    if args.supervise:
        if not args.revision_manifest:p.error('Supervisor requires the sealed execution manifest')
        return supervise(args.config,args.revision_manifest,args.transition_manifest,args.validation_receipt)
    print(json.dumps(launch(args.config,args.transition_manifest,args.validation_receipt,prepare_only=args.prepare_only),indent=2));return 0

if __name__=='__main__':raise SystemExit(main())