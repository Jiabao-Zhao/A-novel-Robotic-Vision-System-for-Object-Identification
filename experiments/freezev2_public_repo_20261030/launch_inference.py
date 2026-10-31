"""Seal isolated orchestration and start one hidden, resumable Windows/WSL suite.

No inference starts with --prepare-only. Detached execution is started only when
this launcher is explicitly invoked without that flag.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import reference_gate as ref

FILES=('suite_runner.py','launch_inference.py','reference_gate.py','onboard_original.py','evaluate_own.py')


def native(value):
    value=str(value)
    if value.startswith('/mnt/') and len(value)>7 and value[6]=='/':value=value[5].upper()+':/'+value[7:]
    return Path(value).resolve()


def mount(value):
    value=str(Path(value).resolve()).replace('\\','/')
    return '/mnt/'+value[0].lower()+value[2:] if len(value)>2 and value[1]==':' else value


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def seal(config_path,source_dir=None):
    config_path=Path(config_path).resolve();config=ref.read(config_path)
    from suite_runner import validate_configuration
    validate_configuration(config)
    source=Path(source_dir or Path(__file__).parent).resolve();output=native(config['paths']['output'])
    root=source.parents[1]
    if not output.is_relative_to(root/'outputs/freezev2_public_repo_20261030'):
        raise ValueError('Own inference output must remain in the isolated public study')
    output.mkdir(parents=True,exist_ok=True);snapshot=output/'source_snapshot'
    hashes={name:ref.sha256(source/name) for name in FILES}
    manifest={'schema':'sealed-public-orchestration-source-v1','configuration_sha256':ref.digest(config),'files':hashes}
    manifest_path=snapshot/'source_manifest.json'
    if manifest_path.exists():
        if ref.read(manifest_path)!=manifest:raise ValueError('Sealed source/configuration differs; use a new versioned output')
        for name,sha in hashes.items():
            if ref.sha256(snapshot/name)!=sha:raise ValueError('Sealed source payload changed')
    else:
        if snapshot.exists() and any(snapshot.iterdir()):raise ValueError('Unidentified source snapshot exists')
        snapshot.mkdir(exist_ok=True)
        for name in FILES:shutil.copyfile(source/name,snapshot/name)
        atomic_json(manifest_path,manifest)
    saved=output/'configuration.json'
    if saved.exists() and ref.read(saved)!=config:raise ValueError('Output configuration differs')
    if not saved.exists():atomic_json(saved,config)
    return config,output,snapshot,saved


def exact_live_bridge(record,psutil):
    try:
        process=psutil.Process(int(record['pid']))
        return abs(process.create_time()-float(record['create_time']))<.01 and process.cmdline()==record['actual_cmdline']
    except (psutil.NoSuchProcess,psutil.ZombieProcess):return False
    except (KeyError,ValueError,TypeError):raise ValueError('Malformed prior native bridge identity')
    # AccessDenied is not proof of death and deliberately propagates.


def probe_suite(config,output,run=subprocess.run):
    code=('import fcntl,sys; p=open(sys.argv[1],"a+b"); '
          'fcntl.flock(p.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)')
    command=['wsl.exe','--exec',config['paths']['inference_python'],'-B','-c',code,mount(output/'suite.lock')]
    result=run(command,capture_output=True,text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if result.returncode:raise RuntimeError('A surviving Linux suite/guard owns the OS lock, or lock probe failed: '+result.stdout+result.stderr)


@contextmanager
def launcher_lock(output):
    import msvcrt
    stream=(Path(output)/'launcher.lock').open('a+b')
    stream.seek(0);stream.write(b'0');stream.flush();stream.seek(0)
    try:msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:stream.close();raise RuntimeError('Another native launcher owns its OS lock')
    try:yield
    finally:stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1);stream.close()


def launch(config_path,*,prepare_only=False):
    if os.name!='nt':raise RuntimeError('This launcher requires native Windows Python')
    config,output,snapshot,saved=seal(config_path)
    if prepare_only:return {'status':'source_sealed_no_inference_launched','source_manifest':str(snapshot/'source_manifest.json'),'configuration':str(saved)}
    import psutil
    with launcher_lock(output):
        bridge=output/'windows_bridge.json'
        if bridge.exists() and exact_live_bridge(ref.read(bridge),psutil):raise RuntimeError('The exact native bridge is live; no duplicate launched')
        probe_suite(config,output)
        command=[sys.executable,'-B',str(snapshot/'launch_inference.py'),'--supervise','--config',str(saved)]
        flags=subprocess.CREATE_NO_WINDOW|subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP
        with (output/'bridge.log').open('ab',buffering=0) as log:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,creationflags=flags)
        handle=psutil.Process(child.pid)
        record={'pid':child.pid,'create_time':handle.create_time(),'actual_cmdline':handle.cmdline(),'command':command,
                'launched_epoch':time.time(),'source_manifest_sha256':ref.sha256(snapshot/'source_manifest.json')}
        atomic_json(bridge,record);return {'status':'launched','bridge':record,'output':str(output)}


def supervise(config_path):
    if os.name!='nt':raise RuntimeError('Supervisor requires native Windows Python')
    config=ref.read(config_path);output=native(config['paths']['output']);snapshot=output/'source_snapshot'
    # Source is already sealed. Never recopy mutable experiment files at resume.
    manifest=ref.read(snapshot/'source_manifest.json')
    if manifest['configuration_sha256']!=ref.digest(config):raise ValueError('Supervisor configuration differs')
    for name,sha in manifest['files'].items():
        if ref.sha256(snapshot/name)!=sha:raise ValueError('Supervisor sealed source changed')
    import psutil
    import ctypes
    kernel=ctypes.windll.kernel32
    kernel.SetThreadExecutionState(0x80000000|0x00000001)
    command=['wsl.exe','--exec','env','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4',
             config['paths']['inference_python'],'-B',mount(snapshot/'suite_runner.py'),'--config',mount(config_path)]
    try:
        for attempt in range(int(config['resources'].get('auto_restart_limit',2))+1):
            # A dead parent may leave an owned guard briefly alive. Its lock
            # remains authoritative until cleanup; never remove a lock file.
            while True:
                try:probe_suite(config,output);break
                except RuntimeError:
                    atomic_json(output/'supervisor_status.json',{'status':'waiting_for_surviving_linux_owner','attempt':attempt,'epoch':time.time()})
                    time.sleep(15)
            with (output/'worker.log').open('ab',buffering=0) as log:
                child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                handle=psutil.Process(child.pid)
                atomic_json(output/'wsl_bridge.json',{'pid':child.pid,'create_time':handle.create_time(),'actual_cmdline':handle.cmdline(),'command':command})
                code=child.wait()
            status=ref.read(output/'status.json') if (output/'status.json').exists() else {}
            if code==0 and status.get('status')=='suite_pass_finished':
                atomic_json(output/'supervisor_status.json',{'status':'suite_pass_finished','exit_code':code,'attempt':attempt,'epoch':time.time()});return 0
            atomic_json(output/'supervisor_status.json',{'status':'retrying' if attempt<int(config['resources'].get('auto_restart_limit',2)) else 'failed',
                        'exit_code':code,'attempt':attempt,'epoch':time.time()})
            if attempt<int(config['resources'].get('auto_restart_limit',2)):time.sleep(15)
        return 1
    finally:kernel.SetThreadExecutionState(0x80000000)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('public_inference_v1.json'))
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--supervise',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.supervise:return supervise(args.config)
    print(json.dumps(launch(args.config,prepare_only=args.prepare_only),indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
