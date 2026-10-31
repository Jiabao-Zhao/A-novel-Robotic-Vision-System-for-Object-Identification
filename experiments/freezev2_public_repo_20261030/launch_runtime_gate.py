"""Native supervisor for the separately sealed LM-O run-control boundary."""
from __future__ import annotations
import argparse
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import runtime_gate_driver as gate


def mount(path):
    value=str(Path(path).resolve()).replace('\\','/')
    return '/mnt/'+value[0].lower()+value[2:] if len(value)>2 and value[1]==':' else value


def load_launcher(out):
    prior,manifest=gate.load_prior(out)
    return gate.load_module(manifest.parent/'launch_runtime_reuse.py','sealed_lmo_gate_native_guards'),prior


def load_original_launcher(out,prior):
    # The sealed control folder intentionally has no reference_gate.py sibling.
    # Verify/load the original five-file snapshot first, which installs its
    # import directory exactly as the validated v1 launcher does.
    prior.load_base(out)
    return gate.load_module(Path(out)/'source_snapshot/launch_inference.py','sealed_lmo_gate_original_lock')


def launch(config_path,policy_path,*,control_manifest=None,revision_manifest=None,transition_manifest=None,validation_receipt=None,prepare_only=False):
    if os.name!='nt':raise RuntimeError('Use native Windows Python')
    control=gate.seal_control(config_path,policy_path)
    if control_manifest and gate.local(control_manifest)!=control:raise ValueError('Unexpected control manifest path')
    config,out,saved=gate.configuration(config_path)
    if prepare_only:return {'status':'control_source_sealed_no_inference_launched','control_manifest':str(control),'control_manifest_sha256':gate.sha256(control)}
    config,out,manifest,prior=gate.verify_control(saved,control,policy_path,revision_manifest,require_sealed_self=False)
    execution=gate.local(revision_manifest);transition=gate.local(transition_manifest);validation=gate.local(validation_receipt)
    prior.require_validation(saved,execution,validation)
    prior.verified_transition(saved,execution,transition,require_sealed_self=False)
    launcher,_=load_launcher(out)
    original=load_original_launcher(out,prior)
    import psutil
    with original.launcher_lock(out):
        for name in ('windows_bridge.json','wsl_bridge.json'):
            p=out/name
            if p.exists() and launcher.exact_live(gate.read(p),psutil):raise RuntimeError('The exact prior native owner is live; no duplicate launched')
        launcher.probe_locks(config,out)
        command=[sys.executable,'-B',str(control.parent/'launch_runtime_gate.py'),'--supervise','--config',str(saved),
                 '--control-manifest',str(control),'--policy',str(gate.local(policy_path)),
                 '--revision-manifest',str(execution),'--transition-manifest',str(transition),'--validation-receipt',str(validation)]
        flags=subprocess.CREATE_NO_WINDOW|subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP
        with (out/'runtime_gate_bridge.log').open('ab',buffering=0) as log:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,creationflags=flags)
        actual=psutil.Process(child.pid)
        record={'pid':actual.pid,'create_time':actual.create_time(),'actual_cmdline':actual.cmdline(),'command':command,
                'launched_epoch':time.time(),'source_manifest_sha256':gate.BASE_MANIFEST_SHA,
                'execution_revision_sha256':gate.EXECUTION_REVISION_SHA,'control_manifest_sha256':gate.sha256(control),
                'policy_sha256':manifest['policy_sha256']}
        prior.atomic_json(out/'windows_bridge.json',record)
    return {'status':'launched_lmo_comparison_gate','bridge':record,'output':str(out)}


def supervise(config_path,control_manifest,policy_path,revision_manifest,transition_manifest,validation_receipt):
    if os.name!='nt':raise RuntimeError('Supervisor requires native Windows Python')
    config,out,manifest,prior=gate.verify_control(config_path,control_manifest,policy_path,revision_manifest,require_sealed_self=False)
    if Path(__file__).resolve()!=Path(control_manifest).resolve().parent/'launch_runtime_gate.py':raise ValueError('Execute the sealed control supervisor')
    prior.require_validation(config_path,revision_manifest,validation_receipt)
    launcher,_=load_launcher(out)
    import psutil
    kernel=ctypes.windll.kernel32;kernel.SetThreadExecutionState(0x80000000|0x00000001)
    command=['wsl.exe','--exec','env','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4',
             config['paths']['inference_python'],'-B',mount(Path(control_manifest).parent/'runtime_gate_driver.py'),
             '--config',mount(config_path),'--control-manifest',mount(control_manifest),'--policy',mount(policy_path),
             '--revision-manifest',mount(revision_manifest),'--transition-manifest',mount(transition_manifest),
             '--validation-receipt',mount(validation_receipt)]
    try:
        for attempt in range(int(config['resources'].get('auto_restart_limit',2))+1):
            while True:
                try:launcher.probe_locks(config,out);break
                except RuntimeError:
                    prior.atomic_json(out/'supervisor_status.json',{'status':'waiting_for_surviving_linux_owner','attempt':attempt,'epoch':time.time(),'control_manifest_sha256':gate.sha256(control_manifest)})
                    time.sleep(15)
            with (out/'runtime_gate_worker.log').open('ab',buffering=0) as log:
                child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                actual=psutil.Process(child.pid)
                prior.atomic_json(out/'wsl_bridge.json',{'pid':actual.pid,'create_time':actual.create_time(),'actual_cmdline':actual.cmdline(),'command':command,
                    'execution_revision_sha256':gate.EXECUTION_REVISION_SHA,'control_manifest_sha256':gate.sha256(control_manifest),'policy_sha256':manifest['policy_sha256']})
                code=child.wait()
            status=gate.read(out/'status.json') if (out/'status.json').exists() else {}
            if code==0 and status.get('status') in (*gate.EXPECTED_STOPS,'suite_pass_finished'):
                prior.atomic_json(out/'supervisor_status.json',{'status':status['status'],'exit_code':code,'attempt':attempt,'epoch':time.time(),
                    'control_manifest_sha256':gate.sha256(control_manifest),'expected_run_control_stop':status['status'] in gate.EXPECTED_STOPS})
                return 0
            prior.atomic_json(out/'supervisor_status.json',{'status':'retrying' if attempt<int(config['resources'].get('auto_restart_limit',2)) else 'failed',
                'exit_code':code,'attempt':attempt,'epoch':time.time(),'control_manifest_sha256':gate.sha256(control_manifest)})
            if attempt<int(config['resources'].get('auto_restart_limit',2)):time.sleep(15)
        return 1
    finally:kernel.SetThreadExecutionState(0x80000000)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path);p.add_argument('--policy',required=True,type=Path)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--control-manifest',type=Path)
    p.add_argument('--revision-manifest',type=Path);p.add_argument('--transition-manifest',type=Path);p.add_argument('--validation-receipt',type=Path)
    p.add_argument('--supervise',action='store_true');a=p.parse_args(argv)
    if not a.prepare_only and not all((a.revision_manifest,a.transition_manifest,a.validation_receipt)):p.error('Existing validated DINO runtime source/transition/validation are required')
    if a.supervise:
        if not a.control_manifest:p.error('The sealed control manifest is required')
        return supervise(a.config,a.control_manifest,a.policy,a.revision_manifest,a.transition_manifest,a.validation_receipt)
    print(json.dumps(launch(a.config,a.policy,control_manifest=a.control_manifest,revision_manifest=a.revision_manifest,
        transition_manifest=a.transition_manifest,validation_receipt=a.validation_receipt,prepare_only=a.prepare_only),indent=2));return 0

if __name__=='__main__':raise SystemExit(main())