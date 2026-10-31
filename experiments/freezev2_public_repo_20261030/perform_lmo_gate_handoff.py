"""One-time idle-boundary activation of the user's LM-O paper-score stop gate.

No inference settings or target receipts change. This helper never changes the
existing DINO execution revision or transition. --execute is required and root
must review the sealed control policy before invoking it.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import runtime_reuse_driver as revision
import perform_runtime_reuse_handoff as original

ORIGINAL_HANDOFF_SHA='7bf09a390369e973c5856c87fc5749f80889abcaae388043752485fb4d0c040c'
CONTROL_FILES=('runtime_gate_driver.py','launch_runtime_gate.py')


def controller_source():
    replacements={
        "'schema':'runtime-reuse-transition-v1'":"'schema':'runtime-gate-cutover-v1'",
        "'execution_revision_sha256':request['execution_revision_sha256'],":"'execution_revision_sha256':request['execution_revision_sha256'],\n          'control_manifest_sha256':request['control_manifest_sha256'],\n          'previous_transition_sha256':request['previous_transition_sha256'],\n          'original_handoff_helper_sha256':request['original_handoff_helper_sha256'],",
        "'reason':'User-authorized validated execution speedup at an idle target boundary'":"'reason':'User-authorized LM-O paper-score stopping policy at an idle target boundary'",
    }
    code=original.CONTROLLER
    for before,after in replacements.items():
        if code.count(before)!=1:raise ValueError('Original handoff controller structure changed')
        code=code.replace(before,after)
    compile(code,'lmo_gate_handoff_controller','exec')
    return code


def verify_control_files(config_path,control_manifest,policy_path,revision_manifest):
    config,out,saved=revision.configuration(config_path)
    path=revision.local(control_manifest)
    if path!=out/'auxiliary_source/runtime_reuse_v2_lmo_gate/control_manifest.json':raise ValueError('Unexpected sealed control location')
    m=revision.read(path)
    if m.get('schema')!='lmo-paper-run-control-source-v1' or m.get('configuration_sha256')!=revision.digest(config) or m.get('base_source_manifest_sha256')!=revision.BASE_MANIFEST_SHA or m.get('execution_revision_sha256')!=revision.sha256(revision_manifest):
        raise ValueError('Control source/configuration identity differs')
    if set(m.get('files',{}))!=set(CONTROL_FILES):raise ValueError('Unexpected control source file set')
    for name,wanted in m['files'].items():
        if revision.sha256(path.parent/name)!=wanted:raise ValueError('Sealed control source changed: '+name)
    policy=revision.local(policy_path)
    if not policy.is_relative_to(out) or revision.local(m.get('policy_path',''))!=policy or revision.sha256(policy)!=m.get('policy_sha256'):
        raise ValueError('Explicit user policy identity differs')
    return config,out,saved


def execute(config_path,control_manifest,policy_path,revision_manifest,transition_path,validation_path,folder):
    if os.name!='nt':raise RuntimeError('Use recorded native Windows Python')
    if os.path.normcase(str(Path(sys.executable).resolve()))!=os.path.normcase(str(Path('D:/Python/python.exe').resolve())):
        raise RuntimeError('Use D:\\Python\\python.exe')
    if revision.sha256(original.__file__)!=ORIGINAL_HANDOFF_SHA:raise ValueError('Original verified handoff helper changed')
    config,out,saved=verify_control_files(config_path,control_manifest,policy_path,revision_manifest)
    revision.verify_revision(saved,revision_manifest,require_sealed_self=False)
    revision.require_validation(saved,revision_manifest,validation_path)
    revision.verified_transition(saved,revision_manifest,transition_path,require_sealed_self=False)
    if revision.sha256(Path(revision.__file__))!=revision.read(revision_manifest)['files']['runtime_reuse_driver.py']:
        raise ValueError('Validation driver differs from the sealed execution revision')
    spec=importlib.util.spec_from_file_location('sealed_lmo_gate_handoff_verifier',Path(control_manifest).resolve().parent/'runtime_gate_driver.py')
    gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
    gate.verify_control(saved,control_manifest,policy_path,revision_manifest,require_sealed_self=False)
    folder=revision.local(folder)
    if not folder.is_relative_to(out/'auxiliary_diagnostics'):raise ValueError('Control handoff evidence escaped own diagnostics')
    folder.mkdir(parents=True,exist_ok=False)
    cutover=out/'run_control/lmo_gate_cutover_v1.json'
    if cutover.exists():raise FileExistsError('Control cutover already exists; preserve it')
    helper_sha=revision.sha256(__file__)
    import psutil
    windows=revision.read(out/'windows_bridge.json');wsl=revision.read(out/'wsl_bridge.json');linux=revision.read(out/'linux_process.json')
    original.native_identity(windows,psutil);original.native_identity(wsl,psutil)
    if not linux.get('actual_cmdline') or not linux.get('start_ticks'):raise ValueError('Old Linux identity is incomplete')
    def mount(value):
        value=str(Path(value).resolve()).replace('\\','/')
        return '/mnt/'+value[0].lower()+value[2:] if len(value)>2 and value[1]==':' else value
    request={'folder':mount(folder),'output':config['paths']['output'],'compute_lock':config['paths']['compute_lock'],
             'transition_path':mount(cutover),'linux_process':linux,'windows_bridge':windows,'wsl_bridge':wsl,
             'configuration_sha256':revision.digest(config),'base_source_manifest_sha256':revision.BASE_MANIFEST_SHA,
             'execution_revision_sha256':revision.sha256(revision_manifest),'control_manifest_sha256':revision.sha256(control_manifest),
             'previous_transition_sha256':revision.sha256(transition_path),'validation_receipt_sha256':revision.sha256(validation_path),
             'handoff_helper_sha256':helper_sha,'original_handoff_helper_sha256':ORIGINAL_HANDOFF_SHA}
    revision.atomic_json(folder/'handoff_request.json',request)
    flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
    command=['wsl.exe','--exec',config['paths']['inference_python'],'-B','-',json.dumps(request,separators=(',',':'))]
    with (folder/'controller.log').open('ab',buffering=0) as log:
        controller=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,creationflags=flags)
        controller.stdin.write(controller_source().encode());controller.stdin.close();deadline=time.monotonic()+245
        try:
            ready=original.wait_file(folder/'ready.json',controller,deadline)
            if ready.get('status')!='ready_for_native_supervisor_stop' or ready.get('handoff_helper_sha256')!=helper_sha or ready.get('compute_flock_held') is not True:
                raise RuntimeError('Unexpected handoff readiness evidence')
            supervisor=original.native_identity(windows,psutil);bridge=original.native_identity(wsl,psutil)
            supervisor.terminate();supervisor.wait(timeout=10)
            if original.same_identity_live(windows,psutil):raise RuntimeError('Exact old native supervisor is still live')
            revision.atomic_json(folder/'native_supervisor_stopped.json',{'status':'exact_native_supervisor_stopped',
                'old_windows_bridge':windows,'old_wsl_bridge':wsl,'handoff_helper_sha256':helper_sha,'epoch':time.time()})
            code=controller.wait(timeout=max(1,deadline-time.monotonic()))
            if code:raise RuntimeError('WSL controller failed closed; inspect controller_failure.json')
            completed=revision.read(folder/'controller_completed.json')
            if completed.get('status')!='old_owners_reconciled_transition_written' or completed.get('handoff_helper_sha256')!=helper_sha:
                raise RuntimeError('Owner reconciliation receipt differs')
            bridge.wait(timeout=15)
            if original.same_identity_live(wsl,psutil) or original.same_identity_live(windows,psutil):raise RuntimeError('An exact old native owner remains live')
            if revision.sha256(cutover)!=completed['transition_sha256']:raise RuntimeError('Control cutover receipt changed')
            # Recheck every immutable control/execution input after stopping the
            # old owner and before launching the new supervisor.
            verify_control_files(saved,control_manifest,policy_path,revision_manifest)
            gate.verify_control(saved,control_manifest,policy_path,revision_manifest,require_sealed_self=False)
            revision.verified_transition(saved,revision_manifest,transition_path,require_sealed_self=False)
            revision.require_validation(saved,revision_manifest,validation_path)
            launcher=Path(control_manifest).resolve().parent/'launch_runtime_gate.py'
            prior_launcher=Path(revision_manifest).resolve().parent/'launch_runtime_reuse.py'
            launcher_spec=importlib.util.spec_from_file_location('sealed_control_handoff_lock_probe',prior_launcher)
            launcher_module=importlib.util.module_from_spec(launcher_spec);launcher_spec.loader.exec_module(launcher_module)
            launcher_module.probe_locks(config,out)
            # The new launcher must retain its own exact native identity and
            # real suite/compute lock checks; no lock is cleared here.
            command=[sys.executable,'-B',str(launcher),'--config',str(saved),'--control-manifest',str(Path(control_manifest).resolve()),
                     '--policy',str(Path(policy_path).resolve()),'--revision-manifest',str(Path(revision_manifest).resolve()),
                     '--transition-manifest',str(Path(transition_path).resolve()),'--validation-receipt',str(Path(validation_path).resolve())]
            result=subprocess.run(command,capture_output=True,text=True,creationflags=flags)
            revision.atomic_json(folder/'launch_result.json',{'command':command,'returncode':result.returncode,'stdout':result.stdout,
                'stderr':result.stderr,'handoff_helper_sha256':helper_sha,'epoch':time.time()})
            if result.returncode:raise RuntimeError('Control gate launch failed closed: '+result.stderr)
            final={'status':'lmo_stop_gate_launched','control_manifest':str(control_manifest),'control_manifest_sha256':revision.sha256(control_manifest),
                   'control_cutover':str(cutover),'control_cutover_sha256':revision.sha256(cutover),
                   'previous_execution_transition_sha256':revision.sha256(transition_path),
                   'handoff_helper_sha256':helper_sha,'legacy_terminal_count':completed['legacy_terminal_count'],'epoch':time.time()}
            revision.atomic_json(folder/'handoff_completed.json',final);return final
        except BaseException as exc:
            revision.atomic_json(folder/'native_failure.json',{'status':'control_handoff_failed_closed','error':type(exc).__name__+': '+str(exc),
                'handoff_helper_sha256':helper_sha,'controller_pid':controller.pid,'epoch':time.time()})
            raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path)
    p.add_argument('--control-manifest',required=True,type=Path);p.add_argument('--policy',required=True,type=Path)
    p.add_argument('--revision-manifest',required=True,type=Path);p.add_argument('--transition-manifest',required=True,type=Path)
    p.add_argument('--validation-receipt',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--execute',action='store_true');args=p.parse_args(argv)
    if not args.execute:p.error('--execute is required for the planned control-only handoff')
    print(json.dumps(execute(args.config,args.control_manifest,args.policy,args.revision_manifest,args.transition_manifest,args.validation_receipt,args.output),indent=2))

if __name__=='__main__':main()