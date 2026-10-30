"""Freeze, validate, run/resume and report the full seven-dataset experiment."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(Path(__file__).with_name('protocol.json')))
    parser.add_argument('--prepare',action='store_true',help='Seal source/config and save inventory; no GPU allocation')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--report',action='store_true')
    parser.add_argument('--validation-only',action='store_true')
    parser.add_argument('--probe-lock',action='store_true')
    args=parser.parse_args()
    config_path=Path(args.config)
    protocol=json.loads(config_path.read_text(encoding='utf-8-sig'))
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]=str(protocol['resources']['cpu_workers'])
    os.environ.setdefault('PYTHONUNBUFFERED','1')
    from freezev2_repro.pipeline import local_path
    from freezev2_repro.suite import SuiteRunner,summarize
    from freezev2_repro.durable import worker_lock,atomic_json,read_json
    from freezev2_repro.data import sha256
    from freezev2_repro.case_inference import InvalidCUDAContext
    root=local_path(protocol['paths']['suite_output_root'])
    if args.probe_lock:
        try:
            with ExitStack() as locks:
                locks.enter_context(worker_lock(root/'supervisor.lock'))
                locks.enter_context(worker_lock(root/'worker.lock'))
                print(json.dumps({'live_lock_owner':False}))
            return 0
        except RuntimeError:
            print(json.dumps({'live_lock_owner':True}));return 3
    try:
        if args.report:
            assets=read_json(local_path(protocol['paths']['assets_manifest']))
            states=read_json(root/'dataset_states.json') if (root/'dataset_states.json').exists() else {}
            result=summarize(root,assets,states,protocol)
        else:
            runner=SuiteRunner(protocol)
            if args.prepare:
                source=Path(__file__).parent
                destination=root/'source_snapshot'
                manifest=destination/'source_manifest.json'
                files=[source/'run_suite.py',source/'supervise_suite.py']+list((source/'freezev2_repro').glob('*.py'))
                expected={str(p.relative_to(source)):sha256(p) for p in files}
                if manifest.exists():
                    prior=read_json(manifest)
                    if prior['files']!=expected:raise ValueError('Prepared immutable source differs; select a new suite_output_root')
                    for relative,checksum in expected.items():
                        if sha256(destination/relative)!=checksum:raise ValueError('Immutable source snapshot was changed')
                else:
                    destination.mkdir(parents=True,exist_ok=True)
                    for file in files:
                        target=destination/file.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True)
                        shutil.copy2(file,target)
                    git=subprocess.run(['git','-C',str(source),'rev-parse','HEAD'],text=True,capture_output=True)
                    atomic_json(manifest,{'files':expected,'recipe_identity':runner.method_identity,'git_parent_revision':git.stdout.strip() if git.returncode==0 else None,'uncommitted_experiment_source_hashes_are_authoritative':True})
                atomic_json(root/'protocol.json',protocol)
                result=summarize(root,runner.assets,runner.states,protocol)
            else:
                # Normal execution uses the sealed snapshot, not mutable working
                # tree source. A resume verifies hashes before loading the worker.
                snapshot=root/'source_snapshot'
                if Path(__file__).resolve().parent!=snapshot.resolve():
                    raise ValueError('Run --prepare once, then execute the immutable source_snapshot/run_suite.py --resume')
                expected=read_json(snapshot/'source_manifest.json')['files']
                for relative,checksum in expected.items():
                    if sha256(snapshot/relative)!=checksum:raise ValueError('Immutable source snapshot hash mismatch')
                if Path(sys.executable).parent.parent.resolve()!=Path(protocol['paths']['inference_python']).parent.parent.resolve():
                    raise RuntimeError('Use the recorded isolated inference interpreter')
                result=runner.run(validation_only=args.validation_only)
        print(json.dumps({'suite_output_root':str(root),'suite_complete_and_evaluated':result['suite_complete_and_evaluated'],
                          'datasets':{name:{k:r[k] for k in ('status','completed_targets','failed_targets','blocked_targets','pending_targets')} for name,r in result['datasets'].items()}},indent=2))
        return 0
    except Exception as exc:
        root.mkdir(parents=True,exist_ok=True)
        atomic_json(root/'worker_exception.json',{'epoch':time.time(),'type':type(exc).__name__,'reason':str(exc),'traceback':traceback.format_exc()})
        print(json.dumps({'status':'blocked_or_failed','type':type(exc).__name__,'reason':str(exc)}),flush=True)
        return 75 if isinstance(exc,InvalidCUDAContext) else 2


if __name__=='__main__':
    raise SystemExit(main())
