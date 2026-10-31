"""Isolated, lock-exclusive runtime trial; never edits inference outputs."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import datetime


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--helper', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    config = read(args.config)
    output = Path(config['paths']['output'])
    if not args.output.resolve().is_relative_to((output/'auxiliary_diagnostics').resolve()):
        raise ValueError('Trial must remain inside auxiliary diagnostics')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = read(output/'run_manifest.json')
    cases = [
        ('healthy_original', 's000002_i000003_o000001', False),
        ('healthy_reuse', 's000002_i000003_o000001', True),
        ('alternate_gedi_failure_reuse', 's000002_i000039_o000011', True),
        ('all_gedi_failure_reuse', 's000002_i000079_o000005', True),
        ('no_mask_reuse', 's000002_i000017_o000006', True),
    ]
    selection = {'schema':'runtime-reuse-fixed-controls-v1',
                 'selection':'First own target, first observed alternate GeDi failure, first all-GeDi-invalid target, first no-mask target; selected before trial without GT or accuracy scores.',
                 'configuration_sha256':sha(args.config), 'helper_sha256':sha(args.helper),
                 'controller_sha256':sha(__file__), 'cases':[]}
    for label,key,reuse in cases:
        receipt=output/'datasets/lmo/targets'/key/'receipt.json'
        value=read(receipt)
        if value['status']!='complete' or value['source_pins']!=manifest['source_pins']:
            raise ValueError('Control identity differs')
        selection['cases'].append({'label':label,'target':key,'reuse':reuse,
                                   'receipt':str(receipt),'receipt_sha256':sha(receipt)})
    write_new(args.output/'selection.json',selection)
    print('WAITING_FOR_COMPUTE_BOUNDARY',flush=True)
    with Path(config['paths']['compute_lock']).open('a+b') as lock:
        started=time.monotonic()
        while True:
            try:
                fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic()-started>240:
                    raise TimeoutError('No compute boundary acquired; current worker is preserved')
                time.sleep(.03)
        print('COMPUTE_LOCK_ACQUIRED',flush=True)
        write_new(args.output/'lock_receipt.json',{'pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'lock':config['paths']['compute_lock'],'wait_s':time.monotonic()-started})
        finished=[]
        for case in selection['cases']:
            folder=args.output/case['label'];folder.mkdir()
            baseline=read(case['receipt']);command=list(baseline['command'])
            command[command.index('--work-dir')+1]=str(folder/'original')
            command[command.index('--output')+1]=str(folder/'original.csv')
            request={'configuration':str(args.config),'source_pins':manifest['source_pins'],
                     'baseline_receipt':case['receipt'],'command':command,
                     'public_revision':config['public_revision'],'source':config['paths']['public_source']}
            write_new(folder/'request.json',request)
            trial=[sys.executable,'-B',str(args.helper),'--request',str(folder/'request.json'),
                   '--report',str(folder/'runtime_report.json'),'--lock-fd',str(lock.fileno())]
            if case['reuse']:trial.append('--reuse-dino')
            env=os.environ.copy()
            env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
                       MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='0',
                       NVIDIA_TF32_OVERRIDE='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                       TORCH_HOME=str(Path(config['paths']['assets_output'])/'runtime/torch'))
            before=time.monotonic();child=None
            print('TRIAL_STARTED',case['label'],flush=True)
            with (folder/'execution.log').open('x') as log:
                try:
                    child=subprocess.Popen(trial,env=env,stdout=log,stderr=subprocess.STDOUT,
                                           pass_fds=(lock.fileno(),),start_new_session=True)
                    code=child.wait(timeout=180)
                finally:
                    if child is not None:
                        try:os.killpg(child.pid,signal.SIGTERM)
                        except ProcessLookupError:pass
                        try:child.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid,signal.SIGKILL);child.wait()
            record={'label':case['label'],'returncode':code,'wall_s':time.monotonic()-before,
                    'report':str(folder/'runtime_report.json')}
            write_new(folder/'controller_receipt.json',record);finished.append(record)
            print('TRIAL_FINISHED',json.dumps(record),flush=True)
            if code:
                write_new(args.output/'finished.json',{'status':'failed_trial','cases':finished})
                raise RuntimeError('Trial failed; live suite will resume after lock release: '+case['label'])
        write_new(args.output/'finished.json',{'status':'executed_not_yet_parity_verified','cases':finished})
        print('ALL_TRIALS_EXECUTED_MAIN_CAN_RESUME',flush=True)


if __name__=='__main__':
    main()
