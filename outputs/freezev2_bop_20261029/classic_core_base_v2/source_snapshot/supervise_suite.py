"""Bounded fresh-process recovery for the single unattended suite worker."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    protocol=json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]=str(protocol['resources']['cpu_workers'])
    from freezev2_repro.pipeline import local_path
    from freezev2_repro.durable import atomic_json,worker_lock
    root=local_path(protocol['paths']['suite_output_root'])
    with worker_lock(root/'supervisor.lock'):
        for attempt in range(4):
            command=[sys.executable,str(Path(__file__).with_name('run_suite.py')),'--config',args.config,'--resume']
            with (root/'worker.log').open('a',encoding='utf-8') as log:
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                try:
                    while process.poll() is None:
                        info={'supervisor_pid':os.getpid(),'worker_pid':process.pid,'attempt':attempt,'updated_epoch':time.time(),'command':command}
                        if Path(f'/proc/{process.pid}/stat').exists():info['worker_start_ticks']=Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
                        atomic_json(root/'supervisor_status.json',info)
                        time.sleep(5)
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:process.wait(timeout=15)
                        except subprocess.TimeoutExpired:process.kill();process.wait()
            atomic_json(root/'supervisor_exit.json',{'epoch':time.time(),'worker_returncode':process.returncode,'attempt':attempt})
            if process.returncode==0:return 0
            if process.returncode!=75:return process.returncode
            time.sleep(15)
        return 75


if __name__=='__main__':raise SystemExit(main())
