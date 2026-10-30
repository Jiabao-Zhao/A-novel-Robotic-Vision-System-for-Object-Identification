"""Linux evaluator ownership: inherited worker lock and parent-death cleanup."""
from pathlib import Path
import argparse
import json
import os
import signal
import subprocess
import sys
import time

from .durable import atomic_json,read_json


def start_ticks(pid):
    try:
        # comm may contain spaces or parentheses; fields after its final ')' are
        # stable. starttime is field22, i.e. index19 after comm.
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    except (FileNotFoundError,ProcessLookupError):return None


def stop_group(process):
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait()
    # The official orchestrator may exit before one of its error-calculation
    # descendants. Do not equate a dead group leader with an empty group.
    try:os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:pass


def run_owned(command,environment,folder,lock_stream,heartbeat):
    """The guard holds the inherited flock until all evaluator children stop."""
    if os.name!='posix':raise RuntimeError('Official evaluator ownership requires the recorded WSL runtime')
    folder=Path(folder)
    request=folder/'owned_request.json'
    atomic_json(request,{'command':command,'parent_pid':os.getpid(),'parent_start_ticks':start_ticks(os.getpid())})
    guard_command=[sys.executable,'-m','freezev2_repro.owned_process','--request',str(request),'--lock-fd',str(lock_stream.fileno())]
    environment=dict(environment)
    environment['PYTHONPATH']=str(Path(__file__).resolve().parent.parent)+os.pathsep+environment.get('PYTHONPATH','')
    with (folder/'execution.log').open('a',encoding='utf-8') as log:
        guard=subprocess.Popen(guard_command,env=environment,stdout=log,stderr=subprocess.STDOUT,
                               pass_fds=(lock_stream.fileno(),),start_new_session=True)
        try:
            while guard.poll() is None:
                heartbeat(guard.pid);time.sleep(2)
            return guard.returncode
        finally:
            if guard.poll() is None:
                guard.terminate()
                try:guard.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # Never release the inherited lock by killing the guard
                    # before its owned child group has been reconciled.
                    status=read_json(folder/'owned_process.json') if (folder/'owned_process.json').exists() else {}
                    pid=status.get('child_pid')
                    if pid and start_ticks(pid)==status.get('child_start_ticks'):
                        try:os.killpg(pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                    guard.kill();guard.wait()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True);parser.add_argument('--lock-fd',type=int,required=True)
    args=parser.parse_args();request=read_json(args.request);folder=Path(args.request).parent
    os.fstat(args.lock_fd)  # fail closed if the inherited worker lock is absent
    stopped=False
    def stop(signum,frame):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    child=None
    try:
        if start_ticks(request['parent_pid'])!=request['parent_start_ticks']:return 143
        child=subprocess.Popen(request['command'],start_new_session=True)
        atomic_json(folder/'owned_process.json',{'guard_pid':os.getpid(),'guard_start_ticks':start_ticks(os.getpid()),
                    'child_pid':child.pid,'child_start_ticks':start_ticks(child.pid),'command':request['command'],'started_epoch':time.time()})
        while child.poll() is None:
            if stopped or start_ticks(request['parent_pid'])!=request['parent_start_ticks']:
                stop_group(child);return 143
            time.sleep(.25)
        return child.returncode
    finally:
        if child is not None:stop_group(child)
        os.close(args.lock_fd)


if __name__=='__main__':raise SystemExit(main())
