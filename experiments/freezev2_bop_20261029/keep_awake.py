"""Hold a process-scoped Windows sleep request while the exact suite bridge lives.

No power-plan/global setting is changed. Windows releases the request when this
thread exits, including termination; the display is allowed to sleep normally.
"""
from pathlib import Path
import argparse
import ctypes
import json
import os
import time
import psutil


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',required=True)
    parser.add_argument('--bridge-file',default='windows_bridge.json')
    parser.add_argument('--state-file',default='keep_awake_process.json')
    args=parser.parse_args();root=Path(args.run_root)
    if any(Path(value).name!=value for value in (args.bridge_file,args.state_file)):
        raise ValueError('Bridge and state names must be plain filenames inside the run root')
    bridge=json.loads((root/args.bridge_file).read_text())
    me=psutil.Process()
    state=root/args.state_file
    if state.exists():
        previous=json.loads(state.read_text())
        try:
            owner=psutil.Process(previous['pid'])
            if abs(owner.create_time()-previous['create_time'])<.01 and owner.cmdline()==previous['cmdline']:
                raise RuntimeError('The exact existing sleep-request helper is still live')
        except psutil.NoSuchProcess:pass
    if os.name!='nt':raise RuntimeError('Windows-only resource helper')
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    set_state=kernel.SetThreadExecutionState
    set_state.argtypes=[ctypes.c_uint];set_state.restype=ctypes.c_uint
    if not set_state(0x80000001):raise ctypes.WinError(ctypes.get_last_error())
    try:
        while True:
            try:
                owner=psutil.Process(bridge['pid'])
                if abs(owner.create_time()-bridge['create_time'])>.01 or owner.cmdline()!=bridge['actual_cmdline']:break
            except psutil.NoSuchProcess:break
            info={'pid':me.pid,'create_time':me.create_time(),'cmdline':me.cmdline(),
                  'bridge_pid':owner.pid,'updated_epoch':time.time(),
                  'scope':'process-scoped system sleep request only; display may sleep; no power plan modification'}
            temporary=state.with_suffix('.tmp')
            temporary.write_text(json.dumps(info,indent=2)+'\n',encoding='utf-8');temporary.replace(state)
            time.sleep(10)
    finally:set_state(0x80000000)


if __name__=='__main__':main()
