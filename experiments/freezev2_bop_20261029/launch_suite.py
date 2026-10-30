"""Windows hidden launcher; Linux OS locks prevent duplicate experiment workers."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time


def mount_path(value):
    path=str(Path(value).resolve()).replace('\\','/')
    return '/mnt/'+path[0].lower()+path[2:] if len(path)>2 and path[1]==':' else path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(Path(__file__).with_name('protocol.json')))
    args=parser.parse_args()
    if os.name!='nt':raise RuntimeError('This launcher is Windows-only; invoke source_snapshot/supervise_suite.py directly on WSL')
    protocol=json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    root=Path(protocol['paths']['suite_output_root'])
    snapshot=root/'source_snapshot'
    if not (snapshot/'source_manifest.json').is_file():raise RuntimeError('Prepare and test the immutable source snapshot first')
    import psutil
    bridge=root/'windows_bridge.json'
    if bridge.exists():
        prior=json.loads(bridge.read_text())
        try:
            process=psutil.Process(prior['pid'])
            if abs(process.create_time()-prior['create_time'])<.01 and process.cmdline()==prior['actual_cmdline']:
                raise RuntimeError('The exact existing Windows WSL bridge is still live; do not launch a duplicate')
        except psutil.NoSuchProcess:pass
    python=protocol['paths']['inference_python']
    config=mount_path(root/'protocol.json')
    probe=['wsl.exe','--exec',python,mount_path(snapshot/'run_suite.py'),'--config',config,'--probe-lock']
    checked=subprocess.run(probe,capture_output=True,text=True)
    if checked.returncode:raise RuntimeError('Another live Linux worker owns the suite lock or probe failed: '+checked.stdout+checked.stderr)
    command=['wsl.exe','--exec',python,mount_path(snapshot/'supervise_suite.py'),'--config',config]
    flags=subprocess.CREATE_NO_WINDOW|subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP
    with (root/'bridge.log').open('ab',buffering=0) as log:
        process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,creationflags=flags)
    handle=psutil.Process(process.pid)
    value={'pid':process.pid,'create_time':handle.create_time(),'actual_cmdline':handle.cmdline(),'command':command,'launched_epoch':time.time()}
    temporary=bridge.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');os.replace(temporary,bridge)
    print(json.dumps(value,indent=2))


if __name__=='__main__':main()
