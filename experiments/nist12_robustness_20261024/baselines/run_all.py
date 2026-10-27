"""One GPU stage at a time, native imports isolated in separate processes."""
from pathlib import Path
import subprocess
import sys
import time
import common

ROOT = Path(__file__).resolve().parent
STAGES = [('cnos','propose'), ('cnos','match'), ('sam6d','propose'), ('sam6d','match'),
          ('nids_basic_finite_templates','prepare'), ('nids_basic_finite_templates','propose'),
          ('nids_basic_finite_templates','match'), ('noctis','prepare'), ('noctis','propose'),
          ('noctis','match'), ('nemo','match')]


if __name__ == '__main__':
    for method, phase in STAGES:
        log = ROOT / 'logs' / f'{method}_{phase}.log'
        log.parent.mkdir(exist_ok=True)
        started = time.perf_counter()
        print(f'START {method} {phase}', flush=True)
        with log.open('a', encoding='utf-8') as stream:
            result = subprocess.run([sys.executable, '-u', str(ROOT / 'adapter.py'), method, phase],
                                    stdout=stream, stderr=subprocess.STDOUT)
        common.save(log.with_suffix('.status.json'), dict(method=method, phase=phase,
            exit_code=result.returncode, seconds=time.perf_counter()-started))
        print(f'END {method} {phase}: exit={result.returncode}', flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)
    common.save(ROOT / 'completion.json', dict(complete=True, scenes=20,
        methods=['cnos','sam6d_rgb','sam6d_rgbd','nids_basic_finite_templates','noctis','nemo'],
        labels_read=False, protocol_sha256=common.digest(ROOT / 'protocol.json')))
