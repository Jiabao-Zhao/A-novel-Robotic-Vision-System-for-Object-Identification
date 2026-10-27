"""Read-only health and one-second CPU deltas for this exact ordinary pool."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

OUT = Path(__file__).resolve().parent


def snapshot():
    rows = []
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit():
            continue
        try:
            stat = (folder/'stat').read_text().rsplit(')', 1)[1].split()
            argv = [v.decode() for v in (folder/'cmdline').read_bytes().split(b'\0') if v]
            if int(folder.name) not in (3733827, 3733828) and int(stat[1]) != 3733828:
                continue
            rows.append(dict(pid=int(folder.name), ppid=int(stat[1]), state=stat[0],
                start_ticks=int(stat[19]), CPU_ticks=int(stat[11])+int(stat[12]), threads=int(stat[17]),
                argv=argv, wchan=(folder/'wchan').read_text().strip()))
        except OSError:
            continue
    return rows


first = snapshot()
started = time.perf_counter()
time.sleep(1)
second = snapshot()
before = {(row['pid'], row['start_ticks']): row for row in first}
for row in second:
    old = before.get((row['pid'], row['start_ticks']))
    row['CPU_delta_seconds'] = None if old is None else (row['CPU_ticks']-old['CPU_ticks'])/os.sysconf('SC_CLK_TCK')
runner = next(row for row in second if row['pid'] == 3733828)
assert runner['start_ticks'] == 2775476
workers = [row for row in second if row['ppid'] == 3733828 and '--multiprocessing-fork' in row['argv']]
now = datetime.now(timezone.utc)
report = dict(utc=now.isoformat(), sample_seconds=time.perf_counter()-started,
    worker_count=len(workers), all8_workers_have_positive_CPU_delta=len(workers)==8 and all(row['CPU_delta_seconds']>0 for row in workers),
    processes=second, progress=json.loads((OUT/'progress.json').read_text()),
    new_fits_run=0, preprocessing_calls=0, process_actions=False, labels_read=False)
file = OUT/('ordinary_pool_health_'+now.strftime('%Y%m%dT%H%M%S')+'.json')
file.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
