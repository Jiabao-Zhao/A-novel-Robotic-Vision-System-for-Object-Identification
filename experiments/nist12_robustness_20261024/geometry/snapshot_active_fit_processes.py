"""Read-only process health for this study's registered fit workers."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

def main():
    rows = []
    uptime = float(Path('/proc/uptime').read_text().split()[0])
    ticks = os.sysconf('SC_CLK_TCK')
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit(): continue
        try:
            argv = [p.decode() for p in (folder/'cmdline').read_bytes().split(b'\0') if p]
            if not any('/nist12_robustness_20261024/' in p and
                any(s in p for s in ('/geometry/run_depth.py', '/geometry/queue_cohorts.py'))
                for p in argv): continue
            stat = (folder/'stat').read_text().rsplit(')',1)[1].split()
            memory = {}
            try:
                for line in (folder/'smaps_rollup').read_text().splitlines():
                    name, _, value = line.partition(':')
                    if name in ('Rss','Pss','Private_Clean','Private_Dirty'):
                        memory[name+'_KiB'] = int(value.strip().split()[0])
            except (OSError, ValueError): pass
            rows.append(dict(pid=int(folder.name), ppid=int(stat[1]), state=stat[0],
                start_ticks=int(stat[19]), elapsed_seconds=uptime-int(stat[19])/ticks,
                CPU_seconds=(int(stat[11])+int(stat[12]))/ticks,
                threads=int(stat[17]), argv=argv, memory=memory))
        except (OSError, ProcessLookupError, ValueError): continue
    report = dict(utc=datetime.now(timezone.utc).isoformat(), processes=rows,
        process_actions=False, labels_read=False)
    with (HERE/'fit_process_health.jsonl').open('a') as output:
        output.write(json.dumps(report)+'\n')
    print(json.dumps(report))

if __name__ == '__main__':
    main()
