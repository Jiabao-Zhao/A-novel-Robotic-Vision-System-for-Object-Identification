"""Read-only Windows host load snapshot for the one controlled query audit."""
import ctypes
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import psutil

destination = Path(sys.argv[1]).resolve()
assert destination.is_relative_to(Path(__file__).resolve().parent)
virtual_processes = []
for item in psutil.process_iter(['pid', 'name']):
    if item.info['name'] == 'vmmemWSL':
        try:
            times = item.cpu_times()
            virtual_processes.append((item, times.user+times.system))
        except psutil.Error:
            pass
started = time.perf_counter()
before = psutil.cpu_times()._asdict()
total_cpu = psutil.cpu_percent(interval=.35)
elapsed = time.perf_counter()-started
virtual_cpu, virtual_available = 0., bool(virtual_processes)
for item, old in virtual_processes:
    try:
        times = item.cpu_times()
        virtual_cpu += 100*(times.user+times.system-old)/(elapsed*psutil.cpu_count())
    except psutil.Error:
        virtual_available = False
memory = (ctypes.c_ulonglong*8)()
memory[0] = 64
assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
sample = {'utc': datetime.now(timezone.utc).isoformat(), 'host_available_GiB': memory[2]/2**30,
          'sample_seconds': elapsed, 'logical_CPUs': psutil.cpu_count(), 'host_cpu_percent': total_cpu,
          'host_cpu_times_before': before, 'host_cpu_times_after': psutil.cpu_times()._asdict(),
          'vmmemWSL_cpu_percent': virtual_cpu if virtual_available else None,
          'non_vmmem_host_cpu_percent_estimate': max(0., total_cpu-virtual_cpu) if virtual_available else None,
          'note': 'Short load sample; vmmemWSL is unavailable when host permissions hide its CPU counters.',
          'process_actions': False, 'labels_read': False}
temporary = destination.with_suffix(destination.suffix+'.part')
temporary.write_text(json.dumps(sample, indent=2, allow_nan=False)+'\n')
temporary.replace(destination)
print(json.dumps(sample, indent=2))
