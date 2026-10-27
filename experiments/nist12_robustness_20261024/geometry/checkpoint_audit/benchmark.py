"""Read-only mirror of one all-scene score checkpoint; never writes live scores."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

TRIAL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TRIAL))
from common import read, relative

OUT = TRIAL / 'geometry'
AUDIT = Path(__file__).resolve().parent


def snapshot_progress():
    file = OUT / 'progress.json'
    return {'utc': datetime.now(timezone.utc).isoformat(),
            'mtime_ns': file.stat().st_mtime_ns, 'value': read(file)}


def main():
    index_file = OUT / 'pair_index.json'
    index_bytes = index_file.read_bytes()
    index = json.loads(index_bytes.decode('utf-8-sig'))['scene_pairs']
    progress_before = snapshot_progress()
    counts = Counter()
    phase = Counter()
    seen = Counter()
    scenes = []
    started = perf_counter()
    for scene_id, rows in index.items():
        pairs = []
        detail = {'scene_id': scene_id, 'total_pairs': len(rows), 'reads': 0}
        complete = True
        for row in rows:
            step = perf_counter()
            fit_file = OUT / 'fits' / row['fit_key'][:2] / (row['fit_key'] + '.json')
            exists = fit_file.exists()
            phase['path_and_exists_seconds'] += perf_counter() - step
            counts['exists_checks'] += 1
            if not exists:
                complete = False
                detail['first_missing_key'] = row['fit_key']
                break
            step = perf_counter()
            record = read(fit_file)
            phase['read_parse_seconds'] += perf_counter() - step
            step = perf_counter()
            pairs.append(dict(row, fit_file=relative(fit_file), status=record['status'],
                              geometry_score=record['geometry_score'],
                              size_score=record['size_score']))
            phase['relative_and_dict_seconds'] += perf_counter() - step
            counts['json_reads'] += 1
            detail['reads'] += 1
            seen[row['fit_key']] += 1
        detail['complete'] = complete
        if complete:
            step = perf_counter()
            serialized = json.dumps({'scene_id': scene_id, 'pairs': pairs,
                'labels_read': False,
                'availability': 'Unavailable candidates remain explicit; no task-level all-candidates veto.'},
                indent=2, allow_nan=False) + '\n'
            phase['serialization_seconds'] += perf_counter() - step
            counts['serialized_characters'] += len(serialized)
            counts['complete_scenes'] += 1
        else:
            counts['incomplete_scenes'] += 1
        scenes.append(detail)
    elapsed = perf_counter() - started
    progress_after = snapshot_progress()
    result = {
        'utc': datetime.now(timezone.utc).isoformat(),
        'mode': 'Read-only: no live score writes, fits, model calls, or labels.',
        'scenes': len(index), 'elapsed_seconds': elapsed, **counts,
        'unique_fit_files_read': len(seen),
        'repeated_fit_reads': sum(seen.values()) - len(seen),
        'phase_seconds': dict(phase),
        'uninstrumented_bookkeeping_seconds': elapsed - sum(phase.values()),
        'file_write_seconds': None,
        'file_write_note': 'Unmeasured; serialization mirrored exactly but mkdir/write/rename deliberately omitted.',
        'measurement_note': 'Per-row timers/counting add small instrumentation overhead; reads and Path.resolve match live scene_scores.',
        'progress_before': progress_before, 'progress_after': progress_after,
        'pair_index_sha256': hashlib.sha256(index_bytes).hexdigest(),
        'scorer_source_sha256': hashlib.sha256((OUT / 'run_depth.py').read_bytes()).hexdigest(),
        'benchmark_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'labels_read': False,
        'scene_details': scenes,
    }
    destination = AUDIT / 'measurement.json'
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'scene_details'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
