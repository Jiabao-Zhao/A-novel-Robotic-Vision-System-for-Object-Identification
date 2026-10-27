"""Execute unchanged frozen scoring with each NPZ array loaded once."""
from pathlib import Path
import numpy as np
import run_benchmarks as trial


if __name__ == '__main__':
    trial.check_sources()
    assert not (trial.HERE/'completion.json').exists()
    assert not (trial.HERE/'candidate_scores.json').exists()
    assert not (trial.HERE/'selections.json').exists()
    original_load = np.load
    cached = {}

    def load_arrays_once(path, *args, **kwargs):
        absolute = Path(path).resolve()
        if absolute.suffix != '.npz':
            return original_load(path, *args, **kwargs)
        if absolute not in cached:
            with original_load(path, *args, **kwargs) as archive:
                cached[absolute] = {key: archive[key] for key in archive.files}
        return cached[absolute]

    trial.save(trial.HERE/'execution_adjustment.json', dict(
        change='Materialize every NPZ array once in memory; no scoring or input changes.',
        reason='Frozen score loop repeatedly decompressed the complete feature archive for indexed reads.',
        stopped_process_pid=295,
        stopped_process_argv=['/home/jiabao/.venvs/lerobot-libero/bin/python',
            '/mnt/d/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/experiments/nist12_matching_benchmarks_20261023/run_benchmarks.py','score'],
        stopped_before_any_scores_or_selections_written=True,
        scoring_rules_changed=False, feature_values_or_dtypes_changed=False,
        protocol_sha256=trial.digest(trial.HERE/'protocol.json'),
        features_sha256=trial.digest(trial.HERE/'features.npz'),
        adapter_sha256=trial.digest(__file__)))
    np.load = load_arrays_once
    trial.score()
