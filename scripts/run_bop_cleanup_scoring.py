"""Finish the frozen comparison with two CPU workers and serialized GPU encoding.

Run only after run_bop_cleanup_lccp has saved all masks, with that serial runner
stopped. Separate processes own the CAD raycasters. All score functions and
input masks are unchanged; a shared lock bounds DINO's GPU memory use.
"""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
from time import perf_counter

from scripts import run_bop_cleanup_lccp as experiment
from scripts import run_surface_verification as scoring
import torch


_prepare_observations = scoring.prepare_observations
_gpu_lock = None
_engine = None


def guarded_observations(*args, **kwargs):
    with _gpu_lock:
        return _prepare_observations(*args, **kwargs)


def initialize(gpu_lock):
    global _gpu_lock, _engine
    os.chdir(experiment.ROOT)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    _gpu_lock = gpu_lock
    scoring.prepare_observations = guarded_observations
    with _gpu_lock:
        _engine = experiment.baseline.cached_templates()


def run_frame(item):
    for variant in experiment.VARIANTS:
        experiment.score_variant(item, variant, _engine)
    return item


def main():
    protocol = experiment.read_json(experiment.OUTPUT / "protocol.json")
    for path, digest in protocol["experiment_sha256"].items():
        assert experiment.baseline.association.content_hash(experiment.ROOT / path) == digest
    for item in protocol["plan"]:
        name = experiment.baseline.frame_folder(item).name
        assert all((experiment.OUTPUT / v / name / "localization.json").exists() for v in experiment.VARIANTS)
    execution = {"workers": 2, "gpu_encoding": "serialized by shared process lock",
        "geometry": "unchanged CPU scoring in independent processes",
        "driver_sha256": experiment.baseline.association.content_hash(experiment.ROOT / "scripts/run_bop_cleanup_scoring.py")}
    experiment.save_json(experiment.OUTPUT / "execution.json", execution)
    start = perf_counter()
    context = mp.get_context("spawn")
    lock = context.Lock()
    with ProcessPoolExecutor(max_workers=2, mp_context=context, initializer=initialize, initargs=(lock,)) as pool:
        for item in pool.map(run_frame, protocol["plan"]):
            print("FRAME COMPLETE", item, flush=True)
    results = {"baseline": experiment.read_json(experiment.BASE / "summary.json")}
    try:
        for variant in experiment.VARIANTS:
            experiment.baseline.OUTPUT = experiment.OUTPUT / variant
            results[variant] = experiment.baseline.evaluate_saved(protocol["plan"])
            experiment.save_json(experiment.baseline.OUTPUT / "summary.json", results[variant])
    finally:
        experiment.baseline.OUTPUT = experiment.BASE
    assert protocol["protected_sha256"] == {p: experiment.baseline.association.content_hash(experiment.ROOT / p)
                                            for p in experiment.baseline.PROTECTED}
    experiment.save_json(experiment.OUTPUT / "comparison.json", {"variants": results,
        "parallel_scoring_and_evaluation_wall_s": perf_counter() - start,
        "protected_sources_unchanged": True, "tuning_performed": False, "execution": execution})
    print("COMPARISON COMPLETE", flush=True)


if __name__ == "__main__":
    main()
