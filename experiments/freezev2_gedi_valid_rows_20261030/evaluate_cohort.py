"""Evaluation-only, failure-inclusive BOP19 scores on one frozen target cohort.

This helper reads no GT in inference and changes no original dataset/target file.
Run with the recorded evaluation venv; each arm gets its own output directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import csv
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

TOOLKIT_REVISION = "cea62d651c7e395b2e1962b9749e4e89693c6ac4"
TARGETS_NAME = "test_targets_fixed_cohort.json"
HEADER = ["scene_id", "im_id", "obj_id", "score", "R", "t", "time"]
METRICS = ["bop19_average_recall", "bop19_average_recall_vsd",
           "bop19_average_recall_mssd", "bop19_average_recall_mspd"]
PUBLIC = {"lmo", "tless", "tudl", "icbin", "ycbv"}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def local_path(value) -> Path:
    text = str(value)
    if os.name != "nt" and re.match(r"^[A-Za-z]:[\\/]", text):
        text = "/mnt/" + text[0].lower() + "/" + text[3:].replace("\\", "/")
    elif os.name == "nt" and re.match(r"^/mnt/[a-z]/", text):
        text = text[5].upper() + ":/" + text[7:]
    return Path(text).expanduser().resolve()


def _key(row):
    return row["scene_id"], row["im_id"], row["obj_id"]


def _targets(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("A nonempty explicit target cohort is required")
    result, seen = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each target must be an official object-image/count row")
        target = {}
        for name, minimum in (("scene_id", 0), ("im_id", 0), ("obj_id", 1), ("inst_count", 1)):
            value = row.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"Invalid target {name}")
            target[name] = value
        if _key(target) in seen:
            raise ValueError("Duplicate target key")
        seen.add(_key(target))
        result.append(target)
    return sorted(result, key=_key)


def validate_cohort(cohort, official, dataset="tless"):
    if isinstance(cohort, dict) and cohort.get("dataset", dataset) != dataset:
        raise ValueError("Cohort dataset differs from requested evaluation dataset")
    selected = _targets(cohort.get("targets") if isinstance(cohort, dict) else cohort)
    full = {_key(row): row for row in _targets(official)}
    for row in selected:
        if _key(row) not in full or row != full[_key(row)]:
            raise ValueError("Cohort must retain the exact official target key and inst_count")
    return selected


def filter_prediction_csv(source: Path, targets):
    """Subset only by preselected localization IDs, never by GT/error/success."""
    expected = {_key(row): row["inst_count"] for row in targets}
    selected, counts, image_times, outside = [], Counter(), {}, 0
    with Path(source).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        if next(reader, None) != HEADER:
            raise ValueError("Unexpected BOP CSV header")
        for row in reader:
            if len(row) != len(HEADER):
                raise ValueError("Malformed BOP CSV row")
            key = tuple(int(x) for x in row[:3])
            if key not in expected:
                outside += 1
                continue
            score, elapsed = float(row[3]), float(row[6])
            rotation = [float(x) for x in row[4].split()]
            translation = [float(x) for x in row[5].split()]
            if len(rotation) != 9 or len(translation) != 3:
                raise ValueError("BOP pose must contain9 rotation and3 translation values")
            if not all(math.isfinite(x) for x in [score, elapsed, *rotation, *translation]) or elapsed < 0:
                raise ValueError("Nonfinite pose/score or negative timing")
            det = (rotation[0]*(rotation[4]*rotation[8]-rotation[5]*rotation[7])
                   - rotation[1]*(rotation[3]*rotation[8]-rotation[5]*rotation[6])
                   + rotation[2]*(rotation[3]*rotation[7]-rotation[4]*rotation[6]))
            gram = [sum(rotation[k*3+i]*rotation[k*3+j] for k in range(3))
                    for i in range(3) for j in range(3)]
            if abs(det-1) > 1e-5 or any(abs(v-float(i//3 == i%3)) > 1e-5 for i,v in enumerate(gram)):
                raise ValueError("Prediction rotation must be a proper SO(3) matrix")
            if key[:2] in image_times and not math.isclose(image_times[key[:2]], elapsed, rel_tol=0, abs_tol=1e-9):
                raise ValueError("Inconsistent per-image CSV timing")
            image_times[key[:2]] = elapsed
            counts[key] += 1
            if counts[key] > expected[key]:
                raise ValueError("Localization CSV exceeds the requested instance count")
            selected.append(row)
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(HEADER)
    writer.writerows(selected)  # original numeric strings, including baseline poses
    missing = [{"scene_id": k[0], "im_id": k[1], "obj_id": k[2],
                "requested_instances": n, "saved_predictions": counts[k],
                "missing_predictions": max(0, n-counts[k])} for k,n in sorted(expected.items())]
    return buffer.getvalue().encode(), {
        "target_rows": len(targets), "requested_instances": sum(expected.values()),
        "target_images": len({k[:2] for k in expected}), "saved_predictions": len(selected),
        "target_rows_without_predictions": sum(counts[k] == 0 for k in expected),
        "missing_prediction_instances": sum(max(0,n-counts[k]) for k,n in expected.items()),
        "excluded_source_rows_outside_fixed_cohort": outside,
        "per_target_coverage": missing,
        "denominator_policy": "All fixed official target rows and inst_count retained, including every failed/blocked/zero-prediction target",
    }


def build_evaluation_command(runtime, *, toolkit: Path, view: Path, results: Path,
                             evaluation: Path, targets_filename=TARGETS_NAME):
    if Path(targets_filename).name != targets_filename or not targets_filename:
        raise ValueError("Targets must be a dataset-local file name")
    if runtime.get("toolkit_revision") != TOOLKIT_REVISION:
        raise ValueError("Unpinned official BOP Toolkit")
    if runtime.get("renderer_type") != "vispy" or runtime.get("protocol", {}).get("visib_gt_min") != -1:
        raise ValueError("Requires recorded BOP19 localization visib_gt_min=-1 protocol")
    python = runtime["python"]
    command = [python, str(toolkit / "scripts/eval_bop19_pose.py"),
               "--result_filenames", results.name, "--results_path", str(results.parent),
               "--eval_path", str(evaluation), "--targets_filename", targets_filename,
               "--renderer_type", "vispy", "--num_workers", "1"]
    env = os.environ.copy()
    env.update(runtime.get("environment", {}))
    env.update(BOP_PATH=str(view), BOP_RESULTS_PATH=str(results.parent), BOP_EVAL_PATH=str(evaluation))
    env["PYTHONPATH"] = str(toolkit) + os.pathsep + env.get("PYTHONPATH", "")
    env["PATH"] = str(Path(python).parent) + os.pathsep + env.get("PATH", "")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "4"
    return command, env


def _immutable(path: Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Incompatible existing evaluation input; use a new output: {path}")
    else:
        with path.open("xb") as stream:
            stream.write(content)


def _dump(path: Path, value):
    content = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    temporary = path.with_name(path.name+f".{os.getpid()}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _pinned_helpers(root: Path):
    snapshot = root / "outputs/freezev2_bop_20261029/classic_core_base_v2/source_snapshot"
    manifest = read_json(snapshot / "source_manifest.json")
    for relative, expected in manifest["files"].items():
        path = (snapshot / relative).resolve()
        if not path.is_relative_to(snapshot.resolve()) or sha256(path) != expected:
            raise ValueError("Frozen baseline helper source identity changed")
    for name, module in list(sys.modules.items()):
        if name == "freezev2_repro" or name.startswith("freezev2_repro."):
            source = getattr(module, "__file__", None)
            if source and not Path(source).resolve().is_relative_to(snapshot.resolve()):
                raise RuntimeError("Use a fresh process for the frozen evaluator helpers")
    sys.path.insert(0, str(snapshot))
    return importlib.import_module("freezev2_repro.durable"), importlib.import_module("freezev2_repro.owned_process")


def _link_dataset_view(source: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    linked = []
    for name in ("models_eval", "test", "test_primesense"):
        original = source / name
        if not original.is_dir():
            continue
        target = destination / name
        if target.exists() or target.is_symlink():
            if not target.is_symlink() or target.resolve() != original.resolve():
                raise ValueError("Existing evaluation view does not point to the recorded assets")
        else:
            target.symlink_to(original.resolve(), target_is_directory=True)
        linked.append({"name": name, "source": str(original.resolve()), "link": str(target)})
    if "models_eval" not in {r["name"] for r in linked} or not ({"test", "test_primesense"} & {r["name"] for r in linked}):
        raise FileNotFoundError("Recorded official evaluation models and sensor split are required")
    return linked


def evaluate_saved(root: Path, cohort_path: Path, csv: Path, output: Path, dataset="tless", *,
                   runtime_path: Path | None = None, execute=True) -> dict:
    root, cohort_path, csv, output = map(local_path, (root, cohort_path, csv, output))
    if dataset not in PUBLIC:
        raise ValueError("This local helper only supports the five public-GT datasets")
    baseline = root / "outputs/freezev2_bop_20261029/classic_core_base_v2"
    if output == baseline or output.is_relative_to(baseline):
        raise ValueError("New evaluation must not change the frozen baseline outputs")
    runtime_path = local_path(runtime_path or root / "outputs/freezev2_bop_20261029/evaluation_runtime.json")
    runtime, cohort = read_json(runtime_path), read_json(cohort_path)
    view_source, toolkit = local_path(runtime["dataset_view"]), local_path(runtime["toolkit_root"])
    source_targets = view_source / dataset / "test_targets_bop19.json"
    targets = validate_cohort(cohort, read_json(source_targets), dataset)
    if isinstance(cohort, dict) and cohort.get("official_targets", {}).get("sha256"):
        if cohort["official_targets"]["sha256"] != sha256(source_targets):
            raise ValueError("Cohort's pinned official target source changed")
    content, coverage = filter_prediction_csv(csv, targets)
    revision = subprocess.check_output(["git", "-C", str(toolkit), "rev-parse", "HEAD"], text=True).strip()
    if revision != TOOLKIT_REVISION:
        raise ValueError("Actual official toolkit revision differs from the pinned runtime")
    if not (toolkit / "scripts/eval_bop19_pose.py").is_file():
        raise FileNotFoundError("Official evaluator script missing")
    result = output / "results" / f"freezev2cohort_{dataset}-test.csv"
    view, evaluation = output / "bop_view", output / "official_evaluation"
    targets_file = view / dataset / TARGETS_NAME
    target_bytes = (json.dumps(targets, indent=2, sort_keys=True)+"\n").encode()
    pins = {"input_csv_sha256": sha256(csv), "cohort_sha256": sha256(cohort_path),
            "official_targets_sha256": sha256(source_targets), "cohort_targets_sha256": hashlib.sha256(target_bytes).hexdigest(),
            "evaluated_csv_sha256": hashlib.sha256(content).hexdigest(), "runtime_sha256": sha256(runtime_path),
            "helper_sha256": sha256(Path(__file__)), "toolkit_revision": revision,
            "official_script_sha256": sha256(toolkit / "scripts/eval_bop19_pose.py"),
            "frozen_helpers_manifest_sha256": (sha256(baseline / "source_snapshot/source_manifest.json")
                if (baseline / "source_snapshot/source_manifest.json").is_file() else None)}
    identity = {"schema": "freezev2-fixed-cohort-official-evaluation-v1", "dataset": dataset, "source_pins": pins}
    output.mkdir(parents=True, exist_ok=True)
    _immutable(output / "identity.json", (json.dumps(identity, indent=2, sort_keys=True)+"\n").encode())
    _immutable(result, content)
    _immutable(targets_file, target_bytes)
    command, environment = build_evaluation_command(runtime, toolkit=toolkit, view=view, results=result, evaluation=evaluation)
    entry = {**identity, **coverage, "status": "prepared_not_evaluated", "command": command,
             "source_csv": str(csv), "evaluated_csv": str(result), "cohort_path": str(cohort_path),
             "targets_path": str(targets_file), "runtime_path": str(runtime_path),
             "metrics": None, "GT_role": "evaluation_only", "visib_gt_min": -1,
             "metric_scope": "Fixed diagnostic cohort only; not the full TLESS benchmark or published-performance parity",
             "predictions_origin": "our saved inference; authors' predictions not used",
             "timing_scope": "CSV timings are preserved. A filtered full-scene baseline may include work outside this cohort; do not treat these as controlled cohort runtime comparisons."}
    receipt = output / "evaluation_receipt.json"
    if receipt.is_file():
        previous = read_json(receipt)
        if previous.get("source_pins") != pins:
            raise ValueError("Existing metric receipt uses incompatible inputs")
        score = Path(previous.get("score_file", ""))
        if previous.get("status") == "complete" and score.is_file() and sha256(score) == previous.get("score_sha256"):
            return previous
    if not execute:
        return entry
    if os.name != "posix" or Path(sys.prefix).resolve() != Path(runtime["python"]).parent.parent.resolve():
        raise RuntimeError("Execute this helper using the recorded WSL evaluation Python")
    durable, owned = _pinned_helpers(root)
    evaluation.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        lock = stack.enter_context(durable.worker_lock(root / "outputs/freezev2_gedi_valid_rows_20261030/evaluation.lock"))
        stack.enter_context(durable.worker_lock(output / "evaluation.lock"))
        entry["asset_links"] = _link_dataset_view(view_source / dataset, view / dataset)
        def heartbeat(pid):
            durable.atomic_json(output / "evaluation_status.json", {"pid": os.getpid(), "start_ticks": owned.start_ticks(os.getpid()),
                                "guard_pid": pid, "epoch": time.time(), "status": "official_evaluation", "source_pins": pins})
        started = time.perf_counter()
        code = owned.run_owned(command, environment, evaluation, lock, heartbeat)
        entry.update(evaluator_returncode=code, runtime_s=time.perf_counter()-started)
        score = evaluation / result.stem / "scores_bop19.json"
        if code != 0 or not score.is_file():
            entry.update(status="failed_official_evaluator", log=str(evaluation / "execution.log"))
            _dump(receipt, entry)
            durable.atomic_json(output / "evaluation_status.json", {"status": "failed", "epoch": time.time(), "source_pins": pins})
            raise RuntimeError(f"Official cohort evaluation failed; inspect {evaluation / 'execution.log'}")
        values = read_json(score)
        metrics = {name: float(values[name]) for name in METRICS}
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in metrics.values()):
            raise ValueError("Official evaluator produced invalid pose recalls")
        entry.update(status="complete", metrics=metrics, score_file=str(score), score_sha256=sha256(score),
                     official_format="BOP19 localization, VSD/MSSD/MSPD, unchanged cohort inst_count most-visible GT semantics")
        _dump(receipt, entry)
        durable.atomic_json(output / "evaluation_status.json", {"status": "complete", "epoch": time.time(), "source_pins": pins})
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="tless")
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    result = evaluate_saved(args.root, args.cohort, args.csv, args.output, args.dataset,
                            runtime_path=args.runtime, execute=not args.prepare_only)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
