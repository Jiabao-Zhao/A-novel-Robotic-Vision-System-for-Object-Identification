"""Validate and summarize the matched ten-task LIBERO-Object comparison."""

import csv
import json
from pathlib import Path

from simulation.libero_experiment import (
    EXPERIMENT_ROOT,
    LIBERO_OBJECT_TASKS,
    PROPOSED_METHOD_FOLDER,
    VLA_METHOD_FOLDER,
    episode_result_dir,
)


COMPARISON_OUTPUT_ROOT = EXPERIMENT_ROOT
EXPECTED_TASK_INDICES = tuple(task[0] for task in LIBERO_OBJECT_TASKS)
EXPECTED_SEED = 1000
EXPECTED_EPISODE_HORIZON_STEPS = 280
EXPECTED_CONTROL_FREQUENCY_HZ = 20
EXPECTED_CONTROL_MODE = "relative"
EXPECTED_OBSERVATION_RESOLUTION_HW = [256, 256]
EXPECTED_SETTLE_STEPS = 10


def build_comparison(proposed_results, vla_runs):
    """Build a success-only comparison from one state-0 run per task and method."""
    proposed_by_task = _index_results(proposed_results, "proposed method")
    vla_by_task = _index_vla_runs(vla_runs)
    _require_exact_task_coverage(proposed_by_task, "proposed method")
    _require_exact_task_coverage(vla_by_task, "VLA method")

    rows = []
    tasks = []
    fairness_checks = {}
    proposed_method_names = set()
    vla_method_names = set()

    for task_index, target_object, expected_instruction in LIBERO_OBJECT_TASKS:
        proposed = proposed_by_task[task_index]
        vla_run = vla_by_task[task_index]
        vla_manifest = vla_run["manifest"]
        vla_eval = vla_run["eval"]
        vla_episode = _normalize_vla_episode(vla_manifest, vla_eval, task_index)

        checks = _matched_protocol_checks(
            proposed,
            vla_manifest,
            vla_episode,
            task_index,
            expected_instruction,
        )
        failures = [name for name, passed in checks.items() if not passed]
        if failures:
            raise ValueError(
                f"Task {task_index} ({target_object}) is not matched; failed checks: "
                + ", ".join(failures)
            )
        fairness_checks[f"task_{task_index:02d}_{target_object}"] = checks

        proposed_success = _require_bool(proposed.get("success"), "proposed success")
        proposed_run_status = proposed.get("run_status", "completed")
        if proposed_run_status not in {"completed", "pipeline_error"}:
            raise ValueError(
                f"Task {task_index} has invalid proposed run_status="
                f"{proposed_run_status!r}."
            )
        if proposed_success and proposed_run_status != "completed":
            raise ValueError(
                f"Task {task_index} succeeded but is not marked completed."
            )
        if (
            proposed_run_status == "pipeline_error"
            and proposed.get("termination_reason") != "pipeline_error"
        ):
            raise ValueError(
                f"Task {task_index} pipeline-error status and termination disagree."
            )
        if (proposed.get("termination_reason") == "success") != proposed_success:
            raise ValueError(
                f"Task {task_index} proposed success and termination_reason disagree."
            )
        proposed_method = _require_text(proposed.get("method"), "proposed method")
        vla_method = _require_text(vla_manifest.get("method"), "VLA method")
        proposed_method_names.add(proposed_method)
        vla_method_names.add(vla_method)

        proposed_artifact = str(
            episode_result_dir(PROPOSED_METHOD_FOLDER, task_index, 0) / "episode.json"
        )
        vla_artifact = str(
            episode_result_dir(VLA_METHOD_FOLDER, task_index, 0) / "eval_info.json"
        )
        task_rows = [
            {
                "method": proposed_method,
                "suite": proposed["suite"],
                "task_index": task_index,
                "target_object": target_object,
                "initial_state_index": 0,
                "initial_state_sha256": proposed["initial_state_sha256"],
                "seed": proposed["seed"],
                "run_status": proposed_run_status,
                "success": proposed_success,
                "termination_reason": proposed.get("termination_reason"),
                "artifact": proposed_artifact,
            },
            {
                "method": vla_method,
                "suite": vla_manifest["suite"],
                "task_index": task_index,
                "target_object": target_object,
                "initial_state_index": 0,
                "initial_state_sha256": vla_episode["initial_state_sha256"],
                "seed": vla_episode["seed"],
                "run_status": "completed",
                "success": vla_episode["success"],
                "termination_reason": None,
                "artifact": vla_artifact,
            },
        ]
        rows.extend(task_rows)
        tasks.append(
            {
                "task_index": task_index,
                "target_object": target_object,
                "instruction": expected_instruction,
                "initial_state_index": 0,
                "initial_state_sha256": proposed["initial_state_sha256"],
                "results": {
                    row["method"]: {
                        "success": row["success"],
                        "artifact": row["artifact"],
                    }
                    for row in task_rows
                },
            }
        )

    if len(proposed_method_names) != 1 or len(vla_method_names) != 1:
        raise ValueError(
            "Each experiment branch must use one consistent method name; found "
            f"proposed={sorted(proposed_method_names)}, VLA={sorted(vla_method_names)}."
        )
    if proposed_method_names == vla_method_names:
        raise ValueError("The proposed and VLA branches must have distinct method names.")

    methods = {}
    for row in rows:
        summary = methods.setdefault(
            row["method"],
            {"episodes": 0, "successes": 0, "failures": 0, "success_rate": 0.0},
        )
        summary["episodes"] += 1
        summary["successes"] += int(row["success"])
    for summary in methods.values():
        summary["failures"] = summary["episodes"] - summary["successes"]
        summary["success_rate"] = summary["successes"] / summary["episodes"]

    return {
        "schema_version": 1,
        "comparison_valid": True,
        "scope": (
            "all ten LIBERO-Object product tasks at official fixed initial-state "
            "index 0; a breadth test, not a within-task robustness benchmark"
        ),
        "primary_metric": {
            "name": "binary_task_success_rate",
            "definition": (
                "Number of episodes satisfying LIBERO check_success() divided by "
                "ten completed task episodes"
            ),
            "speed_or_action_count_used": False,
        },
        "coverage": {
            "expected_task_indices": list(EXPECTED_TASK_INDICES),
            "matched_task_count": len(tasks),
            "episodes_per_method": len(tasks),
            "initial_state_indices_per_task": [0],
        },
        "fairness_checks": fairness_checks,
        "methods": methods,
        "tasks": tasks,
        "episodes": rows,
    }


def main():
    proposed_results = []
    vla_runs = []
    for task_index, _, _ in LIBERO_OBJECT_TASKS:
        proposed_root = episode_result_dir(PROPOSED_METHOD_FOLDER, task_index, 0)
        vla_root = episode_result_dir(VLA_METHOD_FOLDER, task_index, 0)
        proposed_results.append(_load_json(proposed_root / "episode.json"))
        vla_runs.append(
            {
                "manifest": _load_json(vla_root / "experiment_manifest.json"),
                "eval": _load_json(vla_root / "eval_info.json"),
            }
        )

    comparison = build_comparison(proposed_results, vla_runs)
    COMPARISON_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    comparison_path = COMPARISON_OUTPUT_ROOT / "comparison_summary.json"
    csv_path = COMPARISON_OUTPUT_ROOT / "per_episode.csv"
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    _write_rows(csv_path, comparison["episodes"])

    print("Matched LIBERO all-product state-0 comparison validated")
    for method, metrics in comparison["methods"].items():
        print(
            f"  {method}: {metrics['successes']}/{metrics['episodes']} "
            f"success ({metrics['success_rate']:.1%})"
        )
    print("Primary metric: binary task success rate (no speed ranking)")
    print(f"Comparison JSON: {comparison_path}")
    print(f"Per-episode CSV: {csv_path}")


def _index_results(results, label):
    indexed = {}
    for result in results:
        if not isinstance(result, dict):
            raise ValueError(f"Each {label} result must be a dictionary.")
        task_index = result.get("task_index")
        if not isinstance(task_index, int) or isinstance(task_index, bool):
            raise ValueError(f"{label} has an invalid task_index={task_index!r}.")
        if task_index in indexed:
            raise ValueError(f"Duplicate {label} result for task {task_index}.")
        indexed[task_index] = result
    return indexed


def _index_vla_runs(runs):
    indexed = {}
    for run in runs:
        if not isinstance(run, dict) or not isinstance(run.get("manifest"), dict):
            raise ValueError("Each VLA run must contain a manifest dictionary.")
        if not isinstance(run.get("eval"), dict):
            raise ValueError("Each VLA run must contain an eval dictionary.")
        task_index = run["manifest"].get("task_index")
        if not isinstance(task_index, int) or isinstance(task_index, bool):
            raise ValueError(f"VLA manifest has invalid task_index={task_index!r}.")
        if task_index in indexed:
            raise ValueError(f"Duplicate VLA method result for task {task_index}.")
        indexed[task_index] = run
    return indexed


def _require_exact_task_coverage(indexed, label):
    actual = set(indexed)
    expected = set(EXPECTED_TASK_INDICES)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"{label} task coverage mismatch; missing={missing}, unexpected={unexpected}."
        )


def _normalize_vla_episode(manifest, eval_info, task_index):
    task = _find_vla_task(eval_info, manifest.get("suite"), task_index)
    successes = list(task.get("metrics", {}).get("successes", []))
    if len(successes) != 1:
        raise ValueError(
            f"VLA task {task_index} must contain exactly one success value; "
            f"found {len(successes)}."
        )
    success = _require_bool(successes[0], f"VLA task {task_index} success")
    if "success" in manifest:
        manifest_success = _require_bool(
            manifest["success"], f"VLA task {task_index} manifest success"
        )
        if manifest_success != success:
            raise ValueError(
                f"VLA task {task_index} manifest success disagrees with eval_info.json."
            )

    initial_state_index = _single_manifest_value(
        manifest,
        scalar_key="initial_state_index",
        list_key="initial_state_indices",
        label=f"VLA task {task_index} initial state",
    )
    initial_state_sha256 = _single_manifest_value(
        manifest,
        scalar_key="initial_state_sha256",
        list_key="initial_state_sha256s",
        label=f"VLA task {task_index} initial-state hash",
    )
    episode_seed = _single_manifest_value(
        manifest,
        scalar_key="episode_seed",
        list_key="episode_seeds",
        label=f"VLA task {task_index} episode seed",
        fallback_key="seed",
    )
    if not _is_integer(initial_state_index):
        raise ValueError(
            f"VLA task {task_index} initial-state index must be an integer."
        )
    if not _is_integer(episode_seed):
        raise ValueError(f"VLA task {task_index} episode seed must be an integer.")

    overall_episode_count = eval_info.get("overall", {}).get("n_episodes")
    if overall_episode_count is not None and overall_episode_count != 1:
        raise ValueError(
            f"VLA task {task_index} eval_info reports {overall_episode_count} "
            "episodes; expected exactly one."
        )
    return {
        "success": success,
        "initial_state_index": initial_state_index,
        "initial_state_sha256": initial_state_sha256,
        "seed": episode_seed,
    }


def _single_manifest_value(
    manifest,
    scalar_key,
    list_key,
    label,
    fallback_key=None,
):
    scalar_present = scalar_key in manifest
    scalar_value = manifest.get(scalar_key)
    list_present = list_key in manifest
    values = manifest.get(list_key)
    list_value = None
    if list_present:
        if not isinstance(values, list) or len(values) != 1:
            raise ValueError(f"{label} must contain exactly one value.")
        list_value = values[0]
    if scalar_present and list_present and scalar_value != list_value:
        raise ValueError(f"{label} scalar and list forms disagree.")
    if scalar_present:
        return scalar_value
    if list_present:
        return list_value
    if fallback_key is not None and fallback_key in manifest:
        return manifest[fallback_key]
    raise ValueError(
        f"{label} is missing ({scalar_key!r} or {list_key!r} required)."
    )


def _matched_protocol_checks(
    proposed,
    vla_manifest,
    vla_episode,
    task_index,
    expected_instruction,
):
    proposed_hash = proposed.get("initial_state_sha256")
    vla_hash = vla_episode["initial_state_sha256"]
    return {
        "schema_versions": proposed.get("schema_version") == 1
        and vla_manifest.get("schema_version") == 1,
        "distinct_method_names": bool(proposed.get("method"))
        and bool(vla_manifest.get("method"))
        and proposed.get("method") != vla_manifest.get("method"),
        "suite": proposed.get("suite") == vla_manifest.get("suite") == "libero_object",
        "task_index": proposed.get("task_index")
        == vla_manifest.get("task_index")
        == task_index,
        "instruction": proposed.get("instruction")
        == vla_manifest.get("instruction")
        == expected_instruction,
        "seed": _is_integer(proposed.get("seed"))
        and proposed.get("seed") == vla_episode["seed"] == EXPECTED_SEED,
        "initial_state_index_is_zero": _is_integer(
            proposed.get("initial_state_index")
        )
        and proposed.get("initial_state_index")
        == vla_episode["initial_state_index"]
        == 0,
        "initial_state_hash_recorded": _is_sha256(proposed_hash)
        and _is_sha256(vla_hash),
        "initial_state_bytes": proposed_hash == vla_hash,
        "episode_horizon_steps": proposed.get("episode_horizon_steps")
        == vla_manifest.get("episode_horizon_steps")
        == EXPECTED_EPISODE_HORIZON_STEPS,
        "control_frequency_hz": proposed.get("control_frequency_hz")
        == vla_manifest.get("control_frequency_hz")
        == EXPECTED_CONTROL_FREQUENCY_HZ,
        "control_mode": proposed.get("control_mode")
        == vla_manifest.get("control_mode")
        == EXPECTED_CONTROL_MODE,
        "observation_resolution_hw": proposed.get("observation_resolution_hw")
        == vla_manifest.get("observation_resolution_hw")
        == EXPECTED_OBSERVATION_RESOLUTION_HW,
        "settle_steps": proposed.get("initial_physics_settle_steps")
        == vla_manifest.get("settle_steps_before_policy")
        == EXPECTED_SETTLE_STEPS,
        "same_success_predicate": proposed.get("success_predicate")
        == vla_manifest.get("success_predicate")
        == "LIBERO task environment check_success()",
        "vla_eval_completed": vla_manifest.get("return_code") == 0,
        "vla_state_hash_completed": not vla_manifest.get("initial_state_hash_error"),
    }


def _find_vla_task(vla_eval, suite, task_index):
    matches = [
        item
        for item in vla_eval.get("per_task", [])
        if item.get("task_group") == suite and item.get("task_id") == task_index
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one VLA result for {suite}[{task_index}], found {len(matches)}."
        )
    return matches[0]


def _require_bool(value, label):
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be true or false; received {value!r}.")
    return value


def _require_text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    return value


def _is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _is_integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _load_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Required result is missing: {path}. Run both methods for all ten "
            "state-0 tasks first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _write_rows(path, rows):
    fieldnames = list(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"LIBERO comparison unavailable: {error}") from error
