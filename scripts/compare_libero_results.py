"""Validate and summarize a matched task-7 proposed-method/VLA comparison."""

import csv
import json
from pathlib import Path

from simulation.libero_experiment import (
    EXPERIMENT_ROOT,
    PROPOSED_METHOD_FOLDER,
    VLA_METHOD_FOLDER,
    episode_result_dir,
)

PROPOSED_OUTPUT_ROOT = episode_result_dir(PROPOSED_METHOD_FOLDER, 7, 0)
PROPOSED_EPISODE_PATH = PROPOSED_OUTPUT_ROOT / "episode.json"
VLA_OUTPUT_ROOT = episode_result_dir(VLA_METHOD_FOLDER, 7, 0)
COMPARISON_OUTPUT_ROOT = EXPERIMENT_ROOT


def build_comparison(proposed, vla_manifest, vla_eval):
    vla_task = _find_vla_task(vla_eval, proposed["suite"], proposed["task_index"])
    episode_seeds = list(vla_manifest.get("episode_seeds", []))
    initial_state_indices = list(vla_manifest.get("initial_state_indices", []))
    vla_seed = vla_manifest.get("seed")
    checks = {
        "schema_versions": proposed.get("schema_version") == 1
        and vla_manifest.get("schema_version") == 1,
        "method_names": bool(proposed.get("method"))
        and bool(vla_manifest.get("method"))
        and proposed.get("method") != vla_manifest.get("method"),
        "suite": proposed.get("suite") == vla_manifest.get("suite"),
        "task_index": proposed.get("task_index") == vla_manifest.get("task_index"),
        "instruction": proposed.get("instruction") == vla_manifest.get("instruction"),
        "seed": proposed.get("seed") == vla_manifest.get("seed"),
        "episode_seeds": [proposed.get("seed")] == episode_seeds,
        "vla_episode_seed_sequence": isinstance(vla_seed, int)
        and episode_seeds
        == [vla_seed + index for index in range(len(episode_seeds))],
        "initial_state": [proposed.get("initial_state_index")] == initial_state_indices,
        "vla_initial_state_sequence": initial_state_indices
        == list(range(len(initial_state_indices))),
        "initial_state_bytes": [proposed.get("initial_state_sha256")]
        == vla_manifest.get("initial_state_sha256s"),
        "episode_horizon_steps": proposed.get("episode_horizon_steps")
        == vla_manifest.get("episode_horizon_steps"),
        "control_frequency_hz": proposed.get("control_frequency_hz")
        == vla_manifest.get("control_frequency_hz"),
        "control_mode": proposed.get("control_mode")
        == vla_manifest.get("control_mode"),
        "observation_resolution_hw": proposed.get("observation_resolution_hw")
        == vla_manifest.get("observation_resolution_hw"),
        "settle_steps": proposed.get("initial_physics_settle_steps")
        == vla_manifest.get("settle_steps_before_policy"),
        "fixed_initial_state_recorded": bool(proposed.get("initial_state_sha256")),
        "vla_eval_completed": vla_manifest.get("return_code") == 0,
        "vla_state_hash_completed": not vla_manifest.get("initial_state_hash_error"),
        "vla_step_inspection_completed": not vla_manifest.get("rollout_metrics_error"),
        "same_success_predicate": proposed.get("success_predicate")
        == vla_manifest.get("success_predicate"),
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError(
            "Comparison is not matched; failed checks: " + ", ".join(failures)
        )

    successes = list(vla_task["metrics"].get("successes", []))
    expected_episodes = len(initial_state_indices)
    if len(successes) != expected_episodes:
        raise ValueError(
            "VLA result count does not match its fixed initial-state list: "
            f"{len(successes)} results versus {expected_episodes} states."
        )
    overall_episode_count = vla_eval.get("overall", {}).get("n_episodes")
    if overall_episode_count != expected_episodes:
        raise ValueError(
            "VLA overall episode count does not match its fixed initial-state list: "
            f"{overall_episode_count} versus {expected_episodes}."
        )
    rollout_step_counts = list(vla_manifest.get("rollout_step_counts", []))
    first_success_steps = list(vla_manifest.get("first_success_steps", []))
    if len(rollout_step_counts) != expected_episodes:
        raise ValueError(
            "VLA manifest is missing one rollout step count per evaluated episode."
        )
    if len(first_success_steps) != expected_episodes:
        raise ValueError(
            "VLA manifest is missing one first-success value per evaluated episode."
        )

    horizon = proposed["episode_horizon_steps"]
    _validate_episode_metrics(
        proposed["method"],
        bool(proposed["success"]),
        proposed.get("first_success_step"),
        proposed.get("action_steps"),
        horizon,
    )
    if (proposed.get("termination_reason") == "success") != bool(proposed["success"]):
        raise ValueError(
            "Proposed-method success and termination_reason are inconsistent."
        )
    for episode_index, success in enumerate(successes):
        _validate_episode_metrics(
            f"{vla_manifest['method']} episode {episode_index}",
            bool(success),
            first_success_steps[episode_index],
            rollout_step_counts[episode_index],
            horizon,
        )

    rows = [
        {
            "method": proposed["method"],
            "suite": proposed["suite"],
            "task_index": proposed["task_index"],
            "episode_index": 0,
            "initial_state_index": proposed["initial_state_index"],
            "seed": proposed["seed"],
            "success": bool(proposed["success"]),
            "first_success_step": proposed.get("first_success_step"),
            "action_steps": proposed.get("action_steps"),
            "artifact": str(PROPOSED_EPISODE_PATH),
        }
    ]
    for episode_index, success in enumerate(successes):
        rows.append(
            {
                "method": vla_manifest["method"],
                "suite": vla_manifest["suite"],
                "task_index": vla_manifest["task_index"],
                "episode_index": episode_index,
                "initial_state_index": initial_state_indices[episode_index],
                "seed": episode_seeds[episode_index],
                "success": bool(success),
                "first_success_step": first_success_steps[episode_index],
                "action_steps": rollout_step_counts[episode_index],
                "artifact": str(VLA_OUTPUT_ROOT / "eval_info.json"),
            }
        )

    methods = {}
    for row in rows:
        summary = methods.setdefault(
            row["method"], {"episodes": 0, "successes": 0, "success_rate": 0.0}
        )
        summary["episodes"] += 1
        summary["successes"] += int(row["success"])
    for summary in methods.values():
        summary["success_rate"] = summary["successes"] / summary["episodes"]

    return {
        "comparison_valid": True,
        "scope": "task-7 integration pilot; not a statistically meaningful benchmark",
        "fairness_checks": checks,
        "methods": methods,
        "episodes": rows,
    }


def main():
    proposed = _load_json(PROPOSED_EPISODE_PATH)
    vla_manifest = _load_json(VLA_OUTPUT_ROOT / "experiment_manifest.json")
    vla_eval = _load_json(VLA_OUTPUT_ROOT / "eval_info.json")
    comparison = build_comparison(proposed, vla_manifest, vla_eval)

    COMPARISON_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    comparison_path = COMPARISON_OUTPUT_ROOT / "comparison_summary.json"
    csv_path = COMPARISON_OUTPUT_ROOT / "per_episode.csv"
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    _write_rows(csv_path, comparison["episodes"])

    print("Matched LIBERO task-7 comparison validated")
    for method, metrics in comparison["methods"].items():
        print(
            f"  {method}: {metrics['successes']}/{metrics['episodes']} "
            f"success ({metrics['success_rate']:.1%})"
        )
    print(f"Comparison JSON: {comparison_path}")
    print(f"Per-episode CSV: {csv_path}")


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


def _validate_episode_metrics(label, success, first_success_step, action_steps, horizon):
    if not isinstance(action_steps, int) or isinstance(action_steps, bool):
        raise ValueError(f"{label} has an invalid action-step count: {action_steps!r}.")
    if not 0 <= action_steps <= horizon:
        raise ValueError(
            f"{label} action steps {action_steps} exceed the shared horizon {horizon}."
        )
    if success:
        if (
            not isinstance(first_success_step, int)
            or isinstance(first_success_step, bool)
            or not 1 <= first_success_step <= action_steps
        ):
            raise ValueError(
                f"{label} succeeded but has invalid first_success_step="
                f"{first_success_step!r}."
            )
    elif first_success_step is not None:
        raise ValueError(
            f"{label} failed but records first_success_step={first_success_step!r}."
        )


def _load_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Required result is missing: {path}. Run both episode scripts first."
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
    except (FileNotFoundError, KeyError, ValueError) as error:
        raise SystemExit(f"LIBERO comparison unavailable: {error}") from error
