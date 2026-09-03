"""Run the official LeRobot SmolVLA evaluator for LIBERO-Object tasks.

The official evaluator is kept as a separate controller branch. It does not
import or use this repository's depth, VLM, CAD-registration, or scripted
control modules. A single evaluator process handles all requested tasks so the
checkpoint is loaded only once. Its untouched combined output is preserved,
then one validated result package is copied into each task's experiment folder.
"""

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from simulation.libero_experiment import (
    EXPERIMENT_ROOT,
    LIBERO_OBJECT_TASKS,
    VLA_METHOD_FOLDER,
    episode_result_dir,
)


CHECKPOINT = "HuggingFaceVLA/smolvla_libero"
CHECKPOINT_REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
SUITE_NAME = "libero_object"
METHOD_NAME = "smolvla_official_lerobot_evaluator"
SUCCESS_PREDICATE = "LIBERO task environment check_success()"
INITIAL_STATE_SOURCE = "official LIBERO task .pruned_init array"
SETTLE_STEPS_BEFORE_POLICY = 10
METHOD_INPUTS = (
    "rendered agentview RGB",
    "rendered wrist RGB",
    "8D robot state",
    "language instruction",
)

# Retained for compatibility with the completed task-7 pilot and its tests.
TASK_INDEX = 7
TASK_INSTRUCTION = "pick up the milk and place it in the basket"
ALL_TASK_INDICES = tuple(task_index for task_index, _, _ in LIBERO_OBJECT_TASKS)

SEED = 1000
INITIAL_STATE_INDICES = [0]
EPISODE_HORIZON_STEPS = 280
CONTROL_FREQUENCY_HZ = 20
CONTROL_MODE = "relative"
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
OUTPUT_ROOT = episode_result_dir(VLA_METHOD_FOLDER, TASK_INDEX, INITIAL_STATE_INDICES[0])
BATCH_RUNS_ROOT = EXPERIMENT_ROOT / VLA_METHOD_FOLDER / "_batch_runs"


def build_eval_command(output_root=OUTPUT_ROOT, task_indices=None):
    """Build one official evaluator invocation for the requested task IDs.

    Omitting ``task_indices`` retains the original task-7 command behavior for
    callers that use this helper directly. ``main()`` explicitly supplies all
    pending task IDs.
    """
    _validate_initial_state_indices()
    selected = (
        (TASK_INDEX,)
        if task_indices is None
        else _normalize_task_indices(task_indices)
    )
    output_root = Path(output_root).resolve()
    task_ids = ",".join(str(task_index) for task_index in selected)
    return [
        "lerobot-eval",
        f"--policy.path={CHECKPOINT}",
        f"--policy.pretrained_revision={CHECKPOINT_REVISION}",
        "--policy.device=cuda",
        "--env.type=libero",
        f"--env.task={SUITE_NAME}",
        f"--env.task_ids=[{task_ids}]",
        f"--env.control_mode={CONTROL_MODE}",
        f"--env.fps={CONTROL_FREQUENCY_HZ}",
        f"--env.observation_height={IMAGE_HEIGHT}",
        f"--env.observation_width={IMAGE_WIDTH}",
        f"--env.episode_length={EPISODE_HORIZON_STEPS}",
        "--env.init_states=true",
        "--env.hard_reset=true",
        "--env.max_parallel_tasks=1",
        "--eval.batch_size=1",
        "--eval.use_async_envs=false",
        # LeRobot applies n_episodes to every selected task. With batch size 1,
        # one episode selects official fixed initial state 0 for each task.
        f"--eval.n_episodes={len(INITIAL_STATE_INDICES)}",
        f"--seed={SEED}",
        f"--output_dir={output_root}",
    ]


def main(task_indices=None, staging_output_root=None):
    """Evaluate pending tasks, defaulting to every unfinished LIBERO-Object task.

    Passing ``ALL_TASK_INDICES`` explicitly requests the full catalog, but any
    already-complete task is safely skipped. Existing validated task results are
    never overwritten.
    """
    executable = shutil.which("lerobot-eval")
    if executable is None:
        raise SystemExit(
            "lerobot-eval is unavailable. Activate ~/.venvs/lerobot-libero and "
            "install the LeRobot source checkout with `python -m pip install "
            "-e \".[libero,smolvla]\"`."
        )

    requested = (
        ALL_TASK_INDICES
        if task_indices is None
        else _normalize_task_indices(task_indices)
    )
    pending, completed = _partition_completed_tasks(requested)
    if completed:
        labels = ", ".join(_task_label(task_index) for task_index in completed)
        print(f"Keeping completed VLA results unchanged: {labels}")
    if not pending:
        print("All requested VLA task/state-0 evaluations are already complete.")
        return

    _ensure_publish_targets_are_empty(pending)
    output_root = (
        _new_staging_output_root(pending)
        if staging_output_root is None
        else Path(staging_output_root).resolve()
    )
    if output_root.exists():
        raise SystemExit(
            f"Refusing to overwrite the VLA batch staging output: {output_root}. "
            "Choose a new staging_output_root."
        )

    command = build_eval_command(output_root, task_indices=pending)
    environment = os.environ.copy()
    environment.setdefault("MUJOCO_GL", "egl")
    environment.setdefault("PYOPENGL_PLATFORM", environment["MUJOCO_GL"])

    print("VLA baseline: SmolVLA (independent from RGB-D/VLM/CAD branch)")
    print("Tasks: " + ", ".join(_task_label(task_index) for task_index in pending))
    print("Official fixed initial state: 0 for every task")
    print(f"Preserved batch staging output: {output_root}")
    print("Command:")
    command_separator = " \\" + "\n  "
    print("  " + command_separator.join(command))

    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    result = subprocess.run(command, env=environment, check=False)
    elapsed_seconds = time.perf_counter() - started

    # Preserve diagnostics even when the evaluator fails before creating output.
    output_root.mkdir(parents=True, exist_ok=True)
    eval_info_path = output_root / "eval_info.json"
    eval_info = (
        json.loads(eval_info_path.read_text(encoding="utf-8"))
        if eval_info_path.is_file()
        else None
    )

    initial_state_hashes = {}
    initial_state_hash_error = None
    try:
        initial_state_hashes = _initial_state_hashes_by_task(pending)
    except Exception as error:
        initial_state_hash_error = str(error)

    task_records = {}
    rollout_metrics_error = None
    if result.returncode == 0 and eval_info is not None:
        try:
            task_records = _parse_batch_results(eval_info, pending)
        except Exception as error:
            rollout_metrics_error = str(error)

    lerobot_source = _lerobot_source_info()
    batch_manifest = _batch_manifest(
        task_indices=pending,
        command=command,
        output_root=output_root,
        environment=environment,
        started_at=started_at,
        elapsed_seconds=elapsed_seconds,
        return_code=result.returncode,
        initial_state_hashes=initial_state_hashes,
        initial_state_hash_error=initial_state_hash_error,
        rollout_metrics_error=rollout_metrics_error,
        task_records=task_records,
        lerobot_source=lerobot_source,
    )
    batch_manifest_path = output_root / "experiment_manifest.json"
    batch_manifest_path.write_text(json.dumps(batch_manifest, indent=2), encoding="utf-8")

    if result.returncode != 0:
        raise SystemExit(
            f"LeRobot evaluation failed with exit code {result.returncode}. "
            f"Preserved diagnostics: {batch_manifest_path}"
        )
    if not eval_info_path.is_file():
        raise SystemExit(
            f"LeRobot exited successfully but did not create {eval_info_path}. "
            f"Preserved diagnostics: {batch_manifest_path}"
        )
    if rollout_metrics_error is not None:
        raise SystemExit(
            "LeRobot evaluation completed, but per-task rollout inspection failed: "
            f"{rollout_metrics_error}. Preserved batch: {output_root}"
        )
    if initial_state_hash_error is not None:
        raise SystemExit(
            "LeRobot evaluation completed, but fixed-state hashing failed: "
            f"{initial_state_hash_error}. Preserved batch: {output_root}"
        )

    published = _publish_task_results(
        output_root=output_root,
        batch_manifest=batch_manifest,
        task_records=task_records,
        initial_state_hashes=initial_state_hashes,
    )
    print(f"Combined evaluation metrics: {eval_info_path}")
    print(f"Combined experiment manifest: {batch_manifest_path}")
    for task_index, task_output in published.items():
        print(f"Published {_task_label(task_index)}: {task_output}")


def _normalize_task_indices(task_indices):
    if not isinstance(task_indices, (list, tuple)):
        raise TypeError("task_indices must be a list or tuple of LIBERO-Object task IDs.")
    if not task_indices:
        raise ValueError("task_indices must contain at least one task ID.")
    normalized = tuple(int(task_index) for task_index in task_indices)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"task_indices contains duplicates: {normalized}")
    unknown = [task_index for task_index in normalized if task_index not in ALL_TASK_INDICES]
    if unknown:
        raise ValueError(
            f"Unknown LIBERO-Object task IDs {unknown}; available IDs are "
            f"{list(ALL_TASK_INDICES)}."
        )
    return tuple(sorted(normalized))


def _partition_completed_tasks(task_indices):
    completed = tuple(
        task_index for task_index in task_indices if _task_result_is_complete(task_index)
    )
    pending = tuple(task_index for task_index in task_indices if task_index not in completed)
    return pending, completed


def _task_result_is_complete(task_index):
    output_root = episode_result_dir(VLA_METHOD_FOLDER, task_index, 0)
    manifest_path = output_root / "experiment_manifest.json"
    eval_info_path = output_root / "eval_info.json"
    if not manifest_path.is_file() or not eval_info_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        eval_info = json.loads(eval_info_path.read_text(encoding="utf-8"))
        task_result = _find_vla_task(eval_info, task_index)
        official_hash = _official_state_zero_sha256(task_index)
    except Exception:
        return False

    if not _manifest_matches_current_protocol(manifest, task_index, official_hash):
        return False
    metrics = task_result.get("metrics", {})
    if any(
        len(metrics.get(key, [])) != 1
        for key in ("sum_rewards", "max_rewards", "successes", "video_paths")
    ):
        return False
    if eval_info.get("overall", {}).get("n_episodes") != 1:
        return False
    official_video = (
        output_root
        / "videos"
        / f"{SUITE_NAME}_{task_index}"
        / "eval_episode_0.mp4"
    )
    if not official_video.is_file():
        return False

    success = bool(metrics["successes"][0])
    if "success" in manifest and manifest["success"] is not success:
        return False
    rollout_steps = manifest.get("rollout_step_counts", [])
    first_success_steps = manifest.get("first_success_steps", [])
    if len(rollout_steps) != 1 or len(first_success_steps) != 1:
        return False
    if success:
        return first_success_steps[0] == rollout_steps[0]
    return first_success_steps[0] is None


def _manifest_matches_current_protocol(manifest, task_index, official_hash):
    """Require exact task, checkpoint, sensor, control, and state-0 provenance."""
    scalar_aliases_match = (
        ("episode_seed" not in manifest or manifest["episode_seed"] == SEED)
        and (
            "initial_state_index" not in manifest
            or manifest["initial_state_index"] == 0
        )
        and (
            "initial_state_sha256" not in manifest
            or manifest["initial_state_sha256"] == official_hash
        )
    )
    return (
        manifest.get("schema_version") == 1
        and manifest.get("method") == METHOD_NAME
        and manifest.get("controller_branch") == "vla_baseline"
        and manifest.get("checkpoint") == CHECKPOINT
        and manifest.get("checkpoint_revision") == CHECKPOINT_REVISION
        and manifest.get("checkpoint_is_libero_finetuned") is True
        and manifest.get("local_finetuning_performed") is False
        and manifest.get("suite") == SUITE_NAME
        and manifest.get("task_index") == task_index
        and manifest.get("instruction") == _task_instruction(task_index)
        and manifest.get("success_predicate") == SUCCESS_PREDICATE
        and manifest.get("seed") == SEED
        and manifest.get("episode_seeds") == [SEED]
        and manifest.get("initial_state_indices") == [0]
        and manifest.get("initial_state_sha256s") == [official_hash]
        and manifest.get("initial_state_source") == INITIAL_STATE_SOURCE
        and manifest.get("episode_horizon_steps") == EPISODE_HORIZON_STEPS
        and manifest.get("control_frequency_hz") == CONTROL_FREQUENCY_HZ
        and manifest.get("control_mode") == CONTROL_MODE
        and manifest.get("observation_resolution_hw") == [IMAGE_HEIGHT, IMAGE_WIDTH]
        and manifest.get("settle_steps_before_policy")
        == SETTLE_STEPS_BEFORE_POLICY
        and manifest.get("method_inputs") == list(METHOD_INPUTS)
        and manifest.get("depth_or_cad_inputs_used") is False
        and manifest.get("return_code") == 0
        and manifest.get("initial_state_hash_error") is None
        and manifest.get("rollout_metrics_error") is None
        and scalar_aliases_match
    )


def _ensure_publish_targets_are_empty(task_indices):
    for task_index in task_indices:
        output_root = episode_result_dir(VLA_METHOD_FOLDER, task_index, 0)
        existing_files = list(output_root.rglob("*")) if output_root.exists() else []
        existing_files = [path for path in existing_files if path.is_file()]
        if existing_files:
            raise SystemExit(
                f"Refusing to overwrite an incomplete VLA result for "
                f"{_task_label(task_index)}: {output_root}. Move the partial "
                "artifacts aside before retrying."
            )


def _new_staging_output_root(task_indices):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    task_slug = "-".join(f"{task_index:02d}" for task_index in task_indices)
    return (BATCH_RUNS_ROOT / f"{timestamp}_state00_tasks_{task_slug}").resolve()


def _parse_batch_results(eval_info, task_indices):
    expected_count = len(task_indices)
    overall_count = eval_info.get("overall", {}).get("n_episodes")
    if overall_count != expected_count:
        raise ValueError(
            "LeRobot overall episode count does not match one episode per selected task: "
            f"{overall_count} versus {expected_count}."
        )

    records = {}
    for task_index in task_indices:
        task_result = _find_vla_task(eval_info, task_index)
        metrics = task_result.get("metrics", {})
        for key in ("sum_rewards", "max_rewards", "successes", "video_paths"):
            values = metrics.get(key, [])
            if len(values) != 1:
                raise ValueError(
                    f"{SUITE_NAME}[{task_index}] must contain exactly one {key} "
                    f"value, found {len(values)}."
                )
        rollout_steps, first_success_steps = _rollout_step_metrics(
            eval_info, task_index=task_index
        )
        records[task_index] = {
            "metrics": metrics,
            "success": bool(metrics["successes"][0]),
            "action_steps": rollout_steps[0],
            "first_success_step": first_success_steps[0],
        }
    return records


def _find_vla_task(eval_info, task_index):
    matches = [
        item
        for item in eval_info.get("per_task", [])
        if item.get("task_group") == SUITE_NAME and item.get("task_id") == task_index
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one VLA result for {SUITE_NAME}[{task_index}], "
            f"found {len(matches)}."
        )
    return matches[0]


def _publish_task_results(
    *, output_root, batch_manifest, task_records, initial_state_hashes
):
    """Copy validated batch artifacts to empty task folders; write manifests last."""
    task_indices = tuple(task_records)
    _ensure_publish_targets_are_empty(task_indices)

    # Validate every source before creating any task result package.
    for task_index, record in task_records.items():
        for source in record["metrics"].get("video_paths", []):
            if not Path(source).is_file():
                raise FileNotFoundError(
                    f"LeRobot video for {_task_label(task_index)} is missing: {source}"
                )
        for source in record["metrics"].get("predicted_video_paths", []):
            if not Path(source).is_file():
                raise FileNotFoundError(
                    f"LeRobot predicted video for {_task_label(task_index)} is missing: "
                    f"{source}"
                )

    published = {}
    for task_index, record in task_records.items():
        task_output = episode_result_dir(VLA_METHOD_FOLDER, task_index, 0).resolve()
        task_output.mkdir(parents=True, exist_ok=True)
        metrics = dict(record["metrics"])
        metrics["video_paths"] = _copy_artifacts(
            metrics.get("video_paths", []),
            task_output / "videos" / f"{SUITE_NAME}_{task_index}",
        )
        metrics["predicted_video_paths"] = _copy_artifacts(
            metrics.get("predicted_video_paths", []),
            task_output / "predicted_videos" / f"{SUITE_NAME}_{task_index}",
        )

        per_task_eval = _single_task_eval_info(task_index, metrics)
        (task_output / "eval_info.json").write_text(
            json.dumps(per_task_eval, indent=2), encoding="utf-8"
        )
        task_manifest = _task_manifest(
            task_index=task_index,
            output_root=output_root,
            batch_manifest=batch_manifest,
            record=record,
            initial_state_hash=initial_state_hashes[task_index],
            published_video_paths=metrics["video_paths"],
        )
        # The manifest is written last and acts as the completion marker.
        (task_output / "experiment_manifest.json").write_text(
            json.dumps(task_manifest, indent=2), encoding="utf-8"
        )
        published[task_index] = task_output
    return published


def _copy_artifacts(source_paths, destination_dir):
    if not source_paths:
        return []
    destination_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for source_path in source_paths:
        source = Path(source_path)
        destination = destination_dir / source.name
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite published artifact: {destination}")
        shutil.copy2(source, destination)
        copied.append(str(destination.resolve()))
    return copied


def _single_task_eval_info(task_index, metrics):
    successes = list(metrics["successes"])
    sum_rewards = list(metrics["sum_rewards"])
    max_rewards = list(metrics["max_rewards"])
    videos = list(metrics.get("video_paths", []))
    predicted_videos = list(metrics.get("predicted_video_paths", []))
    summary = {
        "avg_sum_reward": float(np.mean(sum_rewards)),
        "avg_max_reward": float(np.mean(max_rewards)),
        "pc_success": float(np.mean(successes) * 100),
        "n_episodes": 1,
        "video_paths": videos,
        "predicted_video_paths": predicted_videos,
    }
    return {
        "per_task": [
            {
                "task_group": SUITE_NAME,
                "task_id": task_index,
                "metrics": metrics,
            }
        ],
        "per_group": {SUITE_NAME: dict(summary)},
        "overall": dict(summary),
    }


def _batch_manifest(
    *,
    task_indices,
    command,
    output_root,
    environment,
    started_at,
    elapsed_seconds,
    return_code,
    initial_state_hashes,
    initial_state_hash_error,
    rollout_metrics_error,
    task_records,
    lerobot_source,
):
    return {
        "schema_version": 1,
        "method": METHOD_NAME,
        "controller_branch": "vla_baseline",
        "run_scope": "multi_task_state_0_batch",
        "checkpoint": CHECKPOINT,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_is_libero_finetuned": True,
        "local_finetuning_performed": False,
        "suite": SUITE_NAME,
        "task_indices": list(task_indices),
        "tasks": [
            {
                "task_index": task_index,
                "instruction": _task_instruction(task_index),
                "initial_state_index": 0,
                "initial_state_sha256": initial_state_hashes.get(task_index),
                "episode_seed": SEED,
                "success": task_records.get(task_index, {}).get("success"),
                "action_steps": task_records.get(task_index, {}).get("action_steps"),
                "first_success_step": task_records.get(task_index, {}).get(
                    "first_success_step"
                ),
            }
            for task_index in task_indices
        ],
        "success_predicate": SUCCESS_PREDICATE,
        "seed": SEED,
        "episode_seed_policy": "Each selected task receives seed 1000 independently.",
        "initial_state_indices_per_task": [0],
        "initial_state_sha256s_by_task": {
            str(task_index): digest
            for task_index, digest in initial_state_hashes.items()
        },
        "initial_state_hash_error": initial_state_hash_error,
        "initial_state_source": INITIAL_STATE_SOURCE,
        "episode_horizon_steps": EPISODE_HORIZON_STEPS,
        "control_frequency_hz": CONTROL_FREQUENCY_HZ,
        "control_mode": CONTROL_MODE,
        "observation_resolution_hw": [IMAGE_HEIGHT, IMAGE_WIDTH],
        "settle_steps_before_policy": SETTLE_STEPS_BEFORE_POLICY,
        "method_inputs": list(METHOD_INPUTS),
        "depth_or_cad_inputs_used": False,
        "command": command,
        "staging_output": str(output_root),
        "staging_preserved": True,
        "mujoco_gl": environment["MUJOCO_GL"],
        "lerobot_version": _package_version("lerobot"),
        "lerobot_source": lerobot_source,
        "torch_version": _package_version("torch"),
        "started_at_utc": started_at.isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "return_code": return_code,
        "rollout_step_count_source": (
            "MP4 frame count; pinned LeRobot evaluator renders one frame per action step"
        ),
        "rollout_metrics_error": rollout_metrics_error,
    }


def _task_manifest(
    *,
    task_index,
    output_root,
    batch_manifest,
    record,
    initial_state_hash,
    published_video_paths,
):
    return {
        "schema_version": 1,
        "method": METHOD_NAME,
        "controller_branch": "vla_baseline",
        "checkpoint": CHECKPOINT,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_is_libero_finetuned": True,
        "local_finetuning_performed": False,
        "suite": SUITE_NAME,
        "task_index": task_index,
        "instruction": _task_instruction(task_index),
        "success_predicate": SUCCESS_PREDICATE,
        "seed": SEED,
        "episode_seed": SEED,
        "episode_seeds": [SEED],
        "initial_state_index": 0,
        "initial_state_indices": [0],
        "initial_state_sha256": initial_state_hash,
        "initial_state_sha256s": [initial_state_hash],
        "initial_state_hash_error": None,
        "initial_state_source": INITIAL_STATE_SOURCE,
        "episode_horizon_steps": EPISODE_HORIZON_STEPS,
        "control_frequency_hz": CONTROL_FREQUENCY_HZ,
        "control_mode": CONTROL_MODE,
        "observation_resolution_hw": [IMAGE_HEIGHT, IMAGE_WIDTH],
        "settle_steps_before_policy": SETTLE_STEPS_BEFORE_POLICY,
        "method_inputs": list(batch_manifest["method_inputs"]),
        "depth_or_cad_inputs_used": False,
        "success": record["success"],
        "action_steps": record["action_steps"],
        "first_success_step": record["first_success_step"],
        "rollout_step_counts": [record["action_steps"]],
        "first_success_steps": [record["first_success_step"]],
        "rollout_step_count_source": batch_manifest["rollout_step_count_source"],
        "rollout_metrics_error": None,
        "video_paths": published_video_paths,
        "command": list(batch_manifest["command"]),
        "batch_task_indices": list(batch_manifest["task_indices"]),
        "batch_staging_output": str(output_root),
        "batch_manifest": str((Path(output_root) / "experiment_manifest.json").resolve()),
        "staging_preserved": True,
        "mujoco_gl": batch_manifest["mujoco_gl"],
        "lerobot_version": batch_manifest["lerobot_version"],
        "lerobot_source": batch_manifest["lerobot_source"],
        "torch_version": batch_manifest["torch_version"],
        "started_at_utc": batch_manifest["started_at_utc"],
        "elapsed_seconds": batch_manifest["elapsed_seconds"],
        "return_code": batch_manifest["return_code"],
    }


def _task_label(task_index):
    _, object_name, _ = next(
        task for task in LIBERO_OBJECT_TASKS if task[0] == task_index
    )
    return f"{SUITE_NAME}[{task_index}] ({object_name})"


def _task_instruction(task_index):
    return next(task[2] for task in LIBERO_OBJECT_TASKS if task[0] == task_index)


def _package_version(package_name):
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _validate_initial_state_indices():
    expected = list(range(len(INITIAL_STATE_INDICES)))
    if INITIAL_STATE_INDICES != expected:
        raise ValueError(
            "The official LeRobot evaluator selects sequential initial states "
            f"{expected}; configured INITIAL_STATE_INDICES={INITIAL_STATE_INDICES}. "
            "A custom evaluator is required for an arbitrary subset."
        )
    if INITIAL_STATE_INDICES != [0]:
        raise ValueError(
            "This all-product pilot publishes only fixed initial state 0; configured "
            f"INITIAL_STATE_INDICES={INITIAL_STATE_INDICES}."
        )


def _lerobot_source_info():
    spec = importlib.util.find_spec("lerobot")
    if spec is None or spec.origin is None:
        return {"path": None, "git_commit": None, "git_dirty": None}
    package_path = Path(spec.origin).resolve()
    repository = next(
        (parent for parent in package_path.parents if (parent / ".git").exists()),
        None,
    )
    if repository is None:
        return {"path": str(package_path.parent), "git_commit": None, "git_dirty": None}
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "path": str(repository),
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "git_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def _initial_state_hashes_by_task(task_indices):
    return {
        task_index: _official_state_zero_sha256(task_index)
        for task_index in task_indices
    }


@lru_cache(maxsize=len(ALL_TASK_INDICES))
def _official_state_zero_sha256(task_index):
    """Hash official state 0 without instantiating a MuJoCo environment."""
    return _initial_state_sha256s(task_index=task_index)[0]


def _initial_state_sha256s(task_index=TASK_INDEX):
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[SUITE_NAME]()
    states = np.asarray(suite.get_task_init_states(task_index))
    hashes = []
    for index in INITIAL_STATE_INDICES:
        value = np.ascontiguousarray(states[index])
        digest = hashlib.sha256()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
        hashes.append(digest.hexdigest())
    return hashes


def _rollout_step_metrics(eval_info, task_index=TASK_INDEX):
    if eval_info is None:
        return [], []
    task_result = _find_vla_task(eval_info, task_index)
    metrics = task_result.get("metrics", {})
    successes = list(metrics.get("successes", []))
    step_counts = []
    for video_path in metrics.get("video_paths", []):
        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Could not inspect rollout video: {video_path}")
            step_counts.append(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        finally:
            capture.release()
    if len(step_counts) != len(successes):
        raise RuntimeError(
            "LeRobot result has a different number of rollout videos and success values."
        )
    first_success_steps = [
        steps if success else None
        for steps, success in zip(step_counts, successes, strict=True)
    ]
    return step_counts, first_success_steps


if __name__ == "__main__":
    main()
