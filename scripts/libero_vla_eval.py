"""Run the official LeRobot SmolVLA evaluator for the matched milk task.

This is deliberately a separate controller branch. It does not import or use
the repository's depth, VLM, CAD-registration, or scripted-control modules.
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
from pathlib import Path

import cv2
import numpy as np

from simulation.libero_experiment import VLA_METHOD_FOLDER, episode_result_dir


CHECKPOINT = "HuggingFaceVLA/smolvla_libero"
CHECKPOINT_REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
SUITE_NAME = "libero_object"
TASK_INDEX = 7
TASK_INSTRUCTION = "pick up the milk and place it in the basket"
SEED = 1000
INITIAL_STATE_INDICES = [0]
EPISODE_HORIZON_STEPS = 280
CONTROL_FREQUENCY_HZ = 20
CONTROL_MODE = "relative"
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
OUTPUT_ROOT = episode_result_dir(VLA_METHOD_FOLDER, TASK_INDEX, INITIAL_STATE_INDICES[0])


def build_eval_command(output_root=OUTPUT_ROOT):
    _validate_initial_state_indices()
    output_root = Path(output_root).resolve()
    return [
        "lerobot-eval",
        f"--policy.path={CHECKPOINT}",
        f"--policy.pretrained_revision={CHECKPOINT_REVISION}",
        "--policy.device=cuda",
        "--env.type=libero",
        f"--env.task={SUITE_NAME}",
        f"--env.task_ids=[{TASK_INDEX}]",
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
        f"--eval.n_episodes={len(INITIAL_STATE_INDICES)}",
        f"--seed={SEED}",
        f"--output_dir={output_root}",
    ]


def main():
    executable = shutil.which("lerobot-eval")
    if executable is None:
        raise SystemExit(
            "lerobot-eval is unavailable. Activate ~/.venvs/lerobot-libero and "
            "install the LeRobot source checkout with `python -m pip install "
            "-e \".[libero,smolvla]\"`."
        )

    output_root = OUTPUT_ROOT.resolve()
    if output_root.exists():
        raise SystemExit(
            f"Refusing to overwrite the existing VLA evaluation: {output_root}. "
            "Move it aside or change OUTPUT_ROOT for a new run."
        )

    command = build_eval_command(output_root)
    environment = os.environ.copy()
    environment.setdefault("MUJOCO_GL", "egl")
    environment.setdefault("PYOPENGL_PLATFORM", environment["MUJOCO_GL"])

    print("VLA baseline: SmolVLA (independent from RGB-D/VLM/CAD branch)")
    print(f"Task: {SUITE_NAME}[{TASK_INDEX}] — {TASK_INSTRUCTION}")
    print(f"Official fixed initial states: {INITIAL_STATE_INDICES}")
    print("Command:")
    print("  " + " \\\n  ".join(command))

    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    result = subprocess.run(command, env=environment, check=False)
    elapsed_seconds = time.perf_counter() - started

    output_root.mkdir(parents=True, exist_ok=True)
    eval_info_path = output_root / "eval_info.json"
    eval_info = (
        json.loads(eval_info_path.read_text(encoding="utf-8"))
        if eval_info_path.is_file()
        else None
    )
    rollout_step_counts = []
    first_success_steps = []
    rollout_metrics_error = None
    if result.returncode == 0 and eval_info is not None:
        try:
            rollout_step_counts, first_success_steps = _rollout_step_metrics(eval_info)
        except Exception as error:
            rollout_metrics_error = str(error)
    initial_state_sha256s = []
    initial_state_hash_error = None
    try:
        initial_state_sha256s = _initial_state_sha256s()
    except Exception as error:
        initial_state_hash_error = str(error)
    lerobot_source = _lerobot_source_info()
    manifest = {
        "schema_version": 1,
        "method": "smolvla_official_lerobot_evaluator",
        "controller_branch": "vla_baseline",
        "checkpoint": CHECKPOINT,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_is_libero_finetuned": True,
        "local_finetuning_performed": False,
        "suite": SUITE_NAME,
        "task_index": TASK_INDEX,
        "instruction": TASK_INSTRUCTION,
        "success_predicate": "LIBERO task environment check_success()",
        "seed": SEED,
        "episode_seeds": [SEED + index for index in range(len(INITIAL_STATE_INDICES))],
        "initial_state_indices": INITIAL_STATE_INDICES,
        "initial_state_sha256s": initial_state_sha256s,
        "initial_state_hash_error": initial_state_hash_error,
        "initial_state_source": "official LIBERO task .pruned_init array",
        "episode_horizon_steps": EPISODE_HORIZON_STEPS,
        "control_frequency_hz": CONTROL_FREQUENCY_HZ,
        "control_mode": CONTROL_MODE,
        "observation_resolution_hw": [IMAGE_HEIGHT, IMAGE_WIDTH],
        "settle_steps_before_policy": 10,
        "method_inputs": [
            "rendered agentview RGB",
            "rendered wrist RGB",
            "8D robot state",
            "language instruction",
        ],
        "depth_or_cad_inputs_used": False,
        "command": command,
        "mujoco_gl": environment["MUJOCO_GL"],
        "lerobot_version": _package_version("lerobot"),
        "lerobot_source": lerobot_source,
        "torch_version": _package_version("torch"),
        "started_at_utc": started_at.isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "return_code": result.returncode,
        "rollout_step_counts": rollout_step_counts,
        "first_success_steps": first_success_steps,
        "rollout_step_count_source": (
            "MP4 frame count; pinned LeRobot evaluator renders one frame per action step"
        ),
        "rollout_metrics_error": rollout_metrics_error,
    }
    manifest_path = output_root / "experiment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if result.returncode != 0:
        raise SystemExit(
            f"LeRobot evaluation failed with exit code {result.returncode}. "
            f"Reproducibility manifest: {manifest_path}"
        )
    if not eval_info_path.is_file():
        raise SystemExit(
            f"LeRobot exited successfully but did not create {eval_info_path}."
        )
    if rollout_metrics_error is not None:
        raise SystemExit(
            "LeRobot evaluation completed, but rollout-step inspection failed: "
            f"{rollout_metrics_error}. Manifest: {manifest_path}"
        )
    if initial_state_hash_error is not None:
        raise SystemExit(
            "LeRobot evaluation completed, but fixed-state hashing failed: "
            f"{initial_state_hash_error}. Manifest: {manifest_path}"
        )
    print(f"Evaluation metrics: {eval_info_path}")
    print(f"Experiment manifest: {manifest_path}")


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


def _initial_state_sha256s():
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[SUITE_NAME]()
    states = np.asarray(suite.get_task_init_states(TASK_INDEX))
    hashes = []
    for index in INITIAL_STATE_INDICES:
        value = np.ascontiguousarray(states[index])
        digest = hashlib.sha256()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
        hashes.append(digest.hexdigest())
    return hashes


def _rollout_step_metrics(eval_info):
    if eval_info is None:
        return [], []
    matches = [
        item
        for item in eval_info.get("per_task", [])
        if item.get("task_group") == SUITE_NAME and item.get("task_id") == TASK_INDEX
    ]
    if len(matches) != 1:
        return [], []
    metrics = matches[0].get("metrics", {})
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
