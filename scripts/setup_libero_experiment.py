"""Create the two-method LIBERO-Object experiment result hierarchy."""

import json

from simulation.libero_experiment import (
    EXPERIMENT_NAME,
    EXPERIMENT_ROOT,
    LIBERO_OBJECT_TASKS,
    METHOD_FOLDERS,
    PROPOSED_METHOD_FOLDER,
    VLA_METHOD_FOLDER,
)


def main():
    for method_folder in METHOD_FOLDERS:
        for task_index, object_name, _ in LIBERO_OBJECT_TASKS:
            (EXPERIMENT_ROOT / method_folder / f"task_{task_index:02d}_{object_name}").mkdir(
                parents=True,
                exist_ok=True,
            )

    manifest = {
        "schema_version": 1,
        "experiment_name": EXPERIMENT_NAME,
        "suite": "libero_object",
        "objective": "Compare two methods for placing every LIBERO-Object target into the basket.",
        "status": "all-product state-0 breadth evaluation",
        "primary_metric": {
            "name": "binary_task_success_rate",
            "definition": (
                "Number of episodes satisfying LIBERO check_success() divided by "
                "ten completed task episodes"
            ),
            "speed_or_action_count_used": False,
        },
        "methods": {
            VLA_METHOD_FOLDER: {
                "description": "SmolVLA through the official LeRobot evaluator",
                "observation_resolution_hw": [256, 256],
                "inputs": [
                    "agent-view RGB",
                    "wrist RGB",
                    "8D robot state",
                    "language instruction",
                ],
            },
            PROPOSED_METHOD_FOLDER: {
                "description": "RGB-D/VLM/CAD perception with task-specific scripted control",
                "observation_resolution_hw": [512, 512],
                "inputs": [
                    "agent-view RGB",
                    "metric depth",
                    "camera calibration",
                    "robot proprioception",
                    "language instruction",
                    "CAD prior selected from VLM classification",
                ],
            },
        },
        "shared_protocol": {
            "task_indices": [task[0] for task in LIBERO_OBJECT_TASKS],
            "initial_state_indices_per_task": [0],
            "index_definition": (
                "task_index selects one of the ten product instructions; "
                "initial_state_index selects a fixed simulator state within that task"
            ),
            "episode_seed_per_task": 1000,
            "episode_horizon_steps": 280,
            "control_frequency_hz": 20,
            "control_mode": "relative",
            "settle_steps_before_policy": 10,
            "success_predicate": "LIBERO task environment check_success()",
        },
        "interpretation": (
            "The proposed branch is currently a 512 x 512 resolution ablation. The existing "
            "VLA baseline remains at its official 256 x 256 setting, so the two branches are "
            "not a matched method comparison. This sweep tests product breadth with one fixed "
            "state per task and does not measure within-task robustness."
        ),
        "task_catalog": [
            {
                "task_index": task_index,
                "object": object_name,
                "instruction": instruction,
                "available_fixed_initial_states": 50,
            }
            for task_index, object_name, instruction in LIBERO_OBJECT_TASKS
        ],
        "ground_truth_policy": (
            "Simulator instance identity, segmentation, and object pose are not method inputs. "
            "Ground truth may only be added later as clearly labeled evaluation-only data."
        ),
        "image_convention_note": (
            "SmolVLA retains LeRobot's checkpoint-specific LIBERO image orientation; the "
            "proposed RGB-D backend uses a conventional top-left-origin view. Any comparison "
            "montage may mirror the VLA row for display alignment only. Policy inputs and "
            "original rollout videos remain unchanged."
        ),
    }
    manifest_path = EXPERIMENT_ROOT / "experiment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Experiment root: {EXPERIMENT_ROOT}")
    print(f"Methods: {', '.join(METHOD_FOLDERS)}")
    print(f"Tasks prepared per method: {len(LIBERO_OBJECT_TASKS)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
