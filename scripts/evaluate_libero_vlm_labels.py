"""Audit saved LIBERO VLM labels against simulator ground truth.

This script is evaluation-only. Simulator object identities and poses are loaded
after the proposed method has produced its saved results; they are never exposed
to the VLM, CAD registration, or controller.
"""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from simulation.libero_cad import resolve_libero_cad_record
from simulation.libero_control import OPEN_GRIPPER, hold_gripper
from simulation.libero_env import LiberoTaskEnvironment
from simulation.libero_experiment import (
    PROPOSED_METHOD_FOLDER,
    episode_result_dir,
    libero_object_task,
)


TASK_INDICES = tuple(range(10))
INITIAL_STATE_INDEX = 0
RANDOM_SEED = 1000
SETTLE_STEPS = 10
MAXIMUM_MATCH_XY_DISTANCE_M = 0.06
OUTPUT_ROOT = Path("outputs/simulation/experiments/put_all_objects_into_basket")
REPORT_JSON = OUTPUT_ROOT / "vlm_label_audit.json"
REPORT_CSV = OUTPUT_ROOT / "vlm_label_audit.csv"


def main():
    tasks = [_audit_task(task_index) for task_index in TASK_INDICES]
    report = {
        "schema_version": 1,
        "evaluation_only": True,
        "ground_truth_used_by_method": False,
        "ground_truth_usage": (
            "Post-hoc nearest-position association between saved depth-cluster "
            "centroids and LIBERO object body positions at the same fixed state."
        ),
        "grounding_error_definition": (
            "A candidate is incorrect when its target/reference role or positive/"
            "negative target match disagrees with the task target and basket ground truth."
        ),
        "exact_type_error_definition": (
            "A predicted_type is incorrect when it does not resolve to the exact "
            "LIBERO product class. This is diagnostic because the current prompt "
            "allows null or coarse labels for non-target distractors."
        ),
        "tasks": tasks,
        "totals": _totals(tasks),
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_csv(tasks)
    _print_summary(tasks, report["totals"])


def _audit_task(task_index):
    task_index, target_slug, instruction = libero_object_task(task_index)
    result_dir = episode_result_dir(
        PROPOSED_METHOD_FOLDER,
        task_index,
        INITIAL_STATE_INDEX,
    )
    localization_path = result_dir / "perception/point_cloud/point_cloud_localization.json"
    transform_path = result_dir / "perception/capture/world_T_camera.npy"
    vlm_path = result_dir / "vlm_result.json"
    for path in (localization_path, transform_path, vlm_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required saved trial artifact is missing: {path}")

    localization = json.loads(localization_path.read_text(encoding="utf-8"))
    vlm_result = json.loads(vlm_path.read_text(encoding="utf-8"))
    evaluations = vlm_result.get("normalized_result", {}).get("object_evaluations", [])
    evaluations_by_id = {item.get("object_id"): item for item in evaluations}
    candidates = localization.get("objects", [])
    candidate_ids = [item.get("object_id") for item in candidates]
    if len(candidates) != 7 or len(set(candidate_ids)) != len(candidates):
        raise RuntimeError(
            f"Expected seven unique localized candidates for task {task_index}; "
            f"received {candidate_ids}."
        )
    if set(evaluations_by_id) != set(candidate_ids):
        raise RuntimeError(
            f"VLM/localization object IDs differ for task {task_index}: "
            f"{sorted(evaluations_by_id)} vs {sorted(candidate_ids)}."
        )

    world_T_camera = np.load(transform_path)
    candidate_world_xyz = np.asarray(
        [_camera_point_to_world(item["centroid_3d_m"], world_T_camera) for item in candidates]
    )
    ground_truth, state_sha256 = _load_ground_truth(task_index, instruction)
    matches = match_candidates_to_ground_truth(
        candidate_world_xyz,
        np.asarray([item["position_world_m"] for item in ground_truth]),
    )

    rows = []
    for candidate_index, ground_truth_index, xy_distance_m in matches:
        candidate = candidates[candidate_index]
        truth = ground_truth[ground_truth_index]
        evaluation = evaluations_by_id[candidate["object_id"]]
        if xy_distance_m > MAXIMUM_MATCH_XY_DISTANCE_M:
            raise RuntimeError(
                f"Task {task_index} candidate {candidate['object_id']} is "
                f"{xy_distance_m:.4f} m from its nearest one-to-one ground-truth "
                "assignment; refusing an unreliable label audit."
            )
        row = evaluate_candidate(
            task_index=task_index,
            target_type=target_slug,
            candidate=candidate,
            evaluation=evaluation,
            ground_truth=truth,
            xy_distance_m=xy_distance_m,
        )
        rows.append(row)
    rows.sort(key=lambda item: item["object_id"])

    grounding_error_count = sum(not item["grounding_correct"] for item in rows)
    exact_type_error_count = sum(not item["exact_type_correct"] for item in rows)
    false_positive_count = sum(
        item["expected_instruction_role"] is None and not item["grounding_correct"]
        for item in rows
    )
    false_negative_count = sum(
        item["expected_instruction_role"] is not None and not item["grounding_correct"]
        for item in rows
    )
    return {
        "task_index": task_index,
        "target_object": target_slug,
        "instruction": instruction,
        "initial_state_index": INITIAL_STATE_INDEX,
        "initial_state_sha256": state_sha256,
        "rgb_path": str(result_dir / "perception/capture/rgb.png"),
        "vlm_contact_sheet_path": str(result_dir / "vlm_roi_contact_sheet.png"),
        "rgb_resolution_hw": [
            int(localization["camera_intrinsics"]["height"]),
            int(localization["camera_intrinsics"]["width"]),
        ],
        "candidate_count": len(rows),
        "grounding_error_count": grounding_error_count,
        "grounding_correct_count": len(rows) - grounding_error_count,
        "false_positive_count": false_positive_count,
        "false_negative_count": false_negative_count,
        "exact_type_error_count": exact_type_error_count,
        "exact_type_correct_count": len(rows) - exact_type_error_count,
        "task_grounding_fully_correct": grounding_error_count == 0,
        "maximum_match_xy_distance_m": max(item["match_xy_distance_m"] for item in rows),
        "objects": rows,
    }


def _load_ground_truth(task_index, expected_instruction):
    with LiberoTaskEnvironment(task_index=task_index) as environment:
        observation = environment.reset(
            seed=RANDOM_SEED,
            init_state_index=INITIAL_STATE_INDEX,
        )
        observation = hold_gripper(
            environment,
            observation,
            OPEN_GRIPPER,
            "evaluation_only_settle",
            SETTLE_STEPS,
        )
        if environment.instruction != expected_instruction:
            raise RuntimeError(
                f"Task {task_index} instruction mismatch: "
                f"{environment.instruction!r} != {expected_instruction!r}."
            )
        inner_environment = environment.env.env
        objects = []
        for object_name, object_model in inner_environment.objects_dict.items():
            body_id = inner_environment.obj_body_id[object_name]
            objects.append(
                {
                    "instance_name": object_name,
                    "category": str(object_model.category_name),
                    "position_world_m": [
                        float(value) for value in environment.sim.data.body_xpos[body_id]
                    ],
                }
            )
        if len(objects) != 7:
            raise RuntimeError(
                f"Expected seven LIBERO objects for task {task_index}; received "
                f"{[item['instance_name'] for item in objects]}."
            )
        return objects, environment.last_init_state_sha256


def match_candidates_to_ground_truth(candidate_world_xyz, ground_truth_world_xyz):
    candidates = np.asarray(candidate_world_xyz, dtype=float)
    truth = np.asarray(ground_truth_world_xyz, dtype=float)
    if candidates.ndim != 2 or truth.ndim != 2 or candidates.shape[1] != 3 or truth.shape[1] != 3:
        raise ValueError("Candidate and ground-truth positions must have shape (N, 3).")
    if len(candidates) != len(truth) or len(candidates) == 0:
        raise ValueError("Candidate and ground-truth positions must be equally nonempty.")
    if not np.all(np.isfinite(candidates)) or not np.all(np.isfinite(truth)):
        raise ValueError("Candidate and ground-truth positions must be finite.")
    cost = np.linalg.norm(candidates[:, None, :2] - truth[None, :, :2], axis=2)
    candidate_indices, truth_indices = linear_sum_assignment(cost)
    return [
        (int(candidate_index), int(truth_index), float(cost[candidate_index, truth_index]))
        for candidate_index, truth_index in zip(candidate_indices, truth_indices)
    ]


def evaluate_candidate(
    task_index,
    target_type,
    candidate,
    evaluation,
    ground_truth,
    xy_distance_m,
):
    ground_truth_type = ground_truth["category"]
    expected_role = (
        "reference_object"
        if ground_truth_type == "basket"
        else "moved_object"
        if ground_truth_type == target_type
        else None
    )
    expected_positive = expected_role is not None
    predicted_role = evaluation.get("instruction_role")
    target_match = evaluation.get("target_match")
    predicted_positive = target_match in {"match", "plausible_match"}
    grounding_correct = (
        predicted_positive == expected_positive
        and predicted_role == expected_role
    )
    predicted_type = evaluation.get("predicted_type")
    return {
        "task_index": int(task_index),
        "object_id": candidate["object_id"],
        "ground_truth_instance": ground_truth["instance_name"],
        "ground_truth_type": ground_truth_type,
        "predicted_type": predicted_type,
        "expected_instruction_role": expected_role,
        "predicted_instruction_role": predicted_role,
        "target_match": target_match,
        "grounding_correct": bool(grounding_correct),
        "exact_type_correct": exact_type_matches(predicted_type, ground_truth_type),
        "match_xy_distance_m": float(xy_distance_m),
        "bbox_2d_xyxy": evaluation.get("bbox_2d_xyxy"),
    }


def exact_type_matches(predicted_type, ground_truth_type):
    if not isinstance(predicted_type, str) or not predicted_type.strip():
        return False
    if ground_truth_type == "basket":
        return predicted_type.strip().lower() in {"basket", "woven basket"}
    try:
        predicted_record = resolve_libero_cad_record(predicted_type)
        truth_record = resolve_libero_cad_record(ground_truth_type)
    except LookupError:
        return False
    return predicted_record["cad_id"] == truth_record["cad_id"]


def _camera_point_to_world(point_camera_m, world_T_camera):
    point = np.asarray(point_camera_m, dtype=float)
    transform = np.asarray(world_T_camera, dtype=float)
    if point.shape != (3,) or transform.shape != (4, 4):
        raise ValueError("Expected a 3D camera point and a 4x4 world_T_camera transform.")
    return (transform @ np.append(point, 1.0))[:3]


def _totals(tasks):
    candidate_count = sum(item["candidate_count"] for item in tasks)
    grounding_errors = sum(item["grounding_error_count"] for item in tasks)
    exact_type_errors = sum(item["exact_type_error_count"] for item in tasks)
    return {
        "image_count": len(tasks),
        "candidate_count": candidate_count,
        "grounding_error_count": grounding_errors,
        "grounding_accuracy": (candidate_count - grounding_errors) / candidate_count,
        "false_positive_count": sum(item["false_positive_count"] for item in tasks),
        "false_negative_count": sum(item["false_negative_count"] for item in tasks),
        "exact_type_error_count": exact_type_errors,
        "exact_type_accuracy": (candidate_count - exact_type_errors) / candidate_count,
        "fully_correct_grounding_image_count": sum(
            item["task_grounding_fully_correct"] for item in tasks
        ),
    }


def _write_csv(tasks):
    fieldnames = [
        "task_index",
        "target_object",
        "object_id",
        "ground_truth_type",
        "predicted_type",
        "expected_instruction_role",
        "predicted_instruction_role",
        "target_match",
        "grounding_correct",
        "exact_type_correct",
        "match_xy_distance_m",
    ]
    with REPORT_CSV.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for task in tasks:
            for item in task["objects"]:
                writer.writerow(
                    {
                        name: task["target_object"] if name == "target_object" else item[name]
                        for name in fieldnames
                    }
                )


def _print_summary(tasks, totals):
    print("LIBERO VLM label audit (evaluation-only ground truth)")
    for task in tasks:
        print(
            f"  {task['task_index']:02d} {task['target_object']}: "
            f"grounding errors {task['grounding_error_count']}/{task['candidate_count']}, "
            f"FP {task['false_positive_count']}, FN {task['false_negative_count']}; "
            f"exact-name errors {task['exact_type_error_count']}/{task['candidate_count']}, "
            f"max match distance {task['maximum_match_xy_distance_m']:.4f} m"
        )
    print(
        "Grounding: "
        f"{totals['candidate_count'] - totals['grounding_error_count']}/"
        f"{totals['candidate_count']} correct "
        f"({100.0 * totals['grounding_accuracy']:.1f}%)"
    )
    print(
        "Exact product names (diagnostic): "
        f"{totals['candidate_count'] - totals['exact_type_error_count']}/"
        f"{totals['candidate_count']} correct "
        f"({100.0 * totals['exact_type_accuracy']:.1f}%)"
    )
    print(f"JSON: {REPORT_JSON}")
    print(f"CSV: {REPORT_CSV}")


if __name__ == "__main__":
    main()
