"""Capture all 500 fixed LIBERO-Object scenes for VLM score evaluation."""

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from simulation.libero_control import OPEN_GRIPPER, hold_gripper
from simulation.libero_env import LiberoTaskEnvironment
from simulation.libero_experiment import EXPERIMENT_ROOT, LIBERO_OBJECT_TASKS
from simulation.libero_sensor import LiberoRGBDSensor
from simulation.perception_adapter import run_libero_localization
from vlm_module import create_roi_contact_sheet


SUITE_NAME = "libero_object"
CAMERA_NAME = "agentview"
IMAGE_WIDTH = 768
IMAGE_HEIGHT = 768
VLM_CONTACT_SHEET_TILE_SIZE_PX = 448
INITIAL_STATE_INDICES = tuple(range(50))
RANDOM_SEED = 1000
INITIAL_PHYSICS_SETTLE_STEPS = 10
MAX_TARGET_XY_MATCH_DISTANCE_M = 0.075
OUTPUT_ROOT = EXPERIMENT_ROOT / "openai_vlm_confidence" / "candidate_normalized_500"


def dataset_partition(initial_state_index):
    index = int(initial_state_index)
    if not 0 <= index < 50:
        raise ValueError("LIBERO-Object initial-state index must be between 0 and 49.")
    if index < 30:
        return "calibration"
    if index < 40:
        return "validation"
    return "test"


def match_target_to_localized_candidate(
    localization,
    world_T_camera,
    target_position_world_m,
    maximum_xy_distance_m=MAX_TARGET_XY_MATCH_DISTANCE_M,
):
    """Match simulator target pose to a depth cluster for evaluation only."""
    target_position = np.asarray(target_position_world_m, dtype=float)
    if target_position.shape != (3,) or not np.all(np.isfinite(target_position)):
        raise ValueError("Target world position must be a finite XYZ vector.")

    distances = []
    transform = np.asarray(world_T_camera, dtype=float)
    for item in localization.get("objects", []):
        centroid_camera = np.asarray([*item["centroid_3d_m"], 1.0], dtype=float)
        centroid_world = (transform @ centroid_camera)[:3]
        distances.append(
            {
                "object_id": str(item["object_id"]),
                "centroid_world_m": centroid_world.tolist(),
                "xy_distance_m": float(
                    np.linalg.norm(centroid_world[:2] - target_position[:2])
                ),
            }
        )
    distances.sort(key=lambda item: (item["xy_distance_m"], item["object_id"]))
    if not distances:
        return None, {
            "target_localized": False,
            "nearest_candidate_id": None,
            "nearest_xy_distance_m": None,
            "second_nearest_xy_distance_m": None,
        }

    nearest = distances[0]
    target_localized = nearest["xy_distance_m"] <= float(maximum_xy_distance_m)
    return (
        nearest["object_id"] if target_localized else None,
        {
            "target_localized": target_localized,
            "nearest_candidate_id": nearest["object_id"],
            "nearest_xy_distance_m": nearest["xy_distance_m"],
            "second_nearest_xy_distance_m": (
                distances[1]["xy_distance_m"] if len(distances) > 1 else None
            ),
            "candidate_distances": distances,
        },
    )


def compact_localization(localization, rgb_path):
    """Keep VLM inputs while dropping temporary point-cloud artifact paths."""
    return {
        "schema_version": 1,
        "frame": localization.get("frame", "camera"),
        "rgb_path": str(rgb_path),
        "camera_intrinsics": localization.get("camera_intrinsics"),
        "object_count": int(localization.get("object_count", 0)),
        "plane_model": localization.get("plane_model"),
        "objects": [
            {
                "object_id": item["object_id"],
                "roi": item["roi"],
                "centroid_3d_m": item["centroid_3d_m"],
                "size_3d_m": item["size_3d_m"],
                "point_count": int(item["point_count"]),
            }
            for item in localization.get("objects", [])
        ],
    }


def capture_dataset(
    task_catalog=LIBERO_OBJECT_TASKS,
    initial_state_indices=INITIAL_STATE_INDICES,
    output_root=OUTPUT_ROOT,
):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    samples = []
    reused_count = 0
    captured_count = 0

    for task_index, target_slug, expected_instruction in task_catalog:
        target_description = target_slug.replace("_", " ")
        task_root = output_root / "scenes" / f"task_{task_index:02d}_{target_slug}"
        with LiberoTaskEnvironment(
            suite_name=SUITE_NAME,
            task_index=task_index,
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT,
        ) as environment:
            if environment.initial_state_count != 50:
                raise RuntimeError(
                    f"Expected 50 fixed states for task {task_index}; received "
                    f"{environment.initial_state_count}."
                )
            sensor = LiberoRGBDSensor(environment, CAMERA_NAME)
            for initial_state_index in initial_state_indices:
                scene_dir = task_root / f"init_state_{initial_state_index:02d}"
                sample_path = scene_dir / "sample.json"
                if sample_path.is_file():
                    sample = json.loads(sample_path.read_text(encoding="utf-8"))
                    if _sample_files_exist(sample, output_root):
                        samples.append(sample)
                        reused_count += 1
                        continue

                scene_dir.mkdir(parents=True, exist_ok=True)
                raw = environment.reset(
                    seed=RANDOM_SEED,
                    init_state_index=initial_state_index,
                )
                raw = hold_gripper(
                    environment,
                    raw,
                    OPEN_GRIPPER,
                    "confidence_dataset_settle",
                    INITIAL_PHYSICS_SETTLE_STEPS,
                )
                observation = sensor.capture(raw)
                if observation.instruction != expected_instruction:
                    raise RuntimeError(
                        f"Instruction mismatch for task {task_index}: "
                        f"{observation.instruction!r}."
                    )

                rgb_path = scene_dir / "rgb.png"
                if not cv2.imwrite(
                    str(rgb_path),
                    cv2.cvtColor(observation.rgb, cv2.COLOR_RGB2BGR),
                ):
                    raise RuntimeError(f"Could not save LIBERO RGB image: {rgb_path}")

                with tempfile.TemporaryDirectory() as temporary_directory:
                    localization, paths, _ = run_libero_localization(
                        observation,
                        rgb_path=rgb_path,
                        output_root=Path(temporary_directory) / "perception",
                    )
                    contact_sheet_path = scene_dir / "vlm_roi_contact_sheet.png"
                    create_roi_contact_sheet(
                        rgb_path,
                        paths["localization"],
                        contact_sheet_path,
                        tile_size_px=VLM_CONTACT_SHEET_TILE_SIZE_PX,
                    )

                localization_path = scene_dir / "localization.json"
                localization_path.write_text(
                    json.dumps(compact_localization(localization, rgb_path), indent=2),
                    encoding="utf-8",
                )
                target_instance, target_position_world_m = _target_state(environment)
                ground_truth_object_id, match = match_target_to_localized_candidate(
                    localization,
                    observation.world_T_camera,
                    target_position_world_m,
                )
                ground_truth_path = scene_dir / "evaluation_ground_truth.json"
                ground_truth_path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "evaluation_only": True,
                            "excluded_from_vlm_prompt": True,
                            "matching_method": (
                                "nearest depth-localized cluster centroid in MuJoCo "
                                "world XY to the target root-body position"
                            ),
                            "maximum_match_distance_m": MAX_TARGET_XY_MATCH_DISTANCE_M,
                            "simulator_target_instance": target_instance,
                            "simulator_target_position_world_m": (
                                target_position_world_m.tolist()
                            ),
                            "ground_truth_object_id": ground_truth_object_id,
                            **match,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                sample = {
                    "sample_id": (
                        f"task_{task_index:02d}_{target_slug}_state_"
                        f"{initial_state_index:02d}"
                    ),
                    "partition": dataset_partition(initial_state_index),
                    "task_index": int(task_index),
                    "initial_state_index": int(initial_state_index),
                    "initial_state_sha256": environment.last_init_state_sha256,
                    "image_path": str(contact_sheet_path.relative_to(output_root)),
                    "localization_path": str(localization_path.relative_to(output_root)),
                    "target_description": target_description,
                    "ground_truth_object_id": ground_truth_object_id,
                    "target_localized": bool(match["target_localized"]),
                    "evaluation_ground_truth_path": str(
                        ground_truth_path.relative_to(output_root)
                    ),
                }
                sample_path.write_text(json.dumps(sample, indent=2), encoding="utf-8")
                samples.append(sample)
                captured_count += 1
                print(
                    f"Captured {len(samples):03d}/500: {sample['sample_id']} "
                    f"candidates={localization['object_count']} "
                    f"ground_truth={ground_truth_object_id} "
                    f"match_xy_m={match['nearest_xy_distance_m']}"
                )

    samples.sort(key=lambda item: (item["task_index"], item["initial_state_index"]))
    manifest = {
        "schema_version": 1,
        "split": "libero_object_candidate_normalized_500",
        "suite": SUITE_NAME,
        "camera": CAMERA_NAME,
        "resolution_hw": [IMAGE_HEIGHT, IMAGE_WIDTH],
        "sample_count": len(samples),
        "partition_rule": {
            "calibration": "initial states 0-29",
            "validation": "initial states 30-39",
            "test": "initial states 40-49",
        },
        "ground_truth_usage": (
            "Simulator target pose is used only after RGB-D localization to assign "
            "the evaluation label; it is never included in the VLM prompt."
        ),
        "samples": samples,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"New captures: {captured_count}")
    print(f"Reused captures: {reused_count}")
    print(f"Dataset manifest: {manifest_path}")
    if len(samples) != len(task_catalog) * len(initial_state_indices):
        raise RuntimeError("The LIBERO confidence dataset is incomplete.")
    return manifest_path


def _target_state(environment):
    inner_environment = environment.env.env
    target_instance = str(inner_environment.obj_of_interest[0])
    target_object = next(
        (item for item in inner_environment.objects if item.name == target_instance),
        None,
    )
    if target_object is None:
        raise RuntimeError(f"Could not find LIBERO target instance {target_instance!r}.")
    body_id = environment.sim.model.body_name2id(target_object.root_body)
    position = np.asarray(environment.sim.data.body_xpos[body_id], dtype=float).copy()
    return target_instance, position


def _sample_files_exist(sample, output_root):
    return all(
        (output_root / sample[key]).is_file()
        for key in (
            "image_path",
            "localization_path",
            "evaluation_ground_truth_path",
        )
    )


if __name__ == "__main__":
    capture_dataset()
