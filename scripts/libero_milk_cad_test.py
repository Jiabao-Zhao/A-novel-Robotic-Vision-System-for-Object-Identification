import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from scripts.libero_task_execution import camera_point_to_world, task_roles_from_vlm_result
from simulation.libero_cad import register_libero_cad_to_observation
from simulation.libero_experiment import PROPOSED_METHOD_FOLDER, episode_result_dir


EPISODE_ROOT = episode_result_dir(PROPOSED_METHOD_FOLDER, 7, 0)
LOCALIZATION_PATH = (
    EPISODE_ROOT / "perception" / "point_cloud" / "point_cloud_localization.json"
)
VLM_RESULT_PATH = EPISODE_ROOT / "vlm_result.json"
WORLD_T_CAMERA_PATH = EPISODE_ROOT / "perception" / "capture" / "world_T_camera.npy"
OUTPUT_DIR = EPISODE_ROOT / "cad_registration" / "milk"
ALIGNMENT_FIGURE_PATH = OUTPUT_DIR / "alignment_views.png"


def main():
    for path in (LOCALIZATION_PATH, VLM_RESULT_PATH, WORLD_T_CAMERA_PATH):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required perception artifact is missing: {path}. "
                "Run `python -m scripts.libero_task_execution` through its perception stage first."
            )

    localization = json.loads(LOCALIZATION_PATH.read_text(encoding="utf-8"))
    grounding = task_roles_from_vlm_result(VLM_RESULT_PATH)
    world_T_camera = np.load(WORLD_T_CAMERA_PATH)
    result = register_libero_cad_to_observation(
        object_type=grounding["milk_object_type"],
        object_id=grounding["milk_object_id"],
        localization=localization,
        world_T_camera=world_T_camera,
        output_dir=OUTPUT_DIR,
    )
    milk = next(
        item
        for item in localization["objects"]
        if item["object_id"] == grounding["milk_object_id"]
    )
    depth_center_world_m = camera_point_to_world(
        milk["centroid_3d_m"],
        world_T_camera,
    )
    save_alignment_figure(
        result["observed_cloud_path"],
        result["aligned_cad_cloud_path"],
        ALIGNMENT_FIGURE_PATH,
    )

    print(f"Milk localized object: {result['object_id']}")
    print(f"CAD: {result['cad_name']} ({result['evaluation_condition']})")
    print(f"CAD scale to metres: {result['cad_scale_to_m']}")
    print(f"Observed partial cloud: {result['observed_cloud_path']}")
    print(f"Depth centroid world XYZ (m): {np.round(depth_center_world_m, 6).tolist()}")
    print(
        "CAD-registered center world XYZ (m): "
        f"{np.round(result['registered_center_world_m'], 6).tolist()}"
    )
    print(f"Constrained registration RMSE (m): {result['constrained_rmse_m']:.6f}")
    print(f"Aligned CAD cloud: {result['aligned_cad_cloud_path']}")
    print(f"Augmented cloud: {result['augmented_cloud_path']}")
    print(f"Alignment figure: {ALIGNMENT_FIGURE_PATH}")
    print(f"Registration report: {result['result_path']}")
    print(
        "MuJoCo instance identity, segmentation, and ground-truth object pose "
        "were not used"
    )


def save_alignment_figure(observed_path, aligned_cad_path, output_path):
    observed = np.asarray(o3d.io.read_point_cloud(str(observed_path)).points)
    aligned_cad = np.asarray(o3d.io.read_point_cloud(str(aligned_cad_path)).points)
    if observed.size == 0 or aligned_cad.size == 0:
        raise RuntimeError("Cannot visualize an empty observed or aligned CAD cloud.")
    aligned_cad = aligned_cad[::max(1, len(aligned_cad) // 5000)]

    views = (
        (0, 2, "X", "Z", "front"),
        (1, 2, "Y", "Z", "side"),
        (0, 1, "X", "Y", "top"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, (
        horizontal,
        vertical,
        horizontal_name,
        vertical_name,
        title,
    ) in zip(axes, views):
        axis.scatter(
            aligned_cad[:, horizontal],
            aligned_cad[:, vertical],
            s=1,
            c="#2878d0",
            alpha=0.35,
            label="aligned CAD",
        )
        axis.scatter(
            observed[:, horizontal],
            observed[:, vertical],
            s=4,
            c="#d62728",
            alpha=0.8,
            label="observed depth",
        )
        axis.set_title(f"{title} view")
        axis.set_xlabel(f"{horizontal_name} camera (m)")
        axis.set_ylabel(f"{vertical_name} camera (m)")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
    axes[0].legend(loc="best")
    figure.suptitle("Milk CAD-to-observation alignment")
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
