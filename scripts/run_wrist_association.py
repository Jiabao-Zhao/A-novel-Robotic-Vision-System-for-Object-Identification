"""Six-object wrist RGB-D association experiment; no pickup or task execution.

Run in the existing WSL simulation environment: python -m scripts.run_wrist_association
Simulator identities are used only by the evaluator, after depth localization.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from simulation.wrist_cluster import CAMERA, SEED, make_environment, prepare_catalog


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs/simulation/wrist_association"
TARGETS = {
    "Gear_Large": "large white gear",
    "Gear_Medium": "medium white gear",
    "KET16_Square_16mm": "rectangular pin",
    "pulley": "pulley",
    "red_block": "red block",
    "blue_block": "blue block",
}
# Fixed, separated scene layout in simulator world metres. This is not input to matching.
PLACEMENTS = {
    "Gear_Large": {"xy_m": [-.18, -.14], "yaw_deg": 15.},
    "Gear_Medium": {"xy_m": [0., -.14], "yaw_deg": -20.},
    "KET16_Square_16mm": {"xy_m": [-.18, 0.], "yaw_deg": 30.},
    "pulley": {"xy_m": [0., 0.], "yaw_deg": 5.},
    "red_block": {"xy_m": [-.18, .14], "yaw_deg": 15.},
    "blue_block": {"xy_m": [0., .14], "yaw_deg": -15.},
}
WORKSPACE_MIN = (-.32, -.27, -.01)
WORKSPACE_MAX = (.12, .27, .17)


def configure_scene(model):
    """Remove the fixed assembly board and lay the existing prismatic pin flat."""
    from scipy.spatial.transform import Rotation

    for body in list(model.worldbody):
        name = body.get("name", "")
        if name == "nist_board" or name.startswith("nist_fixture_"):
            model.worldbody.remove(body)
    name = "KET16_Square_16mm"
    body = model.worldbody.find(f"./body[@name='nist_part_{name}']")
    rotation = (Rotation.from_euler("z", PLACEMENTS[name]["yaw_deg"], degrees=True)
                * Rotation.from_euler("y", 90, degrees=True))
    body.set("quat", " ".join(map(str, rotation.as_quat()[[3, 0, 1, 2]])))
    body.set("pos", " ".join(map(str, [*PLACEMENTS[name]["xy_m"], .009])))


def evaluate_rankings(associations, identity_audit):
    """Compare independent predictions with audited identities, including misses."""
    by_id = {item["object_id"]: item["simulator_instance"] for item in identity_audit
             if item["status"] == "matched"}
    by_instance = {instance: object_id for object_id, instance in by_id.items()}
    rows = []
    for result in associations:
        expected = by_instance.get(result["cad_id"])
        predictions = result["predictions"]
        rows.append({
            "target_description": result["target_description"], "cad_id": result["cad_id"],
            "expected_object_id": expected, "localized": expected is not None,
            "predictions": predictions,
            "predicted_instances": {mode: by_id.get(object_id) for mode, object_id in predictions.items()},
            "correct": {mode: expected is not None and object_id == expected
                        for mode, object_id in predictions.items()},
        })
    return {
        "target_count": len(rows), "localized_target_count": sum(row["localized"] for row in rows),
        "accuracy": {mode: {"correct": sum(row["correct"][mode] for row in rows),
                            "total": len(rows),
                            "fraction": sum(row["correct"][mode] for row in rows) / len(rows) if rows else None}
                     for mode in ("visual", "geometry", "fused")},
        "per_target": rows,
        "ground_truth_policy": "Separate simulator XY identity audit; never passed to association.",
        "protocol": "One fixed scene, known CAD IDs, independent top-1 rankings; no confidence thresholds.",
        "limitations": [
            "One clean simulated scene is a smoke test, not a benchmark accuracy estimate.",
            "Red and blue blocks share geometry; only their supplied CAD material colors distinguish them.",
            "Visual and geometric scores are uncalibrated; 0.5/0.5 fusion is an experimental baseline.",
            "Pulley and blocks are representative procedural assets, not measured physical parts.",
        ],
    }


def save_identity_image(rgb_path, identity_audit, output_path):
    """Evaluation-only image: persistent candidate IDs and audited object names."""
    image = cv2.imread(str(rgb_path))
    for item in identity_audit:
        x1, y1, x2, y2 = (item["roi"][key] for key in ("x1", "y1", "x2", "y2"))
        name = TARGETS.get(item["simulator_instance"], "unresolved")
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 2)
        for row, label in enumerate((item["object_id"], name)):
            width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .42, 1)[0][0]
            x = max(0, min((x1 + x2 - width) // 2, image.shape[1] - width - 4))
            y = min(image.shape[0] - 4, y2 + 15 + row * 16)
            cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 0, 0), 3)
            cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Cannot save identity audit image: {output_path}")


def capture_scene(output_dir):
    from scripts.run_wrist_cluster import prepare_observation, identity_audit
    from simulation.libero_io import save_libero_observation
    from simulation.libero_sensor import LiberoRGBDSensor
    from simulation.nist_peg_task import localize_parts

    catalog = {name: record for name, record in prepare_catalog().items() if name in TARGETS}
    for name in ("Gear_Large", "Gear_Medium"):
        catalog[name]["visual_rgba"] = [.95, .95, .95, 1.]
    library_path = output_dir / "cad_library.json"
    library = [{"cad_id": name, "cad_name": description, "description": description,
                "file_path": catalog[name]["cad_path"], "category": "simulation assembly component",
                "base_color_rgb": catalog[name]["visual_rgba"][:3]}
               for name, description in TARGETS.items()]
    library_path.write_text(json.dumps(library, indent=2), encoding="utf-8")
    with make_environment(catalog, PLACEMENTS, additional_scene=configure_scene) as environment:
        raw = prepare_observation(environment)
        observation = LiberoRGBDSensor(environment, CAMERA).capture(raw)
        capture = save_libero_observation(observation, environment, output_dir / "capture")
        np.save(output_dir / "initial_state.npy", environment.sim.get_state().flatten())
        localization, paths = localize_parts(observation, capture["rgb"], output_dir,
            workspace_min=WORKSPACE_MIN, workspace_max=WORKSPACE_MAX, cluster_in_table_plane=True)
        audit = identity_audit(environment, localization, observation)
    (output_dir / "identity_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    save_identity_image(capture["rgb"], audit, output_dir / "wrist_identities.png")
    manifest = {"seed": SEED, "camera_inputs": [CAMERA], "external_camera_enabled": False,
                "targets": TARGETS, "placements": PLACEMENTS, "catalog": catalog,
                "workspace_min_m": WORKSPACE_MIN, "workspace_max_m": WORKSPACE_MAX,
                "rectangular_pin": "Existing 16 x 10 x 50 mm rectangular prismatic pin, lying flat.",
                "capture": {key: str(path) for key, path in capture.items()},
                "localization_path": str(paths["localization"]), "cad_library_path": str(library_path),
                "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in (Path(__file__), ROOT / "cad_object_association.py",
                                               ROOT / "CADPointCloudRegistration.py", ROOT / "main.py",
                                               ROOT / "helper_function.py",
                                               ROOT / "simulation/wrist_cluster.py",
                                               ROOT / "simulation/nist_peg_task.py")}}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Captured {len(localization['objects'])} candidates for six objects: {output_dir}", flush=True)
    return manifest


def run_association(output_dir):
    """Replay a saved scene without starting MuJoCo; use the unchanged matcher."""
    import torch
    from main import associate_targets_with_cad
    from cad_object_association import load_dino_encoder

    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    targets = manifest["targets"]
    localization = json.loads(Path(manifest["localization_path"]).read_text(encoding="utf-8"))
    start = perf_counter()
    associations, _, _ = associate_targets_with_cad(
        list(targets.values()), localization, manifest["capture"]["rgb"],
        localization.get("plane_model"), output_dir / "association",
        known_cad_ids={description: name for name, description in targets.items()},
        library_path=manifest["cad_library_path"],
    )
    elapsed = perf_counter() - start
    audit = json.loads((output_dir / "identity_audit.json").read_text(encoding="utf-8"))
    evaluation = evaluate_rankings(associations, audit)
    evaluation["association_set_resolution"] = associations[0]["association_set_resolution"]
    evaluation["conflicts"] = associations[0]["conflicts"]
    evaluation["runtime"] = {"association_set_wall_time_s": elapsed,
        **{key: sum(result["runtime"][key] for result in associations) for key in (
            "cad_visual_preprocessing_time_s", "cad_geometry_preprocessing_time_s",
            "cad_cache_load_time_s", "workspace_dino_feature_time_s", "workspace_fpfh_feature_time_s",
            "pairwise_matching_time_s", "fusion_time_s", "scene_association_time_s")},
        "device": str(next(load_dino_encoder().parameters()).device), "torch_version": torch.__version__,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "candidate_pair_count": sum(len(result["candidate_ranking"]) for result in associations),
        "cad_visual_cache_hits": sum(result["runtime"]["cad_visual_cache_hit"] for result in associations),
        "cad_geometry_cache_hits": sum(result["runtime"]["cad_geometry_cache_hit"] for result in associations)}
    (output_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"accuracy": evaluation["accuracy"], "resolution": evaluation["association_set_resolution"],
                      "runtime": evaluation["runtime"]}, indent=2), flush=True)
    return evaluation


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = OUTPUT_DIR / stamp
    output_dir.mkdir(parents=True)
    capture_scene(output_dir)
    run_association(output_dir)
    print(f"Saved wrist association experiment: {output_dir}", flush=True)
    return output_dir


if __name__ == "__main__":
    main()
