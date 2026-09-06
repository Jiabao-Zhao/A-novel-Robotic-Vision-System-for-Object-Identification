"""Evaluation-only identity audit of frozen LIBERO candidates; no VLM calls.

Replay verifies the frozen state, target position and RGB pixels. Localization,
images and predictions are never rewritten. Simulator identities belong ONLY
in this audit directory, never in perception inputs or association prompts.
"""

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path("outputs/simulation/experiments/put_all_objects_into_basket/openai_vlm_confidence")
SCENES = ROOT / "candidate_normalized_500"
AUDIT = ROOT / "candidate_identity_audit"
FROZEN = Path("experiments/put_all_objects_into_basket/proposed_framework/vlm_confidence_500")
MAX_DISTANCE_M = 0.075  # Same geometric association bound as the frozen target labels.
MIN_SEPARATION_M = 0.01  # Conservative audit ambiguity flag, not a VLM threshold.
PILOT_TASKS = (0, 5)
PILOT_STATES = (0, 1, 15)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def match_candidates(localization, world_T_camera, simulator_objects):
    """Accept only separated, mutual-nearest XY pairs; never force assignment.

    This checks cluster-centroid identity, not per-point cluster purity. Multiple
    clusters choosing one body (possible split) or near-ties remain unresolved.
    """
    candidates = localization["objects"]
    ids = [item["object_id"] for item in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate persistent candidate ID.")
    names = [item["instance"] for item in simulator_objects]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate simulator instance.")
    if not candidates or not simulator_objects:
        raise ValueError("Audit needs localized candidates and simulator objects.")
    camera_points = np.array([item["centroid_3d_m"] for item in candidates])
    transform = np.asarray(world_T_camera)
    world_points = camera_points @ transform[:3, :3].T + transform[:3, 3]
    positions = np.array([item["position_world_m"] for item in simulator_objects])
    distances = np.linalg.norm(world_points[:, None, :2] - positions[None, :, :2], axis=2)
    nearest = distances.argmin(axis=1)
    counts = Counter(nearest.tolist())
    result = []
    for i, item in enumerate(candidates):
        j = int(nearest[i])
        ranked = np.sort(distances[i])
        gap = float(ranked[1] - ranked[0]) if len(ranked) > 1 else None
        reverse = np.sort(distances[:, j])
        reverse_gap = float(reverse[1] - reverse[0]) if len(reverse) > 1 else None
        status = "matched"
        if ranked[0] > MAX_DISTANCE_M:
            status = "unmatched_distance"
        elif counts[j] > 1 or distances[:, j].argmin() != i:
            status = "ambiguous_possible_split"
        elif (gap is not None and gap < MIN_SEPARATION_M) or (reverse_gap is not None and reverse_gap < MIN_SEPARATION_M):
            status = "ambiguous_near_tie"
        obj = simulator_objects[j]
        result.append({
            "object_id": item["object_id"], "roi": item["roi"],
            "semantic_identity": obj["semantic_identity"] if status == "matched" else None,
            "simulator_instance": obj["instance"] if status == "matched" else None,
            "status": status, "centroid_world_m": world_points[i].tolist(),
            "nearest_simulator_instance": obj["instance"],
            "nearest_xy_distance_m": float(ranked[0]),
            "next_object_distance_gap_m": gap,
            "next_cluster_distance_gap_m": reverse_gap,
        })
    return result


def source_files(sample):
    scene = (SCENES / sample["localization_path"]).parent
    return {"rgb": scene / "rgb.png", "localization": scene / "localization.json",
            "contact_sheet": SCENES / sample["image_path"],
            "target_ground_truth": SCENES / sample["evaluation_ground_truth_path"]}


def replay_sample(environment, sample):
    import cv2
    from robosuite.utils.camera_utils import get_camera_extrinsic_matrix
    from simulation.libero_control import OPEN_GRIPPER, hold_gripper

    paths = source_files(sample)
    hashes = {key: sha256(path) for key, path in paths.items()}
    localization = json.loads(paths["localization"].read_text())
    target_truth = json.loads(paths["target_ground_truth"].read_text())
    raw = environment.reset(seed=1000, init_state_index=sample["initial_state_index"])
    if environment.last_init_state_sha256 != sample["initial_state_sha256"]:
        raise ValueError("Official initial-state hash differs from frozen capture.")
    raw = hold_gripper(environment, raw, OPEN_GRIPPER, "evaluation_only_replay", 10)
    inner = environment.env.env
    objects = []
    for obj in [*inner.objects, *inner.fixtures]:
        body_id = environment.sim.model.body_name2id(obj.root_body)
        objects.append({"instance": obj.name,
                        "semantic_identity": obj.category_name.replace("_", " "),
                        "position_world_m": environment.sim.data.body_xpos[body_id].tolist()})
    objects.sort(key=lambda item: item["instance"])
    target = next(item for item in objects if item["instance"] == target_truth["simulator_target_instance"])
    target_error = float(np.linalg.norm(np.array(target["position_world_m"]) - target_truth["simulator_target_position_world_m"]))
    saved_rgb = cv2.cvtColor(cv2.imread(str(paths["rgb"])), cv2.COLOR_BGR2RGB)
    # Use the same observation path and rendering settings as the original
    # capture; a separate sim.render call can differ at antialiased boundaries.
    replay_rgb = np.flip(raw["agentview_image"], axis=0)
    pixel_delta = np.abs(replay_rgb.astype(np.int16) - saved_rgb.astype(np.int16))
    replay_check = {"initial_state_sha256": environment.last_init_state_sha256,
                    "target_position_error_m": target_error,
                    "rgb_max_channel_error": int(pixel_delta.max()),
                    "rgb_differing_pixels": int(np.any(pixel_delta != 0, axis=2).sum()),
                    "rgb_exact_match": bool(np.array_equal(replay_rgb, saved_rgb))}
    if target_error > 1e-6 or not replay_check["rgb_exact_match"]:
        raise ValueError(f"Replay does not reproduce frozen scene {sample['sample_id']}: {replay_check}")
    transform = get_camera_extrinsic_matrix(environment.sim, "agentview")
    candidates = match_candidates(localization, transform, objects)
    for item in candidates:
        saved = next(value for value in target_truth["candidate_distances"] if value["object_id"] == item["object_id"])
        if not np.allclose(item["centroid_world_m"], saved["centroid_world_m"], atol=1e-8, rtol=0):
            raise ValueError("Camera transform differs from frozen centroid calibration.")
    audited_target = [item["object_id"] for item in candidates if item["simulator_instance"] == target["instance"]]
    return {"schema_version": 1, "evaluation_only": True, "excluded_from_vlm_inputs": True,
            "sample_id": sample["sample_id"], "partition": sample["partition"],
            "target_description": sample["target_description"],
            "original_ground_truth_object_id": sample["ground_truth_object_id"],
            "audited_ground_truth_object_id": audited_target[0] if len(audited_target) == 1 else None,
            "target_label_agrees": audited_target == [sample["ground_truth_object_id"]],
            "source_sha256": hashes, "replay_check": replay_check,
            "world_T_camera": transform.tolist(), "simulator_objects": objects,
            "candidates": candidates,
            "unassigned_simulator_instances": [item["instance"] for item in objects if item["instance"] not in {row["simulator_instance"] for row in candidates}]}


def run_audit(pilot=False):
    from simulation.libero_env import LiberoTaskEnvironment

    frozen = json.loads((FROZEN / "dataset_manifest.json").read_text())
    local = json.loads((SCENES / "manifest.json").read_text())
    if frozen["samples"] != local["samples"]:
        raise ValueError("Local manifest differs from frozen experiment.")
    samples = frozen["samples"]
    if pilot:
        samples = [row for row in samples if row["task_index"] in PILOT_TASKS and row["initial_state_index"] in PILOT_STATES]
    (AUDIT / "identities").mkdir(parents=True, exist_ok=True)
    for task_index in sorted({row["task_index"] for row in samples}):
        task_samples = [row for row in samples if row["task_index"] == task_index]
        with LiberoTaskEnvironment(task_index=task_index, image_width=768, image_height=768) as env:
            for sample in task_samples:
                destination = AUDIT / "identities" / f"{sample['sample_id']}.json"
                if destination.exists():
                    row = json.loads(destination.read_text())
                    if row["source_sha256"] != {key: sha256(path) for key, path in source_files(sample).items()}:
                        raise ValueError(f"Frozen inputs changed: {sample['sample_id']}")
                else:
                    row = replay_sample(env, sample)
                    destination.write_text(json.dumps(row, indent=2), encoding="utf-8")
                print(sample["sample_id"], "target_agrees=", row["target_label_agrees"],
                      " | ".join(f"{c['object_id']}={c['semantic_identity'] or c['status']}" for c in row["candidates"]), flush=True)


def semantic_prediction(prediction, audit):
    """Join by persistent ID only; never infer an identity from model text."""
    object_id = prediction["predicted_object_id"]
    if object_id is None:
        return "none"
    candidates = {row["object_id"]: row for row in audit["candidates"]}
    if object_id not in candidates:
        return "invalid_candidate_id"
    return candidates[object_id]["semantic_identity"] or "unresolved_identity"


def projected_centroid_in_roi(candidate, intrinsics):
    x, y, z = candidate["centroid_3d_m"]
    if not np.all(np.isfinite([x, y, z])) or z <= 0:
        return False
    u = intrinsics["fx"] * x / z + intrinsics["cx"]
    v = intrinsics["fy"] * y / z + intrinsics["cy"]
    roi = candidate["roi"]
    return bool(roi["x1"] <= u <= roi["x2"] and roi["y1"] <= v <= roi["y2"])


def load_predictions():
    paths = {"frozen_reference": FROZEN / "association_records.json",
             "paired_ablation": ROOT / "output_format_ablation" / "full_records.json"}
    predictions = []
    for source, path in paths.items():
        for row in json.loads(path.read_text())["records"]:
            predictions.append({"source": source if source == "frozen_reference" else f"ablation_{row['condition']}",
                **{key: row[key] for key in ("sample_id", "target_description", "predicted_object_id", "ground_truth_object_id", "correct")},
                "raw_likelihood": row.get("raw_sequence_likelihood", row.get("raw_association_likelihood"))})
    return predictions, {name: sha256(path) for name, path in paths.items()}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_contact_sheets():
    """Round-trip the existing image builder to check every saved crop/ID tile."""
    import cv2
    import tempfile
    from vlm_module import create_roi_contact_sheet

    samples = json.loads((FROZEN / "dataset_manifest.json").read_text())["samples"]
    checks = []
    with tempfile.TemporaryDirectory() as temporary:
        destination = Path(temporary) / "contact_sheet.png"
        for sample in samples:
            paths = source_files(sample)
            create_roi_contact_sheet(paths["rgb"], paths["localization"], destination, tile_size_px=448)
            original = cv2.imread(str(paths["contact_sheet"]))
            regenerated = cv2.imread(str(destination))
            exact = np.array_equal(original, regenerated)
            if not exact:
                raise ValueError(f"Frozen contact sheet differs from its RGB/candidate map: {sample['sample_id']}")
            checks.append({"sample_id": sample["sample_id"], "pixel_exact_reconstruction": exact,
                           "source_sha256": {key: sha256(path) for key, path in paths.items()}})
            if len(checks) % 50 == 0:
                print(f"Contact-sheet round trip: {len(checks)}/500 exact", flush=True)
    (AUDIT / "visual_input_checks.json").write_text(json.dumps({"evaluation_only": True, "checks": checks}, indent=2), encoding="utf-8")


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    audits = {row["sample_id"]: row for path in sorted((AUDIT / "identities").glob("*.json"))
              for row in [json.loads(path.read_text())]}
    if len(audits) != 500:
        raise ValueError(f"Full report needs 500 audited scenes; found {len(audits)}.")
    manifest = json.loads((FROZEN / "dataset_manifest.json").read_text())
    roi_disagreements = []
    for sample in manifest["samples"]:
        if audits[sample["sample_id"]]["source_sha256"] != {key: sha256(path) for key, path in source_files(sample).items()}:
            raise ValueError("Frozen source file changed during audit.")
        localization = json.loads(source_files(sample)["localization"].read_text())
        roi_disagreements.extend({"sample_id": sample["sample_id"], "object_id": item["object_id"]}
                                 for item in localization["objects"] if not projected_centroid_in_roi(item, localization["camera_intrinsics"]))
    if not (AUDIT / "visual_input_checks.json").exists():
        audit_contact_sheets()
    visual_checks = json.loads((AUDIT / "visual_input_checks.json").read_text())["checks"]
    if {row["sample_id"] for row in visual_checks} != set(audits):
        raise ValueError("Incomplete visual-input audit.")
    for check in visual_checks:
        if not check["pixel_exact_reconstruction"] or check["source_sha256"] != audits[check["sample_id"]]["source_sha256"]:
            raise ValueError("Visual-input audit mismatch.")
    predictions, prediction_hashes = load_predictions()
    joined = []
    for row in predictions:
        audit = audits[row["sample_id"]]
        if row["target_description"] != audit["target_description"] or row["ground_truth_object_id"] != audit["original_ground_truth_object_id"]:
            raise ValueError("Prediction/audit target mismatch.")
        joined.append({**row, "partition": audit["partition"],
                       "predicted_semantic_identity": semantic_prediction(row, audit),
                       "audited_ground_truth_object_id": audit["audited_ground_truth_object_id"],
                       "target_label_agrees": audit["target_label_agrees"]})
    write_csv(AUDIT / "semantic_associations.csv", joined)
    candidate_rows = [{"sample_id": sample_id, "partition": audit["partition"],
                       "target_description": audit["target_description"],
                       **{key: item[key] for key in ("object_id", "semantic_identity", "simulator_instance", "status", "nearest_xy_distance_m", "next_object_distance_gap_m")}}
                      for sample_id, audit in audits.items() for item in audit["candidates"]]
    write_csv(AUDIT / "candidate_identity_map.csv", candidate_rows)
    targets = sorted({row["target_description"] for row in joined})
    identities = sorted({row["semantic_identity"] for row in candidate_rows if row["semantic_identity"]})
    columns = sorted(set(identities) | {row["predicted_semantic_identity"] for row in joined})
    sources = sorted({row["source"] for row in joined})
    matrices, summaries = [], {}
    fig, axes = plt.subplots(1, len(sources), figsize=(21, 6.5), layout="constrained")
    for ax, source in zip(axes, sources):
        rows = [row for row in joined if row["source"] == source]
        counts = Counter((row["target_description"], row["predicted_semantic_identity"]) for row in rows)
        matrix = np.array([[counts[target, label] for label in columns] for target in targets])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=50)
        ax.set(xticks=range(len(columns)), xticklabels=columns, yticks=range(len(targets)),
               yticklabels=targets, title=source, xlabel="Selected candidate's simulator identity", ylabel="Requested target")
        plt.setp(ax.get_xticklabels(), rotation=55, ha="right", fontsize=8)
        for i in range(len(targets)):
            for j in range(len(columns)):
                if matrix[i, j]:
                    ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > 25 else "black", fontsize=9)
        for split in ("all", "calibration", "validation", "test"):
            subset = rows if split == "all" else [row for row in rows if row["partition"] == split]
            split_counts = Counter((row["target_description"], row["predicted_semantic_identity"]) for row in subset)
            matrices.extend({"source": source, "partition": split, "target": target, "prediction": label, "count": split_counts[target, label]}
                            for target in targets for label in columns)
        errors = [row for row in rows if not row["correct"]]
        summaries[source] = {"trials": len(rows), "original_correct": sum(row["correct"] for row in rows),
            "errors": len(errors), "errors_with_raw_likelihood_ge_099": sum(row["raw_likelihood"] is not None and row["raw_likelihood"] >= .99 for row in errors),
            "error_pairs": [{"target": target, "selected_identity": identity, "count": count}
                            for (target, identity), count in Counter((row["target_description"], row["predicted_semantic_identity"]) for row in errors).most_common()]}
    fig.suptitle("Evaluation-only semantic confusion matrices | 50 frozen scenes per target")
    fig.savefig(AUDIT / "semantic_confusion_matrices.png", dpi=170)
    plt.close(fig)
    write_csv(AUDIT / "semantic_confusion_counts.csv", matrices)
    summary = {"evaluation_only": True, "vlm_calls": 0, "audited_scenes": len(audits),
        "audited_candidates": len(candidate_rows), "identity_status_counts": dict(Counter(row["status"] for row in candidate_rows)),
        "exact_rgb_replays": sum(row["replay_check"]["rgb_exact_match"] for row in audits.values()),
        "exact_contact_sheet_reconstructions": len(visual_checks),
        "centroid_to_roi_disagreements": roi_disagreements,
        "target_label_disagreements": [key for key, row in audits.items() if not row["target_label_agrees"]],
        "maximum_target_position_error_m": max(row["replay_check"]["target_position_error_m"] for row in audits.values()),
        "maximum_candidate_match_distance_m": max(row["nearest_xy_distance_m"] for row in candidate_rows),
        "minimum_candidate_alternative_gap_m": min(row["next_object_distance_gap_m"] for row in candidate_rows if row["next_object_distance_gap_m"] is not None),
        "unassigned_instances": {key: row["unassigned_simulator_instances"] for key, row in audits.items() if row["unassigned_simulator_instances"]},
        "prediction_source_sha256": prediction_hashes, "sources": summaries}
    (AUDIT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# LIBERO candidate-identity ground-truth audit", "",
        "Evaluation only. No VLM inference, no changed predictions, and no production-pipeline changes.", "",
        f"Audited {len(audits)} scenes and {len(candidate_rows)} localized candidates. "
        f"RGB-exact replays: {summary['exact_rgb_replays']}/500. "
        f"Candidate status counts: {summary['identity_status_counts']}. "
        f"Target-label disagreements: {len(summary['target_label_disagreements'])}. "
        f"Projected-centroid/ROI disagreements: {len(roi_disagreements)}.", "",
        "## Method", "",
        "Replayed official fixed states, seed 1000, two cameras at 768 x 768, and the same ten open-gripper settling steps. "
        "Every replay must match the frozen state hash, target XYZ within 1 micrometre, and original RGB pixels exactly. "
        "Saved localized centroids are transformed with simulator camera calibration and matched to every simulator object/fixture root body in world XY. "
        "Only mutual-nearest, unique pairs within 75 mm and separated from alternatives by at least 10 mm are assigned. "
        "These are conservative geometric audit checks, not VLM confidence thresholds. Ambiguous matches remain unresolved.", "",
        "The semantic name comes from the simulator category, not a model prediction or crop OCR. "
        "Basket is retained as a candidate. All identities and world poses remain in this separate audit directory. "
        "Original images, localization JSON, labels and prediction files are hash-verified and never rewritten.", "",
        "All 500 contact sheets are also independently regenerated into temporary files using the existing RGB/ROI/contact-sheet code, "
        "and must match their frozen counterparts pixel-for-pixel. This verifies that saved candidate titles/crops correspond to "
        "the saved RGB and localization metadata. Each saved 3D centroid is also projected through the saved intrinsics "
        "to check that it lies in its own ROI. These checks do not measure semantic recognizability.", "",
        "## Semantic errors in saved predictions", "",
        "The frozen reference and two fresh output-format conditions are distinct runs; do not pool them as independent scenes. "
        "Correctness remains the original saved outcome. No threshold or model selection is performed here.", "",
        "| Run | Requested target | Selected candidate identity | Error count |", "| --- | --- | --- | ---: |"]
    for source, item in summaries.items():
        for pair in item["error_pairs"]:
            lines.append(f"| {source} | {pair['target']} | {pair['selected_identity']} | {pair['count']} |")
    lines.extend(["", "## Limits", "",
        "Centroid identity matching does not establish per-point cluster purity or full object visibility. "
        "A crop can contain an occluding neighbour or background even when its cluster ID is correct. "
        "Simulator semantic identity does not establish that a human can recognize the product from the image. "
        "The same task layouts, crop positions, and deterministic IDs can correlate with category; confusion counts alone cannot prove an ID/position prior. "
        "A future label/order permutation control would be needed. No human recognition study was performed.", "",
        "## Files", "", "- `identities/`: per-scene simulator catalog, every candidate match, replay checks and source hashes.",
        "- `candidate_identity_map.csv`: one row per localized candidate.",
        "- `semantic_associations.csv`: unchanged predictions joined to evaluation-only semantic identities.",
        "- `semantic_confusion_counts.csv`: complete count matrices, including zero cells, for all/calibration/validation/test.",
        "- `semantic_confusion_matrices.png`: visual matrices for the three saved runs.",
        "- `review/`: representative original contact sheets and bounding-box audit panels; never model inputs.", ""])
    lines.extend(["## Representative visual inspection", "",
        "Four calibration scenes were inspected directly by the coding agent: alphabet soup and tomato sauce, each at states 0 and 1. "
        "The contact-sheet titles and bounding boxes agree in these examples. The alphabet-soup can is partly occluded by milk; "
        "the tomato-sauce can shows its lid and coloured side without clearly exposed product text. "
        "Some selected distractors have visible BBQ Sauce or Chocolate Pudding lettering. "
        "These are agent inspection observations, not blinded human readability measurements. "
        "See [inspection notes](review/INSPECTION.md) for scene-specific observations and limitations.", "",
        "## Reproduction", "", "From the repository root, in the existing WSL LIBERO environment:", "",
        "```sh", "export MUJOCO_GL=egl", "export PYOPENGL_PLATFORM=egl",
        "python -m scripts.audit_libero_candidate_identities pilot",
        "python -m scripts.audit_libero_candidate_identities full",
        "python -m scripts.audit_libero_candidate_identities report", "```", "",
        "`report` only reads saved audit/prediction files; it does not launch the simulator or call a model. "
        "The full replay resumes completed, hash-checked identity files. All outputs stay in the separate candidate_identity_audit directory.", ""])
    (AUDIT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def review_figures():
    """Reuse frozen contact sheets and the existing bounding-box annotator."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from helper_function import RBGAnnotation

    predictions, _ = load_predictions()
    manifest = json.loads((FROZEN / "dataset_manifest.json").read_text())
    (AUDIT / "review").mkdir(exist_ok=True)
    for sample in manifest["samples"]:
        if sample["task_index"] not in PILOT_TASKS or sample["initial_state_index"] not in (0, 1):
            continue
        audit = json.loads((AUDIT / "identities" / f"{sample['sample_id']}.json").read_text())
        paths = source_files(sample)
        annotator = RBGAnnotation()
        rgb = plt.imread(paths["rgb"])
        annotated = annotator.annotate((rgb * 255).astype(np.uint8), [
            {"roi": item["roi"], "label": item["object_id"].removeprefix("object_")}
            for item in audit["candidates"]])
        fig = plt.figure(figsize=(16, 13), layout="constrained")
        grid = fig.add_gridspec(2, 2, height_ratios=[1, .85])
        top = fig.add_subplot(grid[0, :])
        top.imshow(plt.imread(paths["contact_sheet"]))
        top.set_title("Original VLM contact sheet (unchanged)")
        top.axis("off")
        left = fig.add_subplot(grid[1, 0])
        left.imshow(annotated)
        left.set_title("Frozen RGB + existing localization boxes")
        left.axis("off")
        right = fig.add_subplot(grid[1, 1])
        right.axis("off")
        text = ["EVALUATION ONLY — NOT A VLM INPUT", "", f"Target: {sample['target_description']}",
                f"Verified target ID: {audit['audited_ground_truth_object_id']}", ""]
        text.extend(f"{item['object_id']} = {item['semantic_identity'] or item['status']}" for item in audit["candidates"])
        text.extend(["", "Saved predictions:"])
        for row in predictions:
            if row["sample_id"] == sample["sample_id"]:
                text.extend([row["source"], f"  {row['predicted_object_id']} -> {semantic_prediction(row, audit)}",
                             f"  raw likelihood = {row['raw_likelihood']:.9f}"])
        right.text(0, 1, "\n".join(text), ha="left", va="top", fontsize=11, family="monospace")
        fig.suptitle(sample["sample_id"], fontsize=16)
        fig.savefig(AUDIT / "review" / f"{sample['sample_id']}_audit.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) == 2 else "pilot"
    if stage in {"pilot", "full"}:
        run_audit(pilot=stage == "pilot")
    elif stage == "report":
        report()
    elif stage != "review":
        raise SystemExit("Use pilot, full, review or report.")
    if stage in {"review", "report"}:
        review_figures()
