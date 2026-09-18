"""Compare BlenderProc/42 templates with the frozen original 14-view templates.

Run in the existing WSL association environment, with CPU DINO for baseline parity:
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 \
    python -m scripts.run_wrist_blenderproc_semantic

BLENDERPROC_OUTPUT may name an existing render directory to resume scoring.
BlenderProc 2.6.1, Blender 3.3.1 and the official pose array live under outputs/cache.
No production association, observation, localization or geometry changes.
"""

import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

import cad_object_association as association
from scripts.run_wrist_input_ablation import (
    ROOT, DATASET, read_json, read_rgb, save_csv, save_image, save_json, snapshot_hashes,
)
from scripts.run_wrist_patch_ablation import evaluate_variant, verify_artifacts
from scripts.run_wrist_sam_visual_comparison import prepare_masked_input
from scripts.run_wrist_top5_semantic_ablation import score_view_similarities
from scripts.run_wrist_view_ablation import ABLATION_2


OUTPUT_ROOT = ROOT / "outputs/blenderproc42_semantic_wrist_association_2026-09-11"
SOURCE = ROOT / "outputs/cache/sam6d_render_source"
TOP5_BASELINE = ROOT / "outputs/top5_semantic_wrist_association_2026-09-11/20260915T164654190031Z"


def render_templates(output, library):
    if not (output / "render_job.json").exists():
        for name in ("cam_poses_level0.npy", "provenance.json"):
            shutil.copyfile(SOURCE / name, output / name)
        save_json(output / "render_job.json", {"output": str(output), "cad_library": [
            dict(item, absolute_mesh_path=str(ROOT / item["file_path"])) for item in library]})
    job = read_json(output / "render_job.json")
    assert [{k: v for k, v in item.items() if k != "absolute_mesh_path"}
            for item in job["cad_library"]] == library
    assert Path(job["output"]) == output
    if (output / "render_runtime.json").exists():
        return False
    # DINO is deliberately CPU; the isolated renderer can still use the GPU.
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    with (output / "render.log").open("w") as log:
        subprocess.run([str(ROOT / "outputs/cache/blenderproc_env/bin/blenderproc"), "run",
            str(ROOT / "scripts/render_wrist_blenderproc_templates.py"), str(output / "render_job.json"),
            "--custom-blender-path", str(ROOT / "outputs/cache/blender/blender-3.3.1-linux-x64")],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    if not (output / "render_runtime.json").exists():
        raise RuntimeError(f"Blender did not complete; inspect {output / 'render.log'}")
    return True


def prepare_rgba(rgba):
    """Composite antialiased Blender foreground over white, tight crop, letterbox."""
    alpha = rgba[..., 3:4].astype(float) / 255
    mask = rgba[..., 3] > 0
    if not mask.any() or mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        raise ValueError("Rendered foreground is empty or clipped by the image boundary.")
    rgb = np.rint(rgba[..., :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    canvas, _, bbox = prepare_masked_input(rgb, mask)
    return canvas, mask, bbox


def contact_sheet(images, labels, columns=7):
    """Display exact encoder inputs without changing their pixel dimensions."""
    rows = (len(images) + columns - 1) // columns
    sheet = np.full((rows * 252, columns * 224, 3), 255, np.uint8)
    for index, (rgb, label) in enumerate(zip(images, labels, strict=True)):
        y, x = index // columns * 252, index % columns * 224
        cv2.putText(sheet, label, (x + 5, y + 19), cv2.FONT_HERSHEY_SIMPLEX,
                    .43, (0, 0, 0), 1, cv2.LINE_AA)
        sheet[y + 28:y + 252, x:x + 224] = rgb
    return sheet


def evaluate(pairs, score, targets, audit):
    rows = [dict(p, cls_score=p[score], runtime={"cls_comparison_time_s": 0.}) for p in pairs]
    result = evaluate_variant("cls_only", rows, targets, audit, {
        "observed_image_loading_time_s": 0., "observed_joint_feature_extraction_time_s": 0.,
        "observed_mask_preparation_time_s": 0.})
    for target in result["associations"]:
        winner = next(p for p in target["candidate_ranking"] if p["object_id"] == target["predictions"]["visual"])
        target["semantic_match"] = {"object_id": winner["object_id"], "score": winner[score],
            "best_template_id": winner["best_template_id"],
            "best_template_rotation_camera_from_cad": winner["best_template_rotation_camera_from_cad"]}
    return result


def write_report(output, evaluations, changes, targets, runtime):
    lines = ["# BlenderProc 42-view semantic test", "",
        "Fixed six-object scene; all 36 CAD/candidate pairs. Primary comparison: top-five mean versus top-five mean. "
        "The maximum-view score is also reported as a fixed diagnostic. No weights or rendering settings were tuned from scores.", "",
        "| Templates and aggregation | Visual top-1 | Fused top-1 (saved geometry) | Mean visual margin | Mean fused margin |",
        "|---|---:|---:|---:|---:|"]
    for name, result in evaluations.items():
        lines.append(f"| {name} | {result['accuracy']['visual']['correct']}/6 | {result['accuracy']['fused']['correct']}/6 | "
            f"{result['margin_summary']['visual_raw']['mean_correct_minus_best_incorrect']:+.6f} | "
            f"{result['margin_summary']['fused']['mean_correct_minus_best_incorrect']:+.6f} |")
    lines += ["", "Margins are correct minus strongest incorrect candidate. Increased cosine alone does not establish better discrimination.", "",
        "| Target | Correct top-five cosine: old → new | Visual prediction: old → new | Visual margin: old → new | Fused prediction: old → new |",
        "|---|---:|---|---:|---|"]
    for row in changes:
        if row["aggregation"] != "top5_mean":
            continue
        lines.append(f"| {targets[row['cad_id']]} | {row['old_correct_score']:.6f} → {row['new_correct_score']:.6f} | "
            f"{row['old_visual_prediction']} → {row['new_visual_prediction']} | "
            f"{row['old_visual_margin']:+.6f} → {row['new_visual_margin']:+.6f} | "
            f"{row['old_fused_prediction']} → {row['new_fused_prediction']} |")
    lines += ["", "Object IDs: 001 pulley; 002 large gear; 003 blue block; 004 red block; 005 medium gear; 006 rectangular pin.", "",
        "## Controlled inputs and changed template pipeline", "",
        "Unchanged: exact saved localization-masked observation images and normalized CLS features from Ablation 2; "
        "DINOv2-small and its pinned checkpoint; 224×224 white aspect-preserving letterbox; CAD meshes and library colors; "
        "all recorded geometry; exhaustive comparison; `0.5*((cosine+1)/2) + 0.5*geometry`. No SAM or registration runs.", "",
        "Changed together: original 14 orthographic, normal-shaded templates versus BlenderProc 2.6.1 / Blender 3.3.1 "
        "Cycles perspective renders at the authors' exact 42 predefined camera poses. New templates are tightly cropped "
        "using renderer alpha before the existing white letterbox; old templates remain their exact saved inputs. "
        "This is a combined template-pipeline test, not a separate causal test of lighting, cropping or view count.", "",
        "SAM-6D custom-render settings: 512×512, FOV 0.691111 radians, normalized object radius 0.5, camera radius 2, "
        "point light at 2.5×camera position with energy 1000, 50 samples. Object center/radius use exact mesh vertices "
        "instead of random sampled extrema. Existing RGB values are assigned directly as Principled Base Color; "
        "roughness 0.5, metallic 0; no fitted textures/materials. Renderer alpha is composited over white. "
        "The saved render runtime records actual color management, device and denoising. These choices were fixed before scoring.", "",
        "Template rotations map original CAD axes into the OpenCV camera axes (right/down/forward). They are discrete "
        "template orientations, not recovered workspace poses. Top-five is the arithmetic mean of the five largest "
        "individual cosines; the returned best template is the single cosine maximum, with first-index ties.", "",
        "This retains our encoder and observations and is not a full SAM-6D/CNOS reproduction. "
        "Any result is specific to this fixed scene; no general accuracy claim follows from six objects.", "",
        "## Timing", "", f"Recorded renderer work: {runtime['render']['wall_time_s']:.2f}s. "
        f"DINO model load: {runtime['encoder_load_time_s']:.2f}s. "
        f"252 CAD CLS features: {runtime['cad_feature_extraction_time_s']:.2f}s. "
        f"All 1512 cosines, aggregation and four evaluations: {runtime['scoring_evaluation_time_s']:.4f}s. "
        "Dependency installation/download excluded. Observed features and geometry were reused at zero recomputation cost. "
        "The renderer ran as a separate stage; resumed scoring does not include rendering in its wall clock. "
        "Per-object render timings and the scoring wall clock are saved in runtime.json.", "",
        f"Actual final render mode: **{runtime['render']['device']}**; encoder: **{runtime['encoder_device']}**. "
        "BlenderProc 2.6.1's `clean_up()` deletes the scene's custom `cycles` property group, resetting "
        "the device to CPU after startup selected CUDA. This was verified separately in Blender 3.3.1. "
        "The timing above is a CPU rendering measurement, not a GPU benchmark; the CUDA preference alone does not prove GPU use.", "",
        "## Inspection and data", "", "![Observed inputs and matching-CAD templates](template_comparison.png)", "",
        "Columns: saved observation; old best template; old second best; BlenderProc best; BlenderProc second best. "
        "Each row uses the known correct CAD/observation pair; these are not necessarily the predicted candidate.", "",
        "[All pair scores](pair_scores.json), [all per-view scores](per_view_scores.csv), "
        "[all 36 pair-score changes](pair_changes.csv), [prediction/margin changes](target_changes.csv), "
        "[template rotations and crop boxes](templates.json), [protocol](protocol.json), "
        "[validation](validation.json), [runtime](runtime.json).", ""]
    for cid in targets:
        lines += [f"### {targets[cid]}", "", f"![All 42 {cid} templates](templates/{cid}/all_views.png)", ""]
    (output / "COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    started = perf_counter()
    os.chdir(ROOT)
    fixed_hashes, prior_hashes = snapshot_hashes(), verify_artifacts(ABLATION_2)
    verify_artifacts(TOP5_BASELINE)
    production = {p: association.content_hash(ROOT / p) for p in
                  ("cad_object_association.py", "CADPointCloudRegistration.py", "main.py")}
    protocol = read_json(ABLATION_2 / "protocol.json")
    targets = protocol["known_cad_ids"]
    assert association.DINO_MODEL == protocol["encoder"] == "dinov2_vits14"
    assert association.DINO_REPO == protocol["encoder_source"]
    assert association.content_hash(association.DINO_HUB_DIR / "checkpoints/dinov2_vits14_pretrain.pth") == protocol["checkpoint_sha256"]
    assert (association.VISUAL_WEIGHT, association.GEOMETRY_WEIGHT, association.FUSION_METHOD) == (.5, .5, "weighted_mean")
    output = Path(os.environ.get("BLENDERPROC_OUTPUT", OUTPUT_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))).resolve()
    output.mkdir(parents=True, exist_ok=True)
    library = read_json(DATASET / "cad_library.json")
    stage = perf_counter()
    rendered_now = render_templates(output, library)
    runtime = {"render_process_time_s": perf_counter() - stage if rendered_now else None,
               "reused_completed_render_stage": not rendered_now,
               "render": read_json(output / "render_runtime.json")}
    provenance = read_json(output / "provenance.json")
    assert association.content_hash(output / "cam_poses_level0.npy") == provenance["files"]["cam_poses_level0.npy"]["sha256"]
    poses = np.load(output / "cam_poses_level0.npy")
    assert poses.shape == (42, 4, 4)
    rotations = poses[:, :3, :3].transpose(0, 2, 1)
    np.testing.assert_allclose(rotations @ rotations.transpose(0, 2, 1), np.broadcast_to(np.eye(3), (42, 3, 3)), atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(rotations), 1, atol=1e-6)
    np.testing.assert_allclose((rotations @ poses[:, :3, 3, None])[:, :, 0], np.tile([0, 0, -1000], (42, 1)), atol=.001)
    with np.load(ABLATION_2 / "features/observed.npz") as saved:
        observed, object_ids = saved["cls_features"], saved["object_ids"].tolist()
    assert observed.shape == (6, 384)
    (output / "features").mkdir(exist_ok=True)
    (output / "inputs").mkdir(exist_ok=True)
    shutil.copyfile(ABLATION_2 / "features/observed.npz", output / "features/observed.npz")
    for oid in object_ids:
        shutil.copyfile(ABLATION_2 / "inputs" / f"{oid}.png", output / "inputs" / f"{oid}.png")
    torch.set_num_threads(1)
    stage = perf_counter()
    model = association.load_dino_encoder()
    runtime["encoder_load_time_s"] = perf_counter() - stage
    assert str(next(model.parameters()).device) == "cpu", "Use CPU DINO for exact baseline numerical parity."
    features, templates = {}, []
    runtime["cad_feature_extraction_time_s"] = 0.
    for cid in targets:
        directory = output / "templates" / cid
        directory.mkdir(parents=True, exist_ok=True)
        images = []
        for index in range(42):
            rendered_path = output / "renders" / cid / f"view_{index + 1:02}_rgba.png"
            bgra = cv2.imread(str(rendered_path), cv2.IMREAD_UNCHANGED)
            if bgra is None or bgra.shape != (512, 512, 4):
                raise ValueError(f"Missing or invalid RGBA render: {rendered_path}")
            rgb, mask, bbox = prepare_rgba(cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA))
            images.append(rgb)
            save_image(directory / f"view_{index + 1:02}.png", rgb)
            save_image(directory / f"view_{index + 1:02}_mask.png", mask.astype(np.uint8) * 255)
            templates.append({"cad_id": cid, "view_index_1based": index + 1,
                "template_id": f"{cid}/view_{index + 1:02}", "crop_bbox_xyxy": list(map(int, bbox)),
                "R_template_camera_from_cad": rotations[index].tolist(),
                "camera_direction_in_cad_frame": (poses[index, :3, 3] / np.linalg.norm(poses[index, :3, 3])).tolist(),
                "input_sha256": association.content_hash(directory / f"view_{index + 1:02}.png")})
        save_image(directory / "all_views.png", contact_sheet(images, [f"{cid} v{i + 1:02}" for i in range(42)]))
        stage = perf_counter()
        cls = association.extract_dino_features(images).astype(np.float64)
        cls /= np.linalg.norm(cls, axis=1, keepdims=True)
        assert cls.shape == (42, 384) and np.isfinite(cls).all()
        runtime["cad_feature_extraction_time_s"] += perf_counter() - stage
        features[cid] = cls
        np.savez_compressed(output / "features" / f"{cid}.npz", cls_features=cls)
        print(f"ENCODED {cid}: 42 templates", flush=True)
    stage = perf_counter()
    old_pairs = read_json(TOP5_BASELINE / "pair_scores.json")
    pairs = []
    for old in old_pairs:
        cid, oid = old["cad_id"], old["object_id"]
        cosines = np.clip(features[cid] @ observed[object_ids.index(oid)], -1, 1)
        pairs.append({"cad_id": cid, "object_id": oid,
            **score_view_similarities(cid, cosines, rotations),
            "geometry_raw": old["geometry_raw"], "geometry_score": old["geometry_score"]})
    audit = read_json(DATASET / "identity_audit.json")
    evaluations = {}
    for name, source_pairs in (("original14", old_pairs), ("blenderproc42", pairs)):
        for aggregation, key in (("max_view", "max_view_score"), ("top5_mean", "semantic_score")):
            result = evaluate(source_pairs, key, targets, audit)
            result["variant"] = f"{name}_{aggregation}"
            evaluations[result["variant"]] = result
            if name == "original14":
                baseline = read_json(TOP5_BASELINE / aggregation / "evaluation.json")
                assert result["accuracy"] == baseline["accuracy"] and result["margins"] == baseline["margins"]
                assert [a["rankings"] for a in result["associations"]] == [a["rankings"] for a in baseline["associations"]]
    runtime["scoring_evaluation_time_s"] = perf_counter() - stage
    changes = []
    for aggregation in ("max_view", "top5_mean"):
        before, after = (evaluations[f"{name}_{aggregation}"] for name in ("original14", "blenderproc42"))
        for i, cid in enumerate(targets):
            row = {"cad_id": cid, "aggregation": aggregation,
                   "expected_object_id": before["margins"][i]["expected_object_id"]}
            for label, result in (("old", before), ("new", after)):
                row[f"{label}_correct_score"] = result["margins"][i]["visual_raw"]["correct_score"]
                for mode, margin_key in (("visual", "visual_raw"), ("fused", "fused")):
                    row[f"{label}_{mode}_prediction"] = result["associations"][i]["predictions"][mode]
                    row[f"{label}_{mode}_margin"] = result["margins"][i][margin_key]["correct_minus_best_incorrect"]
            for mode in ("visual", "fused"):
                row[f"{mode}_margin_delta"] = row[f"new_{mode}_margin"] - row[f"old_{mode}_margin"]
            changes.append(row)
    pair_changes = [{"cad_id": p["cad_id"], "object_id": p["object_id"],
        **{f"{key}_{label}": source[key] for key in ("semantic_score", "max_view_score")
           for label, source in (("old", old), ("new", p))},
        "top5_delta": p["semantic_score"] - old["semantic_score"],
        "max_delta": p["max_view_score"] - old["max_view_score"]}
        for old, p in zip(old_pairs, pairs, strict=True)]
    save_json(output / "pair_scores.json", pairs)
    save_json(output / "templates.json", templates)
    save_csv(output / "pair_changes.csv", pair_changes)
    save_csv(output / "target_changes.csv", changes)
    save_csv(output / "per_view_scores.csv", [{"cad_id": p["cad_id"], "object_id": p["object_id"],
        "view_index_1based": i, "cls_cosine": score, "in_top5": i in p["top5_view_indices_1based"]}
        for p in pairs for i, score in enumerate(p["per_view_cosines"], 1)])
    for name, result in evaluations.items():
        (output / name).mkdir(exist_ok=True)
        save_json(output / name / "evaluation.json", result)
    panels, labels = [], []
    for row in evaluations["original14_top5_mean"]["margins"]:
        cid, oid = row["cad_id"], row["expected_object_id"]
        panels.append(read_rgb(output / "inputs" / f"{oid}.png"))
        labels.append(f"{cid}: observed {oid[-3:]}")
        for name, source_pairs, directory in (("old", old_pairs, DATASET / "templates"),
                                             ("new", pairs, output / "templates")):
            pair = next(p for p in source_pairs if (p["cad_id"], p["object_id"]) == (cid, oid))
            for index in pair["top5_view_indices_1based"][:2]:
                panels.append(read_rgb(directory / cid / f"view_{index:02}.png"))
                labels.append(f"{name} v{index:02}: {pair['per_view_cosines'][index - 1]:.3f}")
    save_image(output / "template_comparison.png", contact_sheet(panels, labels, columns=5))
    assert snapshot_hashes() == fixed_hashes and verify_artifacts(ABLATION_2) == prior_hashes
    assert all(association.content_hash(ROOT / p) == digest for p, digest in production.items())
    assert len(pairs) == len({(p['cad_id'], p['object_id']) for p in pairs}) == 36
    assert len(templates) == 252
    assert association.content_hash(output / "features/observed.npz") == association.content_hash(ABLATION_2 / "features/observed.npz")
    assert all(association.content_hash(output / "inputs" / f"{oid}.png") ==
               association.content_hash(ABLATION_2 / "inputs" / f"{oid}.png") for oid in object_ids)
    save_json(output / "validation.json", {"pair_count": 36, "template_count": 252,
        "per_view_comparison_count": 1512, "all_masks_nonempty_and_unclipped": True,
        "original_rankings_and_margins_reproduced": True, "production_and_fixed_dataset_unchanged": True,
        "observed_inputs_features_and_geometry_reused_exactly": True,
        "camera_rotations_verified": True})
    save_json(output / "protocol.json", {"baseline": str(TOP5_BASELINE.relative_to(ROOT)),
        "observation_source": str(ABLATION_2.relative_to(ROOT)), "encoder": protocol["encoder"],
        "encoder_source": protocol["encoder_source"], "checkpoint_sha256": protocol["checkpoint_sha256"],
        "pose_source": provenance, "known_cad_ids": targets, "aggregation": "mean of top five individual cosines",
        "diagnostic": "max-view cosine", "fusion": "0.5*((raw+1)/2)+0.5*recorded_geometry",
        "background": "white", "template_preprocessing": "alpha composite, tight crop, existing 224 letterbox",
        "renderer": "BlenderProc 2.6.1; Blender 3.3.1; 512x512; 50 samples; pointlight 1000 at 2.5*camera",
        "mesh_fitting": "exact bbox center; radius from all vertices; radius normalized to 0.5; camera radius 2",
        "material": "unchanged library base colors assigned directly; roughness .5; metallic 0; no textures",
        "source_hashes": {p: association.content_hash(ROOT / p) for p in
            ("scripts/render_wrist_blenderproc_templates.py", "scripts/run_wrist_blenderproc_semantic.py",
             "scripts/run_wrist_top5_semantic_ablation.py")},
        "production_hashes": production, "fixed_dataset_hashes": fixed_hashes,
        "saved_ablation2_hashes": prior_hashes, "full_paper_reproduction": False, "tuning_performed": False})
    runtime["wall_time_before_report_s"] = perf_counter() - started
    runtime["encoder_device"] = str(next(model.parameters()).device)
    save_json(output / "runtime.json", runtime)
    write_report(output, evaluations, changes, targets, runtime)
    (output / "SHA256SUMS").write_text("".join(f"{association.content_hash(p)}  {p.relative_to(output).as_posix()}\n"
        for p in sorted(output.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"), encoding="utf-8")
    print({name: result["accuracy"] for name, result in evaluations.items()}, flush=True)
    print(output, flush=True)
    return output


if __name__ == "__main__":
    main()
