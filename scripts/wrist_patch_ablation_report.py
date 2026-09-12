"""Saved tables and mask-grid inspection for the isolated patch ablation."""

import cv2
import numpy as np

from scripts.run_wrist_input_ablation import save_csv, save_image, save_json


def save_report(output_dir, evaluations, baseline, timing, images, masks, weights):
    names = list(evaluations)
    changes = []
    for index, target in enumerate(evaluations["cls_only"]["per_target"]):
        row = {"cad_id": target["cad_id"], "expected_object_id": target["expected_object_id"]}
        for name, result in evaluations.items():
            prediction = result["per_target"][index]["predictions"]
            margins = result["margins"][index]
            row.update({f"{name}_{mode}_prediction": prediction[mode] for mode in ("visual", "fused")})
            row.update({f"{name}_{mode}_margin": margins[mode]["correct_minus_best_incorrect"]
                        for mode in ("visual_raw", "fused")})
            for mode in ("visual_raw", "fused"):
                row[f"{name}_{mode}_margin_delta_vs_cls"] = (margins[mode]["correct_minus_best_incorrect"]
                    - evaluations["cls_only"]["margins"][index][mode]["correct_minus_best_incorrect"])
        changes.append(row)
    save_csv(output_dir / "discrimination_changes.csv", changes)
    save_json(output_dir / "comparison.json", {
        "accuracy": {name: result["accuracy"] for name, result in evaluations.items()},
        "margin_summary": {name: result["margin_summary"] for name, result in evaluations.items()},
        "per_target_changes": changes,
        "margin_improvement_target_count_vs_cls": {
            name: {mode: sum(row[f"{name}_{mode}_margin_delta_vs_cls"] > 0 for row in changes)
                   for mode in ("visual_raw", "fused")} for name in names[1:]},
    })
    lines = ["# Ablation 2: does foreground patch matching improve discrimination?", "",
             "Exact masked RGB inputs from Ablation 1, white background, existing 224×224 letterbox, "
             "DINOv2-small, 14 fixed colored CAD views, and all 36 recorded geometry results. "
             "Production association code and Ablation 1 artifacts are unchanged.", "",
             "Margin = correct candidate score minus the highest-scoring incorrect candidate. "
             "A positive margin indicates correct separation; the sign is diagnostic and never gates predictions.", "",
             "| Visual variant | Visual top-1 | Fused top-1 | Mean raw visual margin | Mean fused margin | Accounted warm runtime (s) |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, result in evaluations.items():
        summary, accuracy = result["margin_summary"], result["accuracy"]
        lines.append(f"| {name} | {accuracy['visual']['correct']}/6 | {accuracy['fused']['correct']}/6 "
                     f"| {summary['visual_raw']['mean_correct_minus_best_incorrect']:+.6f} "
                     f"| {summary['fused']['mean_correct_minus_best_incorrect']:+.6f} "
                     f"| {result['runtime']['accounted_warm_runtime_s']:.6f} |")
    lines += ["", "The main comparison is accuracy and signed correct-versus-incorrect margins. "
              "Absolute CLS, patch, and blended scores have different empirical distributions; "
              "a larger mean score alone does not demonstrate better discrimination.", "",
              "## Correct-versus-strongest-incorrect visual margins", "",
              "| CAD | CLS | Patch | 0.5 CLS + 0.5 patch | Patch Δ vs CLS | Blend Δ vs CLS |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in changes:
        values = [row[f"{name}_visual_raw_margin"] for name in names]
        values += [row[f"{name}_visual_raw_margin_delta_vs_cls"] for name in names[1:]]
        lines.append(f"| {row['cad_id']} | " + " | ".join(f"{v:+.6f}" for v in values) + " |")
    lines += ["", "## Top-1 predictions", "",
              "Each cell shows visual / fused object IDs. IDs: 001 pulley; 002 large gear; "
              "003 blue block; 004 red block; 005 medium gear; 006 rectangular pin.", "",
              "| CAD | Truth | CLS | Patch | Blend |", "|---|---|---|---|---|"]
    for row in changes:
        predictions = [f"{row[f'{name}_visual_prediction']} / {row[f'{name}_fused_prediction']}" for name in names]
        lines.append(f"| {row['cad_id']} | {row['expected_object_id']} | " + " | ".join(predictions) + " |")
    lines += ["", "All candidates retain independent rankings. Duplicate selections are reported as conflicts; "
              "the existing conflict rule leaves downstream final object IDs null for conflicting sets.", "",
              "## Fixed method", "",
              "1. Extract final layer-normalized CLS and 256 patch tokens, then L2-normalize each descriptor. "
              "Patch tokens retain row-major `(16, 16, 384)` shape.",
              "2. Resize each saved binary observation mask with nearest-neighbor interpolation, using the "
              "same tight crop, dimensions and centering as its saved RGB input. Zero-pad the mask. "
              "Observed patch weight is the fraction of its 196 pixels inside the mask.",
              "3. Read CAD foreground from finite renderer ray hits with the same mesh fitting, "
              "camera directions and orthographic rays. Verify all 84 RGB renders against the saved templates. "
              "A CAD patch is eligible if any pixel has foreground support; no occupancy cutoff is tuned.",
              "4. Compute all 14 CLS cosines and select the highest-CLS view (first index wins exact ties). "
              "For every supported observed patch, take its maximum cosine against foreground CAD patches "
              "in that view. The occupancy-weighted mean of these maxima is the patch score. "
              "Matching is directional and allows multiple observed patches to select the same CAD patch.",
              "5. Evaluate CLS, patch, and their fixed 0.5/0.5 raw-score mean. Every variant uses "
              "`visual_normalized = (visual_raw + 1) / 2`, then "
              "`fused = 0.5 * visual_normalized + 0.5 * recorded_geometry_score`.", "",
              "Patch-only still requires CLS to select its CAD view. No SAM, new views, thresholds, "
              "candidate pruning, geometry recomputation, or weight tuning is used.", "",
              "## Runtime accounting", "",
              "The table sums measured stages required by each variant with CAD features and masks ready: "
              "observed RGB loading, joint token extraction, applicable observation-mask preparation, "
              "CLS comparison, applicable patch matching, and variant fusion/ranking/evaluation. "
              "Shared measurements are reused; these are not three independent end-to-end timing runs. "
              "Integrity checks, artifact saving and CAD/model setup are excluded.", "",
              f"Model loading: {timing['model_load_time_s']:.6f} s; all CAD joint feature extraction "
              f"(including PNG reads): {timing['cad_joint_feature_extraction_time_s']:.6f} s; "
              f"CAD ray-mask preparation: {timing['cad_mask_preparation_time_s']:.6f} s; "
              f"shared observed joint feature extraction: {timing['observed_joint_feature_extraction_time_s']:.6f} s. "
              "CPU, one thread. [Complete timings](runtime.json).", "",
              "## Saved evidence", "",
              "- [All 36 pair records](pair_scores.json): all 14 CLS cosines, selected view, CLS/patch/blended scores, "
              "foreground counts, geometry records, and pair timings.",
              "- [CLS scores](cls_only/all_pair_scores.csv), [evaluation](cls_only/evaluation.json), [margins](cls_only/margins.csv).",
              "- [Patch scores](patch_only/all_pair_scores.csv), [evaluation](patch_only/evaluation.json), [margins](patch_only/margins.csv).",
              "- [Blended scores](cls_patch/all_pair_scores.csv), [evaluation](cls_patch/evaluation.json), [margins](cls_patch/margins.csv).",
              "- [Per-target discrimination changes](discrimination_changes.csv), [comparison JSON](comparison.json), "
              "[protocol and hashes](protocol.json), [validation](validation.json).",
              "- `features/observed.npz` and `features/<CAD ID>.npz`: normalized CLS features, full normalized "
              "patch-token grids, masks and occupancy weights. CAD arrays follow saved view_01…view_14 order.",
              "- `inputs/` contains byte-identical Ablation 1 RGB inputs; `masks/` contains aligned binary masks. "
              "Black mask padding means zero occupancy; RGB padding remains white.", "",
              "![RGB inputs, aligned binary masks, and 16×16 occupancy grids](patch_masks.png)", "",
              "One fixed six-object scene supports this controlled comparison only. No settings were "
              "optimized from these outcomes; broader performance requires separate evaluation data.", ""]
    (output_dir / "COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")
    sheet = np.full((6 * 264 + 42, 720, 3), 255, dtype=np.uint8)
    for x, label in ((8, "Fixed RGB input"), (248, "Aligned object mask"), (488, "Patch occupancy")):
        cv2.putText(sheet, label, (x, 26), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 1, cv2.LINE_AA)
    for index, (image, mask, occupancy, record) in enumerate(zip(images, masks, weights, baseline["inputs"], strict=True)):
        y = 42 + index * 264
        cv2.putText(sheet, f"{record['object_id']} | {np.count_nonzero(occupancy)} supported patches", (8, y + 19),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1, cv2.LINE_AA)
        panels = [image, np.repeat((mask * 255)[..., None], 3, axis=-1),
                  np.repeat(cv2.resize(np.rint(occupancy * 255).astype(np.uint8), (224, 224),
                                       interpolation=cv2.INTER_NEAREST)[..., None], 3, axis=-1)]
        for x, panel in zip((8, 248, 488), panels, strict=True):
            sheet[y + 30:y + 254, x:x + 224] = panel
        for grid in range(0, 225, 14):
            cv2.line(sheet, (488 + min(grid, 223), y + 30), (488 + min(grid, 223), y + 253), (0, 150, 0), 1)
            cv2.line(sheet, (488, y + 30 + min(grid, 223)), (711, y + 30 + min(grid, 223)), (0, 150, 0), 1)
    save_image(output_dir / "patch_masks.png", sheet)
