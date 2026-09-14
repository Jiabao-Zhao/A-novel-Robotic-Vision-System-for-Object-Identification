"""Saved visual evidence and concise tables for the oracle-pose diagnostic."""

import cv2
import numpy as np

from scripts.run_wrist_input_ablation import DATASET, read_rgb, save_image


def save_report(output, diagonal, pairs, evaluations, truth, library, runtime):
    pin = next(r for r in diagonal if r['cad_id'] == 'KET16_Square_16mm')
    evaluation = evaluations['visible_cad']
    pin_ranking = next(r for r in evaluation['per_target'] if r['cad_id'] == pin['cad_id'])
    accepted = sum(r['oracle_full_cad']['registration_fitness'] == 1.
                   and r['oracle_visible_cad']['registration_fitness'] == 1. for r in diagonal)
    improved = sum(r['oracle_pose_localization_cosine'] > r['baseline_visual_raw'] for r in diagonal)
    max_mesh_rmse = max(r['point_to_exact_mesh']['rmse_mm'] for r in diagonal)
    lines = ["# Exact captured-pose diagnostic: all six objects", "",
             f"{accepted}/6 correct pairs obtain fitness 1.0 at their verified poses, with both full "
             f"and visible CAD geometry. Their continuous observed-to-mesh RMSE is at most {max_mesh_rmse:.3f}mm. "
             f"{improved}/6 correct-pair visual cosines increase relative to the 14-view localization-mask baseline.", "",
             f"For the pin, the old full-CAD registration fitness is {pin['baseline_full_cad_registration']['registration_fitness']:.6f}; at the captured "
             f"pose it is {pin['oracle_full_cad']['registration_fitness']:.6f}. Its visual cosine changes from "
             f"{pin['baseline_visual_raw']:.6f} to {pin['oracle_pose_localization_cosine']:.6f}, and to {pin['oracle_pose_same_cad_silhouette_cosine']:.6f} "
             "when both images use the verified CAD silhouette. This supports pose/view and "
             "input-consistency limitations rather than an inability of the geometric fitness formula "
             "to accept the correctly posed pin.", "",
             "Correct-pose acceptance does not establish reliable identity discrimination: "
             f"{evaluation['incorrect_pairs_with_fitness_1']} incorrect pairs still have fitness 1.0. In the oracle-aided 36-pair controls, "
             f"visual top-1 is {evaluation['accuracy']['visual']['correct']}/6 and fused top-1 is {evaluation['accuracy']['fused']['correct']}/6. "
             f"The pin's strongest incorrect visual score is {pin_ranking['visual']['strongest_incorrect_score']:.6f}, "
             f"versus {pin_ranking['visual']['correct_score']:.6f} for the pin. "
             "These are diagnostics with supplied poses, not production accuracy.", "",
             "The original simulator state was restored, and its final 2ms Euler position update "
             "was reversed. The resulting camera matrix, RGB pixels and depth values reproduce the "
             "fixed capture exactly. The six body poses are therefore verified captured poses, "
             "not placement guesses or registration estimates. Native CAD frames were independently "
             "checked against MuJoCo's compiled mesh vertices.", "",
             "Each correct CAD is rendered through the saved perspective camera at its actual "
             "position and orientation. Rendering uses the existing simple CAD lighting law and "
             "saved CAD material colors; the observation uses MuJoCo lighting. All scores retain "
             "DINOv2-small CLS, white 224x224 letterboxing, 5mm geometry voxels, 7.5mm correspondence "
             "distance and fixed 0.5/0.5 fusion. No RANSAC/ICP/FPFH or localization is rerun.", "",
             "## Correct CAD–object pairs at the verified pose", "",
             "| Object | Old max-14-view CLS | Exact-pose CLS | Old full-CAD fitness | Exact-pose full fitness | Exact-pose visible fitness | Observed→exact mesh RMSE (mm) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in diagonal:
        lines.append(f"| {library[r['cad_id']]['description']} | {r['baseline_visual_raw']:.6f} | "
            f"{r['oracle_pose_localization_cosine']:.6f} | {r['baseline_full_cad_registration']['registration_fitness']:.6f} | "
            f"{r['oracle_full_cad']['registration_fitness']:.6f} | {r['oracle_visible_cad']['registration_fitness']:.6f} | "
            f"{r['point_to_exact_mesh']['rmse_mm']:.6f} |")
    lines += ["", "Exact-pose CLS above compares the existing localization-masked observation to the "
              "entire visible silhouette of the independently shaded, perspective CAD rendering. "
              "Cosine need not equal one: masks, lighting and image sampling can differ even at the correct pose.", "",
              "## Isolate mask and appearance mismatch", "",
              "| Object | Existing localization / full CAD mask | Same localization mask on both | Same exact CAD silhouette on both | Self-image control |",
              "|---|---:|---:|---:|---:|"]
    for r in diagonal:
        lines.append(f"| {library[r['cad_id']]['description']} | {r['oracle_pose_localization_cosine']:.6f} | "
            f"{r['oracle_pose_same_localization_mask_cosine']:.6f} | {r['oracle_pose_same_cad_silhouette_cosine']:.6f} | {r['self_image_cosine']:.6f} |")
    lines += ["", "The shared-mask columns are controlled consistency tests. They reuse the same support "
              "and crop on both images and are not used for the main rankings. The oracle silhouette "
              "comes from independent first-hit CAD raycasting at the verified pose. Simulator ID-color "
              "masks were not used: exploratory ID rendering omitted visible pin pixels. The final column is an "
              "encoder arithmetic positive control, not an independent CAD test.", "",
              "## Geometric sampling and camera consistency", "",
              "| Object | Visible C_obs | Visible C_cad | Visible C_f1 | Visible point-cloud RMSE (mm) | Localization/CAD silhouette IoU | Depth MAE (mm) | Mesh RMSE after half-pixel correction (mm), diagnostic only |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in diagonal:
        g = r['oracle_visible_cad']
        lines.append(f"| {library[r['cad_id']]['description']} | {g['C_obs']:.6f} | {g['C_cad']:.6f} | {g['C_f1']:.6f} | "
            f"{g['observed_to_cad_rmse_m'] * 1000:.6f} | {r['localization_vs_oracle_silhouette_iou']:.6f} | "
            f"{r['depth_difference_pixel_centers']['mean_mm']:.6f} | "
            f"{r['half_pixel_corrected_point_to_mesh_diagnostic_only']['rmse_mm']:.6f} |")
    lines += ["", "OpenGL samples pixel centers at u+0.5,v+0.5 with the stored cx=width/2, cy=height/2. "
              "The existing localizer unprojects integer indices. The auxiliary half-pixel correction "
              "is applied only to a copy for continuous mesh-distance diagnostics; all main geometry "
              "scores use unchanged recorded clouds. Point-to-triangle distance avoids conflating "
              "surface agreement with 5mm point-sampling/voxel errors.", "",
              "Every observed-cloud self-match has fitness 1 and RMSE 0. A fixed +30mm camera-X "
              "translation of the correct full CAD provides a pose-error control:", "",
              "| Object | Fitness at correct pose | Fitness after +30mm shift | RMSE after shift (mm) |",
              "|---|---:|---:|---:|"]
    for r in diagonal:
        g = r['translated_30mm_camera_x_control']
        lines.append(f"| {library[r['cad_id']]['description']} | {r['oracle_full_cad']['registration_fitness']:.6f} | "
                     f"{g['registration_fitness']:.6f} | {g['observed_to_cad_rmse_m'] * 1000:.6f} |")
    lines += ["", "## Wrong-CAD controls at the same known candidate poses", "",
              "All 36 CAD/candidate hypotheses were evaluated. For each candidate, every CAD uses "
              "that candidate's true body frame without scaling or pose search. Thus incorrect CADs "
              "are tested at the candidate location, rather than rejected merely because their "
              "original scene locations differ. This supplies ground truth to the diagnostic and "
              "is not a deployable association benchmark. Incorrect CADs could obtain other scores "
              "if their orientations were searched independently.", "",
              "| Geometry used | Visual top-1 | Geometry top-1 | Fused top-1 | Incorrect fitness=1 pairs / 30 |",
              "|---|---:|---:|---:|---:|"]
    for name, evaluation in evaluations.items():
        acc = evaluation['accuracy']
        lines.append(f"| {name} | {acc['visual']['correct']}/6 | {acc['geometry']['correct']}/6 | "
                     f"{acc['fused']['correct']}/6 | {evaluation['incorrect_pairs_with_fitness_1']}/30 |")
    lines += ["", "Rankings are independent per CAD target; exact ties use object ID. "
              "The red and blue blocks have identical CAD geometry, so geometry cannot distinguish "
              "their identity even at perfect poses. Full rankings and correct-minus-strongest-incorrect "
              "margins for all modalities are in [evaluations.json](evaluations.json).", "",
              "## Runtime and evidence", "",
              f"Verified simulator replay: {runtime['state_recovery_and_exact_replay_time_s']:.3f}s. "
              f"36 perspective renders plus geometry and saved diagnostics: {runtime['all_36_rendering_and_fixed_pose_geometry_time_s']:.3f}s. "
              f"DINO loading: {runtime['dino_model_load_time_s']:.3f}s; feature extraction: {runtime['all_dino_feature_extraction_time_s']:.3f}s. "
              "DINO ran on CPU, one thread; MuJoCo replay used EGL. Timings include diagnostic work "
              "and file writing, not just an optimized scoring pipeline.", "",
              "[All 36 pair scores](all_pair_scores.csv), [full pair metrics/transforms](all_pairs.json), "
              "[six correct-pair diagnostics](correct_pairs.json), [captured poses](ground_truth/poses.json), "
              "[replay checks](ground_truth/replay_validation.json), [protocol](protocol.json), "
              "[validation](validation.json). Inputs, visible depths/clouds and aligned inspection PLYs "
              "are saved in `inputs/` and `diagonal/`.", "",
              "The fixed pose controls test numerical and observation consistency. They do not "
              "prove identity discrimination under unknown pose, occlusion or real sensor noise. "
              "No production changes, parameter tuning or new association thresholds were made.", "",
              "![Existing localized observation, exact-pose CAD, shared localization mask CAD, CAD-silhouette observation](input_comparison.png)", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    columns = ["Observed localization", "Exact-pose CAD", "CAD same local mask", "Observed CAD silhouette"]
    sheet = np.full((len(diagonal) * 268 + 40, 960, 3), 245, dtype=np.uint8)
    for col, label in enumerate(columns):
        cv2.putText(sheet, label, (col * 240 + 5, 24), cv2.FONT_HERSHEY_SIMPLEX, .5, (20, 20, 20), 1, cv2.LINE_AA)
    overlay = read_rgb(DATASET / "scene/rgb.png")
    for i, r in enumerate(diagonal):
        cid, oid = r['cad_id'], r['object_id']
        paths = [output / "inputs" / f"{oid}_localization.png", output / "inputs" / f"{cid}__{oid}_cad.png",
                 output / "diagonal" / cid / "cad_same_localization_mask_input.png", output / "inputs" / f"{oid}_oracle_mask.png"]
        y = 40 + 268 * i
        cv2.putText(sheet, library[cid]['description'], (5, y + 21), cv2.FONT_HERSHEY_SIMPLEX, .6, (20, 20, 20), 1, cv2.LINE_AA)
        for col, path in enumerate(paths):
            sheet[y + 32:y + 256, col * 240 + 8:col * 240 + 232] = read_rgb(path)
        mask = cv2.imread(str(output / "diagonal" / cid / "cad_mask.png"), cv2.IMREAD_GRAYSCALE)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 220, 50), 1)
    save_image(output / "input_comparison.png", sheet)
    save_image(output / "pose_overlay.png", overlay)
