# Four-Scene Local Qwen Classification Trial

11 selected target objects, each compared independently with all 11 retained object crops in each of four fixed scenes. No new scene views. M8 and all square pins excluded from targets and candidate analysis.

## Reproducibility
Exact previous selected descriptions and prompts are frozen in selected_prompts.json. The USB query retains its existing shadow-aware paragraph. No description edits or inference score threshold. One image and all saved target descriptions per request. Ground-truth labels are not in model inputs.
Qwen3.5 9B Q4_K_M is saved on D: via Ollama. Runtime versions, model digests, inference options, raw requests/responses and log probabilities are saved under each model folder. Thinking disabled; one greedy output token A/B/C.

## USB Fallback
Depth accepted the cable in baseline and scene 202. In scenes 101/303 its complete cluster was rejected by the fixed aspect-ratio filter. We recovered the existing native RGB rectangles, visually verified them, and verified pixel-for-pixel equality with the corresponding scene crop. Simulation reference boxes assisted identification of those rejected clusters. This is explicitly assisted localization, not SAM or autonomous success. No inpainting, new view, or image enhancement.

## Score Meaning
P(A) = exp(raw model log probability of token A). It is not renormalized over A/B/C and is not a calibrated probability of correct object identity. A means match supported, B mismatch supported, C insufficient evidence. All three probabilities and raw logs are retained.
Gear sizes, hex-nut sizes, and round-pin sizes form separate product families. Same-family alternatives are plausible matches, not proof of exact size identity. An unrelated acceptance means a crop outside the target family passes the cutoff.

## Threshold Results
| Model | Cutoff | Intended retained / 44 | Unrelated retained / 384 |
|---|---:|---:|---:|
| qwen3.5:4b | 0.60 | 39 | 14 |
| qwen3.5:4b | 0.65 | 35 | 12 |
| qwen3.5:4b | 0.70 | 29 | 8 |
| qwen3.5:4b | 0.80 | 15 | 2 |
| qwen3.5:4b | 0.90 | 2 | 0 |
| qwen3.5:4b | 0.95 | 0 | 0 |
| qwen3.5:4b | 0.97 | 0 | 0 |
| qwen3.5:4b | 0.98 | 0 | 0 |
| qwen3.5:4b | 0.99 | 0 | 0 |
| qwen3.5:9b | 0.60 | 44 | 122 |
| qwen3.5:9b | 0.65 | 44 | 116 |
| qwen3.5:9b | 0.70 | 44 | 107 |
| qwen3.5:9b | 0.80 | 44 | 87 |
| qwen3.5:9b | 0.90 | 44 | 53 |
| qwen3.5:9b | 0.95 | 40 | 31 |
| qwen3.5:9b | 0.97 | 38 | 19 |
| qwen3.5:9b | 0.98 | 28 | 5 |
| qwen3.5:9b | 0.99 | 5 | 0 |

Threshold sweeps include both all crops and autonomous-depth-only results; the latter excludes every pair involving either assisted USB crop. The four reused development scenes are not an independent test set. Do not select a final deployment threshold from these data alone.

If both model folders are present, the 4B control was freshly rerun on the same 44 image files and exact prompts/options as 9B. paired_model_scores.csv verifies the comparison. Earlier 4B chart data mixed scene revisions and are not used as this matched control.

## Files
- 9b/four_scene_overview.png and four_scene_heatmaps.pdf: all 484 values across four scenes.
- 9b/*_p_A_matrix.png/csv/json: individual scene matrices.
- Each model: all_scores, per_object_per_scene, object_summary, threshold_sweep, threshold_check, unrelated_passes_0_7 (CSV and JSON).
- model_threshold_comparison.csv/json: selected cutoffs, with and without assisted crops.
- paired_model_scores.csv/json and threshold_comparison.png: matched model comparison when both runs are complete.
- retention_vs_false_matches.png and ranking_metrics.csv/json: threshold-independent comparison; same-family alternatives count as compatible.
- manifest.json, selected_prompts.json, validation.json and responses: full audit trail.
- results.zip includes all data, charts, raw responses and the 44 input crops; no model weights.
