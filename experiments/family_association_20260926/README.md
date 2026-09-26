# Local Product-Family Association Ablation

## GitHub / GPT Cloud Entry Point

This is the **new family-association prompt/description ablation**, not the baseline.
The [unchanged four-model baseline](../vlm_four_scene_20260925/README.md) remains separately versioned.

- [Paired raw scores](paired_scores.csv), [all baseline and revised scores](all_scores_comparison.json).
- [Exact revised prompts](selected_prompts.json), [family descriptions](descriptions.json).
- [Threshold comparison](threshold_comparison.png), [family retention at matched false-match rates](family_retention_vs_false_matches.png).
- [Per-family metrics](per_family_metrics.csv), [threshold sweeps](threshold_sweep.csv), [known failures](known_failure_pairs.csv).
- [Qwen4B heatmaps](qwen3.5-4b/four_scene_heatmaps.pdf), [Qwen9B heatmaps](qwen3.5-9b/four_scene_heatmaps.pdf), [Gemma heatmaps](gemma4-12b/four_scene_heatmaps.pdf).

All 1,452 new raw responses are in the three model folders. Canonical table paths
resolve from this directory. Identical input crops and the raw baseline records
are referenced in the sibling baseline folder, not duplicated. Raw request
metadata retains its original local paths; use the canonical score row for the
portable crop/response link. `provenance/` holds original local scripts and manifests,
not portable inference entry points. No cloud/API rerun was performed.

From the repository root, run:
`python tools/verify_family_results.py experiments/family_association_20260926`

This offline check needs no models, credentials, GPU, or third-party packages.
Read the limitations below before selecting a threshold; no setting was deployed.


Three local models, four frozen scenes, 484 calls/model (1,452 new). No GPT/cloud API calls. Baseline untouched.
This jointly changes the task, labels, comparison rules and descriptions. It is not an isolated description-only effect.

## Inputs and Controls
- Same 44 crop files, 11 target IDs and 11 candidates per scene; same model digests, Ollama version and exact per-model inference controls.
- Size-family descriptions AND full request prompts are identical for all gear sizes, both nuts and all round pins. Each original target ID is still run independently for direct pairing, not supplied to the model.
- Those repeated queries are correlated, not independent evidence. Per-family metrics and unweighted macro-family AUC supplement the original 100-family/384-unrelated pair-weighted metric.
- M8 and square pins remain excluded. Two USB rectangles remain assisted recoveries (101/303); autonomous-depth-only tables exclude all comparisons against those crops, not just own-object rows.
- Waterproof CAD review_sheet.png was visually inspected: a block body with grouped round openings. Descriptions are saved verbatim in descriptions.json. No new render views or localization passes.
- P(A)=exp(raw logp_A), no renormalization or inference gate. A/B/C only. P(C) is raw token probability; C frequency counts emitted C labels, not a threshold on P(C).
- Exact unavailable probabilities remain null, with conservative bounds as in the baseline. Charts use marked upper bounds; no zero imputation.

## Aggregate Results
| Model | Condition | Family AUC | Family at .70 /100 | Unrelated at .70 /384 | Own at .70 /44 | C labels /484 |
|---|---|---:|---:|---:|---:|---:|
| qwen3.5:4b | baseline | 0.9771 | 64 | 8 | 29 | 325 |
| qwen3.5:4b | family_ablation | 0.9862 | 92 | 17 | 40 | 303 |
| qwen3.5:9b | baseline | 0.9781 | 100 | 107 | 44 | 40 |
| qwen3.5:9b | family_ablation | 0.9927 | 100 | 23 | 44 | 186 |
| gemma4:12b | baseline | 0.9720 | 100 | 78 | 44 | 0 |
| gemma4:12b | family_ablation | 0.9955 | 100 | 22 | 44 | 52 |

## Requested Failure Cases
At P(A) >= 0.70. First three cases should have fewer passes; true-nut cases should have more.
| Case | Model | Condition | Passes | Total | C labels |
|---|---|---|---:|---:|---:|
| Waterproof <- DSUB | qwen3.5:4b | baseline | 4 | 4 | 0 |
| Waterproof <- DSUB | qwen3.5:4b | family_ablation | 3 | 4 | 0 |
| Waterproof <- DSUB | qwen3.5:9b | baseline | 4 | 4 | 0 |
| Waterproof <- DSUB | qwen3.5:9b | family_ablation | 3 | 4 | 0 |
| Waterproof <- DSUB | gemma4:12b | baseline | 4 | 4 | 0 |
| Waterproof <- DSUB | gemma4:12b | family_ablation | 4 | 4 | 0 |
| DSUB <- USB | qwen3.5:4b | baseline | 2 | 4 | 0 |
| DSUB <- USB | qwen3.5:4b | family_ablation | 4 | 4 | 0 |
| DSUB <- USB | qwen3.5:9b | baseline | 4 | 4 | 0 |
| DSUB <- USB | qwen3.5:9b | family_ablation | 4 | 4 | 0 |
| DSUB <- USB | gemma4:12b | baseline | 2 | 4 | 0 |
| DSUB <- USB | gemma4:12b | family_ablation | 4 | 4 | 0 |
| Gear <- hex nut | qwen3.5:4b | baseline | 0 | 24 | 13 |
| Gear <- hex nut | qwen3.5:4b | family_ablation | 9 | 24 | 3 |
| Gear <- hex nut | qwen3.5:9b | baseline | 24 | 24 | 0 |
| Gear <- hex nut | qwen3.5:9b | family_ablation | 9 | 24 | 6 |
| Gear <- hex nut | gemma4:12b | baseline | 24 | 24 | 0 |
| Gear <- hex nut | gemma4:12b | family_ablation | 3 | 24 | 0 |
| True nut family crops | qwen3.5:4b | baseline | 3 | 16 | 6 |
| True nut family crops | qwen3.5:4b | family_ablation | 8 | 16 | 0 |
| True nut family crops | qwen3.5:9b | baseline | 16 | 16 | 0 |
| True nut family crops | qwen3.5:9b | family_ablation | 16 | 16 | 0 |
| True nut family crops | gemma4:12b | baseline | 16 | 16 | 0 |
| True nut family crops | gemma4:12b | family_ablation | 16 | 16 | 0 |

## Repeated-Query Diagnostic
Identical size-family prompts for the same crop had a maximum P(A) spread of 0.053744; 0 groups disagreed on the emitted A/B/C label. These are separately executed calls, not copied scores. The cause of likelihood variation was not isolated; fixed settings do not establish bitwise reproducibility. See repeated_family_query_consistency.csv.

## Insufficient Evidence
Mean raw P(C) and emitted-C frequency are different quantities. Family and unrelated pairs are separated below.
| Model | Condition | Family mean P(C) | Family C /100 | Unrelated mean P(C) | Unrelated C /384 |
|---|---|---:|---:|---:|---:|
| qwen3.5:4b | baseline | 0.190526 | 6 | 0.613643 | 319 |
| qwen3.5:4b | family_ablation | 0.0835702 | 0 | 0.624814 | 303 |
| qwen3.5:9b | baseline | 0.0142323 | 0 | 0.240442 | 40 |
| qwen3.5:9b | family_ablation | 0.0214386 | 0 | 0.498241 | 186 |
| gemma4:12b | baseline | 1.21733e-07 | 0 | 6.99585e-05-7.25627e-05 | 0 |
| gemma4:12b | family_ablation | 1.40285e-05 | 0 | 0.141907 | 52 |

## Per-Family Retention
Counts at .70, baseline -> revised. Denominators include repeated size-family query IDs, not independent physical trials.
| Model | Target family | Family retained | Family pairs | Unrelated accepted | Unrelated pairs |
|---|---|---:|---:|---:|---:|
| qwen3.5:4b | gear | 23 -> 36 | 36 | 0 -> 9 | 96 |
| qwen3.5:4b | hex_nut | 3 -> 8 | 16 | 0 -> 0 | 72 |
| qwen3.5:4b | round_pin | 27 -> 36 | 36 | 0 -> 0 | 96 |
| qwen3.5:4b | Waterproof_Male | 3 -> 4 | 4 | 6 -> 4 | 40 |
| qwen3.5:4b | USB_Male | 4 -> 4 | 4 | 0 -> 0 | 40 |
| qwen3.5:4b | DSUB_Male | 4 -> 4 | 4 | 2 -> 4 | 40 |
| qwen3.5:9b | gear | 36 -> 36 | 36 | 24 -> 9 | 96 |
| qwen3.5:9b | hex_nut | 16 -> 16 | 16 | 0 -> 0 | 72 |
| qwen3.5:9b | round_pin | 36 -> 36 | 36 | 58 -> 3 | 96 |
| qwen3.5:9b | Waterproof_Male | 4 -> 4 | 4 | 16 -> 5 | 40 |
| qwen3.5:9b | USB_Male | 4 -> 4 | 4 | 0 -> 0 | 40 |
| qwen3.5:9b | DSUB_Male | 4 -> 4 | 4 | 9 -> 6 | 40 |
| gemma4:12b | gear | 36 -> 36 | 36 | 31 -> 3 | 96 |
| gemma4:12b | hex_nut | 16 -> 16 | 16 | 0 -> 0 | 72 |
| gemma4:12b | round_pin | 36 -> 36 | 36 | 28 -> 0 | 96 |
| gemma4:12b | Waterproof_Male | 4 -> 4 | 4 | 14 -> 8 | 40 |
| gemma4:12b | USB_Male | 4 -> 4 | 4 | 2 -> 1 | 40 |
| gemma4:12b | DSUB_Male | 4 -> 4 | 4 | 3 -> 10 | 40 |

## Files
- selected_prompts.json, descriptions.json, manifest.json: exact prompts, provenance and preserved baseline hashes.
- all_scores_comparison.csv/json and paired_scores.csv/json: all baseline and revised scores, including P(A), P(B), P(C), labels, input hashes and raw-response paths.
- metrics.csv/json: family AUC, macro-family AUC, P(C) means/bounds, C frequency by family/unrelated/own and scene.
- baseline_comparison.csv/json: directly paired aggregate AUC changes and family/unrelated C counts.
- development_operating_points.csv/json: matched false-match budgets with development-selected cutoffs. Optimistic analysis, NOT validated thresholds.
- threshold_sweep.csv/json: every .01 cutoff plus high-score checks, with family/unrelated/own retained counts and rates.
- per_family_metrics.csv/json, per_family_thresholds.csv/json: each target family separately.
- ranked_unrelated_pairs.csv/json: all unrelated pairs sorted within each model/condition, not merely those above .70.
- top_unrelated_pairs.csv/json: the top 20 unrelated pairs per model/condition, retaining repeated size-family queries explicitly.
- known_failures.csv/json and known_failure_pairs.csv/json: requested confusion directions, paired raw scores and emitted labels.
- repeated_family_query_consistency.csv/json: diagnostic score spread across identical size-family requests.
- Model folders: raw responses, exact runtime metadata, per-scene heatmaps and four-page PDF.

## Interpretation Limits
These are reused development scenes, not a held-out validation set. Prompt changes were designed with knowledge of baseline failures. Do not claim calibrated identity correctness, statistical superiority, or a deployment-ready gate. Compare thresholds at matched false-match rates, not just at a universal .70. Own-object retention is secondary and does not imply exact-size identification.
The earlier name-and-color-only proposal was not run. This family ablation is a different, explicitly requested experiment. No model or threshold is deployed automatically.
