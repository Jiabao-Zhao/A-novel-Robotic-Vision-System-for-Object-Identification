# GPT-6 Sol and Gemma 4 12B: Four-Scene Classification

The frozen 11-target, 11-crop, four-scene dataset is unchanged. Each model has 484 independent comparisons. The previous Qwen3.5-4B and 9B runs are reused as controls; no per-scene best-score selection. M8 and square pins remain excluded from both targets and candidate analysis.

## Method
All models receive exactly the same system/user prompt text and crop bytes. They receive no observation identity, crop ID, ROI coordinates or ground-truth metadata. All saved target descriptions are included together. No new render views, prompt changes, inference score thresholds, or forced workspace winner.
GPT-6 Sol uses OpenAI Responses, standard service, reasoning none, temperature 0, high image detail, max output 16 and top logprobs 20. Four requests run concurrently. The user explicitly approved uploading these simulation crops. Requests use store=false.
Gemma 4 12B Q4_K_M is installed in D:/AI/models and runs through the existing D:/AI/Ollama runtime, with GPU/CPU offload. Native image preprocessing and tokenizers differ across model families. Gemma permits 16 generated tokens for end/control tags; only its single A/B/C answer token is scored. Reasoning is disabled.
P(A) is exp(raw token log probability), not a written confidence and not normalized over A/B/C. It is not calibrated identity correctness. GPT may omit exact P(A); those entries remain null with conservative lower/upper bounds. A 0.0001 allowance covers rounded API log probabilities. Heatmap <= labels show upper bounds rounded upward, never zero imputation.

## USB Recovery and Evaluation
Baseline/202 USB rectangles were accepted by depth localization. Scenes 101/303 use visually verified recovered depth-cluster rectangles rejected by the aspect-ratio filter. These are assisted crops, not SAM or autonomous localization. Their pixels match the original scene rectangles exactly. Separate autonomous-depth-only statistics exclude all pairs involving those two crops.
Same-family gear, nut and round-pin sizes count as plausible alternatives, not exact instance identification. Unrelated means outside that family. Four development scenes are not held-out validation. No final model or cutoff is automatically deployed.

## Threshold Comparison
| Model | Cutoff | Intended retained / 44 | Unrelated accepted / 384 |
|---|---:|---:|---:|
| qwen3.5:4b | 0.60 | 39 | 14 |
| qwen3.5:4b | 0.65 | 35 | 12 |
| qwen3.5:4b | 0.70 | 29 | 8 |
| qwen3.5:4b | 0.90 | 2 | 0 |
| qwen3.5:4b | 0.95 | 0 | 0 |
| qwen3.5:4b | 0.98 | 0 | 0 |
| qwen3.5:9b | 0.60 | 44 | 122 |
| qwen3.5:9b | 0.65 | 44 | 116 |
| qwen3.5:9b | 0.70 | 44 | 107 |
| qwen3.5:9b | 0.90 | 44 | 53 |
| qwen3.5:9b | 0.95 | 40 | 31 |
| qwen3.5:9b | 0.98 | 28 | 5 |
| gpt-6-sol | 0.60 | 33 | 14 |
| gpt-6-sol | 0.65 | 33 | 14 |
| gpt-6-sol | 0.70 | 33 | 12 |
| gpt-6-sol | 0.90 | 28 | 7 |
| gpt-6-sol | 0.95 | 27 | 2 |
| gpt-6-sol | 0.98 | 22 | 2 |
| gemma4:12b | 0.60 | 44 | 79 |
| gemma4:12b | 0.65 | 44 | 78 |
| gemma4:12b | 0.70 | 44 | 78 |
| gemma4:12b | 0.90 | 44 | 72 |
| gemma4:12b | 0.95 | 43 | 68 |
| gemma4:12b | 0.98 | 43 | 65 |

## API Usage
321,761 input tokens; 2,420 output tokens; 0 reasoning tokens. Estimated standard API cost: $0.6860, based on saved usage, not a billing invoice.
Rates: $2/M uncached input, $0.20/M cached input, $2.50/M cache writes, $10/M output. https://developers.openai.com/api/docs/models/gpt-6-sol

## Files
- Each new model: four_scene_heatmaps.pdf, four_scene_overview.png, individual scene PNG/CSV/JSON matrices.
- Each model: all_scores, per_object_per_scene, object_summary, threshold_sweep, threshold_check, and unrelated_passes_or_possible_0_7 CSV/JSON.
- all_model_scores.csv/json and model_threshold_comparison.csv/json: combined comparisons.
- retention_vs_false_matches.png: comparison at equal unrelated-acceptance rates, using conservative bounds.
- ranking_metrics.csv/json: family-versus-unrelated AUC intervals, not exact-size correctness.
- selected_prompts.json, manifest.json, runtime.json, validation.json, responses/: complete provenance.
- results.zip: scripts, data, charts, raw responses for all complete models, and the 44 input crops. No weights included.
