# Four-Scene VLM Classification: Cloud Analysis Handoff

Status: completed results packaged on 2026-09-25. Start here for the latest
classification study. The repository's older SAM3/DINO branch protocol is
historical context, not the method used to produce these scores.

## Research Question

Can a vision-language model associate an observed object crop with a requested
CAD product, and can its raw answer-token probability support a useful gate for
human-robot collaboration? Retaining several members of the correct product
family can be useful: the operator can clarify which instance is intended.
Rejecting the intended object before clarification is also a failure.

The earlier pipeline used CAD-view descriptions to prompt SAM3, followed by
DINOv2 global/patch matching and projected CAD mask/depth verification. This
study instead uses saved **depth-localization RGB rectangles** and a VLM as the
association module. SAM3, DINOv2 and the geometric combined score are NOT used
in these 1,936 comparisons. No robot action or live human dialogue was tested.

## Start With These Files

- [All 1,936 scores, CSV](data/all_model_scores.csv) and [JSON](data/all_model_scores.json).
- [Exact system/user prompts for each target](selected_prompts.json).
- [Portable manifest](manifest.json): targets, crop IDs, input hashes and scene paths.
- [Threshold comparison](data/model_threshold_comparison.csv).
- [Ranking metrics](data/ranking_metrics.csv).
- [Threshold curves](charts/threshold_comparison.png) and [equal-error comparison](charts/retention_vs_false_matches.png).
- [GPT-6 Sol charts](models/gpt6-sol/four_scene_heatmaps.pdf).
- [Gemma charts](models/gemma4-12b/four_scene_heatmaps.pdf).
- [Qwen4B charts](models/qwen3.5-4b/four_scene_heatmaps.pdf).
- [Qwen9B charts](models/qwen3.5-9b/four_scene_heatmaps.pdf).
- [Offline verification/summary script](verify_results.py).

Each canonical score row links to its exact crop and byte-preserved raw request/
response using paths relative to this directory. `raw/` contains all 1,936 raw
records; `input_crops/` contains all 44 input images. `scenes/` contains the four
full wrist RGB images, previews, scene metadata and camera calibration.
Ground-truth scene metadata is evaluation-only; never give it to the classifier.

`provenance/` retains the original manifests, runtime settings, analysis scripts,
localization configuration and audit files. Original records can contain old
Windows/WSL paths. `file_index.json` maps packaged files back to their sources.
Only the canonical tables and portable manifest are rewritten for portability;
raw responses are unchanged. Do not assume an original local path exists in cloud.

## Dataset and Inputs

- Four fixed scenes: `original` (baseline), `state_101`, `state_202`, `state_303`.
- 11 target queries x 11 object crops x 4 scenes = 484 calls per model.
- The physical scenes contain 14 objects. Three square pins remain physically
  present but are excluded from both target queries and candidate analysis.
  M8 is absent. Drills and glue bottle are absent from this current scene.
- Targets: large/medium/small gears; waterproof male connector; M12/M16 hex
  nuts; 16/12/8 mm round pins; USB male connector with cable; DSUB male connector.
- One crop is independently compared with one target query per call. All saved
  descriptions for that target are supplied together, not one per call.
- All four models receive identical prompt text and image bytes. No per-scene
  best-score selection, new camera views, re-rendering or inference cutoff.
- Crop identity, simulator poses and ROI coordinates are not supplied to the VLM.
- Target descriptions include earlier VLM outputs and user edits, not exclusively
  machine-generated captions. Some targets use everyday-name revisions; USB
  retains a shadow-aware paragraph. Read the exact per-target prompts.
- The matched Qwen4B control is a fresh matched run, not an older mixed-revision
  chart. Qwen4B/9B results were reused unchanged for the later GPT/Gemma comparison.

## Localization Qualification

Depth localization accepted USB in baseline and scene 202. In 101 and 303 its
cluster was rejected by the fixed aspect-ratio filter (limit 16). Existing RGB
rectangles were recovered with simulator/reference-box assistance and visually
checked, with pixel equality verified against the source image. These are
**assisted recovered depth rectangles**, not SAM3 masks or autonomous localization
successes. They contain the cable as well as the connector. The cable geometry is
a procedural addition, not a measured replica of the physical cable.

Per-model threshold tables include `all_crops` and `autonomous_depth_only` scopes.
The latter removes every pair involving either assisted crop (22 calls/model),
leaving 462 pairs and 42 own-object pairs. It is an exclusion analysis, not a claim
that the autonomous pipeline found the missing USB objects. Depth arrays and CAD
weights are not in this package; it supports classification analysis without
re-running simulation/localization.

## Models and Scoring

| Model | Backend | Reasoning | Output budget |
|---|---|---|---:|
| Qwen3.5-4B, 4-bit | Ollama | disabled | 1 token |
| Qwen3.5-9B, Q4_K_M | Ollama | disabled | 1 token |
| GPT-6 Sol | OpenAI Responses | none | 16 tokens |
| Gemma 4 12B, Q4_K_M | Ollama | disabled | 16 tokens |

Local options: temperature 0, seed 0, context 4096, repeat penalty 1,
presence/frequency penalties 0, top_k 0, top_p 1, min_p 0; top logprobs 20.
GPT: temperature 0, high image detail, top logprobs 20, store=false, default service
tier; four concurrent requests. Native image processing, model templates,
tokenization and quantization differ. The output budget accommodates control/end
tokens; only a single visible A/B/C label is scored, with no written confidence.

```text
A = MATCH SUPPORTED
B = MISMATCH SUPPORTED
C = INSUFFICIENT EVIDENCE
P(A) = exp(raw model log probability of answer token A)
accept candidate at cutoff t when P(A) >= t
```

P(A) is NOT renormalized over A/B/C, not a SAM score, not a combined geometric
score, and not a calibrated probability of correct identity. An A response can
have P(A) below a chosen threshold. Scores for different candidates do not sum
to one. There is no forced winner or automatic execution decision.

GPT omitted A from returned alternatives in 327/484 calls. Their exact P(A) stays
null. The lower bound is zero; the upper bound is the smaller of unreported
probability mass and the smallest reported alternative probability, with a
0.0001 rounding allowance, capped at one. Charts mark upper bounds with `<=`,
rounded upward. The largest missing upper bound is below 0.02, so counts at
0.60/0.65/0.70 are unambiguous. All 44 GPT own-object probabilities are available.
Do not replace missing values with zero or infer fine low-cutoff rankings from them.

## What Counts as Correct?

- `own_object_match`: crop identity equals the requested CAD identity (44/model).
- `same_family`: any gear for a gear query, either nut for a nut query, or any
  round pin for a round-pin query. Other connector types are separate families.
  There are 100 family-compatible pairs/model, including the 44 own-object pairs.
- `unrelated`: outside the target's family (384/model). Same-family alternatives
  are not unrelated false matches, but are NOT proof of exact size identification.
- `intended retained`: own-object P(A) passes the threshold, regardless of what
  other crops pass. It does not mean the system selected the correct unique winner.

These are post-hoc object labels, not segmentation-ground-truth metrics. Crops
are individually resized and have no metric scale supplied to the model; a name
such as '12 mm round pin' does not make absolute size observable.

## Main Results

| Model | Cutoff | Own retained /44 | Family retained /100 | Unrelated accepted /384 |
|---|---:|---:|---:|---:|
| Qwen3.5-4B | 0.60 | 39 | 88 | 14 |
| Qwen3.5-4B | 0.70 | 29 | 64 | 8 |
| Qwen3.5-9B | 0.70 | 44 | 100 | 107 |
| Qwen3.5-9B | 0.98 | 28 | 64 | 5 |
| GPT-6 Sol | 0.70 | 33 | 81 | 12 |
| GPT-6 Sol | 0.95 | 27 | 67 | 2 |
| Gemma 4 12B | 0.70 | 44 | 100 | 78 |
| Gemma 4 12B | 0.98 | 43 | 98 | 65 |

The same numerical cutoff is not an equal operating point across models. Qwen9B
and Gemma give many unrelated candidates very high scores. This does not alone
establish worse ranking ability. Family-versus-unrelated AUC: Qwen4B 0.9771;
Qwen9B 0.9781; GPT bounded 0.9729-0.9814; Gemma 0.9720. These tiny differences
are descriptive, not statistically established superiority.

GPT retains all four large-gear, medium-gear and each round-pin own-object cases
at 0.70. It retains 2/4 each for small gear, waterproof connector, M12, M16 and
DSUB, and 3/4 USB. Inspect per-object/per-scene tables before using averages.

GPT usage: 321,761 input tokens and 2,420 output tokens, zero reasoning tokens;
estimated standard cost $0.686011, not a billing invoice. Local hardware was
RTX 3060 Ti (8 GB VRAM) and approximately 32 GB RAM. Gemma used GPU/CPU offload.
Summed request durations are not end-to-end wall time, especially for parallel
GPT requests. Check runtime records before making speed claims.

## Limitations and Cloud Analysis Request

These are four reused development scenes, not held-out benchmark data. The
queries, crops and product-family pairs are correlated. Prompt choices were
influenced by prior experiments on these scenes. No repeated stochastic trials,
calibration fitting, real-robot test, or independent test-set evaluation is present.
Excluding difficult square pins means this is not accuracy across all products.
Reference color is assigned simulation appearance, not evidence of a physical
material specification in STL files.

Please independently analyze the raw results, rather than merely endorse the
summary. Compare model-specific operating points at matched unrelated-acceptance
rates, with own and family retention reported separately. Inspect scene-level
failures, confusion between families, prompt wording, and assisted USB sensitivity.
If tuning thresholds, distinguish development selection from held-out evaluation;
avoid treating 1,936 correlated pair scores as independent scene trials. Suggest
an operator-clarification policy and the next controlled experiment, including
what to do when no candidates or multiple candidates pass. Do not silently modify
the locked prompts or claim a deployment-safe threshold.

**Proposed, NOT RUN:** replace multi-view descriptions with the catalog name and
verified render color (for example '12 mm round pin; gray'). Metadata was inspected,
but no new prompt files, API calls or results were generated before interruption.
This package contains no evidence for that proposed ablation. No final model or
threshold has been deployed based on this comparison.

## Offline Verification

From this directory, run `python verify_results.py` (Python 3 standard library
only; no credentials, network, models or GPU required). It checks all packaged
hashes, all 1,936 raw label probabilities and request prompts, crop associations,
evaluation flags and the main threshold counts. It prints reproducible summaries.
Original local inference/plotting scripts are preserved for provenance; they are
not portable execution entry points and may depend on omitted local runtimes.
