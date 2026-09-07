# Frozen VLM association results for cloud review

Published at the user's request on 2026-09-07. This directory is a read-only snapshot of five local experiments, not a new inference run or a production configuration change. It contains 698 successful fresh inference responses across those experiments, their protocols, parsed scores, analysis tables, figures, and six representative visual inputs. Reused controls in comparison files are not additional independent responses.

The scripts and tests are already in the repository. Normal experiment execution writes to ignored `outputs/`, not this archive. These files are committed intentionally so GPT cloud and other reviewers can analyze the results without API keys, camera hardware, robot execution, or the full local RGB-D corpus.

## Start here

1. [GPT-5.2 findings and caveats](gpt52_explanation_pilot/FINDINGS.md), then [full report](gpt52_explanation_pilot/REPORT.md) and [endpoint audit](gpt52_explanation_pilot/API_AUDIT.json).
2. [Expanded GPT-4.1-mini explanation experiment](explanation_expansion/REPORT.md).
3. [Earlier model/temperature comparison](model_temperature_pilot/analysis/REPORT.md).
4. For independent checks, load each experiment's `records.json` and its protocol; inspect `responses/` for original provider tokens and alternatives. Use `gpt52_explanation_pilot/comparison.json` for its exactly paired mini controls.

## Experiment inventory

| Directory | Scene / response design | Model(s) | Important distinction |
| --- | --- | --- | --- |
| [model_temperature_pilot](model_temperature_pilot/analysis/REPORT.md) | 20 calibration scenes: ten targets x states 0–1; two models x three temperatures x three repetitions = 360 responses | GPT-4.1-mini and GPT-4o | Older label-only prompt, N allowed; not the newer explanation protocol |
| [known_presence_pilot](known_presence_pilot/REPORT.md) | Ten state-0 scenes x two conditions = 20 responses | GPT-4.1-mini | With-N versus known-present/no-N prompt comparison |
| [explanation_pilot](explanation_pilot/REPORT.md) | Ten state-0 scenes x three output formats = 30 responses | GPT-4.1-mini | Historical pilot includes label-only, label-then-explanation, and explanation-then-label |
| [explanation_expansion](explanation_expansion/REPORT.md) | 100 calibration scenes: ten targets x states 1–10; plus 12 additional-object captures; two formats = 224 responses | GPT-4.1-mini | Only label-only and label-then-explanation; no explanation-first calls |
| [gpt52_explanation_pilot](gpt52_explanation_pilot/FINDINGS.md) | 20 calibration scenes: ten targets x states 1–2; plus the same 12 additional-object captures; two formats = 64 responses | GPT-5.2, with saved mini controls | First successful compatibility probe is included; one rejected top-20 request is separately documented |

Pinned snapshots are `gpt-4.1-mini-2025-04-14`, `gpt-4o-2024-08-06`, and `gpt-5.2-2025-12-11`. The first experiment varies temperature (0, 0.5, 1); the remaining experiments use temperature 0. Read the exact per-experiment settings rather than pooling all rows.

## Current score and output semantics

The generated decision label selects the object. Raw decision-label likelihood is `exp(sum(original decision-bearing output-token log probabilities))`, excluding separable whitespace and explanation tokens, with no sequence-length or candidate normalization. It is not calibrated correctness probability. Provider alternatives remain diagnostics and do not replace the generated prediction.

The JSON commonly calls this field `raw_sequence_likelihood`; production calls the same quantity `association_score`. Invalid/missing decision scores remain null and count as human deferrals for operational coverage. Valid high-likelihood incorrect associations must remain in the analysis. Ground truth is evaluation-only and was not disclosed in VLM requests.

The production architecture remains semantic target -> localized-object association -> likelihood gate / human clarification -> `final_object_id` -> downstream CAD interface. This archive does not change that architecture or set a new threshold.

## Key results and cautions for GPT cloud

- Expanded mini study: label-only 98/112 correct versus label-then-explanation 100/112; pooled correctness AUROC approximately 0.717 versus 0.828. All 12 additional-object captures were correct in both conditions, so those captures cannot establish error-ranking quality on new object classes.
- On the exact 32-scene GPT-5.2 subset, the saved mini controls were 29/32 correct in both formats; GPT-5.2 was 30/32 label-only and 31/32 label-then-explanation.
- GPT-5.2 returned tiny positive decision-token logprobs in 27/64 responses. The existing parser rejected them rather than clamping their scores to one. All 27 affected predictions were correct. Valid-score availability was only 17/32 for label-only and 20/32 for explanation. The numerical cause is not established.
- GPT-5.2's actual endpoint rejected `top_logprobs=20`; five diagnostic alternatives were supported. Reasoning effort was none, and all successful responses reported zero reasoning tokens. See the preserved rejected preflight protocol/error.
- Score availability changes the attainable coverage domain and the sample used for AUROC. Do not declare GPT-5.2's lower AURC superior without accounting for that difference. In the pooled pilot, mini's high-score ties and GPT-5.2's missing scores leave no shared nonzero threshold-attainable coverage for the model comparison.
- Recurring high-score semantic errors remain. On the 32-scene explanation comparison, mini's tomato-sauce error has reported likelihood 1.0; GPT-5.2's alphabet-soup error scores about 0.99327. See error_examples.json and per_target.csv.
- In the older GPT-4o temperature-zero study, many errors were confident `N` responses for present targets (chocolate pudding, cream cheese, butter). These are false target-absence decisions, not necessarily wrong-object robot actions. That prompt differs from the later known-present/no-N experiments.
- Discussion-only thresholds such as GPT-5.2 0.994, mini 0.99999, and older GPT-4o 0.9999 were inspected post hoc on small development subsets. They are not validated final thresholds and have not been installed in production. No automated threshold selection from held-out test data occurred in these pilots.
- Counts of "would request human" are offline deferral estimates. Human correction and robot execution were not performed. Any calculation assuming perfect human correction is a hypothetical semantic-association upper bound, not measured robot task success.

## Suggested review questions

Assess whether the scores discriminate correct from incorrect associations, accounting for missing scores and tied likelihoods. Separate classification accuracy from selective-prediction quality. Identify recurring high-likelihood failures without excluding them merely because they are difficult. Evaluate which model/prompt comparisons are actually paired and which are confounded by protocol or scene differences. Recommend a validation design and error-versus-intervention objective before choosing any deployment threshold. Do not treat raw likelihood as calibrated correctness probability or assume the newer model is necessarily a better gate.

## Archive integrity and paths

- `MANIFEST.json` records relative archive paths, byte sizes, and SHA-256 hashes. Copied original artifacts are byte-identical; `.gitattributes` disables newline rewriting in this archive.
- Existing local `outputs/...`, WSL, or Windows path fields are preserved as provenance. They are not automatically resolvable paths on a cloud checkout. For raw responses, replace the original experiment root with the corresponding directory in this archive, preserving the suffix after the experiment name.
- Reports, summary JSON, and CSV permit offline analysis without loading image_path/localization_path. Do not run an experiment's `run` command to inspect these results: those commands can invoke the API if local checkpoints/inputs are absent.
- `visual_examples/` contains the exact marked images for alphabet soup states 1–2, tomato sauce states 1–2, BBQ sauce state 2, and the fresh red-mug capture from explanation_expansion. Their hashes match the original input manifest. These are representative examples, not all trial images. Full RGB-D, PLY, videos, simulator assets, caches, credentials, and the copyrighted reference-paper PDF are omitted.
- Frozen protocols and original response files are intentionally retained even where parsed comparison tables repeat some data: they permit independent verification of token-level scoring and settings.
- The earlier [500-trial archive](../vlm_confidence_500/README.md) and its [likelihood calibration report](../vlm_confidence_500/likelihood_calibration/REPORT.md) remain available separately. That older archive's root README describes its historical candidate-normalized experiment, not the current raw-likelihood production policy.

The complete dependency-independent test suite last passed 268 tests. The five GPT-5.2 pilot tests were additionally rerun with a regression case for positive provider logprobs. Resumable GPT-5.2 analysis was checked with the API function disabled and left its records byte-identical. Publishing this archive does not rerun inference.
