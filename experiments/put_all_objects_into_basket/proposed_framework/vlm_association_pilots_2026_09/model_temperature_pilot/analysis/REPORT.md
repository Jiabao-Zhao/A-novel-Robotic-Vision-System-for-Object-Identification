# VLM model / temperature pilot

20 fixed LIBERO calibration scenes (initial states 0 and 1 from each class). Two pinned models, three temperatures, three repeats: 360 fresh responses, not 360 independent scenes. Images, prompt text, target descriptions, candidate order, ground truth, and output representation are frozen from the compact-label ablation. No human corrections, CAD registration, planning, or robot execution. No test-set use or threshold selection; all curves are exploratory calibration-subset results.

Score: exp(sum of provider log probabilities for generated decision-bearing tokens). Generated labels determine predictions; alternatives never change scores or predictions.

| Model | T | Accuracy | Mean score | Median score | Exactly 1 | Wrong >=.99 | AUROC | AURC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt-4.1-mini-2025-04-14 | 0 | 48/60 (80.0%) | 0.997478 | 1 | 33 | 8 | 0.953993 | 0.0308942 |
| gpt-4.1-mini-2025-04-14 | 0.5 | 48/60 (80.0%) | 0.997475 | 1 | 34 | 8 | 0.948785 | 0.0325772 |
| gpt-4.1-mini-2025-04-14 | 1 | 48/60 (80.0%) | 0.996067 | 1 | 34 | 7 | 0.948785 | 0.0325565 |
| gpt-4o-2024-08-06 | 0 | 42/60 (70.0%) | 0.905332 | 0.997795 | 0 | 9 | 0.691799 | 0.155009 |
| gpt-4o-2024-08-06 | 0.5 | 43/60 (71.7%) | 0.900963 | 0.997795 | 0 | 9 | 0.690834 | 0.14256 |
| gpt-4o-2024-08-06 | 1 | 39/60 (65.0%) | 0.821048 | 0.997586 | 0 | 9 | 0.764347 | 0.161541 |

## Score range and saturation

| Model | T | Minimum | 10th percentile | Median | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| gpt-4.1-mini-2025-04-14 | 0 | 0.948880498 | 0.999921445 | 1.000000000 | 1.000000000 |
| gpt-4.1-mini-2025-04-14 | 0.5 | 0.948880498 | 0.999924329 | 1.000000000 | 1.000000000 |
| gpt-4.1-mini-2025-04-14 | 1 | 0.939907989 | 0.999739266 | 1.000000000 | 1.000000000 |
| gpt-4o-2024-08-06 | 0 | 0.498421928 | 0.627241229 | 0.997794884 | 0.999992893 |
| gpt-4o-2024-08-06 | 0.5 | 0.287239400 | 0.668752572 | 0.997794884 | 0.999997422 |
| gpt-4o-2024-08-06 | 1 | 0.004905393 | 0.300672003 | 0.997586238 | 0.999989197 |

## Same-token temperature diagnostic

Compare exact token text at output position zero, using only alternatives actually returned in both calls. Whitespace-prefixed variants remain distinct; a later token after an emitted newline is not treated as position zero. Missing top-k alternatives are not zero-filled. same_first_token_logprobs.csv also includes repeated-temperature controls.

For T=.5 vs T=1, compare the log odds of the same two distinct semantic labels at position zero. If logits were fixed and returned probabilities temperature-scaled, the gap ratio would be 2; unchanged returned gaps would give 1. Hosted-model variations prevent interpreting this as an API implementation guarantee. Missing pairs are excluded with availability recorded.

| Model | Available pairs | Median gap ratio (.5 / 1) | Median error: unchanged gap | Median error: doubled gap |
| --- | ---: | ---: | ---: | ---: |
| gpt-4.1-mini-2025-04-14 | 60/60 | 1 | 0.749999 | 18.75 |
| gpt-4o-2024-08-06 | 59/60 | 1 | 0.125 | 6.21875 |

Same-token variability control below reports the median absolute logprob change over common exact token entries (including low-probability alternatives). These are descriptive matched-token summaries, not independent statistical observations.

| Comparison | Matched token entries | Median absolute logprob change |
| --- | ---: | ---: |
| gpt-4.1-mini-2025-04-14: T0 -> T0.5 | 499 | 0.25 |
| gpt-4.1-mini-2025-04-14: T0 -> T1 | 496 | 0.374323 |
| gpt-4.1-mini-2025-04-14: repeated T0 | 332 | 0.375 |
| gpt-4.1-mini-2025-04-14: repeated T0.5 | 326 | 0.374995 |
| gpt-4.1-mini-2025-04-14: repeated T1 | 323 | 0.375 |
| gpt-4o-2024-08-06: T0 -> T0.5 | 558 | 0.046874 |
| gpt-4o-2024-08-06: T0 -> T1 | 559 | 0.0659685 |
| gpt-4o-2024-08-06: repeated T0 | 370 | 0.234359 |
| gpt-4o-2024-08-06: repeated T0.5 | 372 | 0.0843463 |
| gpt-4o-2024-08-06: repeated T1 | 374 | 0.0619855 |

## Paired changes (B minus A)

Accuracy intervals use a paired bootstrap over scenes, preserving repeated calls as a cluster. They remain exploratory and do not quantify transfer to new object classes.

| Comparison | Changed predictions | Accuracy delta | Scene-bootstrap 95% interval |
| --- | ---: | ---: | --- |
| gpt-4.1-mini-2025-04-14: T0 -> T0.5 | 0/60 | +0.0% | [0.0, 0.0] |
| gpt-4.1-mini-2025-04-14: T0 -> T1 | 1/60 | +0.0% | [0.0, 0.0] |
| gpt-4.1-mini-2025-04-14: repeated T0 | 0/40 | +0.0% | [0.0, 0.0] |
| gpt-4.1-mini-2025-04-14: repeated T0.5 | 0/40 | +0.0% | [0.0, 0.0] |
| gpt-4.1-mini-2025-04-14: repeated T1 | 1/40 | +0.0% | [0.0, 0.0] |
| gpt-4o-2024-08-06: T0 -> T0.5 | 5/60 | +1.7% | [-0.03333333333333333, 0.06666666666666667] |
| gpt-4o-2024-08-06: T0 -> T1 | 9/60 | -5.0% | [-0.18333333333333335, 0.06666666666666667] |
| gpt-4o-2024-08-06: repeated T0 | 6/40 | +15.0% | [0.0, 0.3] |
| gpt-4o-2024-08-06: repeated T0.5 | 3/40 | +2.5% | [-0.075, 0.15] |
| gpt-4o-2024-08-06: repeated T1 | 6/40 | +15.0% | [0.05, 0.275] |
| model: mini -> 4o at T0 | 24/60 | -10.0% | [-0.33333333333333326, 0.11666666666666667] |
| model: mini -> 4o at T0.5 | 23/60 | -8.3% | [-0.31666666666666665, 0.13333333333333336] |
| model: mini -> 4o at T1 | 29/60 | -15.0% | [-0.41666666666666663, 0.1] |

## Descriptive thresholds (not selected operating points)

| Model | T | Threshold | Coverage | Autonomous accuracy | False accepts / all calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| gpt-4.1-mini-2025-04-14 | 0 | 0.7 | 100.0% | 0.8 | 12/60 |
| gpt-4.1-mini-2025-04-14 | 0 | 0.9 | 100.0% | 0.8 | 12/60 |
| gpt-4.1-mini-2025-04-14 | 0 | 0.95 | 98.3% | 0.813559 | 11/60 |
| gpt-4.1-mini-2025-04-14 | 0 | 0.99 | 93.3% | 0.857143 | 8/60 |
| gpt-4.1-mini-2025-04-14 | 0.5 | 0.7 | 100.0% | 0.8 | 12/60 |
| gpt-4.1-mini-2025-04-14 | 0.5 | 0.9 | 100.0% | 0.8 | 12/60 |
| gpt-4.1-mini-2025-04-14 | 0.5 | 0.95 | 98.3% | 0.813559 | 11/60 |
| gpt-4.1-mini-2025-04-14 | 0.5 | 0.99 | 93.3% | 0.857143 | 8/60 |
| gpt-4.1-mini-2025-04-14 | 1 | 0.7 | 100.0% | 0.8 | 12/60 |
| gpt-4.1-mini-2025-04-14 | 1 | 0.9 | 100.0% | 0.8 | 12/60 |
| gpt-4.1-mini-2025-04-14 | 1 | 0.95 | 95.0% | 0.842105 | 9/60 |
| gpt-4.1-mini-2025-04-14 | 1 | 0.99 | 91.7% | 0.872727 | 7/60 |
| gpt-4o-2024-08-06 | 0 | 0.7 | 81.7% | 0.693878 | 15/60 |
| gpt-4o-2024-08-06 | 0 | 0.9 | 73.3% | 0.75 | 11/60 |
| gpt-4o-2024-08-06 | 0 | 0.95 | 70.0% | 0.785714 | 9/60 |
| gpt-4o-2024-08-06 | 0 | 0.99 | 66.7% | 0.775 | 9/60 |
| gpt-4o-2024-08-06 | 0.5 | 0.7 | 85.0% | 0.705882 | 15/60 |
| gpt-4o-2024-08-06 | 0.5 | 0.9 | 73.3% | 0.75 | 11/60 |
| gpt-4o-2024-08-06 | 0.5 | 0.95 | 70.0% | 0.785714 | 9/60 |
| gpt-4o-2024-08-06 | 0.5 | 0.99 | 65.0% | 0.769231 | 9/60 |
| gpt-4o-2024-08-06 | 1 | 0.7 | 75.0% | 0.711111 | 13/60 |
| gpt-4o-2024-08-06 | 1 | 0.9 | 68.3% | 0.780488 | 9/60 |
| gpt-4o-2024-08-06 | 1 | 0.95 | 68.3% | 0.780488 | 9/60 |
| gpt-4o-2024-08-06 | 1 | 0.99 | 63.3% | 0.763158 | 9/60 |

## Limitations and audit

A larger score range is not evidence of better correctness ranking. AUROC is undefined if only one correctness outcome occurs. AURC uses tied threshold endpoints and trapezoidal interpolation from an empty-coverage origin; interpolated coverage inside a tied group is not achievable by a threshold. Unavailable scores count as deferred. No class-conditional score adjustment is used. No absent-target scenes are present. The paper did not specify its GPT-4o snapshot or temperature: this is not an exact replication. The first scene checks every condition before bulk work. Calls are interleaved and raw fingerprints/timestamps retained; temperature-zero and cross-condition variation can include server-side numerical differences. No production settings changed.

Audit: {"calls": 360, "scenes": 20, "all_inputs_match_frozen_prompts_and_images": true, "finish_reasons": {"stop": 360}, "fingerprints": {"gpt-4.1-mini-2025-04-14": {"fp_e2d71a067d": 175, "fp_45ff40c11b": 5}, "gpt-4o-2024-08-06": {"fp_ffd8308b42": 180}}, "invalid_choices": 0, "unavailable_scores": 0, "full_vs_decision_differences": 0, "usage": {"gpt-4.1-mini-2025-04-14": {"prompt_tokens": 597915, "completion_tokens": 180}, "gpt-4o-2024-08-06": {"prompt_tokens": 339615, "completion_tokens": 180}}}

Run: `python -m scripts.run_vlm_model_temperature_pilot run` (resumes saved calls). Analyze without inference: `python -m scripts.analyze_vlm_model_temperature_pilot`.

Official references checked September 6, 2026: [Chat Completions parameters](https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create), [GPT-4o snapshots](https://developers.openai.com/api/docs/models/gpt-4o), [GPT-4.1-mini snapshot](https://developers.openai.com/api/docs/models/gpt-4.1-mini). The API reference describes temperature as a sampling parameter but does not explicitly specify pre/post-temperature semantics for returned logprobs. The gap comparison here tests observed behavior in this run, not a documented universal guarantee.
