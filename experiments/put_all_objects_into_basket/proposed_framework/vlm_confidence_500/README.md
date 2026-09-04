# LIBERO VLM confidence experiment

This package contains the compact, publication-safe results of the proposed
framework's semantic object-association experiment. It is intended for review
and analysis without requiring the 1.06 GB local RGB-D capture directory.

## Protocol

- Benchmark: `LIBERO-Object`
- Model: `gpt-4.1-mini-2025-04-14`
- Trials: 10 semantic targets x 50 official fixed initial states = 500
- Visual input: 768 x 768 agent-view rendering plus a contact sheet containing
  the full scene and seven depth-localized candidate crops
- Choices: deterministic labels `A` through `G`, plus `N` for target absent
- Split: states 0-29 calibration, 30-39 validation, and 40-49 test
- Human clarification, CAD registration, planning, and robot execution were not
  run. This experiment isolates raw VLM association behavior.

Simulator object position was used only after localization to assign an
evaluation ground-truth object ID. It was never included in the image,
localization JSON, prompt, or VLM candidate metadata. All 500 targets matched a
localized cluster; the maximum ground-truth-to-cluster XY distance was 19.3 mm.

## Score definition

For every valid choice label with a returned log probability `l_j`, the
candidate-relative distribution is

```text
q_j = exp(l_j) / sum_k exp(l_k)
```

`association_score` is the largest `q_j`. It is available only when the API
returns log probabilities for every valid label `A`-`G` and `N`; otherwise it
is `null` and the trial must defer to a human. The score is not calibrated and
must not be interpreted as probability of correct identity.

## Main results

- Correct associations: 402/500 (80.4%)
- Complete candidate-normalized scores: 181/500 (36.2%)
- Score unavailable: 319/500 (63.8%)
- Accuracy within scored trials: 123/181 (68.0%)
- Accuracy within score-unavailable trials: 279/319 (87.5%)

| Target | Correct | Accuracy | Score available |
| --- | ---: | ---: | ---: |
| alphabet soup | 9/50 | 18% | 84% |
| cream cheese | 49/50 | 98% | 76% |
| salad dressing | 50/50 | 100% | 22% |
| bbq sauce | 50/50 | 100% | 0% |
| ketchup | 50/50 | 100% | 4% |
| tomato sauce | 0/50 | 0% | 28% |
| butter | 50/50 | 100% | 10% |
| milk | 50/50 | 100% | 36% |
| chocolate pudding | 44/50 | 88% | 92% |
| orange juice | 50/50 | 100% | 10% |

Score-unavailable trials are counted as human deferrals in whole-system
coverage:

| Threshold | Autonomous coverage | Autonomous accuracy | False acceptances |
| ---: | ---: | ---: | ---: |
| 0.75 | 33.2% | 71.1% | 48 |
| 0.95 | 31.0% | 75.5% | 38 |
| 0.9999 | 21.4% | 89.7% | 11 |
| 0.99999 | 18.2% | 94.5% | 5 |
| 0.999999 | 15.6% | 96.2% | 3 |
| 0.9999999 | 10.6% | 100% observed | 0 |

The last row is an observation on this dataset, not a calibrated operating
threshold or a guarantee of future accuracy. No threshold was selected from
the final test data.

## Files

- `dataset_manifest.json`: 500 sample IDs, partitions, target descriptions,
  fixed-state hashes, and evaluation labels
- `association_records.json`: full per-trial predictions, raw label likelihood,
  candidate scores when complete, margins, and correctness
- `association_records.csv`: compact analysis table
- `summary.json`: run-level counts and provenance
- `threshold_sweeps/`: whole-dataset and split-specific threshold tables
- `figures/`: representative failure and success contact sheets

The original `image_path` and `localization_path` fields in the detailed JSON
are local provenance references. The complete scene corpus is intentionally not
committed; rerun `scripts/capture_libero_confidence_dataset.py` to reproduce it.

## Limitations

- The split changes object poses but contains the same ten semantic categories
  in calibration, validation, and test. It does not test unseen object classes.
- Complete candidate scores are missing for 63.8% of trials because all eight
  valid labels were not simultaneously present in the returned top-token
  alternatives.
- Alphabet soup and tomato sauce show systematic, sometimes high-score semantic
  failures. Thresholding alone does not correct these class-specific errors.
- The three included images are illustrative examples, not the complete visual
  dataset.
