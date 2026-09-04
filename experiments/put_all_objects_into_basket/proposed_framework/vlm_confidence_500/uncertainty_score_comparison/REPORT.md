# Uncertainty-score comparison

This analysis reuses the 500 frozen `predicted_object_id` values and correctness
labels. It does not rerun the VLM or change the prediction rule. Missing scores
are treated as operational deferrals.

## Development protocol

For each method and requested accuracy target, all thresholds observed in the
calibration and validation partitions were evaluated. The selected operating
point maximizes validation coverage subject to autonomous accuracy meeting the
target independently on both calibration and validation. The final method is
the one with highest validation coverage at the 95% target. Only after this
selection was frozen was its held-out test performance calculated.

## Score comparison

| Score | Overall availability | Validation availability | Correctness ranking AUC (development) | Normalized validation AURC |
| --- | ---: | ---: | ---: | ---: |
| Raw label likelihood | 100.0% | 100.0% | 0.930 | 0.043 |
| Candidate-normalized | 36.2% | 32.0% | 0.866 | 0.136 |
| Candidate margin | 36.2% | 32.0% | 0.865 | 0.137 |

Ranking AUC measures whether a randomly chosen correct prediction tends to have
a higher score than a randomly chosen incorrect prediction. AURC is integrated
over each method's attainable coverage and normalized by that attainable
coverage; lower is better. Neither metric implies calibration.

## Development operating points

| Score | Accuracy target | Threshold | Calibration coverage / accuracy | Validation coverage / accuracy | Validation false accepts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw label likelihood | 90% | 0.999602895 | 81.3% / 92.2% | 83.0% / 90.4% | 8 |
| Raw label likelihood | 95% | 0.999983237 | 76.7% / 95.2% | 73.0% / 97.3% | 2 |
| Candidate-normalized | 90% | 0.999984020 | 20.7% / 90.3% | 13.0% / 100.0% | 0 |
| Candidate-normalized | 95% | 0.999999531 | 14.0% / 95.2% | 10.0% / 100.0% | 0 |
| Candidate margin | 90% | 0.999971013 | 20.7% / 90.3% | 13.0% / 100.0% | 0 |
| Candidate margin | 95% | 0.999999138 | 14.0% / 95.2% | 10.0% / 100.0% | 0 |

Candidate-normalized score and margin are operationally disadvantaged by the
API top-logprob truncation: 63.8% of all trials and 68.0% of validation trials
have neither score. Their apparent high validation precision is obtained at
only 10-13% whole-split coverage.

| Score method | Availability | Validation coverage at >=90% accuracy | Validation coverage at >=95% accuracy | Frozen test accuracy | Frozen test coverage | Frozen test false accepts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw label likelihood | 100.0% | 83.0% | 73.0% | 94.7% | 76.0% | 4 |
| Candidate-normalized | 36.2% | 13.0% | 10.0% | not evaluated | not evaluated | not evaluated |
| Candidate margin | 36.2% | 13.0% | 10.0% | not evaluated | not evaluated | not evaluated |

Only the development-selected raw method was opened on test. This avoids using
the held-out split to compare or reselect methods.

## Development score distributions

| Score | Correct median (IQR) | Incorrect median (IQR) |
| --- | ---: | ---: |
| Raw label likelihood | 1.000000 (1.000000-1.000000) | 0.997285 (0.872489-0.999842) |
| Candidate-normalized | 1.000000 (0.999968-1.000000) | 0.992567 (0.805051-0.999842) |
| Candidate margin | 0.999999 (0.999936-1.000000) | 0.987358 (0.632178-0.999754) |

All three scores are heavily saturated near one. Incorrect predictions can
therefore receive extremely high scores.

The public records do not retain the generated choice label separately from
the final saved prediction. One calibration trial
(`task_08_chocolate_pudding_state_26`) is an essentially exact normalized tie
between two candidates, so its raw label cannot be unambiguously aligned from
the saved fields. The comparison follows the requested fixed-prediction rule
and keeps the saved `predicted_object_id`; this one case does not affect the
selected raw threshold because its raw score is approximately 0.4994.

## Frozen test evaluation

The development selection rule chose raw label likelihood at threshold
`0.9999832372181827`.

| Frozen method | Test accuracy | Test coverage | Deferred | False accepts |
| --- | ---: | ---: | ---: | ---: |
| Raw label likelihood | 94.7% (72/76) | 76.0% (76/100) | 24 | 4 |

The false-acceptance rate is 4.0% of all test trials and 5.3% of autonomous test
decisions. The test accuracy is slightly below the development target of 95%,
so the target did not strictly generalize.

Nonselected methods were not evaluated at chosen operating points on test;
doing so would reuse the held-out split for method selection.

## Confidently wrong classes

On calibration plus validation:

- Alphabet soup: 33/40 predictions were wrong. At the frozen raw threshold,
  29 wrong predictions were deferred, but four wrong predictions were still
  accepted. Incorrect raw scores reached `0.999999687`.
- Tomato sauce: 40/40 predictions were wrong. At the frozen raw threshold,
  31 were deferred, but nine wrong predictions were still accepted. Incorrect
  raw scores reached `1.0`.
- Candidate-normalized score and margin also assigned values near one to some
  failures. At their strict 95% development points, they deferred all tomato
  sauce trials but still accepted two wrong alphabet-soup trials.

On held-out test, the frozen raw gate deferred all nine incorrect alphabet-soup
predictions, but accepted four of the ten incorrect tomato-sauce predictions.
No score reliably recognizes these semantic class failures by itself.

## Recommendation

**None of the three is a reliable standalone HITL gate.** Raw label likelihood
is empirically the strongest single signal and provides far better coverage,
but the frozen 95%-development operating point missed its target on test and
accepted four systematically wrong tomato-sauce predictions. Candidate
normalization and margin lose too much operational coverage through missing
top-logprob alternatives and do not eliminate confidently wrong classes.

Raw label likelihood should remain the strongest baseline or one feature in a
future deferral rule, but this experiment does not justify trusting any one of
these scores alone.
