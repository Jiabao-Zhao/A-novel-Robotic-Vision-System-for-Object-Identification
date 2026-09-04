# Raw VLM likelihood calibration on frozen LIBERO trials

The analysis uses the existing 500 predictions only; no VLM inference was rerun.
The 300/100/100 calibration, validation, and test partitions were preserved.
Raw likelihood is used as a ranking signal here, while probability calibration is evaluated separately.

## Raw-score discrimination

| Split | Association accuracy | Correctness AUROC | Risk–coverage area |
| --- | ---: | ---: | ---: |
| calibration | 0.796667 | 0.943069 | 0.034136 |
| validation | 0.810000 | 0.885315 | 0.043427 |
| test | 0.820000 | 0.862127 | 0.044939 |

### Held-out autonomous accuracy by raw threshold

| Threshold | Coverage | Autonomous accuracy | False accepts |
| ---: | ---: | ---: | ---: |
| 0.9 | 0.950 | 0.832 | 16 |
| 0.99 | 0.890 | 0.888 | 10 |
| 0.999 | 0.830 | 0.904 | 8 |
| 0.9999 | 0.810 | 0.914 | 7 |
| 0.99995 | 0.790 | 0.937 | 5 |
| 0.999983237218183 | 0.760 | 0.947 | 4 |
| 0.999999 | 0.730 | 0.945 | 4 |

## Validation selection

| Mapping | Validation Brier | Validation quantile ECE | Validation AUROC |
| --- | ---: | ---: | ---: |
| isotonic | 0.088891 | 0.027231 | 0.890838 |
| platt | 0.137218 | 0.163848 | 0.885315 |

Selected on validation: **isotonic** (lowest Brier score).

## Held-out test

| Score | ECE (10 quantile bins) | ECE (10 equal-width bins) | Brier | AUROC |
| --- | ---: | ---: | ---: | ---: |
| Raw likelihood | 0.164437 | 0.166066 | 0.171396 | 0.862127 |
| isotonic calibrated | 0.054745 | 0.110590 | 0.106913 | 0.844512 |

The frozen monotonic mapping changed test AUROC by -0.017615; isotonic plateaus create ties, so ranking is not expected to improve.

### Held-out quantile reliability bins

| Score | Count | Mean score | Empirical correctness |
| --- | ---: | ---: | ---: |
| raw | 10 | 0.848150 | 0.300000 |
| raw | 10 | 0.996242 | 0.500000 |
| raw | 10 | 0.999979 | 0.800000 |
| raw | 70 | 1.000000 | 0.942857 |
| isotonic | 6 | 0.123840 | 0.500000 |
| isotonic | 13 | 0.329670 | 0.384615 |
| isotonic | 8 | 0.598684 | 0.625000 |
| isotonic | 73 | 0.976613 | 0.945205 |

Ten quantile bins were requested. Exact score ties and isotonic plateaus reduce these to fewer distinct non-empty bins.

Raw likelihood is not itself calibrated: values concentrated near one substantially overstate empirical correctness. The selected mapping supports an approximate class-agnostic correctness-probability interpretation for this frozen model/prompt/simulation setup. The test sample is only 100 trials and the isotonic output is coarse, so this interpretation must not be generalized to new models, prompts, candidate counts, or domains without fresh validation.

## Systematic held-out failures

| Target | Trials | Accuracy | Incorrect | Incorrect accepted by raw gate |
| --- | ---: | ---: | ---: | ---: |
| alphabet soup | 10 | 0.200 | 8 | 0 |
| tomato sauce | 10 | 0.000 | 10 | 4 |

These failures remain important because a single class-agnostic scalar mapping cannot identify a class-specific semantic confusion when its raw likelihood resembles that of correct predictions.
