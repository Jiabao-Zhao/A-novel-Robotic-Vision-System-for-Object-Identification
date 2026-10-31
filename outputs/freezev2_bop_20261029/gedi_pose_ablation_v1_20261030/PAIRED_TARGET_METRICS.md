# Paired metrics on the fixed seven-target cohort

These are per-target BOP19 localization threshold recalls from saved official matches. All seven requested instances, including missing predictions and failures, remain in the denominator. This is a diagnostic subset, not the full T-LESS benchmark.

| Scene/image/CAD | Baseline AR | Repair224 AR | Repair476 AR | 224âˆ’baseline pp | 476âˆ’224 pp |
|---|---:|---:|---:|---:|---:|
| 1/1/30 | 0.000000% | 90.000000% | 0.000000% | +90.000000 | -90.000000 |
| 3/5/8 | 0.000000% | 0.000000% | 0.000000% | +0.000000 | +0.000000 |
| 3/476/8 | 0.000000% | 6.333333% | 3.666667% | +6.333333 | -2.666667 |
| 4/0/28 | 0.000000% | 0.000000% | 16.666667% | +0.000000 | +16.666667 |
| 5/7/1 | 3.333333% | 3.333333% | 3.333333% | +0.000000 | +0.000000 |
| 5/7/27 | 0.000000% | 0.000000% | 0.000000% | +0.000000 | +0.000000 |
| 5/20/1 | 0.000000% | 0.000000% | 12.666667% | +0.000000 | +12.666667 |

| Scene/image/CAD | Arm | VSD recall | MSSD recall | MSPD recall | AR |
|---|---|---:|---:|---:|---:|
| 1/1/30 | baseline | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| 1/1/30 | repair224 | 80.000000% | 100.000000% | 90.000000% | 90.000000% |
| 1/1/30 | repair476 | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| 3/5/8 | baseline | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| 3/5/8 | repair224 | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| 3/5/8 | repair476 | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| 3/476/8 | baseline | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| 3/476/8 | repair224 | 19.000000% | 0.000000% | 0.000000% | 6.333333% |
| 3/476/8 | repair476 | 11.000000% | 0.000000% | 0.000000% | 3.666667% |
| 4/0/28 | baseline | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| 4/0/28 | repair224 | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| 4/0/28 | repair476 | 50.000000% | 0.000000% | 0.000000% | 16.666667% |
| 5/7/1 | baseline | 0.000000% | 0.000000% | 10.000000% | 3.333333% |
| 5/7/1 | repair224 | 0.000000% | 0.000000% | 10.000000% | 3.333333% |
| 5/7/1 | repair476 | 0.000000% | 0.000000% | 10.000000% | 3.333333% |
| 5/7/27 | baseline | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| 5/7/27 | repair224 | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| 5/7/27 | repair476 | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| 5/20/1 | baseline | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| 5/20/1 | repair224 | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| 5/20/1 | repair476 | 28.000000% | 0.000000% | 10.000000% | 12.666667% |

Each arm was checked against100 VSD,10 MSSD and10 MSPD official match/score pairs. Every per-threshold true-positive count, target denominator and recall agrees with its official score; metric aggregates and the mean of the seven per-target ARs agree with official scores_bop19.json to absolute tolerance1e-12.

A zero-to-positive AR change means some official threshold matches were gained. It does not establish success at every threshold. The numerical GeDi repair and CAD476 preprocessing remain separate comparisons. No GT pose files were read by this report.
