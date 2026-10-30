# FreeZeV2 base: BOP-Classic-Core run

Method: `FreeZeV2-base-paper-v1-independent`; mask configuration: `CNOS-FastSAM-public-BOP23`.

Task: BOP19 model-based 6D localization, official object/count targets, official k-most-visible GT evaluation. GT poses are evaluation-only.

| Dataset | Expected targets | Completed | Failed | Blocked | Pending | Poses | CADs with features | Official AR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lmo | 1445 | 1345 | 100 | 0 | 0 | 1345 | 8/8 | 56.5788% |
| tless | 4904 | 3370 | 1146 | 388 | 0 | 4553 | 27/30 | unavailable |
| tudl | 600 | 574 | 26 | 0 | 0 | 574 | 3/3 | 69.1983% |
| icbin | 200 | 125 | 75 | 0 | 0 | 1653 | 2/2 | 36.5631% |
| itodd | 733 | 509 | 224 | 0 | 0 | 2446 | 28/28 | unavailable |
| hb | 1630 | 1460 | 170 | 0 | 0 | 1460 | 16/33 | unavailable |
| ycbv | 4123 | 3994 | 129 | 0 | 0 | 3994 | 21/21 | 71.9046% |

A completed target produced its requested number of distinct valid poses. A failed target was attempted but produced fewer poses; any surviving predictions remain in the CSV and the full official target list remains the evaluator denominator.

Onboarding and inference runtimes, exact mask/target hashes, metric receipts and per-dataset blockers are in results_summary.json. Inference timing includes per-image candidate work and excludes CAD onboarding/evaluation.

## Resume

Use the immutable source snapshot and the recorded inference interpreter:
```bash
/home/jiabao/.venvs/freezev2-bop-20261029/bin/python /mnt/d/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029/classic_core_base_v2/source_snapshot/run_suite.py --config /mnt/d/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029/classic_core_base_v2/protocol.json --resume
```

No placeholder descriptors, pose constraints, GT initialization, parameter tuning on test GT, or author-prediction substitution is used. Source/recipe/model identities are sealed per run.

Status: SUITE INCOMPLETE; no complete BOP-Classic-Core aggregate is reported.
