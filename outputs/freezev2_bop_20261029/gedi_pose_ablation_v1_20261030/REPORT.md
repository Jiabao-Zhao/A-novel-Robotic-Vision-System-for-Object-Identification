# Controlled GeDi repair and CAD-resolution diagnostic

Independent robustness repair and isolated CAD476 preprocessing ablation; not full public-repository or published-performance reproduction

Seven fixed T-LESS object-image targets / seven requested instances / six images. All failed or blocked targets remain in official evaluation. This cohort was enriched for recorded numerical failures; it is not the full T-LESS benchmark.

| Arm | Saved poses /7 | AR | AR VSD | AR MSSD | AR MSPD |
|---|---:|---:|---:|---:|---:|
| baseline | 3/7 | 0.4762% | 0.0000% | 0.0000% | 1.4286% |
| repair224 | 7/7 | 14.2381% | 14.1429% | 14.2857% | 14.2857% |
| repair476 | 7/7 | 5.1905% | 12.7143% | 0.0000% | 2.8571% |

These are official BOP19 localization recalls. A saved pose is a valid-format output, not a statement that the pose is correct. Additional predictions may still have substantial pose error. No AP, CAD-identification accuracy, public-repository parity or paper-performance reproduction is claimed.

| Scene/image/object | Baseline poses | Repair224 poses | Repair476 poses |
|---|---:|---:|---:|
| 000001/000001/000030 | 0 | 1 | 1 |
| 000003/000005/000008 | 1 | 1 | 1 |
| 000003/000476/000008 | 0 | 1 | 1 |
| 000004/000000/000028 | 0 | 1 | 1 |
| 000005/000007/000001 | 1 | 1 | 1 |
| 000005/000007/000027 | 0 | 1 | 1 |
| 000005/000020/000001 | 1 | 1 | 1 |

All120 per-threshold official score files in each arm retain targets_count=7; each arm also has its verified aggregate score file. CSV poses match saved ledgers to absolute tolerance1e-12 in BOP millimetres. Candidate trace/artifact hashes, frozen baseline sources/results, and full original CAD/PCA/ICP geometry denominators were checked.

Per-target candidate ranking, selected/suppressed poses, coarse/refined/geometric scores, finite-row counts, failure reasons, stage runtimes and provenance hashes are retained in results_summary.json. Representative visual inspection is recorded separately by the supervising reviewer.

The repair preserves original full CAD points for ICP and geometric scoring. CAD476 changes only rendered-CAD visual preprocessing; observed DINO remains224. Parameters were fixed before cohort metrics. Both runs remain independent implementations with recorded assumptions.

Runtime sums include saved cache effects. Original baseline CSV/image timing includes full-scene work outside this cohort, so no baseline speedup ratio is reported.
