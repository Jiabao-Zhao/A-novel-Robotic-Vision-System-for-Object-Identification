# Read-only checkpoint timing

Measured through WSL `/mnt/d` on 2026-10-24, with the existing fit pool and dense pilot active. This single pass mirrored `scene_scores` reads, first-missing-fit returns, path resolution, pair construction, and complete-scene JSON serialization. It never called `scene_scores`/`atomic_save` or wrote live score files. No annotations, model calls, fitting, or numerical changes were involved.

| Measurement | Result |
|---|---:|
| All-scene elapsed time | 194.575 s |
| Scenes complete / incomplete | 83 / 117 |
| File-existence checks | 12,069 |
| Thin-summary JSON reads | 11,952 |
| Distinct thin-summary files | 3,312 |
| Repeated reads | 8,640 |
| Path construction and existence checks | 27.870 s |
| Read and JSON parse | 60.397 s |
| Relative-path resolution and dictionary construction | 105.996 s |
| Complete-scene JSON serialization | 0.119 s |
| Other instrumentation/bookkeeping | 0.192 s |

The complete scene outputs totalled 5,181,054 serialized characters. Actual directory creation, file write, and atomic rename time was deliberately **not measured**, avoiding collisions with the live writer's fixed `.part` paths.

Live progress stayed at 400 completed new fits throughout the timed pass (17:02:15–17:05:30 UTC), consistent with the scorer being occupied by its own checkpoint. The preceding saved progress records advanced from 300 fits / 560.892 s to 400 fits / 723.322 s, a 162.430 s interval. The measured metadata scan alone exceeds that earlier 100-fit interval; intermediate checkpoints can therefore be a material scheduling bottleneck. That interval includes other scorer work and is not an isolated kernel-throughput measurement.

This was an in-run measurement under contention, apparently overlapping a live all-scene checkpoint, rather than an isolated filesystem baseline. Repeated reads account for 72.3% of all reads. Serialization contributes less than 0.1% of measured time; filesystem access and path resolution dominate.

A resource-only improvement is to skip previously complete, unchanged scenes during intermediate checkpoints, while preserving the final full-scene pass and all score contents. No scoring or cache-key changes are needed. No such live-source edit was made by this audit.

The complete machine-readable receipt and source hashes are in `measurement.json`; the reproducible read-only source is `benchmark.py`.
