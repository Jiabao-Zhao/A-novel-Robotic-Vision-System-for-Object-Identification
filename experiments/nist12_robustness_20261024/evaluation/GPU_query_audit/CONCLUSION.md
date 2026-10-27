# Stop this GPU candidate; retain CPU production

The bounded FP64 direct-distance prototype passes exact-array checks on the
tested inputs, but **loses to the existing four-thread CPU tree**:

| Warm cached query test | CPU, four threads | GPU, end to end |
| --- | ---: | ---: |
| 4,096 queries × 10,651 CAD points | 3.14 ms | 19.48 ms |

The GPU path is **6.21× slower** in this test, with zero CPU fallback rows.
Timing includes transfers and final CPU distance recomputation. The first CUDA
case had already warmed the runtime; its initialization delay is not part of
this cached-case comparison. The early-stop rule skipped the larger 16,384-row
test. No full dense GPU query or whole fit was run, so no larger-workload or
whole-study speed claim follows.

Distances and original-tree indices match bitwise for every tested random,
duplicate/equidistant, near-tie, zero-distance, strided, float32-conversion and
cached case. CPU selected-distance recomputation independently matches the
installed tree on 131,072 cached rows. Exact and near ties use the original CPU
query to preserve its index choice.

These tests **do not prove universal frozen-query equivalence**. The prototype's
near-tie guard is not a derived IEEE error interval and must not be used as a
production certificate. Poor timing ends this investigation before further
proof work or integration, as required.

Peak Torch allocation was 44.1 MB (42.0 MiB), with 62.9 MB (60 MiB) reserved.
Observed host RSS was approximately 756 MB (721 MiB), approximately 687 MB
(655 MiB) above the pre-Torch baseline. The process exited fully after the test;
the geometry coordinator was notified. No installation, model/weight loading,
pose search, accuracy-annotation read, frozen-evidence edit or production-kernel
change occurred. The approved CPU route remains active.

Evidence: `CPU_recompute_probe.json`, `GPU_probe.json`, `prototype.py` and
`FEASIBILITY.md`. Existing CPU worker-parity evidence files remain untouched.
