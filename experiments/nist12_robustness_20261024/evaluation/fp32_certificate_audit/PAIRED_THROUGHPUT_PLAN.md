# Bounded paired throughput plan (not executed)

## What the completed replay can establish

The saved bank contains only arm and whole-fit timing: 37.877465 seconds for the arm and 37.914740 seconds for `fit_pair` (37.914782 seconds externally). Runtime receipts separately record 0.169415 seconds for compilation/inspection and 0.190615 seconds for CUDA initialization. They do not contain per-query durations. Query sums, medians and the non-query remainder therefore cannot be recovered from this run without inventing measurements.

The successful path implies 527 tree fingerprints after the initial fingerprint, 527 row-bound evaluations, one CAD upload, 526 cached-CAD equality checks, and four FP/resource checks per supported GPU query. With constructor and first initialization checks, that is 2,110 FP/resource checks. These are source-derived operation counts, not time estimates. Source/review/binary hash verification is setup work; tree data/order/bounds/opaque-state fingerprinting is repeated work. Bounds, guards, copies, cache checks, conversion, transfer, kernel, fallback and recomputation were all inside the measured fit duration.

The previous per-row query pilot has component timing, but it is a different path and workload. Its 262,144-row medians include about 33.65 ms for bounds/maxima, 20.49 ms for kernel/synchronization, 44.66 ms for CPU fallback, and 14.50 ms for CPU selected-distance recomputation. Component medians need not sum to the median total. They cannot be multiplied by 527 to partition the bank replay, particularly because its certification rate and query sizes differ and its reusable-wrapper checks add work.

Both that earlier saved dense-query pilot and this cached nominal replay use exactly **10,651 CAD points** for Gear_Large. Thus their CAD bank sizes match; their observed-query distributions, certification rates, timings and wrapper paths still differ. Other CAD sizes would change exhaustive GPU work, so this does not establish speed across the product catalog.

## Proposed first measurement

Run a new isolated query-only program, with the reviewed numerical modules and kernel unchanged, after independent source review and fresh memory clearance. Geometry owns any exact-PID pause/resume and must guarantee resume in its own finally path; this program will not suspend another process. Do not run until root separately authorizes the measurement.

Use the existing 10,651-point Gear CAD and two label-free query workloads:

1. The already saved 262,144-row dense query bank, with deterministic nested subsets of 16,384, 65,536, 131,072 and 262,144 rows. No new depth preparation is needed.
2. At most 127,530 nominal rows (26 times 4,905 observed voxels) formed from the same cached observation and the first 26 saved predicted Gear poses in bank order, using the exact `observed-minus-translation @ rotation` transform. These are cached fitted poses, never simulator poses. This supplies a nominal distribution but is explicitly not a replay of every intermediate query from the full fit.

Construct the original tree once, initialize the proxy once, and report cold setup separately. For each workload/size, execute three warm paired original-SciPy-four-thread / complete-proxy calls, alternate order deterministically, and compare distance/index bytes after each pair. No timed path bypasses the proxy's fingerprints, FP/resource guards, bounds or cache checks. Enforce a 60-second controlled measurement deadline: start no new pair if the remaining budget is less than a conservative duration based on the slowest completed pair; report an incomplete matrix rather than extend the window. Cleanup is always in finally. Preserve the prior 1 GiB host/128 MiB explicit-device limits.

The 1 GiB increment limit is an isolated-audit budget. Production preparation for clouds exceeding two million voxels may legitimately use more memory before a query. Any future production adapter needs separately measured per-stage allowances and the geometry coordinator's reserve policy. Exceeding a GPU resource guard must retain the original CPU computation, never turn an otherwise available scientific method into an unavailable result. No reviewed source is changed for that prospective consideration.

Report paired differences/ratios, aggregate sum throughput and median per-call latency, certification/fallback rates, query shapes and input/source/build hashes. Also report cleanup, peak RSS, device peak and geometry's exact paused/resumed PID receipt. Compare medians within a workload/size, never combine independent medians as if they formed one call.

Add observational timers in the new harness only, forwarding identical arguments to unchanged functions. Collect exclusive/nested-aware time around tree fingerprinting, FP/resource checks, row bounds, cache equality/upload, GPU `top_two`, CPU selected-distance recomputation and original-tree fallback. Do not sum a parent timer and its nested children. Preserve the uninstrumented full-proxy wall time as the primary number; measure timer overhead with an empty forwarding timer and report it. CUDA-driver call timers can separate transfer/launch/synchronization if the unchanged runtime's `session.call` is observed; allocations/conversion and unassigned Python work remain an explicit residual, rather than being silently assigned to the kernel.

## Full-fit decision

If complete-proxy query savings are small or unstable, stop: the prior parity pass does not justify production integration. If gains are clear but the full-fit benefit remains uncertain, request a separately authorized matched CPU4 and GPU pair of the same already-cached Gear bank, both in the same controlled resource window, with identical non-query code and full nested parity gates. One later CPU replay compared with the earlier live-load GPU replay is still unpaired. Two controlled full fits would likely exceed the 60-second query-only window and require their own explicit duration/resource approval. No scientific cache changes or additional fit key are needed.

Instrumentation of that matched bank pair can then report actual query sum/median and `whole_fit_wall - query_wall` non-query work. Until that exists, the full-bank result is exactness and resource evidence; the measured query-only pilots establish only query throughput.
