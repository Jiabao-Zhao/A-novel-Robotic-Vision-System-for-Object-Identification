# Decision: preserve the audit; do not integrate the tree engine

The exactness checks passed. The new tree proxy did not provide a consistent improvement over the existing exhaustive GPU proxy: summed timing ratios were **1.045× at dense131k, 0.890× at dense262k, and 1.260× at nominal127530**. The largest dense case became slower. These are complete-query measurements on the 10,651-point Gear CAD, not paired full-fit measurements or a per-CAD speed guarantee.

All 27 timing samples are retained below in milliseconds; none was discarded. Methods rotated through each timing position across three repeats. Compilation, setup, warmup and context selection were outside timing; conversion, tree fingerprints, bounds, transfers, kernel work, certification and original SciPy fallback remained inside it.

| Workload | CPU4 samples | Existing exhaustive samples | New tree samples |
|---|---|---|---|
| Dense131072 | 112.623,119.805,122.637 | 48.450,57.643,48.020 | 47.586,49.605,50.311 |
| Dense262144 | 217.905,232.506,229.457 | 93.214,90.349,94.160 | 107.694,104.849,99.626 |
| Nominal127530 | 54.800,55.369,56.935 | 38.888,39.814,38.216 | 30.740,28.545,33.485 |

This run used live scientific CPU load without pausing workers. WSL CPU usage before measurement was 71.5%, with load averages14.57/13.11/13.27. GPU clocks were not measured. Scheduling and clock effects therefore remain unseparated; the table supports no claim about their cause. The timed admission threshold was **16,384 rows**, allowing nominal127530 to exercise both GPU engines; the required full-fit pilot used **131,072 rows**. All 18 timed accelerated calls recorded positive GPU rows and healthy runtimes. Actual GPU visited-node/point counts were not instrumented.

Raw first/second FP32 distance bits and both IDs matched the literal-diff exhaustive reference on **389,738 rows**. CPU exhaustive checks covered **4,096 rows**, including all64 small edge rows. Each complete proxy matched original four-thread SciPy distance/index bytes on **520,810 input rows** across the named workloads; dense131k is nested within dense262k. The finite out-of-cap check used threshold1 and recorded original CPU fallback with no GPU row. No labels, fits, scientific-cache writes or process controls were used. All reviewed sources and17 input dependencies remained unchanged.

Peak combined explicit device allocation was16.37MiB (128MiB cap); incremental host high-water was324.64MiB (1GiB cap). All three contexts reported no cleanup errors, destroyed contexts and zero tracked device bytes. Session35006 exited0. This audit authorizes no further GPU jobs or production changes.

Full evidence: [completion and per-call counts](query_only_audit/20261024T223312Z_query01/completion.json), [unrounded timing samples and ratios](query_only_audit/20261024T223312Z_query01/timing_summary.json), [detailed report](query_only_audit/20261024T223312Z_query01/REPORT.md), [source review](bounded_query_independent_review.json), [fake-guard receipt](query_audit_fake_receipt.json), [core proof review](independent_source_review.json), and [backend review](tree_query_backend_independent_review.json). Completion SHA-256 is `ca4129ad1011247f7ee2a0e4a1913adc0a5cdd3e1d2f4d22925db50fe3e2106d`.
