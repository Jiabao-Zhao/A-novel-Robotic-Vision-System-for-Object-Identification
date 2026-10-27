# Exact FP32 tree query audit

The bounded audit passed every numerical check. It does **not** establish a consistent throughput improvement over the existing exhaustive GPU proxy: the tree proxy was 11% slower on the largest dense workload. Do not integrate it on the strength of this run.

The run used 10 small fixtures (64 rows), saved dense queries and nominal queries reconstructed from the first 26 predicted poses of the cached Gear bank. The Gear CAD contains 10,651 points. No simulator poses, accuracy labels, pose fitting, scientific-cache writes or process pauses were used. All input/source hashes remained unchanged. Execution ran from 22:33:33 to 22:33:43 UTC under live scientific load; WSL load averages before measurement were 14.57/13.11/13.27. Context selection, compilation, setup and warmup were outside the timing samples. Complete proxy conversion, fingerprinting, bounds, transfers, traversal, certification and original CPU fallback remained inside them.

Raw FP32 first/second distance bits and both point IDs matched the audit-only exhaustive reference on all **389,738 rows**. Its source differed from the frozen old kernel only by the function name and second-ID argument/write, checked byte-for-byte. An independent CPU exhaustive calculation checked **4,096 rows**, including all 64 small rows. The small fixtures cover duplicates, signed zero, repeated distances, FP32 subnormal coordinates/squares, a uniquely selected FP64 subnormal squared distance, large offsets, box boundaries and the coordinate cap.

Each complete proxy matched original four-thread SciPy distance/index bytes on **520,810 input rows** across the final workloads. The dense 131,072-row workload is a nested subset of the 262,144-row workload. The finite out-of-cap check explicitly used threshold 1, recorded one unsupported-bound row and no GPU row for each proxy. All 18 timed accelerated calls used the GPU and remained healthy; ordinary uncertified rows continued through original SciPy.

| Input rows | CPU4 median | Exhaustive proxy median | Tree proxy median | CPU4 / tree, all three samples | Exhaustive / tree, all three samples |
|---|---:|---:|---:|---:|---:|
| Dense 131,072 | 119.805 ms | 48.450 ms | 49.605 ms | 2.407 | 1.045 |
| Dense 262,144 | 229.457 ms | 93.214 ms | 104.849 ms | 2.178 | 0.890 |
| Nominal 127,530 | 55.369 ms | 38.888 ms | 30.740 ms | 1.801 | 1.260 |

Ratios use the sum of all three recorded samples, with no samples discarded. The 27 samples and per-call GPU/certified/fallback counts are in `completion.json`; individual samples are also preserved in `timing_summary.json`. The timed GPU threshold was the original 16,384 rows to exercise both engines on the nominal batch. This differs from the separate required full-fit pilot's 131,072-row admission setting. No actual GPU visited-node/point counts were instrumented, so this run cannot attribute the dense loss to a particular kernel component.

Peak combined explicit device allocation was **17,163,628 bytes (16.37 MiB)** against a conservatively precomputed 24,632,420-byte bound and a 128 MiB limit. Incremental host high-water was **340,414,464 bytes (324.64 MiB)** against 1 GiB. The reference context was destroyed before the old complete proxy initialized. All three contexts reported empty cleanup errors, destroyed contexts and zero tracked device bytes. Process session 35006 exited with code 0.

Completion SHA-256: `ca4129ad1011247f7ee2a0e4a1913adc0a5cdd3e1d2f4d22925db50fe3e2106d`.

The separate required Round16 fit completed 432 hypotheses in 3,841.32 seconds with a 3,362-point CAD and 1,358,622 observed voxels. This query audit uses a different CAD and is not a paired full-fit speed measurement. No full-fit speed claim follows from these timings.
