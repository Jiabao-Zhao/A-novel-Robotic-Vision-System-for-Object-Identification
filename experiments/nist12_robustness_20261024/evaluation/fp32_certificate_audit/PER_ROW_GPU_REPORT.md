# Per-row certificate GPU retiming

The coordinator authorized a new isolated pilot after the saved independent
per-row review. It reuses the exact previously prepared262,144-query bank and
the same kernel/PTX. Archived pilots, reviewed proof/kernel and live runner were
not edited. All scientific fit processes continued during this live-load test.

Receipt: `gpu_pilot_per_row/20261024T200924051761Z/result.json`.

| Query rows | Original SciPy4 median(ms) | Complete per-row hybrid median(ms) | Native/hybrid speed | Certified |
| --- | ---: | ---: | ---: | ---: |
|16,384 |17.663 |8.867 |1.99times |14,923(91.08%) |
|65,536 |68.802 |32.894 |2.09times |59,645(91.01%) |
|262,144 |258.952 |106.212 |2.44times |238,401(90.94%) |

Three alternating warm passes per size include normalization/FP guards, all
per-row bound/max computations, both CAD/query FP32 conversion and upload on
every call, kernel/synchronization, download, certificate construction, CPU
selected-distance recomputation and original SciPy4-thread fallback. Coefficient
setup is a separate9.62ms once-per-depth preparation cost. CUDA context/module/
buffer initialization takes0.174s, excluded from warm calls. The full process
startup cost is not separately instrumented.

Every final distance and original CAD-point index matches SciPy bitwise for all
sizes and every repeat. A4,096-row subset also matches CPU ordered-FP32 raw
first/second values and first IDs. The per-row supported-domain mask is combined
with all existing GPU invalid/finite/nonnegative/order/index-range guards before
any certificate. All unsupported/ambiguous rows retain the original CPU query.

At262,144 rows,23,743 rows(9.06%) fall back. Median measured components are
44.664ms CPU fallback,33.651ms per-row bounds/max,20.487ms kernel/synchronization,
14.504ms CPU selected-distance recomputation,4.105ms certificate construction,
1.209ms download/output allocation,0.732ms conversion,0.727ms normalization/FP
guard and0.559ms upload. Independently computed component medians do not have to
sum to the median complete time. Timing variance reflects live CPU load.

The earlier global-bound pilot certified60.31% at262k; this one certifies90.94%
using the same input bank and proof evaluated per row. Their speed measurements
come from separate live-load windows. The paired speed table above compares the
new path directly with a freshly timed CPU reference in its own window.

Peak explicit device buffers are7,467,844bytes(7.12MiB), below128MiB. Peak
process RSS is247.69MiB,180.69MiB above baseline and below the768MiB incremental
host cap. All buffers were freed, module unloaded and context destroyed in
`finally`, with no cleanup errors and zero tracked device bytes remaining. The
process exited successfully and never imported Torch.

No model, pose fit, production edit or accuracy-label access occurred. This is
nearest-neighbor query acceleration, not measured full-fit or whole-study speed.
A cached complete-bank parity replay and a separate production decision remain
necessary before changing the live compute path.
