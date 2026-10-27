# Per-row bound: CPU-only prototype

The same reviewed B(M,H) is a positive quadratic in M. New isolated module
`per_row_bound.py` derives its coefficients using exact Fraction interpolation,
asserts positivity/identity, rounds coefficients upward, and applies nextafter
toward positive infinity after every multiply/add and the final factor2. Each
row uses max(CAD absolute coordinate, that query row's absolute coordinate).
Nonfinite/cap/depth failures receive infinite thresholds and cannot certify.
Reviewed `probe.py`, `DERIVATION.md`, `ordered_top2.cu` and all live files remain
unchanged. Independent review is required before GPU timing or integration.

`per_row_cpu/20261024T200603368188Z/result.json` records five passing focused
tests: exact positive-quadratic identity; exact outwardness over logarithmic,
subnormal, zero/cap/neighbor/random magnitudes at eight tree depths; agreement
with the reviewed expression; invalid-input fallback; and row/strided handling.

The CPU-only analysis reuses16,384 deterministic spread rows from the previously
saved262,144-row native-voxel query bank. It emulates the ordered FP32 top-two
distances and compares the certificates, without GPU calls or accuracy labels.

| Margin | Certified query rows | Original CPU fallback |
| --- | ---: | ---: |
| One global coordinate maximum |9,944 /16,384(60.69%) |6,440 |
| Each row's coordinate maximum |14,923 /16,384(91.08%) |1,461 |

The prototype certifies4,979 additional rows and loses zero on this sample.
Every newly/all certified nearest point index and recomputed binary64 distance
matches the original pinned SciPy tree bitwise. These are nearest-neighbor
execution certificates, not object-classification accuracy or acceptance rates.

Warm vectorized threshold calls take1.553/1.474/1.400ms (median1.474ms) for16k
rows. Exact CPU-FP32 emulation takes0.735s. Peak process RSS is74.08MiB, below
the256MiB additional-memory guard. CPU build/dispatch/FP-control checks and
input/source/output signatures are preserved. Torch/CUDA/models/pose fits/new
accuracy labels are unused. No GPU or whole-fit speed claim follows yet.
