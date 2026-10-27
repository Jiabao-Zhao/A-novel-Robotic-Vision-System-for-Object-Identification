# Independent review of per-row certificate bounds

2026-10-24 20:07 UTC; reviewer: `robustness_capture`.

**Pass for the reviewed bound module under the existing pinned arithmetic and
environment guards.** A separately authorized isolated GPU retiming may use it;
this is not approval for a production proxy or full pose fitting. Source hashes
and exact conditions are preserved in
[per_row_independent_review.json](per_row_independent_review.json).

The algebra proves equality for all supported M, rather than fitting a few
examples. Difference error is affine, e_d=aM+b. Its product-error F and magnitude
bound V have nonnegative quadratic coefficients. A3 and the H+3 box recurrence
are positive constant multiples/additions of those polynomials. Thus
B=3F32+A32+9F64+2A64+P64 is degree<=2 with nonnegative coefficients for fixed H.
`exact_coefficients` evaluates the original expression in Fraction arithmetic at
0,1,2 before any float conversion; interpolation therefore recovers that entire
polynomial exactly. The extra rational checks supplement this degree proof.

`threshold_for_magnitudes` upper-rounds each coefficient and follows every
positive multiply/add with nextafter toward positive infinity, including final
multiplication by2. Monotonicity gives an upper bound on exact2B(M,H) at every
step. At zero or a subnormal product, RN with FTZ/DAZ disabled plus one upward
nextafter still bounds the exact nonnegative result. Nonfinite evaluated
thresholds invalidate the row. M<=2^60 also keeps magnitude squaring far below
binary64 overflow. The original GPU first/second gap must remain rounded down.

`row_thresholds` uses the maximum magnitude of the original complete CAD array
and the original query row. It cannot underestimate coordinates relevant to that
row's point distances or tree boxes. Nonfinite CAD values invalidate all rows;
nonfinite/out-of-cap query rows and unsupported depths produce infinity plus a
false validity mask. Shape/dtype assertions describe this prototype's supported
domain; a future transparent query wrapper must delegate unsupported calls to
the original SciPy API.

The CPU analysis uses finite exhaustive emulation, so its mask is adequate there.
The next GPU pilot must AND row validity with all previous kernel flag,
index-range, finite, nonnegative and ordered-top-two guards. It must preserve
original fallback row order, dtypes/shapes, binary hashes, actual tree depth,
and RN/FTZ/DAZ controls. A passing row bound alone never validates GPU output.

Memory use in this module is linear in CAD/query rows; it creates no query by
CAD distance matrix. Float64 row vectors cost2MiB each at262,144 rows. Fraction
coefficient caches have32 entries and supported H<=1024, rather than retaining
observed clouds. The existing16,384-row CPU analysis reports74.08MiB total peak
RSS; this is not a measured larger GPU memory bound.

Five focused tests passed in the saved CPU analysis. Its certified fraction rose
from60.69% to91.08%, with4,979 additional and zero lost certified rows; all
certified distances/indices matched the fixed SciPy reference bitwise. These
are nearest-neighbor certificate checks, not classification accuracy or proof
of GPU speed. No material source defect was found. This independent review ran
no GPU, query, fit or new accuracy-label operation and changed no live source.
