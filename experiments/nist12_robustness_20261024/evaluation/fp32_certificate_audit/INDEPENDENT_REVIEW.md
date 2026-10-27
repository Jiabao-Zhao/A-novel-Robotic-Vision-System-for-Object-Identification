# Independent certificate and kernel review

2026-10-24; reviewer: `robustness_capture`. **Conditional pass for a bounded,
isolated GPU pilot only, after the guards below are enforced and the coordinator
authorizes execution. No production replacement is approved.** This review ran
no GPU, nearest-neighbor query, pose fit, model, or accuracy-label operation.

The machine-readable [independent_review.json](independent_review.json) pins the
reviewed derivation, probe, results and CUDA source. Read-only disassembly and
commands are preserved in [assembly_evidence.txt](assembly_evidence.txt).

## Index proof: conditional mathematical pass

The derivation covers the relevant distinction: a unique exhaustive real-distance
minimum alone does not prove equality to rounded cKDTree traversal. Its E64 and
depth-dependent Ebox terms cover leaf arithmetic and accumulated box pruning.
With B = E32 + 2 E64 + Ebox, a downward-rounded top-two gap greater than upward
2B leaves more than the required margin between any competitor's CPU distance
and every ancestor box containing the selected point. No unsupported/tied row
may bypass the original tree.

The real update is new-side minus old-side, followed by addition to the parent
state. This matches both official
[query.cxx](https://raw.githubusercontent.com/scipy/scipy/v1.18.1/scipy/spatial/ckdtree/src/query.cxx)
and the inspected binary at0x587d4/0x587d8. Initialization invokes the same update
three times from zero, so H+3 is conservative. The 7u coefficient is loose enough
for the stated subtraction/addition recurrence. Signed accumulated state does
not invalidate the additive underflow bound: clearing two signed subnormal
add/subtract operands changes the exact result by at most two minimum-normal
units in absolute magnitude; four units also cover rounding/flushing. This
argument does not extend to arbitrary multiplication of unrelated DAZ operands.

The coordinate cap2^60 prevents FP32 conversion, squared-difference and three-term
sum overflow. Exact/near ties and ambiguous underflow cases fall back. The
existing4096-row emulation supports, but does not by itself prove, the claims.

## Distance bytes: conditional pinned-build pass

The inspected binaries are:

- SciPy `_ckdtree.so` SHA256
  `29316ed1bf1f2b1c4915740fcfe8014df17524a6537b707c1f1347b51c75ddf4`.
- NumPy `_multiarray_umath.so` SHA256
  `e6241082eaab01d0f68b2b661522ecb9d5d2160915104df8b112cc161dfa18b4`.

For dimension3, SciPy takes the n<=3 branch at0x589f7 to positive-zero
initialization at0x58d48. It computes x/y differences and self-products with
subpd/mulpd, adds them in order at0x58add and0x58ae5, then computes/adds z with
subsd/mulsd/addsd at0x58af8–0x58b02. This path contains no FMA or reassociation.
The valid nonnegative result uses sqrtsd at0x59627. This agrees with the
[fixed-version point-distance source](https://raw.githubusercontent.com/scipy/scipy/v1.18.1/scipy/spatial/ckdtree/src/distance.h).

NumPy reports active float64 sqrt dispatch `SSE41`; its contiguous loop uses
sqrtpd at0x37543f and its single-element tail uses sqrtpd at0x375638. Separate
NumPy subtraction/multiplication/addition ufunc calls materialize their results
and cannot fuse across calls. Under the same RN/FTZ/DAZ state, these elementary
SSE/AVX operations reproduce the inspected SciPy operation sequence, including
square root. This is a claim about this pinned build, dtype and dispatch.

The evidence recorder read actual MXCSR through installed glibc's fenv_t layout:
main and four fresh metadata threads all had RN, FTZ=false, DAZ=false.
`fegetround()==0` alone is not an adequate SSE-state guard. The sampled threads
are not instrumented SciPy internal threads; an implementation must preserve
the inherited FP environment. Grouped one-point original-SciPy trees remain an
alternative for returned distances if operation/environment correspondence
cannot be enforced; their overhead has not been measured here.

## CUDA draft: source merge review passes

Each lane retains the two lexicographically smallest (distance, original ID)
pairs with distinct IDs. The top two of a union lie in the union of its local
top-two sets; inserting both partner minima therefore preserves the invariant
through the shuffle tree. Distinct IDs at equal distance remain two minima and
force fallback. Per-lane invalid flags are OR-reduced through the same tree to
lane0. A NaN may disrupt local ordering but flags the entire query, which must
then use the original CPU tree.

The early query guard is uniform within a warp, and all remaining lanes execute
the full-mask shuffles, provided the required1D launch shape is used. Explicit
RN subtraction, multiplication and addition intrinsics match the proposed
distance model. This is source review; generated PTX has not been inspected or
executed by this reviewer.

## Mandatory pilot guards and remaining tests

- Preserve immutable original finite contiguous binary64 values, CAD order/tree
  and actual tree depth. Normalize other layouts exactly as the CPU reference;
  restore original row order, output shapes and dtypes.
- Support only nonperiodic m=3,k=1,p=2,eps=0,infinite-bound calls. Unsupported
  states, fewer than two CAD points, empty/other shapes and failed guards use the
  unchanged original SciPy query.
- Enforce M<=2^60; use exact outward bound/downward gap calculations. Check
  kernel output flags, finiteness, nonnegativity and index range per row.
- Pin both binary hashes, active float64 sqrt dispatch and actual MXCSR RN/
  FTZ/DAZ state on the relevant threads before conversion/recomputation.
- Use1D grid/block,128 threads, valid int32 counts and buffer lengths. Bound the
  **padded maximum query ID**, not only query_count, by INT_MAX to prevent signed
  overflow before the early guard. Point/query offsets remain64-bit.
- Compile without fast math/TF32/FMA/FTZ and inspect/preserve PTX/options/hash
  before launch. Raw top-two values must match exhaustive ordered-FP32 CPU
  emulation; every final index/distance must match the original tree bitwise.
- Add mixed-scale unique subnormal-distance winners with distant competitors
  (the present all-tiny case entirely falls back), coordinate-cap/large-offset
  boundaries, duplicate/tie arrangements across lanes and shape/layout cases.
- Passing a small isolated pilot permits no live proxy or full fit. Production
  consideration still needs a separately authorized complete-bank replay and
  demonstrated net throughput benefit including transfers and CPU fallback.

No mathematical or source reduction counterexample was found under these
explicit conditions. Without those conditions, the claimed exactness is not
established. Original full geometry searches remain required and unchanged.
