# Conservative FP32 nearest-neighbor certificate

This is an isolated mathematical/CPU-emulation audit. The production runner is
unchanged. The old FP64 GPU experiment remains a failed speed experiment. No GPU
speed claim follows from this note or the certificate fraction.

## Supported operation and guards

The proposed operation is the frozen nonperiodic, three-dimensional SciPy
`cKDTree.query`, with `k=1`, `p=2`, `eps=0`, and infinite initial distance bound.
The original CAD point order, tree and query binary64 values remain available.
Each proposed GPU squared distance must use coordinate conversion, direct
subtraction, self-product, and ordered nonnegative addition. A norm expansion
`q*q + p*p - 2*q*p`, TF32, approximate arithmetic, or an uninspected compiler
transformation is outside this proof. Use round-to-nearest basic arithmetic;
enforce/inspect this in any future kernel and CPU environment. Separate CUDA
operations are one straightforward way to prevent unreviewed reassociation.

For every certified row require finite input, at least two CAD points, and a
maximum absolute original coordinate `M <= 2^60`. Otherwise use the unchanged
CPU query. This deliberately conservative cap prevents binary32 overflow in
every conversion, subtraction, square and three-term sum, including the bounds
below. Returned GPU values/indices also require finite/range checks. Any
unsupported state falls back; it must not become an invented score.

Let `u32 = 2^-24`, `u64 = 2^-53`, `eta32 = 4*2^-126`, and
`eta64 = 4*2^-1022`. For the operations above use

    |round(t) - t| <= u*|t| + eta.

The absolute term covers gradual underflow and FTZ/DAZ conservatively. For
addition/subtraction, clearing two subnormal operands changes their exact result
by at most two minimum-normal units; rounding/flushing adds at most another unit
plus the relative term, bounded by four units. For a self-product, clearing a
subnormal operand removes a square smaller than one minimum-normal unit. This
argument does not authorize general multiplication of unrelated DAZ operands.

## Squared-distance bounds

All following calculations in the audit use exact `Fraction` arithmetic. The
final threshold is rounded upward; the measured top-two gap is rounded downward.

For binary32 conversion let `e_c = u*M + eta`; for original binary64 values
let `e_c = 0`. For either precision define

    e_d = 2*e_c + 2*u*(M + e_c) + eta
    D   = 2*M + e_d
    F   = e_d*(4*M + e_d) + u*D^2 + eta
    V   = (1 + u)*D^2 + eta.

Here `e_d` bounds error in one coordinate difference, `F` bounds error in its
rounded square against the exact square of the original inputs, and `V` bounds
the absolute rounded square. The first term of `F` follows from
`|a^2-b^2| <= |a-b|*(|a|+|b|)`.

Set `A_0 = 0` and, for `j = 1,2,3`,

    A_j = (1 + u)*A_(j-1) + u*j*V + eta.

The bound against an exact real squared distance is `E = 3*F + A_3`.
Using the two precisions gives `E32` and `E64` respectively. This includes
binary64 subtraction, squaring and summation error; a GPU-to-real margin alone
would be insufficient.

## SciPy box-pruning roundoff

A unique brute-force minimum is not by itself a proof of equality to an
independently rounded tree traversal. SciPy's box lower bounds also accumulate
roundoff, which this certificate covers explicitly.

Let `H` be the maximum depth of the unchanged tree. Its box state stores three
rounded squared side distances, each bounded by `V64` and differing from the
exact corresponding side square by at most `F64`. Each update computes a rounded
`new_side - old_side` followed by rounded addition to the previous total.
Relative to the exact sum of the three stored floating-point sides, one update
satisfies

    C = (7*u64 + 2*u64^2)*V64 + (2 + u64)*eta64
    P_0 = 0
    P_(j+1) = (1 + u64)*P_j + C.

To see the constants, subtraction error is at most `2*u64*V64 + eta64`.
The next addition has magnitude at most the previous three-side sum plus two
side magnitudes and accumulated error. Combining these gives the recurrence.
There are at most `H+3` updates on a node's ancestry, including initialization.
Thus

    Ebox = 3*F64 + P_(H+3)

bounds the computed box lower bound's error against the exact real box distance.
Tree copies, comparisons and heap moves add no arithmetic error. This relies on
the unchanged tree's valid geometric split/containment invariants; changing its
construction or layout requires separate validation.

## Certificate and exact fallback

Let `f1` and `f2` be the two smallest **computed** GPU binary32 squared distances
over every CAD point. The GPU reduction must return those values exactly under
ordinary floating-point comparison; approximate top-k is unsupported. Define

    B = E32 + 2*E64 + Ebox.

Certify the selected point only if the downward-rounded gap satisfies

    f2 - f1 > upward(2*B).

For every other point its rounded CPU squared distance then exceeds the selected
point's rounded CPU squared distance by more than `E64 + Ebox`. Any tree box
containing the selected point has computed lower bound at most the selected
rounded CPU distance plus `E64 + Ebox`. Therefore no current bound supplied by
another point can prune the selected point's path. Before that point is visited,
the heap cannot terminate past all boxes on its path; after it is visited, no
other rounded point distance can replace it. This establishes the original
tree's selected index, not only the real-distance argmin. All exact or
numerically ambiguous ties fall back.

For a certified index, recompute the selected distance on CPU using SciPy's
three ordered binary64 squared differences and square root. The fixed installed
build's operation correspondence and returned distance bytes must be validated;
the existing CPU recomputation audit and this emulation pass that check. For
every uncertified row call the original complete SciPy tree. A future production
implementation still needs kernel inspection, installed-build/environment
checks, complete output parity tests and a full-bank replay. This mathematical
note does not certify a GPU implementation that does not yet exist.

Relevant official fixed-version sources:
[point distance](https://raw.githubusercontent.com/scipy/scipy/v1.18.1/scipy/spatial/ckdtree/src/distance.h),
[tree query/update/heap](https://raw.githubusercontent.com/scipy/scipy/v1.18.1/scipy/spatial/ckdtree/src/query.cxx),
[row wrapper](https://raw.githubusercontent.com/scipy/scipy/v1.18.1/scipy/spatial/_ckdtree.pyx).

## Bounded emulation result

`results.json` records exactly 4,096 total query rows, SciPy1.18.1/NumPy2.2.6,
16-row blocks, and 10.8MiB maximum additional process RSS. No Torch, CUDA, model,
pose fit or accuracy annotation was used.

| Case | Certified | Fallback |
| --- | ---: | ---: |
| Cached label-free depth queries | 2,431 / 3,072 (79.13%) | 641 |
| Synthetic random | 509 / 512 | 3 |
| Exact ties/duplicates | 0 / 256 | 256 |
| Near ties | 0 / 64 | 64 |
| Zero distance, unique point | 64 / 64 | 0 |
| Tiny underflow coordinates | 0 / 64 | 64 |
| Coordinate-cap guard | 0 / 64 | 64 |

Every output index and distance matches the original tree bitwise. The cached
queries use 3,072 raw foreground pixels transformed by a previously saved
predicted pose; they are not all voxelized observations or a full ICP trajectory.
These acceptance fractions cannot be extrapolated to other CADs/poses/iterations.
The CPU emulation establishes no GPU transfer, reduction, fallback-throughput or
whole-fit speed. Proceeding to a bounded performance pilot is mathematically
plausible; production replacement remains a no-go until the checks above pass.
