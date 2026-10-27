# Bounded GPU nearest-neighbor feasibility

The installed nonperiodic three-dimensional cKDTree distance code computes the
three squared coordinate differences in order, then takes a square root. GPU
argmin alone cannot preserve its tie choice: equal or numerically ambiguous
minima must use the original CPU tree. Separate eager FP64 kernels prevent
cross-operation multiply/add contraction; CPU selected-distance recomputation
must first match the installed binary bitwise.

This prototype changes no production source or frozen kernel and performs no
pose fitting. Torch is imported only by the explicit GPU command, after a
coordinator-approved memory window. It uses query blocks of 128 and caps Torch
allocation at half the eight-GiB GPU, with an explicit four-GiB allocated-byte
check. It compares complete output bytes against the original tree, including
duplicates, exact/near ties, float32 input conversion and bounded cached queries.
Any mismatch fails the investigation; no numerical tolerance is accepted.

Efficiency is uncertain. At 10,651 CAD samples, 131,072 queries require about
1.4 billion brute-force pairs, whereas the CPU tree prunes most distances.
Separate eager kernels also materialize intermediate arrays. The current
four-thread CPU reference already handles that bounded input in about 0.05 s
in isolation. The prototype therefore tests only 4,096 and 16,384 cached queries
first. If end-to-end GPU timing loses to the CPU path, larger probes and
integration are unjustified. Timings must include host/device transfer, CPU
recomputation, and original-tree fallback.

The ambiguity guard is a deliberately broad diagnostic safeguard, not a formal
proof for arbitrary coordinates, all floating-point environments or tree
traversal roundoff. Passing tested arrays alone does not authorize production
replacement. The frozen CPU production route remains active unless the parent
explicitly approves further work.

Source: [tagged distance kernel](https://raw.githubusercontent.com/scipy/scipy/v1.18.1/scipy/spatial/ckdtree/src/distance.h),
[tagged query kernel](https://raw.githubusercontent.com/scipy/scipy/v1.18.1/scipy/spatial/ckdtree/src/query.cxx).
