# Source-only FP32 tree traversal proposal

Status: the CUDA source is uncompiled, unexecuted and unmeasured. Root authorized only small CPU structural/emulation fixtures while the sole required GPU pilot runs. No scientific runner imports this folder. The current required pilot, its 42 pins, exhaustive kernel, certificate and scientific writer remain unchanged. GPU execution requires pilot exit and separate authorization for an isolated query-only audit.

## Target equality

The proposed kernel must return exactly the current exhaustive kernel's two smallest distinct original point IDs, ordered lexicographically by `(ordered_FP32_squared_distance, original_ID)`. It preserves the same original FP64-to-FP32 conversion, three separate RN subtractions/self-products and ordered X+Y+Z RN additions. Only the candidate search changes. The existing per-row `E32 + 2E64 + Ebox` certificate, outward threshold, original SciPy fallback and pinned CPU distance recomputation do not change.

This is not a new SciPy search or an approximation to its pruning. It searches the converted FP32 point bank exactly, then applies the old certificate against the unchanged original FP64 SciPy tree.

## Safe lower bound against rounded FP32 distances

Let `P` be the fixed converted FP32 points in a node. For axis j, let `l_j=min(P_j)` and `u_j=max(P_j)`, selected directly from those finite FP32 values. No arithmetic conversion of original FP64 cell bounds is used. For a converted query `q_j`, compute:

```
b_j = RN(l_j - q_j) if q_j < l_j
      RN(q_j - u_j) if q_j > u_j
      +0 otherwise
```

For every descendant point p, exact real interval containment gives `max(l_j-q_j,0,q_j-u_j) <= |p_j-q_j|`. RN is monotone. RN subtraction is sign-symmetric under nearest-even; signed zero becomes positive zero when squared. Therefore:

```
0 <= b_j <= |RN(p_j - q_j)|.
```

Nonnegative RN multiplication and RN addition are monotone in each operand. Consequently the **same ordered RN operation sequence** gives a directed-safe lower inequality:

```
LB = RN(RN(RN(b_x*b_x) + RN(b_y*b_y)) + RN(b_z*b_z))
   <= ordered_FP32_squared_distance(p,q).
```

This lower relation is to the rounded FP32 point distance itself, not merely to a real Euclidean norm. It uses no empirical tolerance, norm expansion, approximate reciprocal, FMA, reassociation or epsilon. Gradual underflow preserves the monotonic inequality; FTZ/DAZ remain prohibited by the existing build/FP guards. The old finite original-coordinate cap `M<=2^60` ensures finite converted inputs, differences bounded by `2^61` and a three-square sum below FP32 overflow. Defensive nonfinite or malformed-bound flags still force original CPU fallback.

If the current second distinct score is s, a node may be discarded only when **LB > s**. Every point in that node then has score strictly worse than the already held second item. When `LB==s`, the node must be visited: an equal-score lower original ID can change the lexicographic pair. With fewer than two known distinct IDs, s is positive infinity and finite nodes cannot be pruned. Point IDs, not coordinate equality, define distinctness.

Inductively, the retained pair is the smallest two distinct items of all visited points. Discarded nodes cannot improve it. At exhaustion the pair equals the exhaustive scan's pair, including duplicate coordinates, distance ties and the second-item ordering. The existing strict certificate still sends exact/near ties to original SciPy.

## Original tree and containment obligations

`flatten_original_tree.py` is an isolated builder draft tested only on small CPU fixtures. It reads only the original bare nonperiodic 3D SciPy tree and keeps its original point array/order. The skeleton uses `tree.tree`, `split_dim`, `lesser/greater`, `start_idx/end_idx` and `indices`; these read-only views are defined by the [pinned SciPy 1.18.1 source](https://raw.githubusercontent.com/scipy/scipy/v1.18.1/scipy/spatial/_ckdtree.pyx). No binary node-layout interpretation is required.

Required checks before any future upload:

- Exact original tree fingerprint, including data, original indices, bounds and opaque node state, remains unchanged before/after flattening and at every query.
- Original indices are a permutation of all original point IDs. Root covers `[0,n)`. Every internal node's children partition its exact contiguous original index range. Every leaf is nonempty; in-order leaf concatenation equals the original index order with no omission/duplication. Node count matches `tree.size`, nodes have single parents and depth is bounded.
- Convert the exact same original point array to contiguous FP32 under the existing pinned conversion/FP guards. Leaf extrema are chosen from those converted points. Internal bounds are elementwise extrema of child bounds. Verify all leaf points lie inside their leaf box and both child boxes lie in each parent. Thus containment for all ancestors follows inductively, independently of original FP64 split positions.
- Freeze/fingerprint the converted points, node arrays and leaf indices. Cache keys must include the original tree fingerprint and all uploaded metadata hashes; original trees and GPU buffers are immutable under the single owner thread.
- Independently require both `2<=tree.n<=INT32_MAX` and `1<=tree.size<=INT32_MAX` before data access or int32 conversion: a supported point count alone does not prove a supported node count. Count and launch integer/buffer guards remain in force. The builder's memory must enter the host/device budgets, including temporary leaf gathers and validation arrays. Reject/fallback before an unsupported allocation.

## Bounded traversal

The uncompiled kernel draft gives one warp a query and one 64-entry shared-memory stack. Blocks contain exactly four warps (128 threads), with grid/block Y/Z equal to one. Only lane zero accesses its stack; warp shuffles publish uniform control, and leaves use the old comparison-only distinct-pair merge. The stack uses 1 KiB shared memory per block, not a per-query global matrix or per-thread local stack.

For a binary DFS tree of edge depth H, pending stack entries never exceed H+1: descending through an internal node retains at most one sibling at each ancestor. Host support is therefore limited to H<=63. The kernel checks capacity before every push and flags overflow rather than writing out of range. Node-ID/range/bound checks and a visited-node count bounded by the validated node count prevent malformed metadata from creating an unbounded traversal. Every flagged row must be AND-rejected by the existing raw-output guards and recomputed through the original CPU tree. Unsupported signatures, periodic trees, excessive depth and failed flattening remain original CPU calls.

The bound is recomputed on pop; child bounds order the traversal and prune only by the same strict rule. Ordering affects cost, not the resulting pair. An adversarial valid tree can still visit every point, so no speedup is promised.

## Required audit after the current pilot exits

1. Independent source/math review of this builder/kernel and a new host wrapper. Pin source/compiler/PTX, exact launch/buffer/depth bounds, FP state and cleanup. Reject FMA/FTZ/reassociation just as before.
2. CPU-only structural and exact-FP32 emulation tests: all leaf coverage, converted-box containment, empty/one-point/periodic/deep/degenerate trees, duplicates, signed zero, repeated distances, queries on box boundaries, mixed selected subnormal distances, cap neighbors, large offsets, maximum supported stack depth and forced overflow/malformed metadata.
3. Separately authorized isolated GPU parity: raw first/second FP32 score bits and first/second IDs versus exhaustive FP32, then final distance/index bytes versus original SciPy with the unchanged certificate. Every flagged row falls back. Stop on any mismatch.
4. A bounded alternating throughput comparison of the complete tree engine, the current complete exhaustive proxy and original CPU4 on the same already saved dense/nominal query banks, with all preparation, transfers, guards, certificates and fallback included. Report visited nodes/points and uncertainty fractions; retain outliers. Include node preparation/upload and lifetime memory costs separately.

Only measured parity and throughput can justify another full-bank replay or integration. The plausible benefit is reducing candidate distance evaluations; the blockers are warp-serial traversal, leaf/reduction overhead, irregular memory access and degenerate boxes. They remain unmeasured.
