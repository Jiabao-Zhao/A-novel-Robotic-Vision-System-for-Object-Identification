# Source-only optimization feasibility audit

Date: 2026-10-24. This note records a read-only analysis of frozen source. No
optimization was implemented, no fitting job was skipped, no model was called,
and no robustness accuracy annotation was read for this audit. Original complete
searches remain authorized and required. This note does not amend any protocol,
evaluation source, calibration, prediction, or completion requirement.

## Frozen computation and evidence contract

The task is **retrieved CAD to scene proposal**, rather than choosing a CAD for
each proposal. The native search refines every declared support-face/36-yaw
initialization, then ray-scores the best-RMSE pose returned by each refinement.
It takes the maximum available geometry score over that bank. An arbitrary
initial/contact pose is not a member of the scored bank unless refinement
returns it. One unsuccessful seed does not prove candidate unavailability.

Sources:

- [Original protocol](../protocol.json): frozen contact/native-scale search,
  availability, diagnostics, and resource policy.
- [Native geometry](../../nist12_qwen_clip_reranking_20261023/depth_verification_20261023/geometry.py),
  `contact_pose` at line 155, `refine_pose` at 166, `score_projection` at 215,
  `projection_score` at 270, and `fit_pair` at 301.
- [Batched execution](../geometry/batched.py), lines 92–123: complete declaration
  order, refinement, projection, best geometry/RMSE records, and full pose bank.
- [Fit runner](../geometry/run_depth.py), lines 99–125: summary, size control,
  compressed hypothesis archive, and archive SHA256.
- [Evaluator](evaluate.py), lines 163–177: all CAD/proposal pairs and referenced
  full hypothesis archives are required in the prediction inventory. Lines
  115–127 verify and deduplicate bank payload hashes.

CLIP-unavailable descriptions do not disable `depth_all` or `size_only`.
`arm_decisions` in [evaluate.py](evaluate.py), lines 91–100, ranks their scores
independently. A single proposal can therefore yield the same selected proposal
for several supplied CAD queries. This is a property of the ranking task, not
evidence that the proposal has every queried identity or that a CAD is present.

## A. Options that can preserve complete banks

These are execution opportunities, not measured speedups or implemented changes.
They need archived source changes and focused full-bank parity checks, including
unavailability branches. They do not require removing hypotheses or diagnostics.
Wall times may change; all native numerical results and non-time evidence fields
must remain intact.

1. **Reuse exact observation preparation.** `observation` already computes the
   voxel array in [run_depth.py](../geometry/run_depth.py), line 43, and stores it
   at line 99. Nevertheless, the fit receives raw points at line 103 and
   [batched.py](../geometry/batched.py) voxelizes those points again. Passing the
   already computed, byte-identical array into an explicitly prepared-input
   execution path could remove repeated voxel sorting/averaging. Reusing exact
   observed dimensions and the observed mask likewise removes repeated work.
   It must not apply voxelization again to voxel means, change summation order,
   substitute another centroid, downsample further, or change observed pixels.

2. **Return provably identical unavailable projections before raycasting.**
   [CadSurface.render](../../../scripts/surface_verification.py), lines 113–139,
   computes projected vertex bounds and a clipping flag, then raycasts. Native
   `projection_score` subsequently discards clipped renders at geometry.py line
   293. Reproducing the exact preflight can avoid those discarded raycasts.
   Preserve branch order: a behind-camera mesh or empty image intersection
   returns `None` before the caller handles clipping. Principal-point offsets,
   floor/ceil expansion, supplied image dimensions, and exact boundary tests
   must match. An available projection must still receive its complete native
   raycast and score.

3. **Memoize genuinely identical projection evaluations.** Byte-identical CAD,
   final pose, intrinsics/ROI convention, depth and observed mask permit reuse
   of a deterministic projection result. Different ICP histories remain in
   their original bank rows. Similar poses, approximate symmetry, rounded
   transforms and matching RMSE are insufficient cache keys.

The runner already reuses exact observation/CAD fitting inputs across paired
RGB-only conditions. Its threaded nearest-neighbor execution has separate parity
evidence. Additional benefit from the options above remains unmeasured.

Geometric symmetry alone does not justify removing yaw seeds: the frozen random
CAD samples, voxel means, nearest-neighbor ties and floating-point refinement need
not be symmetric even when the mesh is.

## B. Classification-decision certificates

These can preserve a selected proposal in specified cases, but do **not** satisfy
the present full-bank contract. They require a separately archived prospective
supplementary protocol, certificate records, and evaluator support. Unknown
results must never be encoded as unavailable, zero, or the witnessed lower score.

### Exactly one proposal

For each supplied CAD, one completed frozen seed whose returned best-RMSE pose
has an available native projection proves that the maximum over the full search
is available. Because only one proposal exists, its `depth_all` selected proposal
and `ok` status are then fixed, regardless of the remaining scores. This does not
prove identity correctness, CAD presence, absence, or duplicate count.

An availability witness must prove positive finite CAD projection, no native
clipping, and at least one valid observed-depth overlap pixel. In principle, a
ray at an observed pixel could witness overlap after the native pose is fixed;
any such partial-ray implementation would itself require exact-ray parity and
prospective evidence rules. No arbitrary pre-ICP pose or failed seed suffices.
If every attempted seed fails, availability remains unknown until the rest are
exhausted or separately proved unavailable.

If the sole proposal has no CLIP-valid descriptions and the task has no candidate
IDs, both CLIP arms already have exact unavailable results. This creates no
geometric proof by itself. The size control uses exact observed footprint/height
and CAD support dimensions and can be evaluated without ICP; see
[dimensions.py](../geometry/dimensions.py), lines 37–78.

### Several proposals

Let `L_w` be an available witnessed lower bound on the maximum score of candidate
`w`, and let `U_j` bound the complete maximum of each competitor. The winner is
certified only when

\[
L_w-U_j>10^{-12}\quad\text{for every competitor }j,
\]

with additional conservative numerical enclosure. A current low score, poor
RMSE, mismatched dimensions, or one losing support face does not bound unexplored
hypotheses. An unresolved interval close to the tie tolerance does not certify
either a unique winner or an exact tie.

### Which diagnostics remain exact?

| Output | Status with decision certificates |
|---|---|
| Selected proposal and primary availability/tie status | Exact only when individually certified; failed attempts alone leave status unknown. |
| Primary correctness and per-object/cohort accuracy | Exact after authorized annotation access, if every corresponding decision/status is certified. No accuracy was computed here. |
| Fixed/introduced errors, paired-condition changes and scene-cluster bootstrap | Exact from certified decisions, with the same query denominators and resampling. |
| Proposal recall and CLIP recall | Unchanged; their predictions do not depend on unfinished geometry. |
| Candidate-level depth availability | Exact for a witnessed available candidate; a skipped or unsuccessful candidate remains unknown. Single-proposal witnesses settle that candidate's availability. |
| Target-depth availability among recalled proposals/top five | Exact only when the required existential availability statement is certified; unknown losing candidates can matter in multi-proposal scenes. |
| Exact maximum score and complete ranked score lists | Unknown unless bounds collapse to the native exact result. A witness score is only a lower bound. |
| Best geometry pose/index, best RMSE across the bank, full histories, number of available projections | Generally unknown for skipped hypotheses. |
| Full hypothesis archive and its integrity record | Incomplete; cannot claim the original archive contract was met. |
| Fixed threshold curves | Exact only at thresholds resolved by every relevant score interval; otherwise interval-valued or explicitly not computed. Candidate-count curves also need every relevant candidate. |
| Known12 CLIP-top-five plausibility overlay | Exact only with the required availability statuses and proof that every shortlisted maximum is either at least frozen tau or below it. A witnessed score at least tau proves passing; failure to witness one does not prove non-passing. T-LESS/new-CAD cohorts remain outside the confirmatory overlay. |

The evaluator currently consumes exact scores in `rank_available` at line 75,
uses all ranked scores for threshold/count curves at lines 479–506, and reports
candidate depth availability at lines 566–570. The supplementary overlay's exact
availability and threshold requirements are in [plausibility.py](plausibility.py),
lines 47–66. These fields cannot silently accept incomplete certificates.

## Conservative bounds for large observed regions

Let `O` be the full observed foreground mask and `C` a native projected CAD mask.
For an available pose, the frozen score is algebraically

\[
g=1-\frac{p_{obs}+p_{cad}+p_{depth}}3
 =\frac{2+J-p_{depth}}3,\qquad
J=\frac{|O\cap C|}{|O\cup C|}.
\]

Since the clipped depth penalty is nonnegative, any certified pixel envelope
`E` containing `C` yields

\[
g\le\frac{2+|O\cap E|/|O|}{3}
\le\frac{2+\min(1,|E|/|O|)}3.
\]

Thus a small CAD envelope within a very large observed region can have a useful
upper bound. It is not automatically unavailable or zero: this upper bound
approaches two thirds as the overlap fraction becomes small. Observed pixel
counts, not observed voxel counts, belong in the denominator.

**After refinement:** the exact rectangle that native rendering would raycast
is a certified envelope, because the native mask cannot contain pixels outside
that rectangle. Its overlap count can be obtained without rays. A per-pose bound
can avoid a losing raycast in a decision-only search, but refinement is still
needed and the omitted pose's verification fields remain unknown. A complete
candidate bound must cover every unfinished pose.

**Possible before-refinement envelope, not implemented or numerically proved:**
let `mu` be the exact voxel-cloud mean, `q` its projection onto the estimated table
plane, `D` a conservative native-CAD diameter bound, and `H` the height for one
support face. In real arithmetic every initial/accepted ICP pose places its
tangential CAD extent within radius `D` of `q`, with height in `[0,H]`: the
in-plane matched centroid is aligned to `mu`, and that centroid lies in the
native CAD convex hull. A CAD bounding-box diagonal can conservatively bound its
diameter. The resulting cylinder encloses all yaw/refinement outcomes for that
face. Projecting it gives an envelope if its camera-Z lower bound is positive;
otherwise the whole image remains the safe fallback. All support faces must be
covered.

Before this could support any pruning, proof work is required for floating-point
rotation/translation accumulation, conservative interval division and outward
rounding, voxel/vertex convex-hull bounds, exact pixel and clipping conventions,
all support faces, and the strict tie margin. No current low-score observation or
estimated runtime establishes those properties.

## Disposition

Continue the full original search. Options in A are candidates for separately
approved execution changes with parity evidence. Options in B are incomplete
evidence certificates under a possible supplementary protocol, not replacements
for the frozen exact scores and complete banks. No speedup, dataset count of
eligible cases, numerical equivalence of an unimplemented optimization, or new
classification result is claimed by this note.
