# Sole-proposal availability certificate: dormant draft

This is a separate prospective design, not an experiment protocol amendment or a
prediction. The human choice remains pending. Original full-bank jobs continue;
no job is skipped and no native score/cache/completion file is written here. The
module only interprets supplied records, and its tests use fabricated records.
No model, fit, image, depth input or accuracy annotation is loaded. This source
may later join the ordinary source inventory as an unused draft; it is not an
inference dependency of any current result.

## Narrow claim and proof

Fix one supplied CAD and the **complete frozen localization output containing
exactly one proposal**, p. For the frozen finite support/yaw bank H, let A be the
set of hypotheses whose exact native projection returns an available finite
score. The native candidate maximum is max(h in A) g(h), if A is nonempty.
If one member h0 of that bank has an available finite score, A is nonempty.
Thus the sole proposal is available and is the unique proposal-level argmax.
This proves neither CAD identity correctness nor physical CAD presence, absence,
duplicate count, or successful completion of the original full-bank job.

The score g(h0) is a lower bound on the native maximum. It is **never written as
that maximum**, even when it happens to equal 1. The best geometry pose/index,
global best RMSE, exact available-projection count and full bank remain unknown.
Within-bank ties do not create a proposal-level tie when there is only one
proposal; no within-bank tie or pose winner is certified.

An unavailable first projection, incomplete refinement or technical failure
leaves availability and the depth decision **unknown**. It does not prove that
all remaining projections are unavailable. The current draft has no exhaustive
unavailability certificate, multi-proposal bound or decision-only job scheduler.

## Exact first-group restriction

Only observations with more than 131,072 frozen voxel points are supported.
The original `batched.fit_pair` computes
`min(72, max(1, 131072 // observed_voxel_points)) == 1` in this case. A future
producer may execute the original first support/yaw group, with support index0,
yaw0 and clearance0m, using the identical native preparation, one-element
`refine_batch`, native returned best-RMSE pose, native `projection_score` and table
contact check. It must preserve every original point and complete projected mask.
An arbitrary initial/contact pose is insufficient. A completed native refinement
may legitimately return its initial pose after rejecting an update; its actual
completion, not whether it moved, is what matters.

No producer or partial fitter is implemented. The boolean provenance fields in
this record interpreter are **claims to be established by a separately reviewed
producer**, not independent evidence of execution. Before any integration, replay
the extracted first row against cached complete-bank row0 and compare every
non-time field/float bit using identical prepared input bytes and constants.
Finite availability must come from the original projection path, which rejects
clipping and requires a positive CAD projection and valid observed-depth overlap.

## Record schema

`binding` contains the query identifier, the full localization proposal ID list,
observed voxel count, native support count, exact source hashes, fixed constants,
and hashes of raw/voxel observations, observed mask, depth, intrinsics, plane,
CAD native mesh/vertices/points/support declarations. Hashes must encode array dtype, shape
and bytes, with explicit camera/CAD frames and metre units in the producer
receipt; native CAD object/raycaster construction and its input mesh provenance
must also be checked by that producer. `binding_sha256` hashes the canonical
record. The interpreter does not read files or verify arrays itself.

`evidence` contains that binding hash, completed-refinement/projection flags,
returned-best-RMSE-pose and exact-first-group provenance flags, and the unchanged
native row: support/yaw/clearance, T_camera_from_cad, fitting and verification.
The available projection must have a finite native geometry score, positive
projected pixels, positive valid-depth overlap and consistent native union counts.
The returned witness retains the raw row and its evidence hash. Geometry score
lower bound and its float64 hex value refer only to that witnessed row.

`clip_task` is the original query's score/shortlist/unavailable-ID record, supplied
unchanged by the trusted future adapter. For one CLIP-invalid proposal, scores
and candidate IDs are empty and unavailable IDs contain p. This fixes CLIP-only
to `unavailable_no_finite_scores` and CLIP-top5/depth to the original
`unavailable_no_CLIP_candidates`, independently of any depth witness. For a valid
sole CLIP score, CLIP-only is already exact; top5/depth still needs the witness.
The producer must bind this record to its frozen shortlist file and original
CLIP computation. This prototype only checks its internal partition.

Optional `dimensions` contains the matching binding hash, a
`complete_native_dimensions` flag, and unchanged native `score_dimensions` output.
It is exact only when the original observed dimensions and **all** native support
dimensions were evaluated under the frozen source and input bytes. It requires
no ICP; an existing exact result can be reused. A partial dimension evaluation
is unknown. A finite zero size score remains a valid exact score. This prototype
does not calculate dimensions or establish that the supplied flag is truthful.

Output decisions separate `decision_known`, native status, proposal ID, score and
`score_state`. Certified depth decisions have score=null and state=unknown;
their exact native maximum is not available. Failed evidence has
`unknown_incomplete_geometry`, never native-unavailable or zero. The record always
states full_bank_complete=false, frozen_full_bank_completion_satisfied=false,
normal_fit_cache_written=false, and no new accuracy annotation access.

## Minimal prospective evaluator changes, not implemented

1. Keep the original exact full-bank path. Add an explicitly authorized separate
   certificate input/lock inventory; do not fabricate fit summaries or gzip banks.
   A certificate cannot pass the present `prediction_files` full-bank checks.
2. At `arm_decisions`, consume the exact certified proposal/status separately from
   score ranking. Never feed a lower bound into `rank_available` as an exact score.
   Unknown decisions remain unresolved, retain their denominator, and prevent a
   claim of complete original-protocol results. After separate annotation GO,
   settled proposal decisions can support correctness/fixes/regressions and the
   same scene-cluster bootstrap without using a maximum score.
3. Score/ranked-list curves and best-pose/full-bank diagnostics remain not computed.
   At a fixed threshold t, lower_bound>=t proves passing; lower_bound<t does not
   prove rejection. With one proposal, the predicate 'at least two proposals pass'
   is false, but this never certifies the number of physical object instances.
4. The frozen plausibility overlay cannot consume the lower bound as an exact
   maximum. A shortlisted sole witness at or above tau can establish one passing
   proposal; below tau remains unknown. An empty CLIP shortlist instead keeps its
   exact unavailable flag, not zero-passing or certified absence. Unknown fields
   must be shown as unknown/not computed rather than silently counted as failures.
5. Keep original bank completion/resource markers untouched. Any later producer,
   protocol change, row0 parity, prediction locking and human execution choice
   need their own approval; the tests below supply none of those gates.

Sources inspected: original `geometry/run_depth.py:97`, `geometry/batched.py:78`,
native `geometry.py:215/270`, `geometry/dimensions.py:57`,
`evaluation/evaluate.py:75/91/130/479`, and `evaluation/plausibility.py:47`.
