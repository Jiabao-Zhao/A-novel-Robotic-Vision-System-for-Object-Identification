# Controlled CAD grasp generation after alignment

This study compares bounded proposal construction and three explicit simulation
contact policies on the same eight saved NIST scenes. Correct CAD identity and
the saved observation-fitted CAD alignment are supplied. The known final CAD
poses, protected mating regions, full hand and receivers are fixed. Recognition
policies are unchanged. VLM selection is excluded; robot motion, pickup,
assembly, insertion paths and release remain paused.

The current callable is `pipeline.select_grasp(request, ...)`, with defaults
`strategy='task_informed'`, `mode='combined'` and at most 100 candidates. It
returns the best valid sampled grasp under the unchanged geometric priorities,
or no valid grasp. It evaluates only the chosen policy; it does not run the two
unused benchmark arms.

The full controlled run is in
[`outputs/cad_grasp_controlled_20261029/runs/20261029T190821Z/`](../../outputs/cad_grasp_controlled_20261029/runs/20261029T190821Z/).
The frozen trial is complete: **71/96** placements retain a selected grasp in the
preregistered task-informed combined seed-0 branch. Across three seeds, combined
availability is **203/288** for task-informed and **131/288** for random proposals.
Total shared benchmark wall time is **2249.87 s**. This includes all policies,
seeds and the frozen-angle sensitivity; independent policy runtimes were not measured.

| Proposal strategy | Policy | Seed 0 /96 | Seed 1 /96 | Seed 2 /96 | Total selected /288 | Valid / evaluated candidates |
|---|---|---:|---:|---:|---:|---:|
| random | pad | 35 | 38 | 41 | 114 | 1661 / 28800 |
| random | finger | 37 | 35 | 32 | 104 | 236 / 28800 |
| random | combined | 46 | 44 | 41 | 131 | 924 / 28800 |
| task_informed | pad | 61 | 52 | 61 | 174 | 4430 / 24546 |
| task_informed | finger | 57 | 56 | 61 | 174 | 1024 / 24546 |
| task_informed | combined | 71 | 66 | 66 | 203 | 2751 / 24546 |

Read the [published report](../../results/grasp/controlled_20261029/controlled_trial/REPORT.md),
[all policy decisions](../../results/grasp/controlled_20261029/controlled_trial/decisions.csv),
[primary selected-grasp gallery](../../results/grasp/controlled_20261029/controlled_trial/primary/gallery/index.html),
and [independent audit](../../results/grasp/controlled_20261029/controlled_trial/independent_audit/verification.json).
The audit passed its saved-data, source, score, transform and table-distance checks.
It does not rerun or independently certify the target/receiver collision engines,
the per-angle acceptance decisions or physical contact. The detailed-source report
separately measures visual/collision mismatch in the primary valid finger-contact subset.

The primary selection contains 43 pad-mode and 28 finger-mode grasps. All 56
contact points from those 28 selected finger-mode poses lie outside the detailed
visual finger solid, at distances of 0.038–1.327 mm from its surface; none is
within 1 micrometre. They lie on the declared coarse collision geometry. This
is a measured disagreement between source models, not a measured hardware error
or physical validation. The 43 pad-mode selections are outside this finger-only
audit's scope. The primary round 8 mm pin retains a grasp in one of eight
placements.

The primary display is fixed in advance to task-informed combined mode, seed 0;
we do not choose the best seed after seeing the results.

## Proposal construction

Both strategies use the existing complete CAD rays, exact allowed-region
clipping, verified round-side normals where available, 200 surface-origin draws,
at most ten approach directions per pair, and a 100-complete-pose cap. Both
contacts must lie in the allowed CAD region. Protected insertion surfaces are
still part of the complete collision/ray geometry; they are not deleted.

The random reference draws origins by allowed surface area and retains a uniform
random subset of the pair/orientation pool. The task-informed strategy distributes
draws across mesh components and three equal length intervals along the mating
axis. Within each resulting region it samples by triangle area. This gives
small permitted regions a chance to contribute without changing their geometry.

For each contact pair, task-informed construction includes projected pickup-down
and final-palm-up approach directions when defined, plus broad seeded angles
around the closing axis. With ten direction slots, at least eight slots remain
for broad angular exploration. Before the pose cap, a necessary table bound
rejects a configuration only if both an exact pad endpoint bound and an
optimistic independent-jaw finger holding bound fail. It does not certify a pose
as collision free; full geometry validation still follows.

At the 100-pose cap, the target quotas are 20 pickup-down, 20 final-palm-up and
60 broad poses, balanced across component pairs, axial intervals and closing
directions. If a category lacks enough poses, the remaining eligible pool fills
the unused quota. These 20/20/60 quotas, three axial intervals and region-balancing
rules are our controlled engineering choices. They are not a paper's optimality
theorem, a learned affordance model or a demonstrated guarantee of better assembly.
No rejection-resampling loop adds surface draws or independent finger depths.
The task-informed arm changes region/length allocation, explicit approach
directions and the necessary table prefilter together. Any observed difference
belongs to that bundle of choices; this comparison cannot attribute a gain to
one component without separate ablations.

## Contact policies and the actual computation budget

| Policy | Contact interpretation | Candidate budget |
|---|---|---:|
| `pad` | Original saved finite pad surfaces and their existing calibration | At most 100 |
| `finger` | Static first touch on the original inward distal finger collision faces | At most 100 |
| `combined` | Pad/finger labels assigned before checking, balanced within proposal groups | At most 100 total; 50/50 when all 100 are retained |

The combined policy is a predefined subset. Its 50/50 split is a fixed
engineering choice, not an experimentally established optimal contact ratio.
It does not inspect two sets of valid candidates and then select from their
union. The runtime callable fits and
checks only the mode assigned to each of its at most 100 poses. The benchmark
separately evaluates both pad and finger versions of the shared raw bank, so up
to 200 underlying candidate evaluations support its three logical 100-pose
policies. It replays the combined subset from those existing evaluations.

The same complete original hand meshes remain in all checks. Finger mode fits
each jaw's first contact to the complete CAD without simulating closure or
moving the part. The coarse collision hull bridges a groove in the detailed
display mesh. A separate source-surface audit measures that discrepancy; the
display mesh is not physical calibration. Neither contact policy establishes
the actual pad material, friction, compliance or physical holding accuracy.

## Geometry stages and deterministic selection

Only explicit `pass` statuses count. Known failures, unavailable checks and
unevaluated checks remain visible. Stages retain their required category statuses:

| Stage | Required geometry |
|---|---|
| Contact | Candidate input, jaw opening, contact reconstruction, surface-normal alignment, finite contact support, complete-hand/part closing clearance |
| Pickup endpoint | Contact checks plus complete-hand table clearance |
| Final endpoint/task | Contact checks plus complete-hand receiver clearance and protected-surface compatibility |
| Joint | All contact, pickup, final endpoint and protected-surface checks |

Protected-surface compatibility is separately reported, as is contact with that
additional rule. Stage availability can come from different candidates; a grasp
is selected only if one unchanged pose passes every mandatory check. Endpoint
checks do not establish a feasible insertion path, release or physical assembly.
CAD-local and fixed-final-goal geometry may be cached under a binding of the
complete task. Pickup table clearance and approach preference are recomputed for
every observed placement.

The priority selector is unchanged. For contact midpoint `m`, mating-axis point
`a`, unit mating direction `d`, area-weighted CAD surface centroid `c_s`, and
complete CAD vertices `v_i`, its first two criteria are

\[
r_\tau=\frac{\|(m-a)\times d\|}{\max_i\|(v_i-a)\times d\|},
\qquad
r_s=\frac{\|m-c_s\|}{\max_i\|v_i-c_s\|}.
\]

It first minimizes `r_tau`, then `r_s` among numerical ties, then the pickup
approach angle to table-down, then candidate ID. The existing `1e-10` numerical
tie tolerances remain explicit. Zero CAD bounds are handled by the preserved
criterion implementation. These are fixed geometric proxies: `r_tau` is a
normalized moment arm, not measured assembly torque, and `c_s` is a surface
centroid, not center of mass. No insertion force or material density is supplied.
No weighted score or new physical optimality claim is introduced.

## Paired trial and frozen-pose sensitivity

The completed preregistered matrix is 96 placements × three paired seeds × two proposal
strategies × three contact policies = 1,728 policy-placement records. The base
seeds are `20261028`, `20262028` and `20263028`, each offset by the fixed sorted
part index. Scene/CAD bank reuse is recorded; these are not 1,728 independent
scenes, new objects or physical trials. All original failures remain in the
complete saved matrix.

The main trial fixes the hand-surface labeling, proposal-normal and validation-
normal angles separately at one degree. A 0.5/1/2/3/5-degree validation sensitivity
uses exactly the random seed-0 finger proposals, calibration and fitted poses.
It performs no new contact fits, surface draws or proposal-angle changes.
Rescued candidates whose other checks already passed are distinguished from
candidates whose other checks were previously unavailable or unevaluated.

Timing is total shared benchmark wall time for all policies/seeds and the
separately recorded validation-only sweep, excluding reporting, rendering and
publication. Work may be shared or cached across arms. Independent per-policy
elapsed times are unavailable, so this does not measure one policy's speedup.

## Evidence and publication

The [published evidence index](../../results/grasp/controlled_20261029/README.md)
preserves the preceding 35/96 support baseline and 37/96 finger-contact trial,
including all candidate banks and exact hashes. Their historical numerical
outcomes are unchanged. They use different contact models, and the derived
35/96 set has no standalone full-run wall time; they do not replace the paired
comparison in this study.

The controlled runner saves 576 shared bank directories at full size, each with
sampling metadata and both complete contact-mode diagnostics, plus all 1,728
policy selections. Existing banks are gzip files. The new post-launch
`publish_controlled.py` adapter preserves their exact compressed bytes and decoded
content hashes, compresses exact source selection/result JSON bytes, and publishes
reports, calibration, manifests, task inputs, part receipts and the primary
selected-grasp gallery. The publisher performs no generation, fitting, selection,
git staging, commit or push. See [publication.md](publication.md).

Postprocessing sources and their hashes are recorded separately from the launch
source lock; they are not presented as source frozen before the original trial.
Galleries show selected contact lines on white CAD views, with optional complete-
hand pickup side views and no final assembly views. VLM selection and physical
execution remain excluded from the evidence.
