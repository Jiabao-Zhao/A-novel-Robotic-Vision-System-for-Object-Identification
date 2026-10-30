# Current NIST CAD/Depth Pipeline

1. The target request retrieves a CAD model from `CAD/cad_library.json` by cosine
   similarity between supplied text embeddings.
2. Depth localization separates observed objects. Calibrated projection provides
   the RGB crops and corresponding camera-frame partial point clouds.
3. DINO global/foreground-patch appearance and native-scale CAD/depth agreement
   identify the matching candidate and estimate the aligned CAD pose.
4. Grasp research generates and checks candidates using that aligned CAD and the
   partial observation. The pipeline then selects one final valid grasp using
   geometric criteria and deterministic priorities. If none passes, it reports
   no valid grasp. VLM ranking/selection is not part of this pipeline. Physical
   holding and assembly quality remain separate questions under investigation.

The historical October 28 grasp selector proposed at most 100 complete, calibrated
pad-center poses and rejects invalid geometry before selecting one by explicit
priority: assembly moment arm, area-weighted CAD surface centrality, then pickup
downward-angle preference. It has no mandatory downward gate, grasp grouping,
independent finger-depth sampling or weighted score. Required mating-axis,
protected-surface, receiver and calibration inputs must be explicit; missing
inputs never produce an invalid fallback. The CAD-derived eight-scene geometry
trial selected grasps in 4 of 96 placements under that revision's checks. This is
candidate availability, not pickup or assembly performance. Geometric priority
selection is the final decision, not a preliminary ranking for a VLM. Pickup
and assembly execution remain paused.

The current selector is
`experiments/cad_grasp_controlled_20261029/pipeline.py:select_grasp`.
Its default is the task-informed proposal strategy with a predefined combined
contact policy. The complete three-seed comparison contains 1,728 policy rows.
The primary task-informed combined branch selects 71/96 placements; combined
availability across three seeds is 203/288 for task-informed and 131/288 for random.
Total shared benchmark wall time is 2249.87 s, not an independent policy
runtime. The saved-result audit passed; physical success remains unmeasured. The preceding
`cad_fingertip_contacts_20261029.pipeline.select_grasp` is preserved with its
37/96 result as the finger-contact comparison.
The prior `cad_grasp_refinement_20261029.support_pipeline.select_grasp` and its
35/96 result are preserved as the modeled pad-contact baseline.
The preceding `experiments/cad_grasp_selection_20261029` implementation is
preserved as the region-guided comparison and supplies the shared task and
fingertip calibration helpers.
The finger-contact model keeps the physical fingertip pose reference separate from
actual modeled contact points on the original inward finger hull faces. Static
first-contact fitting determines where each jaw first touches the complete CAD;
it does not simulate closing or move the part. Complete finger/pad/palm geometry
is retained with no pad extension or removed vertices. Its task-input builder uses the
corrected goals and receivers rather than the old nominal-goal adapter.
Its default `assembly_region_guided` mode samples fixed-budget surface origins
from exact clipped patches: allowed surface = positive CAD labels minus protected
insertion/mating regions. The NIST builder requires explicit reviewed positive
labels; a protected-only comparison requires an explicit opt-in. Missing labels
are unavailable inputs, never an unrestricted fallback. Generic requests may
explicitly choose `legacy_unrestricted`.
Both ray endpoints must be allowed before forming the pair/orientation bank and
randomly retaining at most 100 complete poses. Rays still intersect the complete
original mesh; protected surfaces and bores are not removed. Full pad/contact,
finger/palm, table and receiver checks remain downstream, and criterion references
use the complete CAD. Partial triangles are clipped, original face IDs and mesh
hashes are retained, and no rejection-resampling loop expands the fixed budget.
For 50 mm pins, the chosen 20 mm inserted end is protected and outward shaft
sides on the remaining 30 mm are proposed. Connector housing limits and external
gear/nut labels are explicit task annotations. They are our training-free sampler
adaptation, not learned affordances or physical graspability evidence.
`experiments/nist_scene_corrections_20261029` implements vertical waterproof
mounting, USB/DSUB locations, M4/M12 hole assignments, underside bolt mounting
and derived nominal finished-hole geometry. Plate, receivers and goals share an
explicit right-handed frame. Native/STL geometry mappings and mating datums are
recorded; source revision mismatches are not called exact rigid registrations.
Current pin-tip depth is 20 mm below the board top with chamfer-first direction
and protected regions derived from those goals. The earlier 12 mm review remains
historical. CAD placements remain reference targets, not physical fit validation.

The surface-contact refinement makes the proposal and mandatory contact check use the
same one-degree normal-alignment tolerance. Only the three round pins pass the
complete constant-radius side verification for radial contact normals. BNC
retains its original triangle normals; the code does not impose a cylinder on
unverified geometry. Original CAD triangles remain in all ray and collision
queries. Rays begin on the source surface and skip only their immediate self-hit,
instead of jumping a fixed 0.5 mm inward. Verified round-side origins may move
to an actual axial mesh edge at the same sampled height; no new sampling loop
or finger-depth offset is introduced. Complete hand geometry is unchanged.

The exhaustive validator records separate input, opening, contact reconstruction,
surface-normal, pad-contact, part-overlap, table, final-receiver and protected-
surface outcomes in one traversal. These category counts are nonexclusive.
Unavailable checks cannot become passes. Selection retains the same moment-arm,
surface-centrality, downward-angle and candidate-ID priorities.

The preserved eight-scene geometric comparison is complete. The controlled previous
construction selected a grasp in 33/96 placements, with 368 valid poses from
9,600 proposals. The final refined set selected one in 35/96 placements, with
495 valid poses from 9,600 proposals: four round-pin placements gained a valid
grasp and two waterproof placements lost one. Both use the same saved alignments,
corrected goals/regions, complete hand, seeds and 100-pose cap.

After the initial full trials, the finite-pad checker was corrected to distinguish
actual flat-area support from numerical strips around a true line contact on
verified round sides. The one-degree tolerance and collision meshes did not
change. Only 24 round-pin cases were rerun, and their generated candidate banks
matched the original banks exactly. The final `support_corrected` set combines
those cases with 72 unchanged cases. It has no new standalone 96-case wall time;
initial full-trial and supplemental correction times are recorded separately.
The original trials and exact case provenance remain preserved.

Read the [final report](../outputs/cad_grasp_refinement_20261029/runs/20261029T160919Z/support_corrected/report.md)
and [selected-contact gallery](../outputs/cad_grasp_refinement_20261029/runs/20261029T160919Z/support_corrected/gallery/index.html).

The subsequent full finger-contact trial selected grasps in 37/96 placements,
with 75 valid poses from 9,600 proposals in 230.30 seconds. The comparison to
35/96 and 495 valid poses is a modest net gain of two placements: 13 gained and
11 lost a selected grasp. A different contact mode can reduce the number of
valid candidates while changing which placements retain any candidate. It does
not establish universally better contact or assembly performance. The original
CAD contact-pair/approach banks were retained; first-contact fits change the
actual modeled contacts, holding opening and resulting hand placement.

In the [separately declared N00 8 mm side-grip audit](../outputs/cad_fingertip_contacts_20261029/declared_side_grip_audit.json), complete-hand table
clearance changed from −1.103 mm to +2.238 mm. That pose still fails the one-degree
contact condition (about 1.107 degrees) and final board clearance. This isolated
height check is not a selected grasp or pickup result. The full trial retained
no grasp for the round 8 mm pin in all eight placements. The convex collision
hull bridges a groove in the detailed visual finger, so contact there remains
a model approximation requiring physical calibration.

Read the [finger-contact report](../outputs/cad_fingertip_contacts_20261029/runs/20261029T180942Z/report.md)
and [pickup gallery](../outputs/cad_fingertip_contacts_20261029/runs/20261029T180942Z/gallery/index.html).
The recorded runtime includes cached first-contact fits for identical per-part
CAD banks, geometry checking and result writes; it excludes galleries/reports.
These counts measure geometric candidate availability, not physical performance.
VLM selection is excluded; pickup, assembly, insertion and release execution
remain paused. No corrected-scene physics/capture is part of this trial.
Galleries show selected contacts from above, with optional pickup side views of
the complete modeled hand and table; they contain no final assembly views.

The controlled revision separates proposal normal tolerance, target-contact
validation tolerance and fixed hand-surface labeling tolerance. Its main trial
uses one degree for all three, while a validation-only 0.5/1/2/3/5-degree
sensitivity reuses the primary random seed's exactly frozen calibration and
fitted finger poses. It performs no new proposals or fits. Other failed or
unavailable checks cannot become assumed passes when explaining a rescued pose.

Two proposal strategies share the same 200 surface-origin and ten-orientations-
per-pair limits, allowed CAD patches and final geometry. The random strategy is
the reference. Task-informed generation balances mesh components and three equal
length intervals along the mating axis. It includes projected pickup-down and
final-palm-up approaches while retaining broad angular exploration, then balances
the retained pose bank across regions and closing directions. A common necessary
table bound rejects a pose only if neither the exact pad endpoint bound nor the
optimistic independent-jaw finger holding bound can clear the table. Full
complete-hand geometry validation remains mandatory after the 100-pose cap.
This is an experimental construction, not a claim that the new proposals improve
assembly or solve physical contact uncertainty.

Each policy evaluates at most 100 complete candidates. Pad mode uses the saved
finite-pad model; finger mode uses static first contact on the unchanged distal
inward finger hull faces. Combined mode assigns alternating pad/finger labels
within balanced proposal groups before validation, giving 50/50 when the bank
has 100 poses. It is a predefined subset, not a union selected after seeing both
models' outcomes. The callable evaluates only the requested policy. The benchmark
evaluates both pad and finger versions of the shared raw bank (up to 200
underlying evaluations), then replays the combined subset. Its logical per-policy
budget and actual benchmark work are reported separately.

The preregistered complete comparison contains 96 placements, three paired seeds, two
proposal strategies and three contact policies: 1,728 policy-placement records.
This is not 1,728 independent scenes or physical trials. Contact validity combines
local input/opening/reconstruction/normal/contact-support/closing checks. Pickup
validity additionally requires table clearance. Final-goal validity additionally
requires receiver clearance and protected-surface compatibility. Joint validity
requires all of them. Only explicit passes count; unavailable checks do not pass.
Local CAD and fixed-final-goal checks may be cached under a full task binding,
but table clearance and approach preference are recomputed for every observed
pickup pose. Total benchmark wall time includes all policies, seeds and the
separate validation-only sweep; independent per-policy elapsed time is not
measured. No CAD alignment, VLM inference, pickup, assembly, insertion or release
execution is performed.
Historical numerical
results and selector sources remain unchanged. The changed active scene source
has an exact old snapshot recorded in
`experiments/nist_scene_corrections_20261029/source_migrations.json`; do not claim
that the active file still has its historical launch hash.

NIST Task Board 1 is the only enabled scene/CAD-generation family. Individual
studies declare their chosen NIST products, sizes, scene variants and cameras.
Keep the shared simulator/camera/Panda backend and required installed assets.

The older likelihood classifier and CAD-description/SAM3 scene-selection protocol
are not the active pipeline. Optional saved baseline comparisons do not replace
the depth-localization stage or establish an end-to-end deployed system.

Use metres and explicit camera/CAD/world/robot transforms. Preserve native scale,
unavailable fits, tied candidates and symmetry/visibility limits. Ground-truth
identities, masks and simulator poses are evaluation-only. Report the searched
pose degrees of freedom, not an unsupported full-6D claim.

The cleanup changes source organization and asset paths, not saved numerical
results. The cleanup record documents removed artifacts and preserved shared
inputs. Historical manifests/source locks remain descriptions of their original
runs, not certification of the reorganized checkout.
