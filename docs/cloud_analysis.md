# Analysis from a GitHub or cloud checkout

Start with [README.md](../README.md),
[simulation/pipeline_protocol.md](../simulation/pipeline_protocol.md),
[the experiment index](../experiments/README.md), and
[the published evidence index](../results/README.md).

## Available for analysis

The repository contains the current framework source and selected exact copies
of saved result tables, protocols and images under `results/`. The evidence
manifest links every copied file to its original local path and SHA256. Use
the small summaries first, then read per-task/per-object CSVs for concrete
failures. The thirteen-megabyte grasp decision file is detailed evidence;
filter it to a scene or part before discussing individual hand poses.

The published [October 28 grasp run](../outputs/cad_grasp_selection_20261028/runs/20261028T190844Z/README.md)
is also published at its original output paths. It contains all 96 case decisions
and 9,600 proposal diagnostics, including failed searches, plus the eight gallery
sheets and derived CAD validation meshes. Its publication manifest preserves
the saved evidence hashes. The gallery HTML can be opened after cloning; GitHub
displays the linked PNG sheets directly. Saved observation/calibration inputs
and installed dependencies outside that run are still required for a rerun.

The [October 29 evidence index](../results/grasp/controlled_20261029/README.md)
includes complete saved banks for the 35/96 support baseline and 37/96 finger
trial. Large JSON files are gzip copies with exact original/decompressed hashes,
not trimmed candidate subsets. Start with each summary, `decisions.csv`, report
and gallery; read a case's compressed diagnostics only when needed. Shared hand,
observed-alignment, task-goal and source-contact references and the saved stage
audit are included. Nine required receiver meshes plus the source virtual plate
and finished-hole receipt are individually allowed at original asset paths.
The receiver manifest documents exact roles/hashes. Installed libraries, weights,
raw captures and all other local assets remain separate requirements.

The pipeline is target-CAD retrieval by cosine similarity, depth localization,
calibrated RGB crops, DINO/depth matching and CAD alignment, followed by grasp
research. Some tested fitting and fusion code lives in the historically named
`experiments/nist12_qwen_clip_reranking_20261023/` directory. Its directory name
does not make Qwen or CLIP mandatory in the current framework.

The current grasp pipeline makes the final selection itself: mandatory geometry
checks, then moment arm, surface centrality, downward preference and a
deterministic candidate-ID tie break. VLM ranking/selection is not a pipeline
stage. Historical VLM studies and captions do not define the current policy.
Earlier candidate-generation and geometric-selection comparisons across eight
saved NIST scenes are complete; the new controlled comparison is complete (1,728 policy rows).
VLM selection is excluded; pickup and assembly execution remain paused.
The [implemented NIST corrections](../experiments/nist_scene_corrections_20261029/README.md)
supersede the earlier horizontal waterproof reference and wrong USB/DSUB and
M4/M12 assignments. Current task inputs use 20 mm pin depth and explicit native/
STL/board frame conversions; the current selector uses a calibrated physical
fingertip reference. Historical numerical grasp results remain unchanged.
The [preserved region-guided selector](../experiments/cad_grasp_selection_20261029/README.md)
samples origins on exact allowed CAD patches from explicit positive labels minus
protected regions. Both endpoints are filtered before the orientation bank and
100-pose cap. The complete CAD remains in rays and collision checks. Labels cover
the exposed 30 mm of each 50 mm pin after 20 mm insertion, selected connector
housing sides and external gear/nut surfaces. These are our initial training-free
annotations; they do not establish physical graspability. Missing labels fail
explicitly; unrestricted generic sampling requires an explicit mode choice.
The [preserved surface-contact refinement](../experiments/cad_grasp_refinement_20261029/README.md)
shares the one-degree alignment tolerance between proposal and contact checking.
Only the three round pins pass its cylinder-side verification; BNC retains
triangle normals. Rays start on the source CAD surface rather than at a fixed
0.5 mm inward offset. Complete CAD/hand collision geometry remains unchanged.
Independent check categories expose several simultaneous rejection reasons
instead of reporting only the first. The selected-contact gallery can also show
the complete modeled hand, part and table from the side at pickup.

The controlled previous construction selected grasps in 33/96 placements, with
368 valid poses. The final refined set selected grasps in 35/96 placements, with
495 valid poses. Each set contains 9,600 proposals. Four round-pin placements
gained a valid grasp and two waterproof placements lost one. Both use the same
corrected goals/regions, complete hand, seeds and 100-pose cap.

That preserved callable is `experiments.cad_grasp_refinement_20261029.support_pipeline.select_grasp`.
It includes a post-trial finite-pad witness correction: actual flat-area support
uses the supporting mesh-face normal, while a true line contact on a verified
round side uses its radial normal. Numerical strips around a line contact are
not treated as real flat contact area. No CAD or hand geometry was trimmed.
Only the 24 round-pin cases were rerun; the regenerated banks matched the original
banks. The final set preserves 72 unchanged cases, and its
[derivation](../outputs/cad_grasp_refinement_20261029/runs/20261029T160919Z/support_corrected/derivation.json)
records exact case sources. It has no separately measured complete-run wall time;
the original full trials and the supplemental 24-case correction have separate
timings. A post-processing table audit was also repaired to compare each
component's saved minimum with that component, preserving its earlier source
and leaving grasp decisions unchanged.

Read the [final report](../outputs/cad_grasp_refinement_20261029/runs/20261029T160919Z/support_corrected/report.md),
[verification](../outputs/cad_grasp_refinement_20261029/runs/20261029T160919Z/support_corrected/verification.json),
[placement table](../outputs/cad_grasp_refinement_20261029/runs/20261029T160919Z/support_corrected/placement_counts.csv)
and [selected-contact/pickup-side gallery](../outputs/cad_grasp_refinement_20261029/runs/20261029T160919Z/support_corrected/gallery/index.html).
The historical October 28 4/96 result used different goals and contact conventions.
No pickup or assembly improvement is measured by these geometric trials.

The preserved finger-contact callable is
`experiments.cad_fingertip_contacts_20261029.pipeline.select_grasp`.
The [finger-contact revision](../experiments/cad_fingertip_contacts_20261029/README.md)
uses existing inward-facing triangles near the ends of the saved finger collision
hulls. A static first-contact calculation fits each jaw to the complete CAD.
No pad is extended and the original complete hand geometry is retained. The
coarse collision hull bridges the detailed visual finger's central groove;
these contacts are a model approximation, not validated physical surfaces.

Its complete eight-scene trial selected 37/96 placements, with 75 valid poses
from 9,600 proposals in 230.30 s. The preserved baseline has 35/96 and 495 valid
poses: 13 placements gained a selected grasp and 11 lost one. This is a modest
net gain of two placements with fewer valid candidates under the changed
contact mode. The raw CAD contact-pair/approach banks match; actual fitted
contacts, finger opening and hand placement differ. There is no standalone
complete-run wall time for the derived baseline, so do not claim a measured
speedup against it.

The [declared N00 round 8 mm side-grip audit](../outputs/cad_fingertip_contacts_20261029/declared_side_grip_audit.json) changed its complete-hand table clearance
from −1.103 mm (entering the table) to +2.238 mm (above it). It still fails contact
alignment at about 1.107 degrees against the one-degree limit, and the palm
intersects the board at the declared final goal. The full trial selects no round
8 mm grasp in any of its eight placements. Improved table clearance alone is
not a valid grasp or assembly result.

Read the [new report](../outputs/cad_fingertip_contacts_20261029/runs/20261029T180942Z/report.md),
[selected contact measurements](../outputs/cad_fingertip_contacts_20261029/runs/20261029T180942Z/selected_contacts.csv)
and [selected contacts with pickup side views](../outputs/cad_fingertip_contacts_20261029/runs/20261029T180942Z/gallery/index.html).
VLM selection is excluded; pickup and assembly execution remain paused.
The active scene source changed, with exact old bytes retained in a recorded
hash-checked snapshot. Source CAD assets remain unchanged.

The current callable is
`experiments.cad_grasp_controlled_20261029.pipeline.select_grasp`.
It defaults to task-informed proposals and a predefined combined contact policy.
The controlled experiment compares random and task-informed proposals with pad,
finger and combined contact policies under three paired seeds. Task-informed
generation balances allowed components/shaft-length thirds, includes two explicit
task-related approach directions plus broad angles, and applies a necessary
complete-hand table bound before retaining at most 100 poses. The same complete
geometry and mandatory final checks remain in use. Combined mode assigns about
half the raw poses to each contact model before validation, rather than selecting
from a union of two already evaluated 100-pose sets.

The callable evaluates at most 100 mode-specific candidates. The comparison
evaluates both versions of the shared 100-pose bank, so up to 200 underlying
evaluations support three logical policies. The preregistered 1,728 policy-placement rows
represent 96 placements, three seeds, two strategies and three policies; they
are not independent physical trials. Contact, pickup-endpoint, final-goal/task
and joint pass counts must be read separately. Total benchmark time includes
all policies and the validation-only sensitivity, not a measured runtime for
one deployed policy. Calibration and hand-surface labels stay fixed when testing
0.5/1/2/3/5-degree validation tolerances on the exactly frozen primary random
finger poses. No retraining or new pose estimation occurs. The audited primary
combined branch selects 71/96 placements. Across three
seeds, task-informed combined selects 203/288 versus random combined 131/288.
Shared benchmark time is 2249.87 s; per-policy elapsed times remain unavailable.
Read the [full published report](../results/grasp/controlled_20261029/controlled_trial/REPORT.md),
[all policy decisions](../results/grasp/controlled_20261029/controlled_trial/decisions.csv),
and [primary gallery](../results/grasp/controlled_20261029/controlled_trial/primary/gallery/index.html).
The independent audit verifies saved counts, source hashes, rigid transforms,
priority math and full-hand table distances; it does not rerun or independently
certify target/receiver collision engines or physical contact.
The primary selection contains 43 pad and 28 finger-mode grasps. All 56 selected
finger contact points are 0.038–1.327 mm from the detailed visual finger surface,
outside its closed solid, with none within 1 micrometre. They match the declared
collision geometry. This exposes a source-model mismatch; the visual mesh is
not physical material truth. The 43 pad selections were not assessed by that
finger-only comparison.

## Questions the evidence can answer

1. Which pin sizes or connector geometries fail, and at which stage: proposal,
   shortlist retrieval, appearance matching, unavailable depth fit or geometry
   ranking? Compare fixed and introduced errors on the same saved requests.
2. When does adding DINO appearance to depth change a correct geometric match,
   and what do the illustrated single-view cases reveal about scale, visible
   cross-section, contact clearance or partial observation?
3. Which aligned-CAD placements lose every grasp under the one-degree contact
   setting, and which modeled rejection explains each loss? Compare this with
   the fifteen-degree reference without calling either condition a pickup test.
4. In the October 28 selector, which checks reject the complete poses before
   priority selection? Compare the 4/96 selected placements with earlier studies
   only after accounting for their different budgets and contact conventions.
5. In the completed comparison, how do proposed/valid
   counts, selected placements and wall time differ between its two controlled
   variants? Read each nonexclusive failure category separately; do not sum them
   as if each failed pose had only one cause.
6. Which placements gained or lost a valid grasp after contact moved onto the
   modeled finger ends? Compare measured contact coordinates and the unchanged
   full-hand checks, keeping collision-hull assumptions separate from physical
   contact accuracy.

## Interpretation requirements

- Cite the result file and exact scene/part/method when making a numerical claim.
- Keep supplied CAD identity, assumed correspondence and observed CAD pose
  distinct from a measured full retrieval-to-grasp result.
- Separate common-crop matching from native full-scene pipelines. Keep native
  failures, declared compatibility controls, ties, unavailability, absence and
  duplicate-instance ambiguity visible.
- Preserve metric scale and state fitted degrees of freedom. Single-view partial
  clouds and symmetric parts can leave orientation unresolved.
- Simulator labels and poses are audit evidence only. Do not use them to improve
  recognition, pose fitting, grasp proposals or thresholds.
- Original source hashes describe the original run. Source and asset paths were
  reorganized on October 27; old source locks do not validate the new checkout.
- The robustness scope receipt is not completed accuracy evaluation. Candidate
  grasp availability is not physical holding, pickup or assembly accuracy.

## Local data and execution limits

The twenty native-scale CAD catalog meshes are included for inspection. Large
captures, full scene/CAD-render banks, complete pose banks, model checkpoints and
installed simulator/baseline dependencies remain local. Some tests and runners
therefore require local assets that a cloud checkout does not contain. Report
missing dependencies rather than inventing inputs or claiming a reproduction.
The full interactive retrieval-to-grasp runtime has not been validated.

Exact-copied historical reports retain their original relative links. Some point
to local-only galleries; use the published evidence index for included images.

The old C: experiment workspace was removed. Required local inputs now live in
`outputs/datasets/nist_task_board/legacy_workspace_inputs/` on D:. Existing path
readers map that exact retired prefix for immutable historical records. XML
path changes have a separate verified receipt; original full-run source locks
remain historical. See [the migration record](c_workspace_cleanup_20261027.json).

Do not restore retired likelihood classifiers or non-NIST layouts. Do not
download models, start paid APIs, alter environments or command a robot as a
side effect of reading or testing the repository. This handoff supplies evidence
for analysis; it does not request fresh experiments or robot execution.
