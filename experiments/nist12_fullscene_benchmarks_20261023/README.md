# Full-scene CAD identification: eight scenes, twelve products

Completed on 2026-10-24 UTC using an RTX 3060 Ti (8 GB). Every method generates
its own scene masks. Our CLIP + depth pipeline gives **74 correct / 96 requests**,
with **four wrong selections and 18 unavailable decisions**. Native baseline
size filters remove many small-part proposals; the totals therefore measure
detection and selection together, not descriptor classification in isolation.

[Per-object figure](analysis/comparison.png) · [PDF figure](analysis/comparison.pdf) ·
[Every failure with boxes and masks](analysis/index.html) ·
[Per-request CSV](analysis/per_task.csv) · [Per-object CSV](analysis/per_object.csv) ·
[Exact fixes and introduced failures](analysis/paired_changes.json)

![Correct selections for every product](analysis/comparison.png)

## Results

A request supplies a CAD identity and asks for its matching scene instance.
The highest-scored retained detection of that identity is correct when its box
overlaps the target audit box with IoU at least 0.5. IoU is intersection area
divided by union area. No retained detection, an unresolved tie or an unavailable
depth decision does not count as correct. This is a target-localization measure,
not BOP average precision, pose accuracy or assembly success.

| Method | Correct / 96 | Answer exported | Fixes an unsuccessful ours query | Loses a correct ours query |
| --- | ---: | ---: | ---: | ---: |
| Ours: CLIP top five → table-constrained depth | **74** | 78 | — | — |
| CNOS | 28 | 65 | 4 | 50 |
| SAM-6D RGB identification ablation | 35 | 71 | 4 | 43 |
| SAM-6D RGB-D identification | 36 | 71 | 4 | 42 |
| Basic NIDS-Net, native | **Unavailable** | 0 | N/A | N/A |
| Basic NIDS-Net, finite-reference control | 41 | 49 | 5 | 38 |
| NOCTIS | 38 | 60 | 5 | 41 |
| NeMO | 31 | 96 | 5 | 48 |

The native NIDS run contains invalid reference features. Its zero detections
are preserved as an execution failure, not reported as measured 0% classification.
The separately declared control excludes six invalid CAD-view scores, while
retaining all classes, thresholds, finite scores, appearance scoring and NMS.
It is a numerical compatibility control, not an untouched official result.
There are no exact selection ties in any completed condition.

| Product: correct / eight scenes | Ours | CNOS | SAM RGB | SAM RGB-D | NIDS control | NOCTIS | NeMO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Large gear | 8 | 8 | 8 | 8 | 8 | 8 | 6 |
| M16 hex nut | 7 | 0 | 0 | 0 | 0 | 0 | 0 |
| Waterproof connector | 6 | 8 | 8 | 8 | 8 | 8 | 8 |
| USB connector + cable | 8 | 1 | 8 | 8 | 8 | 8 | 7 |
| D-sub connector | 6 | 7 | 7 | 8 | 8 | 7 | 0 |
| BNC connector | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| Round pin, diameter 8 mm | 7 | 1 | 1 | 1 | 0 | 1 | 0 |
| Round pin, diameter 12 mm | 7 | 0 | 0 | 0 | 0 | 0 | 1 |
| Round pin, diameter 16 mm | 8 | 2 | 2 | 2 | 5 | 4 | 3 |
| Rectangular pin, 8 mm family | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Rectangular pin, 12 mm family | 6 | 0 | 0 | 0 | 2 | 0 | 1 |
| Rectangular pin, 16 mm family | 5 | 1 | 1 | 1 | 2 | 2 | 5 |

All pins are 50 mm long. Rectangular cross sections are 8 × 7, 12 × 8 and
16 × 10 mm; the family label is not a square-width claim.

## Where the errors occur

The [stage audit](analysis/stage_counts.json) separates missing proposals from
the later matching decision. A target box is present under the same IoU ≥ 0.5
rule. These are counts of target queries with a usable proposal, not proposal
counts or classification percentages.

| Method | Target box present before native size filter / 96 | After native size filter / 96 | Final correct / 96 |
| --- | ---: | ---: | ---: |
| Ours | 96 | No additional baseline size filter | 74 |
| CNOS | 78 | 34 | 28 |
| SAM-6D, both conditions | 87 | 41 | 35 / 36 |
| NIDS control | 96 | 50 | 41 |
| NOCTIS | 96 | 50 | 38 |

Native size filtering removes the last usable target proposal in 44 CNOS
queries and 46 SAM-6D/NIDS/NOCTIS queries. This includes every M16-nut query,
many pin queries, and most BNC queries. These failures cannot establish that
their descriptors confuse these products. All native thresholds are preserved;
no small-object retuning was performed after evaluation.

NeMO predicts masks conditioned on each CAD rather than a class-agnostic bank.
It has a target-overlapping conditioned box in 83/96 queries before and after
its native 1,024-pixel component filter, but selects the correct highest-scored
component in only 31/96. It makes 36 wrong-instance selections, including twelve
wrong-size selections within a pin family, and 27 selections with weak overlap
to any audited object. Custom CAD-reference rendering remains a material
departure from its paper's PBR reference condition.

Our fresh depth localizer finds a target-overlapping box in all 96 queries.
CLIP's top five retain that box in **93/96** queries. All 18 unavailable queries
still contain the target in the shortlist, and its own depth fit is available:
another candidate has no valid projected/observed depth overlap. The frozen rule requires all five candidate
scores to be available, so it exports no answer. The 8 mm rectangular-pin
request is unavailable in all eight scenes. This is a decision-availability
failure, not eight wrong-size selections. See the exact candidates, crops and
availability flags in [ours_shortlist_audit.json](analysis/ours_shortlist_audit.json)
and the independent [availability-policy diagnosis](analysis/depth_availability_policy_diagnostic.json).

The four actual wrong selections are:

| Scene | Requested CAD | Selected physical product |
| --- | --- | --- |
| N03 | 12 mm rectangular pin | 8 mm rectangular pin |
| N04 | 8 mm round pin | 8 mm rectangular pin |
| N06 | 12 mm round pin | 12 mm rectangular pin |
| N07 | BNC connector | 12 mm rectangular pin |

CNOS and both SAM-6D conditions rescue our unavailable D-sub requests in N00/N06
and waterproof requests in N01/N03. NIDS control and NOCTIS also rescue the
N05 16 mm rectangular-pin request. NeMO rescues N00/N05 16 mm rectangular pins,
N01/N03 waterproof requests, and N06's 12 mm round pin. Every introduced failure,
including absent detections and unavailable decisions, is listed separately in
[paired_changes.json](analysis/paired_changes.json). The failure gallery shows
the requested target in green and the selected box/mask in red. Physical product
names are audit diagnostics assigned by highest mask IoU ≥ 0.1; weaker matches
remain unassigned. That rule does not determine the primary success score.

## Shared inputs and actual implementations

The frozen [input manifest](inputs.json) supplies the same full N00–N07
balanced/middle-light RGB frames, metric single-camera depth, intrinsics and
twelve native-scale colored CAD meshes. No method receives scene crops,
segmentation labels, simulator poses, world extrinsics or a forced count of
twelve detections. RGB-only methods use their required RGB subset. Models are
frozen and receive no target-specific training. Each method keeps its own
preprocessing, proposal filters, class assignment, score thresholds and NMS.

| Method | Official source pin | Own mask generation / reference condition |
| --- | --- | --- |
| [CNOS](https://github.com/nv-nguyen/cnos) | `298d1f3366171464ca271659f0e2f7a6eb8e39b4` | Native SAM ViT-H, stability 0.97; native 42-view Pyrender variant |
| [SAM-6D](https://github.com/JiehongLin/SAM-6D) | `1c2543b3b6faa1f1d81b3c7291f8b371d71e50c2` | Separate native SAM ViT-H, stability 0.85; native 42-view loader; RGB is an explicit identification-score ablation |
| [NIDS-Net](https://github.com/IRVLUTD/NIDS-Net) | `c7685a442157a1f28f2d7771e10dd9c7afdd7154` | GroundingDINO Swin-T + SAM ViT-H; adapter disabled, native appearance enabled; 42 views |
| [NOCTIS](https://github.com/code-iai/noctis) | `ba76bbc5176e2e3604bc596e656f8334188ad486` | GroundingDINO Swin-B + Ultralytics SAM2.1-L; released Pyrender configuration; 42 native slots / 21 distinct views |
| [NeMO](https://github.com/DLR-RM/NeMO) | `76478815baf2c99002cccbee98426773fa4e59cd` | Native CAD-conditioned full-image detector; 32 fixed custom object-centric CAD views |
| Ours | Existing localizer and frozen prior trial scorer | Fresh camera-frame depth proposals, unchanged Qwen/CLIP image-and-text top five, table-constrained native-scale x/y/yaw fitting and depth-only score |

These runs call official model, preprocessing and scoring components through
small experiment drivers. They do not claim stock command-line execution or
exact BOP paper reproduction. The CAD renderer is the released CNOS Pyrender
variant with original colors, not the papers' public BOP PBR onboarding data.
NeMO's released loader expects pre-made object-centric references; its custom
onboarding is documented separately. A raw-background onboarding attempt was
corrected before labels and remains preserved as a diagnostic, not a condition
chosen by accuracy. SAM-6D's pose-estimation stage is outside this identification
test. Native SAM-6D projection retains the USB mesh's off-center original origin;
it does not receive our support-orientation fitting or ICP.

Our method retains the previous zero-contact table geometry arm: CAD scale is
fixed; CAD-only support faces and yaw seeds initialize table x/y/yaw fitting;
the visible CAD silhouette uses full mask union and valid-overlap depth
disagreement. No new penetration rejection or external occlusion carving is
added. This is the user's earlier CLIP + depth experimental condition, not the
separate active human-guided SAM3 branch protocol. The old supplied-crop 94/96
result is not reused as a full-scene baseline.

## Numerical failure, GPU use and verification

Six USB CAD masks have nonempty 224 × 224 support but disappear under native
NIDS bilinear mask sampling to 16 × 16. Its unguarded foreground average divides
by zero. Max over views and classes propagates NaNs to all queries. Independent
CPU replay verified the loader and this cause; scene query features are finite.
The [native diagnosis](nids_noctis/nids_zero_mask_diagnostic.json),
[independent audit](cnos_sam6d/nids_reference_audit.json) and per-scene control
receipts preserve the exact evidence. The control was declared without labels.

Neural models and image encoding run on CUDA, sequentially to fit 8 GB. Native
SAM decoder batches and descriptor batches are reduced without altering inputs
or thresholds. The peak recorded SAM allocation is 5.61 GiB. Our 480 depth fits
run on eight CPU workers for 850.8 seconds; no GPU geometry solver is substituted.
The 704 Qwen descriptions include 508 exact image/prompt cache hits and 196 fresh
local requests. No paid API or GPT model is used.

New Python packages and compatibility libraries are scoped to this trial;
the global environment is unchanged. GroundingDINO uses its shipped PyTorch
deformable attention on CUDA because the compiled extension is unavailable.
Reference renders, source archives, checkpoint receipts and failed compatibility
probes are retained. See implementation details in
[CNOS/SAM-6D](cnos_sam6d/README.md), [NIDS/NOCTIS](nids_noctis/README.md) and
[NeMO](nemo_runner/README.md).

All predictions were [hashed and frozen](predictions_lock.json) before this
evaluation read audit identities. Four synthetic evaluator tests, four native
CNOS/SAM tests and eighteen NIDS/NOCTIS checks pass. Independent replay verifies
all 768 CAD requests and paired results. Native CNOS/SAM math and exports replay
exactly; all 480 stored depth-fit records and 96 final decisions were audited.
The depth search evaluated 152,640 pose hypotheses.
All 768 protected historical files are unchanged, including the locked CAD-view
prompt. The active runtime is unchanged.

Key verification receipts:
[native CNOS/SAM outputs](cnos_sam6d/native_output_audit.json),
[ours replay](cnos_sam6d/ours_replay_audit.json),
[finite-reference NIDS replay](cnos_sam6d/nids_finite_output_audit.json),
[NeMO outputs](nemo/verification.json),
[independent evaluation](analysis/independent_results_verification.json),
[audit camera alignment](analysis/audit_scene_alignment_verification.json),
[historical preservation](historical_source_verification.json).
The exact initial evaluator is retained beside its completion record; the
[subsequent lock recheck adjustment](analysis/evaluator_recheck_adjustment.json)
allows new audit receipts without changing or extending frozen predictions.

## Limits and read-only reproduction

These are eight reused development simulator scenes with ideal depth, known
CADs and one instance per product. Results do not establish real-camera,
held-out-CAD, duplicate-instance, absent-target or assembly performance. Custom
references, small image footprints and native size filters materially affect
the baseline scores. Any future small-part filter adaptation should be a new
declared condition with separate outputs; none was selected using these labels.

Completed predictions should be preserved. Read-only audits are
`cnos_sam6d/audit_native_outputs.py`, `cnos_sam6d/audit_ours.py`, and
`nemo_runner/independent_results_audit.py`. `evaluate.py`, `audit_stages.py` and
`plot_results.py` regenerate analysis from frozen predictions; they never call
the models. The first evaluator verifies the prediction lock before using labels.

## User-directed scope for the next experiment — 2026-10-24

Exclude the robot gripper regions from the product candidate pool before CLIP
retrieval and CAD/depth verification. Keep all 96 product queries in the
denominator; excluding robot hardware does not remove any target product.
This is a prospective scope decision. The completed full-scene predictions,
original scores and 18 recorded abstentions remain unchanged. No new run or
gripper-exclusion implementation has been performed under this instruction.

The previous benchmark driver calls voxel downsampling, table-plane removal,
DBSCAN and proposal checks directly. It does not call `clean_raw_cluster`.
The statistical/radius cleanup in the shared localizer is called by
`save_outputs` when exporting point clouds, after proposal rectangles have
already been formed. It does not serve as a gripper-exclusion rule. In N00,
the two gripper clusters have 161 and 146 downsampled points (minimum 50),
smallest 3D extents about 13.6 and 13.5 mm (minimum 6 mm), and large image
rectangles. They pass the current cluster/edge checks. No workspace crop or
robot-region exclusion was active in this full-scene benchmark.
