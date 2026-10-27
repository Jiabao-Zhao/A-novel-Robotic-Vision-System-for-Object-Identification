# CAD-to-crop matching comparisons on the eight twelve-product scenes

Completed 2026-10-23. **CLIP top-five retrieval followed by depth-only selection
remains strongest in this controlled comparison: 94/96 correct.** The best new
matching module is SAM-6D RGB at 70/96 over all twelve scene crops, or 71/96
when restricted to the same frozen CLIP top five.

These are matching-module adaptations using common inputs, not reproductions of
the papers' complete detection/segmentation systems or their published BOP scores.
Each request supplies one known CAD and independently selects its scene crop.
No all-class assignment, one-to-one scene assignment, confidence gate, NMS,
human correction or robot execution is added.

## Main results

Fixes and introduced errors below compare each method with the saved 94/96
zero-contact CLIP + depth condition. An introduced error means that switching
selectors changes a correct depth selection into a wrong selection.

| Matching method | Candidates per request | Correct /96 | Wrong | Fixes | Introduces errors | Pins /48 | Other products /48 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **CLIP -> depth only, zero contact** | 5 | **94** | 2 | — | — | **46** | **48** |
| CNOS | 12 | 67 | 29 | 1 | 28 | 20 | 47 |
| SAM-6D RGB | 12 | 70 | 26 | 1 | 25 | 22 | 48 |
| SAM-6D RGB-D | 12 | 63 | 33 | 1 | 32 | 17 | 46 |
| NIDS-Net without adapter | 12 | 65 | 31 | 1 | 30 | 19 | 46 |
| NIDS foreground averaging only, auxiliary | 12 | 64 | 32 | 1 | 31 | 18 | 46 |

All methods have zero tied or unavailable task decisions. The separately saved
1 mm simulator-clearance sensitivity condition gives CLIP + depth 95/96; it is
not the primary result or a physical-sensor calibration.

| Product | CLIP + depth | CNOS | SAM-6D RGB | SAM-6D RGB-D | NIDS no adapter |
| --- | ---: | ---: | ---: | ---: | ---: |
| Large gear | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| M16 nut | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| Round pin, 8 mm | 7/8 | 3/8 | 4/8 | 2/8 | 4/8 |
| Round pin, 12 mm | 7/8 | 3/8 | 2/8 | 2/8 | 1/8 |
| Round pin, 16 mm | 8/8 | 6/8 | 5/8 | 3/8 | 5/8 |
| Rectangular pin, 8 mm | 8/8 | 2/8 | 3/8 | 2/8 | 3/8 |
| Rectangular pin, 12 mm | 8/8 | 1/8 | 2/8 | 1/8 | 1/8 |
| Rectangular pin, 16 mm | 8/8 | 5/8 | 6/8 | 7/8 | 5/8 |
| Waterproof connector | 8/8 | 8/8 | 8/8 | 7/8 | 7/8 |
| USB connector | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| D-sub connector | 8/8 | 7/8 | 8/8 | 7/8 | 7/8 |
| BNC connector | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |

![Overall and per-object results](analysis/comparison.png)

## Exact same-five-crop controls

The new matching scores were also prospectively evaluated on the exact five
crops passed to our depth verifier. This isolates the selector from candidate
retrieval. No extra image encoding, revised shortlist or parameter tuning is used.

| Selector on the same CLIP top five | Correct /96 | Fixes depth errors | Introduces errors |
| --- | ---: | ---: | ---: |
| **Depth only, zero contact** | **94** | — | — |
| CNOS | 68 | 1 | 27 |
| SAM-6D RGB | 71 | 1 | 24 |
| SAM-6D RGB-D | 64 | 0 | 30 |
| NIDS-Net without adapter | 66 | 1 | 29 |
| NIDS foreground averaging only, auxiliary | 66 | 1 | 29 |

## Which errors change

Our primary depth condition has two wrong selections:

- **N04, round 8 mm:** correct crop 9; depth chooses rectangular 8 mm crop 10.
  CNOS, SAM-6D RGB and NIDS without the adapter select the correct crop, including
  in the same-five controls. SAM-6D RGB-D instead selects round 16 mm crop 4.
  The previously declared 1 mm clearance sensitivity also resolves this depth
  failure; the zero-contact condition remains reported unchanged.
- **N06, round 12 mm:** correct crop 8 is CLIP rank eight and absent from the
  five-crop shortlist. Full-twelve SAM-6D RGB-D selects crop 8 correctly. Its
  same-five control cannot recover the excluded crop. CNOS, SAM-6D RGB and NIDS
  select round 16 mm crop 4 in this scene.

Examples of introduced errors shared by all four main comparison methods are
**N00 round 8 mm -> round 16 mm** and **N04 rectangular 8 mm -> rectangular
12 mm**. Most new errors are wrong sizes within the same pin family. SAM-6D RGB
correctly matches every non-pin request but only 22/48 pin requests.

Adding SAM-6D's bounding-box geometry to its RGB score fixes two cases but loses
nine, reducing 70/96 to 63/96. On N04 round 8 mm, the true crop's RGB score is
0.757661 versus 0.669577 for the wrong round 16 mm crop. Bounding-box IoU instead
favors the wrong crop, 0.675641 versus 0.370692, and the combined score reverses
the decision. The true crop's fixed-template projection is 30x182 pixels while
its observed box is 79x171 pixels. This illustrates a limitation of using one
template rotation and bbox overlap as geometry evidence; it does not establish
that depth itself is unhelpful. Our verifier searches supported orientations
and compares visible surfaces. No score or view is retuned after these results.
See the [separate SAM-6D diagnostic](diagnostics_sam6d.html).

Every changed and wrong decision, its selected CAD reference, native crop,
ranked candidate scores and true identity is available in the
[full visual report](analysis/index.html). The [per-task CSV](analysis/per_task.csv),
[per-object CSV](analysis/per_object.csv), [matched-five CSV](analysis/matched_top5.csv)
and [changes by method](analysis/changes_by_method.json) retain all cases.

## What was held fixed

- Exactly N00-N07, twelve products per scene, approved middle lighting, original
  RGB colors, original single-camera depth/intrinsics and accepted localizer ROIs.
  No new scene, rendered reference, VLM description or CLIP score is generated.
- All new methods share freshly encoded cached pretrained **DINOv2 ViT-L/14**,
  FP32, without training or an adapter. All 336 input images are encoded once.
  Raw patch tokens are retained before NIDS foreground averaging.
- Exact saved tight masked 224x224 images, white backgrounds and aspect-preserving
  letterboxes are reused byte for byte. The masks come from observed depth
  distance to the fitted plane greater than 2 mm. Thus RGB denotes the matching
  score's inputs; this fixed-crop experiment does not evaluate RGB-only object
  localization.
- The intended bank is twenty existing CAD views per product. Two USB end views
  have no strict-majority foreground patch at this resolution and remain explicit
  unavailable references. All new methods share the CAD-only valid bank: eighteen
  USB views and twenty for every other product. This is decided before feature
  extraction and without scene identities or outcome-based view selection.
- Native CAD scale is fixed in meters. All 240 stored CAD-to-camera rotations
  reproduce their rendered silhouette bounding boxes within 2.5 pixels. Simulator
  scene poses, world transforms and object labels do not enter scoring.

## Matching definitions and adaptations

**CNOS:** top-five mean of nonnegative CLS cosine similarity over the available
CAD views. **SAM-6D RGB:** equal mean of that semantic score and the directional
foreground-patch matching score at the best CLS view. **SAM-6D RGB-D:** adds
visibility-weighted projected bounding-box IoU using that view's fixed rotation
and the mean observed 3D point as translation. It uses 2,048 deterministic native
CAD surface samples, integer projection/clamping, and bbox area without +1. It
does not use our table fit, orientation search or rendered surface-depth error.

**NIDS without adapter:** maximum cosine of soft-mask-weighted foreground patch
averages selects a CAD view; final proposal confidence averages that score with
patch appearance at the selected view. Its query/reference patch-mask asymmetry
and source nonzero-channel-sum denominator are retained. FFA-only is a separate
predeclared diagnostic, not substituted for the full final matching score.

Common white letterboxing, twenty-view references and fixed depth-localized
proposals differ from native full-pipeline configurations. Default detector
class-assignment filters and proposal generation are intentionally outside this
known-CAD selection test. CPU FP32 matching follows the audited arithmetic; GPU
and CPU reduction kernels can differ near zero. Source details and departures
are recorded in [CNOS](cnos_source_audit.json), [SAM-6D](sam6d_source_audit.json)
and [NIDS](nids_source_audit.json) audits. The standalone CNOS upstream commit
hash was unavailable during retrieval; its verified formula, primary links and
actual frozen shared implementation are recorded explicitly.

## Verification and reproduction

- **29 focused tests pass**, including source-arithmetic parity for NIDS and
  official SAM-6D backprojection, rotation, integer projection and bounding-box IoU.
- Predictions are frozen before this run's separate identity-evaluation process.
  **1,152 candidate pairs, 5,760 matching scores and all 960 selections** replay
  independently with zero score differences and exact projected boxes/IoUs.
  The verifier imports no trial scorer and reads no identity or simulator pose.
- All **768 protected files** pass before/after hashing. The protected CAD prompt,
  earlier results, weights, shared scoring code and active runtime are unchanged.
- Encoding takes **19.96 seconds** and completed scoring **70.82 seconds**, excluding
  model loading, preparation, source review, verification and reporting. The first
  scoring attempt was stopped before score/selection files were written because
  indexed NPZ reads repeatedly decompressed the feature archive. A separately
  [hashed execution adapter](execution_adjustment.json) caches identical arrays in
  memory and calls the unchanged frozen scorer. It changes no numerical rule.

Frozen protocol SHA-256:
`4673466470ebffa212a028e1e51e21169168512b0939902d7529c196e33e9b8d`.

See [protocol](protocol.json), [completion](completion.json),
[independent verification](independent_verification.json),
[analysis verification](analysis/verification.json) and
[analysis source hashes](analysis/source_lock.json).

Run in the existing WSL Python environment, with no package installation or model
download. `run_benchmarks.py prepare` and `encode` were run once before scoring;
`run_cached.py` completed the frozen run. Preserve these completed outputs. For
independent read-only replay, use `independent_verification.py`.

These eight previously inspected development scenes use ideal simulated depth,
known CADs and one instance per product. They establish neither held-out real
sensor performance nor superiority over the papers' complete published systems.
Absent targets, duplicate instances, assembly success and a full interactive
runtime are not evaluated. The 1 mm clearance control is simulator-specific.
