# Reviewable research results

This directory contains exact copies of selected saved NIST results for GitHub
and cloud analysis. No models, fitting, classification evaluation or robot actions
were run to create the bundle. Originals remain in their study folders.

[`bundle_manifest.json`](bundle_manifest.json) records each original path, byte
count and SHA256. Historical protocols and source locks describe their original
runs; copying them does not certify the current reorganized checkout as that run.

## Reading order

| Evidence | Start here | Detail available |
|---|---|---|
| Earlier CLIP retrieval and DINO selection | [Appearance summary](classification/appearance/summary.json) | Per-object counts, all decisions, paired changes, shortlist retention and within-family errors |
| DINO plus native-scale depth verification | [Depth summary](classification/depth/summary.json) | Per-task/per-object CSVs, fixed and introduced errors, fitting protocol and illustrated pin cases |
| Matching modules with common crops | [Matching summary](classification/matching_modules/summary.json) | Per-task/per-object CSVs, confusions and the common top-five control |
| Full-scene methods with their own proposals | [Full-scene summary](classification/fullscene/summary.json) | Per-task errors, proposal-stage counts, unavailable-depth and shortlist diagnostics |
| Current grasp contact checks | [One-degree contact summary](grasp/contact_alignment_1deg/summary.json) | All 7,826 saved hand-pose decisions, zero-option placements, audits and contact-witness diagnostics |
| Current contact-pair visuals | [Contact atlas](grasp/contact_atlas/index.html) | Eight scene sheets; each displays twelve CAD parts and retained sampled contact pairs |
| Earlier sampler comparison | [Reference report](grasp/reference_sampler_comparison_15deg/report.md) | Per-placement summary and quality diagnostics at the former fifteen-degree limit |
| Earlier extra contact-rule ablation | [Ablation report](grasp/contact_rule_ablation/report.md) | Saved proposals evaluated under different additional rules |

The depth [case images](classification/depth/cases/) show CAD/observation
alignment and pin failures. The [partial-cloud illustration](classification/depth/cases/partial_cloud_alignment.png)
explains the single-camera geometry. GitHub can display PNGs directly; the atlas
HTML is a small local viewer whose eight PNGs are included beside it.

## Keep these experimental scopes separate

- The recognition comparisons query a known CAD on N00–N07: eight scenes with
  twelve products. They do not evaluate upstream text-to-CAD retrieval.
- The depth follow-up uses the frozen CLIP top-five crop shortlist. Its DINO/depth
  fusion gives 87/96 correct, while its zero-contact geometry-only arm gives
  94/96. A separately declared 1 mm simulator-clearance arm is a sensitivity
  control, not physical-camera calibration. These are historical comparisons,
  not a new benchmark of the currently intended complete framework.
- Matching-module comparisons share crops. Full-scene methods generate their
  own proposals. Do not combine their denominators or treat shared-crop results
  as full-pipeline results. The saved full-scene `ours` arm has 74/96 correct
  with eighteen unavailable decisions under its historical depth-availability
  rule; diagnostics explain that gate. Native execution failures and numerical
  compatibility controls are documented in the source study README.
- Grasp experiments assume the correct CAD/crop association and use saved
  observation-fitted CAD poses. The current one-degree check retains options in
  89/96 placements. The former 95/96 result used fifteen degrees. Candidate
  availability is neither pickup success nor measured assembly precision.
- The current grasp stage order is `one_budget512` contact sampling, saved 5 mm
  endpoint-pair reduction, downward Panda conversion and modeled finite-hand,
  target/table checks with one-degree local contact selection. Review
  [the current recommendation](../experiments/cad_grasp_method_comparison_20261026/recommendation.md).
- The [retained robustness scope](classification/robustness_scope_only.json)
  contains 220 NIST inputs and 2,720 queries. Its classification accuracy
  evaluation is incomplete; this file is a scope receipt, not accuracy evidence.

## What stays local

The twenty native-scale CAD catalog meshes are included in Git. Full RGB-D
captures, point clouds, complete scene/CAD-render banks, model weights, installed
third-party environments, exhaustive geometric hypothesis banks and larger
historical galleries stay in local storage. Their absence in a cloud checkout
must be reported explicitly; do not claim to have rerun or inspected missing
inputs. Selected diagrams and contact-atlas images are included for visual review.

See [cloud analysis guidance](../docs/cloud_analysis.md) and the
[experiment index](../experiments/README.md) for source locations and assumptions.
