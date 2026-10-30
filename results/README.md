# Reviewable research results

This directory contains selected saved NIST results for GitHub and cloud
analysis. Copies preserve the exact source bytes, including after decompression
for the explicitly published October 29 grasp banks. No models, fitting,
classification evaluation or robot actions were run to publish the copies.
Originals remain in their study folders.

[`bundle_manifest.json`](bundle_manifest.json) records each original path, byte
count and SHA256. Historical protocols and source locks describe their original
runs; copying them does not certify the current reorganized checkout as that run.

## Reading order

| Evidence | Start here | Detail available |
|---|---|---|
| October 29 grasp evidence | [Saved trials and controlled follow-up](grasp/controlled_20261029/README.md) | Preserved 35/96 support-contact and 37/96 finger-contact results, all decisions and complete compressed candidate banks, shared task/contact references, stage counts, source manifests and selected-grasp galleries |
| October 28 grasp selector | [Published run](../outputs/cad_grasp_selection_20261028/runs/20261028T190844Z/README.md) | All 96 case decisions, 9,600 proposal diagnostics, derived CAD meshes, audits and eight selected-grasp sheets at the original output paths |
| Earlier CLIP retrieval and DINO selection | [Appearance summary](classification/appearance/summary.json) | Per-object counts, all decisions, paired changes, shortlist retention and within-family errors |
| DINO plus native-scale depth verification | [Depth summary](classification/depth/summary.json) | Per-task/per-object CSVs, fixed and introduced errors, fitting protocol and illustrated pin cases |
| Matching modules with common crops | [Matching summary](classification/matching_modules/summary.json) | Per-task/per-object CSVs, confusions and the common top-five control |
| Full-scene methods with their own proposals | [Full-scene summary](classification/fullscene/summary.json) | Per-task errors, proposal-stage counts, unavailable-depth and shortlist diagnostics |
| Earlier grasp contact checks | [One-degree contact summary](grasp/contact_alignment_1deg/summary.json) | All 7,826 saved hand-pose decisions, zero-option placements, audits and contact-witness diagnostics |
| Earlier contact-pair visuals | [Contact atlas](grasp/contact_atlas/index.html) | Eight scene sheets; each displays twelve CAD parts and retained sampled contact pairs |
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
  observation-fitted CAD poses. The earlier one-degree check retains options in
  89/96 placements. The former 95/96 result used fifteen degrees. Candidate
  availability is neither pickup success nor measured assembly precision.
- The earlier grasp stage order is `one_budget512` contact sampling, saved 5 mm
  endpoint-pair reduction, downward Panda conversion and modeled finite-hand,
  target/table checks with one-degree local contact selection. Review
  [its historical recommendation](../experiments/cad_grasp_method_comparison_20261026/recommendation.md).
- The October 28 selector generated 100 complete poses per placement and selected
  grasps in 4/96 placements after modeled geometry checks. It uses explicit
  criterion priorities without grouping or a mandatory downward gate. These
  results have a separate publication manifest in the run directory; the
  historical 83-file copy manifest above is unchanged. The current pipeline uses
  geometric priority selection as its final decision and excludes VLM selection.
  Later geometric generation/checking trials are documented in the October 29
  evidence index. Pickup and assembly execution remain paused.
- The October 29 publication keeps the 35/96 support result and 37/96 finger
  result as separate historical trials. The support set combines 72 unchanged
  cases and 24 corrected round-pin cases, so it has no standalone 96-case wall
  time. The finger result changes the contact mode and retains 75 valid poses
  rather than 495; those are not identical-checker counts or physical success
  rates. The separately declared controlled follow-up reports its own shared
  conditions. Every publication has a manifest of exact source/stored hashes;
  the historical 83-file `bundle_manifest.json` remains unchanged.
- The [retained robustness scope](classification/robustness_scope_only.json)
  contains 220 NIST inputs and 2,720 queries. Its classification accuracy
  evaluation is incomplete; this file is a scope receipt, not accuracy evidence.

## What stays local

The twenty native-scale CAD catalog meshes and the explicitly listed October 29
receiver/plate references are included in Git. Full RGB-D
captures, point clouds, complete scene/CAD-render banks, model weights, installed
third-party environments, most exhaustive geometric hypothesis banks and larger
historical galleries stay in local storage. The explicit October 29 publication
includes its complete grasp candidate banks as verified gzip files. Other missing
inputs in a cloud checkout
must be reported explicitly; do not claim to have rerun or inspected missing
inputs. Selected diagrams and contact-atlas images are included for visual review.

See [cloud analysis guidance](../docs/cloud_analysis.md) and the
[experiment index](../experiments/README.md) for source locations and assumptions.
