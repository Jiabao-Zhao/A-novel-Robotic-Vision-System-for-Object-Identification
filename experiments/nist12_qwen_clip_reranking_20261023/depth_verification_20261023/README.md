# Single-camera CAD/depth verification on the twelve-product scenes

Completed 2026-10-23. This isolated follow-up reuses the eight N00–N07 scenes,
the original twelve products, middle lighting, native calibrated depth, observed
table planes, and exactly the frozen CLIP top-five shortlists. Qwen, CLIP and
DINO are not rerun. The supplied CAD is independently fitted to each of five
candidate observations: 96 target requests and 480 CAD/crop pairs.

## Results

| Selector within the same five crops | Correct /96 | Wrong | Baseline errors fixed | Previously correct made wrong |
| --- | ---: | ---: | ---: | ---: |
| Frozen CLIP + DINO | 66 | 30 | — | — |
| Zero-contact DINO + geometry, primary | 87 | 9 | 21 | 0 |
| Zero-contact geometry only | 94 | 2 | 29 | 1 |
| Zero-contact ICP RMSE only | 81 | 15 | 22 | 7 |
| Declared 1 mm clearance: DINO + geometry | 89 | 7 | 23 | 0 |
| Declared 1 mm clearance: geometry only | 95 | 1 | 29 | 0 |
| Declared 1 mm clearance: ICP RMSE only | 92 | 4 | 27 | 1 |

Every method has zero tied or unavailable task decisions. Pin correctness changes
from 20/48 to 39/48 with the primary fusion and 46/48 with zero-contact geometry.
All 48 non-pin requests are correct in both of these depth conditions, versus
46/48 for the baseline. No selector is chosen or tuned separately per product.

The zero-contact geometry regression is **N04, round 8 mm → rectangular 8 mm**.
The true crop has G=0.907214; the rectangular crop has G=0.932600. The true
round pin's measured maximum height is about 8.99 mm; the rectangular crop's is
8.00 mm, while the zero-contact round CAD predicts 8.00 mm. The capture generator
deliberately leaves a 1 mm gap. The separately declared clearance condition
resolves this regression; the primary fusion retains the correct crop even with
zero contact. The 1 mm condition is a simulator-specific sensitivity control,
not a physical-camera calibration or an estimated parameter.

**N06 round 12 mm cannot be recovered within the frozen shortlist:** its true
crop is CLIP rank eight and is never passed to depth verification. This limits
the strict five-crop protocol to 95/96. The geometry-only 1 mm condition reaches
that ceiling; it is not perfect full-scene retrieval.

## Fitting and verification

- Original full-resolution depth, original camera intrinsics and inclusive crop
  offsets; pixel convention zero and the same 2 mm plane-foreground exclusion as
  the frozen DINO preprocessing. Metric coordinates remain in the camera frame.
- Native CAD scale, without fitted rescaling. CAD-only convex-hull support
  hypotheses use a uniform-density hull center proxy, signed-axis coverage and
  additional separated faces, capped at twelve. All three round pins retain
  upright end and side hypotheses. These are finite support hypotheses, not a
  proof of dynamic stability or unrestricted full-6D search.
- Thirty iterations of continuous table-plane x/y/yaw point-to-point ICP from
  36 yaw starts per support orientation, with fixed 1 mm samples and 40,000
  deterministic area-uniform CAD samples. Exact mesh vertices define contact.
  ICP fits the partial observation against the full sampled CAD surface.
- Raycast the camera-visible CAD at every fitted hypothesis. Select its best
  pose using full-union silhouette penalties and valid-overlap depth residuals.
  Hidden CAD surfaces are not depth observations. No external-occluder carving
  or separate estimated-table penetration rejection is applied.

```text
U = observed mask union projected CAD mask
P_obs   = observed-only pixels / |U|
P_cad   = CAD-only pixels / |U|
P_depth = mean(min(abs(Z_observed - Z_CAD) / 0.005 m, 1))
          on valid observed-object / CAD overlap
G = 1 - (P_obs + P_cad + P_depth) / 3
fusion = (frozen global + frozen patch + frozen r * G) / (2 + frozen r)
```

The raycast rectangle may cover only the projected CAD; all observed pixels
outside it are still counted analytically in the full union. Regression checks
verify exact parity with explicit full-image union scoring. The 5 mm residual
scale is distinct from 1 mm fitting voxels and is not a hard rejection threshold.
Frozen `r` is old-view patch-derived coverage, not measured physical visibility
of the fitted pose. Geometry contributes at most one-third of fusion weight.

For example, N04 rectangular 8 mm has stronger geometry for the true crop
(0.948682 versus 0.859687), but its DINO score is lower (0.695088 versus
0.821284). Fusion therefore still chooses the wrong rectangular 12 mm crop
(0.779620 versus 0.834085). Geometry-only selects correctly. No post-hoc
weight tuning is performed.

## Evidence and reproducibility

- [Full results and every changed decision](analysis/index.html)
- [CAD/depth overlays, regression evidence and partial-cloud alignment](figures_v2/index.html)
- [Per-object CSV](analysis/per_object.csv), [all 672 decisions](analysis/per_task.csv),
  [paired changes](analysis/paired_changes.csv)
- [Literature review: fourteen methods and primary sources](literature.html)
- [Frozen protocol](protocol.json), [label-free inputs](inputs.json),
  [label-free selections](selections.json)
- [All-pose arithmetic verification](analysis/verification.json),
  [independent ray verification](independent_ray_verification.json)

Protocol SHA-256:
`cfcf15d79e12e820fde9b0366056f7e779c9844482e48c6eb7605d4cb4748bf6`.
The protocol protects 649 source/input files. All 305,280 saved pose records,
480 fits and 672 selections pass the pose/score/ranking replay before evaluation
labels are read. A separate implementation re-renders 126 deterministic pose
samples across sixteen fits, reproducing mask counts and G exactly. Twelve
focused geometry regressions pass in the existing WSL environment.

The separately hashed `run_parallel.py` execution adapter resumes the same frozen
fitting code with eight rather than four CPU workers; 28 completed fits are
preserved verbatim and recorded in `execution_adjustment.json`. It changes no
fitting parameter, score, input or selection. Sources and all raw fits remain
available. No shared production code, protected prompt or historical experiment
is modified. No training, model downloads, paid APIs or robot execution occur.

This is reused, known-CAD development simulation with ideal sensor depth and one
instance per product. It is not held-out real-sensor accuracy, a target-absence
test, pose ground-truth evaluation, assembly success, or the full interactive
SAM3 runtime. Scores are uncalibrated. Simulator labels are evaluation-only;
fitting reads neither object labels nor simulator poses or world transforms.

The closest literature includes FreeZe (ECCV 2024) for choosing among candidate
masks by supplied-CAD registration, NIST/IEEE ROSE 2023 for partial line-of-sight
industrial-part fitting, IEEE CASE 2019 for CAD stable-pose/table constraints,
and ECCV 2016 PPF work for projective ICP and rendered-depth/silhouette
verification. These precedents do not establish superiority on these exact pin
size tasks. FoundationPose, FreeZeV2 and other reviewed alternatives are not
benchmarked in this run.
