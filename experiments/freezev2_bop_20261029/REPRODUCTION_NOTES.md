# Reproduction notes and execution evidence

## Original GeDi planar-frame failure observed during the full run

At2026-10-30T04:21UTC, T-LESS geometry was30/30 complete and trained feature
onboarding27/30 complete. CAD27,28,30 failed because the original GeDi LRF
normalizes an exactly zero weighted tangent vector. An exact CPU replay found
12,10,6 affected queries respectively out of5000 at radius0.3diameter. The
successful CAD1 control had none. A synthetic exact plane and one saved
observed candidate reproduced the same division0/0 before any CUDA/network
operation. Inputs were finite; model/driver replacement cannot define this
missing tangent direction.

The batch-wide effect is specific to this implementation's strict finite-output
guard: any nonfinite descriptor row rejects that CAD feature cache (or observed
candidate), even though most rows are finite. Rejecting the whole cache is not
claimed as verified FreeZeV2 author behavior. The raw GeDi failure and our guard
policy are distinct, and resulting scores describe this declared independent
baseline. The authors' handling of such rows remains unspecified.

The frozen baseline and all finite outputs remain unchanged. No replacement
descriptor, jitter, radius change, invalid-row exclusion or tangent fallback
has been inserted. These would be additional method choices, not verified
author behavior. CAD-dependent target blocks and observed extraction failures
remain explicit in the saved ledgers and in the full official target set.

CPU-only evidence and reproducible helpers are retained under
`outputs/freezev2_bop_20261029/validation/gedi_lrf_20261030/`, especially
`GEDI_LRF_FAILURE_DIAGNOSIS_20261030.md`. The active worker continues unaffected
targets and independent datasets. A separate evaluation-only continuation
will score finalized saved predictions against the entire official target
list, including extraction failures, and label the failure counts and partial
execution clearly. It writes separate failure-inclusive receipts and does not
change the original inference or its completion claims.

## Scope and source authority

This separate experiment was explicitly authorized on2026-10-29. Root `AGENTS.md`, `README.md`, `simulation/pipeline_protocol.md`, and `experiments/README.md` were read before implementation. Existing dirty NIST/grasp/README edits and historical outputs were preserved. No production files, robot controls, global environments or remote branches were changed.

The primary specification is [FreeZeV2 v1](https://arxiv.org/pdf/2506.09784), with the [original FreeZe paper](https://arxiv.org/pdf/2312.00947) consulted separately. The author-linked repository was verified to contain website source rather than released inference code. The third-party repository, its design and both requested plans were accessible at revision `12f82a2f8afd435466a87cf2dafce404087aeb3f`; it has no verified code license, so its implementation was inspected but not copied. Its later challenge defaults were not adopted as the original base. The full per-component source/assumption/validation checklist is [METHOD_SPEC.md](METHOD_SPEC.md).

The active configuration is `FreeZeV2-base-paper-v1-independent` with
`CNOS-FastSAM-public-BOP23`, schema `FreeZeV2-Classic-Core-base-v2`. The all-seven
run uses `outputs/freezev2_bop_20261029/classic_core_base_v2`. `protocol.json`
freezes assumptions before pose-GT evaluation; configurability does not make an
assumption an author-specified setting. The earlier small `config.json` and
`base_v1` artifacts remain historical development evidence.

## Active all-seven protocol and current evidence

The deployed experiment is
`D:\GitHub\A-novel-Robotic-Vision-System-for-Object-Identification\experiments\freezev2_bop_20261029`.
It reuses all3,871 retained BOP19 target images and125 CADs across LM-O, T-LESS,
TUD-L, IC-BIN, ITODD, HB and YCB-V. Original RGB/depth/CAD inputs remain under
`D:\BOP_SAM3_Pilot\data`; no duplicate datasets were downloaded or originals
edited. All seven public CNOS-FastSAM mask files, official object/count target
lists and official evaluation models are bound by `assets_manifest.json`.

The initial trained-dependency blocker is **resolved**. Inference Python is
`/home/jiabao/.venvs/freezev2-bop-20261029/bin/python`, with Torch2.2.2+cu121,
NumPy1.26.4 and compatible original Open3D/PointNet2 operations. Missing official
giant weights and original GeDi checkpoint were acquired once after cache
searches. Original GeDi stays an external ignored dependency; no unlicensed
algorithm code was copied into tracked source. A local CUDA12.1 toolchain builds
unmodified original PointNet2 kernels for architecture8.6. No system toolkit,
driver or production environment was changed; old core versions were rechecked.

`runtime_ready.json` records source/checkpoint/extension hashes and the passed
`runtime/trained_feature_smoke_20261030_fp32_v2.json` gate. Two actual CAD renders
produced finite16Ãƒâ€”16Ãƒâ€”1536 FP32 giant maps. Sixty-four CAD queries with20,000
support points produced finite64D trained GeDi descriptors. Both models were
frozen/evaluation-only. Peak allocated CUDA memory was4,635,732,480bytes on the
8GB RTX3060Ti. Feature tests passed14/14. See [TRAINED_RUNTIME.md](TRAINED_RUNTIME.md).
This establishes feature validity, not trained pose accuracy or benchmark AR.

The base recipe uses162 CNOS views at480px, Ã¢â€°Â¥18-view visibility,5000 retained
native CAD points, visualPCA64 plus GeDi32 at.3/.4 diameter, sparse16Ãƒâ€”16 andÃ¢â€°Â¤3000
dense observed points, top10 correspondences,10,000 unrestricted feature-aware
rigid hypotheses, and.03diameter ICP/inliers. Candidate scoring multiplies
coarse-feature, refined-feature and geometric terms. TopN+1 same-CAD masks enter
from the single segmentor; translation NMS and pose scores selectÃ¢â€°Â¤N poses. GT
poses are strictly evaluation-only. No support/table/yaw/grasp constraint enters.

Assumptions remain explicit: no-register giant block18, input224/ImageNet
normalization, FP32/TF32-disabled arithmetic, rendering/crop photometry, uniform
visible-view aggregation, CAD-only PCA fit, raw20k Poisson support, GeDi
patch4000Ã¢â€ â€™512/samplingbatch500, triplet/pruning details, point-to-point ICP
stopping and translation-NMS.10diameter. Prepared network forward inputs alone
are microbatched32;16 is a validated resource fallback, with original sampling
boundaries preserved500. Maximum trained difference versus64-query forwarding
was6.09e-7 (rtol1e-5/atol1e-6). No test-GT score tuning chose these values. Exact
CNOS variant parity with the paper is unresolved; no Accurate/SAR/challenge
ensemble settings are silently introduced.

The fixed T-LESS object1 four-image validation precedes its full inference.
Each other dataset uses three target rows for a fixed first object before
expansion. The runner saves immutable source/config identities, durable image
checkpoints, per-target completed/failed/blocked states and candidate diagnostics.
Missing masks/poses remain failures in the official denominator. Resource
retries preserve the method/sampling identity. Source changes require a new run.

Exact PowerShell entry points:

```powershell
$Exp = 'D:\GitHub\A-novel-Robotic-Vision-System-for-Object-Identification\experiments\freezev2_bop_20261029'
$WslExp = '/mnt/d/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/experiments/freezev2_bop_20261029'
$Run = '/mnt/d/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029/classic_core_base_v2'
$Py = '/home/jiabao/.venvs/freezev2-bop-20261029/bin/python'
# Once only, before first launch:
wsl.exe --cd $WslExp --exec $Py run_suite.py --config "$WslExp/protocol.json" --prepare
& 'D:\Python\python.exe' "$Exp\launch_suite.py" --config "$Exp\protocol.json"
# Read/report without starting inference:
wsl.exe --exec $Py "$Run/source_snapshot/run_suite.py" --config "$Run/protocol.json" --report
# Manual recovery only when no live supervisor/worker owns this run:
wsl.exe --exec $Py "$Run/source_snapshot/run_suite.py" --config "$Run/protocol.json" --resume
```

The hidden launcher starts bounded supervision; never run the manual worker
alongside it. Verify actual process identities and Linux OS locks, not status
files alone. Immutable `source_snapshot` hashes are checked on resume. The run's
`results_summary.json`, per-dataset receipts and logs are authoritative for
progress; this document's readiness statements do not claim completed inference.

Evaluation Python is
`/home/jiabao/.venvs/freezev2-bop-evaluation-20261030/bin/python`. Its official
Toolkit/EGL/Vispy runtime passed analytical MSSD/MSPD plus an actual VSD smoke;
`evaluation_runtime.json` records its exact evidence and command. The old
evaluation-layout/environment blockers below are historical. Official BOP19
localization uses VSD/MSSD/MSPD AR, `visib_gt_min=-1`, and target-count
most-visible GT selection, **without an added0.10 cutoff**. These pose metrics
are not2D mask AP or pose-detection AP.

Five datasets have local public GT. HB/ITODD test poses/info are private:
generate their predictions, record local metrics as unavailable rather than
zero, and use official benchmark service scoring. No complete seven-dataset
Classic Core aggregate may be computed from five public scores. Before full
evaluation no real trained benchmark score had been measured in this ledger.
The dated completed-evidence section below now reports our measured scores;
matching the paper's published performance remains unproven.

## Historical pilot execution (before runtime provisioning)

The following results describe the earlier small development run, principally
`base_v1`, in the previously available interpreter. Failed trained-feature probes
and old dependency findings remain historical evidence; they do not describe
the now-ready all-seven runtime and are not trained benchmark measurements.

- Located/reused the actual research checkout and existing BOP dataset root; no duplicate dataset/CAD/model downloads.
- Selectively acquired only the missing public T-LESS masks (58,667 detections), official4,904 object/count targets, and the exact CNOS162-view camera array. Hashes/provenance are retained in `outputs/freezev2_bop_20261029/external`.
- Rendered the actual T-LESS CAD1 at162 views,480Ãƒâ€”480. Mesh:42,134 vertices/84,260 triangles. From20,000 seeded Poisson samples,8,142 pass visibility inÃ¢â€°Â¥18 views;5,000 retained. Full geometry/render cache took13.10s in the existing WSL environment. Nine representative views were visually inspected; the native axes/scale are retained and cavities/thread/connector shapes are visible.
- The native raycaster was independently checked against the software rasterizer for a rotated metric mesh: identical foreground coverage, depth difference<1e-7m, RGB differenceÃ¢â€°Â¤1 level. No table or yaw restriction is used.
- Final complete tests:64passed/1skipped on Windows (5.98s) and WSL (7.45s), including the actual official parser/preflight integration. Four orchestration tests use explicitly mocked descriptors to check real API wiring, cache visibility/PCA, single/multiple-mask flow, all score terms, artifacts and output format; they do not claim trained inference. Windows skips missing Open3D, WSL skips optional pycocotools; the alternative environment executes each missing check.
- Actual observed-input geometry:4images,7selected masks, sparse counts223/186/180/156/179/219/196; dense counts2413/2939/2933/2898/1758/2597/2485 (all available valid pixels when fewer than3000). Reprojection errorÃ¢â€°Â¤8.53e-14px. This establishes mathematical mapping, not correct segmentation or CAD identity.
- Visually inspected the unmasked crops and sparse overlays for scene5/image7 and scene9/image0, in addition to the nine CAD renders. The crops expose a lobed end-cap and a threaded diagonal component; dense overlay dots can hide their visible surfaces. The two suspect source RGBs were independently matched byte-for-byte to their pinned official archive members and have the same720Ãƒâ€”540 dimensions as the masks. No coordinate correction, GT-based mask replacement or CAD-identity accuracy conclusion was introduced.
- `verify_outputs.py` checked326cache file hashes, all162 saved depth/pose arrays, all5000 retained surface mappings and actual RGB/depth/camera hashes. Minimum visibility18; maximum retained-point distance to the original CAD mesh4.05e-8m; depth-consistency residualÃ¢â€°Â¤0.000317573m against configured0.000317576m tolerance.
- The CUDA synthetic pipeline uses200CAD points,100sparse observed points,20% observed outliers,40% CAD overlap, top10 and10,000 hypotheses. Coarse rotation error3.13e-7rad and translation3.60e-8m. Partial-overlap one-sided ICP produced0.003716rad (0.213Ã‚Â°) and0.00007427m (0.0743mm), and all three score components are retained. The small ICP shift is reported rather than disguised as an improvement. This is a synthetic descriptor diagnostic, not real learned-feature BOP inference.
- Synthetic candidate ranking suppresses a near-duplicate and rejects an unavailable pose. The generated synthetic CSV passes the actual pinned official BOP parser.
- Independent registration backend audit compares NumPy/CUDA and our ICP/Open3D. Official synthetic MSSD/MSPD tests establish identity0, known10mm X translationÃ¢â€ â€™10mm MSSD/6px MSPD, and symmetry handling. These checks do not establish full official evaluator compatibility or any dataset score.

The initial slow software CAD render was interrupted only after verifying its exact new process command/PID/Linux start ticks; its partial files remain in the parent output as debug evidence. The initial synthetic diagnostic expected exact refined recovery under partial overlap and failed; that failure/CSV are preserved. The corrected diagnostic separates coarse recovery from measured refinement drift and uses declared numerical tolerances. No BOP pose-GT feedback was involved.

Executed entry points in the existing WSL interpreter: `run.py doctor`, `render`, `synthetic`, `infer-one`, `infer`; `inspect_inputs.py`; `verify_outputs.py`; and `evaluate.py --dry-run` with the synthetic CSV. The two real inference commands both stopped explicitly at the absent giant checkpoint and retained failure JSONs, with no real-prediction CSV. The evaluator dry-run reports missing official models/target-layout and png dependency; it did not launch metrics. Native/WSL focused tests ran with the process-scoped official toolkit/support paths documented in the README. One test invocation initially used the wrong support environment-variable name; correcting that invocation made all integration tests pass without changing the implementation.

## Historical dependency blockers and unresolved fidelity

The trained assets/custom operations and evaluator issues below were observed
during the pilot and have since been resolved above. Method ambiguities remain
declared assumptions. Preserve these earlier findings rather than treating
them as the active runtime status.

Real one-mask/multiple-mask trained inference has **not completed**. Required assets are absent from the existing caches:

1. Official DINOv2 **ViT-giant/14** checkpoint. Existing S/L weights were found, verified as different architectures, and not substituted.
2. Original external GeDi source at `b3dd86776750d8221f89d39975118da9839b39f7` and trained32D checkpoint. Source `gedi.py` is pinned by SHA256 `50f23311ad3173c92f834c637e63c9c0bb6fbd8f12c58dd69773a7aba6350249`; no license was invented or its code vendored.
3. Compatible original GeDi custom operations: Pointnet2 extension, `torchgeometry`, and validated `open3d.ml.torch` integration. Existing WSL has CUDA runtime/PyTorch but no `nvcc`/CUDA development toolchain, `cmake` or `ninja`. Custom-op compatibility with PyTorch2.11/CUDA13/Python3.12 remains unproven. A dedicated compatible environment/build is needed; no shared environment was modified.

Official T-LESS metrics additionally need **official `models_eval`**, an evaluation-only layout exposing existing cached scene GT/depth/calibration and fixed target rows, and a compatible evaluator renderer/dependency environment. The toolkit declares NumPy<2 while this execution environment has2.2.6; analytical MSSD/MSPD pass, but full evaluator compatibility is not established. Vispy is missing. No full VSD run or benchmark AR/AP is reported.

Source ambiguities that remain configurable are detailed in the checklist: giant feature block/register choice, photometry/render occupancy, view aggregation weights, PCA fitting scope, CAD neighborhood support, Poisson algorithm details, triplet/pruning policy, ICP objective/stopping and translation-NMS radius. CAD rendering/visibility geometry was checked, but appearance-feature fidelity and trained GeDi compatibility still need real execution. Parameters must not be selected by repeatedly optimizing against BOP test GT.

## Current claim ledger

| Claim | Demonstrated? |
|---|---|
| Authors' saved predictions evaluated to their published score | No; not executed or used for tuning. |
| Our synthetic plumbing generates valid BOP-format rigid poses | Yes, including actual CUDA and official-format check. |
| Our native CAD rendering and observed RGB-D/mask mappings execute | Yes, on the fixed T-LESS development inputs. |
| Original frozen giant+trained GeDi produce finite actual CUDA features | Yes; real trained smoke and14 adapter tests passed. |
| All-seven public masks/targets/evaluation-model assets are ready | Yes; hash-bound assets manifest. HB/ITODD pose GT remains private. |
| Official evaluator renderer/metric runtime is ready | Yes; separate isolated environment, analytical metrics and VSD smoke passed. |
| Our own frozen giant+GeDi inference generates real BOP predictions | Yes: all-seven fixed correctness gates produced59 poses on22 images, official CSV checks passed. Full-target passes subsequently finished with recorded GeDi failures; see the dated completed-evidence section and run receipts. |
| Our own inference reproduces published performance | No. |

Generated evidence belongs outside version control under the configured output root; source/config/tests/notes stay in this isolated experiment. Cache manifests bind configuration, source files, CAD and camera-asset hashes. A changed method requires a new output root rather than mixed cached results.

## Reporting-only final evidence gate (2026-10-30; initially pending, completed below)

The separate `finalize_report.py` verifies all-seven exact finalized target/count
coverage, CSV/export bindings and five official failure-inclusive metric
receipts before writing only NEW `FINAL_REPORT.md` and `final_results.json`.
Fourteen focused CPU tests passed. Its actual current-run check refused partial
YCB-V coverage and wrote no final artifacts. Original ledgers, sources, workers,
locks, prediction bytes, metric receipts and environments are untouched.

The final report will use108 target-required CADs and125 available inventory
CADs, preserve all failed/blocked requested instances and surviving poses, and
show the official runtime's CSV-image denominator beside all-image saved active
time. It links the sealed19-file source/helper identities, honest selected-file
DINO revision-hint scope, model/runtime evidence, scoped audits, raw reviewed
artifact hashes and separate GeDi failure diagnosis. This pending gate does not
supersede dated measured receipts or claim completion; a dated final-evidence
appendix is still required after the gate passes.


## Completed evidence (2026-10-30T10:45:39Z)

The finalized [FINAL_REPORT.md](D:/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029/classic_core_base_v2/FINAL_REPORT.md) and [final_results.json](D:/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029/classic_core_base_v2/final_results.json) record our own independent frozen base inference, all seven exact official target passes, and five public-dataset official failure-inclusive evaluations. This supersedes the preparation/run-progress statements above; frozen run reports and historical numerical evidence are preserved.

Coverage is **3,871 images, 13,635 object/count target rows and 19,048 requested instances**: 11,377 target rows saved their requested pose count, 1,870 saved fewer poses and 388 remain CAD-blocked. There are 16,025 saved valid-format poses and 3,023 missing requested predictions. A completed target or valid CSV pose does not mean GT-correct 6D pose. Partial surviving poses remain in the CSV, and all requested instances remain in the full official denominator.

| Dataset | Target rows | Requested instances | Completed | Failed | Blocked | Saved poses | AR_VSD | AR_MSSD | AR_MSPD | AR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lmo | 1445 | 1445 | 1345 | 100 | 0 | 1345 | 46.9059% | 60.0484% | 62.7820% | 56.5788% |
| tless | 4904 | 6423 | 3370 | 1146 | 388 | 4553 | 25.1979% | 27.1478% | 28.1815% | 26.8424% |
| tudl | 600 | 600 | 574 | 26 | 0 | 574 | 65.7783% | 71.1333% | 70.6833% | 69.1983% |
| icbin | 200 | 1786 | 125 | 75 | 0 | 1653 | 40.0588% | 35.5095% | 34.1209% | 36.5631% |
| itodd | 733 | 3041 | 509 | 224 | 0 | 2446 | private GT | private GT | private GT | private GT |
| hb | 1630 | 1630 | 1460 | 170 | 0 | 1460 | private GT | private GT | private GT | private GT |
| ycbv | 4123 | 4123 | 3994 | 129 | 0 | 3994 | 71.7216% | 74.7465% | 69.2457% | 71.9046% |

These are official BOP19 model-based 6D localization recalls, using the unchanged object/count targets and visib_gt_min=-1 most-visible-instance semantics. They are not CAD-ID classification accuracy, 2D mask AP or pose-detection AP. Author predictions were not substituted or evaluated; published FreeZeV2 performance parity remains unproven. HB/ITODD test GT is private, local pose metrics are unavailable, no remote submission was performed, and no seven-dataset aggregate or five-dataset macro is claimed.

Only 108 of 125 available CADs occur in these localization targets. HB is complete for 16/16 required CADs; its other 17 inventory CADs are unused, so the frozen 16/33 display is not missing target work. T-LESS has 27/30 required feature caches. Original GeDi produced nonfinite point descriptors on CADs 27, 28 and 30; our strict finite-output guard rejected those entire caches, leaving 388 target rows / 432 requested instances blocked. The raw GeDi failure and our cache-wide guard are distinct; author handling is unknown. No fallback, jitter, descriptor substitution or test-GT tuning changed this frozen baseline.

Runtime columns in the final report distinguish official CSV-image means from saved all-image active work. T-LESS official timing represents 996/1000 images and TUD-L 574/600; their official means are 4.687014 s and 1.080406 s, versus all-image active means 4.669233 s and 1.051420 s. ITODD has 703/721 CSV-represented images but no official local score. These exclude precomputed external segmentation, model loading, onboarding and evaluation. Onboarding sums completed stages; checkpointed partial work does not cover the entire failed GeDi tail. These are not complete cold-start end-to-end runtimes.

The final JSON binds the 19-file primary source snapshot, sealed evaluation helper, runtime/model evidence, exact CSV/target/score bytes, recipe assumptions and raw reviewed artifacts. DINO source verification covers selected files and a revision hint, not a complete Git tree. Model checkpoint hashes are linked to recorded trained-runtime evidence; final reporting did not rehash multi-GB weights. Official Toolkit revision: cea62d651c7e395b2e1962b9749e4e89693c6ac4.

Scoped full-pass audits and representative visual reviews (the reviews are sanity checks, not complete GT verification):

- lmo: [lmo_full_pass_audit_20261030.json](D:/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029/classic_core_base_v2/lmo_full_pass_audit_20261030.json); [lmo_visual_review_20261030.json](D:/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029/classic_core_base_v2/lmo_visual_review_20261030.json).
- tless: [tless_full_pass_audit_20261030.json](D:/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029/classic_core_base_v2/tless_full_pass_audit_20261030.json); [validation_visual_review.json](D:/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029/classic_core_base_v2/validation_visual_review.json).
- tudl: [tudl_full_pass_audit_20261030.json](D:/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029/classic_core_base_v2/tudl_full_pass_audit_20261030.json); [tudl_visual_review_20261030.json](D:/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029/classic_core_base_v2/tudl_visual_review_20261030.json).
- icbin: [icbin_full_pass_audit_20261030.json](D:/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029/classic_core_base_v2/icbin_full_pass_audit_20261030.json); [icbin_visual_review_20261030.json](D:/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029/classic_core_base_v2/icbin_visual_review_20261030.json).
- itodd: [itodd_full_pass_audit_20261030.json](D:/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029/classic_core_base_v2/itodd_full_pass_audit_20261030.json); [itodd_visual_review_20261030.json](D:/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029/classic_core_base_v2/itodd_visual_review_20261030.json).
- hb: [hb_full_pass_audit_20261030.json](D:/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029/classic_core_base_v2/hb_full_pass_audit_20261030.json); [hb_visual_review_20261030.json](D:/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029/classic_core_base_v2/hb_visual_review_20261030.json).
- ycbv: [ycbv_full_pass_audit_20261030.json](D:/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029/classic_core_base_v2/ycbv_full_pass_audit_20261030.json); [ycbv_visual_review_20261030.json](D:/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029/classic_core_base_v2/ycbv_visual_review_20261030.json).
- [failure_visual_review_20261030.json](D:/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029/classic_core_base_v2/failure_visual_review_20261030.json): two visually reviewed failure traces, one absent external mask and one observed GeDi nonfinite extraction. These explain saved failure inputs without claiming GT pose accuracy or why the segmentor omitted a mask.
- [GEDI_LRF_FAILURE_DIAGNOSIS_20261030.md](D:/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029/validation/gedi_lrf_20261030/GEDI_LRF_FAILURE_DIAGNOSIS_20261030.md): original point-level LRF 0/0 and the separate cache-wide rejection policy.
- [final_reporting_test_audit_20261030.json](D:/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029/validation/final_reporting_test_audit_20261030.json): exact copy of the prior focused CPU-test and presentation-revision evidence; no tests were rerun for documentation.

Final report SHA256:`a5dacaa4d15eb13aff23dfb7d66ed02f37f834bfb3f185eb6edd67852fb72491`. Final JSON SHA256:`d05a2e2908b2fab5a21cb05c9b68b9e9b9cec18e47206cc143990034dbc0f878`.
Failure visual review SHA256:`7d3ce5d16f97864b2ef6aca3da416c857953f6309290939e17bd34ca9e10b6a9`.

Final documentation adds no inference, evaluation, tests, GPU work, environment changes or scientific source edits. Deterministic T-LESS cache failures are preserved evidence, not a reason to relaunch or silently repair this baseline.
