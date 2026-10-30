# FreeZeV2 base BOP reproduction

This isolated implementation runs **FreeZeV2-base-paper-v1-independent** with
external **CNOS-FastSAM-public-BOP23** predicted masks. The active all-seven
protocol is `protocol.json`, schema `FreeZeV2-Classic-Core-base-v2`, with output
namespace `classic_core_base_v2`. It estimates unrestricted rigid poses; no
NIST production code, table/support-face/yaw constraint, grasp generation,
training or GT-pose initialization enters inference.

**Preparation history (superseded by completed evidence below):** rendering/RGB-D mapping and synthetic registration execute.
Original DINOv2 giant and original trained GeDi passed actual CUDA feature
extraction together in the new isolated runtime. The all-seven runner conducts
fixed real validation before full inference. At that preparation stage no trained
benchmark score had been measured. The dated completed-evidence section below
now reports our measured scores; published-performance parity remains unproven.

The sealed `classic_core_base_v2` worker has now been launched with automatic
all-seven validation followed by full inference/evaluation. The final focused
suite passed **113 tests**, with one optional pycocotools interoperability test
skipped in the inference environment. See `validation/full_tests_20261030.xml`
and the run's `launch_verification.json`. The temporary continuation monitor
is `finish-freezev2-classic-core`; it remains quiet during normal progress.

All seven real correctness gates subsequently passed: 22 fixed images and59
valid pose predictions, each CSV accepted by the official parser. Full CAD
onboarding then began automatically. See `validation_summary.json` in the run
output. These checks do not establish full-target accuracy. The optional
Windows `keep_awake.py` helper currently holds a process-scoped system-sleep
request while the exact suite bridge lives; it changes no power plan and
releases its request on exit. Its state is `keep_awake_process.json`.

During full T-LESS onboarding, CAD27/28/30 exposed an original GeDi planar-LRF
division0/0:12/10/6 of5000 queries had undefined tangents at0.3diameter. Exact
CPU replay and a synthetic plane confirmed the pre-network cause. Their
targets remain explicitly blocked, and unaffected inference continues.
No fallback has changed the frozen method. See REPRODUCTION_NOTES.md and
the retained `validation/gedi_lrf_20261030/` diagnosis. Separate failure-inclusive
official evaluation will retain every official target in its denominator and
report these extraction failures alongside the measured scores.

The evaluation-only queue is now active and waits for both original OS locks;
it runs after the main inference pass releases them. Its separately sealed
source is `auxiliary_source/evaluation_v1/complete_evaluation.py` in the run
output. It reuses valid existing official metric receipts and evaluates the
remaining finalized CSVs, retaining surviving poses and every requested
instance in the official denominator. Results go to
`results_with_recorded_failures.json` and `FAILURE_INCLUSIVE_EVALUATION.md`.
Original inference receipts, CSVs and source remain unchanged. The additional
queue/launcher tests passed22+8 checks; actual evaluator-runtime imports,
source pins and partial-ledger identities were also checked without GPU work.

To resume this separate evaluation queue after verifying its prior actual
process has exited, use:

```powershell
& 'D:\Python\python.exe' 'D:\GitHub\A-novel-Robotic-Vision-System-for-Object-Identification\experiments\freezev2_bop_20261029\launch_evaluation.py' --config 'D:\GitHub\A-novel-Robotic-Vision-System-for-Object-Identification\experiments\freezev2_bop_20261029\protocol.json'
```

Track `evaluation_bridge.json`, `failure_inclusive_evaluation_process.json` and
`failure_inclusive_evaluation_status.json` with actual process/start-tick
checks. A normally finished main worker's old status timestamp is expected;
do not relaunch inference while the evaluation queue owns its locks.

See [REPRODUCTION_NOTES.md](REPRODUCTION_NOTES.md),
[METHOD_SPEC.md](METHOD_SPEC.md), [TRAINED_RUNTIME.md](TRAINED_RUNTIME.md),
[FEATURE_NOTES.md](FEATURE_NOTES.md), [REGISTRATION_NOTES.md](REGISTRATION_NOTES.md)
and [RUNTIME_NOTES.md](RUNTIME_NOTES.md).

## Resolved local paths

| Input/output | Location |
|---|---|
| Deployed experiment | `D:\GitHub\A-novel-Robotic-Vision-System-for-Object-Identification\experiments\freezev2_bop_20261029` |
| Original BOP data | `D:\BOP_SAM3_Pilot\data` |
| Assets/runtime evidence | `D:\GitHub\A-novel-Robotic-Vision-System-for-Object-Identification\outputs\freezev2_bop_20261029` |
| Active output | `outputs\freezev2_bop_20261029\classic_core_base_v2` |
| Inference Python | `/home/jiabao/.venvs/freezev2-bop-20261029/bin/python` |
| Evaluation Python | `/home/jiabao/.venvs/freezev2-bop-evaluation-20261030/bin/python` |
| BOP Toolkit | `outputs\cache\bop_toolkit`, revision `cea62d651c7e395b2e1962b9749e4e89693c6ac4` |

Existing inputs cover all3,871 BOP19 target images: LM-O200, T-LESS1000,
TUD-L600, IC-BIN150, ITODD721, HB300, YCB-V900 and125CADs. This is the Classic
Core target set, not every image in every BOP release. Original RGB/depth/CAD
files are reused without dataset copies. `assets_manifest.json` pins all seven
official localization target files and public CNOS-FastSAM masks. Missing masks
remain failures; no oracle masks or another segmentor fill them.

After cache inspection, only absent public mask/target/evaluation-model assets,
the exact162-view CNOS array, official giant weights and original GeDi
source/checkpoint were acquired. Existing ViT-S/L weights were not substituted.
`runtime_ready.json` binds the passed trained feature gate to checkpoint/source
hashes and environment provenance. No global/production environment was changed.

## Exact launch and resume commands

Run in PowerShell. Do not launch another worker while the existing supervisor
or worker is live. The launcher checks actual Windows identity and Linux OS
locks; status files alone do not establish liveness.

```powershell
$Exp = 'D:\GitHub\A-novel-Robotic-Vision-System-for-Object-Identification\experiments\freezev2_bop_20261029'
$WslExp = '/mnt/d/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/experiments/freezev2_bop_20261029'
$Run = '/mnt/d/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029/classic_core_base_v2'
$Py = '/home/jiabao/.venvs/freezev2-bop-20261029/bin/python'

# Once only, before first launch: seal exact source/config and inventory.
wsl.exe --cd $WslExp --exec $Py run_suite.py --config "$WslExp/protocol.json" --prepare

# Native Windows launcher starts the hidden bounded supervisor.
& 'D:\Python\python.exe' "$Exp\launch_suite.py" --config "$Exp\protocol.json"

# Report the durable ledger without starting inference.
wsl.exe --exec $Py "$Run/source_snapshot/run_suite.py" --config "$Run/protocol.json" --report
```

Workers use the immutable `source_snapshot`, verified on resume, and the sealed
run-root `protocol.json`. Changes require a new output namespace. For manual
recovery **only after confirming no live supervisor/worker owns this run**:

```powershell
wsl.exe --exec $Py "$Run/source_snapshot/run_suite.py" --config "$Run/protocol.json" --resume
```

Prefer the normal launcher for unattended supervision; never run the manual
worker alongside it. Runtime records include `windows_bridge.json`,
`supervisor_status.json`, `status.json`, `worker.log`, `dataset_states.json` and
`results_summary.json` and the run README. Resumes verify method/input identities before cache
reuse. Bounded resource retries do not retune inference parameters.

## Frozen base inference recipe

1. Convert native BOP CAD/depth units explicitly to metres. Sample20,000 CAD
   surface points, render162 exact CNOS orientations at480Ãƒâ€”480, require visibility
   inÃ¢â€°Â¥18 distinct views, then retain5000 points without altering the CAD frame.
2. Lift frozen giant intermediate descriptors onto visible CAD points, aggregate
   views, fit visual PCA64 on that CAD only, and apply it unchanged to observations.
   Independently normalize visual64 and GeDi32 at each .3/.4 diameter, concatenate128D.
3. Extract a square external-mask RGB crop, up to16Ãƒâ€”16 sparse depth-valid patch
   centres andÃ¢â€°Â¤3000 dense observed points. GT poses are evaluation-only.
4. Search top10 cosine matches; generate10,000 feature-aware unrestricted rigid
   hypotheses. Inliers and ICP use.03diameter. Point-to-point ICP refines the pose.
   Score = coarse-feature Ãƒâ€” refined-feature Ãƒâ€” CAD geometric-inlier terms.
5. Shortlist topN+1 same-CAD masks from this single segmentor for each official
   countN. Rank poses by the base score, apply same-CAD translation NMS, retainÃ¢â€°Â¤N.
   Segmentation confidence shortlists masks; it does not multiply the final score.
6. Retain each candidate's mask/crop/cloud, coarse/refined poses, scores, runtime
   and failure state. Export BOP CSV with object-to-camera transforms and convert
   translation metresÃ¢â€ â€™millimetres once. Missing poses remain in the full official
   target denominator.

T-LESS validation uses object1 at scene5/images7,20,25 and scene9/image0. Other
datasets use the first three target rows for their first sorted target's object,
without inspecting GT outcomes. A dataset must pass its fixed real validation
gate before full inference. Order: T-LESS, LM-O, TUD-L, IC-BIN, ITODD, HB, YCB-V.

Explicit assumptions include giant block18/no registers, input224/ImageNet
normalization, FP32/TF32-disabled arithmetic, render/crop photometry, uniform
visible-view weights, CAD-only PCA fit, Poisson/raw support details, GeDi patch
handling, triplet/pruning/ICP stopping and translation-NMS .10diameter. GeDi
sampling batch500 stays fixed; only prepared forward inputs are microbatched32
(validated fallback16). No BOP test-GT score tuning selected these choices.
This is not FreeZeV2-Accurate, SAR or a challenge ensemble. Exact parity with
the paper's unspecified CNOS variant remains unresolved.

## Official evaluation and limitations

The isolated evaluator and EGL/Vispy renderer passed analytical MSSD/MSPD plus
an actual official VSD smoke. `evaluation_runtime.json` records the toolkit,
runtime and command template. Evaluation-only views reuse linked sensor/GT data;
GT poses never enter inference.

The task is **BOP19 model-based 6D localization**: VSD, MSSD, MSPD and their
average recall. Official `visib_gt_min=-1` selects target-count most-visible GT;
there is no added0.10 visibility filter. Pose-localization AR, pose-detection AP
and2D segmentation AP are different metrics and must not be relabeled.

LM-O/T-LESS/TUD-L/IC-BIN/YCB-V support local public-GT scoring. HB/ITODD test
poses/info are benchmark-private: generate valid predictions, report local
metrics unavailable rather than zero, and use the official evaluation service
for scoring. No complete seven-dataset Classic Core aggregate is reported from
five public scores. The runner withholds a full aggregate until all seven have
compatible official evaluation receipts.

Historical `base_v1` render/synthetic evidence and failed dependency probes stay
preserved. They are not trained benchmark measurements and are not merged into
`classic_core_base_v2`. Feature validity, own valid predictions and matching
published performance require separate evidence.

## Final reporting gate (2026-10-30; initially pending, completed below)

`finalize_report.py` is a separate reporting-only utility. It writes NEW
`FINAL_REPORT.md` and `final_results.json` only when all seven exact official
image/object/count ledgers and CSVs are finalized, all five public datasets have
hash-valid failure-inclusive official metric receipts, and all seven scoped
full-pass audits and representative visual reviews exist. It starts no inference,
locks, GPU work or evaluator, and changes no frozen run source, status, CSV or
receipt. Its actual first check refused the still-partial YCB-V pass; final
artifacts are therefore not claimed here.

```powershell
wsl.exe --exec '/home/jiabao/.venvs/freezev2-bop-evaluation-20261030/bin/python' -B '/mnt/d/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/experiments/freezev2_bop_20261029/finalize_report.py' --run-root '/mnt/d/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029/classic_core_base_v2' --check-only
```

After those gates pass, run the same command without `--check-only` once. The
report separates125 available CADs from108 target-required CADs (HB16/16),
requested instances from target rows/partial poses, and official CSV-image
runtime from all-attempt active runtime. HB/ITODD metrics and a seven-dataset
aggregate remain unavailable. Passing this reporting gate is not published
FreeZeV2 performance parity, CAD classification accuracy or segmentation AP.


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
