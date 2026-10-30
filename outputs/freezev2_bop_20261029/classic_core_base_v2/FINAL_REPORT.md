# Final saved-result report: independent FreeZeV2 base

Verified 2026-10-30T10:45:39Z.

Method `FreeZeV2-base-paper-v1-independent` with `CNOS-FastSAM-public-BOP23`. All7 official target passes are recorded; individual failed/blocked targets remain explicit. This does not reproduce the authors' published performance.

| Dataset | Images | Target rows | Instances | Completed | Failed | Blocked | Saved poses | AR_VSD | AR_MSSD | AR_MSPD | AR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lmo | 200 | 1445 | 1445 | 1345 | 100 | 0 | 1345 | 46.9059% | 60.0484% | 62.7820% | 56.5788% |
| tless | 1000 | 4904 | 6423 | 3370 | 1146 | 388 | 4553 | 25.1979% | 27.1478% | 28.1815% | 26.8424% |
| tudl | 600 | 600 | 600 | 574 | 26 | 0 | 574 | 65.7783% | 71.1333% | 70.6833% | 69.1983% |
| icbin | 150 | 200 | 1786 | 125 | 75 | 0 | 1653 | 40.0588% | 35.5095% | 34.1209% | 36.5631% |
| itodd | 721 | 733 | 3041 | 509 | 224 | 0 | 2446 | private GT | private GT | private GT | private GT |
| hb | 300 | 1630 | 1630 | 1460 | 170 | 0 | 1460 | private GT | private GT | private GT | private GT |
| ycbv | 900 | 4123 | 4123 | 3994 | 129 | 0 | 3994 | 71.7216% | 74.7465% | 69.2457% | 71.9046% |

Completed means the requested number of valid-format distinct poses was saved, not GT-correct poses. Failed means fewer saved poses; surviving partial poses remain in the CSV. Every requested instance stays in the unchanged official denominator. No dummy predictions or missing-instance removal are used.

This is model-based localization with supplied official object IDs/counts, not unknown-CAD classification, segmentation AP or pose-detection AP. HB/ITODD test GT is private. No seven-dataset aggregate or five-dataset macro is reported.

125 CADs are available;108 occur in the official targets. HB has16/16 required CAD feature caches; its17 unused inventory CADs are not pending work. T-LESS has27/30 required feature caches: original GeDi nonfinite outputs caused our strict finite-output guard to reject CAD27/28/30 caches. Author handling is unknown. Frozen16/33 inventory summaries are preserved.

| Dataset | Required feature CADs | Available CADs | CSV timing images | Official CSV mean s | All-image active mean s | Completed onboarding s | Evaluator elapsed s |
|---|---:|---:|---:|---:|---:|---:|---:|
| lmo | 8/8 | 8 | 200/200 | 7.738515 | 7.738515 | 436.066 | 70.303 |
| tless | 27/30 | 30 | 996/1000 | 4.687014 | 4.669233 | 1522.690 | 664.620 |
| tudl | 3/3 | 3 | 574/600 | 1.080406 | 1.051420 | 158.095 | 68.365 |
| icbin | 2/2 | 2 | 150/150 | 8.745177 | 8.745177 | 109.113 | 194.843 |
| itodd | 28/28 | 28 | 703/721 | unavailable | 2.850241 | 2150.823 | unavailable |
| hb | 16/16 | 33 | 300/300 | unavailable | 4.329494 | 835.518 | unavailable |
| ycbv | 21/21 | 21 | 900/900 | 4.018262 | 4.018262 | 1331.997 | 158.762 |

Official timing averages only images represented by a CSV pose. The all-image column includes each attempted image's saved active work. These exclude external precomputed segmentation, model loading, CAD onboarding and evaluation. Onboarding sums saved completed stages; checkpointed partial work is separate in JSON and does not include the entire failed GeDi tail. These are not full cold-start end-to-end runtimes.

Source, model, runtime, recipe assumptions and all file hashes are in final_results.json. The primary19-file snapshot and sealed evaluation helper are verified. DINO provenance verifies selected files and a revision hint, not a complete Git tree. Checkpoint hashes are linked to recorded trained-runtime evidence; this reporting pass does not rehash multi-GB weights.

Representative evidence (visual sanity checks do not establish full6D correctness):
- lmo: [lmo_full_pass_audit_20261030.json](D:/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029/classic_core_base_v2/lmo_full_pass_audit_20261030.json); [lmo_visual_review_20261030.json](D:/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029/classic_core_base_v2/lmo_visual_review_20261030.json) (3 cases). Failure examples and per-status instance/pose counts are bound in JSON.
- tless: [tless_full_pass_audit_20261030.json](D:/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029/classic_core_base_v2/tless_full_pass_audit_20261030.json); [validation_visual_review.json](D:/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029/classic_core_base_v2/validation_visual_review.json) (3 cases). Failure examples and per-status instance/pose counts are bound in JSON.
- tudl: [tudl_full_pass_audit_20261030.json](D:/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029/classic_core_base_v2/tudl_full_pass_audit_20261030.json); [tudl_visual_review_20261030.json](D:/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029/classic_core_base_v2/tudl_visual_review_20261030.json) (3 cases). Failure examples and per-status instance/pose counts are bound in JSON.
- icbin: [icbin_full_pass_audit_20261030.json](D:/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029/classic_core_base_v2/icbin_full_pass_audit_20261030.json); [icbin_visual_review_20261030.json](D:/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029/classic_core_base_v2/icbin_visual_review_20261030.json) (3 cases). Failure examples and per-status instance/pose counts are bound in JSON.
- itodd: [itodd_full_pass_audit_20261030.json](D:/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029/classic_core_base_v2/itodd_full_pass_audit_20261030.json); [itodd_visual_review_20261030.json](D:/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029/classic_core_base_v2/itodd_visual_review_20261030.json) (3 cases). Failure examples and per-status instance/pose counts are bound in JSON.
- hb: [hb_full_pass_audit_20261030.json](D:/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029/classic_core_base_v2/hb_full_pass_audit_20261030.json); [hb_visual_review_20261030.json](D:/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029/classic_core_base_v2/hb_visual_review_20261030.json) (3 cases). Failure examples and per-status instance/pose counts are bound in JSON.
- ycbv: [ycbv_full_pass_audit_20261030.json](D:/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029/classic_core_base_v2/ycbv_full_pass_audit_20261030.json); [ycbv_visual_review_20261030.json](D:/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029/classic_core_base_v2/ycbv_visual_review_20261030.json) (3 cases). Failure examples and per-status instance/pose counts are bound in JSON.
- [Original GeDi LRF diagnosis](D:/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029/validation/gedi_lrf_20261030/GEDI_LRF_FAILURE_DIAGNOSIS_20261030.md): raw point-level nonfinite output and our whole-cache rejection are distinct.

Recovery commands are for an unexpected process failure only, after verifying all original Windows/Linux owners have exited. A finished worker's old timestamp is normal. Deterministic T-LESS cache rejection is preserved evidence, not a reason to restart or silently repair this baseline.

```powershell
& 'D:\Python\python.exe' 'D:\GitHub\A-novel-Robotic-Vision-System-for-Object-Identification\experiments\freezev2_bop_20261029\launch_suite.py' --config 'D:\GitHub\A-novel-Robotic-Vision-System-for-Object-Identification\experiments\freezev2_bop_20261029\protocol.json'
& 'D:\Python\python.exe' 'D:\GitHub\A-novel-Robotic-Vision-System-for-Object-Identification\experiments\freezev2_bop_20261029\launch_evaluation.py' --config 'D:\GitHub\A-novel-Robotic-Vision-System-for-Object-Identification\experiments\freezev2_bop_20261029\protocol.json'
```

No inference/evaluation was launched by this reporter. Frozen run reports and receipts remain unchanged.
