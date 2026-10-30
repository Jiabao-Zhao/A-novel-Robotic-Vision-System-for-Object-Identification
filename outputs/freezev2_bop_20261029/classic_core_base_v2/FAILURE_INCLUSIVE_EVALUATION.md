# Failure-inclusive evaluation of the unchanged FreeZeV2 baseline

All requested instances stay in the unmodified official denominator. Surviving saved poses are retained; missing predictions reduce the score. No dummy/zero predictions are inserted. Original execution limitations remain explicit; no paper-performance parity claim.

Method: `FreeZeV2-base-paper-v1-independent`; masks: `CNOS-FastSAM-public-BOP23`. Exact original settings: `/mnt/d/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029/classic_core_base_v2/protocol.json` (SHA256 `33b805086fdad5c37f7a3619209a37f83b10f6b701b814f078d99c5cddc0006c`).

No inference was rerun or altered. Original status, CSVs and primary evaluation receipts remain unchanged. A measured score does not establish that strict inference execution completed.

| Dataset | Expected targets | Attempted | Completed | Failed | Blocked | Missing | Poses | Official AR | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| lmo | 1445 | 1445 | 1345 | 100 | 0 | 0 | 1345 | 56.5788% | complete_with_recorded_inference_failures |
| tless | 4904 | 4904 | 3370 | 1146 | 388 | 0 | 4553 | 26.8424% | complete_with_recorded_inference_failures |
| tudl | 600 | 600 | 574 | 26 | 0 | 0 | 574 | 69.1983% | complete_with_recorded_inference_failures |
| icbin | 200 | 200 | 125 | 75 | 0 | 0 | 1653 | 36.5631% | complete_with_recorded_inference_failures |
| itodd | 733 | 733 | 509 | 224 | 0 | 0 | 2446 | unavailable | unavailable_private_GT |
| hb | 1630 | 1630 | 1460 | 170 | 0 | 0 | 1460 | unavailable | unavailable_private_GT |
| ycbv | 4123 | 4123 | 3994 | 129 | 0 | 0 | 3994 | 71.9046% | complete_with_recorded_inference_failures |

| Dataset | Geometry CADs | Feature CADs | Completed onboarding active s | Partial checkpointed active s | Saved inference active s | Evaluation elapsed s |
|---|---:|---:|---:|---:|---:|---:|
| lmo | 8/8 | 8/8 | 436.066 | 0.000 | 1547.703 | 70.303 |
| tless | 30/30 | 27/30 | 1522.690 | 108.607 | 4669.233 | 664.620 |
| tudl | 3/3 | 3/3 | 158.095 | 0.000 | 630.852 | 68.365 |
| icbin | 2/2 | 2/2 | 109.113 | 0.000 | 1311.777 | 194.843 |
| itodd | 28/28 | 28/28 | 2150.823 | 0.000 | 2055.024 | unavailable |
| hb | 16/33 | 16/33 | 835.518 | 0.000 | 1298.848 | unavailable |
| ycbv | 21/21 | 21/21 | 1331.997 | 0.000 | 3616.435 | 158.762 |

Completed onboarding time sums only saved complete-stage active runtimes. Partial checkpointed time is separate and does not include all failed/unfinished work after its last checkpoint, including a failing GeDi tail. Saved inference time follows the original image-ledger timing policy; evaluator elapsed time is separate and may be reused from a valid primary receipt.

All failed/blocked targets and requested instances remain in the official target denominator. Surviving saved poses are retained, even within a failed/blocked target; missing predictions reduce the official score. No dummy predictions are inserted. Evaluation uses inst_count most-visible GT selection (visib_gt_min=-1), with GT confined to the official evaluator.

HB/ITODD local scores and the seven-dataset aggregate are unavailable because their test GT is private. No five-dataset macro or paper-performance parity claim is made.

Per-dataset VSD/MSSD/MSPD, inference/evaluation runtimes, exact ledger/CSV/target hashes, source pins and blockers are in results_with_recorded_failures.json and datasets/<dataset>/evaluation_with_recorded_failures.json.

Auxiliary helper SHA256: `9eddc4b53a176b9aa61d7fdfd99876d65714c3b3d9fbdc1c0b62d7bfd905872a`.
Primary frozen source-manifest SHA256: `237f10948fddc3a02fd42176e5acca3a9db1b7598e5b7a34d6d6766190e85c50`.
Official toolkit revision: `cea62d651c7e395b2e1962b9749e4e89693c6ac4`.

lmo recorded target reasons: `[{"reason": "fewer_valid_distinct_poses_than_requested", "status": "failed", "targets": 22}, {"reason": "no_predicted_masks", "status": "failed", "targets": 78}]`.

tless CAD feature-cache blockers: `{"27": "original GeDi extraction failed (check local neighborhoods, LRF degeneracy and CUDA ops): ValueError: trained GeDi descriptors must contain finite numeric values", "28": "original GeDi extraction failed (check local neighborhoods, LRF degeneracy and CUDA ops): ValueError: trained GeDi descriptors must contain finite numeric values", "30": "original GeDi extraction failed (check local neighborhoods, LRF degeneracy and CUDA ops): ValueError: trained GeDi descriptors must contain finite numeric values"}`.
The original GeDi forward produced nonfinite point descriptors. Our finite-output guard rejected the affected entire CAD feature cache. Author handling is unknown; no paper-performance parity is claimed.

tless recorded target reasons: `[{"reason": "CAD_onboarding_unavailable", "status": "blocked", "targets": 388}, {"reason": "fewer_valid_distinct_poses_than_requested", "status": "failed", "targets": 544}, {"reason": "no_predicted_masks", "status": "failed", "targets": 602}]`.

tudl recorded target reasons: `[{"reason": "fewer_valid_distinct_poses_than_requested", "status": "failed", "targets": 23}, {"reason": "no_predicted_masks", "status": "failed", "targets": 3}]`.

icbin recorded target reasons: `[{"reason": "fewer_valid_distinct_poses_than_requested", "status": "failed", "targets": 75}]`.

itodd recorded target reasons: `[{"reason": "fewer_valid_distinct_poses_than_requested", "status": "failed", "targets": 208}, {"reason": "no_predicted_masks", "status": "failed", "targets": 16}]`.

hb recorded target reasons: `[{"reason": "fewer_valid_distinct_poses_than_requested", "status": "failed", "targets": 30}, {"reason": "no_predicted_masks", "status": "failed", "targets": 140}]`.

ycbv recorded target reasons: `[{"reason": "fewer_valid_distinct_poses_than_requested", "status": "failed", "targets": 33}, {"reason": "no_predicted_masks", "status": "failed", "targets": 96}]`.
