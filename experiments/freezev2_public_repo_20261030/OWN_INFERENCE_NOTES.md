# Own generated inference: evaluation boundary

This is a separate execution of the unmodified third-party `tfwang08/FreezeV2-Re` at `12f82a2f8afd435466a87cf2dafce404087aeb3f`. It is not verified author-released FreeZeV2 inference code. The authoritative executable configuration is `public_inference_v1.json`; its canonical JSON digest and byte hash accompany own evaluation receipts.

The own output directory is `outputs/freezev2_public_repo_20261030/own_inference_v1`. The earlier independent implementation, geometry smoke, and author-CSV reference gate remain separate evidence. Evaluating author CSVs can verify evaluator agreement; it does not show that our generated inference reaches those scores.

## Fixed inference recipe and assumptions

The configured method is `FreeZeV2-Re-base-l30-public-v1`: frozen DINOv2 giant, DINO layer30, original trained GeDi, original uniform view aggregation/PCA64/fusion, original top10 feature matching and10000 RANSAC iterations, original30-iteration ICP, and original multi-mask selection/scoring. CNOS-FastSAM-public-BOP23 masks are external predicted inputs. No neural training, GT initialization, pose tuning, finite-row GeDi repair, table constraints, or new score weights are introduced.

Layer30 follows a public README candidate. Raw5000 CAD samples are the public API/example count before visibility filtering; the retained descriptor count may be smaller. A1mm visibility depth tolerance and10mm translation NMS threshold are explicit fixed reproduction assumptions, not verified author constants and not selected by test scores. The canonical configuration records the full values and scientific boundaries. This study must not describe these assumptions as an official complete validated preset.

## Full target accounting

Each dataset retains a `target_ledger.json` with one terminal record for every exact official `(scene_id, im_id, obj_id, inst_count)` target. A record can be complete, failed, or blocked. Targets with no pose remain recorded and remain in the unchanged official evaluation target denominator. Partial successes are not presented as the full dataset.

`evaluate_own.py` audits exact target and instance counts, source/configuration identities, per-target receipt hashes, unchanged original generated CSVs, and original multi-mask selected pose geometry. The aggregate own CSV must match all selected saved rotations/translations/scores at the public writer's nine significant digit precision. Unknown, missing, duplicate, or pending target records block scoring. Author/reference CSV paths and filenames are rejected. The generated aggregate filename is `freezev2public_<dataset>-test.csv`.

The original per-target pose values are preserved. Aggregate CSV time is the sum for each image of original per-target CSV localization time when a pose exists; a no-pose target contributes its guarded attempt runtime, and a pre-inference blocked target contributes zero. This is an explicit execution timing policy. It excludes subprocess startup where original localization timing exists and must not be presented as a directly comparable end-to-end author timing result. CAD onboarding remains separate; missing stage timings remain unavailable.

## Official evaluation and receipts

The runner calls:

`evaluate_dataset(root, output, dataset, csv_path, configuration, ledger_path, lock_stream, durable=None, owned=None)`

while holding `base_public_v1/compute.lock`. The module invokes the pinned original `run_bop.py evaluate-submission` using the existing evaluation Python and official BOP Toolkit `cea62d651c7e395b2e1962b9749e4e89693c6ac4`. The frozen owned-process guard inherits the same real OS lock through the evaluator process group. It never starts a competing inference worker.

Evaluation uses BOP19 model-based 6D localization, split `test`, unchanged `test_targets_bop19.json`, `visib_gt_min=-1`, and the official `inst_count` most-visible-GT rule. GT poses are read only by evaluation; inference uses the separate GT-free dataset view.

Public GT is available locally for LM-O, T-LESS, TUD-L, IC-BIN, and YCB-V. HB and ITODD test GT is private. Their own CSV and exact coverage are still mandatory; metrics are explicitly null and no remote submission is made. No complete seven-dataset aggregate is reported.

Per-dataset outputs include `own_evaluation_receipt.json`, `official_evaluation/`, evaluator logs, full target coverage/failure counts, source/configuration/input pins, and measured evaluator wall time. Scores reuse only when every pinned identity and saved metric file hash agrees. Evaluator failures remain explicit receipts. `write_summary(root, output, durable=None)` writes `own_results_summary.json` and `OWN_INFERENCE_REPORT.md` from recorded receipts.

Official reported values are pose AR/VSD/MSSD/MSPD. They are not mask AP or joint CAD-identity AP. Own valid generated predictions, evaluator agreement on author predictions, and published-performance reproduction remain separate claims.

## Current validation evidence

The evaluator adapter has24 focused offline CPU tests passing. They cover exact full-denominator accounting, missing/unknown/duplicate targets, source and receipt mismatches, author-CSV rejection, selected-pose joins, per-image time aggregation including failed attempts, private-GT coverage requirements, inherited evaluator locking, score reuse, and explicit evaluator/metric validation failures. These tests mock official evaluation and do not measure pose performance. Actual own inference and official scores must be reported from completed run receipts, not from this document.

CPU-only coverage and representative pose/failure diagnostics are documented in [AUXILIARY_DIAGNOSTICS.md](AUXILIARY_DIAGNOSTICS.md). They remain separate from sealed inference and official evaluation.


## Validated DINO loading reuse: execution revision 2026-10-31

The user requested a runtime speedup. Profiling showed that the unchanged public target extractor reconstructs and loads the 4,546,108,579-byte DINOv2 giant model for each candidate mask. In the fixed healthy two-mask replay, the two hub loads consumed 26.174 and 26.790 seconds; GeDi and registration were much smaller components.

A separately sealed execution wrapper now reuses only that exact local, frozen DINO model within each existing target subprocess. The original fresh extractor constructor, preprocessing and feature forward remain in use. Original GeDi models, reseeding and sampling, 10,000 RANSAC trials, ICP, NMS, scoring, FP32 and TF32 settings remain unchanged. The cache does not persist across targets. Skipping the repeated DINO model constructor changes unused constructor RNG consumption; source analysis and the fixed actual replay controls verify that downstream original seeded outputs remain equal on those controls.

The fixed healthy replay measured 61.422 seconds without reuse and 33.523 seconds with reuse (1.832x, 45.4 percent less wall time), with the same 5.119 GiB peak allocated CUDA memory. Original and reused healthy descriptors, full candidate outcomes/scores/poses and CSV pose fields matched exactly. Additional controls retained the identical alternate-candidate GeDi rejection, whole-target GeDi rejection, and absent-mask outcome. This is a fixed LM-O control measurement, not a seven-dataset speed benchmark, a GT accuracy test, or published-performance reproduction. Single-mask targets still pay one complete DINO load.

The active output remains `outputs/freezev2_public_repo_20261030/own_inference_v1`. Its five original sealed orchestration files, canonical configuration, public checkout and existing predictions remain unchanged. The added three-file execution source is sealed under `auxiliary_source/runtime_reuse_v1/`; `execution_manifest.json` has SHA256 `f0ecc7957d9862b647677c57a9e34df438a8f642c07b07970437ab3e126d7e4b`. These execution pins supplement the original scientific source/configuration pins; the old source manifest alone does not identify the added wrapper.

The actual GPU trial, complete CPU parity comparisons and their bound artifacts are in `auxiliary_diagnostics/runtime_reuse_trial_20261031_v2/validation_receipt.json` (SHA256 `6021a6781882a7cbfa207b55950f568115f7e488b3f6c29e7af61648bed3f550`). The first auxiliary trial was deliberately interrupted to verify lock behavior and remains separately preserved; it was not used as a passed validation.

A planned idle-boundary handoff preserved all 406 terminal target receipts byte-for-byte. `execution_revisions/transition_runtime_reuse_v1.json` binds those hashes (SHA256 `7e8773ebf81edd8e64893d91cf7cd4b27770ff10e781f596ffed4ec7e7a20da8`), and `auxiliary_diagnostics/runtime_reuse_handoff_20261031_v1/` records exact old-owner reconciliation and replacement launch. No GPU target was interrupted and no lock file was removed. New target receipts add execution revision, transition and runtime-report hashes. The wrapper audits both execution histories before delegating unchanged official evaluation.

For a genuine future execution failure, use the sealed `auxiliary_source/runtime_reuse_v1/launch_runtime_reuse.py` with `--config configuration.json`, `--transition-manifest execution_revisions/transition_runtime_reuse_v1.json`, and `--validation-receipt auxiliary_diagnostics/runtime_reuse_trial_20261031_v2/validation_receipt.json`, all resolved inside the own output. First reconcile actual native and Linux owner identities, surviving children and both real OS locks. Do not resume the old launcher or edit either sealed source set. Current logs are `runtime_reuse_worker.log` and `runtime_reuse_bridge.log`; old empty or stale logs and old process IDs are not evidence of failure.


## User LM-O comparison stop gate (2026-10-31)

The user superseded unconditional continuation through seven datasets: finish full LM-O, compare its official pose AR with the paper, and stop remaining benchmark work if not aligned. The closest declared comparator is Table II row11 CNOS-only base LM-O69.4 percent. The tolerance remains pending an explicit user answer; no one- or two-percentage-point tolerance has been assumed. Until a criterion is configured, the run holds after LM-O evaluation.

Scheduling-only control source is separately sealed under `own_inference_v1/auxiliary_source/runtime_reuse_v3_lmo_gate`, control manifest SHA256 `1f18518aedeab52f86215b0e615c9e5eab04e7f16adf4955f191a1a04e66ab38`. It delegates the unchanged validated DINO-reuse driver and original inference/evaluation. The immutable pending policy is `run_control/lmo_policy_awaiting_user_v1.json`, SHA256 `5fbd216a93d660b3aa96ac52224ec2d984d1434298bedcc31d9efa62939ffe8b`. New explicit user criteria require a separately recorded policy/control revision.

After LM-O evaluation, the gate reopens the actual own score file and full1445-target/1445-instance accounting, including extraction failures and no-pose targets, and validates source, CSV, ledger and metric hashes. It reports own AR,69.4 percent reference, and signed percentage-point difference. It exits before any T-LESS onboarding if outside the configured criterion, if the criterion is pending, or if evaluation cannot be verified. The supervisor treats these as expected stops and does not restart. Integrity/reporting errors cannot fall through into later datasets.

The planned control cutover preserved461 terminal target receipts byte-for-byte (`run_control/lmo_gate_cutover_v1.json`, SHA256 `9e6587ab65abdcc4e62aa2a084ffc8b51307b77ff14f823ccee5f207f4f7ff13`). The first v2 control launcher failed on an import path before starting inference; its immutable source and failure evidence remain preserved. V3 fixes the bootstrap, passed42 CPU tests and a real native isolated import check, and is active. `run_control/lmo_gate_activation_v3.json` binds actual owners and verification. The primary five-file source snapshot, validated three-file DINO execution revision, canonical scientific configuration and all saved predictions remain unchanged.

Use the current sealed `launch_runtime_gate.py` with the saved configuration, control manifest, policy and existing v1 revision/transition/validation arguments. Directly starting either older launcher bypasses the user gate and is prohibited. Current logs are `runtime_gate_worker.log` and `runtime_gate_bridge.log`. After a deliberate gate stop, report the comparison before considering any resume; no complete-seven-dataset claim is warranted.


## Verified own LM-O result and comparison stop (2026-10-31)

Full own LM-O inference and official failure-inclusive evaluation completed: pose
AR **66.91372549%** versus the paper Table II row11 CNOS-only base **69.4%**,
a **-2.48627451 percentage-point** difference. All1,445 targets remain in the
official denominator:1,346 saved poses and99 zero-pose targets (78 without mask
candidates,21 with all candidates rejected). Authors' saved predictions were
not used. The other six datasets were not started under the user's LM-O gate.

The recorded state is `stopped_lmo_awaiting_alignment_tolerance`; the user has
not supplied a tolerance. No criterion was chosen automatically and no restart
is authorized. Actual native/Linux owners exited and both real locks were
verified free. The complete own result, exact coverage, candidate failures,
source/checkpoint/visual audits, timing scopes and comparison limitations are in
[LMO_FINAL_REPORT.md](../../outputs/freezev2_public_repo_20261030/own_inference_v1/LMO_FINAL_REPORT.md).
This remains third-party public-code replication with unresolved official
configuration/mask-archive parity, not a verified author release or completed
seven-dataset reproduction. Historical readiness and author-CSV reference
evaluations remain separate evidence.


## User-confirmed alignment criterion (2026-10-31)

The user explicitly answered **Match the rounded 69.4%**. The verified own
LM-O AR of 66.91372549% rounds to **66.9%**, so this criterion is not met:
`aligned: false`, `continue: false`, `stopped_lmo_not_aligned`. All six
remaining datasets stay stopped and were not started. The temporary monitor
remains deleted; this chat remains open.

This resolves the historical pending-criterion state above through a separate
[configured policy](../../outputs/freezev2_public_repo_20261030/own_inference_v1/run_control/lmo_policy_rounded69_4_user_v1.json)
and [post-stop comparison](../../outputs/freezev2_public_repo_20261030/own_inference_v1/run_control/lmo_comparison_rounded69_4_user_v1.json).
The [revised final report](../../outputs/freezev2_public_repo_20261030/own_inference_v1/LMO_FINAL_REPORT_ROUNDED69_4.md)
records the result and comparison limitations. Original pending-policy and
runtime-stop records remain historical evidence. No inference, evaluation or
worker restart was performed, and scientific configuration, source, poses,
targets and metrics remain unchanged.
