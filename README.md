# Human-Guided CAD Object Identification

## Latest Classification Study: Start Here

**2026-09-25 cloud handoff:** read the
[four-scene VLM report](experiments/vlm_four_scene_20260925/README.md) for the
latest classification experiments, exact prompts, input images, all 1,936 raw
responses, portable CSV/JSON scores, charts and an offline verification script.
The completed comparison uses Qwen3.5-4B, Qwen3.5-9B, GPT-6 Sol and Gemma 4 12B
on the same 44 depth-localization RGB crops across four scenes.

This newer study revisits raw A-token probability for human-assisted association.
It does **not** use SAM3, DINOv2 or the geometric combined score during classification.
Its report supersedes the older scope, scene inventory and claims below that the
likelihood study is retired. The sections below document the earlier CAD/SAM3
branch framework and retained software, not the current four-model experiment.

The proposed catalog-name-and-color-only ablation has **not run**. No final model
or deployment threshold has been validated. Cloud analysis should use the report
and its versioned evidence rather than assume access to Windows/WSL directories.

## Earlier CAD/SAM3 Framework

Current direction, 2026-09-24: human-robot collaboration through operator-selected
CAD views, local VLM descriptions, SAM3 text-prompted masks, and independent
CAD-to-mask verification. We are evaluating locked-prompt per-object trials. This is a research
prototype, not a validated autonomous robot or a calibrated classifier.

## Start Here, Including GPT Cloud

Read this README and [the current protocol](simulation/human_cad_sam3_protocol.md)
before changing prompts, gates, scoring, or scene assets. They supersede the
retired VLM decision-token likelihood gate and six-object wrist ablations.

**Repository versus local experiments:** this repository contains shared capture,
simulation, DINOv2, SAM3 utilities and CAD projection code. The current experiment
runners and full results are in a separate local workspace, not in this checkout:

```text
C:/Users/Jiabao Zhao/Documents/Codex/2026-06-02/can-you-add-this-skill-to/
```

GPT Cloud cannot assume access to that Windows directory, D-drive checkpoints,
WSL, or ignored outputs. Ask for the relevant source/artifacts to be attached or
versioned before claiming to inspect or rerun them. Do not recreate a different
pipeline just because the local experiment files are absent from a cloud checkout.
Local edits become visible to a remote checkout only after commit and push.

## Framework

1. The human supplies the task, target and quantity. The intended runtime VLM
   extracts a target phrase; text-embedding cosine retrieval proposes a CAD model.
   Current controlled experiments supply a known CAD target directly: runtime
   extraction/retrieval is not yet the evaluated end-to-end path.
2. The human chooses useful CAD render angles and count. Views are object-dependent,
   not an automatic fixed six-view bank. Review revision 4 retains only top/bottom
   views for all gears; waterproof-male side views use 0-degree elevation. Both
   drills have two views; the glue bottle has no bottom view or tip-down placement.
3. Local Qwen3.5-4B, 4-bit, receives ONE CAD image per request in the latest trial.
   The user-supplied prompt requests one description, preferably at most five words.
   This is a soft instruction: retain and score overlength outputs unchanged.
   The actual request puts the prompt in a user message, not a custom system message.
   The CAD name and dimensions are not supplied to rescue an ambiguous image.
   [The exact current prompt](docs/cad_view_prompt.txt) is copied here for cloud
   review. Do not change it without explicit user instruction. Notify the user if
   it is accidentally changed. The original supplied prompt, including the five-word
   rule, was reinstated per the latest user instruction; there is no length-based
   rejection, trimming, or automatic re-prompt. The local canonical copy and SHA-256 lock
   are `cad-view-review/vlm_prompt.txt` and `vlm_prompt.lock.json`.
   The current template also includes the user-approved bad example for calling
   a gear a circular saw blade. The latest authorized addition prohibits exact
   feature counts in digits or words and includes the bad example
   `Gray rectangular panel with ten circular holes`. Preserve fresh outputs even
   when the VLM violates a rule; flag errors rather than silently correcting them.
   The large-gear bottom branch uses the human
   correction `gray teeth gear with central hole`, not a fresh VLM response.
   Earlier batch runs used a different multi-image, five-word prompt; preserve
   those historical records rather than rewriting them to match the new trial.
4. Each description is an independent text query to the same SAM3 model/scene
   embedding. Do not concatenate descriptions or merge repeated detections.
   Branch i keeps its description, masks, and corresponding CAD render i.
5. Compare each candidate with its own branch's CAD render using DINOv2 global
   and foreground patch descriptors, plus projected CAD mask/depth agreement.
6. Rank candidates inside each branch. If branch winners represent the same
   physical instance, select the highest-scoring winner. Disagreement, no usable
   candidate, or several plausible instances for a singular request requires human
   clarification. Identical products remain different physical instances. A plural
   request needs explicit quantity/selection handling, not silent deduplication.
7. CAD-to-observation pose alignment and robot execution are downstream. Existing
   registration/controller code is retained, but this experiment stops at analysis
   and selection/clarification decisions. It does not command the real robot.

The broader VLM-as-brain dialogue loop and real operator interaction remain design
goals; saved decision flags are not evidence that the full runtime loop is built.

## Scores and Gates

SAM3 text-detection score is not calibrated target correctness and is not the
combined score. The saved text pipeline multiplies sigmoid instance and presence
logits. The automatic point-prompt tracker is a different SAM3 mode; its predicted
mask IoU must not be substituted for this text-detection score.

Current per-branch verification:

```text
S_global = max(0, cosine(DINO CLS candidate, DINO CLS corresponding CAD view))
S_patch  = foreground patch appearance matching
r        = patch-derived coverage diagnostic, NOT measured physical visibility

O = observed SAM3 mask; C = projected CAD mask; U = O union C
P_obs   = |O minus C| / |U|
P_cad   = |C minus O| / |U|
P_depth = mean(min(|Z_observed - Z_CAD| / 0.005 m, 1))
          over O intersect C with valid measured depth
G       = 1 - (P_obs + P_cad + P_depth) / 3
S_combined = (S_global + S_patch + r * G) / (2 + r)
```

Use native metric CAD geometry, the branch render's fixed orientation, and
translation-only fitting. This is not unconstrained 6D pose estimation. Depth is
retained for projection placement and disagreement, including possible evidence
at gear openings. Partial depth is not reliable proof of product-family identity.
**No estimated-table penetration rejection and no external-occluder carving** in
the active branch penalty. Some historical helper functions still offer those
behaviors; they are not the current scoring contract.

| Quantity | Current role |
| --- | --- |
| SAM3 detection gate | Candidate pruning before expensive verification, not target acceptance. Gear and remaining-product trials: >0.5. New waterproof four-side trial: >0.4. Historical ungated follow-up retains all 200 raw slots |
| Mask-pixel probability > 0.5 | Binary mask creation, not a detection-confidence gate |
| Patch occupancy > 0.5 | Foreground descriptor support |
| Patch cosine > 0.5 | Coverage calculation, not candidate acceptance |
| Winner-mask IoU > 0.8 | Cross-branch instance agreement, not SAM3 confidence |
| Post-hoc bbox IoU >= 0.5 | Audit label only; never supplied to ranking |
| Final combined cutoff | Not selected or validated; chart sliders are exploratory |

Empty masks, too little valid depth, invalid projections, and no valid depth
overlap have explicit unavailable statuses/null scores. Do not fabricate zero
scores or silently treat these structural failures as confidence gates.
The detection cutoff does not reduce SAM3's internal 200 query slots. Record
retained candidates and verification time when tuning it; protect useful-candidate
recall. Fewer candidates are not evidence of proportional end-to-end speedup.

## Current Scene and Evidence

The active capture is:
`outputs/simulation/industrial_workbench/20260922T214852Z/`.
It contains calibrated RGB-D plus evaluation-only simulator object poses.
Those poses/identities must never influence candidate scoring or fitting.

The 13 targets are large/medium/small NIST gears; M8/M12/M16 hex nuts; Waterproof
Male; DSUB Male; round and rectangular pins; LM-O drill (8); LM-O glue bottle (11);
and YCB-V power drill (15). Spacer, bearing, pulley, Cable Shark, BNC Male and USB
Male are excluded from this scene. CAD assets for other uses are not silently erased.

Local workspace directories worth preserving:

| Directory | Purpose |
| --- | --- |
| `cad-view-review` | Operator-approved renders and view manifest |
| `large-gear-branches` | Historical independent-branch baseline and its earlier prompt template |
| `gear-pose-trial` | New locked-prompt, two-pose large-gear trial; per-mask CSV/JSON, raw masks and scores; no HTML report |
| `medium-gear-trial` | Medium gear, fresh top/bottom VLM descriptions, two scene poses, same >0.5 gate and shared gear-pose runner |
| `small-gear-trial` | Small gear, fresh top/bottom VLM descriptions, two scene poses, unchanged >0.5 gate and shared gear-pose runner |
| `remaining-product-trial` | Ten other products, 25 approved views, baseline and target-only yaw180 scenes (50 branches); same locked prompt, >0.5 gate, per-mask scores and decisions |
| `waterproof-all-sides-trial` | New no-count prompt; four approved CAD branches across Top, Bottom, Side A and Side /180 poses (16 branches), >0.4 pruning; 49 scored masks; Side A is a residual-motion diagnostic |
| `object-branch-batch` | 13-target, 34-branch original screening and score dossiers |
| `object-branch-ungated` | 21 previously blocked branches rerun without detection gating |
| `large-gear-side-test` | Side-view VLM failure diagnostic |
| `classification-score-charts` | Correct/incorrect/unresolved distributions and cutoff exploration |
| `cad-sam3-pilot` | Older pilot; its SAM3 query implementation is still imported |

The remaining-product trial keeps the existing resting face in its second scene:
only the target rotates 180 degrees about world vertical and settles. It does not
put the glue bottle tip-down or claim a new visible face for symmetric parts.
All twelve other object poses and the camera/calibration remain fixed. Fresh VLM
outputs, including wrong names and overlength descriptions, are retained without
manual correction. Root CSV/JSON tables record descriptions, every retained mask's
scores, branch winners, and scene decisions; no HTML report is generated.
The completed batch retained and fully scored 76 masks; 11 of 50 branches were
empty. Across 20 scene cases, five provisional selections matched the target in
the post-hoc audit and fifteen required clarification. This is controlled-scene
evidence, not independent benchmark accuracy or a calibrated acceptance gate.

The later waterproof-only trial keeps exactly the four approved view directions,
not six geometric faces. Top, Bottom and Side /180 pass the pose checks; Side A
stays within 0.07 degrees of the requested orientation but fails the strict motion
check, so its rows are flagged as diagnostic. All 49 retained masks are scored;
four branches are empty because the Side A CAD description incorrectly says
`White sofa with vertical back slats`. All four scenes require clarification.
On these same outputs, >0.5 would keep 37 masks but lose a correct Side A/04A
candidate at 0.4406. This is a pruning trade-off, not a calibrated correctness gate.
The original top scene is unchanged, but both prompt and detector cutoff changed
from the prior run; the saved comparison is not a prompt-only ablation.

The ungated follow-up has 4,200 query slots, 3,971 nonempty masks and 3,357 complete
combined scores. It is NOT an ungated rerun of all 34 original branches. The score
charts use those 21 branches for cutoff totals; large gear, M8 nut and YCB-V drill
have older gated reference data only in that chart. Bounding-box audit labels are
not ground-truth segmentation masks. Unresolved masks are separate from incorrect
named-object matches. Correlated proposals in one scene do not establish accuracy
or calibrated probabilities on independent scenes.

A complete side view of the large gear produced the raw VLM description
"White rectangular object with ribbed front surface". This is a useful failure
case, not a statement that the object is a rectangular part. The prompt's five-word
limit in that older trial did not guarantee compliance. The latest user-approved
prompt retains five words as a soft instruction; do not trim, reject, or re-prompt
based on description length. Preserve raw responses and separate prompt versions.

## Source Map

| Path | Responsibility |
| --- | --- |
| `scripts/cad_descriptors.py` | Current 224x224 preprocessing and DINOv2 CLS/patch encoding |
| `scripts/sam6d_patch_matching.py` | Reused patch matching; the active branch uses one corresponding view, not top-five view aggregation |
| `scripts/surface_verification.py` | Native CAD raycasting, depth points and translation fitting |
| `scripts/capture_industrial_workbench.py` | Capture the approved 13-object simulation layout |
| `simulation/industrial_workbench.py`, `benchmark_products.py` | Current scene and metric benchmark products |
| `simulation/nist_task_board_1.py`, `wrist_cluster.py`, `libero_*` | Shared geometry, sensor and simulation environment support |
| `point_cloud_localization.py`, `camera_capturing.py` | Retained depth localization and RealSense capture |
| `helper_function.py` | CAD library/retrieval and projection helpers |
| `CADPointCloudRegistration.py`, `robot_controller.py` | Existing downstream integration, not active classification gates |
| `scripts/local_sam3.py` | Optional local point-grid tracker adapter; not the active text-query classifier |

The exact branch penalty/fusion implementation currently lives in the local
`object-branch-batch/branch_metrics.py`; scoring is in
`object-branch-ungated/score_branch.py`. Their source must accompany cloud-side
changes to the score calculation. The shared DINO helpers were extracted from
retired wrist ablation drivers without intentionally changing their numerics.

## Local Setup and Checks

Existing local setup: Ubuntu-22.04 WSL, Python at
`/home/jiabao/.venvs/lerobot-libero/bin/python`, an RTX 3060 Ti (8 GB), Ollama model
`qwen3.5:4b`, and SAM3 checkpoint `/mnt/d/AI/models/sam3`. Rendering uses the cached
Blender/BlenderProc installation under `outputs/cache`. These are this computer's
paths, not portable cloud defaults. Keep model downloads/caches and source assets.

From the repository root in the existing WSL environment:

```bash
PY=/home/jiabao/.venvs/lerobot-libero/bin/python
$PY -m pytest -q tests
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl $PY -m scripts.capture_industrial_workbench
```

The capture command creates a new timestamped scene; it does not overwrite the
frozen active capture or automatically replace experiment inputs. Asset-dependent
tests require the local caches. A fresh cloud checkout is not a self-contained
model/dataset distribution. There is no replacement `main.py` pretending to run
the full human-robot framework.

## Next Experiment

Start with the large gear, including the ambiguous side view. Compare prompt
variants with fixed views and scoring first; then examine score components and
gates in controlled comparisons. Preserve exact prompts, model settings, raw VLM
outputs, branch/view mappings, all candidate scores and unavailable statuses.
Use different scenes/poses for validation before claiming an optimal prompt or
threshold. Do not silently change multiple factors or enable a cutoff from a chart.

## Cleanup and Data Policy

The obsolete VLM-likelihood path, wrist ablation/report drivers, hosted SAM/BOP
comparison drivers, assembly experiments and their superseded results were removed
from the working tree. Current local experiment results were preserved. Source
history remains in Git; ignored deleted results are not recoverable from Git.

Keep source, required tests and small protocol documents in Git. Keep large
generated masks, RGB-D captures, checkpoints, downloaded assets, environments and
reports under ignored local output paths. Never delete a shared helper or cached
asset solely because its filename or directory contains an older experiment name.
