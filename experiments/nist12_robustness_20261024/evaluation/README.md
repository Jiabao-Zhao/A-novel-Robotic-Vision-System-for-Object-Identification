# Four-arm evaluation

The source is prepared before classification-accuracy evaluation accesses
ground-truth annotations. Earlier `capture/audit_capture.py` input QA used
simulator truth for counts, projection and paired visibility checks, without
classification scoring or selection feedback. `lock`
validates the retained 220 NIST inputs and 2,720 CAD queries, then hashes final proposals,
scores, descriptions, features, input images, catalogs, protocols and source.
Compressed depth hypothesis archives are chained through their recorded SHA256
in locked fit summaries. A subsequent run must match the same inventory exactly.

Synthetic depth-corruption testing and the real T-LESS cohort are retired.
Their exclusive scripts, tests and generated artifacts are deleted. Active
cohorts and CLIP task collection are NIST-only.
Existing image encodings and their row mappings are retained verbatim. Old
380-input/5,000-query receipts do not constitute a retained-scope prediction lock.
The supplementary calibration keeps its original lock and verifies its exact
original source snapshots in `plausibility/frozen_sources/`. Current sources
join the new prediction lock separately. Its retained scope is the 2,400
simulation queries; no calibration data or threshold was changed.

Run only synthetic tests while predictions are in progress:

```powershell
& D:/Python/python.exe experiments/nist12_robustness_20261024/evaluation/test_evaluate.py
```

After all predictions, including the separately run native baselines, are final:

```powershell
& D:/Python/python.exe experiments/nist12_robustness_20261024/evaluation/evaluate.py lock
```

**Do not run the following until the coordinating agent explicitly confirms GO.**

```powershell
& D:/Python/python.exe experiments/nist12_robustness_20261024/evaluation/evaluate.py evaluate --read-audit-after-go
```

CLIP alone, CLIP top-five + depth, all-candidate depth, and size-only each rank
available scores. Null scores and ties remain unavailable. A bad distractor fit
does not block a valid candidate. Top-five cutoff ties block only the top-five
arm. Primary correctness requires the selected box to overlap any visible
instance of the requested CAD with IoU at least 0.5. Boxes from simulation and
proposals have inclusive final pixels; BOP boxes use x/y/width/height.

Present but fully hidden objects remain in the physically-present denominator;
observable-only accuracy is also reported. An absent query with no returned
candidate is unavailable, not certified absence. Duplicate-instance class
correctness and the need for human clarification are recorded separately.
Mask overlap identifies the selected physical part for error diagnostics only.

Reports separate cohorts, conditions, objects, newly queried female connectors
and twelve known controls. Paired fixes/regressions compare against CLIP alone
and all-candidate depth. Confidence intervals resample base scenes, preserving
paired conditions and objects. Fixed 101-point score threshold curves are
diagnostic and do not choose an operating threshold or claim calibrated
absence/ambiguity rejection. Multiple above-threshold proposals may be fragments
of one part; their count is not a ground-truth instance count.

Audit paths: simulation `audit/<scene>/`, new CADs
`new_cads/audit/Uxx/`. No robot or camera is connected. No new accuracy evaluation
was run during cleanup.

The label-free visual audit can replay historical saved CLIP embeddings without
loading a model:

```powershell
& D:/Python/python.exe experiments/nist12_robustness_20261024/evaluation/check_visual_parity.py
```

Its recorded replay matches every one of 1,152 candidate scores, their selected
CAD views, and all 96 fully available top-five lists. It also checks exact prompt
and model-control equality and verifies hash-identical cached images have no
conflicting raw descriptions.

After GO and completed evaluation, generate the five PNG/PDF figures and HTML
report/gallery:

```powershell
& D:/Python/python.exe experiments/nist12_robustness_20261024/evaluation/render_report.py --after-go
```

The source verifies the same prediction lock before reading evaluated result
tables. Gallery categories use a declared deterministic order. It preserves raw
CAD/crop images, both exact prompts, verbatim Qwen outputs, CLIP ranks, depth and
size scores, and links to component records. No output is generated for a failure
category without an eligible case. Recall-conditional accuracy, paired changes
from nominal, and measured occlusion pixel ratios are exported separately.
