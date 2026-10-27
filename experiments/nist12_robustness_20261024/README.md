# NIST classification robustness study

This earlier CLIP/depth study is retained for comparison. It is incomplete:
no new robustness classification accuracy has been evaluated. The current
project framework is described in the [repository README](../../README.md).

## Retained scope after October 27 cleanup

| Cohort | Inputs | CAD queries |
|---|---:|---:|
| New nominal NIST layouts | 20 | 240 |
| Paired appearance, viewing, occlusion and cardinality changes | 180 | 2,160 |
| Four additional NIST CAD queries alongside the original twelve | 20 | 320 |
| **Total** | **220** | **2,720** |

The paired scenes vary lighting/shadows, background, finish, camera distance,
resolution, fixture width, target absence and an extra instance. The query is a
supplied CAD; the output is a scene candidate. Shared fitting preserves native
CAD scale and calibrated depth. New female-connector queries previously appeared
as fixed fixtures, so they are not wholly unseen geometry in the scene generator.

The four saved ranking arms are CLIP alone, CLIP top-five followed by depth,
all-candidate depth and dimensions only. Native full-scene baselines are separate
from shared-crop matching adaptations. Consult each baseline's recorded execution
and numerical limitations.

## Removed scope

Synthetic depth-noise, missing-depth and depth-scale experiments were canceled.
Their dedicated code, tests and exclusive generated artifacts are deleted.
The real T-LESS cohort and its exclusive artifacts are also removed because
current scene generation and testing are NIST-only. Do not restart either cohort.

The [retained-scope receipt](retained_scope_20261027.json) and
`geometry/nist_pair_index_20261027.json` define the NIST subset. All 25,360 unique
NIST fit summaries were retained. Seven mixed visual-cache files remain intact
because deleting or rewriting them would discard retained NIST embeddings or
break their row mappings; retired rows are excluded from active task collection.
The [deletion manifest](cleanup_nist_only_20261027/deletion_paths.json) and
[execution receipt](../../outputs/cleanup_20261027/phase_b_execution.json)
record the cleanup.

Older protocols, logs and hashes describe their original design and checkout.
They do not certify the reorganized source or authorize canceled work. A future
prediction lock must cover the current NIST scope before accuracy evaluation.
Cleanup did not run inference, access new accuracy labels or recompute results.

Correctness, unavailable scores, ties, target absence, hidden targets and multiple
plausible instances must remain distinct. Scores are uncalibrated. These results
do not establish physical pickup, assembly or a complete autonomous runtime.

See [evaluation details](evaluation/README.md) and
[continuation status](CONTINUATION.md).
