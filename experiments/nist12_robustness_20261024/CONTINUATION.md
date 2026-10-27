# Continuation status — October 27, 2026

Repository cleanup is not an instruction to resume experiments. No inference or
accuracy evaluation was run during cleanup.

The retained scope is NIST-only: **220 inputs / 2,720 CAD queries**, defined by
`common.ACTIVE_INPUTS` and `retained_scope_20261027.json`. All 25,360 NIST fit
summaries remain. Classification accuracy evaluation is incomplete.

Synthetic depth corruption and real T-LESS are retired. Their exclusive artifacts
and code are removed; do not recreate them. Older manifests, logs, source hashes
and completion counts describe the original experiment and are not current
execution gates. Keep shared embeddings and their row mappings together.

Before any future evaluation, inspect actual process identities, current source
and input dependencies, and build a fresh retained-scope prediction lock. Do not
reuse all-380-input finalizers or report fitting completion as classification
accuracy. Gripper regions remain excluded as specified in the study contract.
