# Repository Guidance

Read `README.md` and `simulation/pipeline_protocol.md` first. The current scope is
NIST Task Board 1 and CAD-guided recognition/alignment, followed by grasp research.

## Current Contract

- Retrieve the requested CAD from the library using cosine similarity.
- Segment observed RGB-D through depth localization, project regions into RGB and
  crop them. Keep camera calibration and metric depth explicit.
- Use DINO global/foreground-patch descriptors and metric CAD/depth verification
  for final object matching and pose estimation. Do not restore VLM token
  likelihood classifiers or their retired experiments/results.
- Grasp generation uses the aligned CAD and partial observed cloud. It remains
  research; geometric candidates do not establish pickup or assembly success.
- The pipeline makes the final grasp decision using mandatory geometry checks
  and deterministic geometric priorities. VLM ranking/selection is not a current
  or planned stage of this pipeline. Preserve older VLM studies as historical
  evidence; do not reconnect them to the current selector.
- Generate NIST scenes/CAD references only. LIBERO/robosuite camera and Panda
  support remain dependencies; household/FMB/workbench layout generators do not.
- Keep native CAD scale, explicit frames and units, unavailable scores and ties.
  Do not describe translation-only fits or unresolved symmetric orientation as
  full observable 6D pose. Simulator labels/poses are audit-only.
- Do not claim the complete retrieval-to-grasp runtime is implemented or validated
  merely because its component experiments exist.

## Scope and Engineering

- Prefer small existing helpers and controlled local trials over new frameworks,
  long interfaces or speculative configuration.
- Keep source under the existing module directories, study code under experiments,
  and generated assets/results under outputs. See the experiment index.
- Preserve genuine CAD/model/tool dependencies and physical calibration/captures.
  Delete obsolete experiments only within the user's authorized scope.
- The October 27 user instruction explicitly authorizes removal of likelihood
  experiments AND results and non-NIST layouts. It supersedes the older SAM3
  experiment contract and older retention rules for those retired artifacts.
- For retained studies, do not rewrite numerical outcomes or claim migrated source
  hashes were frozen before the original run. Record path-only migrations.
- Add focused tests for shared preprocessing, projection, scoring or mapping changes.
  Keep tests offline unless an integration check is explicitly necessary.
- Do not download weights, install/modify global environments, start paid APIs,
  command a physical robot or add execution/planning features during cleanup.
- Preserve unrelated user edits and current grasp research. Explain results with
  concrete parts, steps, inputs and measured limits rather than vague labels.
