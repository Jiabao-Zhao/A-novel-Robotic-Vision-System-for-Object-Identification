# Repository Guidance

Read `README.md` and `simulation/human_cad_sam3_protocol.md` first. They describe
the current human-guided CAD/VLM/SAM3 direction, local-only experiment locations,
and limitations. The old VLM decision-token likelihood classifier is retired.

## Current Contract

- Human supplies intent/quantity, chooses CAD views, reviews descriptions and
  resolves ambiguity. Do not claim the full interactive runtime is implemented.
- VLM receives one CAD render per request in the latest trial, with one description
  per view. The five-word prompt rule is soft: retain overlength responses verbatim,
  with no length-based rejection, trimming, or re-prompt. It is not the final selector. Preserve the
  exact user-approved prompt and output. Do not change the prompt without explicit
  authorization; stop and notify the user if it is accidentally changed. The current
  prompt is `docs/cad_view_prompt.txt`; local canonical text/hash are in
  `cad-view-review/vlm_prompt.txt` and `vlm_prompt.lock.json`.
  The latest authorized rule prohibits feature counts in digits or words, with
  `Gray rectangular panel with ten circular holes` as a bad example. Preserve
  raw violations for analysis; do not silently rewrite VLM output.
- One SAM3 text query per description; reuse the scene embedding. Never concatenate
  queries or merge masks across branches. Score against the corresponding CAD view.
- Keep DINOv2 global and foreground-patch descriptors. Do not substitute VLM token
  likelihoods or confuse SAM3 score, tracker IoU and target correctness.
- Keep native metric CAD scale, fixed branch-view rotation, translation-only fit,
  full projected/observed mask union and valid-overlap depth disagreement.
- No estimated-table penetration rejection or external-occlusion carving in the
  active branch score. Retain depth localization; it is not replaced by SAM masks.
- Current combined score is `(global + patch + r * geometry) / (2 + r)`; `r` is
  patch-derived coverage, not measured physical visibility. See README for penalties.
- Gear and remaining-product trials export and score only SAM3 confidence >0.5.
  The waterproof-only four-side trial uses >0.4. These are compute-pruning gates,
  not correctness or acceptance gates, and do not change SAM3's internal 200 slots.
  Historical ungated trials retain all query slots. Preserve explicit unavailable
  statuses. Pixel/patch thresholds and branch-agreement IoU are distinct.
- Waterproof testing uses exactly the four approved Top/Bottom/Side A/Side /180
  directions. Side A fails the strict motion check and is explicitly diagnostic;
  do not call it settled or add end-face tests. Keep pose flags in score tables.
- Choose the best per branch; require human clarification for disagreement or
  multiple plausible instances for a singular request. Never silently execute.
- Simulator labels/poses are audit-only. Scores are uncalibrated. The 21-branch
  follow-up is not the full 34-branch ungated dataset. Avoid leakage and overclaiming.

## Scope and Engineering

- Prefer deletion of obsolete drivers and outputs over new framework abstractions.
  Keep genuine dependencies, CAD assets, physical calibration/captures and models.
- Do not add robot execution, grasp planning, a GUI or a new end-to-end framework
  during prompt/scoring experiments unless specifically requested.
- Use existing helpers and structured parsers. Keep source separate from generated
  artifacts, small functions, explicit units, and concise meaningful comments.
- Prefer constants or a small config for controlled local trials. Do not add long
  argparse interfaces, compatibility layers or speculative configurability.
- Work in meters; label camera/CAD/world/robot frames. Never invent calibration.
  Do not call translation-only fitting or unobservable symmetric orientation full 6D.
- Preserve frozen experiment evidence. Create new trial outputs; do not rewrite old
  prompts/results to make a new protocol appear historically true.
- Add focused regression tests when changing shared preprocessing, branch scoring,
  projection, thresholds or query/view mapping. Validate before deleting dependencies.
- See README for the current source map. Some active runners live outside this
  repository. GPT Cloud must obtain those files before editing or claiming to run them.
- Do not download model weights, start paid APIs, modify global environments, or
  connect to a physical robot as a side effect of repository tests or cleanup.
