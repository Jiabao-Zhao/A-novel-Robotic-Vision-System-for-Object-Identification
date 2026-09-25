# Current Human-Guided CAD / SAM3 Protocol

This document supersedes the earlier pooled/deduplicated proposal protocol and
the SAM3 >0.8 gate. The full method, equations, source map and local artifact
locations are in the root [README](../README.md).

## Fixed Decisions

- Keep the approved 13-object industrial workbench and native metric CAD assets.
- The operator determines useful CAD render count/angles, task intent and quantity.
  Gear top/bottom views and nut top/side views are distinct branches. Revision 4
  removed all gear side-oblique views and set both waterproof-male side elevations
  to zero degrees. The glue
  bottle excludes bottom views. Both drills retain only the two approved views.
- Send one CAD image per request to local Qwen3.5-4B using the locked user prompt
  in `docs/cad_view_prompt.txt`. Generate one description per view. The five-word
  rule is retained as a soft instruction: keep overlength descriptions verbatim and
  score them normally, without trimming, rejecting, or automatically re-prompting.
  The latest user-authorized rule prohibits numeric feature counts (digits or
  words), with `Gray rectangular panel with ten circular holes` as a bad example.
  Do not change any other wording or examples; notify the user of accidental changes.
  Historical multi-image prompts/results remain historical, not the current template.
- Query SAM3 separately for each description, reusing its scene embedding. Do not
  concatenate descriptions and do not merge duplicate masks across branches.
- For every branch, compare each mask only to that branch's corresponding CAD view
  with DINOv2 CLS, foreground patches and projected mask/depth penalties.
- The gear and remaining-product trials use SAM3 confidence strictly >0.5 for
  exported/scored masks; no NMS or cross-branch merging. Raw tensors retain the
  internal query slots. The waterproof-only four-side trial instead uses >0.4.
  This cutoff prunes expensive downstream work, not target correctness or final
  acceptance, and does not reduce the 200 internal SAM3 slots. Record counts and
  stage times; verify useful candidates survive. Historical ungated trials remain ungated. Preserve
  null/unavailable statuses for structural failures; no invented scores.
- Fit CAD translation with fixed branch orientation and native scale. No estimated
  table penetration filter, no external-occluder carving, no ground-truth pose input.
- Use equal mean observation-only/CAD-only/depth penalties and the documented
  `(global + patch + r * geometry) / (2 + r)` fusion. Depth residual scale is 5 mm.
- Rank inside each branch. Winner masks must agree on one instance (IoU >0.8) for
  the existing agreement decision. Multiple plausible instances for a singular
  request, disagreement or no usable candidate requires human clarification.

## Evidence and Next Work

The new remaining-product batch uses the same locked prompt and score computation
for the other ten products and all 25 approved CAD views (50 branches across two
scenes per target). The second scene is a target-only world-vertical yaw180 turn,
with the same resting face, fixed other objects and fixed calibration. This is not
a bottom flip; the glue bottle stays upright. Symmetric parts can remain visually
equivalent. Save raw descriptions unchanged, all retained per-mask component scores,
zero-detection branches and clarification decisions under `remaining-product-trial`
in the local workspace. No weights, final acceptance gate or CAD views are changed.

The later `waterproof-all-sides-trial` uses the authorized no-count prompt and
exactly four approved directions: Top, Bottom, Side A and Side /180. No additional
end-face tests. All four descriptions query each scene independently, giving
16 branches and 49 fully scored masks at >0.4. Side A is a residual-motion
diagnostic, not a settled pose; the other three pass the checks. Four branches
are empty due to the erroneous sofa description, and all scenes require
clarification. A post-hoc >0.5 filter on the same outputs would retain 37 masks
but remove a correct Side A/04A candidate at 0.4406. Do not equate lower confidence
with uselessness. The top scene is frozen, but both prompt and pruning cutoff
changed versus the prior experiment, so this is not a prompt-only ablation.

The completed batch covers 13 targets and 34 views. The later ungated run covers
only 21 previously uncomputed branches, not every branch. The active statistics
must retain that distinction. Post-hoc bbox IoU >=0.5 assigns audit labels only;
an unresolved label is not automatically an incorrect object or poor-quality mask.

No final SAM3/combined gate or score weighting is validated. Tune one object and
one controlled factor at a time, beginning with the large-gear side-view failure.
Keep raw prompts, output, masks and score components. Validate on different poses
and scenes before claiming generalization. Prompt optimization is not permission
to change the scoring equation or the approved CAD views silently.

The broader VLM task-extraction/dialogue and cosine CAD-retrieval stages are the
intended runtime design; the present controlled tests use known CAD targets.
The clarification decision is not yet a full deployed human interaction loop.
