# Human-guided CAD / SAM3 experiment decisions

Status: revised scene and asset preparation; the complete scoring experiment has
not yet been run on this scene. Historical captures and reports remain unchanged.

## Products

Retain the three NIST gears, three hex nuts, Waterproof Male, DSUB Male, round
pin, and rectangular pin. Add official BOP LM-O drill (8), LM-O glue bottle (11),
and YCB-V power drill (15): 13 objects in total. Exclude spacer, bearing, pulley,
Cable Shark, BNC Male, and USB Male from this experiment. The later screwdriver
and tape-measure demonstration assets are not part of this original-scene revision.

`python -m scripts.capture_industrial_workbench` saves a new timestamped capture.
Source BOP dimensions are converted from millimetres to metres without resizing.
LM-O vertex colors are baked into textures; the YCB-V texture is retained.
The LM-O drill rests on its side. The glue bottle rests upright, nozzle up.

## Human and VLM roles

The operator supplies task intent and quantity, chooses useful CAD description
views, reviews descriptions, and resolves ambiguous physical instances at runtime.
View selection should reflect feasible tabletop poses, not a fixed all-sides list.
The glue bottle excludes the bottom view and tip-down placement. Front, side, and
upper-oblique views are proposed, but the operator has not fixed their exact angles
or count. A supported side-resting pose could be added later; do not assume every
product must always be upright. Description views and matching templates are
separate sets, both subject to the experiment's support/visibility assumptions.

Use independent short VLM descriptions as separate SAM3 queries, reusing the scene
embedding. Pool candidate hypotheses and deduplicate repeated physical instances,
not different objects. Never concatenate the phrases into a single query for this
protocol. SAM3 score >0.8 is an exploratory proposal cutoff, not target correctness.
Preserve lower-score outputs for candidate-recall diagnostics.

## Restored descriptors and fusion

Restore frozen DINOv2 global CLS and foreground patch descriptors for CAD/candidate
comparison. Existing reusable code: `scripts/sam6d_patch_matching.py` and the local
`dinov2_vitl14` feature path in `scripts/run_surface_verification.py`.

The proposed SAM6D starting fusion is:

    S = (S_semantic + S_appearance + r_visible * S_geometry) / (2 + r_visible)

Do not silently reuse the historical comparator's 0.25/0.25/0.50 weights. Do not
change historical experiment scripts or results just to declare this new protocol.
SAM6D top-five semantic aggregation needs at least five matching templates; the
operator's two or three VLM description views must not be mistaken for this bank.

## Recovered image-space penalty

Source: cloud conversation "CAD Based Classification Methods", conversation
`6aa80c4e-8a90-83ea-a5af-0c68440fab82`, turn
`b0297f08-6865-464c-a98f-4ed4810ef9e9`.

For observed mask O and aligned visible CAD mask C, U = O union C:

    P_obs = |O minus C| / |U|
    P_cad = |C minus O| / |U|
    P_depth = mean(clip(abs(D_obs - D_cad) / sigma_d, 0, 1)) on O intersect C
    S_geometry = 1 - (P_obs + P_cad + P_depth) / 3

This retrieved version is evaluated in the 2D image plane but includes a depth
residual. It is not a pure silhouette-only equation. The existing implementation
is `equal_penalty_scores` in `scripts/surface_verification.py`, with sigma_d=5 mm.
It excludes external occlusion and invalid depth; the target's own pixels cannot
excuse a wrong CAD depth as occlusion. No overlap yields an unavailable score.
Do not substitute a 3D nearest-neighbor penalty or silently remove the depth term.

If a mask-only variant is requested explicitly, the symmetric-difference penalty
P_obs + P_cad equals 1 - mask IoU. It must be labeled as a separate ablation.
The historical one-third weighting compresses the shape contribution; thresholds
must be evaluated, not interpreted as probabilities.

## Decision and evaluation

Retain distinct candidates through verification. No passing candidate means reject
or clarify; multiple passing candidates for a singular request mean ask the human.
Plural requests require quantity/selection constraints. A lone high-scoring mask
is not proof of correct identity. Keep raw component scores and view provenance.
No score, acceptance threshold, or expected accuracy has been validated on this
revised easier object set. It is a development scene, not a representative benchmark.
