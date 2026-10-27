# Newly queried CAD cohort

The addon protocol was frozen before captures and inference. It contains twenty
new static CAD-supported layouts U00–U19 with sixteen products: the existing
twelve-product catalogue plus BNC Female, USB Female, DSUB Female and Waterproof
Female. The four female connector identities are newly queried targets in the
searched experiment JSON, but appeared as fixed fixtures in earlier scenes.
They are not globally unseen project assets.

All sixteen identities are queried in every layout: report **80 female-target
queries** as the primary new-query result and **240 known-target queries** as
contextual controls. No target labels or rendered masks choose layouts. Robot
and gripper geometry is absent from the new scenes. No robot execution or
dynamic settling is claimed.

`catalog.json` preserves native metric CAD geometry and the original fixture
colors. New CAD meshes are snapshotted under `cad_assets/`. The twelve known CAD
reference banks are reused exactly; the four new banks use the same fixed
twenty directions, 448-pixel images, 30-degree field of view, bounding-sphere
framing, white background and balanced lights. No scene observations select
reference views. `renderer_parity/completion.json` verifies all twenty RGB and
mask images of a known gear are bitwise equal to the prior bank.

Preparation commands use the existing WSL Python runtime:

1. `real_data/prepare.py` freezes both addon protocols and snapshots assets.
2. `real_data/render_references.py new_cads` writes `cad_inputs.json`.
3. `new_cads/capture.py` writes captures under `inputs/` and audit-only truth
   under `audit/`. `capture.py preflight` checks CAD-only placement without
   rendering or reading classification results.

The twenty-layout seed base is 9821040. Capture uses the shared scene generator;
the addon wrapper corrects its cosmetic base-scene label to U00–U19 in the final
manifest. All model inputs and calibration are in `inputs.json`; labels and
simulator transforms are audit-only and must not reach inference.
