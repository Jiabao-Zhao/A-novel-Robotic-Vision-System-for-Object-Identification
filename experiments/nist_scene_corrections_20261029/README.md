# Implemented NIST scene and assembly-reference corrections

This revision implements the corrections identified in the
[official source audit](../../docs/nist_task_board_1_reference.md). It changes
scene construction and task inputs; it does not run grasp generation, pickup,
assembly, physics, new pose estimation or a VLM.

## Scene changes

- Waterproof receiver mounted with its opening upward using a full rigid
  rotation, rather than the former yaw-only mounting.
- USB and the complete DSUB receiver/housing restored to their official mounting
  groups. M4/M12 bolts restored to their threaded holes, freeing the designated
  round-pin holes.
- Bolt mounting uses the plate underside as the head-shoulder reference.
- A separate finished board mesh represents nominal 4.1, 8.1, 12.2 and 16.5 mm
  round holes. Visual geometry and hole-preserving collision prisms use the same
  derived mesh. All other cut boundaries and board thickness are retained.
- The plate, receivers and goals share one explicit right-handed board frame.
  The native assembly working face −Z becomes +Z by a 180-degree rotation about
  X; plate geometry and Y coordinates rotate with it.

Source CAD/STL files are unchanged. A few original cuts exceed the documented
finished sizes; the derived mesh is a nominal simulation board, not a claim that
drilling can reduce an existing physical hole. Component convex collision hulls
remain the perception scene's approximation. Geometric grasp checks consume
actual receiver meshes; insertion/thread/contact dynamics remain unvalidated.

## Goals and grasp inputs

Current straight-pin tip depth is **20 mm below the board top**, including the
chosen chamfer-first direction. This leaves 11.0084 mm below the plate underside
and 8.9916 mm above the support. Protected insertion regions follow this updated
goal. It is a chosen partial-insertion endpoint, not bottomed-out seating.

Native assembly placements are transferred using explicit native/STL geometry
correspondences and mounting/mating datums. A source revision mismatch is never
called an exact rigid registration: the cached native KET12 part is 12 × 12 mm,
whereas our official task STL is 12 × 8 mm; its goal follows the actual STL and
matching rectangular receiver instead.

The [current selector](../cad_grasp_selection_20261029/README.md) uses an explicit
calibrated fingertip reference, near-tip contact anchors and complete saved
finger/pad/palm geometry. Its `task_inputs.build_request` composes the corrected
part and receiver poses with an explicit board-to-world transform. Pickup
poses remain supplied observed CAD alignments. No simulator object pose is read.

See the [corrected side views](../../outputs/nist_scene_corrections_20261029/gallery/index.html),
[goal registry](../../outputs/nist_scene_corrections_20261029/assembly_goals.json)
and [offline verification](../../outputs/nist_scene_corrections_20261029/verification.json).
CAD goal references do not certify physical connector contact or thread seating.

## Verification and reproduction

With the existing local Python environment and cached official assets:

```powershell
D:\Python\python.exe -B -m experiments.nist_scene_corrections_20261029.native_goals --goals
D:\Python\python.exe -B -m pytest -q -p no:cacheprovider tests/test_nist_scene_corrections.py experiments/nist_scene_corrections_20261029 experiments/cad_grasp_selection_20261029
D:\Python\python.exe -B -m experiments.nist_scene_corrections_20261029.verify
D:\Python\python.exe -B -m experiments.nist_scene_corrections_20261029.gallery
```

These checks construct XML/static geometry and inspect input transforms. They
do not instantiate/step a simulator or sample grasps on the NIST parts. A small
set of synthetic contact candidates verifies the selector's frame conversions.

The current targeted aggregate passed **68 offline tests**, including the
assembly-region sampler, strict endpoint checks and fingertip/frame checks.
The corrected input builder accepted
all **96 saved observed part placements across eight scenes** without calling
grasp generation. All **15 actual fixture poses** were constructed into static
[scene XML](../../outputs/nist_scene_corrections_20261029/corrected_board_scene.xml).
All twelve target CAD hashes match the CAD frames used by the observed alignments,
including the frozen cable-containing USB model. Thirteen side-view PNGs and
the gallery links were checked; this is not a new grasp-success measurement.
The [region atlas](../../outputs/cad_grasp_selection_20261029/region_guidance/index.html)
shows the explicit allowed proposal surfaces. It compiles CAD annotations only;
actual NIST candidate trials and all pickup/assembly execution remain paused.

M16's mounting, coaxial target and seating height are corrected. Its static
thread phase remains explicitly nominal; physical threading/contact engagement
has not been validated and no threading controller is introduced.

## Historical evidence

The original October 28 selector and numerical results remain unchanged. The
active scene module had to change. Its exact old bytes were copied and checked
against the original launch hash before editing; see
[source_migrations.json](source_migrations.json). The snapshot was created during
this correction, not before the original experiment. `history.py` verifies all
249 published files plus the original launched code/protected inputs, resolving
that one changed source path to the recorded snapshot. This does not claim that
the active scene source still has its historical hash.

The earlier 12 mm side-section review remains historical. Its incorrect mounting
and destination assumptions are visibly superseded; those measurements were
not overwritten with new numbers.
