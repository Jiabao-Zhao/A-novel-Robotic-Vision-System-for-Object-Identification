# NIST mating-goal CAD review

**October 29 correction:** the
[official NIST source audit](../../docs/nist_task_board_1_reference.md) supersedes
this review's horizontal waterproof reference and some destination assignments.
The receiver should face upward; our USB/DSUB locations were swapped, and M4/M12
bolts were assigned to pin holes. In particular, the alternate round-12 hole
selected below is the M12 bolt location. Treat this registry as historical
inspection data pending explicit native-CAD frame conversion, not corrected
goals ready for use. The user-selected 12 mm depth is retained. Mesh sections,
measured numbers and historical grasp outcomes are unchanged.

Review the twelve products in the current eight-scene grasp study against their
actual receiver CADs before resuming grasp generation. This work runs no grasp
sampler, VLM selection, new pose estimation, robot motion or physics stepping.

The pipeline itself will make the final grasp decision using geometric checks
and deterministic criterion priorities. VLM ranking/selection is not a pipeline
stage. Grasp generation, pickup and assembly execution remain paused; VLM
selection is excluded rather than awaiting resumption.

The [side-view gallery](../../outputs/nist_mating_goals_20261028/gallery/index.html)
shows previous goals above reviewed references. The drawings cut through the
actual triangle meshes so holes and internal contact features are visible.
Gray is the receiver, blue is the reviewed part, and red marks overlapping solid
sections when both section boundaries are complete. A clear sampled section
does not certify full three-dimensional fit or physical assembly.

## Corrected references and limits

- All six pins now point their modeled chamfers downward. The user chose a tip
  depth of **12 mm below the board top**, or about **3.0084 mm through the underside**.
  Actual hole boundaries, including the small square-hole rotations, have positive
  side clearance. The depth is a task requirement; the CAD contains no axial stop.
- Gear Large's source gear/shaft exports recover the existing seated height.
  Its static CAD reference does not validate insertion, the set screw or other gears.
- USB's catalog connector and frozen cable-containing model are related by an
  explicit rigid transform. Composing that transform with the original male/female
  export relationship fixes the incorrect identity goal. Twenty-five complete
  axial sections show no solid overlap; this remains sampled evidence.
- BNC, DSUB, Waterproof and M16 exact final goals remain unavailable. Their review
  transforms are separate display data, never fallbacks. The report identifies
  contact interference, uncertain mechanical stops or thread phase explicitly.
- The M16 screw reference mounting is lowered by one measured board thickness:
  its head shoulder belongs at the board underside. This is stored in the new
  reference data; the simulation scene and historical results are unchanged.

The [goal registry](../../outputs/nist_mating_goals_20261028/assembly_goals.json)
contains eight non-null CAD/task references and four unresolved goals. All have
`grasp_generation_enabled=false`. Full physical calibration remains separate.
The corrected pin orientation also reverses which CAD end is protected for
assembly; the registry supplies the corresponding inserted-region half-space.
Historical negative-end protected masks must not be reused with the new goals.

## Files and local reproduction

- `pins.py`: native chamfer direction, free board holes, actual polygon clearance
  and the chosen task-depth transforms.
- `connectors.py`: receiver opening directions, source paired placements, USB
  model mapping and sampled interference measurements.
- `m16_audit.py`: mounting-plane and sampled thread-phase review.
- `gears_nut.py`: gear source relationship and unresolved nut reference.
- `cad_tools.py`: unchanged binary STL loading and plane sections with explicit
  completeness reports; no mesh filling, scaling or collision hull substitution.
- `build_review.py`: corrected registry, frame checks and historical hash audit.
- `gallery.py`: previous/reviewed side sections for all twelve parts.

```powershell
python -B -m experiments.nist_mating_goals_20261028.pins
python -B -m experiments.nist_mating_goals_20261028.connectors
python -B -m experiments.nist_mating_goals_20261028.m16_audit
python -B -m experiments.nist_mating_goals_20261028.gears_nut
python -B -m experiments.nist_mating_goals_20261028.build_review
python -B -m experiments.nist_mating_goals_20261028.gallery
python -B -m pytest -q -p no:cacheprovider experiments/nist_mating_goals_20261028
```

These CAD-only commands require the existing local NumPy/SciPy/Shapely/Matplotlib
environment, original NIST STL exports, board/receiver meshes and frozen USB CAD.
They do not install dependencies. Some assets are local and absent in a fresh
GitHub checkout. Original exports share useful placement information for some
pairs; they are not assumed to share a global assembled board frame.

See the [report](../../outputs/nist_mating_goals_20261028/report.md),
[complete measurements](../../outputs/nist_mating_goals_20261028/review.json)
and [verification receipt](../../outputs/nist_mating_goals_20261028/verification.json).
The October 28 grasp run is preserved byte-for-byte.

Verification in the existing local environment passed **19 offline tests**:
section construction, hole parity, pin frames/depth/clearance, USB composition,
M16 mounting and unavailable-goal behavior. All gallery links and thirteen PNGs
were checked. Browser interaction is not claimed as tested.
