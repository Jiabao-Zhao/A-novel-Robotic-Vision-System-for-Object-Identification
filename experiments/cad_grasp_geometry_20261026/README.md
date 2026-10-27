# Eight-scene CAD grasp geometry test before VLM selection

Authorized on 2026-10-26 after refining the earlier sampler. This experiment
tests whether modeled Panda fingers can be placed around the CAD without
unintended overlap. It does not execute pickups, predict slipping, measure
assembly precision, or call a VLM. Other parts, the arm, and robot self-collision
are outside this isolated check. A passing pose is a geometric candidate, not
a demonstrated successful pickup.

## Completed results

The corrected initial search finds passing poses for **88/96 placements**.
All eight misses are USB: sphere-based seeds concentrate on the cable and miss
usable upper connector surfaces. A separate, explicitly exploratory CAD surface
sampler recovers all eight, giving **96/96 placements with geometric candidates**.
It changes candidate coverage, not the checker thresholds.

Across the initial search plus surface follow-up, **401,424 proposals** are
checked; **75,622 pass**, represented by **16,506 groups**. Every distinct proposal
is checked before grouping, and no winning grasp is chosen. Per-placement grouped
sets range from **2 to 912**, so compact candidate presentation remains a limitation.
Median generation/check time is **14.97 s**, mean **26.55 s**, range **0.51--202.11 s**
with four concurrent CPU workers. These include CAD loading/checking and evidence
writing, but exclude imports and input pose fitting.

Candidate availability is not assembly suitability. In N00 the 8 mm round pin
has only two retained poses; both hold its ends, about 50 mm apart, with the jaws
exchanged. The VLM would not have another contact region to choose in that case.
Thus 96/96 geometric availability does not establish 96/96 high-quality assembly
grasps. The gallery keeps this limitation visible rather than filtering it away.

[Review every retained pose](../../outputs/cad_grasp_geometry_20261026/runs/20261026T172654Z_surface_coverage/gallery/index.html)
and the [per-part report](../../outputs/cad_grasp_geometry_20261026/runs/20261026T172654Z_surface_coverage/report.md).
The [corrected initial report](../../outputs/cad_grasp_geometry_20261026/runs/20261026T171446Z/report.md)
preserves the original 88/96 outcome. The surface follow-up is an improvement
developed after observing those misses, not a held-out comparison.

Verification: **21 focused tests pass**. The corrected initial 96-case audit and
the [combined independent audit](../../outputs/cad_grasp_geometry_20261026/runs/20261026T172654Z_surface_coverage/audit.json)
both pass with zero failures. The combined audit checks inventory, every saved
candidate decision/count, group membership, rigid downward frames, native CAD
contact witnesses, pad bounds, full observed table plane, opening limits,
first-contact brackets, cumulative times and preserved critical source/input
hashes. It independently replays saved geometry quantities; it does not rerun
the collision kernel or establish physical grasp stability.

## Inputs and method

N00--N07 each contain twelve products: 96 placements. They are the existing
eight classification scenes, not copies of the older fourteen-part pickup scene.
Correct CAD-to-crop identity is explicitly supplied. Saved observed-depth CAD
fits provide position/orientation, transformed using the saved camera calibration.
The table plane is fitted from observed depth. No simulator object pose is read.
One missing correct-crop fit is recomputed with the unchanged depth fitter, and
saved separately. See [INPUTS.md](INPUTS.md).

The candidate generator uses the pinned EmbodiChain GraspKit ray sampler for
seeds (512 surface rays). It recasts horizontal rays through those seeds so the
two contacts have the same world height. This is necessary for a downward hand
whose fingers close horizontally. Both contact normals must face the closing
axis within 15 degrees, with opposing normals within 30 degrees. Repeated pairs
are removed on a 0.01 mm coordinate grid. There is no 24-pair cap.

For every distinct pair, test both jaw assignments and the six historical depths
-8, -4, 0, +4, +6, +8 mm. The approach is exactly world down. Rotation in the
table plane follows the contact direction. All generated configurations are
validated, with a primary rejection reason saved for each; checking does not
stop after finding a passing candidate.

The actual Panda collision components are exported from the existing model
without stepping physics. Each hand component uses its collision convex hull,
as in MuJoCo; the target retains its native triangles and holes. The maximum
commanded opening is 80 mm, but the actual inner-pad gap is opening minus 1 mm.
Retaining the historical 12 mm total extra approach opening therefore gives a
maximum contact separation of 67 mm. The smallest separation is a 1 micrometre
numerical guard, not a physical lower gripping limit.

## Declared geometry checks

- The proposed point must be inside each finite inner pad, at least 0.25 mm
  from its depth boundary. The +6/+8 mm historical positions are still tested;
  their proposed points lie beyond the current pad and fail this requirement.
- Complete open-hand components and their continuous 100 mm downward sweep
  must clear the target using a 0.1 mm conservative supporting-plane guard.
  This guard is not an exact measured Euclidean distance.
- Every hand component must remain at least 0.1 mm above the full observed table
  plane. Endpoints suffice for each straight translating component's plane test.
- Locate each pad's first contact by bisection on continuous swept geometry.
  The fraction bracket is at most 0.0001 of finger travel (at most 4 micrometres).
  Examine all simultaneous target triangles; an arbitrary first face at a CAD
  edge must not determine the answer.
- Contact must occur on an inner pad face, within 25 micrometres numerical
  tolerance, on a target surface facing finger closing within 15 degrees.
- First-contact opening widths must agree within 0.05 mm. Stop at the earlier
  contact: the other pad is within a declared near-contact bracket, not pushed
  through the CAD to force two exact contacts. Exact simultaneous physical
  contact is not established by that tolerance.
- Finger bodies and palm must avoid the target throughout modeled closing up
  to that stop. Actual robot joints, forces and object settling are not simulated.

These clearances, angles and tolerances are explicit local trial settings, not
calibrated physical assembly tolerances. Candidate quality here means passing
these stated geometry requirements. Pad support area, friction, gravity torque,
pose uncertainty and physical holding success are not certified.

The sampled pair proposes a hand placement. The finite modeled pads can touch
nearby CAD features first, especially gear teeth, at a different opening.
Save those recomputed first contacts separately. The gallery distinguishes the
sampled pair from the modeled contacts; it must not draw the sampled pair as
though those were the actual finger touches. The 12 mm extra opening is relative
to the sampled width; actual open-hand clearance is established by the checker.

After validation, group nearby passing poses using 4 mm contact/hand position,
10 degree yaw and 2 mm opening differences. Preserve different finger depths
and opposite jaw assignments. Each group records all member IDs. Representative
IDs are for inspection, not a quality ranking. There is no target of twelve
poses, padding, VLM call, or hidden final winner.

## Native mesh checks

Exact-coordinate STL vertex welding changes no geometry. The large gear also
requires an explicit 50 nm seam weld: maximum vertex displacement 10.70 nm,
all 22,884 faces and both components preserved, no surface filling or target
hull. Source CAD files are unchanged. Closed disconnected components are treated
as a union, which avoids cancelling overlapping USB component volumes. Internal
component surfaces remain and can conservatively reject a grasp. Topological
closure is checked; arbitrary self-intersecting CAD is not certified.

## Run and inspect

Use the existing WSL Python environment; no installation or weights are needed:

```text
python -m pytest experiments/cad_grasp_geometry_20261026 -q
python -u -m experiments.cad_grasp_geometry_20261026.run_trial
python -m experiments.cad_grasp_geometry_20261026.gallery RUN_PATH
```

`outputs/cad_grasp_geometry_20261026/runs/<UTC>/` contains the frozen settings,
per-candidate decisions, per-placement results and gallery. Development smoke
runs are separate from the full 96-placement run. Timings exclude imports and
input pose fitting; each placement includes CAD loading, collision setup,
sampling, validation and grouping. Four CPU workers run concurrently, so summed
placement times and elapsed experiment time are different measures.

## Separately declared height-coverage follow-up

The first full run's USB proposals did not sample sufficiently high side
contacts. An isolated N00 diagnostic found a passing pose at 6.426 mm contact
height with 1.125 mm table clearance, while the highest original contact was
4.141 mm and its best hand/table clearance was -1.160 mm. This is a search miss,
not an impossible grasp and not a reason to loosen the checker.

The exploratory follow-up revisits every zero-grasp placement, regardless of
part identity. Recast its existing horizontal lines at a fixed 2 mm height grid
from CAD minimum height +1 mm through maximum height -1 mm, including the upper
endpoint. Validate every resulting pose with the same checker/depths/tolerances;
do not stop after the first passing pose. Two millimetres is a declared search
resolution, not a physical optimality claim. Passing initial placements are
reused unchanged. Initial evidence remains frozen and is reported separately.

```text
python -m experiments.cad_grasp_geometry_20261026.height_followup BASE_RUN
```

The follow-up saves a new combined 96-placement result, its eight (or other
actual number of) retried cases, all additional candidate checks, cumulative
times and source hashes. This is an exploratory improvement after observing
the initial failure, not a fresh held-out benchmark.

## Separate surface-coverage variant

The height grid fixes only one USB scene: its existing horizontal lines still
mostly pass through the cable. A second exploratory variant samples the native
CAD directly: 512 points distributed by triangle area, with deterministic
Halton barycentric positions inside each chosen triangle. Project each usable
surface normal horizontally, recast both contacts against the CAD at that
point's height, and apply the same angle/width/contact/hand/table checks. This
is a geometric seeding variant we added; it is not unchanged GraspKit code.

The standalone eight-scene USB diagnostic checks 10,044 proposals and finds
26 valid poses / 10 groups in every scene. Its independent audit passes. No
checker tolerances, hand geometry or CAD poses were changed to achieve this.
When applied to a completed baseline, the same part-independent rule retries
every zero-grasp placement and preserves all passing baseline records.

```text
python -m experiments.cad_grasp_geometry_20261026.surface_coverage BASE_RUN
```

This is an exploratory coverage repair after observing the initial misses,
not a preregistered performance claim or a new grasping theory.

## Numerical correction and preserved development evidence

The development run `20261026T165914Z` failed its independent contact audit:
24 saved contact witnesses in five placements lay outside their recorded CAD
triangles, by up to 0.1933 mm. Triangle clipping used different numerical
boundaries for inside classification and edge intersection, allowing an
interpolation to leave its original edge. The corrected kernel uses one
boundary and clamps interpolation to the edge. The old kernel and driver are
archived under the new study's `diagnostics`; old results/audits are unchanged.
Rechecking all 24 witnesses passes. A full new run `20261026T171446Z` uses the
corrected kernel and double-precision hand-frame construction; 21 focused tests
pass. Treat the earlier failed-audit run as development evidence only.
