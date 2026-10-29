# CAD grasp selection with explicit priorities

This is the historical October 28 revision requested after the Fabrica studies.
The [October 29 fingertip adapter](../cad_grasp_selection_20261029/README.md)
and [corrected scene/task inputs](../nist_scene_corrections_20261029/README.md)
are current. This revision's Python sources and measured results are preserved.
It implements a callable selector and a small JSON entry point. It has offline
tests and a completed CAD-derived geometric trial over N00–N07. No new perception
run, VLM call, robot motion, pickup or assembly execution has run.
Earlier study sources, numerical results and galleries remain unchanged.

The [eight-scene trial report](../../outputs/cad_grasp_selection_20261028/runs/20261028T190844Z/report.md)
records **4 selected placements out of 96**, from 9,600 proposed poses. The
[selected-only gallery](../../outputs/cad_grasp_selection_20261028/runs/20261028T190844Z/gallery/index.html)
contains all eight scenes, including explicit failed-search labels.

The sequence is:

**At most 100 complete poses → mandatory geometry checks → geometric criteria
→ one selected grasp.**

This geometric decision is the final grasp selection. There is no VLM ranking,
approval or pose-selection stage after it. The output is the best valid sampled
grasp under the declared priorities; an empty valid set returns no grasp. Older
VLM experiments and frozen gallery captions describe their historical scope.
Grasp generation, pickup and assembly remain paused during the
[mating-goal review](../nist_mating_goals_20261028/README.md).

There is no mandatory downward approach, grasp grouping, clustering, diversity
reduction, independent finger-depth parameter, center-of-mass criterion or
weighted combined score. The primary result contains one grasp, not a shortlist.

## Files and existing utilities

- `sampler.py`: reuses the area-uniform surface sampler and all-forward-hit ray
  query from `../fabrica_candidates_20261027/sampler.py`. It does not call that
  study's pair-thinning or centroid-sorting function.
- `geometry.py`: reuses `../cad_grasp_geometry_20261026/geometry_kernel.py`
  for unchanged CAD triangles, convex hand components and solid containment.
- `criteria.py`: fixed CAD reference quantities and sequential priority selection.
- `pipeline.py`: input validation, generation, validation and one selected output.
- `task_geometry.py`: measured CAD axes, bore/seat labels and conservative
  connector feature envelopes, with full-polygon protected-surface classification.
- `run_nist.py`: controlled 96-placement trial using the frozen observed fits.
- `verify_nist.py`: saved-result transform, scoring, selection and source audits.
- `gallery.py`: selected-only eight-scene CAD contact sheets.
- `test_*.py`: offline numerical, calibration, geometry and integration checks.

This is **Fabrica-derived**, not execution of the complete original Fabrica
system. The original native baseline under `../fabrica_original_20261027/`
continues to describe its original run. This revision uses the existing NumPy
ray port and SciPy collision kernel, not Fabrica's entire FCL/robot/fixture stack.

## Required inputs and frames

`select_grasp(request, ...)` takes an explicit dictionary:

| Key | Meaning |
|---|---|
| `vertices_cad_m`, `faces` | Complete native-scale CAD mesh: `(V,3)` coordinates and integer `(F,3)` triangle indices |
| `world_T_cad_pickup` | Estimated CAD-to-workspace pickup transform |
| `world_T_cad_final` | Known nominal final transform, including orientation, in the same workspace frame |
| `axis_point_cad_m`, `axis_direction_cad` | Explicit mating-axis point and unit direction in CAD coordinates |
| `table_plane_world` | Upward unit plane `[nx,ny,nz,offset]`, with `n·x + offset = 0` |
| `hand` | Complete saved convex hand components, usable pads, frame calibration and opening limits |
| `protected_contact_check` | Explicit protected-surface classifier described below |
| `receivers` | Nonempty list of `{name, vertices, faces, world_T_mesh}` receiver solids |

All lengths are metres and all angles radians. The selector reads no simulator
object state. It does not run or repair pose estimation. Task geometry must be
supplied explicitly; final translation alone does not define the mating axis.

The supported saved hand is
`outputs/cad_grasp_geometry_20261026/hand_model.json`: the Panda grip-site frame
has X closing and +Z approaching the object. Its five convex collision solids
represent the saved simulator hand model. They are **not measured physical hand
calibration**. Unknown frame mappings return unavailable rather than using a
guessed Fabrica hand-base offset.

The CAD/receiver solids must have closed, consistently outward-oriented manifold
shells after exact vertex welding. The existing collider treats disconnected
closed components as a union of solids. Separately encoded nested cavity shells
and arbitrary nonconvex hand components are not silently converted into a new
solid model. Self-intersection-free solids are an input assumption; the topology
audit is not a complete self-intersection certification.

## Candidate construction

Sample triangles in proportion to area and sample points uniformly inside them.
From point `p1` with outward normal `n1`, cast inward along `-n1`, beginning at
`p1 - epsilon*n1`. Consider all forward ray intersections, not just the first or
farthest one. A span across a gear or nut hole is permitted.

For an opposing pair:

\[
u=(p_2-p_1)/\|p_2-p_1\|,\quad
w=\|p_2-p_1\|,\quad m=(p_1+p_2)/2.
\]

The proposal tests are `n1·(-u) >= 0.95` and `n2·u >= 0.95`. These are broad
proposal conditions, not the final pad-contact tolerance. Degenerate spans fail.

Sample approach orientations around `u` over the full circle, using a seeded
random phase per pair. The approach remains perpendicular to `u`; upward,
sideways and tilted orientations are not removed by an angle gate. Randomly
choose at most 100 pair/orientation combinations **before full validation**.
Do not sort or truncate them by centroid distance. Numerical merging of a ray's
coincident triangle-edge intersections is not grasp grouping.

Both sampled points are the intended centers of the usable inner pad faces.
For calibrated pad-center functions `s1(q), s2(q)`, solve the opening command
`q` from their actual separation. Then use

\[
o_H(q)=\tfrac12(s_1(q)+s_2(q)),\qquad
t_H^O=m^O-R_H^O o_H(q).
\]

Both reconstructed centers must match the sampled points. Do not replace `m`
with an observed contact-patch centroid or shift the hand using a compensating
depth parameter. Different contact locations produce different complete poses.

For the saved Panda, actual gap is `w = q - 0.001 m`. Thus `q = w + 0.001 m`.
The pad-center midpoint is approximately `[0,0,-0.0036] m` in grip-site coordinates.
This geometry is used directly rather than equating command opening with gap.

Defaults are seed 0, 200 sampled surface points, 10 orientations per pair,
100 complete-pose maximum, and a 0.5 mm inward ray offset. There is one fixed
surface budget, without an unbounded loop seeking successful grasps. A thin
feature can be missed by the ray offset, and a finite bank can miss feasible
grasps. The existing query's ray batch is 8 and coincident-hit tolerance 1 nm.

Default pre-opening adds 2 mm to the physical total gap, converted through the
calibrated gap-per-command slope. This is 1 mm per side for the saved symmetric
Panda. Both holding and pre-opening commands must satisfy the actual limits.

## Mandatory geometry checks

All of the following must pass; a good criterion cannot compensate for failure:

1. Holding and pre-opening commands lie within calibrated opening limits.
2. Both intended pad centers reconstruct correctly and lie on CAD surfaces with
   acceptable opposing normals. Surrounding finite pad contact must be compatible.
   Cylindrical tangent contact is allowed; positive-area flat CAD faces are not
   required for round parts.
3. The complete hand does not penetrate the target. Part contact belongs only
   to the usable inner pad faces. Solid containment is checked, in addition to
   triangle intersections.
4. The complete modeled closing sweep is clear. Each saved convex component
   translates linearly with opening. Its swept volume is exactly the convex hull
   of its endpoint vertices under this model, so no discrete closing samples are
   used. Numerical collision tolerance still applies.
5. Every component clears the table throughout closing. The minimum signed
   plane distance occurs at an endpoint for this fixed-orientation linear model.
6. Both intended centers and complete modeled contact polygons avoid protected
   mating surfaces.
7. The complete held hand clears the receiver at the supplied nominal final pose:

\[
T^W_{H,p}=T^W_{O,p}T^O_H,\qquad
T^W_{H,f}=T^W_{O,f}T^O_H.
\]

The last calculation assumes a fixed part-to-hand transform. It does not measure
post-pickup movement or certify an insertion path.

Default numerical/model settings are recorded with every validation:

| Setting | Value / meaning |
|---|---|
| Pad normal tolerance | 1 degree; explicit local requirement, not Fabrica's 0.95 proposal threshold |
| Intended-point/reconstruction tolerance | 1 micrometre |
| Intended pad-center edge margin | 0.25 mm; actual flat contact can reach pad boundaries |
| Table clearance | 0.1 mm |
| Pad collision allowance | 10 nm outward numerical retraction before the closing sweep |
| Existing kernel boundary tolerance | 0.1 nm |

Actual contact-region queries use only the kernel boundary tolerance, not the
1 micrometre intended-point tolerance. This avoids treating nearby curved CAD
triangles as actual contact. Boundary-only witnesses from an adjacent face do
not invalidate aligned support solely because their normals differ. The
numerical area threshold is `8 * kernel_tolerance * max(pad_width,pad_height)`
square metres and is recorded per pad. Positive-area incompatible support still
fails; the independent full-pad penetration sweep remains mandatory. Boundary
witnesses are still passed to the protected-surface classifier.

These are declared implementation settings, not proven assembly-accuracy margins.
The collision model may reject conservative convex-component overlaps. Do not
describe its results as measured physical stability.

### Protected-surface callback

The callback must support:

```python
check(points_cad, *, contact_polygons, contact_face_ids)
# returns:
{
    "status": "available_...",
    "eligible": boolean_array_of_length_N,
    "polygon_eligible": boolean_array_of_length_P,
}
```

All supplied points and polygons are in native CAD coordinates. Source face IDs
retain the input face order. The callback must certify **entire** polygons,
including their interiors, and support an empty availability query. Checking
vertices alone cannot exclude an arbitrary protected island within a contact
patch. Unavailable/missing/unknown statuses never count as passing.

`protected_triangle_check(...)` builds this callback for explicit protected CAD
face IDs. It does not discover mating surfaces. For partial-face regions such
as a pin's inserted segment, supply a geometric region callback instead of
reserving the entire long CAD face. A convex clear half-space can be checked
using all polygon extrema; arbitrary curved regions need their own conservative
polygon intersection check.

## Geometric criteria and priority selection

Compute the **area-weighted surface centroid**, not a vertex average or CoM:

\[
c_S=\frac{\sum_f A_f(v_{f,1}+v_{f,2}+v_{f,3})/3}{\sum_f A_f}.
\]

For explicit unit mating direction `d` and axis point `a`:

\[
q_\tau(g)=\|(m(g)-a)\times d\|,\qquad q_S(g)=\|m(g)-c_S\|.
\]

The first is a moment arm for an assumed axial load, not an actual torque
measurement or complete stability calculation. The second is geometric surface
centrality; it does not claim to minimize gravitational torque.

Compute fixed scales once from the complete CAD vertices:

\[
D_\tau=\max_v\|(v-a)\times d\|,\quad D_S=\max_v\|v-c_S\|,
\qquad r_\tau=q_\tau/D_\tau,\quad r_S=q_S/D_S.
\]

Both ratios lie in [0,1] for the stipulated surface-contact midpoint: the
midpoint lies in the convex hull of the CAD vertices and both distance functions
are convex. A zero denominator contributes zero/no distinction; inconsistent
nonzero values are rejected. Roundoff beyond `1e-10` outside the normalized
bounds fails rather than being hidden by clipping. The zero-distance allowance
is `1e-12 m`. There is no candidate-dependent min–max normalization.

At pickup only, let `a_g` point from the pregrasp position toward the object:

\[
\theta_{down}=\arccos(\operatorname{clip}((a_g^W)^T(-z_T^W),-1,1)).
\]

This is an approach-direction preference, not a jaw-direction constraint or a
claim about gravitational torque or robot loading. Downward approach tends to
put a conventional hand's palm above its contacts; the complete geometry checks
still decide clearance.

Select sequentially, without a combined score:

1. Keep candidates within `epsilon_tau` of the current minimum `r_tau`.
2. Within that set, keep those within `epsilon_surface` of minimum `r_surface`.
3. Within that set, keep those within `epsilon_angle` of minimum downward angle.
4. Choose the smallest deterministic candidate ID among final numerical ties.

Defaults are `1e-10`, `1e-10`, and `1e-10 rad`. They are equality tolerances,
not adjustable preference weights. Each stage compares with its own minimum;
there is no nontransitive tolerance-based pairwise comparator. A lower-priority
improvement cannot override a higher-priority difference outside its tolerance.
Similar-looking poses are never suppressed.

## Calling the revision

Programmatic integration:

```python
from experiments.cad_grasp_selection_20261028.pipeline import select_grasp

debug = []
result = select_grasp(request, seed=0, max_poses=100,
                      surface_budget=200, orientation_samples=10,
                      diagnostics=debug)
```

The JSON entry point accepts the same array-based request, replacing the callback
with explicit `protected_face_indices`. `hand_model_file` may replace inline
`hand`; relative paths resolve against the request file. Optional
`geometry_config` and `tie_tolerances` retain the same meanings described above.

```powershell
python -B -m experiments.cad_grasp_selection_20261028.pipeline request.json --output outputs/cad_grasp_selection_20261028/selection.json --diagnostics outputs/cad_grasp_selection_20261028/diagnostics.json
```

The eight-scene trial uses the programmatic entry point with explicit CAD-derived
region callbacks. Do not substitute nominal or simulator object state for
missing observed pickup poses.

The output status is one of:

- `selected`: one pickup pose, corresponding nominal final pose, contact pair,
  opening values, approach direction, raw/normalized criteria, check results,
  seed, candidate ID and proposed/valid counts.
- `unavailable_inputs`: explicitly missing/invalid geometry, task information,
  calibrated mapping or numerical inputs; no selected fallback.
- `no_valid_grasp_within_budget`: "No valid grasp found within the sampling
  budget." This does not assert that the part has no feasible grasp.

Candidate diagnostics are optional and separate from the primary result. No
VLM, physical robot, simulator execution, pose fitting or model training runs.
Robot-arm reachability, full pregrasp approach, insertion path, release, friction,
closing-induced part motion and actual assembly precision remain unchecked.

## CAD-derived task inputs and remaining assembly limits

The frozen manifest has 96 observed-fit records: twelve parts across N00–N07,
not the older fourteen-product inventory. Read it directly rather than invoking
the old helper that can automatically refit missing poses.

`task_geometry.derive_task(part, vertices, faces)` now supplies explicit axis
points/directions and protected-region callbacks for all twelve CAD products.
The [trial report](../../outputs/cad_grasp_selection_20261028/runs/20261028T190844Z/report.md)
and per-part `derived_tasks/` receipts describe the measurements and thresholds.
Pins use longitudinal geometry and the unchanged nominal engagement plane;
gear/nut axes use both circular bore-mouth loops rather than a threaded middle
section. Connector shoulders/open ends and USB plug/body interfaces supply
conservative protected feature envelopes, not inferred engagement depths.

The older task helper's unavailable connector-region placeholders are not used
as passing checks. The new callbacks classify entire modeled contact polygons.
Their feature rules are specific to these CAD families, not a general semantic
CAD recognition claim.

Existing task-goal/receiver transforms use board-centered XY with original world
Z; the pickup records use their saved workspace frame. A caller must explicitly
convert both into one workspace frame. The selector does not guess that mapping.
The trial explicitly uses nominal task-board XY=(0,0), while leaving observed
pickup transforms unchanged. This does not claim that a board was observed there.
Gear/nut goals and pin engagement depths remain nominal assumptions, not verified
physical mating transforms. Connector goals still have zero engagement; DSUB,
Waterproof and the frozen USB cable model expose feature/goal orientation
mismatches. The goals were preserved and these limits are recorded. Physical
hand/camera/board calibration must be supplied before a physical test.

## Eight-scene run

```powershell
python -u -B -m experiments.cad_grasp_selection_20261028.run_nist --workers 4
python -B -m experiments.cad_grasp_selection_20261028.verify_nist RUN_DIRECTORY
python -B -m experiments.cad_grasp_selection_20261028.gallery RUN_DIRECTORY
```

The saved run is `outputs/cad_grasp_selection_20261028/runs/20261028T190844Z`.
Each case generated 100 complete poses. Eight poses passed, supporting selected
grasps for KET16 in N02/N05 and DSUB in N04/N07. Selection took 116.04 seconds
wall time on four workers, excluding tests, gallery and post-run audits.

The independent table audit found 8,793 of 9,600 proposals fail the 0.1 mm table
clearance. Small parts lying on the table can be incompatible with contacting
the centers of full 16×16 mm pads. Round-pin facet normals also differ by about
1.2675°–2.5353°, exceeding the unchanged 1-degree final contact rule. These are
limitations of the current sampling/contact convention and modeled checks;
the run did not relax them, add fingertip-depth variants or resample failures.
First-rejection counts are not independent counts of every violated constraint.

Gear Large uses the previously documented 50 nm seam-neighbor weld in a recorded
validation copy; maximum vertex movement is 10.703 nm, and the original mesh is
unchanged. All inputs, original results, protected VLM code/prompts and launched
source hashes were checked unchanged. Geometric selection does not establish
pickup or assembly success.

## Offline verification

```powershell
python -B -m pytest -q -p no:cacheprovider experiments/cad_grasp_selection_20261028
```

Initial implementation verification passed 40 revision tests and ten existing
collision-kernel regressions. Before the eight-scene trial, the same command
passed **69 tests**, including the added CAD-feature, preparation and gallery
tests (5.58 seconds in the existing Windows Python environment):

```powershell
python -B -m pytest -q -p no:cacheprovider experiments/cad_grasp_selection_20261028 experiments/cad_grasp_geometry_20261026/test_geometry_kernel.py
```

Tests cover actual saved pad-center reconstruction, unrestricted approaches,
round-side contact, whole-hand intersection and containment, an obstruction
between clear closing endpoints, table/final receiver checks, complete protected
patches, normalization bounds and zeros, priority ties, the complete-pose cap,
empty/missing inputs and one-output integration. These are synthetic/calibration
checks, not NIST pickup or assembly success measurements.
