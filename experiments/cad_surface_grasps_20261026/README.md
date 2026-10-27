# Surface-derived CAD grasps with a straight-down approach

This experiment replaces arbitrary native CAD X/Y/Z grasp directions with
directions obtained from opposing flat faces and verified local circular bands.
It starts with the same saved observed-depth CAD alignment and real modeled
Panda hand. The approach remains exactly down. It does not execute a pickup,
assemble a part, call a VLM, or read simulator object positions/orientations.

[Review all revised grasps](../../outputs/cad_surface_grasps_20261026/runs/20261026T200202Z/gallery/index.html)
and the [full comparison](../../outputs/cad_surface_grasps_20261026/runs/20261026T200202Z/report.md).

| Same 96 placements | Native-axis revision | Surface-derived revision |
|---|---:|---:|
| Placements with a candidate | 88 | 87 |
| Full hand proposals | 414 | 518 |
| Passing hand alternatives | 347 | 359 |
| Displayed families | 175 | 182 |
| Families per placement | 0--5 | 0--4 |
| Median recorded generation/check time | 0.121 s | 0.702 s |

This is a correction to direction generation and a stricter contact test, not
an overall coverage or speed improvement over the native-axis revision. The
12 mm and 16 mm pins now have round-side options in all eight scenes. The 8 mm
pin has a round-side option in one upright placement and only end grips in the
seven lying placements. BNC has round-side options in N02/N05/N07; no option in
N00/N01/N03/N04/N06. Four large-gear placements still have no option. All KET
pins, nuts, USB, DSUB and waterproof connectors retain options in every scene.
The largest retained flat-normal error is 0.093745 degrees. These are geometric
results only. Time includes new feature extraction and contact-polygon checks;
it omits startup, pose fitting, VLM/rendering and some final result writing.
The independent final audit passes 187,925 checks with zero failures, including
circle fits, fitted-band triangle membership, contact geometry and centering.
All 28 focused tests pass. The failed development run `195205Z` is preserved;
its contact points could leave the classified band. The corrected final run
`200202Z` restricts paired points to the same band and fully revalidates the hand.

The useful open-source reference is Wan's MIT `one` antipodal planner. Its
surface-normal and opposing-ray principles are adapted through existing local
helpers; the upstream robot framework is not installed or executed. The
[source review](source_review.md) and [pinned source manifest](source_manifest.json)
separate upstream behavior from our additions. There is no claim that this is
a full reproduction of that planner or the assembly paper.

## The three problems being corrected

1. **An arbitrary CAD roll blocked round-side grasps.** A circular pin has a
   meaningful shaft direction, but rotating its CAD X/Y frame around that shaft
   need not change its round surface. Requiring those arbitrary axes to be
   horizontal removed otherwise useful directions. Local round BNC sections
   have the same issue; this does not make the entire connector symmetric.
2. **The old green line could misrepresent the hand.** In N02 KET16, one
   independent witness-to-witness line was 44.853 degrees from closing, but
   actual hand-to-flat-face normal error was only 0.001561 degrees. The source
   faces had substantial finite-pad projected overlap, with only about
   0.294 micrometers of plane-height variation across the patch. These are
   geometric calculations, not measurements of loaded contact or zero slip.
3. **One retained family sometimes meant only an end grip.** N00 16 mm round
   pin lost its side direction because its arbitrary transverse CAD axis was
   tilted 1.26790 degrees. N00 8 mm round pin lost its arbitrary transverse
   directions too, but also has a separate pad/table height limitation. The
   [reproducible diagnosis](../../outputs/cad_surface_grasps_20261026/compact_diagnostics.json)
   records these differences rather than treating all missing sides alike.

The new viewer uses paired native-surface locations at the same hand Y/Z to
draw the green pair. It also shows both finite near-contact regions and actual
normal errors. Original independent first-touch witnesses remain optional.
The new pair is a geometric construction inside a declared proximity band;
it is not a measurement of the center of pressure.

## Directions from physical surfaces

World up is `u=(0,0,1)`, hand approach is `a=-u`, and finger closing is `x`:

\[
\|x\|=1,\quad x^Tu=0,\quad R_{WG}=[x,\;a\times x,\;a].
\]

Native CAD frame labels do not enter direction generation.

### Flat faces

Cluster native surface normals and require opposing supported faces. A connected
flat patch must have at least 16 square millimeters of area and spans of at
least 2 mm in both in-plane directions. Disconnected tooth strips do not become
one broad face merely because their normals agree. These are explicit local
controls, not universal published constants or calibrated holding thresholds.

Keep a direction only when its surface normal is within 1 degree of horizontal,
then project it horizontally. Recheck the actual native first-touch normals
after full hand validation:

\[
\epsilon_i=\arccos(|n_i^Tx|)\leq1^\circ\quad(i=1,2).
\]

For flat faces, the inner pad **plane** should be parallel to its object face;
the plane **normals** align or oppose. The pad surface itself is not perpendicular
to the object face. Nearly opposing faces and usable finite contact regions are
additional requirements.

### Round sides

Use native triangle normals and positions to form finite shaft-axis hypotheses,
then verify circular bands around them. For fitted shaft direction `s` that is
not vertical, choose

\[
x=\frac{u\times s}{\|u\times s\|}.
\]

This direction is horizontal, transverse to the shaft and independent of an
arbitrary CAD rotation around it. For a nearly vertical shaft, use two canonical
world horizontal directions to represent its radial family. They remain subject
to the complete target checks; no whole-object or environment symmetry is assumed.

Round-band verification requires at least 18 of 24 angular bins, axial span of
at least 3 mm, and 95th-percentile native-vertex radius residual no larger than
`max(25 micrometers, 0.2% of radius)`. Strong support is required to introduce
a shaft direction. Thin native bands can be separately identified as contact
geometry when coaxial with a strongly supported shaft; they do not introduce
additional directions or claim sufficient holding area. Nearby distinct radii
are separated before refitting, preserving the same residual limits.

The fitting procedure tests a finite set of axes from area-weighted normal and
position statistics. It is not an exhaustive detector of every cylindrical
feature. A verified local band is not a declaration that a gear or BNC is a
perfect cylinder as a whole.

## From directions to a small set of hand poses

Reuse the previous compact contact-region sampling, pad-size grouping and one
calculated height per region. There is no six-depth sweep, arbitrary yaw sweep,
top-twelve cap or stop at the first successful hand. Both jaw assignments are
checked because the complete hand is not perfectly symmetric. Actual identical
final grips merge after checking, while all valid checked alternatives remain.

Before checking the two jaw assignments, center the hand on the actual finite
pad first contacts. If the two commanded openings at first contact are `w_L`
and `w_R`, the Panda's finger motion of minus/plus half the opening gives

\[
\Delta x=(w_R-w_L)/4,\qquad t'=t+\Delta x\,x.
\]

This is a contact-midpoint correction from the declared finger kinematics,
not a weighted quality score. Calculate it once in the positive-jaw frame and
use the same corrected position for both jaw assignments. Fully recheck both
hands afterward, including the approach and table. The trial records 518 extra
finite-pad first-contact queries for this calculation, separately from its 518
full hand proposals; no physical closing action or position sweep is performed.

The unchanged checker handles full native target geometry, open hand and its
100 mm approach, finite pads, first contact during closure, finger/palm
interference and the full observed table plane. Other parts, arm reach and robot
self-contact remain outside this controlled study. Target holes and the earlier
declared numerical gear seam treatment are preserved.

The added checker then requires two opposed near-contact regions inside the
finite pads, excluding the existing 0.25 mm pad-edge guard. Both regions must
share an in-plane location. Use the shared location nearest the intended pad
center, rather than connecting unrelated triangle witnesses.
This shared-region rule is our conservative centering screen, not a published
necessary condition for every stable soft-pad grasp. It can reject grasps that
real compliant pads could hold; its effect on physical shift remains untested.

For each pad, native triangles are clipped to its finite face bounds and a
25 micrometer band around its own first-contact plane. The band is a numerical
geometry tolerance, **not rubber compression, loaded contact area or pressure**.
The paired points lie on the original triangles and have the same hand Y/Z.

### Curved CAD surfaces and mesh resolution

A triangulated cylinder has flat facets and vertices; these differ from an ideal
smooth cylinder. Record three distinct quantities instead of calling them all
exact tangency:

- actual native facet-normal error, limited by the unchanged 15-degree mesh screen;
- closing-versus-shaft departure from perpendicular, limited to 1 degree;
- fitted radial-normal error at each paired native point.

For the third quantity, the explicit allowance is 1 degree plus that paired
native triangle's radial angular span. The independent verifier recalculates
the span from its three native vertices. This is a mesh-resolution allowance,
not a relaxation of the flat-face 1-degree check. The viewer reports the raw
errors and allowances separately. It does not claim every curved contact has
a normal error below 1 degree.

## Why the 8 mm side can remain unavailable

For the lying N00 8 mm pin, the lowest guarded pad location that clears the
table is 5.279086 mm; the sampled qualified side region ends at 5.057108 mm.
The shortfall is 0.221978 mm. N02 has a 0.222096 mm shortfall. Thus recovering
a transverse direction does not make the stock pad reach that side region.
The corresponding retained end grip must be labeled as an end grip.

This is a bounded result for the current pad, table and approach rules, not proof
that the part can never be grasped. A support fixture, different pad extension or
different approach would be a separate setup change. None was silently applied
to this experiment to make the count improve.

## BNC exclusions after recovering its radial direction

The arbitrary CAD-roll restriction is removed, but stricter finite contact
checking still excludes five BNC placements. For N00, two representative hands
have no common opposing region within the 25 micrometer bands: the nearest
projected strip-box gaps are both about 0.2605 mm. An independent convex
halfspace/linear-program check confirms the empty intersections; it does not
reuse the contact polygon clipping routine. N01 examples have gaps of
about 0.1667--0.1857 mm. These are individual disconnected near-contact strips,
not a gap between the whole objects. Some other candidates fail actual normal
alignment. Thus these are additional conservative contact
rejections, not continuing failures to propose a radial direction.

See [BNC diagnosis](../../outputs/cad_surface_grasps_20261026/runs/20261026T200202Z/bnc_region_diagnosis.json).
The nine unavailable placements are preserved; neither band tolerances nor the
table are altered to inflate coverage.
There is also residual near-duplication: N02 BNC families S0005 and S0007 are
only 0.486 mm apart on the same fitted band. They remain distinct under the
current physical-pose equality test and are visible in the gallery. Thus this
small set is not claimed to be an optimal minimal set of task-distinct grasps.

## What the result can establish

The output is a variable-size set of surface-aligned geometrically feasible
options before VLM selection. It is not a universal grasp-quality equation.
Friction, force, pad deformation and movement during closing are not validated.
Even exactly parallel flat faces do not prove zero shift or reliable assembly.
No automatic assembly-error optimizer or after-execution precision measurement
is added. Known assembly context remains intended input to later VLM selection.

## Reproduce

Use the existing WSL environment; no package installation or weight download:

```text
python -m pytest experiments/cad_surface_grasps_20261026 -q
python -m experiments.cad_surface_grasps_20261026.run
python -m experiments.cad_surface_grasps_20261026.verify_results RUN_PATH
python -m experiments.cad_surface_grasps_20261026.analyze RUN_PATH
python -m experiments.cad_surface_grasps_20261026.gallery RUN_PATH
node RUN_PATH/gallery/verify_viewer.cjs
```

Each new UTC run preserves settings, generation source hashes, original observed
inputs, candidate checks, explicit failures and an evidence-preservation audit.
Earlier compact/exhaustive outputs and locked CAD description prompts are not
rewritten. Development runs remain development evidence, including failed fits
and missing options.
