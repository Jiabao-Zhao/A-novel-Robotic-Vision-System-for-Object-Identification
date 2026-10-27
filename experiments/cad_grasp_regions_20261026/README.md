# Compact CAD grasp generation before VLM selection

This revision generates a few distinct finger placements from an already aligned
CAD model. A **grasp pose** is the position and orientation of the hand, together
with the finger opening. A **contact family** groups checked hand alternatives
that use the same contact region or an equivalent final grip. It is a review
option, not a quality score or the VLM's chosen grasp.

The user's selected rule is: **approach straight down; use CAD closing axes that
are approximately horizontal.** The implementation uses a declared 1-degree
tolerance. It does not tilt the hand to follow a tilted CAD axis.

[Final gallery](../../outputs/cad_grasp_regions_20261026/runs/20261026T184853Z/gallery/index.html)
and [comparison report](../../outputs/cad_grasp_regions_20261026/runs/20261026T184853Z/report.md).
The viewer includes native CAD X/Y/Z arrows, a separate black finger-closing
arrow, every contact family and every valid checked hand alternative.

## What changed and what the test found

The previous sampler expanded many nearby contact pairs into two jaw assignments
and six depth labels. Its grouping treated different depth labels and different
contact witnesses as distinct even when they described essentially the same
hand placement. The new generator forms contact regions **before constructing
full hand poses**, calculates one finger height per region, and checks both jaw
assignments. The two assignments exchange which physical finger touches each
side; the complete Panda hand is not exactly symmetric, so both are checked.

| Measure, across the same 96 placements | Previous | Compact revision |
|---|---:|---:|
| Full hand proposals checked | 401,424 | 414 |
| Passing full hand proposals | 75,622 | 347 |
| Displayed groups/contact families | 16,506 | 175 |
| Placements with at least one passing option | 96/96 | 88/96 |
| Median generation/check time per placement | 14.967 s | 0.121 s |
| KET16 options per scene | 42--488 | 2 |
| Circular-pin options per scene | 2--880 across three sizes | 1--2 |

These are the existing N00--N07 scenes with twelve parts per scene, not the older
single scene with fourteen parts. Both runs use four CPU workers, the same saved
observed-depth CAD fits, and the same hand/target/table checker. Correct CAD/crop
identity is supplied. Simulator object poses are not used. Times exclude input
pose estimation, imports, gallery rendering and VLM selection. The old result
also omits some final result/report writing. The old USB times accumulate the
initial search and declared exploratory follow-up; the new method has lower
coverage. This comparison is not an equal-output speed benchmark.

The compact generator still makes 130,359 cheap horizontal CAD ray queries to
find regions. It does not construct that many hand poses. It finds 939 raw
regions, merges them into 524 pad-size regions, and prepares one placement for
207 regions. Two jaw assignments per prepared placement give 414 full checks.
The output is not padded or capped at twelve, five or another target number.

The large gear has no option in N01/N03/N05/N07; the BNC connector has no option
in N01/N03/N04/N06. In all eight cases the selected region placements fail
open-hand approach clearance against the target during preparation. The report
gives concrete examples. The strict direction rule and the single centered
representative both restrict this search. A missing option is not proof that
every CAD-axis grasp is impossible. No oblique fallback was silently added.

The final independent audit passes 20,123 checks with zero failures. Sixteen
focused tests pass. Minimum retained whole-hand table clearance is 0.120 mm;
maximum difference between the two fingers' first-contact openings is 0.044434 mm
against the unchanged 0.050 mm limit. These are model calculations. No pickup,
assembly, VLM call or force/slip test was performed.

## How the compact generator works

All lengths in the code are meters. World Z points up. The hand's local X is the
finger-closing direction; its local Z is the approach direction.

### 1. Use a small, explicit set of directions

Let `R_WC` rotate native CAD coordinates into world coordinates, `e_i` be one CAD
axis, and `u = (0,0,1)` be world up. Its world direction and departure from the
horizontal plane are

\[
a_i=R_{WC}e_i,\qquad
\theta_i=\arcsin(|u^T a_i|).
\]

Use the axis only if `theta_i <= 1 degree`. Project it into the horizontal plane:

\[
x_i=\frac{(I-uu^T)a_i}{\|(I-uu^T)a_i\|},\qquad
a=-u,\qquad R_{WG}=[x_i,\;a\times x_i,\;a].
\]

The opposite jaw assignment uses `-x_i`. Both approaches remain exactly down.
The saved result and viewer report `theta_i`; projection is not described as
exact native-axis alignment. One degree is our local engineering setting, not
a published optimum or a measured pose-estimation uncertainty.

### 2. Find opposing surface regions, rather than thousands of hand poses

For each allowed closing direction, cast a horizontal ray from each side of the
native CAD mesh on a lateral/height grid with spacing no larger than 1 mm. Both
rays must hit. If their contact separation is `w`, require

\[
1\,\mu\mathrm m < w \leq 67\,\mathrm{mm},\quad
|n_-^T x_i|, |n_+^T x_i|\geq\cos15^\circ,\quad
n_-^T n_+\leq-\cos30^\circ.
\]

The absolute direction tests do not require a fixed sign convention; the two
normals must still oppose each other. Inconsistent relative mesh winding can
therefore still cause rejection. These are geometric screens,
not a friction or force-closure guarantee. The full hand checker follows.

The 67 mm maximum comes from the actual modeled gripper: maximum commanded
opening is 80 mm, the inner-pad gap is 1 mm smaller, and the approach requires
12 mm additional total gap. Therefore `w + 1 mm + 12 mm <= 80 mm`. A nonpositive
or numerically degenerate separation is discarded.

Neighboring grid cells join a region if their width changes by at most 2 mm,
their midpoint along the closing direction changes by at most 1 mm, and each
corresponding surface normal changes by at most 30 degrees. Connectivity is
through the four lateral/vertical grid neighbors. Occupied cell edges are
sampling bounds, not exact CAD boundaries.

Nearby fragments merge when their lateral/vertical bounding gaps are within
the half-size of the modeled pad footprint, their median widths differ by at
most 2 mm, and their median closing-direction midpoints differ by at most 1 mm.
For these pads the two half-sizes are 8 mm. This is a connected-component merge,
not a proof that a whole region can be supported by one pad. Later checking
determines whether its representative fits.

### 3. Calculate one height for each region

Start at the mean lateral/height coordinates of the region. Recast the two CAD
contacts there. If the mean falls in a hole, use the qualified grid cell nearest
the mean, rather than the first cell whose hand happens to pass.

The inner pads span local Z `[-11.6, +4.4] mm`, so their center is `-3.6 mm`.
For world contact height `z_c`, the preferred hand-reference height is

\[
h_0=z_c-3.6\,\mathrm{mm}.
\]

Raise this height as needed for the observed table plane. For plane
`n^T p + d = 0`, upward `n_z > 0`, hand point `q`, horizontal translation `t_xy`,
and required table clearance `c`, the lower bound is

\[
h\geq\max_q
\frac{c-n^T(Rq+t_{xy})-d}{n_z}.
\]

The maximum includes every hand component, both jaw assignments, and commanded
openings of zero and this candidate's approach opening. Add a 0.020 mm numerical reserve. The nominal clearance
remains 0.100 mm. The target-clearance calculation then finds a clear approach
height, if present in the contact-compatible interval, to a 0.010 mm bracket.
It uses the same modeled downward sweep as final validation, not simulated
pickup attempts. When height changes, recast the contacts within the overlap
of the region and the finite pads and update the opening. Three geometry
updates are allowed. If this process fails, record a preparation failure.

This is our deterministic representative policy. It is not a proof of the
globally best height, nor an exhaustive search for every feasible height.
The height projection relies on upward clearance within this bounded, low
task-board setup; it is not claimed as a general search for overhangs or cages.

### 4. Validate every proposed hand and remove physical duplicates

Use the unchanged `cad_grasp_geometry_20261026` checker for the open hand, full
100 mm approach, nominal pad placement, first contact while closing, finger and
palm overlap, and full observed table plane. Target holes stay present. No robot
execution or other-part/robot-self collision test is added.

Two final grips are duplicates when hand translations differ by at most
1 micrometer, hand rotation differs by at most `1e-6` radians, and the actual
first-contact openings differ by at most 1 micrometer. Sampling witness
coordinates, depth labels and different already-checked starting openings do
not make a new final grip. All valid checked alternatives are preserved in the family.

For the default display only, prefer the positive CAD closing direction when
valid, then the smallest already-checked starting opening. This is a canonical
display convention, not an assembly-quality score. The viewer lets the user
inspect every valid alternative. No final winner is selected here; later VLM
selection remains part of the intended pipeline.

## Which ideas are published and which are ours?

[Wan, Harada and Kanehiro, *Planning Grasps for Assembly Tasks*](https://arxiv.org/html/1903.01631v1)
removes contacts near surface-region boundaries and screens nearby contacts
using a radius related to finger-pad size (§III-B2). It then plans parallel-finger
grasps from opposing surface regions and applies a stability model. This supports
reducing redundant contacts before expanding hand poses. It does not prescribe
our CAD-axis restriction, ray grid, region-merge thresholds or single-height rule.
Our implementation is not a reproduction of its superimposed surface segmentation
or soft-finger stability test.

[GraspIt!'s official planning manual](https://www.cs.columbia.edu/~cmatei/graspit/html-manual/graspit-manual_9.html)
also describes controlling sampling density and sorting tested grasps by a chosen
quality measure. Thus score-based ranking is established, but no single score or
universal shortlist length is required. The documented primitive planner has
hand/model restrictions; it is precedent, not our drop-in implementation.

One established analytic score is Ferrari--Canny's worst-direction force/torque
resistance, often written

\[
Q_\epsilon(g)=\min_{w\in\partial\mathcal W(g)}\|w\|,
\]

when the origin is inside the achievable force/torque region `W(g)`, and zero
otherwise. Its meaning depends on contact/friction assumptions, force limits and
force/torque scaling. It is a holding-robustness measure, not assembly precision.
See the [original paper](https://people.eecs.berkeley.edu/~jfc/papers/92/FCicra92.pdf).
This revision does not invent those physical inputs or add that score.

The exact mathematical decision here is **allowed direction + qualified region
+ modeled hand feasibility + physical duplicate removal**. The grid and merge
constants are our explicit local settings. This is a training-free adaptation
of established geometric principles, not a newly established universal grasp
quality equation. Region centering is a heuristic and may discard a better
off-center grasp.

## Alignment does not guarantee that the part stays still

The earlier KET16 audit found 99.18% of its retained hand directions already
within 1 degree of a native CAD axis. Much of its excess was repeated positions,
depth labels and contact witnesses. A line between two contact witnesses is not
necessarily the finger-closing axis; the revised viewer shows both separately.

CAD-axis alignment can help produce understandable, repeatable placements on
flat-sided parts. But CAD coordinate axes can be arbitrary, particularly around
a circular shaft, and a hex nut's opposite flat faces need not match all three
native axes. The cited assembly paper uses actual surface directions and reports
reduced rather than eliminated closure movement. An oblique stable grip does
not inherently move the part, and an axis-aligned grip can still slide or roll.

If a rigid part truly stays fixed in the hand, its relative transform is

\[
T_{GP}=(T_{WG})^{-1}T_{WP},\qquad
T_{WG}^{\mathrm{goal}}=T_{WP}^{\mathrm{goal}}(T_{GP})^{-1}.
\]

Here `P` is the part, `G` the hand and `W` the world. These equations apply to
any grasp orientation. Correctness depends on knowing the **actual** held-part
transform; initial CAD alignment alone does not measure a shift during closing.
The present test establishes neither zero shift nor precise assembly.

## Run and evidence

Use the existing WSL environment; do not install packages or download weights:

```text
python -m pytest experiments/cad_grasp_regions_20261026 -q
python -m experiments.cad_grasp_regions_20261026.run
python -m experiments.cad_grasp_regions_20261026.verify_results RUN_PATH
python -m experiments.cad_grasp_regions_20261026.analyze RUN_PATH
python -m experiments.cad_grasp_regions_20261026.gallery RUN_PATH
```

`run.py`, `regions.py` and their hashes define generation. The audit checks all
saved alternatives, frames, counts, source preservation, native-triangle contact
evidence and the unchanged checker limits. The gallery records its own renderer
hash after generation. Old outputs and CAD prompt files are preserved.

The development run `184044Z` retains its failed duplicate audit. The corrected
final run is `184853Z`; its audit passes. It should not be represented as a
successful pickup or assembly result.
