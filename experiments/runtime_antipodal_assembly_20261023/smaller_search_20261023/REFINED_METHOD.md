# Refinement: downward CAD grasps with modeled contact validation

Implementation follow-up: the user subsequently authorized the
[eight-scene pre-VLM geometry trial](../../cad_grasp_geometry_20261026/README.md).
Its checker and exploratory CAD surface-coverage variant are implemented in
that separate experiment. This note preserves the preceding design discussion;
its original statement that no eight-scene run was authorized is historical.

Design revision after user clarification on 2026-10-26. This is a proposed
method specification, not a completed implementation or eight-scene experiment.
The saved smaller-search results and their [historical mathematics](MATH.md)
remain unchanged.

## Current decisions

- Assume CAD identification/alignment and final assembly targets are supplied.
- Use the actual simulated Panda hand/fingers; physical RG2 geometry is a later
  explicit change, not an interchangeable finger model.
- Restrict pickup approach to workspace down. Retain rotation in the table plane
  to align the fingers with the part. Do not add tilted approaches as a fallback.
- Use CAD and modeled fingers to validate placement, approach and nominal closing.
  Do not execute a pickup or step contact physics to decide candidate validity.
- Check the target part and table together. Other-part collision, arm/self
  collision, reachability and robot path planning remain outside this study.
- Remove near-duplicate configurations without filling a target shortlist size.
  Keep meaningful finger-depth alternatives. There is no requirement to reach 12.
- Validate every distinct proposed configuration before VLM review. Do not select
  the first passing configuration. The VLM makes the final choice using task context.
- Refine this specification before testing the eight scenes. No eight-scene run,
  new weights, paid inference or physical robot connection is authorized by this note.

## What the existing sampler actually qualifies

The old ray sampler first casts 512 directions around the aligned CAD to obtain
surface hits and their surface directions. For each hit p and direction n, it
casts back toward p from the other side along n, using two starting distances.
The other intersection supplies candidate point q. The active path uses no
random-direction perturbation, despite that option's configuration description.

Let w=||q-p|| and u=(q-p)/w, with outward CAD surface normals n_p,n_q. The old
checks require finite points, strictly 0.002 < w < 0.075 m, an upstream hit-normal
angle strictly below 15 degrees, and these additional local tests:

\[
|n_p^T u|\ge\cos15^\circ,\quad
|n_q^T u|\ge\cos15^\circ,\quad
n_p^T n_q\le-\cos30^\circ.
\]

Nonfinite pairs, unsuitable separation or surface directions fail. Repeated
endpoints are canonicalized and removed on a 0.01 mm coordinate grid. P is the
number left after these tests; it is not a requested number of successful grasps.
These are point-pair qualifications. They do not test finite pad support,
gripper-body clearance or an unobstructed approach to the part.

The 2/75 mm and 15/30 degree values are settings of the old local trial, not
universal physical limits. Refined width limits must come from the actual hand.

## Fixed downward approach and the meaning of 0/180 degrees

Let table normal n_T=(0,0,1) and hand approach a=-n_T. An approach from above is
perpendicular to the table. The finger-contact normals on side surfaces are
usually approximately horizontal. These are different directions.

For a horizontal unit closing direction x=(cos(psi),sin(psi),0), use:

\[
R(\psi)=[x\ \ a\times x\ \ a],\qquad Re_3=a.
\]

The hand retains one orientation variable, yaw psi in the table plane. Replacing
psi by psi+pi exchanges jaws while the approach remains down. A part turned
35 degrees in the workspace can therefore require hand yaw 35/215 degrees;
restricting absolute yaw to 0/180 degrees would miss those contacts.

The old roll variable rotated around the contact line. Its 180-degree value
usually turned a downward approach upward, so simply keeping old roll values
0/180 is not the proposed restriction.

Pairs and actual pad geometry must be compatible with horizontal finger closing.
One implementation option is sampling CAD cross-sections at contact heights.
Do not merely project a tilted contact line onto the table and assume the old
contact points still fit. Recompute compatible contacts or reject the pair,
and locate the hand using actual pad positions and tool offsets.

Primary precedent: [Dex-Net 2.0, Figure 3 and sections III-B/IV-A1](https://arxiv.org/html/1703.09312v3)
uses table-perpendicular grasps with a table-plane angle. [GR-ConvNet, section III](https://arxiv.org/html/1909.04810v3)
represents orientation as rotation around Z. These support a reduced grasp domain;
they do not prove that downward grasping is universally best for precise assembly.

## Which parts of the old score were our design

The weighted-sum structure and rotation-angle distance are common mathematical
tools. This exact old score and its constants were written for our local trial:

\[
J(g)=\|m-c\|/0.05+0.25\,\theta(R,R_0)+0.5(1+a_z).
\]

R_0 is the robot gripper's starting orientation. Its turn term measures hand
rotation from that starting orientation to the pickup pose, not robot joint
travel or the rotation needed to assemble the part. Changing the starting hand
orientation can change this preference without changing the object or task.
c was the CAD frame origin, not a measured centre of mass.

With a_z=-1 for every proposed grasp, the approach term equals zero for all
candidates and must be removed. Neither the initial-hand term nor the arbitrary
CAD-origin distance is adopted as a task-quality score in the refined method.
Keep the old score only as a clearly labeled historical comparison if needed.

For a fixed relative grasp R_OG, initial/final hand orientations obey
R_G0=R_O0 R_OG and R_G*=R_O* R_OG. Hence:

\[
R_{G*}R_{G0}^T=R_{O*}R_{O0}^T.
\]

The rigid pickup-to-goal rotation angle itself is candidate-independent. Do not
invent a candidate-dependent assembly-turn objective from the known final pose.
The final pose remains task context and provides the nominal assembly hand pose.

## One model-only validity checker

Let O be the target CAD solid and H(g,w) the actual hand at pose g and opening w.
Partition the hand into intended inner contact pads and other finger/palm surfaces.
The CAD remains fixed during this geometric calculation.

1. Check physical width limits and proposed contacts against the finite pads.
2. At the intended open width, check the complete hand against target and table.
   If a straight downward approach is part of the proposal, check its swept
   geometry too. A clear final pose alone does not establish a clear approach.
3. Move hypothetical finger geometry through its known closing motion. Locate
   the first contact of each finger and its opening width. Do not continue
   pushing a modeled finger through the fixed CAD to make the other finger touch.
4. Require first contacts on the intended opposing inner pads, near the proposed
   regions, at compatible widths. Check for unintended earlier contacts elsewhere.
5. Permit intended pad-to-part touching within declared numerical tolerances.
   Reject excessive penetration, palm/finger-back interference and table overlap.

For each pad j, denote its first-contact opening by w_j*. A symmetric-jaw
compatibility condition is |w_L*-w_R*| <= epsilon_w. The tolerance must be declared
before testing; it is not inferred from which grasps succeed. Failure indicates
asymmetric modeled contact, not a prediction of how the real part would settle.

For a straight open-hand approach g_s=(R,t-sa), 0<=s<=L, require a positive
declared open-hand clearance epsilon_open from the CAD solid:

\[
\operatorname{dist}(H(g_s,w_o),O)\ge\epsilon_{\rm open}>0
\quad\text{for every }s.
\]

For a horizontal table and mesh/box vertex set V(w_o), the open-hand plane test is:

\[
\min_{v\in V(w_o)}(Rv+t)_z-z_T+\min(0,-La_z)\ge\epsilon_T.
\]

This excludes both unintended touching and solid overlap before closing; intended
pad contact is allowed at the modeled closing endpoint instead.

The table check also applies to modeled closing configurations; asymmetric or
non-translating mechanisms cannot be assumed to preserve their lowest point.
For our fixed downward approach the open-hand minimum is at the final pose.
The old 0.1 mm plane margin may be retained as a declared baseline setting; it
is not an established physical safety or assembly-accuracy tolerance.

Use signed-distance/solid-overlap or equivalent collision geometry that handles
containment, not only triangle crossings. Keep native holes and concavities;
do not replace an entire target by a hull that closes its CAD openings. Numerical
checking must state mesh resolution, tolerance and whether the sweep is continuous
or sampled. Sampled checks alone are not a continuous clearance certificate.

Define combined validity as:

\[
V(g)=C_{\rm width}(g)\land C_{\rm target/approach}(g)
\land C_{\rm closing/contact}(g)\land C_{\rm table}(g).
\]

This checks modeled contact/placement feasibility. It does not simulate forces,
predict slipping or settling, or establish pickup success/assembly precision.

Primary implementation precedent: [NVLabs GraspDataGen, geometric grasp guesses](https://github.com/NVlabs/GraspDataGen/blob/main/docs/components/grasp-guess.md)
needs a triangle mesh and tests modeled finger placement/openings; its physics
validation is a separate stage. [Wan's parallel-jaw planner, section IV-B](https://arxiv.org/html/1903.01631v1#S4.SS2)
also checks hand geometry against the target before execution. These references
support geometric validation; they do not establish our proposed checker as implemented.

## Variable-size grouped set and VLM final selection

\[
\mathcal G_{\rm valid}=\{g\in\mathcal G_{\downarrow}:V(g)=1\},
\qquad \mathcal C=\operatorname{deduplicate}(\mathcal G_{\rm valid}).
\]

Grouping/deduplication must consider actual contact locations, yaw, hand position,
opening and finger depth. Define position/angle/depth tolerances before the run.
Do not discard a meaningful depth change merely because its contact pair is the
same. Merge exchanged jaws only when the actual geometry is equivalent, keeping
their orientation aliases in the record. No target size of 12, second-pass padding
to twelve, or first-twelve-valid rule is used.

Sampling remains finite at declared surface/contact-height resolution. Replace
the arbitrary every-Nth-entry 24-pair truncation with geometric spacing/coverage
at the chosen resolution. If a computation limit is necessary, state it as a
limit and report truncation; do not describe an incomplete list as exhaustive.

Validate all distinct proposed configurations, record every rejection reason,
then give all retained representatives and stable IDs to the VLM. If the set is
large, compare all candidates in declared batches without dropping unseen ones.
The task description and supplied final pose are context. Preserve the selected
ID and raw explanation; no hidden numeric reranking or first-passing fallback.
Zero candidates is explicitly unavailable; one candidate provides no ranking
comparison. A few candidates remain a few; do not manufacture extra alternatives.

## Historical overlap evidence and later scene mapping

Of 156 old shortlisted poses, 16 had numerical pickup-path checks: 14 passed,
two failed, and 140 were not checked. Large-gear G01 overlapped the target by
up to 4.737912 mm; medium-gear G01 by 0.121301 mm. Both exceeded the old 0.050 mm
rejection threshold during modeled lowering with open fingers. G02 then passed
for both parts. The checks were numerical, not failed physical pickup attempts.
The new refinement moves target/table/contact validity before final selection
and removes stopping validation at the first passing pose.

Existing N00-N07 are twelve-part classification layouts, not eight repetitions
of the old fourteen-part pickup scene. Their inventory excludes medium/small
gear and M12 nut, and adds BNC. Before the later experiment, explicitly map scene
IDs to actual parts and supplied estimated poses. Do not silently report 8x14
tests or use simulator object state for runtime pose updates.

No after-execution position/angle precision measurement is required. Later
evaluation can report available candidates, VLM choices, observable pickup/hold
and assembly outcomes when attempted, and time. It must distinguish those results.
