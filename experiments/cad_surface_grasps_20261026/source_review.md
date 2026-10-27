# Source review for surface-based CAD grasp generation

Inspected October 26, 2026. This is a code and method review, not a claim that
the external frameworks were installed or executed. Exact upstream revisions
and file URLs are in [source_manifest.json](source_manifest.json). No full
upstream implementation is copied here.

## Recommended reference: Wan's current `one` antipodal planner

[`one/grasp/antipodal.py`](https://github.com/wanweiwei07/one/blob/ec1bdae83ec33ca6af9de42904d9851647fe012c/one/grasp/antipodal.py)
is MIT licensed. `_antipodal_candidates` samples actual mesh surfaces, shoots
inward rays along their normals and finds opposing hits. `antipodal_iter`
checks opposing-normal agreement and gripper width, constructs the hand frame,
and tests the grasp and open pre-grasp against the object. Neither operation
selects native CAD X/Y/Z directions.

Its generic implementation expands roll angles and uses a weighted normal/
opening preference. `antipodal` returns up to fifty passing grasps by default.
Those are implementation choices, not assembly-quality guarantees. Our fixed
downward frame, contact-region reduction, cylinder handling, finite-pad checks,
continuous approach checks and table rules are local adaptations; they must not
be attributed to this code. A complete framework import is unnecessary.

The upstream normal test is the useful principle to retain:

\[
\frac{n_1^T(-n_2)}{\|n_1\|\|n_2\|}\geq\cos\delta.
\]

This requires the two surface normals to oppose within tolerance `delta`.
It does not by itself prove sufficient contact area, friction or holding force.

## Historical relation: WRS and the assembly paper

[WRS `antipodal.py`](https://github.com/wanweiwei07/wrs/blob/552ce472a2614e487dceda6930efcaae4877b97e/wrs/grasping/planning/antipodal.py)
is an older MIT implementation from the same author. The current WRS README
labels the project archived. `plan_contact_pairs` uses sampled surface normals,
inward rays and radius-neighbor suppression; `plan_gripper_grasps` builds and
checks hand configurations.

The associated research context,
[Wan, Harada and Kanehiro, *Planning Grasps for Assembly Tasks*](https://arxiv.org/html/1903.01631v1),
removes contacts near region boundaries and neighboring contacts using spacing
related to finger-pad size. Its planner uses actual surface regions and includes
a soft-contact stability test. Our revision borrows the geometric reduction
principle, not its full segmentation or stability implementation.

Do not copy the older WRS default normal inequality without checking it:
`dot_thresh = -cos(angle_between_contact_normals)` at the default 160 degrees
becomes approximately +0.94, and the code accepts `n0 dot n1 < dot_thresh`.
That inequality does not enforce 160-degree opposition. The current `one` test
above has a clearer direct interpretation.

## Alternative reviewed: official EmbodiChain GraspKit

Official current source is Apache-2.0 licensed:

- [`antipodal_sampler.py`](https://github.com/DexForce/EmbodiChain/blob/81116eb3b4a005fd266cd093f6e4e1d8b5874ea6/embodichain/toolkits/graspkit/pg_grasp/antipodal_sampler.py)
  obtains mesh contacts and normals by ray casting and filters opposite contact
  separation and normal direction.
- [`_antipodal_backend.py`](https://github.com/DexForce/EmbodiChain/blob/81116eb3b4a005fd266cd093f6e4e1d8b5874ea6/embodichain/toolkits/graspkit/pg_grasp/_antipodal_backend.py)
  suppresses nearby poses before collision checking, then uses a weighted
  approach/opening/center-distance cost and retains a configured top set.
- [`pose_generator.py`](https://github.com/DexForce/EmbodiChain/blob/81116eb3b4a005fd266cd093f6e4e1d8b5874ea6/embodichain/toolkits/graspkit/pg_grasp/pose_generator.py)
  exposes the public mesh, object-pose, gripper and approach-direction interface.

Two inspected constraints need care: the sampler uses
`normal_angle < max_angle`, so setting `max_angle=0` rejects every hit in this
revision. The backend's frame projects the requested approach perpendicular to
the contact-pair direction; supplying world down does not guarantee a perfectly
downward hand when that pair is tilted. Our explicit frame constraints avoid
inheriting either behavior.

## Local geometry equations and terminology

These equations describe our proposed adaptation and basic geometry. They are
not a claim that upstream implements cylindrical symmetry reduction.

Let `u` be world up, `a=-u` the required approach and `x` the finger-closing
direction. Require

\[
\|x\|=1,\qquad x^T u=0,\qquad R_{WG}=[x,\;a\times x,\;a].
\]

For an opposing contact pair `p1,p2` with consistently outward normals `n1,n2`,
choose the sign of `x` so that

\[
x=\frac{p_2-p_1}{\|p_2-p_1\|},\qquad
(-n_1)^T x\geq\cos\epsilon,\qquad n_2^T x\geq\cos\epsilon.
\]

`epsilon` is a declared normal-alignment tolerance. The candidate must satisfy
both the horizontal requirement and these contact constraints. Merely projecting
an arbitrary direction horizontally and assuming it still matches both normals
is insufficient.

For **flat faces**, the inner pad plane should be parallel to its object face;
the two plane normals should align or oppose. It is the pad's normal, not the
pad surface plane, that follows the closing direction. Contacts away from face
edges and balanced pad placement are additional checks.

For a **cylindrical side**, a flat pad is tangent at contact, and its normal is
radial. Let `s` denote the physical cylinder shaft direction. If the shaft is not
vertical, the direction

\[
x=\frac{u\times s}{\|u\times s\|}
\]

is both horizontal and perpendicular to the shaft. This construction is
independent of arbitrary CAD rotation around the shaft. A vertical cylinder
leaves all horizontal radial directions equivalent for its round section; keep
one canonical direction or a few directions justified by other geometry.
A BNC's local round section does not make the entire connector rotationally
symmetric: adjoining features still require the full target checks.

An ideal fitted-cylinder normal and the original triangle normal are distinct
quantities. Record both residuals rather than calling a faceted mesh exactly
tangent. Neither surface alignment nor symmetry handling establishes zero
movement during closing or precise assembly.

## Attribution and integration boundary

The intended integration reuses existing native CAD ray helpers, the real Panda
pad model and the current target/table checker. It adapts source-backed
opposing-normal/raycast principles without importing a new robot framework,
training a model or reading simulator object poses. If upstream code is later
copied or substantially adapted, retain its applicable license and attribution.
Source citations alone do not mean we executed the original implementation.
