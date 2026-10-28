# Fabrica-derived CAD grasp candidates

This trial generates candidates only for the frozen eight-scene dataset: twelve
parts in each scene, or 96 placements. It does not run VLM selection, pickup,
assembly, physics stepping, or new pose estimation. The VLM implementation and
prompt are protected by before/after hashes. No final assembly views are made.

## What comes from Fabrica

The source is [Fabrica](https://fabrica.csail.mit.edu/), with the contact code
vendored from [commit 215a30f](https://github.com/yunshengtian/Fabrica/blob/215a30fe51e59299588a5b5e417a9cb934fb393e/planning/robot/util_grasp.py).
The installed environment has no `trimesh` or FCL. Consequently, this trial is a
NumPy port of its contact construction, followed by our existing finite-gripper
checker. It is not execution of the complete original Fabrica system.

1. Sample a point `p1` on the native CAD surface, in proportion to triangle area.
2. With outward unit normal `n1`, start a ray at `p1 - 0.0005*n1` metres and cast
   in direction `-n1`. Test every forward intersection `p2`.
3. Retain an intersection when its outward normal satisfies
   `n2 dot (-n1) > 0.95`, and the separation is at most 76 mm. The angular
   threshold is about 18.2 degrees; this proposal test alone does not establish
   flat finger contact.
4. Execute Fabrica's original nearby-pair removal function with its 0.5 mm
   spacing, then order the remaining pairs by the distance of their midpoint to
   the area-weighted CAD surface center.

We use 100 sampled surface points per batch, stop after a complete batch reaches
100 accepted raw pairs, and allow at most eight batches. The raw count can exceed
100. These are bounded trial settings, not a desired number of good grasps.
Sampling is done once per CAD model and reused across the eight placements.

The original triangle sampler and ray library are replaced; coincident ray hits
are merged within 1 nanometre. Fabrica's 10 mm ground gate and clearance spheres
are replaced by checks using the actual saved finger, pad and palm geometry.

## How a contact pair becomes a gripper pose

All distances are in metres. `world_T_cad` is the saved observed CAD alignment.
`world_T_grip` maps the saved Panda `gripper0_grip_site` frame into world
coordinates. Local X closes the fingers; local Z points downward during approach.
No simulator object pose is read to update the alignment.

A pair whose line is within 15 degrees of horizontal proposes a horizontal closing
direction. The gripper approaches straight down. The actual finger contact must
then pass the tighter existing 1 degree surface-normal check. Projecting the
proposal does not bypass that check.

Three height variants put the sampled pair at 25%, 50% and 75% of the usable pad
height. The middle variant is checked first. Table clearance can raise a pose;
identical heights after this adjustment are checked once. Both finger assignments
are checked. These depth variants are our adaptation, not a native Fabrica feature.

The existing CAD checker holds the part fixed and computes first contact while
the modeled fingers close. It requires:

- Both actual contacts on the inner pad faces, with surface normals within
  1 degree of the closing direction.
- The two first-contact gripper openings within 50 micrometres of each other.
- The existing 0.25 mm pad-edge margin and 0.1 mm target/table clearance.
- No earlier contact by the finger bodies or palm, and a clear straight approach.

These tests can reject a sampled opposing pair. They do not predict friction,
closing force, part motion, slip, or assembly accuracy. Well-aligned cylindrical
side contact is allowed; the CAD faces need not be planar.

## Groups in the gallery

Every passing pose is saved under `all_valid_grasps`; every checked pose is also
saved in its case's `candidate_checks.json`. Nothing is chosen for execution.

The gallery groups nearby passing poses to make them easier to browse. A new pose
joins an existing representative only when closing axes differ by at most
10 degrees (allowing reversed fingers), horizontal center and both sampled
endpoints differ by at most half a pad's width, reference heights differ by at
most one eighth of the pad height, and contact openings differ by at most 2 mm.
These are local display rules. They do not prove mechanical equivalence or
identify every cylinder symmetry. There is no fixed count such as twelve.

The first passing member in the saved proposal order represents each display
group. This is not a quality score or VLM selection. All group members remain
available for future selection, including the depth and reversed-finger variants.
Top and side CAD renders show the original pair, actual modeled contacts, fingers
and pads. Virtual CAD views do not require additional physical cameras.
The green points are geometric contact witnesses. For face-to-face contact, they
do not identify the contact-force center or a unique physical contact point.

## Reading the result

A useful first sign is that valid candidates exist across the saved placements,
include the required side-grip and height options, and remain manageable to review.
Report placements with no passing candidate and the modeled rejection reasons.
Candidate availability is not pickup success or insertion success. This run cannot
establish that Fabrica is better than another method or that VLM selection helps.

The source budget, depth variants and display grouping differ from older runs.
Their counts and timings are not a controlled method comparison. Sampling time
is recorded once per CAD; validation task times are recorded separately in a
four-worker run. Rendering time is separate from generation.

From the repository root, using the existing environment:

```powershell
wsl -- /home/jiabao/.venvs/lerobot-libero/bin/python -B -m pytest -q -p no:cacheprovider experiments/fabrica_candidates_20261027/test_sampler.py
wsl -- /home/jiabao/.venvs/lerobot-libero/bin/python -B -m experiments.fabrica_candidates_20261027.run_trial
wsl -- /home/jiabao/.venvs/lerobot-libero/bin/python -B -m experiments.fabrica_candidates_20261027.gallery RUN_DIRECTORY
```

Use a new output directory for each trial. Preserve prior candidate evidence and
the input alignments. Do not start the VLM or physical/simulated robot execution
as part of this command sequence.
