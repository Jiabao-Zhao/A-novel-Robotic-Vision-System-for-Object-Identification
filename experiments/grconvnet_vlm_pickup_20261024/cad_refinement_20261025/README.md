# CAD-assisted GR-ConvNet pickup retry

Authorized 2026-10-25: use aligned CAD geometry to improve GR-ConvNet height and
finger placement, retry the large gear, medium gear and M16 nut, and compare
pickup success and time. No assembly views, assembly goals or assembly movement
are used. This is a new controlled follow-up; the previous 14-part records stay
unchanged.

## Completed result

The [new three-part report](../../../outputs/grconvnet_vlm_pickup_20261024/cad_refinement_20261025/runs/20261025T192913Z/report.md)
and [pickup gallery](../../../outputs/grconvnet_vlm_pickup_20261024/cad_refinement_20261025/runs/20261025T192913Z/gallery/index.html)
contain all 15 condition records, actual pickup frames, exact VLM inputs and
part-in-gripper measurements. Every executed pickup passed the same 180 mm
commanded lift and one-second hold check.

| Method | Parts picked up and held | Actual pickup attempts | Decision time, all three | Total recorded time, all three |
| --- | ---: | ---: | ---: | ---: |
| Geometric small search | 3/3 | 3 | 9.84 s | 46.50 s |
| Original GR-ConvNet, score order | 0/3 | 0 | 8.57 s | 17.87 s |
| Original GR-ConvNet, VLM order | 0/3 | 0 | 26.47 s | 35.22 s |
| CAD-refined GR-ConvNet, score order | 3/3 | 3 | 9.52 s | 46.99 s |
| CAD-refined GR-ConvNet, VLM order | 3/3 | 3 | 15.38 s | 59.07 s |

The original conversion again stopped before movement: large/medium gear
approaches would intersect the target, and M16's fingers would extend below the
table. Its shorter totals include zero physical pickup attempts and are not
successful-pickup speed measurements.

CAD refinement rescued all three using unchanged network weights and jaw
directions. It shifted contact centres by 3.206 mm, 3.480 mm and 6.376 mm in XY,
respectively. The large gear uses its raised upper section at 15.005 mm contact
height; medium gear and M16 use CAD middle heights of 10.008 mm and 6.509 mm.
Only one refined candidate survived per part, so VLM review had no alternatives
to rank and did not change the chosen grasps or pickup outcomes.

The common VLM preload took 12.06 s outside the timed conditions. Despite that
attempt, the first original-GR VLM request reported a further 10.51 s model load;
this remains included in its decision and total time. The other four calls
reported 0.004–0.007 s model loads. The manifest records the intended warm-model
control; the actual first-call reload means it was not fully achieved. No
recorded times have been adjusted or replaced. This does not affect the direct
CAD score-order versus geometric comparison, which has three successful pickups
in each condition. Differences from the older 14-part timings are not a speed
comparison across days or runs.

Four focused geometry tests passed before the run. The [held-part pose records](../../../outputs/grconvnet_vlm_pickup_20261024/cad_refinement_20261025/runs/20261025T192913Z/held_part_poses.json)
save actual positions/orientations after closure and after lifting, including
movement relative to the gripper. These use simulator poses for audit; they do
not claim an implemented held-part vision tracker or successful assembly.
Closing shifted the medium gear's CAD origin by 2.563 mm and turned M16 by
4.556 degrees relative to the gripper. Their subsequent changes from closure to
the end of the lift/hold were 0.206 mm / 1.244 degrees and 0.067 mm / 0.057 degrees.
The initial planned relation alone therefore does not measure the current held
part accurately enough to assume precise assembly alignment.
The [independent audit](../../../outputs/grconvnet_vlm_pickup_20261024/cad_refinement_20261025/runs/20261025T192913Z/independent_audit.json)
passes all 15 condition records and verifies 1,191 protected files unchanged.
Only these three parts were retried; this is not a fresh all-14-parts result.

## Before-run protocol

Use the same prepared older `objects_centered` scene, RGB-D camera, visible
target mask, frozen Cornell RGB-D network, local Qwen3.5-9B, grasp-ranking prompt,
Panda geometry and contact controller. Other loose parts remain visible with
collisions disabled. Robot/table checks and open-finger/target checks remain.
The assumed correct CAD alignment is supplied from the simulator, as in the
earlier geometric control; the actual CAD pose estimator is not run here.

Run five fresh conditions on each of the three parts:

- Geometric small search, matching the previous control.
- Original GR-ConvNet conversion, network-score order.
- Original GR-ConvNet conversion, VLM order.
- CAD-assisted conversion, network-score order.
- CAD-assisted conversion, VLM order.

Every learned condition captures and infers afresh. The network still supplies
at most four image proposals. CAD refinement keeps each proposal's finger-closing
direction and permits only bounded nearby changes. It does not run the full
geometric sampler or use a saved part-specific grasp as a fallback.

The [refinement protocol](refinement.py) fixes local offsets, CAD height slices,
normal checks and the six existing finger-depth settings before pickup outcomes.
There are 15 CAD lines per network proposal: five offsets across its closing
line (0, -2, +2, -4, +4 mm) at the CAD's middle, quarter and three-quarter heights.
The final contact midpoint must stay within 10 mm of the image proposal; endpoint
surface normals must agree with the closing line within 30 degrees. Opening
margins are +12, +16 and +20 mm, limited to the same 80 mm gripper opening.
At most 270 depth/opening combinations are formed per proposal, with early exit
on the first geometry-feasible choice. These internal checks produce at most
one displayed pose per original network proposal.
Outer CAD ray hits provide contact points and spacing. The complete open gripper
must clear the target and table in static geometry checks. Only one pose per
network proposal is retained before VLM ranking. The VLM sees the exact corrected
pose and opening it ranks; it never chooses unseen depth variants. Candidates
then undergo the same robot-path checks and at most one physical pickup attempt.

A pass requires at least 120 mm rise after a one-second hold, both fingers still
touching, and robot overlap with itself, the table or fixed board hardware below
1 mm. The lift command is 180 mm. There are no attachment constraints or target
pose corrections in the motion controller. One observation per part/condition
tests these known starting arrangements, not general performance.

Decision time includes capture, generation/refinement, VLM review when used,
evidence preparation and path checks. Total recorded time also includes setup,
actual contact simulation and saved evidence. A rejected case without movement
remains in the three-part denominator. One-time network loading and one common
VLM preload are reported separately; the timed VLM calls use the same warmed
model. Any residual loading remains included in its call. Five conditions run serially
with rotating order. Exact source hashes and constants are frozen in the run
manifest before trials.

## Knowing how the part sits in the gripper

Define `world_T_cad` as the CAD coordinate frame's position and orientation in
the workspace, and `world_T_grip` as the actual gripper coordinate frame's
position and orientation in the same workspace. Our gripper frame is
`gripper0_grip_site`, not the robot flange or finger tip. All translations are
metres. Then:

```
grip_T_cad = inverse(world_T_grip) @ world_T_cad
world_T_grip_goal = world_T_cad_goal @ inverse(grip_T_cad)
```

The first expression gives the part's position and orientation relative to the
gripper. The second gives the gripper pose that would put the part at a known
CAD destination, provided the part keeps the same relation to the gripper.
The CAD destination must describe the CAD frame, including the offset from its
origin to the actual mating feature. The same rule follows from standard
[coordinate-frame composition](https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-3-1-homogeneous-transformation-matrices/).

Before closure, this is a planned relation. Finger closure can shift or turn the
part; later slip changes it again. Initial CAD pose plus robot position alone
does not measure those changes. A real system should reobserve the held part or
validate the rigid-hold assumption to the needed assembly tolerance. Camera,
robot and gripper-frame calibration errors also propagate. A visually symmetric
part can leave an orientation unobservable; an assembly key or tooth may require
additional evidence.

This retry records planned, end-of-descent, post-closure, post-lift and held-part
relations. Simulator poses used for those measurements are audit-only, not an
implemented held-part vision tracker. Pickup success alone does not verify
assembly alignment or insertion.
