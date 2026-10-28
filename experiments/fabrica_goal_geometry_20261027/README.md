# Fabrica grasps filtered using nominal assembly geometry

This trial takes the **existing Fabrica candidates** and rejects grasps that
cover a specified mating surface or put a finger/palm into the receiving CAD
model at the final part position. It checks the fully seated configuration only.
It does not move the robot, close the fingers, or simulate assembly.

The input contains **12 parts in each of 8 scenes, N00–N07: 96 placements**. This
is the current twelve-part CAD-alignment study, not the older fourteen-part
pickup study. Every candidate keeps its original contact points, hand position,
orientation and holding opening. No new grasp is sampled, adjusted, ranked or
selected. VLM prompts/selection and all prior trial results remain unchanged.

**Insertion motion, finger opening/release, withdrawal, pickup and assembly
execution are paused and were not checked.** No pose fitting, training, model
download, physical robot connection or simulator-state read occurs here.

## Recorded result

Run: `outputs/fabrica_goal_geometry_20261027/runs/20261027T213516Z/`.

| Quantity | Result |
|---|---:|
| Original Fabrica hand poses checked | 129,170 |
| Remaining hand poses | 37,594 |
| Contact pairs supporting remaining poses | 15,093 |
| Placements with at least one remaining pose | 92 / 96 |
| Placements with both contact-region and seated-hand checks | 64 |
| Connector placements with seated-hand check only | 32 |
| Filtering, serialization and audits, four workers | 17.14 s |

The timing excludes gallery rendering and the earlier candidate generation. A
remaining pose is a geometric candidate, not a successful pickup or insertion.

| Part | Original poses, all 8 scenes | Remaining poses | Remaining contact pairs |
|---|---:|---:|---:|
| BNC Male | 5,961 | 3,146 | 1,176 |
| DSUB Male | 8,854 | 6,082 | 1,915 |
| Gear Large | 7,484 | 417 | 132 |
| KET12 Square 12 mm | 14,157 | 3,170 | 1,537 |
| KET16 Square 16 mm | 13,965 | 2,462 | 1,087 |
| KET8 Square 8 mm | 13,838 | 2,163 | 1,171 |
| M16 Hex Nut | 8,750 | 2,573 | 995 |
| RGOCG12-50 12 mm | 14,405 | 3,204 | 1,465 |
| RGOCG16-50 16 mm | 14,242 | 3,607 | 1,630 |
| RGOCG8-50 8 mm | 14,055 | 3,834 | 1,708 |
| USB Male | 6,009 | 4,243 | 1,385 |
| Waterproof Male | 7,450 | 2,693 | 892 |

Gear Large has no remaining pose in **N01, N03, N05 and N07**. In each of these
four cases, every saved hand pose intersects the board when transported to the
declared nominal goal. The original pose counts are respectively 878, 952,
1,002 and 904. This is a result for that frozen bank and goal; it does not prove
that no suitable gear grasp exists. The bank is not repaired or regenerated.

## Inputs, units and reference frames

- Original candidates:
  `outputs/fabrica_original_20261027/runs/20261027T193122Z/`.
- Saved observed-depth CAD alignments: the original candidates' `world_T_cad`
  fields, sourced from the frozen NIST alignment inputs. No engine object pose
  supplies the pickup alignment.
- Previously user-authorized nominal goals:
  `experiments/runtime_antipodal_assembly_20261023/goal_audit.json`.
- Local receiver CAD:
  `outputs/simulation/nist_task_board_1/assets/meshes/`.
- Static mounting conventions: `simulation/nist_task_board_1.py` and the saved
  CAD dimensions in `assets/provenance.json`.

All exported geometry uses metres. `task_inputs.py` uses a board frame with
XY=(0,0) at the plate centre and preserves the existing world Z direction and
height. It subtracts the saved legacy board centre `[-0.520, -0.275] m` from
the goal XY. It does not mix these goals with the simulator source's different
default board XY placement. Board bottom is 0.020 m; top is approximately
0.028991600 m.

The original pose is the **Fabrica Panda hand frame**, whose local Y is the
closing axis. It is not the earlier grip-site frame. Let O denote the part,
H the hand, W the saved pickup workspace, and B the board frame. For candidate g:

\[
{}^O T_H(g)=({}^W T_O^{pick})^{-1}{}^W T_H^{pick}(g),\qquad
{}^B T_H^*(g)={}^B T_O^*\,{}^O T_H(g).
\]

This transports the original hand around the part to the supplied goal without
changing the part-to-hand relationship. It assumes a rigid hold; it does not
predict part movement during closing.

## What is checked

### 1. Keep the specified mating-contact region available

For all six round/rectangular pins, the nominal target specifies 12 mm of the
lower end below the plate top. A pair is rejected if either contact lies at or
below the plate top after applying the goal transform. For the native centred
50 mm pin, that is approximately `CAD z <= -0.013 m`. The rest of its round side
stays available. A whole long triangle is not banned merely because part of it
extends into the reserved segment.

For Gear Large and M16 nut, the actual lowest downward-facing seating triangles
and central bore surfaces are reserved. The nearest native CAD face identifies
the contact surface. An inward-facing surface is identified as the central bore
only if a horizontal ray from the CAD Z axis reaches it first. This avoids
mislabeling a recessed outer gear surface as the bore. Contacts that cannot be
classified on an available surface are explicitly rejected as unclassified.

The **four connectors have no defined engaged mating region**. Their saved
targets place the male lower bounding plane against the female upper bounding
plane, with zero assigned engagement. All 32 connector placements therefore
carry `unavailable_zero_engagement` for this check. Their contact-region
eligibility flag merely lets the separate seated-hand check run; it is not a
successful mating-region assessment.

### 2. Keep the complete hand outside the local receiver at the goal

The saved holding opening is used exactly. Panda palm and both finger collision
meshes are transformed to the goal hand pose. The following receiver solids are
included, with their existing CAD mounting transforms:

| Target | Receiver CAD checked |
|---|---|
| Six round/rectangular pins | Board plate |
| Gear Large | Board, gear plate, assigned gear shaft 1 |
| M16 nut | Board and M16 screw |
| BNC, USB, Waterproof male | Board and matching female |
| DSUB male | Board, DB housing and DSUB female |

Other loose parts, neighboring fixtures, robot links and unrelated table space
are outside this trial. Receiver meshes retain their cavities and openings;
they are not replaced by the old simulator's convex collision hulls.

FCL checks triangle-boundary intersections. Closed-solid containment checks in
both directions also reject a hand component entirely inside a receiver, or a
receiver component entirely inside the hand. Exactly equal STL vertices are
welded without changing surfaces. The native right-finger reflection is baked
into its geometry so FCL receives proper rigid transforms; tests compare all
transformed hand vertices with the original native convention.

With H(g) the occupied hand volume at its saved opening and R the receiver
solids, the decision is the conjunction of contact-region eligibility and
non-overlap of the seated hand and receiver. For connectors, only non-overlap
is assessed. There is no weighted score, fixed shortlist size, deduplication or
automatic winner selection.

### Numerical tolerances and physical margins

The **added physical clearance margin is zero**. Numerical comparison tolerances
are not manufacturing allowances. Plane comparison uses 1e-8 m; nearest-surface
association allows 1e-6 m; radial first-hit comparison uses 1e-7 m.

Trimesh nearest-surface and containment queries use uniformly converted
centimetre copies, matching Fabrica's numerical units. Direct metric queries
can mishandle tiny triangles under Trimesh's absolute tolerances. Face IDs and
physical geometry are preserved, distances are converted back to metres, and
the native CAD and exported poses are unchanged. FCL endpoint intersections use
the metre-scale geometry.

## Meaning and limits of the nominal goals

None of these targets is an official verified complete assembly transform map.
Pins use assumed 12 mm insertion; the 12 mm round pin uses a larger clearance
opening rather than a verified official destination. Gear shaft assignment,
tooth phase and seating are nominal. The nut target represents underside-at-
plate placement, not threading or tightening. Connector targets have zero
assigned engagement. Fixture mounting also uses the existing approximations.

The retained counts establish only that these frozen grasps pass the available
static checks at those targets. They do not establish insertion-path clearance,
release feasibility, contact-force resistance, unchanged closing pose, physical
accuracy or assembly success. The existing native sampler's target/gripper
checks remain historical input evidence; this trial adds no new strict pad-
normal alignment rule or physical closing test.

## Gallery and evidence

The new `gallery/index.html` keeps the requested white CAD-silhouette grid,
with colored contact-pair lines/dots and one tile per part. It contains all
eight scenes. A line represents one pair supporting at least one surviving hand
orientation; it does not show height or every distinct hand angle.

Twelve pairs per tile is a **display page size**, not a shortlist or selection.
All remaining pairs are reachable, and all-pair images are saved. No assembly
views or failure-motion charts replace the contact gallery. Connector tiles
state that only the endpoint check is available. Gallery audits verify 15,093
plotted pairs and their supporting 37,594 poses. Static images and saved data
are checked; browser interactions are not certified by that audit.

Each case saves the original candidate ID, separate contact-region and seated-
hand decisions, rejection reason, and surviving goal hand transforms. The run
manifest records source/receiver hashes. `audit.json` verifies that original
grasp fields, supporting pairs, fixed part-to-hand transforms and protected
files remain unchanged. Region and endpoint rejection counts can overlap and
must not be summed as independent removals.

## Local commands

Use the existing WSL environment and isolated Fabrica dependencies. These
commands install nothing. Run from the repository directory inside WSL:

```bash
PY=/home/jiabao/.venvs/lerobot-libero/bin/python
export PYTHONPATH="$PWD/outputs/cache/fabrica_native_20261027/python"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
$PY -B -m pytest -q -p no:cacheprovider experiments/fabrica_goal_geometry_20261027/test_task_inputs.py experiments/fabrica_goal_geometry_20261027/test_endpoint.py
$PY -B -m experiments.fabrica_goal_geometry_20261027.run_trial
$PY -B -m experiments.fabrica_goal_geometry_20261027.gallery outputs/fabrica_goal_geometry_20261027/runs/20261027T213516Z
```

The runner creates a fresh timestamped run. For a new run, pass its new directory
to the renderer. The gallery command above renders the already recorded run;
it does not regenerate or filter candidates. Focused tests cover coordinate
mapping, pin segment restriction, unknown connector regions, exact bore/seat
classification, preserved outer gear recesses, real FCL intersections, nested
solids and the native mirrored Panda geometry.

## Published basis and our adaptation

[AutoMate, RSS 2024, Appendix D](https://www.roboticsproceedings.org/rss20/p064.pdf)
checks gripper geometry against mating CAD in the assembled configuration.
[Robust Task-Based Grasping as a Service, CASE 2020](https://mjd3.github.io/files/2020-task-grasp.pdf)
excludes task-reserved contact regions before evaluating grasps.
[Fabrica, CoRL 2025, Appendix C.2](https://arxiv.org/html/2506.05168v1#A3.SS2)
checks grasps in assembly context, including sampled paths in its full system.

This trial adapts those principles to our fixed CAD set and existing candidates.
Its explicit pin-segment and bore/seat classification are our implementation
choices, not a claim of copying one paper's exact filtering equations. It does
not reproduce those papers' full planners, trained controllers or assembly
results. The broader review is saved at
`outputs/literature_cad_assembly_20261027/goal_conditioned_filtering_review.md`.
