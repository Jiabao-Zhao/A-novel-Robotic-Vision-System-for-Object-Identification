# Fresh observed gear angles and board attempts

**Physical deployment status:** the user rejected this four-camera arrangement.
This run remains a historical diagnostic, not a proposed hardware setup. The
[current review](../../review_gentle_closure_20261025/README.md) instead considers
the existing camera, controlled closing and grasp selection using the assembly
goal. The capture protocol and frozen run outputs below are preserved.

The completed run is `20261025T203906Z`: **3/3 pickups executed in simulation**,
**five actual board-lowering moves per part**, **27/27 available camera pose
estimates**, **0/3 releases** and **0/3 completed assemblies**. All updates came
from observed RGB-D and measured robot forward kinematics. The private simulator
poses were withheld until every online decision was frozen and the server shut
down, then used only for a separate accuracy check.

| Part | Last observed height above nominal goal | Exact observed stopping reason |
| --- | ---: | --- |
| Large gear | 19.985 mm | Requested another 2.001 mm descent; measured only 0.016 mm downward movement. |
| Medium gear | 19.830 mm | Requested another 2.000 mm descent; measured only 0.170 mm downward movement. |
| M16 nut | 24.804 mm | Estimated part-to-gripper position changed 4.129 mm, exceeding the fixed 3 mm limit. |

These are accepted camera estimates, not commanded heights or simulator truth.
An available fit means it passed the depth checks, not that every pose parameter
is correct. Initial table fits still choose incorrect gear angles; the fresh
full-circle search corrects those choices after pickup. Report post-pickup
accuracy separately from initial localization and from fit availability.
The same saved 14-part scene was reset for each trial; the requested three parts
were tested individually. Other loose products stayed visible with their
contacts disabled. Four fixed simulated RGB-D views were used. Their physical
camera arrangement and calibration have not been validated.

[Detailed report](../../../../outputs/observed_held_assembly_20261025/board_feedback_followup_20261025/yaw_feedback/runs/20261025T203906Z/report.md)
and [camera views / recorded motion](../../../../outputs/observed_held_assembly_20261025/board_feedback_followup_20261025/yaw_feedback/runs/20261025T203906Z/gallery/index.html)
preserve each update, movement, score and unavailable historical result.

This third trial adds multi-view gear-angle selection to the preserved first
run and board-filter/route follow-up. All three parts start again from fresh
raw RGB-D observations and new physical simulated pickups. No earlier selected
pose, simulator product pose or audit error initializes this run.

The first local fitter found good gear centers but chose incorrect rotations
around the upright axes. A separate four-camera diagnostic searched a full
circle, ranked native-CAD surface and projected-depth agreement, and froze its
choices before any truth check. Its two selected gear poses were accurate in
that ideal simulated observation. This motivates a new run; those saved poses
are never copied into its controller.

For each fresh gear pickup, fit a full circle using 6-degree starts for the
large gear and 9-degree starts for the medium gear. These are explicit
CAD-specific search spacings for this local trial. Later updates test five
starts around the previous observed pose transported by measured robot motion:
zero, plus/minus one spacing and plus/minus two spacings. The unchanged local
fitter refines each start. Choose by observed-to-CAD and full projected CAD-to-
raw-depth agreement. Preserve every candidate, its score, the next different
orientation and the uncalibrated score gap. A finite ranking gap is evidence in
these images; it is not a probability of correctness with a physical camera.

Fitting uses depth after removing known robot/board surfaces. Ranking compares
against raw depth and accounts for known robot/static occlusion. The known
static render contains no product body or estimated product pose. The M16 nut
uses the original held-part fitter because its first-run full orientation was
already accurate; its pose is still remeasured after every motion segment.

The gripper-to-part relationship is computed from the selected observed CAD
pose and simultaneous robot forward kinematics. The original nominal goals,
robot path checks, stepwise lowering, observed-shift/progress stop rules and
full-rotation/position release checks remain explicit in the run manifest.
Waiting for completed file responses never resends a physical command.

Run in the existing WSL environment:

```
python -m experiments.observed_held_assembly_20261025.board_feedback_followup_20261025.yaw_feedback.run_trial
```

Generated outputs are under
`outputs/observed_held_assembly_20261025/board_feedback_followup_20261025/yaw_feedback/runs`.
All source hashes and protocols are saved before movement. Online decisions
are frozen and the simulation server shuts down before an independent evaluator
may compare against private simulator truth. That evaluation cannot change the
already selected poses or moves.

This is an observation-based simulation experiment for later physical testing.
It does not validate real depth noise, real calibration or a physical robot.
The current contact model closes CAD bores and omits gear meshing/nut threading;
nominal alignment and verified mechanical assembly are separate outcomes.
