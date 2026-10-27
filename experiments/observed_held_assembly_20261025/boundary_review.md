# Observation-only held-part boundary review

This new experiment must keep held-part pose updates separate from simulator
answers. The previous pickup evidence remains frozen. Correct CAD identity and
native CAD scale are supplied inputs; an RGB-D pose update must still find how
the held part sits relative to the robot.

## Inputs the estimator may read

- RGB and metric optical-camera-Z depth images.
- Image dimensions, camera intrinsics, and known camera-to-world calibration.
- Robot arm/finger encoder values and grip-site forward kinematics.
- The supplied native metric CAD mesh and CAD-derived dimensions/normals.
- Frozen fitting constants and camera/robot inspection positions specified from
  calibration, robot state or CAD dimensions.

A crop centered on the grip-site position, sized from the CAD and gripper, is a
robot/CAD prior. A crop centered on the simulator's part body is an answer leak.
Any initial orientation/position prior must have explicit observation, robot or
CAD provenance. A prior copied from the simulator target pose remains a leaked
answer even if the fitter later adjusts it.

## Inputs to exclude

- `RobotTrial`, its MuJoCo model/data, a scene XML or objects-centered keyframe.
- `target_pose()`, `target_bid`, target free-joint addresses, `d.xpos/xmat`, full
  scene `qpos`, saved `states`, trajectories, contacts, and old result files.
- Engine segmentation, target body/geometry labels, perfect target masks, true
  bounding boxes, target-only depth ranges, or normals sampled by engine ID.
- The previous closeup pictures or their camera settings: that rendering helper
  aimed at a midpoint containing the simulator's true CAD position.
- Assembly goals, final assembly pictures, evaluation errors or outcomes as
  fitting/selection evidence.

The old `capture_scene` cannot be used unchanged. It renders segmentation,
selects pixels using `trial.target_bid`, performs target-labelled ray checks,
and exports target-mask/depth-range information. A new capture helper must
export RGB-D and ordinary calibration only.

## Data handoff

The trusted capture/execution harness owns the simulator. It writes a fresh,
hashed observation-only inbox. The estimator receives paths to that inbox and
the supplied CAD; it receives no simulator object. Its output records the
estimated `world_T_cad`, corresponding
`grip_T_cad = inverse(world_T_grip_FK) @ world_T_cad`, fit evidence, ambiguity,
availability and elapsed time. Preserve every unavailable result.

Freeze the input manifest, estimate and selected plan before evaluating target
truth or executing a move based on that estimate. A separate offline evaluator
may compare the frozen estimate with simulator truth after decisions are fixed;
its files must never return to the estimator or planning inputs. File/source
audits establish the data used by this implementation; a process boundary alone
is not an operating-system access-control sandbox.

The planner receives the observed part-in-gripper transform, robot state, known
static world/calibration and user-approved static goals. It does not receive
`RobotTrial` or the live target body state. A robot/static-world-only planning
model avoids hidden target free-joint access. If held CAD is included in a path
check, place it using `world_T_grip_FK @ estimated_grip_T_cad`.

## Existing controller limits

The old torque controller uses robot joint/site state rather than target pose
feedback. Its recorder writes target truth and labelled forces for audit,
however, so recorder outputs are forbidden online inputs here. Its IK and
collision helpers also hold a full scene state; passing the whole helper object
to the estimator or planner would break the intended boundary.

Existing `validate_path`/`collision_report` rejects robot self/table overlap.
Robot/board and held-part/board contacts are separately recorded, and the checker
is not a complete assembly collision certificate. Observation-only estimation
and a passed IK path do not establish that a gear is seated or a nut is threaded.

## Goal and outcome language

The earlier user authorized nominal simulation goals for parts without verified
final mating transforms. Preserve that distinction for the large gear, medium
gear and M16 nut. Copy goal transforms from the frozen goal source with hashes
and explanations; do not infer new destinations from a part's live pose.

Report separately: an estimate was available, its frozen fit/error diagnostics,
a move was planned, a move was actually executed, the part reached a nominal
pose, and a mechanical mating operation was verified. Define position/angle and
contact limits before examining outcomes. Preserve failures and all three parts
in each denominator. Repeated visible gear/nut surfaces can leave rotation
around the axis ambiguous even when they fit closely. That does not make
different complete CAD orientations equivalent.

## Separate static-depth and gear-angle follow-ups

Known table/board depth may be predicted from the public sanitized model,
calibration and current robot feedback. The model is checked to contain exactly
the seven arm and two finger joints, no product body and no saved keyframe.
The follow-up removes raw depth matching this prediction within 1 mm. It saves
raw images, predicted depth, removal masks, filtered images and all input hashes.
The independent audit recomputes that exact subtraction from saved arrays and
checks that RGB, calibration and robot feedback were copied unchanged. Part
surfaces within 1 mm of known geometry can also be removed; report that limit.

The fresh gear-angle worker receives both raw and filtered observations plus
known-static depth. Its initial pose remains the preceding observed pose moved
by measured robot forward kinematics. After pickup, declared CAD-angle starts
span a full circle; later updates use five declared local starts. Each start
uses the unchanged bounded local fitter. Ranking uses point-to-native-CAD
distance and full projected-CAD-to-raw-depth error. Known static geometry may
exclude occluded image pixels. No stored diagnostic-selected pose, private
trajectory, target ID or offline error is an input.

The audit preserves every candidate and replays the declared starts, per-seed
local bounds, ranking arithmetic, distinct-orientation grouping, numerical
exact-tie rule and final position/axis/full-rotation gate. A finite geometric
score gap is uncalibrated. Accuracy against withheld truth is reported only
after all three parts finish and online evidence is frozen; it cannot correct
the controller's already selected frames or moves.
