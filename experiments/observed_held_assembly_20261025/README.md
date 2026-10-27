# Camera-observed held-part pose and nominal assembly trial

**Current direction after user review:** the four-camera setup below is a
preserved diagnostic experiment. It is not the proposed physical arrangement.
The [current research scope](review_gentle_closure_20261025/precision_grasp_scope.md)
starts after CAD alignment and studies grasp precision from the aligned CAD,
current 6D pose and known final pose. Camera layout, robot reachability and
environment/robot collision planning are excluded from that study.
Historical protocols and recorded results remain unchanged.

The 2026-10-25 request is to estimate CAD pose again after pickup, update the
part-to-gripper position/orientation, and try the large gear, medium gear and
M16 nut at their known board destinations. This trial is for preparation for
physical testing. It does not connect to robot or camera hardware.

The [completed multi-view follow-up](board_feedback_followup_20261025/yaw_feedback/README.md)
records 3/3 simulated pickups and five executed lowering moves per part, with 27/27
available camera pose estimates. No part was released or fully assembled. It
adds fresh global gear-angle ranking after the first local fitter selected
incorrect gear rotations; all earlier runs and their failures remain preserved.

## What the controller can observe

The existing simulator supplies physics, raw colour/depth images and robot
encoder/forward-kinematics feedback. Those images are synthetic RGB-D, not new
physical-camera validation. The estimator and controller receive no simulator
object coordinates, instance segmentation, target mask, target box or
truth-derived camera aim. Historical close-up images and old perfect CAD poses
are excluded from their inputs.

Four fixed calibrated cameras cover the loose parts, raised hand and task board.
The predicted depth of the known robot at its measured joints removes robot
pixels. This prediction contains no object positions. The initial CAD pose is
found from full-image depth components above the calibrated table, followed by
the existing native-scale CAD support-orientation/yaw search, table-constrained
alignment and full projected/observed mask/depth scoring. The target mask given
to GR-ConvNet is that observed depth component, not an engine label. CAD identity
is supplied by the task; object classification is outside this trial.

The new held-part step fits the observed depth from multiple cameras to native
CAD triangle surfaces. All six local pose parameters can change; the table
constraint is removed. A small search around the motion prediction handles
changes during closing. Repeated visible teeth and faces can produce similar
images at distinct CAD orientations. Whole-CAD equivalence is not established;
the output is not a uniquely observed absolute six-dimensional
pose. Unavailable/ambiguous fits are explicit, and frozen depth-agreement gates
are development checks rather than calibrated reliability probabilities.

## Position and orientation inside the gripper

The first post-pickup prediction uses the initial observed CAD pose and measured
grip-site pose at the end of descent, before closing. It assumes the part stayed
stationary during approach. Later predictions use the last observed CAD pose
and simultaneous robot feedback:

```
predicted_world_T_cad = current_world_T_grip
                       @ inverse(previous_world_T_grip)
                       @ previous_observed_world_T_cad

observed_grip_T_cad = inverse(current_world_T_grip)
                     @ current_observed_world_T_cad

world_T_grip_goal = desired_world_T_cad @ inverse(observed_grip_T_cad)
```

All translations are metres. The gripper frame is `gripper0_grip_site`, not the
robot flange or finger tip. Each prediction saves the exact previous estimate,
measured robot-feedback files, hashes and composition rule. After release, a
prior stays in workspace coordinates; it is no longer transported with the hand.

## Boundary between perception and the simulated plant

`plant.py` runs separately and opens the scene. Its online responses contain only
camera-file locations, ordinary robot joint/grip-site feedback and a known
robot/static-board planning model. No loose-product body or saved state appears
in that planning model. `PlanningModel` inserts a fresh CAD surrogate at the
**observed estimate**, not the plant's current pose. Its four native-CAD height
bands preserve stepped gear outlines conservatively; convex bands close bores.

The one-shot `estimate_from_files.py` reads only declared RGB, metric optical-Z
depth, robot-only predicted depth, camera calibration, native CAD and an explicitly
declared observation/FK-derived prior. It cannot receive a target mask or scene
XML through that file interface. Input and source hashes are retained.

The plant may record true object positions and labelled contacts under
`plant_private` for a later independent evaluation. Online code never reads that
directory. Observation-based errors and decisions are frozen before any such
evaluation. A successful offline replay does not mean an assembly succeeded.

## Controlled movement and interpretation

Use the same pretrained Cornell RGB-D GR-ConvNet and bounded CAD refinement as
the pickup follow-up, with the observed initial pose/component replacing the
earlier perfect inputs. No retraining, paid API, new model weights or VLM calls
are needed. Known robot/table/board geometry is checked before movement. The
held object's pose in those checks comes from the observed gripper relationship.
Other loose parts stay visible with collisions disabled, as previously requested.

After pickup and a one-second hold, reobserve the held part. Require at least
120 mm observed rise before moving toward the board. Reobserve after turning,
above the board and between short lowering moves. Stop on unavailable pose,
rejected robot route, large observed part-to-gripper change or insufficient
observed lowering progress. The full constants and geometry/pose protocols are
frozen in the run manifest before robot movement. This is stepwise visual
feedback between moves: simulation time pauses during fitting. It does not
demonstrate continuously running real-time tracking on physical hardware.

The destinations are the previously user-authorized **nominal CAD goals**, not
new verified physical mating transforms. Their positions and positive CAD-Z
directions can be assessed visually. Tooth phase, screw engagement, thread
tightening and full mechanism success are not validated. Existing physical
contact models close gear/nut bores and do not represent nut threading. Those
models can prevent descent even with correct visual alignment; the controller
must not force a part through a shaft to produce a successful picture.

Fixed-camera video frames are recorded during execution. Their camera does not
track the simulator's object pose. They show the robot's actual commanded run,
not an animation that sets object coordinates to the desired answers.

Historical RealSense RGB-D capture code and calibration files exist locally, but
their current hardware calibration has not been verified here. Physical testing
still requires the actual robot/gripper model, measured camera/robot/tool
calibration, validated sensor noise and continuous controller integration.
