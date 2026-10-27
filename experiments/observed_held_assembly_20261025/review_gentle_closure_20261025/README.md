# Review: controlled closing and grasp selection for assembly

**Scope superseded after user clarification:** the
[current research definition](precision_grasp_scope.md) assumes successful CAD
alignment and uses current 6D pose plus final assembly pose to investigate grasp
precision. Robot/environment collision checking, reachability and camera layout
are excluded. The source summaries below remain background research; the
broader deployment recommendation and checklists are not current requirements.

Reviewed 2026-10-25. This is a literature and implementation review, not another
robot experiment. No model, controller, approved CAD prompt, physical hardware
or frozen trial output was changed.

The four virtual cameras in the previous trial are a retained diagnostic setup.
The user rejected that arrangement for physical deployment. The proposed
direction below uses the existing camera, known CAD, robot joint feedback,
controlled closing and the known assembly goal. No four-camera arrangement is
proposed.

**Recommendation.** Select the pickup grasp for the complete assembly task,
prefer contact surfaces that give a repeatable held position, and close slowly
with controlled force. Test whether the resulting part-to-gripper relationship
is repeatable enough for the actual mating clearance. Make extra observation or
a deliberate alignment action conditional on the remaining error. Do not make
four-camera re-estimation a mandatory pipeline stage.

## What gentle closing can and cannot accomplish

There are three separate choices:

| Choice | Intended benefit | What it does not establish |
| --- | --- | --- |
| Slower finger motion at contact | Reduce impact and sudden disturbance | That the part never moves as the fingers settle against it |
| Lower squeeze force | Limit contact load | Sufficient grip against weight, acceleration, insertion force or turning force |
| Better contact locations and finger shape | Make the final position more repeatable, sometimes deliberately centering the part | That every position and angle becomes known |

The official [Franka interface](https://frankarobotics.github.io/docs/franka_ros2.html)
specifies closing speed and holding force separately. Its grasp success test
uses finger separation within an allowed interval. This is useful feedback,
but it is not a measurement of the part's complete pose inside the fingers.
The review does not prescribe numerical hardware speed or force settings.

If one finger reaches the part first, it can push or turn the part before the
second finger arrives. Moving slowly can still produce that motion. Sometimes
this is useful: contact with known jaw surfaces can bring several uncertain
starting positions to a more repeatable final position. Classical geometric
planning explicitly studies this effect. See
[Brost, 1988](https://journals.sagepub.com/doi/10.1177/027836498800700101)
and [Goldberg's mechanical analysis](https://goldberg.berkeley.edu/feeder/sec_3_2.html).

For this project, the useful target is **a small, repeatable and sufficiently
known closing outcome**. Almost no average movement is not enough if individual
pickups vary widely. Some predictable movement can be acceptable if the planner
uses the resulting relationship rather than the original nominal one. This is
our engineering interpretation of the contact-planning literature.

Two flat jaws also leave some movements weakly constrained. They may center the
part across the closing direction while leaving uncertainty along the fingers
or in a rotation. For a round outline, a centered grasp does not identify a
hidden keyway. Suitable nut flats can improve repeatability, but jaw separation
alone cannot identify every angular alternative. Actual constraints depend on
the CAD surfaces, finger pads and friction.

## What the final assembly pose contributes

**Yes: use it during generation or selection, before closing the gripper.** A
good pickup can still be unsuitable for assembly if the fingers occupy the space
needed by the shaft, cannot clear the board, cannot open after seating, or force
the robot into an unreachable wrist position. Geometric assembly planning uses
both the current and assembled configurations when finding usable grasps and
connecting motions. See [Wan and Harada, 2016](https://link.springer.com/article/10.1186/s40638-016-0050-2).

For each current runtime candidate, check:

1. The fingers can approach and close at the observed resting pose without
   hitting the table.
2. The same relative grasp permits the required final part position and
   orientation.
3. The fingers leave the mating hole, shaft approach and other required contact
   surfaces clear throughout the final approach.
4. The robot can connect pickup, lift, turn, transfer and assembly approach.
   Separately reachable endpoints do not establish a connecting path.
5. At the destination, the fingers can open and the hand can retreat.
6. The contacts can resist the expected holding, insertion and turning loads.
7. Small plausible pose/closing errors do not immediately make the candidate
   collide or miss the mating feature.

These are proposed project checks, informed by the literature. The final pose
by itself does not specify the insertion direction, clearances, required forces,
thread pitch or a complete contact controller; those come from the mating CAD
and task definition. Ignore other loose parts as previously requested, while
still checking the robot/table and the gripper's clearance from the known board
and its installed mating features.

Candidates can also be generated backwards: choose feasible gripper positions
around the part at its assembly goal, then map those relative grasps to the
part's observed current pose. Retain only candidates that work at both ends and
along the connecting motion. This can reduce wasted pickup-only candidates.
It does not require manually teaching a fixed grasp for each CAD model.

For implementation, let `world_T_part` locate the part in the robot workspace,
and `part_T_grip` describe the chosen hand pose relative to the part:

```
world_T_grip_pickup = world_T_part_pickup @ part_T_grip
world_T_grip_goal   = world_T_part_goal   @ part_T_grip
```

These are nominal predictions. If closing changes the actual relationship, use
the observed or validated post-closing `part_T_grip`, including any known
systematic offset. All translations are metres; orientations are rotation
matrices. The hand frame must be the calibrated grip-site frame, rather than an
unlabelled wrist or finger-tip frame.

The goal constrains the desired future state. It does not observe the present
held pose, correct a wrong initial gear angle, or prove that no slipping occurs.
Our active CAD/SAM3 branch scoring keeps a fixed branch orientation and fits
translation. This review assumes that the task-relevant initial orientation or
allowed orientation range is sufficiently established; it does not claim that
the full interactive system already provides a unique arbitrary 6D pose.

## The most relevant research and implementations

The table separates useful geometric components from systems that need training
or additional hardware. Each result belongs to its paper's own objects and
setup; it is not a performance claim for our gears or nut.

| Work | Relevant contribution | Fit and limitation for this project |
| --- | --- | --- |
| [Brost, Automatic Grasp Planning in the Presence of Uncertainty, 1988](https://journals.sagepub.com/doi/10.1177/027836498800700101) | Analytical planning of parallel-jaw actions that reduce uncertainty through contact | Training-free foundation; planar shape/friction assumptions do not prove arbitrary full-3D accuracy |
| [Wan and Harada, Integrated assembly and motion planning using regrasp graphs, 2016](https://link.springer.com/article/10.1186/s40638-016-0050-2) | Selects grasps usable at initial and assembled poses, with direct or regrasp paths | Strong geometric model for goal-aware selection; demonstrated assembly is translational and kinematic, not nut threading |
| [Wan, Harada and Kanehiro, Planning Grasps for Assembly Tasks, 2019](https://arxiv.org/html/1903.01631v1) | Automatically synthesizes contacts from CAD surface regions, with grasp precision considered | Training-free; detailed closing tests still show roughly 1–2 mm horizontal movement for stricter tested grasps, not a zero-error or full-6D guarantee |
| [von Drigalski et al., Uncertainty-Aware Manipulation Planning Using Gravity and Environment Geometry, 2022](https://doi.org/10.1109/LRA.2022.3207565) | Predicts how known fingers, gravity and grasp/place/push/touch/look actions affect remaining pose uncertainty | Closest conceptual match to repeatable geometric settling; permissible grasps and stability checks must be supplied |
| [Lin and Sun, Task-Based Grasp Quality Measures for Grasp Synthesis, 2015](https://rpal.cse.usf.edu/products/2015_iros_lin.pdf) | Evaluates grasps against the forces and turning forces required by a task | Supports different grasp rankings for lifting, insertion and turning; needs credible contact/friction/load assumptions |
| [MoveIt Task Constructor, official tutorial](https://moveit.picknik.ai/main/doc/tutorials/pick_and_place_with_moveit_task_constructor/pick_and_place_with_moveit_task_constructor.html) | Connects grasp generation, arm reachability, approach, lift, transfer, place, release and retreat | Practical training-free planning reference; an attached-object model assumes a fixed held transform and does not measure closing error |
| [ICG, Stoiber et al., CVPR 2022](https://arxiv.org/abs/2203.05334), [official implementation](https://github.com/DLR-RM/3DObjectTracking/tree/master/ICG) | CAD-based RGB-D pose refinement/tracking without neural training | Optional single-camera correction; needs initialization and visible evidence, and cannot guarantee recovery of a hidden gear angle |
| [SWRI ConnTact](https://github.com/swri-robotics/ConnTact) | Force-guided insertion with compliant motion and a spiral hole search example | Training-free contact-control reference, separate from grasp generation; robot interface must be adapted and nut threading is not supplied as a validated solution for this setup |
| [Zhao, Troniak and Kroemer, Towards Robotic Assembly by Predicting Robust, Precise and Task-oriented Grasps, CoRL 2020/PMLR 2021](https://proceedings.mlr.press/v155/zhao21c.html) | Separates reliable holding, precise holding and assembly suitability, including gear-on-peg tasks | Directly relevant problem formulation, but uses three trained networks; not the requested training-free implementation |
| [Fabrica, CoRL 2025](https://fabrica.csail.mit.edu/) | Combines assembly, grasp and motion planning with contact-aware execution | Supports planning the grasp for assembly; complete system uses dual arms, generated pickup fixtures and learned local controllers |
| [ManiDreams, 2026](https://arxiv.org/abs/2603.18336), [official project](https://rice-robotpi-lab.github.io/ManiDreams/) | Tests candidate actions over uncertain predicted outcomes and task constraints without retraining the base policy | Broader robustness framework; backends/policies can be learned, and examples do not validate our precision gear/nut assembly |

Two particularly useful details follow from the deeper review:

- The **2019 CAD planner** automatically generates grasps; it does not require
  manually annotated grasp poses. Its paper's workflow can preprocess candidates
  in object coordinates and filter them against current geometry later. Our
  runtime can generate a small set on demand instead. Its old PYHIRO code and
  archived WRS successor are algorithm references rather than an assumed modern
  dependency.
- The **2022 uncertainty planner** offers a model of how contact can reduce
  uncertainty, not a complete automatic grasp generator. Its known-shape,
  centre-of-gravity, friction and settling assumptions must be checked. The
  [public research code](https://github.com/omron-sinicx/uncertainty-aware-manipulation-planning)
  uses ROS 1/C++ and O2AC dependencies; `package.xml` lists its licence as `TODO`.
  Treat direct reuse as requiring licence clarification. Its geometric ideas
  can guide our own small implementation without importing that stack.

The existing geometric sampler is therefore the lowest-disruption place to add
assembly-aware checks. GR-ConvNet can remain an optional pretrained proposal
source; it is not needed for the new geometric checks. VLM ranking can remain
after quantitative feasibility filtering, if desired. Neither a VLM preference
nor a high pickup score establishes precise contact geometry or held-pose error.

## Do we need a complete pose estimate after every pickup?

Not necessarily. A fixed part-to-gripper relationship is a defensible working
model if repeated tests show that initial localization, closing, lifting and
subsequent motion leave sufficiently small error for the intended task.
This is a condition to validate, not a property conferred by slow closing.

Keep three sources of error separate:

| Error source | Relevant response |
| --- | --- |
| Initial CAD location or orientation is wrong | Improve or clarify the initial estimate; gentle closing cannot restore a hidden missing angle automatically |
| Closing pushes/turns the part unpredictably | Improve contact locations, closing control, or use a deliberate alignment action |
| The held part slips during lifting or assembly | Improve holding/contact geometry and load control; a correct pre-lift estimate is no longer sufficient |

Set error margins from measured camera/calibration and closing variability.
Do not convert an uncalibrated SAM3 confidence or CAD-matching score into a
millimetre error bound or a probability of successful insertion.

Use the actual assembly tolerance as the decision criterion. For an illustrative
geometry calculation only, a 1-degree tilt over a 20 mm length produces about
0.35 mm sideways offset. Those are example numbers, not measured dimensions or
tolerances for our parts. A small-looking angular error can therefore matter in
a tight fit.

Do not insist on a unique full CAD rotation if the actual mating geometry does
not require it. A verified round shaft/hole pair may permit rotation around its
axis. A keyway, gear mesh, thread or gripper-clearance restriction can make that
rotation important. Define allowed rotations from the mating geometry and task;
do not infer them from a circular-looking image or declare the whole CAD model
symmetric without checking.

## Applying this to the large gear, medium gear and M16 nut

These are proposed candidate checks, not newly verified grasps or robot moves:

| Part | What to prefer or test |
| --- | --- |
| Large gear | Use CAD to find suitable opposing contacts, if present, with stable finger-pad support. Keep the bore and shaft approach clear. Test clearance through the entire seating depth and hand retreat. Avoid relying on a shallow tooth-edge contact when a better supported alternative exists. |
| Medium gear | Apply the same contact/clearance checks and compare the wrist path needed by each candidate. Choose a grasp that reaches assembly directly when possible, rather than accepting the best pickup and trying to solve a difficult turn afterward. |
| M16 nut | Test opposite external flats and whether the fingers remain clear of the screw and board. Distinguish putting the nut onto a shaft from threading it. Threading needs controlled turning and axial motion, load limits and possibly regrasp or an appropriate tool. A final pose alone does not provide that procedure. |

No specific contact band is guaranteed before inspecting the actual CAD and
checking the proposed finger geometry. Keep the finger depths already found to
work as candidates; do not assume the same depth is best for every assembly.

The previous simulator's solid collision approximations close the holes and
omit threading. They must be corrected before using simulated seating success
as an assembly metric. The review does not reinterpret the earlier stopped
attempts as successful assembly.

## A small, useful next comparison

The following is a proposal, not an executed trial:

1. Use one existing calibrated camera and the current CAD estimate. Keep CAD
   identity and known final task-board goals as supplied inputs.
2. Retain a small shortlist from the existing sampler or CAD-refined network
   proposals. Apply the goal, approach, release and retreat checks before pickup.
3. Compare baseline closing with slower closing at unchanged contact locations
   and finger depths. Test holding-force changes separately, using values
   appropriate to the actual hardware and measured retention needs.
4. Repeat pickups at multiple initial offsets. Measure the mean and variation
   of part movement during closing and lifting. Use visible CAD features and the
   same camera; report unobservable directions rather than inventing a full pose.
5. If those errors fit inside the mating tolerance with margin, use the
   validated held relationship for transfer. Otherwise add one single-camera
   check or a tested mechanical alignment action.
6. Record complete pickup-to-assembly attempts, including failed pickups,
   rejected paths, part slips, intervention and total time. Do not count a
   finger-width success flag as pose accuracy or insertion success.

If the one camera cannot see a required feature, a small robot inspection
motion is an optional fallback. Combining views acquired at different times
requires the part to remain fixed in the fingers during those motions; it is
not a free replacement for four simultaneous views. A new global orientation
should not be invented when that condition or visibility fails.

**Recommended implementation order:** add goal-aware geometric filtering first;
then test closing repeatability. Consider the 2022 uncertainty/contact model
when contact can make the result more repeatable. Add ICG only if an observed
pose correction is actually needed. Use a separate force-guided insertion or
threading procedure for mating. None of these steps requires new neural-network
training for our parts.

Additional primary-source notes are retained in
[gentle_closure_sources.md](gentle_closure_sources.md),
[assembly_grasp_sources.md](assembly_grasp_sources.md) and
[single_camera_sources.md](single_camera_sources.md).
