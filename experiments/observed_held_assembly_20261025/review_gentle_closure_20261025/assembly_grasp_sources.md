# Review: use the assembly goal when choosing the pickup grasp

Review date: 2026-10-25. This is a literature/code review, not a new robot test.
No runtime, model, approved CAD prompt or historical experiment was changed.
The proposed physical arrangement does not require four cameras.

## Main finding

The final assembly pose should be used before pickup to reject grasps that cannot
complete the later task. A grasp that lifts a part successfully may leave the
fingers blocking its hole, require an impossible wrist turn, or lose the part
when insertion forces act on it. This is supported by geometric assembly/grasp
planning, task-dependent force analysis and current open-source motion planning.
The goal is a constraint on the desired future motion. It does not measure the
actual part position inside the hand or remove a wrong initial orientation.

For this project, the smallest useful change is to add destination and approach
checks to the existing small CAD-assisted candidate set. A wholesale planner
replacement is unnecessary for the first controlled comparison.

## Primary sources and what they establish

### 1. Integrated assembly and motion planning using regrasp graphs — Wan and Harada, 2016

[Publisher full text](https://link.springer.com/article/10.1186/s40638-016-0050-2)
and [PMC mirror](https://pmc.ncbi.nlm.nih.gov/articles/PMC5101345/).

This model-based system takes part models, their relative assembled poses and an
assembly location. It computes grasps that are collision-free and reachable at
table placements and assembled placements. It searches direct motions or
sequences involving putting down and regrasping a part. A failure can cause it
to choose another assembly candidate. The essential transferable idea is to use
the same object-relative grasp at the pickup and destination, and verify that
the robot can connect the two. This is geometric planning without neural-network
training. It assumes model/pose information; it is not a camera-based estimate
of a part moving inside the fingers. The paper explicitly limits assembly to
translation and calls the system kinematic, with force control left for future
work. It does not demonstrate threaded nut assembly or rotational mating.

### 2. Planning Grasps for Assembly Tasks — Wan, Harada and Kanehiro, 2019

[Paper, full text](https://arxiv.org/html/1903.01631v1).

The planner divides native mesh surfaces into overlapping approximately flat
regions and synthesizes grasps for industrial hands, including parallel jaws.
It explicitly considers movement caused by closing. The tested method uses
position control without tactile or wrist force sensors. Crucially, the detailed
experiment does not guarantee zero shift: with stricter opposing-surface
conditions, one bunny grasp shifted 0.25 mm and 1.00 mm in the measured horizontal
directions; bearing-housing maxima were 1.95 mm and 1.88 mm. A less strict bunny
setting reached 4.5 mm in one direction. AR markers measured before/after motion;
these are not full 6D error bounds. The evidence supports choosing better contact
surfaces to reduce closing movement, rather than assuming gentle closing makes
the initial grasp transform exact. The paper links PYHIRO as its implementation.
Its CAD candidates are automatically generated rather than manually taught.
Object-frame candidate generation is separate from runtime filtering at the
observed current pose. Our proposed adaptation generates or refreshes the small
set at runtime and tests table access and the assembly goal; CAD preprocessing
alone does not establish either environmental check.

### 3. MoveIt Task Constructor — current official open-source implementation

[Official ROS 2 tutorial](https://moveit.picknik.ai/main/doc/tutorials/pick_and_place_with_moveit_task_constructor/pick_and_place_with_moveit_task_constructor.html)
and [source](https://github.com/moveit/moveit_task_constructor).

MTC combines grasp generation, robot inverse kinematics, approach, closing, lift,
transfer, destination pose, release and retreat. Its serial containers accept
connected end-to-end solutions. The placing stage uses the grasp attachment to
calculate the required robot pose at the specified object destination. Candidate
count is adjustable through angular spacing. It requires a robot model and a
planning scene containing the object's shape and pose. Its example's attached
object is a planning representation: closing the hand and calling
`attachObject` do not measure or physically validate the actual object-to-hand
transform. It is a practical reference for goal-aware filtering, not an insertion
force controller or a solution to closing error. Adopting the full ROS 2 stack
would be a separate integration effort.

### 4. Task-Based Grasp Quality Measures for Grasp Synthesis — Lin and Sun, IROS 2015

[Author-hosted paper](https://rpal.cse.usf.edu/products/2015_iros_lin.pdf)
and [university record](https://digitalcommons.usf.edu/esb_facpub/79/).

This paper separates a grasp's ability to apply the forces/turning forces needed
for a task from the effort required of the robot to carry out the tool motion.
It proposes task-force coverage and manipulator-efficiency measures and tests
them in simulation and with a real robot. The relevant lesson is that lifting
against gravity and pressing a part onto a shaft impose different demands.
For a nut, turning ability matters in addition to vertical holding. Applying
such a calculation here would require credible contact positions, friction,
finger force limits and expected insertion/turning loads. A known destination
alone does not supply those quantities.

### 5. Grasp Planning: How to Choose a Suitable Task Wrench Space — Borst, Fischer and Hirzinger, ICRA 2004

[DLR author-hosted paper](https://www.robotic.dlr.de/fileadmin/robotic/borst/Borst-ICRA2004-TaskWrenchSpace.pdf).

This analytic work describes grasp quality relative to the forces and turning
forces expected during a task. Its distinction is useful here: a generally
stable grasp need not efficiently resist the particular loads generated during
assembly. Contact geometry and friction determine what forces the hand can
transmit. This supports checking task loads when ranking candidates rather than
choosing exclusively by pickup score. The theory is training-free, but its
physical predictions depend on the contact and friction assumptions.

### 6. Developing Compatible Grasp and Fixture Plans — Carriker, Mattikalli and Khosla, 1995

[CMU Robotics Institute publication](https://publications.ri.cmu.edu/developing-compatible-grasp-and-fixture-plans).

This early analytic method jointly chooses parallel-jaw grasp and fixture
positions using equilibrium and geometric compatibility constraints. It seeks
small required closing forces while maintaining stability. It directly supports
the idea that the assembly fixture and the pickup grasp should be planned
together. It is not evidence that minimum force or gentle closing alone removes
position error in an arbitrary grasp. The page establishes the formulation,
not a maintained drop-in implementation for the present gripper.

### 7. Towards Robotic Assembly by Predicting Robust, Precise and Task-oriented Grasps — Zhao, Troniak and Kroemer, CoRL 2020/PMLR 2021

[Official proceedings](https://proceedings.mlr.press/v155/zhao21c.html).

This is unusually close to our application: gear-on-peg, bracket and slot tasks.
It separately optimizes grasp robustness, precision and task performance, and
explicitly addresses displacement after grasping. However, it learns three
cascaded networks using large-scale simulated training. Its real-robot reported
insertion errors are 4.28 mm for brackets and 1.44 mm for gears. It is useful
evidence for the problem formulation, but is not a training-free method and
should not be presented as a guaranteed pretrained solution for our parts.

### 8. ASAP — Tian et al., ICRA 2024

[Official project](https://asap.csail.mit.edu/),
[official code](https://github.com/yunshengtian/ASAP),
[robot planning code directory](https://github.com/yunshengtian/ASAP/tree/main/plan_robot).

ASAP checks assembly paths, stability and robot execution. It has geometric
heuristics as an alternative to its learned part-selection component. The
released robot integration includes grasp planning, inverse kinematics and
collision checks. Its hardware demonstration used a laser-cut part-location
template. The authors explicitly discuss fabrication/localization errors and
the usefulness of vision/force feedback. This is a broad assembly-sequence
system; our already-known destinations do not require importing that entire
system. Its results do not establish precise grasp retention with one camera.

### 9. Available historical code and maintenance considerations

[PYHIRO](https://github.com/wanweiwei07/pyhiro) documents grasp generation,
stable table placements, reachable grasps and regrasp graph planning. It has
older Panda3D/SQL-era dependencies and is better used as algorithm reference
than assumed to be a modern plug-in.

[WRS](https://github.com/wanweiwei07/wrs) now explicitly describes itself as an
archived museum project. Its geometry-based grasp/regrasp ideas remain relevant;
its maintenance status should be disclosed before recommending integration.

[OMG-Planner](https://github.com/liruiw/OMG-Planner) jointly optimizes motion and
online grasp synthesis/selection, but its paper uses online-learning techniques
for selecting grasp goals. Its documented older CUDA/Python stack and emphasis
on pickup trajectories make it a less direct first integration for this task
than adding destination checks to the current sampler or using MTC's structure.

### 10. Uncertainty-Aware Manipulation Planning Using Gravity and Environment Geometry — von Drigalski et al., RA-L 2022

[Official paper](https://ieeexplore.ieee.org/document/9894673/), DOI
10.1109/LRA.2022.3207565, and
[author code](https://github.com/omron-sinicx/uncertainty-aware-manipulation-planning).

This is a particularly close alternative to adding cameras. Known gripper,
support-surface and part geometry are used to plan actions that make the actual
part pose more predictable. The paper adds Grasp, Place and Push actions to its
earlier Touch and Look actions; its abstract reports sub-mm positioning and
over-95% success on tested simulation/robot cases. Those results are scoped to
the tested objects/actions, not a guarantee for our gears and nut. The released
C++/ROS package represents a pose distribution, predicts how actions change it,
and plans uncertainty-reducing sequences; separate base files do not require ROS.
Its two-flat-finger grasp model is much closer to deliberately aligning suitable
surfaces against known pads than merely choosing a lower closing force. Models,
initial uncertainty and action assumptions must be checked for our parts. A
cylindrical contact alone need not constrain rotation around the cylinder axis.

## Proposed project approach — engineering inference from the review

Generate a modest set of grasps from the observed current CAD placement, using
the existing geometric sampler or GR-ConvNet proposals refined against CAD.
Do not select the highest pickup score immediately. For each candidate:

1. Keep its contact points away from fragile edges and prefer broad, nearly
   opposing contact surfaces where the actual part offers them.
2. Check robot/table access at pickup and predict the hand pose for the known
   final part pose using that candidate's object-to-hand transform.
3. Check the fingers against the task board and installed shafts/screws at the
   destination, throughout the final approach, during opening and during retreat.
   Keep the part's hole and required mating surfaces accessible.
4. Check robot joint limits and a connected path through lift, turn, transfer and
   approach. A reachable pickup and reachable endpoint alone are insufficient.
5. Prefer candidates that can resist expected insertion or turning loads, need
   smaller awkward wrist turns, and remain feasible under small declared errors
   in CAD pose and closing position. These error sizes must be measured or used
   as clearly labeled sensitivity tests, not claimed physical guarantees.
6. If no candidate survives, report that this set has no direct solution. Try a
   larger candidate set, another approach direction, or a planned regrasp.

With workspace-to-part transform `W_T_P` and object-relative grasp `P_T_G`, the
nominal pickup hand pose is `W_T_P @ P_T_G`. The nominal assembly hand pose is
`W_T_P_goal @ P_T_G`. The same relative grasp is used at both ends. Actual
closing movement changes `P_T_G`; knowing the desired assembly pose cannot
reconstruct that unknown change. Controlled closing and a small-lift validation
should therefore be tested as an error-reduction strategy, not treated as an
exact replacement for all post-pickup information.

For gears, a grip that avoids blocking the bore and has sufficient support
against insertion loads is preferable to a convenient grip that blocks the
shaft. For the M16 nut, gripping external flats may transmit turning force, but
the task still needs a thread-starting and turning motion. These are proposed
part-specific checks, not conclusions already demonstrated on this hardware.

The required initial information is the CAD model, its currently observable
placement, the robot/gripper model, camera-to-robot calibration and the final
assembly pose/direction. This can use one calibrated camera where the needed
features are visible. The known future pose cannot resolve a current hidden
keyway or an ambiguous orientation. True task symmetry may remove an irrelevant
angle only after verifying the mating operation is invariant to that angle.

