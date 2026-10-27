# Gentle closing and repeatable part position: primary-source review

Reviewed 2026-10-25. Research notes only; no controller, pose-estimation source,
physical robot, model weights or frozen experiment output was changed.

The practical direction is to use the existing camera and CAD estimate, choose
contacts that produce little and repeatable movement during closing, close
slowly, and use sufficient holding force for lifting and assembly. Gentle
closing is useful, but it is not a measurement of the final position of the
part inside the fingers. Low closing speed and low holding force are different
choices.

## What the evidence supports

### Separate closing speed from holding force

The official [Franka gripper API](https://frankarobotics.github.io/libfranka/0.20.4/classfranka_1_1Gripper.html)
accepts width, closing speed and grasping force separately. Its success condition
checks the measured distance between fingers against a width interval. The
[reported gripper state](https://frankarobotics.github.io/libfranka/0.20.4/structfranka_1_1GripperState.html)
contains width and a grasped flag, not a measured six-dimensional part pose.
Therefore, a successful close cannot by itself certify assembly alignment.
These are interface facts, not a proposed physical force setting.

The contact model in [Lynch and Park, Modern Robotics, Chapter 12](https://hades.mech.northwestern.edu/images/archive/b/b2/20170504034507%21MR-2up.pdf)
limits the sideways force a contact can support by the finger's normal squeeze
force and the friction coefficient. A lighter squeeze provides less resistance
to slip. Weight, acceleration, contact position and assembly forces matter;
there is no universal sufficiently gentle force for these parts. Slow movement
can remove inertial effects while gravity and contact forces still require
balance. Thus, slowing closure does not automatically prevent geometric settling.

### Closing can reduce uncertainty by moving the part

[Brost, Automatic Grasp Planning in the Presence of Uncertainty (1988)](https://journals.sagepub.com/doi/10.1177/027836498800700101)
analyses two-dimensional polygonal parts, friction and parallel-jaw operations.
It finds regions of initial uncertainty whose grasps lead to the same final
configuration. Some uncertainty is removed by the grasp itself, and more general
push/offset operations extend what a simple squeeze can accomplish. It is an
analytical planning method, not learned grasp prediction. Its planar assumptions
do not establish full 3D accuracy for our gears and nut.

[Goldberg's mechanical parts-feeding analysis](https://goldberg.berkeley.edu/feeder/sec_3_2.html)
explicitly models the part rotating as the jaws close. The associated
[physical gripper](https://goldberg.berkeley.edu/feeder/sec_2_1.html) uses a sliding
plate to approximate low friction in the plane while retaining grip in the other
direction. This supports mechanical self-alignment as a deliberate design, not
an assumption that standard rubber pads automatically align every part.

[Qian, Hong and Tso, Synthesizing Two-Fingered Grippers for Positioning and Identifying Objects (2001)](https://pubmed.ncbi.nlm.nih.gov/18244824/)
distinguishes partial uncertainty removal by ordinary parallel jaws from a
four-contact finger design that can constrain a locally unique part pose.
This is evidence that finger geometry matters. It requires different fingers
and is not a software-only recommendation for the present gripper.

### Most relevant training-free implementation

[von Drigalski et al., Uncertainty-Aware Manipulation Planning Using Gravity and Environment Geometry (2022)](https://doi.org/10.1109/LRA.2022.3207565)
tracks pose uncertainty through grasp, place, push, touch and camera actions.
Its geometry-based formulation requires no training data or machine learning.
The authored paper's [available full text](https://www.researchgate.net/publication/363746364_Uncertainty-Aware_Manipulation_Planning_Using_Gravity_and_Environment_Geometry)
assumes known shape and centre of gravity, negligible friction during settling,
stable placement, and no rolling/toppling during pushing. It explicitly omits
automatic grasp generation and grasp/push stability evaluation: permissible
grasps must be supplied. Use it to reason about closing outcomes and remaining
uncertainty alongside our grasp generator, not as a drop-in grasp generator.
The reported hardware experiment covers limited objects and action combinations.

The authors' [ROS/C++ implementation](https://github.com/omron-sinicx/uncertainty-aware-manipulation-planning)
represents an estimated pose together with its uncertainty and predicts updates
after the actions. Its camera action accepts one image; it does not require four
surrounding cameras. Touching uses known environmental geometry; its camera
action requires calibration. Extra contact actions and environment assumptions
would have to be checked for this task board.

Integration status checked directly through GitHub: the repository's
[package.xml](https://github.com/omron-sinicx/uncertainty-aware-manipulation-planning/blob/master/package.xml)
declares its licence as `TODO` (blob
`08cd26129b281b6b9fcc9a617593ad91b62c0318`). Do not describe it as a verified
permissively licensed package. It also depends on ROS 1/catkin, MoveIt,
`o2ac_msgs` and `o2ac_skills`; it is not a small Python function we can import
unchanged. The paper remains a strong method reference, while direct source
reuse needs licence clarification.

### Stable pickup and precise pickup are different objectives

[Zhao, Liang and Kroemer, Towards Precise Robotic Grasping by Probabilistic Post-grasp Displacement Estimation (2019)](https://arxiv.org/abs/1909.02129)
separates the chance of lifting an object from the distribution of movement
during grasping. It chooses grasps with both high lift success and low movement
variance, then compensates the expected displacement. Its method uses trained
networks, so it is background evidence rather than the training-free method to
integrate. Its real results do not demonstrate sub-millimetre precision.

[Liu et al., Gentle Grasping: A Method With Low-Cost Magnetic Tactile Sensors (2025)](https://ieeexplore.ieee.org/document/11224760)
uses force measurements and slip detection to adjust squeeze force during
lifting. It evaluates stable low-force grasping, not full part pose accuracy.
It needs additional tactile sensors, so it is an optional hardware path rather
than a prerequisite for a first single-camera trial.

[Funahashi and Yamada (1994)](https://www.jstage.jst.go.jp/article/kikaic1979/60/580/60_580_4161/_article/-char/en)
provide an analytical alternative: rotate a held polyhedral part around vertices
touching the environment and use robot joint readings to recover its pose.
This requires deliberate contact motions and known geometry; ordinary robot
forward kinematics alone does not supply the missing part-to-gripper transform.

## Implications for the present parts — our inference

- Opposite contact surfaces reached almost together reduce the period in which
  one finger pushes the part alone. Prefer broad, stable contacts over a shallow
  contact on a tooth edge or corner, subject to the actual CAD geometry.
- A symmetric jaw motion can centre a part between the fingers while leaving its
  location along the fingers or its rotation insufficiently constrained. A
  circular gear rim does not reveal the unique orientation of a keyway. For a
  nut, opposite flats may settle repeatably while leaving discrete angle choices.
- Preserving an initially inaccurate pose is not always the best goal. A small,
  predictable movement to a repeatable contact configuration can be better for
  assembly than nearly zero but variable movement.
- The known final pose helps choose the pickup: leave the mating hole and the
  shaft approach clear, allow the robot to reach the full insertion path, and
  place finger forces where the part can resist the planned assembly load.
  It does not measure the actual part position after closing.
- The required remaining accuracy depends on the mating clearance and approach
  length. A grasp only needs to constrain the directions that matter for the
  particular mating task; a verified circular shaft fit may tolerate some axial
  rotation, whereas a keyway or thread may require it. Mechanical equivalence
  must come from the part/task geometry, not apparent image symmetry.

## First experiment worth doing — our proposal, not yet executed

Use one calibrated existing camera. Compare normal and slower closing for a
small fixed shortlist of grasps chosen for each assembly goal. Keep finger depth
and contact geometry constant in the speed comparison; vary holding force in a
separate comparison. Avoid picking a numerical force until the actual gripper,
part mass, finger material and retention limits are known. Lift slowly and test
retention before attempting insertion. Measure repeated closing/lift movement
with the one camera when the required features are visible, and report the
unobservable directions instead of assigning exact poses. Include repeated trials,
slips, part movement and the full pickup-to-assembly success count. Use the spread
of resulting part-to-gripper positions, rather than only their mean, to decide
whether extra sensing or a deliberate alignment action is needed.
