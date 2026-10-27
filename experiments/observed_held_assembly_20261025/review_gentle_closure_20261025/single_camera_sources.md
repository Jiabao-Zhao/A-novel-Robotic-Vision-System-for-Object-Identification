# Single-camera alternatives for the held-part update

Research notes, 2026-10-25. This is a literature review, not a new experiment or
hardware validation. No runtime code, frozen results, camera calibration, model
weights or protected prompts were changed.

The user rejected four cameras as a practical requirement. The old four-camera
run should remain a diagnostic record. It does not establish that the existing
physical system can achieve its simulated pose errors.

## Suitable existing code: ICG and M3T

[Stoiber, Sundermeyer and Triebel, ICG, CVPR 2022](https://arxiv.org/abs/2203.05334)
fits object pose using CAD geometry, image regions and depth. It is a model-based
tracker intended for textureless objects; a newly trained object network is not
required. Its occlusion handling is relevant when fingers cover part of a gear
or nut. It is a tracker, so a usable initial pose is still needed. A published
tracking speed is not the latency of our complete acquisition, initialization
and planning pipeline.

[Official ICG implementation and documentation](https://github.com/DLR-RM/3DObjectTracking/blob/master/ICG/readme.md)
describe initialization, pose refinement, image updates, region/depth modalities,
and pose publishers. CAD geometry is precomputed into tracking models. These are
geometric caches, not predefined grasp poses. This is a plausible way to retain
one-camera CAD tracking after closure, provided enough visible geometry remains.
It does not guarantee that a hidden keyway or ambiguous gear angle becomes
observable.

[Official M3T documentation](https://github.com/DLR-RM/3DObjectTracking/blob/master/M3T/readme.md)
provides physical RealSense colour/depth camera classes, recorded-image loaders
and a `run_on_camera_sequence.cpp` example. It explains how to replace the Azure
Kinect classes with RealSense classes and export pose estimates through a
publisher. A single physical RGB-D unit supplies colour and depth; the software
does not require four physical cameras. The documented example still needs
object/model and detector configuration plus calibration. Its geometric region
and depth components are the relevant no-neural-training option here; proposing
an integration does not mean it has been installed or tested in this repository.

## One camera can obtain another view by moving the held part

[Yuan, Ge and Yang, Single-Camera Multi-View 6DoF pose estimation for robotic grasping, 2023](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2023.1136882/full)
uses a fixed camera and robot motion to obtain additional views of a held part.
It optimizes a common part-to-hand transform from images and robot motion. Its
per-view estimator uses deep learning; its assembly demo uses a learned planner.
It is not an entirely training-free implementation. It assumes accurate robot
motion and no part sliding, and acknowledges real slip and calibration error.
Its printed parts do not validate our metal gears or nut.

Engineering inference: our CAD/depth fitting could use the same one-camera
geometry without its learned estimator. Combine calibrated camera geometry,
measured robot joint angles and one unknown part-to-gripper transform, fitting
that transform to observed images. Only request another wrist pose when a
necessary feature is hidden. This is optional recovery, not four views on every
pickup. Moving a weakly held part can invalidate the constant-transform assumption.

## Vision need not remove every small assembly error

[SWRI ConnTact official implementation](https://github.com/swri-robotics/ConnTact)
provides contact-guided assembly behaviours, including a spiral hole search and
compliant insertion. Its example targets circular pegs on a NIST assembly board
with known hole locations and a UR10e. It uses measured force/motion events to
switch between approach, surface finding, search and insertion. This is a
concrete geometric/control alternative to demanding visually perfect alignment.
It requires a suitable robot interface, contact feedback and task parameters;
it is not a ready Panda integration or a complete M16 screw-threading routine.
Example force values in its source are not calibrated force settings for our
hardware and should not be copied as such.

[NIST: Multi-Robot Assembly Strategies and Metrics](https://pmc.ncbi.nlm.nih.gov/articles/PMC5828242/)
describes passive remote-centre compliance and force-guided insertion/search
strategies for accommodating small alignment errors. This supports separating
coarse visual alignment from the final contact operation. Compliance means that
the tool/robot can yield a little during contact; it is different from merely
closing the fingers with a smaller force. Available clearance, chamfers, task
geometry and allowed loads determine what errors can actually be tolerated.

[Franka Control Interface official overview](https://frankaemika.github.io/docs/overview.html)
lists measured robot joint data, externally applied force/torque estimates,
contact information and model forward kinematics. Those are potential physical
feedback interfaces, not evidence that this repository currently has a validated
hardware assembly controller. Robot kinematics locates the hand; it cannot by
itself detect an object sliding between the fingers.

## Metallic parts require real depth tests

[Intel RealSense depth-quality testing white paper](https://www.realsenseai.com/wp-content/uploads/2019/11/RealSense_DepthQualityTesting.pdf)
calls for testing the actual application materials and lighting, and identifies
reflective surfaces as difficult for vision-based depth cameras. Its discussion
also distinguishes low texture from poor infrared reflectance. A geometrically
correct CAD mesh and excellent synthetic fits do not establish real metal depth
quality.

[Basler influencing factors documentation](https://docs.baslerweb.com/influencing-factors)
explains that specular surfaces can redirect illumination and that multiple
reflections can corrupt depth measurements. The relevance is a sensor limitation,
not a requirement to buy extra cameras. A region/silhouette term can complement
missing depth, but it cannot create a distinguishing geometric feature that the
camera never sees.

## Integration facts and alternatives outside the present setup

The [3DObjectTracking repository licence](https://github.com/DLR-RM/3DObjectTracking/blob/master/LICENSE)
is MIT. Its documented ICG build uses CMake, Eigen 3, GLEW, GLFW 3 and OpenCV 4;
RealSense streaming adds the optional `realsense2` dependency. An integration must
preserve mesh units and the CAD-frame transform. Initializing from our current
pose estimate and publishing updates would require an adapter; none was built
or installed in this review.

The [ConnTact licence](https://github.com/swri-robotics/ConnTact/blob/master/License.md)
is Apache 2.0. Its documented development environment is Ubuntu 20.04 with ROS
Noetic, and it names `transitions` and `modern_robotics` Python dependencies plus
ROS source dependencies. The robot-interface work and contact-control validation
are substantial compared with adding a grasp filter to our existing simulation.

[MultiSCOPE, RSS 2023](https://www.roboticsproceedings.org/rss19/p078.pdf)
estimates held-object poses from known geometry, robot measurements and contact
interactions. Its demonstration uses two arms, force/torque feedback and a planar
pose state constrained by parallel jaws. It assumes exploratory contacts do not
slip. This is evidence that contact can disambiguate pose, but replacing four
cameras with two-arm probing would not match the requested simple setup.

[Constrained in-hand needle tracking, ICRA 2023](https://arxiv.org/abs/2210.11973)
uses feasible grasp constraints inside a pose filter. It supports rejecting
impossible tool/object configurations. Its needle and stereo-vision setting
does not establish that jaw opening alone determines our gear or nut pose.

## Practical recommendation for this project

1. Use the actual existing camera and its measured robot-to-camera calibration.
   Keep CAD identity supplied by the identification stage. Do not insert synthetic
   extra camera views into the physical proposal.
2. Select a grasp that works at both pickup and assembly, including jaw clearance
   around the hole/shaft and retention under the expected loads. Prefer a grip
   that leaves a useful part feature visible to this camera.
3. Close slowly with controlled force, then use enough holding force for lifting
   and the intended assembly operation. Slow approach and low impact help; a
   weak hold is not a pose guarantee.
4. Test the simple version first: retain the pre-pickup CAD-to-gripper relation,
   make a small test lift, and verify the visible part with the same camera.
   Measure repeatability across repeated pickups. If the resulting uncertainty
   is comfortably inside the actual assembly clearance, a full new global pose
   search on every cycle may be unnecessary.
5. If a correction is needed and the useful surfaces are visible, initialize a
   one-camera model-based tracker such as ICG from the existing CAD pose. If the
   image cannot distinguish the relevant angle, keep that uncertainty explicit.
6. If necessary, make a small, planned wrist movement to expose the missing
   feature to the same camera. Reject this option when the grasp cannot retain
   the part during that motion. Do not assume FK reveals slip.
7. Use bounded contact-guided insertion where required. An M16 nut needs a
   screw-starting/turning operation, not just a desired endpoint pose.

The above workflow is an engineering proposal from the cited evidence, not a
published guarantee of our success rate. A simple error check helps explain why
it needs validation: a point L metres from the held reference point moves about
L times the angular error in radians. At L=50 mm, a one-degree orientation error
corresponds to about 0.87 mm of sideways movement at that point. This is a
geometric illustration, not an assertion of our part clearance or measured error.

Gentle closing also cannot repair a wrong *initial* CAD angle. The retained first
run found incorrect initial gear angles before pickup. A stable grip could
preserve those wrong estimates perfectly. Any decision to omit a post-pickup
correction must therefore validate both the initial pose and the closure/lift
disturbance against the assembly's actual requirements.

For round pins, rotation around the pin's own axis may be irrelevant. For a keyed
gear, tooth engagement or a thread, the relevant rotational requirements differ.
Derive allowed orientation alternatives from the actual mating geometry and
task; do not infer whole-CAD symmetry or assembly equivalence from a similar
outline.
