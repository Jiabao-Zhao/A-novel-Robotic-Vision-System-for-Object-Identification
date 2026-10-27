# CAD grasp candidates with VLM final selection

Current research scope, clarified by the user on 2026-10-26. This is a literature
review and proposed method, not a newly implemented experiment. The earlier
proposal to predict and rank assembly error before choosing a grasp is withdrawn.
It was a suggested extension, not an established component of this project.
The VLM should make the final choice among a small set of CAD-derived grasps.
The user also clarified that measuring assembly precision after execution is
impractical. Such measurement is not a requirement of this study.

**Latest refinement:** the [downward-grasp and modeled-contact specification](../../runtime_antipodal_assembly_20261023/smaller_search_20261023/REFINED_METHOD.md)
supersedes the earlier orientation sampling, shortlist sizing and table-exclusion
recommendations below. It fixes downward approach while retaining table-plane
rotation, adds combined modeled hand/target/table validation without pickup
execution, uses a variable-size grouped set and lets the VLM compare the retained
alternatives. Validation must not stop at the first passing candidate. This
specification is not an implemented checker or an eight-scene result.

## Inputs and output

Assume CAD alignment has succeeded. Inputs are the native-scale CAD, the current
estimated object position/orientation, gripper/finger geometry, and the supplied
assembly task and final object pose. RGB-D supports identification and alignment;
the grasp generator need not infer the part shape from RGB-D alone. Keep the
supplied final orientation; do not silently relax it because a part looks symmetric.

Each candidate records the gripper position/orientation, two intended finger
contacts, finger separation and contact depth along the finger pads. The VLM
returns a candidate ID, mapped back to the existing numerical pose. It does not
need to invent exact positions or rotation matrices.

## How CAD methods generate a set

For a two-finger parallel gripper, start with two opposing contacts. Do not
assign one grasp to every CAD triangle or assume every side must have a grasp.

1. Sample CAD surface points, or group nearby triangles into contact regions.
   Include geometry hidden from the current camera.
2. Find pairs whose surface directions oppose one another and whose distance
   fits the gripper's opening range. Check that the pads can make the intended
   contact with the target part.
3. Place the pads at the contacts. Their connecting line determines the
   finger-closing direction. Rotate the hand around that line to generate
   different approach directions and hand orientations.
4. Use finger geometry and contact depth to place the gripper body. The midpoint
   of the contacts is a contact centre, not automatically the robot tool origin.
   Preserve the previously successful pin contact depths in a later comparison.
5. Use CAD alignment to express candidates in workspace coordinates. Omit the
   resting contact face and approaches from underneath, as requested.
6. Remove near-duplicates while keeping representatives of different contact
   regions and approach directions. This reduces the choice set without
   choosing its best assembly grasp.

For example, the same left/right contacts on a box may be approached from above
or from the front. Where the fingers touch and where the hand approaches from
are separate choices. An approach from above can grip two side surfaces. Some
surfaces have no suitable opposing partner. An exhaustive sweep of all point
pairs is unnecessary for our proposed bounded runtime sampler; its budget and
time still need testing.

## Published precedents

| Method | What it establishes | Limit for this study |
| --- | --- | --- |
| [Wan, Harada and Kanehiro, Planning Grasps for Assembly Tasks (2019), section IV-B](https://arxiv.org/html/1903.01631v1#S4.SS2) | Groups CAD surfaces, finds opposing contacts and varies hand rotation about their connecting line. | Automatic CAD generation; no VLM final selection. |
| [Kleeberger et al., Automatic Grasp Pose Generation for Parallel Jaw Grippers (2021), sections 3 and 4.3](https://arxiv.org/html/2104.11660v1) | Samples CAD points/directions, tests contact pairs, varies hand orientation and retains a diverse reduced set. | The original method is offline. Its clustering took minutes to hours; it is not a verified fast runtime implementation. |
| [TOG: Training-free Task-oriented Grasp Generation (2025), sections III-B/C](https://arxiv.org/html/2502.04873v4#S3.SS3) | Pretrained Contact-GraspNet generates 6D candidates. Clustering gives representatives; numbered drawings/contact marks and task text let the VLM choose a candidate. | Direct precedent for VLM final choice. RGB-D generation, not precise CAD-guided assembly. Simulation used three representatives. |
| [OWG: Towards Open-World Grasping with Large Vision-Language Models (2024), section 3.2](https://arxiv.org/html/2406.18722v3#S3) | GR-ConvNet generates grasps; a VLM ranks numbered drawings and the highest-ranked grasp is executed. | Four degrees of freedom with a fixed overhead approach. Ranking concerns nearby-object contacts, not precision assembly. |

Implementation references: [TOG](https://github.com/StevenLiudw/TOG) and
[OWG's ranking prompt](https://github.com/gtziafas/OWG/blob/main/prompts/grasp_ranking.txt).
These are research references, not newly integrated or rerun implementations.
The papers need no new task training but use pretrained grasp generators. Our
proposed CAD generator would be geometric; the VLM would remain pretrained
without fine-tuning.

## Proposed division of responsibilities

**Aligned CAD and fingers -> diverse numerical candidates -> numbered candidate
illustrations plus assembly context -> VLM final candidate ID.**

Show the candidate contacts, complete hand orientation and finger depth, rather
than only a centre dot. Give the VLM assembly intent and the supplied final pose
as context. Do not insert a hand-designed assembly-error score, predictive contact
simulator or goal-based optimizer before its final choice. After selection,
coordinate transforms can determine a nominal assembly hand pose. That calculation
is not grasp ranking or a measurement of movement inside the fingers.

This combines established CAD generation with established VLM selection.
Using aligned CAD candidates and richer assembly context is our proposed
adaptation. Generating a set, or asking a VLM to select from a set, is not by
itself a new contribution. The current comparison can investigate candidate
variety, task suitability through explicitly labeled human review, pickup and
holding outcomes, and time. It does not establish improved assembly precision.

## Study boundaries

Modeled hand/target/table geometry validation is now included by the user's latest
request. Other-part and robot-self collisions, arm reachability, inverse kinematics,
route search, camera layout and new insertion/threading controllers remain outside
this study. Fix irrelevant simulator effects as setup work. The resting-face
exclusion remains a sampling boundary. Keep target-to-finger contact for later
pickup/holding evaluation; do not use actual pickup to filter proposed candidates.

For a later comparison, fix the candidates and closing settings when isolating
VLM selection or assembly context. Record candidate counts per part and how many
parts receive at least one candidate; show the contact regions and approach
directions. Preserve the VLM-selected ID and raw explanation. Record pickup and
holding outcomes using a declared observable criterion, plus total time.
A human's assessment of task suitability is a review, not a measured
assembly position/angle error. Do not require extra cameras, held-part pose
re-estimation or simulator truth to produce an after-execution precision score.
No replacement precision-measurement protocol is assumed. If an assembly is
later attempted, report its observed outcome separately; pickup alone is not
evidence of successful assembly or improved assembly precision.

The supplied current 6D pose is an assumption for this grasp study, not a claim
that the fixed-branch/translation-only identification scorer estimates arbitrary
6D poses. Use system estimates as planning inputs. Do not use live simulator
object state to generate or update a grasp. Keep the active identification
protocol, approved prompt and frozen experiment evidence unchanged.
