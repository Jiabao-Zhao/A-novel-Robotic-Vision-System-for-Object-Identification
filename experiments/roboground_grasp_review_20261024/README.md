# RoboGround review and a VLM grasp reviewer for our CAD pipeline

The user confirmed **RoboGround: Robotic Manipulation with Grounded
Vision-Language Priors** on 2026-10-24. The earlier QuickGrasp identification was
incorrect. QuickGrasp remains a different analytical method in our comparison.

RoboGround sends an image and instruction to a VLM, which marks the target and
placement area. A trained policy then receives images, masks and robot state
and produces movements. The banana figures show four views for attribute
descriptions. They do not show proposed grasps for VLM ranking. The method
fine-tunes its VLM and trains its policy; its limitations suggest adding
AnyGrasp in future. See [Figure 3 and Section 4.4](https://arxiv.org/html/2504.21530v1#S4),
[banana Figures 6–7, PDF pages 14–15](https://arxiv.org/pdf/2504.21530#page=14),
and [Appendix D](https://arxiv.org/html/2504.21530v1#A4).

The [pinned released-code audit](code_audit.json) separately verifies the data
flow. In [GR1's rollout](https://github.com/ZzZZCHS/GR1/blob/16b63e577e09d1c2025d996d6d891003e790882d/utils/eval_utils_robocasa.py#L357-L381),
the grounding model receives an instruction and camera image and returns masks.
The [policy wrapper](https://github.com/ZzZZCHS/GR1/blob/16b63e577e09d1c2025d996d6d891003e790882d/utils/eval_utils_robocasa.py#L234-L242)
receives robot state as well as images, masks and text. Those state values are
the robot's current pose, not a list of candidate CAD grasps for the VLM.
The [GPT data-generation prompt](https://github.com/ZzZZCHS/RoboGround/blob/25a862634398bc0407bc7b36b5d8b7c61d7a502b/data_gen/gpt/generate_common_instructions.py#L95-L111)
requests instructions from scene views and colored target/destination masks.

## A released example of the proposed grasp-ranking idea

**[Towards Open-World Grasping with Large Vision-Language Models (OWG)](https://arxiv.org/html/2406.18722v3#S3)**
does show generated grasp candidates to a VLM. This is a separate reference,
not another guess about the paper the user meant.

The [official prompt](https://github.com/gtziafas/OWG/blob/main/prompts/grasp_ranking.txt)
shows a numbered grasp in each subplot. Colored lines depict the fingers and
gripper direction. The VLM ranks IDs using likely contact with nearby objects
and closeness to the object's center. Its
[visual prompt code](https://github.com/gtziafas/OWG/blob/main/owg/visual_prompt.py#L361-L491)
also has an optional gripper-mesh rendering path. That path starts from position
and yaw with the other angles fixed; it does not establish arbitrary-6D results.

The released [demo configuration](https://github.com/gtziafas/OWG/blob/main/config/pyb/env.yaml)
requests four candidates but disables the ranker by default. Its
[grasp configuration](https://github.com/gtziafas/OWG/blob/main/config/pyb/OWG.yaml)
also disables the mesh-rendering option. These are configurable implementation
settings, not a claim that the paper always uses four grasps. OWG uses a learned
proposal network; only its prompted VLM selection avoids task-specific training.

## Proposed use with our existing geometry

**2026-10-24 scope correction:** the user rejected the earlier step that showed
grasps at the final assembly pose. The VLM review below uses current pickup
views only. Final assembly views and goal transforms are excluded from its
inputs. The earlier destination-view proposal is withdrawn.

Keep the training-free CAD contact generator and small search. Use a fixed,
already-pretrained VLM only to order supplied candidates. Do not ask it to
invent millimetre coordinates or certify contact forces.

1. Generate grasps at the aligned CAD's current pose. Keep the full-finger +6
   and +8 mm options that worked for the low pin. Reject numerical gripper/table
   overlap and unintended open-hand overlap with the target. Other loose-part
   collisions remain excluded under the requested simplification.
2. Present a few meaningfully different candidates, each with a neutral ID,
   full gripper geometry and finger-touch locations. Show two fixed camera
   directions at the same scale. Jaw-swapped duplicates do not create a new
   contact location; the saved 8 mm pin's two alternatives share one pair.
3. Ask for candidate IDs in order, short visible observations, and an explicit
   unavailable/clarification result when the necessary view is missing. Keep
   all positions and orientations unchanged. Parse JSON and validate IDs;
   do not import the upstream `eval`-based response parser.
4. Check actual robot joint limits and the pickup movement numerically.
   Include the table and any fixed hardware encountered consistently in planning and
   physics. An attractive picture does not establish that the robot can make
   that movement. A VLM preference cannot override a failed numerical check.

The VLM considers the current part, fingers and approach. Actual robot movement
and contact checks determine pickup feasibility. All lengths stay in metres;
the existing world, native CAD and Panda grip-site frames are preserved.
The shortlist must contain useful alternatives: if none appears suitable, the
VLM should report that rather than inventing a missing grasp.

The small-search baseline already picked up 14/14 parts in one known-pose scene.
A comparison can measure changed choices, pickup failures on additional initial
poses, and added time. It must not assume that VLM ranking improves success.

## How OWG generates grasps and whether it needs object pose estimation

OWG uses a pretrained **GR-ConvNet** grasp network. Its paper supplies RGB-D and
the target mask, predicts image maps of grasp quality, finger angle and opening,
then forms grasp candidates and asks the VLM to order them. It does not require
a CAD model aligned to the object. See [Sections 3.1–3.2](https://arxiv.org/html/2406.18722v3#S3)
and [the released generator](https://github.com/gtziafas/OWG/blob/main/third_party/grconvnet/grconvnet.py).

The output is the gripper's position and one turning angle, with a fixed approach
direction. This is four degrees of freedom, not the object's full position and
orientation. Depth plus camera-to-robot calibration puts the predicted grasp in
robot coordinates. This conversion is different from estimating a CAD object's
six-dimensional pose. The released demo uses fixed camera scale/background
constants; those are not a substitute for our own camera calibration.

The paper reports Jacquard-pretrained weights; the released PyBullet generator's
default checkpoint name instead identifies a Cornell RGB-D model. Neither
requires training on each new object to run inference, but both are learned
grasp generation. Our geometric sampler supplies an alternative without a
trained grasp network. The candidate-image VLM ranking can be considered
separately from which generator produced the candidates.

## Prepared material and a fair next comparison

The [neutral input example](../../outputs/roboground_grasp_review_20261024/neutral_input_example/index.html)
illustrates how to show candidates without revealing saved outcomes. No VLM is
called and no robot or contact simulation is run to create it. Its initial-grasp
views follow the current pickup-only VLM input scope. The
[builder](build_neutral_example.py) records its exact
source images/poses, display-ID mapping and hashes in the new output folder.

Do not feed the old gallery tiles to a selector: their green borders and
pickup/path labels reveal the previous answer. In a controlled comparison,
freeze the candidate pool and new grasp-review prompt before inference, randomize
display IDs, and retain raw responses. Compare geometric ordering with VLM
ordering on the same candidates and the same numerical movement checks. Log
generation, rendering, VLM, checking and total time separately. VLM latency has
not been measured here.

This can avoid new training by using our geometric proposals and a frozen local
VLM. That is a proposed adaptation of candidate-image ranking, not a RoboGround
reproduction or a demonstrated assembly method. The existing CAD-description
prompt, perception scoring, saved trials and runtime have not been changed.
