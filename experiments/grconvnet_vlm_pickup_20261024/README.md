# Pretrained GR-ConvNet and VLM pickup comparison

The user authorized this experiment on 2026-10-24: run pretrained GR-ConvNet on
the same 14 parts without retraining, let a VLM order pickup grasps, and compare
pickup success and total time with the geometric method. Final assembly views
and goal transforms are excluded from generation, VLM inputs and movement.

## Completed result

The [completed run and per-part report](../../outputs/grconvnet_vlm_pickup_20261024/runs/20261024T135155Z/report.md)
used all 14 parts in the prepared older `objects_centered` simulation scene.
The [14-part gallery](../../outputs/grconvnet_vlm_pickup_20261024/runs/20261024T135155Z/gallery/index.html)
shows the exact pickup images supplied to the VLM, its raw observations and
ordering, the selected poses, and actual closed-finger and lifted-part frames.
There are no final assembly views.

| Method | Parts picked up and held | Actual pickup attempts | Decision time, all 14 | Total recorded time, all 14 |
| --- | ---: | ---: | ---: | ---: |
| Geometric small search | 14/14 | 14 | 21.37 s | 115.24 s |
| Pretrained GR-ConvNet, network-score order | 11/14 | 11 | 21.27 s | 98.18 s |
| Pretrained GR-ConvNet, VLM order | 11/14 | 11 | 62.61 s | 138.46 s |

Every attempted pickup passed the same lift-and-hold check. The three learned
cases without an attempt were rejected before the robot moved:

- **Large gear:** all four candidate approaches would put the open fingers
  through the gear by 4.144–8.327 mm. The approach check allows 0.050 mm.
- **Medium gear:** both candidate approaches would put the open fingers through
  the gear by 0.356–3.502 mm, above the same limit.
- **M16 nut:** its one network proposal sampled a visible nut point only
  2.447 mm above the table. The host's half-height conversion placed the grasp
  centre 1.224 mm above the table. Even with the highest +8 mm finger setting,
  a finger extended 0.026 mm below the table, missing the required 0.100 mm
  clearance by 0.126 mm. All six settings were rejected. This is a limitation
  of this proposal and height conversion, not evidence that the nut has no
  usable grasp. The [saved-depth diagnostic](../../outputs/grconvnet_vlm_pickup_20261024/runs/20261024T135155Z/static_failure_diagnostics.json)
  records the calculation without rerunning or changing the benchmark.

The 8 mm round pin passed with all three methods. The VLM returned 13 valid
orders; M16 had no surviving pose to show. It changed the candidate order on
five parts, changed the pose actually used on two, and avoided two rejected
robot-path checks. It rescued zero pickups and introduced zero pickup failures
relative to the same network candidates in score order.

Decision time includes fresh RGB-D capture, generation, VLM review when used,
and robot-path checks. Total recorded time also includes scene setup, actual
contact simulation and evidence saving. GR-ConvNet's lower score-order total
includes three cases without a pickup attempt, so it does not establish faster
successful pickup. The first VLM call includes 9.45 seconds of cold model
loading. The one-time 0.13-second GR-ConvNet load is reported separately.
These are computer wall times in simulation, not physical robot cycle times.

The [independent audit](../../outputs/grconvnet_vlm_pickup_20261024/runs/20261024T135155Z/independent_audit.json)
passed all 42 records, confirmed identical learned candidate pools before
ordering, and verified 394 protected historical files unchanged. Six focused
adapter, camera and earlier geometric-search checks passed before the run.
This single-scene trial uses supplied identity, a perfect visible target mask,
and correct CAD alignment for the geometric method. It does not establish
performance on new arrangements, real camera depth or assembly.

## Frozen protocol

The three conditions are:

- **Geometric:** the previous small CAD search, 512 rays, at most 24 contact
  pairs, six finger depths, and a shortlist of at most 12 poses.
- **GR-ConvNet score:** pretrained proposals checked in network-score order.
- **GR-ConvNet + VLM:** the same learned proposals ordered by local
  `qwen3.5:9b`, with fixed weights and no new training.

The last condition is an adaptation of OWG's candidate-image review. It is not
an exact reproduction of its published robot experiments. The model is the
official Cornell RGB-D checkpoint referenced by OWG's released demo; the paper
instead reports Jacquard-pretrained weights. Exact sources, checksums, license
and the safe tensor-only conversion are in [vendor/provenance.json](vendor/provenance.json).
Dependencies missing from the existing runtime are stored only in this
experiment's output dependency directory. No global environment is changed.

The frozen input preprocessing uses 224 pixels, a mask-derived square crop
twice the target bounding-box side with a 128-pixel minimum, and the released
OWG depth normalization followed by the author's `CameraData` processing.
Metric rendered depth replaces the demo's graphics-buffer depth and known
camera calibration replaces its hardcoded pixel/metre conversion. The model,
crop, depth normalization and four-proposal limit are fixed before real-part
prediction or pickup outcomes. Native postprocessing and peak detection are
retained. [Adapter](grconvnet_adapter.py), [frozen adapter settings](vendor/adapter_protocol.json).

Every condition captures fresh calibrated RGB-D and a perfect visible target
mask in the prepared older `objects_centered` scene. Target identity is supplied
and classification is not tested. The camera is fixed independently of the
target CAD pose. The geometric condition also receives the assumed correct CAD
alignment. Learned grasp generation uses RGB-D, mask and camera/table calibration
only; it does not receive CAD identity geometry or pose. Simulator part poses
remain available for evaluation and the common held-target collision check.

For each learned proposal, depth gives its local visible surface height. The
host assumes a table-supported object and places the closing line halfway
between the table and that height. The approach is vertically downward. Of two
exchanged-jaw orientations, choose the one nearest the initial robot hand
orientation. Try the common finger depths +4, 0, -4, -8, +6 and +8 mm and retain
the first that clears the table. The last two preserve the earlier low-pin
option. This is a generic Panda conversion, not a trained depth prediction or
a CAD-specific adjustment. No deeper variant is tried after VLM selection.
Model jaw endpoints are proposed finger positions, not verified CAD contacts.

The VLM sees at most four shuffled, numbered pickup drawings on the actual RGB
scene. Red joins the jaws; blue represents the open fingers. Tool height is
shown in millimetres. No score, selected-grasp badge, prior path result, pickup
outcome, or assembly destination is supplied. Other loose-product contacts are
explicitly ignored. The [new grasp-review prompt](grasp_rank_prompt.txt) is
separate from the protected CAD-description prompt. Responses are saved raw;
the JSON parser requires each supplied ID once and performs no `eval` or
correctness-driven re-prompt. A single candidate still receives a model call;
no candidate means an explicit unavailable ranking.

Both learned conditions capture and infer independently. Identical inputs and
frozen inference must yield the same pool before ordering. All arms use the
same existing contact-driven Panda controller, numerical pickup checks and
physical geometry. The other 13 loose parts stay visible and noncolliding;
their recorded one-physics-step reset/gravity offset does not accumulate.
Each arm starts fresh and makes at most one physical pickup attempt. A pass
requires at least 120 mm lift after a one-second hold, both fingers touching,
and recorded robot overlap with itself, the table and fixed board hardware
below 1 mm. The commanded lift is 180 mm. No attachment holds the part.

Time is recorded for scene setup, capture, preprocessing, inference, gripper
position construction, input-image creation, VLM request, robot movement checks,
actual simulated pickup and evidence saving. Decision time includes capture
and file overhead through the selected path. Total time includes setup and
the executed simulation, including failures. The one-time GR-ConvNet load is
reported separately; the first VLM request includes its actual cold load.
Conditions run serially with rotating order, one observation per part/condition.
All 14 parts remain in each denominator, including empty masks, zero proposals,
invalid VLM responses, rejected robot paths and failed physical grasps. Shorter
failed trials must not be presented as faster successful pickups.

Two focused synthetic adapter tests and two calibrated-capture tests verify
native preprocessing/inference and camera mapping before the benchmark. No
training, paid API, model-weight update, physical robot, shared perception
change or modification of frozen earlier results is part of this experiment.
