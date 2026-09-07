# LIBERO RGB-D bridge

## NIST Task Board #1 custom scene

The custom scene uses the official NIST Task Board #1 STL geometry, a fixed
384 x 384 x 8.992 mm board on 20 mm standoffs, mounted fixtures, and 20 separate
movable components beside the board. It retains the LIBERO floor arena, Panda
arm, agent-view RGB-D camera, and wrist RGB-D camera. This is a perception and
scene-integration baseline, not a validated NIST assembly benchmark.

Download the official archive once from Windows PowerShell at the repository root:

```powershell
New-Item -ItemType Directory -Force outputs/simulation/nist_task_board_1/assets
Invoke-WebRequest -Uri 'https://www.nist.gov/document/taskboard1stlzip' -OutFile 'outputs/simulation/nist_task_board_1/assets/stl.zip' -UserAgent 'Mozilla/5.0'
```

Then run in the existing WSL virtual environment:

```bash
cd /mnt/d/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl ~/.venvs/lerobot-libero/bin/python -m scripts.capture_nist_task_board
```

The script saves a 3840 x 3840 PNG, a preview, and calibrated 768 x 768 RGB-D
observations from both cameras under `outputs/simulation/nist_task_board_1/`.
Mesh conversion, downloaded assets, and provenance also remain under this
ignored output directory. Existing LIBERO tasks and results are unchanged.
`NistTaskBoardEnvironment` exposes the existing sensor/step interface, but has
no official LIBERO initial-state index or assembly success predicate.

The eight pegs, three gears, four nuts, and five male connectors are free bodies.
Their layout is a custom parts arrangement, not NIST's official kit tray.
Board openings are retained using triangular-prism collision geometry; a single
convex plate collision mesh would incorrectly fill them. Other components use
convex collision hulls. The board STL retains its supplied pilot holes, and does
not simulate the drilling/tapping called for in the fabrication instructions.
Connector seating heights are approximate. Thread engagement, gear meshing,
connector latches, cables, and insertion tolerances are not validated. Materials
and component densities are illustrative, not measured physical properties.
The DSUB male mesh is simplified to fit MuJoCo's 200,000-triangle STL limit.

The board changes the support geometry. The basket-task localization crop and
single-table-plane assumptions must be revalidated before running automatic
localization or CAD alignment here. No VLM association, CAD-registration result,
or robot assembly success is claimed by this capture script.

Sources: [NIST Task Board #1 downloads and description](https://www.nist.gov/el/intelligent-systems-division-73500/robotic-grasping-and-manipulation-assembly/assembly),
[official STL archive](https://www.nist.gov/document/taskboard1stlzip), and
[replication instructions](https://www.nist.gov/document/assemblyinstructionsv4docx).
The generated `assets/provenance.json` records the archive checksum, source
units, conversion, component dimensions, and limitations. These third-party
assets are downloaded locally and are not redistributed as repository source.

### Known-hole round-peg pilot

`python -m scripts.run_nist_peg_pilot` runs one episode per method for the
16 mm round peg. The part pose is unknown to both methods; its semantic target
description is given. The hole pose is supplied from the scene design, with its
entrance at world XYZ `(0.074450, -0.004824, 0.028992)` metres and outward axis
`+Z`. The actual arena floor is at world Z=0; the LIBERO sampler's -25 mm offset
must not be used as a floor elevation. The plate underside is at Z=20 mm.

The framework uses RGB-D localization of the parts work area, independent VLM
association, and the selected partial point cloud aligned to the semantic CAD
prior. It then attempts a grasp, lift, transfer, and straight insertion using
robot-proprioception feedback. The 0.75 raw-label-likelihood gate is provisional
and uncalibrated for NIST. A deferral is saved and stops autonomous motion.
An explicitly confirmed candidate can be continued separately using
`run_method("framework_human", new_output_dir, confirmed_trial=(source_dir, object_id))`;
the source and resumed initial-state hashes must match. Human-assisted results
must not be reported as autonomous recognition success.

The VLA branch uses the cached, pinned SmolVLA LIBERO checkpoint without
fine-tuning. It receives agent/wrist RGB, 8D robot state, and the same task text
including numeric hole coordinates. Official LIBERO image rotation, state
conversion, and checkpoint normalization are retained. This checkpoint has no
dedicated goal-pose channel; numeric goal following is unvalidated. Failure can
therefore reflect goal conditioning or control as well as visual recognition.
The framework uses 768-pixel images and VLA uses its native 256-pixel images.
This is a diagnostic pilot, not a controlled ranking of the two methods.

Each timestamped run saves videos, action logs, association evidence, CAD
alignment, and evaluation JSON. Initial simulator states are hash-checked across
methods. Simulator object poses and contacts are used only by `PegEvaluator`
for scoring, never to choose a part or generate an action. Stages are bilateral
target grasp, target lift by at least 15 mm, approach within 10 mm XY of the hole
while grasped, and insertion. Insertion requires at least 5 mm engagement,
positive estimated shaft clearance, less than one degree tilt, and five
consecutive physics/control observations after lifting the correct peg. It is
a simulation geometry criterion, not NIST's official completion protocol.

The 16.2 mm hole and 16 mm peg provide only 0.1 mm nominal radial clearance.
Their original dimensions are retained. Peg slip within the gripper, registration
error, and OSC tracking error can prevent insertion even with an exact hole pose.
The controller performs no force search or visual re-estimation after grasping.
This pilot does not validate the other components' insertion physics.

Run its integration and metric checks inside WSL:

```bash
MUJOCO_GL=egl python -m unittest discover -s tests -p 'test_nist*.py'
```

This bridge targets Ubuntu 22.04 under WSL2. It uses the current LeRobot
`libero` optional dependency to install LIBERO, then calls the direct LIBERO
environment API because the LeRobot Gym wrapper currently does not expose depth
observations.

## Where the simulation runs

LIBERO, MuJoCo, robosuite, Open3D, and the repository's Python pipeline all run
inside Ubuntu under WSL2. The Windows repository path is mounted into that
Linux environment:

```text
Windows: D:\GitHub\A-novel-Robotic-Vision-System-for-Object-Identification
WSL:     /mnt/d/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification
```

MuJoCo renders through WSL GPU passthrough and EGL. `environment.reset()` and
`environment.step(action)` return Python dictionaries directly in the same WSL
process; there is no image-transfer service between Windows and WSL. The sensor
adapter selects the allowed camera and robot fields, converts depth, and passes
NumPy arrays to the local perception pipeline. For the VLM stage only, the
generated contact-sheet image, instruction, and localized candidate metadata
are sent from WSL to the configured cloud API; the returned JSON association is
then consumed locally by the controller.

## WSL and Python environment

Run the Windows command from an Administrator PowerShell, then restart Windows
if requested:

```powershell
wsl --install -d Ubuntu-22.04
```

The current LeRobot source (`0.6.2` at the tested revision) requires Python
3.12. Ubuntu 22.04 ships Python 3.10, so use LeRobot's documented `uv` route
to install a managed Python 3.12 without replacing Ubuntu's system Python:

```bash
sudo apt-get update
sudo apt-get install -y git python3.10-venv build-essential libegl1 libgl1
python3 -m venv ~/.venvs/uv
~/.venvs/uv/bin/python -m pip install --upgrade pip uv
~/.venvs/uv/bin/uv python install 3.12
~/.venvs/uv/bin/uv venv --seed --python 3.12 ~/.venvs/lerobot-libero
source ~/.venvs/lerobot-libero/bin/activate
mkdir -p ~/src
git clone https://github.com/huggingface/lerobot.git ~/src/lerobot
cd ~/src/lerobot

# Work around the current EGL-probe build ordering and CMake-4 policy issues.
python -m pip install "cmake>=3.29.0.1,<4.2.0" "setuptools<82" wheel
export CMAKE_POLICY_VERSION_MINIMUM=3.5
python -m pip install --no-build-isolation "egl-probe==1.0.2" "hf-egl-probe==1.0.2"
unset CMAKE_POLICY_VERSION_MINIMUM

python -m pip install -e ".[libero,smolvla]"
python -m pip install open3d openai
python -m pip check
```

If `~/src/lerobot` already exists, do not clone it again. Inspect its status and
update it deliberately before running the editable installation.

Verify the simulator imports independently:

```bash
python -c "import mujoco; print('mujoco', mujoco.__version__)"
python -c "import robosuite; print('robosuite', robosuite.__version__)"
python -c "import libero; print('libero', libero.__file__)"
```

On its first import, LIBERO asks where its dataset folder should live. To use
the packaged default paths noninteractively and create `~/.libero/config.yaml`:

```bash
printf "n\n" | python -c "import libero.libero"
```

The first environment creation downloads the current LIBERO simulator assets
to `~/.cache/libero/assets`. This is separate from benchmark demonstration
datasets and from this repository.

### Dependency notes from the tested WSL installation

- Ubuntu 22.04's Python 3.10 cannot install current LeRobot, which requires
  Python 3.12 or newer.
- `egl-probe` and `hf-egl-probe` are source distributions. They require a C/C++
  compiler, and pip otherwise attempts to build them before its declared CMake
  dependency is usable.
- `egl-probe 1.0.2` declares a legacy CMake policy level rejected by CMake 4;
  `CMAKE_POLICY_VERSION_MINIMUM=3.5` is scoped only to building those probes.
- Open3D is not part of LeRobot's `libero` extra. `open3d 0.19.0` installed
  alongside LeRobot's NumPy 2.2.6 without broken requirements.

## Run from the repository root

Generated artifacts are separated by backend:

```text
outputs/physical/     RealSense and physical-robot pipeline results
outputs/simulation/   LIBERO captures, localization, point clouds, and episodes
```

```bash
cd /mnt/d/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
python -m scripts.libero_smoke_test
python -m scripts.capture_libero_rgbd
python -m scripts.libero_pointcloud_test
python -m scripts.libero_localization_test
python -m scripts.libero_milk_cad_test
python -m scripts.setup_libero_experiment
python -m scripts.libero_task_execution
python -m scripts.libero_vla_eval
python -m scripts.compare_libero_results
```

If EGL cannot create an offscreen renderer, verify that `nvidia-smi` works
inside WSL. For a CPU-rendered fallback:

```bash
sudo apt-get install -y libosmesa6
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
python -m scripts.libero_smoke_test
```

The scripts also probe EGL and OSMesa in that order when `MUJOCO_GL` is unset.
They never fall back to an interactive GUI renderer silently.

Use the isolated WSL virtual environment. The existing Windows interpreter has
NumPy 2.3.5, while the current LeRobot source declares `numpy>=2.0,<2.3`; adding
LIBERO to that Windows environment would both violate the supported platform
and force a dependency downgrade.

## Coordinate and image conventions

The raw LIBERO camera arrays use the OpenGL bottom-left origin. The sensor
bridge flips the rows of both RGB and depth once, producing conventional
top-left-origin images. It does not apply the additional horizontal flip used
by some VLA training pipelines.

`robosuite.utils.camera_utils.get_camera_extrinsic_matrix` returns the camera
pose in the world frame after applying robosuite's OpenCV camera-axis
correction. This bridge names it `world_T_camera`; it maps homogeneous points
from the camera frame into the MuJoCo world frame. `camera_T_world` is its
numerical inverse.

Depth is converted with `robosuite.utils.camera_utils.get_real_depth_map`. The
saved `depth.npy` is `float32` in metres. The point cloud is reconstructed only
from that rendered metric depth, RGB, and the retrieved intrinsic matrix.

## Existing localization compatibility

`scripts.libero_localization_test` passes the simulated capture into
`PointCloudLocalization.run_from_arrays`. Before that shared algorithm runs,
the simulation adapter masks depth to a calibrated MuJoCo-world tabletop box:

```text
minimum XYZ: (-0.25, -0.35, -0.02) m
maximum XYZ: ( 0.35,  0.35,  0.23) m
```

This mask uses only metric depth, camera intrinsics, and `world_T_camera`; it
does not use simulator segmentation, object IDs, or object poses. The LIBERO
test uses a 2 mm voxel and plane tolerance plus smaller cluster, ROI, and raw
point-cloud cleanup cutoffs appropriate for its clean 256 x 256 rendered
depth. RealSense defaults remain unchanged.

Outputs are saved beneath `outputs/simulation/libero_localization/`. The localization JSON
contains camera-frame ROI, centroid, axis-aligned size, accepted downsampled
cluster point count, cleaned saved PLY point count, and cluster path for each
candidate. The test also prints each centroid transformed into the MuJoCo world
frame for calibration verification.

## Milk CAD-to-observation experiment

`scripts.libero_milk_cad_test` isolates the first CAD experiment from robot
motion. It reads the saved initial observation, depth-localization result, VLM
semantic association result, and `world_T_camera` from task 7. It then:

```text
VLM association: "milk" -> final_object_id
  -> that candidate's RGB-D-derived partial point cloud
  -> semantic lookup in CAD/libero_object_library.json
  -> tabletop-constrained CAD-to-observation registration
  -> camera_T_cad
  -> world_T_cad = world_T_camera @ camera_T_cad
```

The registered object center is the CAD bounding-box center transformed by
`world_T_cad`; it is not assumed to be the CAD file origin. Results are saved
beneath
`outputs/simulation/experiments/put_all_objects_into_basket/proposed_framework/task_07_milk/init_state_00/cad_registration/milk/`,
including the sampled, aligned, and augmented point clouds, candidate scores,
transforms, provenance, RMSE, and `alignment_views.png`.

Before the pose is allowed into task execution, the adapter requires a
camera-frame localization result, a finite table plane, a checksum-verified
milk asset, no registration warnings, and constrained RMSE at or below 15 mm.
Failure aborts the task instead of silently reverting to the partial-cloud
centroid.

The experiment intentionally uses LIBERO's exact HOPE visual mesh for each of
the ten target products as a known CAD prior. Every product has its own
cataloged source scale, support axis, and SHA-256. The meshes are not copied
into this repository; they are resolved from the installed
`~/.cache/libero/assets` directory. Results are labeled `exact simulator CAD
prior` rather than being presented as independently retrieved real-world
models. See `CAD/LIBERO_ASSET_NOTICE.md` for provenance and license information.

Neither the isolated registration test nor the full task pipeline reads a
MuJoCo object name, object ID, segmentation mask, or ground-truth object pose.
The simulator contributes only rendered RGB-D, camera calibration, the language
instruction, and robot proprioception. The known mesh is an experimental CAD
prior, just as a physical deployment would obtain a model from its CAD library.

## Perception-driven task execution

`scripts.libero_task_execution [task_index]` runs one LIBERO-Object basket task;
omitting the index retains task 7 (milk) as the backward-compatible default. It
uses the rendered agent-view RGB-D observation, the calibration bridge, the
existing depth localizer, the repository's OpenAI VLM
association, target CAD-to-observation alignment, an LLM plan, robot proprioception,
and normalized OSC pose actions. The VLM receives the original task instruction
and a full-scene plus enlarged-crop contact sheet marked with candidate letters.
It returns one `letter: object name` entry per mentioned object using the joint
prompt from `scripts.run_vlm_multi_object_pilot`. No pre-extracted target list,
candidate metadata, or task roles are supplied to the VLM. Letter-to-object-ID
mapping and output-completeness checks happen locally. The LLM infers task roles
from the original instruction and returns object IDs in its action sequence. The target object
pose comes from CAD registration. A simulation-only grasp conversion then
handles the CAD's declared Y-up or Z-up convention, keeps the gripper tool axis
vertical, selects the closer 180-degree-equivalent wrist yaw, and chooses a
grasp height from the registered CAD extent. Flat packages use their center,
medium-height products use a small upward offset, and tall products are grasped
40 mm below the top. The basket place position remains its depth-localized
centroid.
The controlled simulation condition assumes a correct human response whenever
the confidence gate defers. `ASSUME_CORRECT_HUMAN = True` resolves those cases
using simulator identity matched to an unambiguous localized candidate. This is
an explicitly logged substitute for a human response, not an actual human trial.
Accepted VLM decisions are not corrected. Missing or ambiguous localized targets
still fail. Only an object ID is returned; CAD registration, grasp generation,
and execution continue using estimated geometry. The run records whether this
identity assistance was used. Simulator segmentation and ground-truth execution
poses are not supplied to the pipeline. LIBERO's task success predicate is checked during
execution for evaluation and episode termination; it is not a planner input.

Before each RGB-D frame is captured, the environment advances ten zero-pose,
open-gripper physics steps (0.5 seconds at 20 Hz). This matches LeRobot's
settling sequence and lets MuJoCo resolve initial support contacts. The CAD pose
treats the segmented table plane as a hard support constraint and optimizes
only translation along the plane plus yaw.

Set `OPENAI_API_KEY` in the WSL process environment before running the task.

Each episode saves its perception inputs, localization JSON, enlarged VLM
visual prompt, joint VLM request and full provider response/token usage,
resolved associations, any assumed-human identity audit, CAD transforms and aligned/augmented point
clouds, normalized action log, final agent and wrist RGB-D observations,
success value, and MP4 video beneath its corresponding
task/state folder. It uses bounded top-grasp/container-placement skills, not
a general grasp planner or VLA policy.

The current higher-resolution trial renders the proposed method at 768 x 768,
uses 448-pixel VLM contact-sheet tiles, and queries all mentioned objects jointly.
Its artifacts are isolated under
`proposed_framework/_resolution_trials/768x768_joint_instruction_assumed_human_v2/`.
The old one-target `768x768_raw_likelihood_gate_llm_plan_v1` pilot remains intact.
The CAD catalog now declares BBQ sauce Z-up and alphabet soup/tomato sauce Y-up,
matching their source mesh frames. These corrections were checked against frozen
observations before the new pilot. This simulation-only controller maps the registered
`world_T_cad` pose to a collision-aware robosuite grip-site target and uses its
OSC pose controller; it does not alter the physical RealSense/UR5e path.

### Executable LLM plans and failure evidence

The runner now calls `simulation.libero_planning.generate_simulation_plan` after
CAD alignment. It reuses the OpenAI client in `LLM_planner.py`, with
`OPENAI_LLM_MODEL` (default `gpt-4.1-mini`) and temperature 0. Set this to a fixed
supported model snapshot for an experiment. The joint VLM defaults to
`gpt-4.1-mini-2025-04-14`. The physical one-target VLM path and RTDE action schema are
not used by the simulation.

The planner receives the original task text and resolved objects with descriptions,
estimated centroids, CAD transforms/dimensions where available, grasp bindings,
placement capabilities, and robot proprioception. All geometry uses meters in
explicit frames. It receives no expected plan or source/destination role labels.
The supported structured actions are `pick(object_id)` and
`place(object_id, destination_id, relation="in")`. Numeric motion parameters are
bound by the executor from perception, never generated by the LLM. Only the
existing ten LIBERO-Object basket tasks are connected; plate placement and the
proposed broader task selection still require separate integration. Only the
associated product currently has a CAD-derived grasp binding, so this is a
limited instruction-to-skill evaluation, not an unconstrained planning benchmark.

The complete plan is checked before any task motion: known IDs, exact arguments,
supported capabilities, one held object, placement after picking, and no reuse
of an object's stale initial pose after moving it. Rejected plans, blocked plans,
refusals, incomplete responses, and API failures stop the episode without a
scripted fallback. A controller exception remains an execution error. JSON
schema compliance alone is not evidence that a plan satisfies the task.

Each run saves `llm_plan.json` with the prompt, context, model settings, raw provider
response/token usage, parsed plan, and validation outcome. `execution_inputs.json`
freezes the perception values and controller/environment configuration.
`episode.json` includes planning status, success, and action logs linked to plan
action indices. Task consistency is graded against the instruction's object
relationships only in the evaluation layer; this grade never guides execution.
It is conditional on correct upstream identities and geometry, and does not
automatically assign a causal failure stage. Pre-planning exceptions record
`failure_observed_at` while leaving causal attribution unresolved.

Run a selected initial arrangement programmatically inside WSL:

```python
from scripts.libero_task_execution import main
episode_path = main(7, initial_state_index=4)
```

The existing `python -m scripts.libero_task_execution 7` command uses state 0.
It exits unsuccessfully when the task fails; the Python function returns the
episode path for completed attempts, including planning stops. Existing output
directories cannot be overwritten.

For an unsuccessful episode that reached planning, make an explicit diagnostic
replay in a new directory:

```python
from scripts.libero_task_execution import replay_with_reference_plan
episode_path = replay_with_reference_plan(
    source_dir="outputs/simulation/experiments/put_all_objects_into_basket/proposed_framework/"
               "_resolution_trials/768x768_joint_instruction_assumed_human_v2/task_07_milk/init_state_00",
    output_root="outputs/simulation/planning_diagnostics/milk_state00_reference",
)
```

Replay uses the same frozen perception values, initial state, settling actions,
controller and executor source hashes, library versions, and episode budget.
It verifies both the official initial-state hash and the full simulator-state
hash after settling. State values used for this integrity check are never
exposed to the planner. No VLM, CAD, or LLM calls are repeated. Only the plan is
replaced with a checked pick/place reference using the same resolved IDs.
The original episode remains unchanged. `reference_comparison.json` records the
paired outcomes; these diagnostic executions are excluded from the main trial
count. Reference success supports a planning contribution only when the
executable plan changed, subject to upstream correctness. The original planning
status distinguishes rejected model output from blocked plans or service failures.
Two failures, or different outcomes from identical executable plans, leave the
cause unresolved.

Each joint VLM entry contains a decision letter mapped deterministically back to
an `object_id`. Its gate score is
`exp(sum(decision-bearing token logprobs))`, without counting standalone
formatting tokens. It is a raw generated-label likelihood, not a calibrated
probability of correct object identity. Whole-response and letter-plus-name
likelihoods are saved only as diagnostics. The inherited threshold
`0.9999832372181827` is provisional and has not been validated for the joint
prompt; later entry probabilities also depend on earlier generated output.
Malformed/incomplete output or missing decision-letter log probabilities defers
to the assumed human. Both paths produce the same `final_object_id` field before CAD
retrieval. A `none` decision leaves `final_object_id` null and stops CAD
retrieval for that target.

The physical single-target VLM retains its existing prompt and threshold settings.

## SmolVLA baseline and matched comparison

The VLA is a parallel baseline, not another stage after CAD registration:

```text
outputs/simulation/experiments/put_all_objects_into_basket/
├── experiment_manifest.json
├── comparison_summary.json             # after both matched methods run
├── per_episode.csv                      # after both matched methods run
├── VLA/
│   ├── task_00_alphabet_soup/init_state_00/...
│   ├── ...
│   └── task_09_orange_juice/init_state_00/...
└── proposed_framework/
    ├── task_00_alphabet_soup/init_state_00/...
    ├── ...
    └── task_09_orange_juice/init_state_00/...
```

Run `python -m scripts.setup_libero_experiment` once to create the two method
folders and the ten actual LIBERO-Object task folders. Generated artifacts stay
under `outputs/` and are intentionally ignored by Git.

```text
Same LIBERO task + same official fixed state 0
├── Proposed method
│   agent RGB-D -> depth localization -> VLM association -> target CAD alignment
│   -> task-specific scripted OSC controller -> 7D actions
└── VLA baseline
    agent RGB + wrist RGB + 8D robot state + instruction
    -> SmolVLA -> 7D actions
                         |
                         v
              same LIBERO success predicate
```

`scripts.libero_vla_eval` delegates policy loading, image orientation,
normalization, state construction, action unnormalization, and rollout to the
official LeRobot evaluator. It intentionally does not import the RGB-D sensor,
VLM, CAD, or scripted controller. The selected
`HuggingFaceVLA/smolvla_libero` checkpoint is already LIBERO-fine-tuned; no
local fine-tuning is needed for this benchmark baseline. The exact tested
checkpoint revision is
`6721902bc4d61e50a3bfdb11dfb4cb626f05d102`.

The all-product breadth test evaluates task indices 0 through 9 once per method.
A **task index** selects the product instruction: task 0 is alphabet soup, task
7 is milk, and task 9 is orange juice. An **initial-state index** selects one of
the 50 saved simulator arrangements inside that task. These are independent
indices. The completed VLA baseline remains at its official 256 x 256 input
setting. The completed proposed-method breadth sweep uses 512 x 512, while the
current Alphabet Soup resolution trial uses 768 x 768. Neither proposed result
is a resolution-matched cross-method comparison with the VLA baseline. All runs
still use:

- official `.pruned_init` state index 0, including a byte hash check
- seed 1000
- relative OSC control at 20 Hz
- 10 pre-policy physics-settling actions
- a 280-action episode horizon
- LIBERO's own `check_success()` task predicate

Run the VLA branch inside the WSL environment:

```bash
source ~/.venvs/lerobot-libero/bin/activate
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
python -m scripts.libero_vla_eval
```

The VLA runner preserves every complete task package, including the existing
milk pilot, and evaluates all missing task indices in one official LeRobot
batch. It preserves the combined batch under `VLA/_batch_runs/` and writes a
separate manifest, `eval_info.json`, and copied video beneath each task's
`init_state_00/` directory. A nonempty but incomplete task directory is treated
as an error and is never overwritten silently.

Run the proposed branch for each task index from the same repository and WSL
environment. The default remains task 7 for backward compatibility:

```bash
python -m scripts.libero_task_execution 0
python -m scripts.libero_task_execution 1
# continue through task index 9
```

This branch sends the generated letter-marked contact sheet and task instruction
to the configured OpenAI VLM. The VLA branch makes no cloud API call. Obtain
authorization for the intended cloud evaluation scope before starting it;
existing user authorization for that scope covers the batch. Only run the cross-method comparator when both
branches use the same image resolution:

```bash
python -m scripts.compare_libero_results
```

The comparison script writes `comparison_summary.json` and `per_episode.csv`
beneath
`outputs/simulation/experiments/put_all_objects_into_basket/`, but only after
both methods contain exactly tasks 0 through 9 and each pair matches on the
instruction, seed, byte-hashed state 0, settling, horizon, control
mode/frequency, resolution, and success predicate. The primary metric is binary
task success rate across the ten products. Action count and execution speed are
not used to rank the methods. Because there is only one arrangement per product,
this is a breadth test rather than a within-task robustness benchmark; a later
study should repeat every product over multiple matched fixed states.
