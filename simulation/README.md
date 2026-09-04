# LIBERO RGB-D bridge

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
python -m pip install open3d google-genai openai
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
existing depth localizer, the repository's Gemini-first/OpenAI-fallback VLM
association, target CAD-to-observation alignment, robot proprioception, and
normalized OSC pose actions. The VLM receives a full-scene plus enlarged-crop
contact sheet labeled only with localized `object_id` values. It independently
associates the semantic descriptions for the product and basket; it does not
infer which one is moved or used as a destination. The task-execution layer
retains those relationships from the LIBERO task definition. The target object
pose comes from CAD registration. A simulation-only grasp conversion then
handles the CAD's declared Y-up or Z-up convention, keeps the gripper tool axis
vertical, selects the closer 180-degree-equivalent wrist yaw, and chooses a
grasp height from the registered CAD extent. Flat packages use their center,
medium-height products use a small upward offset, and tall products are grasped
40 mm below the top. The basket place position remains its depth-localized
centroid.
Simulator segmentation, simulator object identities, and ground-truth object
poses are not method inputs. LIBERO's task success predicate is read only after
execution as the evaluation result.

Before each RGB-D frame is captured, the environment advances ten zero-pose,
open-gripper physics steps (0.5 seconds at 20 Hz). This matches LeRobot's
settling sequence and lets MuJoCo resolve initial support contacts. The CAD pose
treats the segmented table plane as a hard support constraint and optimizes
only translation along the plane plus yaw.

Set `GEMINI_API_KEY` and/or `OPENAI_API_KEY` in the WSL process environment
before running the task.

Each episode saves its perception inputs, localization JSON, enlarged VLM
visual prompt, independent semantic association results, CAD transforms and aligned/augmented point
clouds, normalized action log, final agent and wrist RGB-D observations,
success value, and MP4 video beneath its corresponding
`proposed_framework/task_XX_product/init_state_00/` folder. This remains a
top-grasp execution baseline, not a general grasp planner or VLA policy.

The current higher-resolution trial renders the proposed method at 768 x 768,
uses 448-pixel VLM contact-sheet tiles, and queries one explicit semantic target
description at a time. Its artifacts are isolated under
`proposed_framework/_resolution_trials/768x768_semantic_association_grasp_pose_v5/`
so the completed 512 x 512 breadth sweep and the earlier XYZ-controller trial
are not overwritten. This simulation-only controller maps the registered
`world_T_cad` pose to a collision-aware robosuite grip-site target and uses its
OSC pose controller; it does not alter the physical RealSense/UR5e path.

The VLM response is a compact choice label mapped deterministically back to an
`object_id`. When provider top-logprob data contains every valid choice label,
the association score is normalized over those labels. Otherwise it falls back
to `exp(sum(decision-bearing token logprobs))`, without counting standalone
formatting tokens. The saved diagnostics identify the score type. Neither score
is a calibrated correctness probability. The provisional threshold of 0.75
controls autonomous acceptance versus human clarification and must be calibrated
on held-out experiments. Both paths produce the same `final_object_id` field
before CAD retrieval. A `none` decision leaves `final_object_id` null and stops
CAD retrieval for that target.

With the locally tested Google Gen AI 2.21.0 Developer API,
`gemini-2.5-flash` rejects `response_logprobs` as not enabled. The Gemini
provider therefore preserves its label with a null score and defers to a human;
it never fabricates confidence. The locally tested OpenAI 3.7.0 Chat
Completions client and official API schema expose up to 20 top-token alternatives
per output position. Candidate-normalized scores are emitted only when those
alternatives cover every valid localized-object label plus `N`.

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

This branch sends the generated contact-sheet image and localized candidate
metadata to the configured Gemini or OpenAI VLM. The VLA branch makes no cloud
API call. Each new proposed-method image payload requires explicit approval
before it is sent to an external VLM; do not treat the ten-task proposed sweep
as an unattended cloud job. Only run the cross-method comparator when both
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
