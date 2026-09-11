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

### Wrist-only scattered-object pickup

`python -m scripts.run_wrist_cluster` runs Try 2: one seeded scatter of all 14
active movable targets and one independent pickup attempt for each target.
Use `main(trial_number=N)` for a specific try from 1 through 10. Each try uses
seed `20260908 + N`, changing every object's XY position and yaw while retaining
its supported resting orientation. The same saved simulator state and gripper
command are restored before each attempt within a try. Only
`robot0_eye_in_hand` RGB and metric depth are enabled; there is no external-camera
perception input. Camera calibration is read at the robot's observation pose.
The seeded scatter uses a 380 by 740 mm world-XY region closer to the robot,
with at least 12 mm between conservative object footprints and 45 mm of extra
clearance from the excluded board bounds for the fingers. The board footprint
is excluded from placement and the depth workspace using its fixed workcell
layout bounds. This is not a simulator-instance segmentation mask.
Motion subsequently uses the initial estimated object pose and robot proprioception;
the wrist recording is not a claim of continuous visual pose correction.

The seven NIST parts are large/medium gears, M12/M16 nuts, D-sub connector, and
16 mm round/square pins. The square-pin instruction uses "square pin"
and its visual material is white; its CAD and collision geometry are unchanged.
BBQ sauce and cream cheese reuse the installed LIBERO assets. Cable shark has
been removed from the active test bed. Its original repository CAD remains available.
Bearing, pulley, spacer, red block, and blue block use explicitly
labeled representative procedural CAD. Their dimensions are prototype choices,
not physical calibration or official NIST component dimensions. Ring collisions
preserve center openings; the pulley's belt recess is visual, with an outer-envelope
collision approximation. The NIST board remains background; basket, plate,
placement, insertion, and final orientation are outside this pickup-only run.

BBQ sauce replaces butter in this study.
The two common assets start upright on their CAD Z support plane, consistent with
the current alignment baseline and placement footprints. This overrides LIBERO's
sideways BBQ preset; it is a controlled initial-state assumption, not a measured
orientation or a demonstration of arbitrary resting-state pose estimation.

The boxed-scene VLM prompt explicitly distinguishes similar-looking products
within a family using the target's size, color, and model modifiers, while
accounting for perspective. Exact physical size cannot be inferred from pixel
size alone. The input remains one annotated image and the task instruction;
output remains `label: description` with the 0.99 raw decision-label gate.
The prompt distinguishes a square peg's cross-section and elongated shape from
a block's square face. It explicitly requires any stated object color to match,
while excluding annotation-box and label colors from that judgment. The model
must match the object category and modifiers together.
The instruction is followed by: "You are given a near-top-down wrist-camera
view of the workspace."
Assumed-human identity correction applies only after deferral and cannot repair a
missing or ambiguous depth candidate. The LLM receives the explicit `held`
completion condition; ordinary basket plans still require an empty final gripper.
Gear contacts use separate lower-section and hub hulls within one rigid body,
preserving the original nominal mass/inertia. Their grip heading follows the
robot's current heading rather than the gear's arbitrary CAD yaw. A grasp estimate
up to 1 mm below the controller's 5 mm lower bound is raised to 5.1 mm;
larger violations remain rejected. These are simulation pickup settings.
Before wrist pickup, bounded inverse kinematics checks the pregrasp and grasp
against the robot's joint limits. The controller can use the equivalent 180-degree
parallel-jaw orientation at the same grasp point. These checks do not establish
collision-free motion. Opening, closing, and final holding maintain one fixed hand
pose with feedback, so relative zero commands cannot accumulate position drift.
For upright parts whose CAD height is at least twice their maximum cross-section
width and whose cross-section is no wider than 20 mm, the final descent requires
position error within 1 mm before closing. This prevents the coarser approach
tolerance from initiating an off-center grasp on narrow pegs. Other parts retain
the 8 mm tolerance; applying 1 mm to all parts blocked previously successful
nut, bearing, and BBQ-sauce pickups. This controller setting uses registered CAD
geometry, not object names or simulator object poses.
Confirmed robot-execution faults are recorded separately from the four framework
failure stages. Diagnostic controller replays preserve the original task outcomes
and do not count as additional study attempts.
Success requires the requested target to rise at least 20 mm and maintain bilateral
finger contact for 20 consecutive control steps (one second) ending at trial end,
with no other object lifted during the trial. Logs separately report the longest
continuous hold and whether a one-second hold occurred at any earlier point.
These are pickup outcome measures, not causal framework-stage diagnoses.
Simulator identity, positions, and contacts are used only for
evaluation and the previously authorized assumed-human correction.

This wrist setup clusters connectivity in the estimated table plane. That joins
visible surfaces separated vertically by occlusion (such as a pulley's rim and
upper face), while retaining their original 3D points for localization and CAD
alignment. Raw-cloud cleanup uses the same connectivity rule. The setting is
specific to this scattered scene; existing physical and other simulation runs
retain 3D clustering by default. It assumes separated tabletop footprints and
does not resolve stacked or horizontally overlapping objects.

Artifacts live under `outputs/simulation/wrist_cluster/`: initial wrist RGB-D,
boxed input, exact initial state, catalog provenance, per-target VLM scores,
CAD alignment, LLM plans, action logs, and wrist video when motion is attempted.
The recorded stopping stage is not by itself a causal diagnosis; execution/grasp
failures remain unresolved unless supported by additional evidence. The first
scene is exploratory development, not held-out validation or a VLA comparison.
Try 1 is preserved at `run_20260909T190843Z/`, including its original results and
a snapshot of the recorded source files. `study.json` indexes the ten-try study
(140 planned pickup attempts). Older simulation results and generated reports
were removed, together with the obsolete classification sweeps, basket/VLA
comparison runners, and insertion diagnostics. The current runner, reusable
simulation code, regression tests, CAD assets, and BDDL scene definitions remain.
Local preview regeneration is `main(execute=False, trial_number=N)`;
it saves `setup_try_NN/` without calling the VLM or attempting a pickup.
The configured translation bounds are not an inverse-kinematics or collision
check; a scattered pose inside those bounds is not guaranteed to be reachable.

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
single boxed wrist-camera image and task instruction are sent from WSL to
the configured cloud API. Candidate metadata and point clouds remain local.
The generated letter association is resolved locally before CAD alignment.

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
# The numbered study runner is described above; these commands only check/capture the simulator.
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

## Verification and saved-state diagnostics

### Six-object CAD association experiment

Run the frozen DINOv2-small + FPFH/RANSAC/ICP baseline under the wrist camera:

```bash
cd /mnt/d/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ~/.venvs/lerobot-libero/bin/python -m scripts.run_wrist_association
```

This separate experiment places a large white gear, medium white gear, rectangular
pin (the existing 16 x 10 x 50 mm rectangular pin lying flat), pulley, red block, and blue
block in a fixed, separated layout. It removes the fixed board and fixtures, uses
only `robot0_eye_in_hand`, and moves the simulated wrist to the existing calibrated
observation pose. There are no pickup, planner, VLM, or final pose-registration calls.

The existing depth localizer supplies candidates. Exact IDs from a generated
simulation CAD library bypass text retrieval. Every target compares against every
localized candidate using both modalities; all six targets share batched scene
DINOv2 features and FPFH features. CAD descriptors reuse the usual persistent cache.
Simulator positions enter only the separate identity audit and accuracy calculation.
The audit's existing distance/separation checks validate evaluation labels; they
do not filter association candidates or change ranking decisions.

Each timestamped directory under `outputs/simulation/wrist_association/` contains
calibrated RGB-D, localization and raw object clouds, `wrist_identities.png`
(ground-truth labels, not predictions), `manifest.json`, the exact CAD library,
six `association/target_NNN.json` score/ranking/runtime records, the conflict summary,
and `evaluation.json` with visual-only, geometry-only, and fused top-1 results.
The runtime summary separates CAD preprocessing/cache loading from scene association;
the association-set wall time also includes CAD lookup and result writing, but not
simulation startup, wrist motion, capture, or localization. Cache state is recorded.

The generated CAD library stores `base_color_rgb` from each object's material:
the CAD views of the red and blue blocks now use red and blue. White gears/pin and
the dark pulley use their supplied colors as well. Color participates through
DINOv2 descriptors; no color gate or separate color score is used.

To run association on CPU, set `DINO_DEVICE = "cpu"` in `cad_object_association.py`
before launching, or add `CUDA_VISIBLE_DEVICES=-1` to the command above. This
controls inference independently of the simulator's EGL graphics backend. Saved
RGB-D replay through `scripts.run_wrist_association.run_association(output_dir)`
needs no simulator renderer. For software camera rendering too, use the OSMesa
setup above.

This is a single-scene smoke test. The two blocks intentionally share identical CAD
geometry, so geometry-only matching cannot distinguish their color identities. Independent
queries may select the same candidate, which is recorded as a conflicting set.
The 0.5/0.5 fusion scores are uncalibrated. Do not tune weights or thresholds to this
scene or treat these six outcomes as benchmark accuracy. Layout constants live in
`scripts/run_wrist_association.py`; matching parameters are unchanged.

### Regression checks

Run regression checks in the WSL environment without calling either model:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m unittest discover -s tests
```

`scripts/diagnose_wrist_pickups.py` retains only the reusable `restore` and
`measure` helpers for exact saved-state replay and contact/joint measurements.
It has no hardcoded historical run or batch entry point. Select the saved study
manifest and initial state explicitly when investigating a recorded failure.
Current controller diagnoses are recorded in Try 2's `failure_review.json` and
indexed by `outputs/simulation/wrist_cluster/study.json`.
