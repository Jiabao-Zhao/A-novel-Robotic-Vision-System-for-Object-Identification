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

python -m pip install -e ".[libero]"
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

```bash
cd /mnt/d/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
python -m scripts.libero_smoke_test
python -m scripts.capture_libero_rgbd
python -m scripts.libero_pointcloud_test
python -m scripts.libero_localization_test
python -m scripts.libero_task_execution
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

Outputs are saved beneath `outputs/libero_localization/`. The localization JSON
contains camera-frame ROI, centroid, axis-aligned size, accepted downsampled
cluster point count, cleaned saved PLY point count, and cluster path for each
candidate. The test also prints each centroid transformed into the MuJoCo world
frame for calibration verification.

## Perception-driven task execution

`scripts.libero_task_execution` runs LIBERO-Object task 7, "pick up the milk
and place it in the basket." It uses the rendered agent-view RGB-D observation,
the calibration bridge, the existing depth localizer, the repository's
Gemini-first/OpenAI-fallback VLM association, robot proprioception, and
normalized OSC pose actions. The VLM receives a full-scene plus enlarged-crop
contact sheet labeled only with localized `object_id` values. Simulator
segmentation, simulator object identities, and ground-truth object poses are
not method inputs. LIBERO's task success predicate is read only after execution
as the evaluation result.

Set `GEMINI_API_KEY` and/or `OPENAI_API_KEY` in the WSL process environment
before running the task. `simulation.libero_clip_baseline` remains available as
an explicitly labeled local comparison, but the primary task runner does not
use it.

The episode saves its perception inputs, localization JSON, enlarged VLM visual
prompt, complete VLM result, normalized action log, final agent and wrist RGB-D
observations, success value, and MP4 video beneath
`outputs/libero_task_execution/episode/`. This is a task-specific top-grasp
execution baseline, not yet a general grasp planner or VLA policy.
