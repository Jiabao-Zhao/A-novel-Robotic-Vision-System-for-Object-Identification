# LIBERO RGB-D bridge

This bridge targets Ubuntu 22.04 under WSL2. It uses the current LeRobot
`libero` optional dependency to install LIBERO, then calls the direct LIBERO
environment API because the LeRobot Gym wrapper currently does not expose depth
observations.

## WSL and Python environment

Run the Windows command from an Administrator PowerShell, then restart Windows
if requested:

```powershell
wsl --install -d Ubuntu-22.04
```

Inside Ubuntu, create an isolated environment and install the current LeRobot
source checkout:

```bash
sudo apt-get update
sudo apt-get install -y git python3-venv libegl1 libgl1
python3 -m venv ~/.venvs/libero
source ~/.venvs/libero/bin/activate
python -m pip install --upgrade pip
mkdir -p ~/src
git clone https://github.com/huggingface/lerobot.git ~/src/lerobot
cd ~/src/lerobot
python -m pip install -e ".[libero]"
python -m pip install open3d matplotlib
```

If `~/src/lerobot` already exists, do not clone it again. Inspect its status and
update it deliberately before running the editable installation.

Verify the simulator imports independently:

```bash
python -c "import mujoco; print('mujoco', mujoco.__version__)"
python -c "import robosuite; print('robosuite', robosuite.__version__)"
python -c "import libero; print('libero', libero.__file__)"
```

## Run from the repository root

```bash
cd /mnt/d/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
python -m scripts.libero_smoke_test
python -m scripts.capture_libero_rgbd
python -m scripts.libero_pointcloud_test
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
