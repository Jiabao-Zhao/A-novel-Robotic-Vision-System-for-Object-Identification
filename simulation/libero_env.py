import importlib.util
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path


LIBERO_INSTALL_HINT = (
    "Activate the WSL LeRobot environment and install the current source checkout "
    "with `python -m pip install -e \".[libero]\"`."
)


class LiberoIntegrationError(RuntimeError):
    """Base error for actionable LIBERO integration failures."""


def configure_mujoco_rendering(requested_backend=None):
    """Select an offscreen MuJoCo backend before importing MuJoCo or LIBERO.

    An explicitly configured MUJOCO_GL value is respected. In automatic mode,
    EGL is probed first and OSMesa is used only when EGL is unavailable.
    """
    requested = requested_backend or os.environ.get("MUJOCO_GL", "auto")
    if requested not in {"auto", "egl", "osmesa", "glfw"}:
        raise LiberoIntegrationError(
            f"Unsupported MuJoCo rendering backend {requested!r}. "
            "Use auto, egl, osmesa, or glfw."
        )

    if requested != "auto":
        _set_rendering_environment(requested)
        return requested

    if importlib.util.find_spec("mujoco") is None:
        raise LiberoIntegrationError(f"The `mujoco` package is unavailable. {LIBERO_INSTALL_HINT}")

    failures = {}
    for backend in ("egl", "osmesa"):
        result = _probe_mujoco_backend(backend)
        if result.returncode == 0:
            _set_rendering_environment(backend)
            return backend
        failures[backend] = (result.stderr or result.stdout).strip()

    details = "\n".join(
        f"  {backend}: {message or 'probe exited without diagnostics'}"
        for backend, message in failures.items()
    )
    raise LiberoIntegrationError(
        "Neither EGL nor OSMesa could create an offscreen MuJoCo renderer.\n"
        f"{details}\n"
        "For EGL, verify WSL GPU access with `nvidia-smi` and install `libegl1`. "
        "For the CPU fallback, install `libosmesa6` and run with MUJOCO_GL=osmesa."
    )


def _set_rendering_environment(backend):
    os.environ["MUJOCO_GL"] = backend
    if backend in {"egl", "osmesa"}:
        os.environ["PYOPENGL_PLATFORM"] = backend


def _probe_mujoco_backend(backend):
    probe = """
import mujoco
model = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><light pos="0 0 1"/><geom type="sphere" size="0.1"/></worldbody></mujoco>'
)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
renderer = mujoco.Renderer(model, height=64, width=64)
renderer.update_scene(data)
renderer.render()
renderer.close()
"""
    environment = os.environ.copy()
    environment["MUJOCO_GL"] = backend
    environment["PYOPENGL_PLATFORM"] = backend
    try:
        return subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=20,
            env=environment,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        return subprocess.CompletedProcess(
            args=error.cmd,
            returncode=124,
            stdout=error.stdout or "",
            stderr=f"renderer probe timed out after {error.timeout} seconds",
        )


class LiberoTaskEnvironment:
    """One directly accessible RGB-D LIBERO task environment.

    The direct LIBERO API is used because the current LeRobot Gym wrapper
    exposes RGB observations but not the simulator's depth observations.
    """

    def __init__(
        self,
        suite_name="libero_object",
        task_index=0,
        camera_names=("agentview", "robot0_eye_in_hand"),
        image_width=256,
        image_height=256,
        rendering_backend=None,
    ):
        self.suite_name = str(suite_name)
        self.task_index = int(task_index)
        self.camera_names = tuple(camera_names)
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.rendering_backend = configure_mujoco_rendering(rendering_backend)
        self.last_observation = None

        try:
            from libero.libero import benchmark, get_libero_path
            from libero.libero.envs import OffScreenRenderEnv
        except EOFError as error:
            raise LiberoIntegrationError(
                "LIBERO needs its one-time path configuration. Run "
                "`printf \"n\\n\" | python -c \"import libero.libero\"` inside "
                "the WSL environment to use the packaged default paths."
            ) from error
        except ModuleNotFoundError as error:
            raise LiberoIntegrationError(
                f"LIBERO import failed ({error}). {LIBERO_INSTALL_HINT}"
            ) from error

        suites = benchmark.get_benchmark_dict()
        if self.suite_name not in suites:
            available = ", ".join(sorted(suites))
            raise LiberoIntegrationError(
                f"Unknown LIBERO suite {self.suite_name!r}. Available suites: {available}"
            )

        self.suite = suites[self.suite_name]()
        task_count = getattr(self.suite, "n_tasks", len(getattr(self.suite, "tasks", [])))
        if not 0 <= self.task_index < task_count:
            raise LiberoIntegrationError(
                f"Task index {self.task_index} is outside [0, {task_count - 1}] "
                f"for {self.suite_name}."
            )

        self.task = self.suite.get_task(self.task_index)
        self.task_name = str(getattr(self.task, "name", Path(self.task.bddl_file).stem))
        self.instruction = str(getattr(self.task, "language", ""))
        self.bddl_path = Path(get_libero_path("bddl_files")) / self.task.problem_folder / self.task.bddl_file
        if not self.bddl_path.is_file():
            raise LiberoIntegrationError(
                f"LIBERO task definition is missing: {self.bddl_path}. "
                "Check the installed hf-libero assets and LIBERO config paths."
            )

        try:
            self.env = OffScreenRenderEnv(
                bddl_file_name=str(self.bddl_path),
                camera_names=list(self.camera_names),
                camera_heights=self.image_height,
                camera_widths=self.image_width,
                camera_depths=True,
                use_camera_obs=True,
                has_renderer=False,
                has_offscreen_renderer=True,
                render_gpu_device_id=-1,
                control_freq=20,
            )
        except Exception as error:
            raise LiberoIntegrationError(
                "LIBERO environment creation failed with "
                f"MUJOCO_GL={self.rendering_backend!r}: {error}. "
                "If EGL failed under WSL, verify `nvidia-smi`; then retry with "
                "MUJOCO_GL=osmesa after installing libosmesa6."
            ) from error

    @property
    def sim(self):
        simulator = getattr(self.env, "sim", None)
        if simulator is None:
            simulator = getattr(getattr(self.env, "env", None), "sim", None)
        if simulator is None:
            raise LiberoIntegrationError(
                "The installed LIBERO environment does not expose its MuJoCo simulator, "
                "so camera calibration cannot be retrieved."
            )
        return simulator

    def reset(self):
        try:
            self.last_observation = self.env.reset()
        except Exception as error:
            raise LiberoIntegrationError(
                f"LIBERO reset failed for {self.suite_name}[{self.task_index}]: {error}"
            ) from error
        if not isinstance(self.last_observation, Mapping):
            raise LiberoIntegrationError(
                "LIBERO reset returned an unexpected observation type: "
                f"{type(self.last_observation).__name__}"
            )
        return self.last_observation

    @property
    def action_dim(self):
        return int(self.env.env.action_dim)

    @property
    def robots(self):
        return self.env.robots

    def step(self, action):
        try:
            self.last_observation, reward, done, info = self.env.step(action)
        except Exception as error:
            raise LiberoIntegrationError(f"LIBERO action step failed: {error}") from error
        return self.last_observation, reward, done, info

    def check_success(self):
        return bool(self.env.check_success())

    def close(self):
        if getattr(self, "env", None) is not None:
            self.env.close()
            self.env = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def observation_lines(observation):
    """Describe actual observation leaves without assuming LIBERO key names."""
    lines = []
    for key, value in _flatten_mapping(observation):
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        if shape is not None:
            lines.append(f"  {key}: shape={tuple(shape)}, dtype={dtype}")
        else:
            lines.append(f"  {key}: {type(value).__name__}")
    return lines


def _flatten_mapping(value, prefix=""):
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten_mapping(child, child_prefix)
    else:
        yield prefix, value
