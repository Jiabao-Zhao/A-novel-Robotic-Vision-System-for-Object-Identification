"""BOP output validation and evaluation-only official-toolkit invocation.

All poses use CAD/model -> OpenCV camera R,t. Internal translation is metres;
the BOP CSV conversion is the only conversion to millimetres in this module.
Ground-truth access occurs only in a separately invoked official evaluator.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable

import numpy as np


def _official_path(template, **values):
    """Format official numeric placeholders without interpreting literal braces."""
    def replace(match):
        return format(values[match[1]], match[2] or "")
    return Path(re.sub(r"\{(obj_id|scene_id|im_id)(?::([^}]*))?\}", replace, template))


@dataclass(frozen=True)
class PosePrediction:
    scene_id: int
    image_id: int
    object_id: int
    score: float
    R_m2c: np.ndarray
    t_m: np.ndarray
    image_time_s: float

    def validate(self) -> None:
        for name, minimum in (("scene_id", 0), ("image_id", 0), ("object_id", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"Invalid {name}")
        rotation, translation = np.asarray(self.R_m2c), np.asarray(self.t_m)
        if rotation.shape != (3, 3) or translation.size != 3:
            raise ValueError("A pose requires a 3x3 rotation and a three-vector translation")
        if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            raise ValueError("Pose contains nonfinite values")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5):
            raise ValueError("Rotation must be a proper unrestricted SO(3) matrix")
        if not math.isfinite(float(self.score)):
            raise ValueError("Pose score must be finite")
        if not math.isfinite(float(self.image_time_s)) or self.image_time_s < 0:
            raise ValueError("Report measured nonnegative per-image wall time")


def write_bop_csv(predictions: Iterable[PosePrediction], path: str | Path) -> Path:
    """Write valid BOP19 localization predictions, preserving deterministic order."""
    predictions = list(predictions)
    if not predictions:
        raise ValueError("No valid poses: do not write an empty CSV as successful inference")
    timings: dict[tuple[int, int], float] = {}
    for pose in predictions:
        pose.validate()
        key = pose.scene_id, pose.image_id
        if key in timings and not math.isclose(timings[key], pose.image_time_s, rel_tol=0, abs_tol=1e-9):
            raise ValueError("Every prediction from one image must share the complete image wall time")
        timings[key] = pose.image_time_s
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["scene_id", "im_id", "obj_id", "score", "R", "t", "time"])
        for pose in predictions:
            writer.writerow([pose.scene_id, pose.image_id, pose.object_id, format(pose.score, ".17g"),
                             " ".join(format(float(x), ".17g") for x in np.asarray(pose.R_m2c).reshape(-1)),
                             " ".join(format(float(x) * 1000, ".17g") for x in np.asarray(pose.t_m).reshape(-1)),
                             format(pose.image_time_s, ".17g")])
    return path


def read_bop_csv(path: str | Path) -> list[PosePrediction]:
    output = []
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["scene_id", "im_id", "obj_id", "score", "R", "t", "time"]:
            raise ValueError("Unexpected BOP CSV header")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Malformed BOP CSV row")
            rotation = np.asarray([float(value) for value in row["R"].split()])
            translation = np.asarray([float(value) for value in row["t"].split()])
            if rotation.size != 9 or translation.size != 3:
                raise ValueError("BOP R/t must contain exactly9/3 numbers")
            pose = PosePrediction(int(row["scene_id"]), int(row["im_id"]), int(row["obj_id"]),
                                  float(row["score"]), rotation.reshape(3, 3), translation / 1000,
                                  float(row["time"]))
            pose.validate()
            output.append(pose)
    if not output:
        raise ValueError("Empty BOP predictions")
    times: dict[tuple[int, int], float] = {}
    for pose in output:
        key = pose.scene_id, pose.image_id
        if key in times and not math.isclose(times[key], pose.image_time_s, abs_tol=1e-9, rel_tol=0):
            raise ValueError("Inconsistent per-image wall time")
        times[key] = pose.image_time_s
    return output


def validate_bop_csv(path: str | Path, toolkit_root: str | Path,
                     extra_python_paths: Iterable[str | Path] = ()) -> dict:
    """Use both our strict physical checks and the actual official BOP parser."""
    poses = read_bop_csv(path)
    toolkit_root = Path(toolkit_root).resolve()
    if not (toolkit_root / "bop_toolkit_lib/inout.py").is_file():
        raise FileNotFoundError("The official BOP Toolkit checkout is required")
    # A subprocess avoids importing a different toolkit revision into this process.
    # Optional cached pure-Python dependencies are appended after the selected
    # interpreter's installed libraries, avoiding a cached Linux NumPy shadowing
    # the native interpreter's NumPy. No environment is modified on disk.
    code = "import json,sys;sys.path.extend(json.loads(sys.argv[2]));from bop_toolkit_lib import inout;ok,msg=inout.check_bop_results(sys.argv[1],version='bop19');print(json.dumps({'official_format_valid':bool(ok),'message':msg}));sys.exit(0 if ok else 2)"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(toolkit_root) + os.pathsep + environment.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, "-c", code, str(Path(path).resolve()),
                             json.dumps([str(Path(p).resolve()) for p in extra_python_paths])], env=environment,
                            capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"Official BOP validation failed: {result.stdout}\n{result.stderr}")
    official = json.loads(result.stdout.strip().splitlines()[-1])
    revision = subprocess.run(["git", "-C", str(toolkit_root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    return dict(status="valid_predictions_format_only", poses=len(poses),
                images=len({(p.scene_id, p.image_id) for p in poses}),
                toolkit_revision=revision, **official,
                claim="Valid prediction format; this does not measure inference accuracy or reproduce a published score")


def build_evaluation_command(*, toolkit_root: str | Path, results_file: str | Path,
                             datasets_root: str | Path, evaluation_root: str | Path,
                             targets_filename: str, python_executable: str | None = None,
                             renderer_type: str = "vispy", num_workers: int = 1) -> tuple[list[str], dict]:
    """Official BOP19 localization AR (VSD/MSSD/MSPD), evaluation-only process.

    The toolkit's fixed script evaluates all three metrics. No custom metric or
    inference initialization is introduced. Its subprocess 'python' resolves to
    this same selected interpreter via PATH.
    """
    toolkit_root, results_file = Path(toolkit_root).resolve(), Path(results_file).resolve()
    script = toolkit_root / "scripts/eval_bop19_pose.py"
    if not script.is_file() or not results_file.is_file():
        raise FileNotFoundError("Official evaluator script and completed result CSV are required")
    if Path(targets_filename).name != targets_filename or not targets_filename:
        raise ValueError("targets_filename must be a dataset-local file name")
    if renderer_type not in {"vispy", "cpp", "python"} or num_workers < 1:
        raise ValueError("Invalid official evaluator renderer/worker options")
    python = python_executable or sys.executable
    command = [python, str(script), "--result_filenames", results_file.name,
               "--results_path", str(results_file.parent), "--eval_path", str(Path(evaluation_root).resolve()),
               "--targets_filename", targets_filename, "--renderer_type", renderer_type,
               "--num_workers", str(num_workers)]
    environment = os.environ.copy()
    environment["BOP_PATH"] = str(Path(datasets_root).resolve())
    environment["BOP_RESULTS_PATH"] = str(results_file.parent)
    environment["BOP_EVAL_PATH"] = str(Path(evaluation_root).resolve())
    environment["PYTHONPATH"] = str(toolkit_root) + os.pathsep + environment.get("PYTHONPATH", "")
    environment["PATH"] = str(Path(python).parent) + os.pathsep + environment.get("PATH", "")
    return command, environment


def preflight_official_evaluation(*, toolkit_root: str | Path, results_file: str | Path,
                                  datasets_root: str | Path, targets_filename: str,
                                  dataset: str, extra_python_paths: Iterable[str | Path] = ()) -> dict:
    """Resolve official evaluation paths without reading GT pose contents.

    This belongs to the explicit evaluation entry point, never to inference.
    Official dataset_params determines sensor/model conventions; CAD meshes are
    never substituted for the required evaluation model family.
    """
    from .masks import file_sha256, load_localization_targets

    root, toolkit = Path(datasets_root).resolve(), Path(toolkit_root).resolve()
    if Path(targets_filename).name != targets_filename or not targets_filename:
        raise ValueError("targets_filename must be a dataset-local file name")
    target_path = root / dataset / targets_filename
    blockers = []
    if not target_path.is_file():
        blockers.append(dict(component="localization_targets", path=str(target_path),
                             reason="Official object-ID/instance-count localization targets are missing from the explicit evaluation view"))
        targets = []
    else:
        targets = load_localization_targets(target_path)
    rows = read_bop_csv(results_file)
    code = r'''
import json,sys
sys.path.insert(0,sys.argv[1]);sys.path.extend(json.loads(sys.argv[5]))
from bop_toolkit_lib import inout,dataset_params
_,_,dataset,split,split_type,_=inout.parse_result_filename(sys.argv[3])
if dataset!=sys.argv[4]:raise ValueError('Result filename dataset disagrees with config')
model=dataset_params.get_model_params(sys.argv[2],dataset,model_type='eval')
params=dataset_params.get_split_params(sys.argv[2],dataset,split,split_type)
print(json.dumps(dict(dataset=dataset,split=split,split_type=split_type,model=model,
    scene_gt_tpath=params['scene_gt_tpath'],scene_gt_info_tpath=params['scene_gt_info_tpath'],
    scene_camera_tpath=params['scene_camera_tpath'],depth_tpath=params['depth_tpath'])))
'''
    result = subprocess.run([sys.executable, "-c", code, str(toolkit), str(root),
                             Path(results_file).name, dataset,
                             json.dumps([str(Path(p).resolve()) for p in extra_python_paths])],
                            text=True, capture_output=True, check=False)
    if result.returncode:
        blockers.append(dict(component="official_dataset_parameters", reason=result.stderr.strip()))
        return dict(status="blocked", blockers=blockers, target_path=str(target_path),
                    gt_pose_contents_read=False)
    official = json.loads(result.stdout.strip().splitlines()[-1])
    expected: dict[str, list[Path]] = {"evaluation_models": [], "evaluation_model_info": [],
                                     "scene_gt": [], "scene_gt_info": [], "scene_camera": [], "sensor_depth": []}
    model = official["model"]
    expected["evaluation_models"] = [_official_path(model["model_tpath"], obj_id=obj) for obj in model["obj_ids"]]
    expected["evaluation_model_info"] = [Path(model["models_info_path"])]
    for scene in sorted({t.scene_id for t in targets}):
        for component in ("scene_gt", "scene_gt_info", "scene_camera"):
            expected[component].append(_official_path(official[f"{component}_tpath"], scene_id=scene))
    for scene, image in sorted({(t.scene_id, t.image_id) for t in targets}):
        expected["sensor_depth"].append(_official_path(official["depth_tpath"], scene_id=scene, im_id=image))
    for component, paths in expected.items():
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            reason = ("Official models_eval assets are required; inference models_cad is not an evaluation substitute"
                      if component.startswith("evaluation_model") else "Required official evaluator input is absent from the explicit evaluation view")
            blockers.append(dict(component=component, missing_count=len(missing),
                                 missing_paths=missing, reason=reason))
    target_keys = {(t.scene_id, t.image_id, t.object_id) for t in targets}
    prediction_keys = {(p.scene_id, p.image_id, p.object_id) for p in rows}
    if targets and prediction_keys - target_keys:
        blockers.append(dict(component="prediction_targets", reason="CSV includes scene/image/object IDs outside the explicit localization target file",
                             unexpected_keys=sorted(prediction_keys - target_keys)))
    return dict(status="ready" if not blockers else "blocked", blockers=blockers,
                dataset=official["dataset"], split=official["split"], split_type=official["split_type"],
                target_path=str(target_path), target_sha256=file_sha256(target_path) if targets else None,
                targets=len(targets), target_instances=sum(t.instance_count for t in targets),
                target_images=len({(t.scene_id, t.image_id) for t in targets}),
                predicted_target_keys=len(prediction_keys & target_keys),
                target_keys_without_predictions=len(target_keys - prediction_keys),
                gt_pose_contents_read=False,
                scope="Evaluation-only filesystem preflight; no benchmark metric has been measured")


def run_official_evaluation(*, diagnostics_log: str | Path, **kwargs) -> dict:
    """Run only when explicitly requested after inference and dataset preflight."""
    command, environment = build_evaluation_command(**kwargs)
    log = Path(diagnostics_log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(command=command, task="model-based_6D_localization",
                                    metrics=["VSD", "MSSD", "MSPD"], ground_truth="evaluation_only")) + "\n")
        stream.flush()
        process = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if process.returncode:
        raise RuntimeError(f"Official evaluation failed with exit{process.returncode}; inspect {log}")
    return dict(status="official_evaluator_completed", command=command, log=str(log),
                claim="Our inference accuracy is measured by the evaluator outputs; paper-performance reproduction remains a separate comparison")
