"""Run and resume the current 100-episode development batch, with identity audits."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time
import traceback


ROOT = Path("outputs/simulation/study_1200/development/object_states_02_to_12")
STATE_INDICES = (2, 3, 4, 5, 6, 7, 8, 9, 11, 12)
WORKERS = 3
MODEL = "gpt-4.1-mini-2025-04-14"


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def audit_episode(case, task_index, state_index):
    """Restore the original scene for evaluation; never rerun or repair predictions."""
    import numpy as np
    from simulation.libero_assumed_human import match_candidates
    from simulation.libero_control import OPEN_GRIPPER, hold_gripper
    from simulation.libero_env import LiberoTaskEnvironment
    from simulation.libero_experiment import libero_object_task
    from simulation.libero_joint_association import joint_inferences
    from vlm_module import candidate_choice_map

    report = json.loads((case / "episode.json").read_text())
    local_path = case / "perception/point_cloud/point_cloud_localization.json"
    if not local_path.is_file():
        return {"status": "localization_unavailable", "associations": [],
                "failure_observed_at": report.get("failure_observed_at")}
    localization = json.loads(local_path.read_text())
    transform = np.load(case / "perception/capture/world_T_camera.npy")
    with LiberoTaskEnvironment(task_index=task_index, image_width=768, image_height=768) as env:
        raw = env.reset(seed=1000, init_state_index=state_index)
        if env.last_init_state_sha256 != report["initial_state_sha256"]:
            raise ValueError("Identity audit initial state differs from the episode.")
        hold_gripper(env, raw, OPEN_GRIPPER, "initial_physics_settle", 10)
        settled_hash = hashlib.sha256(
            np.ascontiguousarray(env.sim.get_state().flatten()).tobytes()
        ).hexdigest()
        recorded_hash = report.get("settled_state_sha256")
        if recorded_hash is not None and recorded_hash != settled_hash:
            raise ValueError("Identity audit settled state differs from the episode.")
        inner = env.env.env
        objects = [{
            "instance": obj.name,
            "semantic_identity": obj.category_name.replace("_", " "),
            "position_world_m": env.sim.data.body_xpos[
                env.sim.model.body_name2id(obj.root_body)
            ].tolist(),
        } for obj in [*inner.objects, *inner.fixtures]]
        matches = match_candidates(localization, transform, objects)

    _, slug, instruction = libero_object_task(task_index)
    names = [slug.replace("_", " "), "basket"]
    raw_path = case / "vlm_provider_response.json"
    inference = []
    parsed = None
    if raw_path.is_file():
        inference, parsed = joint_inferences(
            json.loads(raw_path.read_text()), instruction,
            candidate_choice_map(localization["objects"]), names,
        )
    result_path = case / "vlm_result.json"
    resolved = (json.loads(result_path.read_text())["associations"]
                if result_path.is_file() else [])
    decisions = []
    for name in names:
        candidates = [m for m in matches if m["semantic_identity"] == name]
        expected = candidates[0]["object_id"] if len(candidates) == 1 else None
        predicted = next((a for a in inference if a["target_description"] == name), {})
        final = next((a for a in resolved if a["target_description"] == name), {})
        if final:
            if (final["vlm_object_id"] != predicted["vlm_object_id"]
                    or final["association_score"] != predicted["association_score"]):
                raise ValueError("Saved original prediction or raw score does not reproduce.")
            if (final["resolution"] == "vlm_accepted"
                    and final["final_object_id"] != final["vlm_object_id"]):
                raise ValueError("An accepted prediction was modified.")
        decisions.append({
            "target_description": name, "expected_object_id": expected,
            "audit_status": "resolved" if expected is not None else "unresolved",
            "vlm_object_id": predicted.get("vlm_object_id"),
            "raw_association_score": predicted.get("association_score"),
            "original_prediction_correct": (predicted.get("vlm_object_id") == expected
                                            if expected is not None and predicted else None),
            "resolution": final.get("resolution"),
            "final_object_id": final.get("final_object_id"),
            "final_identity_correct": (final.get("final_object_id") == expected
                                       if expected is not None and final else None),
        })
    frozen_path = case / "execution_inputs.json"
    frozen_verified = None
    cad = {}
    if frozen_path.is_file():
        frozen = json.loads(frozen_path.read_text())
        digest = hashlib.sha256(json.dumps(
            frozen["perception"], sort_keys=True, allow_nan=False
        ).encode()).hexdigest()
        if digest != frozen["inputs_sha256"]:
            raise ValueError("Frozen perception hash does not match.")
        frozen_verified = True
        cad = frozen["perception"]["target_cad_registration"]
    return {
        "status": "completed", "task_index": task_index, "initial_state_index": state_index,
        "initial_state_verified": True, "settled_state_verified": recorded_hash is not None,
        "scene_group_id": f"libero_object:{task_index}:{settled_hash}",
        "ground_truth_use": "Post-execution identity audit only; no poses supplied to execution.",
        "included_in_final_1200": False, "task_success": report["success"],
        "candidate_identity_audit": matches, "associations": decisions,
        "joint_format_valid": None if parsed is None else parsed["format_valid"],
        "format_errors": [] if parsed is None else parsed["format_errors"],
        "frozen_perception_verified": frozen_verified,
        "cad_alignment": {key: cad.get(key) for key in
                          ("registration_accepted", "constrained_rmse_m", "max_accepted_rmse_m")},
        "planning_evaluation": report.get("plan_evaluation"),
        "failure_observed_at": report.get("failure_observed_at"),
        "causal_failure_stage": None,
    }


def run_episode(task_index, slug, state_index):
    from scripts import libero_task_execution as runner

    case = ROOT / f"task_{task_index:02d}_{slug}" / f"init_state_{state_index:02d}"
    case.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with (case.parent / f"init_state_{state_index:02d}.log").open("a") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            # Resume audits without paying for another VLM/LLM call or rerunning a failure.
            if not (case / "episode.json").is_file():
                try:
                    runner.main(task_index, initial_state_index=state_index, output_root=case)
                except Exception:
                    traceback.print_exc()
                    if not (case / "episode.json").is_file():
                        raise
            report = json.loads((case / "episode.json").read_text())
            try:
                audit = audit_episode(case, task_index, state_index)
            except Exception as error:
                traceback.print_exc()
                audit = {"status": "audit_error", "error": f"{type(error).__name__}: {error}"}
            save_json(case / "evaluation.json", audit)
    return {
        "task_index": task_index, "target": slug, "initial_state_index": state_index,
        "success": report["success"], "run_status": report["run_status"],
        "planning_status": report.get("planning_status"),
        "failure_observed_at": report.get("failure_observed_at"),
        "termination_reason": report.get("termination_reason"),
        "execution_error": report.get("execution_error"), "audit_status": audit["status"],
        "seconds_including_audit": round(time.perf_counter() - started, 1),
        "episode": str(case / "episode.json"),
    }


def main():
    from scripts import libero_task_execution as runner
    from simulation.libero_experiment import LIBERO_OBJECT_TASKS

    trials = [(i, slug, state) for state in STATE_INDICES for i, slug, _ in LIBERO_OBJECT_TASKS]
    source_paths = [Path("scripts/libero_task_execution.py"), Path(__file__),
                    Path("vlm_module.py"), Path("LLM_planner.py"), Path("point_cloud_localization.py"),
                    Path("CADPointCloudRegistration.py"), *sorted(Path("simulation").glob("*.py"))]
    manifest = {
        "split": "development", "included_in_final_1200": False,
        "planned_trials": len(trials), "initial_state_indices": list(STATE_INDICES),
        "prior_development_state_indices": [0, 1, 10, 20],
        "model": MODEL, "workers": WORKERS, "omp_threads_per_worker": 4,
        "threshold": runner.LIBERO_RAW_ASSOCIATION_THRESHOLD,
        "threshold_validated": False, "prompt_changed": False,
        "assumed_correct_human_on_deferral": True,
        "execution_config": runner._execution_config(0, STATE_INDICES[0]),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Batch settings/source changed; use a separate output root.")
    save_json(manifest_path, manifest)
    results = []
    pending = []
    for task, slug, state in trials:
        result_path = ROOT / f"task_{task:02d}_{slug}" / f"init_state_{state:02d}" / "batch_result.json"
        if result_path.is_file():
            results.append(json.loads(result_path.read_text()))
        else:
            pending.append((task, slug, state))
    print(f"BATCH_START planned={len(trials)} completed={len(results)} workers={WORKERS}", flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(run_episode, *trial): trial for trial in pending}
        for future in as_completed(futures):
            result = future.result()
            case = Path(result["episode"]).parent
            save_json(case / "batch_result.json", result)
            results.append(result)
            results.sort(key=lambda r: (r["initial_state_index"], r["task_index"]))
            save_json(ROOT / "results.json", results)
            print(json.dumps({"completed": len(results), **result}), flush=True)
    print(f"BATCH_COMPLETE {sum(r['success'] for r in results)}/{len(results)} successful", flush=True)


if __name__ == "__main__":
    os.environ.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", OMP_NUM_THREADS="4",
                      OPENAI_VLM_MODEL=MODEL, OPENAI_LLM_MODEL=MODEL)
    main()
