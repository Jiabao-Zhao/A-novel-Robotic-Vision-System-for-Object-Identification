"""Evaluation-only object expansion: one fresh reset per target, no saved states.

Stages are catalog, capture, run, analyze. Capture and API inference are separate
so the frozen RGB/localization and evaluation labels can be inspected first.
"""

import json
from pathlib import Path
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

from scripts.run_vlm_candidate_verification_pilot import (
    MODEL, VERIFICATION_INSTRUCTION, VERIFICATION_SYSTEM, run_scene, save_json,
)
from scripts.run_vlm_output_format_ablation import build_prompt, digest, request_arguments
from vlm_module import candidate_prompt_records, load_localized_objects


OUTPUT_ROOT = Path("outputs/simulation/experiments/new_objects_verification")
OLD_TARGETS = {
    "alphabet_soup", "cream_cheese", "salad_dressing", "bbq_sauce", "ketchup",
    "tomato_sauce", "butter", "milk", "chocolate_pudding", "orange_juice",
    "basket",  # Already present in every previous LIBERO-Object scene.
}
# One instance per category in each chosen BDDL; no position/task-role phrases.
# Selected from the installed catalog BEFORE any capture or model response.
TARGETS = (
    ("akita_black_bowl", "black bowl", "libero_90", 6),
    ("black_book", "book", "libero_90", 73),
    ("chefmate_8_frypan", "frying pan", "libero_90", 18),
    ("cookies", "cookies", "libero_spatial", 0),
    ("glazed_rim_porcelain_ramekin", "ramekin", "libero_spatial", 0),
    ("moka_pot", "moka pot", "libero_90", 18),
    ("plate", "plate", "libero_90", 69),
    ("porcelain_mug", "white mug", "libero_90", 65),
    ("red_coffee_mug", "red mug", "libero_90", 65),
    ("white_bowl", "white bowl", "libero_90", 35),
    ("white_yellow_mug", "yellow and white mug", "libero_90", 65),
    ("wine_bottle", "wine bottle", "libero_90", 22),
    ("wooden_tray", "wooden tray", "libero_90", 55),
)
CAPTURE_SETTINGS = {
    "reset_policy": "one fresh seeded environment.reset per target; no saved state files or state sweep",
    "base_reset_seed": 5000, "resolution_hw": [768, 768], "camera": "agentview",
    "contact_sheet_tile_px": 448, "physics_settle_steps": 10,
    "localization": "unchanged simulation.perception_adapter.run_libero_localization",
    "workspace_crop": "existing bounds translated by installed arena workspace_offset, not object poses",
}


def target_ground_truth(matches, instance):
    """Evaluation-only identity; an ambiguous/missed target is not target absence."""
    ids = [m["object_id"] for m in matches
           if m["status"] == "matched" and m["simulator_instance"] == instance]
    return ids[0] if len(ids) == 1 else None


def table_workspace_bounds(offset_m):
    from simulation.perception_adapter import LIBERO_WORKSPACE_MIN_XYZ_M, LIBERO_WORKSPACE_MAX_XYZ_M

    return [[float(x + delta) for x, delta in zip(bound, offset_m)]
            for bound in (LIBERO_WORKSPACE_MIN_XYZ_M, LIBERO_WORKSPACE_MAX_XYZ_M)]


def selected_tasks(inventory):
    selected = []
    for index, (category, description, suite, task_index) in enumerate(TARGETS):
        task = next(t for t in inventory[category] if t["suite"] == suite and t["task_index"] == task_index)
        selected.append({**task, "category": category, "target_description": description,
                         "sample_index": index, "sample_id": f"fresh_{category}",
                         "reset_seed": CAPTURE_SETTINGS["base_reset_seed"] + index})
    return selected


def capture_scene(task):
    from scripts.audit_libero_candidate_identities import match_candidates
    from scripts.capture_libero_confidence_dataset import compact_localization
    from simulation.libero_control import OPEN_GRIPPER, hold_gripper
    from simulation.libero_env import LiberoTaskEnvironment
    from simulation.libero_io import save_libero_observation
    from simulation.libero_sensor import LiberoRGBDSensor
    from simulation.perception_adapter import run_libero_localization
    from vlm_module import create_roi_contact_sheet

    directory = OUTPUT_ROOT / "captures" / task["sample_id"]
    sample_path = directory / "sample.json"
    config = {"task": task, "settings": CAPTURE_SETTINGS}
    capture_hash = digest(json.dumps(config, sort_keys=True).encode())
    if sample_path.exists():
        sample = json.loads(sample_path.read_text())
        if sample["capture_config_sha256"] != capture_hash:
            raise ValueError("Capture configuration changed; refusing to overwrite a frozen scene.")
        for name, expected in sample["files_sha256"].items():
            if digest((directory / name).read_bytes()) != expected:
                raise ValueError(f"Frozen capture changed: {directory / name}")
        return sample

    with LiberoTaskEnvironment(suite_name=task["suite"], task_index=task["task_index"],
                               image_width=768, image_height=768) as environment:
        if environment.task_name != task["task_name"]:
            raise ValueError("Installed task ordering changed.")
        # Deliberately omit init_state_index: never load/set a benchmark state.
        raw = environment.reset(seed=task["reset_seed"])
        raw = hold_gripper(environment, raw, OPEN_GRIPPER, "capture_settle_only",
                           CAPTURE_SETTINGS["physics_settle_steps"])
        observation = LiberoRGBDSensor(environment, "agentview").capture(raw)
        if environment.last_init_state_index is not None or environment.last_init_state_sha256 is not None:
            raise ValueError("Capture unexpectedly loaded a saved initial state.")
        paths = save_libero_observation(observation, environment, directory)
        bounds = table_workspace_bounds(environment.env.env.workspace_offset)
        # Reuse the unmodified depth-localization and ROI visualization. Temporary
        # PLYs are not retained for this association-only experiment.
        with tempfile.TemporaryDirectory() as temp:
            localization, _, _ = run_libero_localization(observation, paths["rgb"], Path(temp),
                                                         workspace_bounds=bounds)
        localization_path = directory / "localization.json"
        save_json(localization_path, compact_localization(localization, paths["rgb"]))
        marked_path = directory / "vlm_roi_contact_sheet.png"
        create_roi_contact_sheet(paths["rgb"], localization_path, marked_path, tile_size_px=448)

        # Only the evaluation file below receives simulator identities/positions.
        inner = environment.env.env
        objects = []
        for obj in [*inner.objects, *inner.fixtures]:
            body_id = environment.sim.model.body_name2id(obj.root_body)
            objects.append({"instance": obj.name, "semantic_identity": obj.category_name.replace("_", " "),
                            "position_world_m": environment.sim.data.body_xpos[body_id].tolist()})
        matches = match_candidates(localization, observation.world_T_camera, objects)
        target_id = target_ground_truth(matches, task["target_instance"])
        save_json(directory / "evaluation_ground_truth.json", {
            "evaluation_only": True, "excluded_from_vlm_inputs": True,
            "target_instance": task["target_instance"], "ground_truth_object_id": target_id,
            "matching_method": "separated mutual-nearest world-XY centroid pairs; no forced assignment",
            "simulator_objects": objects, "candidates": matches,
        })
    sample = {**task, "capture_config_sha256": capture_hash, "partition": "exploratory_new_objects",
              "saved_initial_state_used": False, "ground_truth_object_id": target_id,
              "status": "ready" if target_id is not None else "target_localization_unresolved",
              "candidate_count": localization["object_count"],
              "workspace_bounds_world_m": bounds,
              "image_path": str(marked_path), "localization_path": str(localization_path)}
    sample["files_sha256"] = {p.name: digest(p.read_bytes()) for p in sorted(directory.iterdir()) if p.is_file()}
    save_json(sample_path, sample)
    print(f"Captured {task['category']}: candidates={sample['candidate_count']} target={target_id} "
          f"status={sample['status']} reset_seed={task['reset_seed']}", flush=True)
    return sample


def prepare_scene(sample):
    if sample["status"] != "ready" or sample["ground_truth_object_id"] is None:
        raise ValueError("Unresolved localization must not be treated as a VLM trial or an absent target.")
    image_path, localization_path = Path(sample["image_path"]), Path(sample["localization_path"])
    for name, expected in sample["files_sha256"].items():
        if digest((image_path.parent / name).read_bytes()) != expected:
            raise ValueError("Frozen capture changed before inference.")
    detections = load_localized_objects(localization_path)
    prompt, choices = build_prompt(sample["target_description"], detections, "compact")
    # This allowlisted geometric metadata is the ONLY candidate data sent to API.
    candidates = candidate_prompt_records(detections, {k: v for k, v in choices.items() if v is not None})
    scene = {**sample, "prompt": prompt, "candidate_map": choices, "candidate_metadata": candidates,
             "image_sha256": digest(image_path.read_bytes()),
             "localization_sha256": digest(localization_path.read_bytes()),
             "prompt_sha256": digest(prompt.encode())}
    scene["input_sha256"] = digest(json.dumps(scene, sort_keys=True).encode())
    return scene, image_path.read_bytes()


def run_comparison():
    manifest = json.loads((OUTPUT_ROOT / "manifest.json").read_text())
    samples = manifest["samples"]
    scenes = [prepare_scene(s) for s in samples if s["status"] == "ready"]
    args = request_arguments(b"", "")
    protocol = {"model": MODEL, "request_settings": {k: v for k, v in args.items() if k != "messages"},
                "baseline_system": args["messages"][0]["content"], "image_detail": "high",
                "verification_system": VERIFICATION_SYSTEM, "verification_instruction": VERIFICATION_INSTRUCTION,
                "capture_settings": CAPTURE_SETTINGS,
                "sample_inputs": {s["sample_id"]: s["input_sha256"] for s, _ in scenes},
                "analysis": "one fresh reset per new target; exploratory only, no held-out claim",
                "excluded_before_api": [s["sample_id"] for s in samples if s["status"] != "ready"]}
    for folder in ("responses", "scenes"):
        (OUTPUT_ROOT / folder).mkdir(exist_ok=True)
    protocol_path = OUTPUT_ROOT / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Protocol/input mismatch; refusing to mix runs.")
    save_json(protocol_path, protocol)
    print(f"{len(scenes)} ready targets; {sum(1 + len(s['candidate_metadata']) for s, _ in scenes)} calls; "
          f"{len(samples) - len(scenes)} localization-unresolved captures excluded.", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_scene, scene, image, OUTPUT_ROOT) for scene, image in scenes]
        records = [future.result() for future in futures]
    save_json(OUTPUT_ROOT / "records.json", {"protocol": protocol, "records": records})


def analyze():
    from scripts.analyze_vlm_candidate_verification_pilot import (
        matched_coverage, plot_results, summary, sweep, validate_records, write_csv,
    )

    payload = json.loads((OUTPUT_ROOT / "records.json").read_text())
    records = validate_records(payload, expected_count=len(payload["protocol"]["sample_inputs"]),
                               fixed_calibration_states=False)
    manifest = json.loads((OUTPUT_ROOT / "manifest.json").read_text())
    points = sweep(records)
    frontiers, matched = matched_coverage(points)
    stats = summary(records)
    stats["verification_unresolved"] = sum(r["verification"]["predicted_object_id"] is None for r in records)
    stats["verification_wrong_resolved"] = sum(r["verification"]["predicted_object_id"] is not None
                                                and not r["verification"]["correct"] for r in records)
    stats.update(capture_attempts=len(manifest["samples"]),
                 excluded_before_api=payload["protocol"]["excluded_before_api"], matched_coverage=matched)
    directory = OUTPUT_ROOT / "analysis"
    directory.mkdir(exist_ok=True)
    save_json(directory / "summary.json", stats)
    write_csv(directory / "threshold_sweep.csv", points)
    write_csv(directory / "matched_coverage.csv", [p for r in matched for p in r["methods"].values()])
    rows, candidate_rows = [], []
    for r in records:
        scene, baseline, verify = r["scene"], r["baseline"], r["verification"]
        common = {k: scene[k] for k in ("sample_id", "target_description", "ground_truth_object_id", "input_sha256")}
        rows.append({**common, "baseline_prediction": baseline["predicted_object_id"],
                     "baseline_correct": baseline["correct"], "baseline_likelihood": baseline["raw_sequence_likelihood"],
                     **verify})
        candidate_rows.extend({**common, "object_id": k, **c} for k, c in r["candidates"].items())
    write_csv(directory / "scene_predictions.csv", rows)
    write_csv(directory / "candidate_scores.csv", candidate_rows)
    plot_results(records, frontiers, points, directory)
    n = len(records)
    lines = ["# Additional LIBERO objects: fresh-reset comparison", "",
             "One fresh seeded reset per target. No saved initial-state files, state-index sweep, "
             "test-set access, robot task execution, CAD registration or production VLM/HITL changes. "
             "Both methods use identical frozen RGB/localization/contact-sheet inputs.", "",
             "The simulation adapter now accepts optional world-frame crop bounds: the existing crop "
             "is translated by each arena's workspace_offset. Segmentation settings, algorithms and "
             "the original LIBERO-Object default bounds are unchanged. See ../INPUT_INSPECTION.md "
             "for pre-inference visual checks and the initial crop-configuration failure.", "",
             f"{len(manifest['samples'])} captures attempted; {n} targets have unambiguous evaluation labels. "
             "Unresolved localization is excluded before model calls, not scored as a classification error.", "",
             f"GPT-4.1-mini-2025-04-14; temperature 0; detail high; native 768x768; contact-sheet tiles 448; "
             f"{n} baseline calls plus {stats['candidate_calls']} independent candidate Y/N calls.", "",
             f"Baseline {stats['baseline_correct']}/{n} correct; verification {stats['verification_correct']} correct, "
             f"{stats['verification_wrong_resolved']} wrong resolved selections and {stats['verification_unresolved']} "
             "unresolved/deferred scenes. Unresolved is NOT an incorrect selected object or target absence. "
             f"Verification corrects {stats['baseline_errors_corrected']} baseline errors; "
             f"{stats['baseline_correct_lost']} baseline-correct scenes become wrong OR unresolved.", "",
             f"Score availability: baseline {stats['baseline_score_available']}/{n}; fully scored verification "
             f"{stats['verification_complete_scenes']}/{n}; candidate scores "
             f"{stats['candidate_scores_available']}/{stats['candidate_calls']}.", "",
             "| Target | GT ID | Baseline ID / correct | Raw likelihood | Verification ID / correct | YES support | Gap |",
             "| --- | --- | --- | ---: | --- | ---: | ---: |"]
    for r in rows:
        lines.append(f"| {r['target_description']} | {r['ground_truth_object_id']} | "
                     f"{r['baseline_prediction']} / {r['baseline_correct']} | {r['baseline_likelihood']} | "
                     f"{r['predicted_object_id']} / {r['correct']} | {r['support']} | {r['gap']} |")
    lines += ["", "## Localization-unresolved captures (no API calls)", ""]
    lines += [f"- {s['target_description']}: {s['candidate_count']} clusters, no separated mutual-nearest target match."
              for s in manifest["samples"] if s["status"] != "ready"] or ["None."]
    lines += ["", "## Fixed illustrative gates (not calibrated or coverage-matched)", "",
              "| Gate | Accepted | Accuracy | False accepts | Same baseline errors deferred |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for p in stats["reference_gates"]:
        accuracy = f"{p['autonomous_accuracy']:.1%}" if p["autonomous_accuracy"] is not None else "undefined"
        lines.append(f"| {p['method']} ({p['tau_support']}, {p['tau_gap']}) | {p['autonomous_decisions']}/{n} | "
                     f"{accuracy} | {p['false_autonomous_acceptance_count']} | {p['baseline_errors_deferred']} |")
    lines += ["", "## Matched-coverage error counts", "",
              "Support+gap is an optimistic same-pilot envelope, not held-out performance. "
              "No tie splitting; missing scores count as deferred. Full sweeps and both false-acceptance "
              "rate denominators are saved in CSV.", "",
              "| Accepted | Baseline false accepts | Support false accepts | Support+gap false accepts |",
              "| ---: | ---: | ---: | ---: |"]
    for row in matched:
        errors = [str(row["methods"][m]["false_autonomous_acceptance_count"])
                  for m in ("baseline", "support", "support_gap")]
        lines.append("| " + " | ".join([f"{row['accepted_count']}/{n}", *errors]) + " |")
    lines += ["", "## What happened in this pilot", ""]
    lookup = {r["scene"]["target_description"]: r for r in records}
    for target in ("ramekin", "black bowl", "book", "red mug"):
        r = lookup.get(target)
        if r is None:
            continue
        b, v = r["baseline"], r["verification"]
        ranked = sorted(((k, c["yes_score"]) for k, c in r["candidates"].items() if c["yes_score"] is not None),
                        key=lambda pair: -pair[1])
        missing = {k: c.get("score_unavailable_reason") for k, c in r["candidates"].items() if c["yes_score"] is None}
        lines.append(f"- {target}: baseline {b['predicted_object_id']} (raw={b['raw_sequence_likelihood']}); "
                     f"verification {v['predicted_object_id']}; largest available YES scores={ranked[:2]}; "
                     f"missing alternatives={missing}.")
    base_gate, support_gate, gap_gate = stats["reference_gates"]
    lines += ["", f"At the pre-existing illustrative settings, baseline accepts {base_gate['autonomous_decisions']}, "
              f"support-only accepts {support_gate['autonomous_decisions']}, and support+gap accepts "
              f"{gap_gate['autonomous_decisions']} of {n}. Their false-acceptance counts are "
              f"{base_gate['false_autonomous_acceptance_count']}, {support_gate['false_autonomous_acceptance_count']}, "
              f"and {gap_gate['false_autonomous_acceptance_count']} respectively. "
              "At the exactly matched coverage points in the table, compare error counts directly; "
              "fewer accepted scenes alone does not establish better error detection.", ""]
    if matched and all(len({p["false_autonomous_acceptance_count"] for p in row["methods"].values()}) == 1
                       for row in matched):
        lines += ["**This pilot shows no matched-coverage error-detection advantage for verification.** "
                  "It may change which errors are corrected and which correct scenes are deferred. "
                  "Support+gap can reveal conflicting high YES scores, but here this also costs correct accepts. "
                  "Keep the production gate unchanged; do not generalize from this small exploratory sample.", ""]
    lines += ["", "## Score distributions and discrimination", "",
              "Full correct/incorrect score distributions and ranking AUROC are in summary.json. "
              "An unavailable AUROC means only one correctness class has usable scores, not poor performance.", ""]
    for key in ("baseline_distribution", "support_distribution"):
        distribution = stats[key]
        lines.append(f"- {key}: correctness-ranking AUROC={distribution['correctness_ranking_auc']}.")
        for outcome in ("correct", "incorrect"):
            d = distribution[outcome]
            if d is not None:
                lines.append(f"  - {outcome}: n={d['count']}, min={d['minimum']}, median={d['median']}, max={d['maximum']}.")
    lines += ["", "## Interpretation limits", "",
              "One observation per new asset cannot establish generalization or calibrate thresholds. "
              "These new scenes differ from the old ten-product pilot in background, geometry and distractors too. "
              "Raw likelihood and YES support are not calibrated correctness probabilities. "
              "Low YES support is not proven absence. Complete Y/N alternatives at the same answer position "
              "are mandatory; any missing candidate score or exact top tie defers the whole scene. "
              "Compare actual errors at matched coverage, not score spread alone. "
              "Simulator identity matching is evaluation-only and never enters either prompt.", "",
              "## Reproduction (existing WSL venv, repository root)", "",
              "```sh", "source /home/jiabao/.venvs/lerobot-libero/bin/activate",
              "python -m scripts.run_libero_new_object_verification catalog",
              "python -m scripts.run_libero_new_object_verification capture",
              "python -m scripts.run_libero_new_object_verification run",
              "python -m scripts.run_libero_new_object_verification analyze", "```", "",
              "Capture reuses frozen samples after hash checks; run reuses all successful saved API responses "
              "after input/settings checks; analyze is offline. Inspect inputs before a new API run. "
              "No new package installation was needed.", "",
              "![Exploratory comparison](comparison.png)", ""]
    (directory / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Report: {directory / 'REPORT.md'}")
    print(json.dumps({k: stats[k] for k in ("scenes", "baseline_correct", "verification_correct",
          "baseline_errors_corrected", "baseline_correct_lost", "candidate_calls", "excluded_before_api")}, indent=2))


def catalog():
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.bddl_utils import robosuite_parse_problem

    by_category = {}
    for suite_name in ("libero_90", "libero_spatial", "libero_goal", "libero_10"):
        suite = benchmark.get_benchmark_dict()[suite_name]()
        for task_index in range(suite.n_tasks):
            task = suite.get_task(task_index)
            path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            problem = robosuite_parse_problem(str(path))
            for category, instances in problem["objects"].items():
                if category in OLD_TARGETS or len(instances) != 1:
                    continue
                by_category.setdefault(category, []).append({
                    "suite": suite_name, "task_index": task_index, "task_name": task.name,
                    "target_instance": instances[0], "objects": problem["objects"],
                    "bddl_path": str(path),
                })
    return by_category


def main(stage):
    if stage == "catalog":
        inventory = catalog()
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        save_json(OUTPUT_ROOT / "catalog.json", inventory)
        for category, tasks in sorted(inventory.items()):
            first = tasks[0]
            print(f"{category}: {len(tasks)} tasks; first={first['suite']} "
                  f"{first['task_index']} {first['task_name']}")
    elif stage == "capture":
        tasks = selected_tasks(json.loads((OUTPUT_ROOT / "catalog.json").read_text()))
        samples = [capture_scene(task) for task in tasks]
        save_json(OUTPUT_ROOT / "manifest.json", {"settings": CAPTURE_SETTINGS, "samples": samples})
    elif stage == "run":
        run_comparison()
    elif stage == "analyze":
        analyze()
    else:
        raise ValueError("Use catalog, capture, run or analyze.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) == 2 else "catalog")
