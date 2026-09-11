"""Paired relabeling of frozen wrist images; VLM association only, no motion."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from openai import OpenAI

import vlm_module
from simulation.libero_joint_association import joint_inferences, request_arguments


BASE = Path("outputs/simulation/wrist_cluster")
MODEL = "gpt-4.1-mini-2025-04-14"
SEED = 20260919
PAIRS = {"gears": ("Gear_Large", "Gear_Medium"),
         "nuts": ("M12_Hex_Nut", "M16_Hex_Nut")}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label_pixels(detections, mappings, shape):
    """Union of old/new label regions, for checking unchanged scene pixels."""
    mask = np.zeros(shape[:2], dtype=bool)
    boxes = {d["object_id"]: d["bbox_2d_xyxy"] for d in detections}
    height, width = shape[:2]
    for mapping in mappings:
        for letter, object_id in mapping.items():
            x, y, _, _ = boxes[object_id]
            (tw, th), baseline = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, .65, 2)
            w, h = tw + 8, th + baseline + 6
            left, top = min(x, max(0, width-w)), max(0, y-h)
            mask[max(0, top-3):min(height, top+h+4),
                 max(0, left-3):min(width, left+w+4)] = True
    return mask


def prepare(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    study = json.loads((BASE / "study.json").read_text())
    rng = np.random.default_rng(SEED)
    jobs, scenes = [], []
    for trial in study["trials"]:
        source = BASE / trial["run_directory"]
        assert sha(source / "results.json") == trial["original_results_sha256"]
        manifest = json.loads((source / "manifest.json").read_text())
        detections = vlm_module.load_localized_objects(source / "point_cloud/point_cloud_localization.json")
        original = vlm_module.candidate_choice_map(detections)
        labels = list(original)
        audit = json.loads((source / "identity_audit.json").read_text())
        truth = {r["object_id"]: r["simulator_instance"] for r in audit}
        ids = {name: object_id for object_id, name in truth.items()}
        variants = {"baseline": original}
        for family, pair in PAIRS.items():
            mapping = original.copy()
            first, second = [next(k for k, v in original.items() if v == ids[name]) for name in pair]
            mapping[first], mapping[second] = mapping[second], mapping[first]
            variants[family + "_swapped"] = mapping
        for i in range(3):
            mapping = dict(zip(labels, rng.permutation(list(original.values())).tolist()))
            assert mapping not in variants.values()
            variants[f"random_{i+1}"] = mapping
        baseline = cv2.imread(str(source / "wrist_boxed.png"))
        scene = root / f"scene_{trial['trial_number']:02d}"
        for variant, mapping in variants.items():
            image = scene / (variant + ".png")
            with patch.object(vlm_module, "candidate_choice_map", return_value=mapping):
                vlm_module.annotate_candidate_boxes(source / "capture/rgb.png",
                    source / "point_cloud/point_cloud_localization.json", image)
            rendered = cv2.imread(str(image))
            mask = label_pixels(detections, [original, mapping], baseline.shape)
            assert np.array_equal(rendered[~mask], baseline[~mask])
            if variant == "baseline":
                assert sha(image) == sha(source / "wrist_boxed.png")
            for family, pair in PAIRS.items():
                if variant.endswith("_swapped") and variant != family + "_swapped":
                    continue
                inverse = {v: k for k, v in mapping.items()}
                earlier = min(pair, key=lambda name: labels.index(inverse[ids[name]]))
                for target in pair:
                    request = json.loads((source / target / "vlm_request.json").read_text())
                    args = request_arguments(image.read_bytes(), request["instruction"], MODEL,
                                             camera_description=manifest["camera_description"])
                    assert args["messages"][0]["content"] == request["system_prompt"]
                    assert request["model"] == MODEL
                    jobs.append({"scene": trial["trial_number"], "source_directory": str(source),
                        "family": family, "target": target, "pair": pair, "variant": variant,
                        "condition": "pair_swapped" if variant.endswith("_swapped") else
                                     "randomized" if variant.startswith("random_") else "baseline",
                        "image_path": str(image), "image_sha256": sha(image),
                        "mapping_local_only": mapping, "truth_evaluation_only": truth,
                        "target_label": inverse[ids[target]], "earlier_labeled_family_member": earlier,
                        "instruction": request["instruction"], "system_prompt": request["system_prompt"],
                        "target_description": request["instruction"].removeprefix("Pick up the ").removesuffix(" and hold it above the workspace."),
                        "camera_description": manifest["camera_description"], "threshold": request["threshold"]})
        scenes.append({"trial": trial["trial_number"], "source_directory": str(source),
                       "rgb_sha256": sha(source / "capture/rgb.png"),
                       "original_results_sha256": sha(source / "results.json")})
    assert len(jobs) == 200
    order = np.random.default_rng(SEED + 1).permutation(len(jobs))
    jobs = [jobs[int(i)] for i in order]
    save(root / "jobs.json", jobs)
    save(root / "manifest.json", {"model": MODEL, "random_seed": SEED,
        "queries": len(jobs), "scenes": scenes, "workers": 4,
        "conditions": "Fresh original labels; swap only queried pair; three random full-scene label permutations.",
        "controls": "Original image size, boxes, pixels outside label regions, instructions, prompt and API settings preserved.",
        "label_rendering_note": "Letters and their fitted white backgrounds change; no object or box is moved.",
        "new_robot_trials": 0, "human_correction": False,
        "source_hashes": {str(p): sha(p) for p in (Path(__file__), Path('vlm_module.py'),
                                                   Path('simulation/libero_joint_association.py'))}})
    print(f"PREPARED: {root}; 60 images, {len(jobs)} requests; all pixel and prompt controls verified.", flush=True)


def query(root, job):
    case = root / f"scene_{job['scene']:02d}" / job["variant"] / job["target"]
    result_path = case / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    image = Path(job["image_path"])
    assert sha(image) == job["image_sha256"]
    args = request_arguments(image.read_bytes(), job["instruction"], MODEL,
                             camera_description=job["camera_description"])
    assert args["messages"][0]["content"] == job["system_prompt"]
    save(case / "request.json", {**job, "api_settings": {k: v for k, v in args.items() if k != "messages"}})
    raw_path = case / "provider_response.json"
    if raw_path.exists():
        raw = json.loads(raw_path.read_text())
    else:
        with OpenAI(timeout=60.) as client:
            raw = client.chat.completions.create(**args).model_dump(mode="json")
        save(raw_path, raw)
    inferences, parsed = joint_inferences(raw, job["instruction"], job["mapping_local_only"],
                                         [job["target_description"]])
    inference = inferences[0]
    valid = inference["diagnostics"]["joint_format_valid"] and inference["diagnostics"]["required_names_complete"]
    selected = job["truth_evaluation_only"].get(inference["vlm_object_id"]) if valid else None
    score = inference["association_score"]
    result = {k: job[k] for k in ("scene", "family", "target", "variant", "condition", "target_label", "earlier_labeled_family_member")}
    result.update(selected_object=selected, choice_label=inference["diagnostics"]["choice_label"],
        valid_response=valid, raw_score=score, correct=valid and selected == job["target"],
        accepted=score is not None and score >= job["threshold"],
        selected_earlier_family_member=valid and selected == job["earlier_labeled_family_member"],
        generated_output=parsed["generated_output_text"], provider_model=raw.get("model"),
        system_fingerprint=raw.get("system_fingerprint"))
    save(result_path, result)
    return result


def run(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    for name, digest in manifest["source_hashes"].items():
        assert sha(Path(name)) == digest, name
    jobs = json.loads((root / "jobs.json").read_text())
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(query, root, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                results.append(future.result())
            except Exception as error:
                errors.append({"scene": job["scene"], "target": job["target"], "variant": job["variant"], "error": repr(error)})
                save(root / "errors.json", errors)
            save(root / "results.json", results)
            n = len(results) + len(errors)
            if n % 10 == 0 or errors:
                print(f"PROGRESS {n}/200; completed={len(results)}; errors={len(errors)}", flush=True)
    print(f"COMPLETE: {len(results)}/200 responses; {len(errors)} errors; {root}", flush=True)


if __name__ == "__main__":
    output = BASE / ("label_order_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    prepare(output)
    run(output)
