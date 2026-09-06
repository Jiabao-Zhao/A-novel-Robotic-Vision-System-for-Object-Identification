"""Evaluation-only 50-scene Y/N verification pilot; use `check` or `run`.

No production gate, HITL, CAD, localization, or planning changes. Only calibration
states 0--4 are opened. Each successful API response is immutable on resume.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

from scripts.run_vlm_output_format_ablation import (
    MODEL, OUTPUT_ROOT as REFERENCE_ROOT, SCENE_ROOT, SYSTEM_PROMPT,
    build_prompt, digest, extract_decision, paced_completion, request_arguments,
)
from vlm_module import candidate_prompt_records, load_localized_objects


OUTPUT_ROOT = SCENE_ROOT.parent / "candidate_verification_pilot"
VERIFICATION_SYSTEM = (
    "Verify whether the named already-localized candidate matches the semantic "
    "target. Return exactly Y or N."
)
VERIFICATION_INSTRUCTION = """Target description: {target_description}
Evaluate candidate: {object_id}

Does this specific candidate match the target description?
Evaluate only the named candidate; other objects provide context.
Do not assume the target is present. Multiple candidates may match, or none may match.
Return exactly Y for yes or N for no. No explanation."""


def verification_prompt(target_description, object_id, candidates):
    return (VERIFICATION_INSTRUCTION.format(target_description=target_description,
                                            object_id=object_id)
            + "\n\nCandidate map and depth-derived metadata:\n"
            + json.dumps(candidates, separators=(",", ":")))


def verification_score(raw):
    """YES likelihood relative to NO at one shared answer-token position.

    Require a single decision-bearing token and the counterpart with the exact
    same whitespace affixes. Never merge token spellings or search later steps.
    Alternative token strings/bytes returned by the API verify tokenization.
    This Y/N normalization is within one candidate, never across candidates.
    """
    output = raw["choices"][0]
    text = output["message"].get("content") or ""
    tokens = (output.get("logprobs") or {}).get("content") or []
    result = {"generated_output_text": text, "yes_score": None,
              "answer_label": text.strip() if text.strip() in ("Y", "N") else None}
    if output["finish_reason"] != "stop" or result["answer_label"] is None:
        return {**result, "score_unavailable_reason": "invalid or unfinished Y/N answer"}
    decision = extract_decision(text, tokens, {"Y": "Y", "N": "N"})
    result["decision_tokens"] = decision["decision_tokens"]
    if decision["decision_token_count"] != 1 or decision["raw_log_probability"] is None:
        return {**result, "score_unavailable_reason": "single scored answer token unavailable"}
    selected = decision["decision_tokens"][0]
    position = tokens[selected["index"]]
    return {**result, "answer_token_position": selected["index"], **answer_token_score(position)}


def answer_token_score(position):
    """Shared evaluation-only Y/N scoring at one actual provider token position."""
    result = {"yes_score": None}
    token = position["token"]
    label = token.strip()
    if label not in ("Y", "N"):
        return {**result, "score_unavailable_reason": "unsupported answer token spelling"}
    chosen = position.get("logprob")
    if (not isinstance(chosen, (float, int)) or not math.isfinite(chosen)
            or not -9999 < chosen <= 0):
        return {**result, "score_unavailable_reason": "chosen answer logprob unavailable"}
    offset = token.index(label)
    spellings = {choice: token[:offset] + choice + token[offset + 1:] for choice in ("Y", "N")}
    result.update(answer_token_spellings=spellings, available_answer_logprobs={})
    for choice, spelling in spellings.items():
        matches = [item for item in [position, *(position.get("top_logprobs") or [])]
                   if item["token"] == spelling]
        values = [item.get("logprob") for item in matches]
        if not values or any(value is None or not isinstance(value, (float, int))
                             or not math.isfinite(value) or not -9999 < value <= 0
                             for value in values):
            continue
        if len(set(values)) != 1:
            return {**result, "score_unavailable_reason": "inconsistent duplicate answer logprobs"}
        result["available_answer_logprobs"][choice] = values[0]
    available = result["available_answer_logprobs"]
    if set(available) != {"Y", "N"}:
        return {**result, "score_unavailable_reason": "Y or N alternative missing/invalid at answer position"}
    high = max(available.values())
    log_normalizer = high + math.log(math.fsum(math.exp(v - high) for v in available.values()))
    result.update(yes_log_probability=available["Y"], no_log_probability=available["N"],
                  yes_score=math.exp(available["Y"] - log_normalizer))
    return result


def scene_prediction(candidate_scores):
    if not candidate_scores or any(value is None for value in candidate_scores.values()):
        return {"predicted_object_id": None, "support": None, "gap": None,
                "top_tie": False, "score_available": False}
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in candidate_scores.values()):
        raise ValueError("Invalid candidate verification score.")
    ranked = sorted(candidate_scores, key=lambda key: (-candidate_scores[key], key))
    support = candidate_scores[ranked[0]]
    gap = support - candidate_scores[ranked[1]] if len(ranked) > 1 else None
    tied = gap == 0
    # No arbitrary identity assignment in a tie; this is unresolved, not absent.
    return {"predicted_object_id": None if tied else ranked[0], "support": support,
            "gap": gap, "top_tie": tied, "score_available": True}


def verification_gate(prediction, tau_support, tau_gap=0):
    if not 0 <= tau_support <= 1 or not 0 <= tau_gap <= 1:
        raise ValueError("Provisional thresholds must be in [0, 1].")
    if (not prediction["score_available"] or prediction["top_tie"]
            or prediction["gap"] is None):
        return "deferred"
    return ("vlm_accepted" if prediction["support"] >= tau_support
            and prediction["gap"] >= tau_gap else "deferred")


def prepare_scenes():
    scenes = []
    # Enumerate task directories, not the 500-response file or test scenes.
    for task_dir in sorted((SCENE_ROOT / "scenes").glob("task_*")):
        for state in range(5):
            sample = json.loads((task_dir / f"init_state_{state:02d}" / "sample.json").read_text())
            if sample["partition"] != "calibration" or sample["initial_state_index"] != state:
                raise ValueError("Only calibration states 0--4 are allowed.")
            image_path = SCENE_ROOT / sample["image_path"]
            localization_path = SCENE_ROOT / sample["localization_path"]
            image_bytes = image_path.read_bytes()
            detections = load_localized_objects(localization_path)
            prompt, choices = build_prompt(sample["target_description"], detections, "compact")
            candidates = candidate_prompt_records(detections, {k: v for k, v in choices.items() if v is not None})
            hashes = {"image_sha256": digest(image_bytes),
                      "localization_sha256": digest(localization_path.read_bytes()),
                      "prompt_sha256": digest(prompt.encode())}
            # Verify the original marked input/prompt. Old predictions are never reused.
            reference = json.loads((REFERENCE_ROOT / "responses" / f"{sample['sample_id']}_compact.json").read_text())
            if (any(reference[k] != v for k, v in hashes.items())
                    or reference["ground_truth_object_id"] != sample["ground_truth_object_id"]
                    or reference["candidate_map"] != choices):
                raise ValueError(f"Frozen input mismatch: {sample['sample_id']}")
            scene = {**sample, **hashes, "image_path": str(image_path),
                     "localization_path": str(localization_path), "prompt": prompt,
                     "candidate_map": choices, "candidate_metadata": candidates,
                     "ground_truth_sha256": digest((SCENE_ROOT / sample["evaluation_ground_truth_path"]).read_bytes())}
            scene["input_sha256"] = digest(json.dumps(scene, sort_keys=True).encode())
            scenes.append((scene, image_bytes))
    pairs = {(s["task_index"], s["initial_state_index"]) for s, _ in scenes}
    if len(scenes) != 50 or pairs != {(task, state) for task in range(10) for state in range(5)}:
        raise ValueError("Expected exactly 50 distinct calibration scenes.")
    return scenes


def call_arguments(image_bytes, prompt, verification=False):
    args = request_arguments(image_bytes, prompt)
    if verification:
        args["messages"][0]["content"] = VERIFICATION_SYSTEM
    return args


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def run_scene(scene, image_bytes, output_root=None):
    from openai import OpenAI

    output_root = OUTPUT_ROOT if output_root is None else Path(output_root)
    ids = [row["object_id"] for row in scene["candidate_metadata"]]
    calls = [("baseline", scene["prompt"])] + [
        (object_id, verification_prompt(scene["target_description"], object_id, scene["candidate_metadata"]))
        for object_id in ids]
    # Counterbalance block order; every call is a fresh two-message request.
    block_order = (scene["task_index"] + scene["initial_state_index"]
                   if "initial_state_index" in scene else scene["sample_index"])
    if block_order % 2:
        calls = calls[1:] + calls[:1]
    rows = {}
    with OpenAI(timeout=60, max_retries=2) as client:
        for call_id, prompt in calls:
            args = call_arguments(image_bytes, prompt, call_id != "baseline")
            settings = {k: v for k, v in args.items() if k != "messages"}
            settings.update(system_prompt=args["messages"][0]["content"], prompt=prompt,
                            image_detail="high", image_sha256=scene["image_sha256"])
            settings_hash = digest(json.dumps(settings, sort_keys=True).encode())
            path = output_root / "responses" / f"{scene['sample_id']}_{call_id}.json"
            if path.exists():
                row = json.loads(path.read_text())
                if row["settings_sha256"] != settings_hash or row["input_sha256"] != scene["input_sha256"]:
                    raise ValueError(f"Resume input/settings mismatch: {path}")
            else:
                response = paced_completion(client, args)
                raw = response.model_dump(mode="json", exclude_none=True)
                # Save the full response BEFORE parsing so failures remain auditable.
                row = {"sample_id": scene["sample_id"], "call_id": call_id,
                       "input_sha256": scene["input_sha256"], "settings_sha256": settings_hash,
                       "settings": settings, "provider_response": raw,
                       "captured_at": datetime.now(timezone.utc).isoformat()}
                save_json(path, row)
            raw = row["provider_response"]
            if raw["model"] != MODEL:
                raise RuntimeError(f"Returned snapshot differs: {raw['model']}")
            if call_id == "baseline":
                output = raw["choices"][0]
                text = output["message"].get("content") or ""
                result = extract_decision(text, (output.get("logprobs") or {}).get("content") or [], scene["candidate_map"])
                if output["finish_reason"] != "stop":
                    result.update(raw_log_probability=None, raw_sequence_likelihood=None,
                                  score_unavailable_reason="unfinished baseline response")
                result.update(generated_output_text=text,
                              correct=result["choice"] is not None and result["predicted_object_id"] == scene["ground_truth_object_id"])
            else:
                result = verification_score(raw)
            rows[call_id] = {**result, "response_path": str(path), "settings_sha256": settings_hash}
    candidates = {key: rows[key] for key in ids}
    prediction = scene_prediction({key: row["yes_score"] for key, row in candidates.items()})
    prediction["correct"] = (prediction["predicted_object_id"] is not None
                             and prediction["predicted_object_id"] == scene["ground_truth_object_id"])
    result = {"scene": scene, "baseline": rows["baseline"], "candidates": candidates,
              "verification": prediction}
    save_json(output_root / "scenes" / f"{scene['sample_id']}.json", result)
    print(f"{scene['sample_id']}: baseline={rows['baseline']['correct']} "
          f"verification={prediction['correct']} support={prediction['support']} gap={prediction['gap']} "
          f"scored={sum(row['yes_score'] is not None for row in candidates.values())}/{len(ids)}", flush=True)
    return result


def run(stage):
    if stage not in ("check", "run"):
        raise ValueError("Use check or run. There is no full-experiment mode.")
    scenes = prepare_scenes()
    protocol = {
        "model": MODEL, "temperature": 0, "image_detail": "high", "top_logprobs": 20,
        "max_completion_tokens": 8, "baseline_system": SYSTEM_PROMPT,
        "verification_system": VERIFICATION_SYSTEM, "verification_instruction": VERIFICATION_INSTRUCTION,
        "sample_inputs": {s["sample_id"]: s["input_sha256"] for s, _ in scenes},
        "selection": "calibration states 0--4 for each of ten tasks; no test scenes",
        "analysis": "exploratory same-pilot threshold sweeps only; no held-out claims or deployment",
        "token_pair": "one answer token; Y/N with matching whitespace affixes at the same position",
        "score": "exp(lY-logsumexp(lY,lN)); always YES, never normalized across candidates",
        "gate": "all candidates scored, no exact top tie, support>=tau_support AND gap>=tau_gap",
        "source_sha256": digest(Path(__file__).read_bytes()),
    }
    for directory in (OUTPUT_ROOT, OUTPUT_ROOT / "responses", OUTPUT_ROOT / "scenes"):
        directory.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_ROOT / "protocol.json"
    if path.exists():
        saved_protocol = json.loads(path.read_text())
        # Source provenance can differ after a nonsemantic reuse refactor. All
        # actual prompts, request settings and input hashes still must match.
        if ({k: v for k, v in saved_protocol.items() if k != "source_sha256"}
                != {k: v for k, v in protocol.items() if k != "source_sha256"}):
            raise ValueError("Protocol changed; refuse to mix pilot settings.")
        protocol = saved_protocol
    if not path.exists():
        save_json(path, protocol)
    calls = sum(1 + len(s["candidate_metadata"]) for s, _ in scenes)
    print(f"Verified {len(scenes)} frozen calibration scenes; {calls} planned calls.", flush=True)
    if stage == "check":
        return
    results = [run_scene(*scenes[0])]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_scene, *scene) for scene in scenes[1:]]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda r: r["scene"]["sample_id"])
    save_json(OUTPUT_ROOT / "records.json", {"protocol": protocol, "records": results})
    print(f"Completed {len(results)} scenes. No production code or test split was changed/accessed.", flush=True)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) == 2 else "check")
