"""Joint Y/N pilot on 12 frozen new-object scenes; no capture or production calls.

Use check, run or analyze. Forward order is the primary arm; reverse answer order
is a sensitivity control, never an ensemble or a result-selected replacement.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

from scripts.run_libero_new_object_verification import OUTPUT_ROOT as REFERENCE_ROOT, prepare_scene
from scripts.run_vlm_candidate_verification_pilot import answer_token_score, save_json, scene_prediction
from scripts.run_vlm_output_format_ablation import MODEL, digest, paced_completion, request_arguments


OUTPUT_ROOT = REFERENCE_ROOT / "joint_verification"
CONDITIONS = ("forward", "reverse")
MAX_OUTPUT_TOKENS = 128  # The only numeric request-setting change: allow K labeled lines, not one token.
JOINT_SYSTEM = (
    "Verify whether each named already-localized candidate matches the semantic "
    "target. Return one Y or N judgment per candidate in the requested order."
)
JOINT_INSTRUCTION = """Target description: {target_description}
Answer order: {answer_order}

For each named candidate: does this specific candidate match the target description?
Evaluate each named candidate; other objects provide context.
Do not assume the target is present. Multiple candidates may match, or none may match.
Return exactly one line per candidate in the answer order above, using its persistent
object ID followed by a colon and Y for yes or N for no. No explanation or confidence numbers."""


def answer_order(scene, condition):
    ids = [c["object_id"] for c in scene["candidate_metadata"]]
    if condition not in CONDITIONS:
        raise ValueError("Unknown joint-verification condition.")
    return ids if condition == "forward" else list(reversed(ids))


def joint_prompt(scene, condition):
    # Candidate metadata/image ordering does not change in the reverse control.
    return JOINT_INSTRUCTION.format(target_description=scene["target_description"],
                                   answer_order=", ".join(answer_order(scene, condition))) + (
        "\n\nCandidate map and depth-derived metadata:\n"
        + json.dumps(scene["candidate_metadata"], separators=(",", ":")))


def joint_arguments(scene, image_bytes, condition):
    args = request_arguments(image_bytes, joint_prompt(scene, condition))
    args["messages"][0]["content"] = JOINT_SYSTEM
    args["max_completion_tokens"] = MAX_OUTPUT_TOKENS
    return args


def parse_joint_response(raw, ids):
    """Locate each Y/N by text span and score ONLY its actual answer token.

    Full responses/alternatives are retained externally. Scores at later slots
    condition on the already-generated prefix: they are NOT independent marginals.
    Malformed, duplicate, missing or reordered rows invalidate the whole format.
    A missing Y/N alternative invalidates its score and thus scene-level selection.
    """
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Expected distinct localized object IDs.")
    output = raw["choices"][0]
    text = output["message"].get("content") or ""
    tokens = (output.get("logprobs") or {}).get("content") or []
    result = {"generated_output_text": text, "format_valid": False,
              "candidates": {key: {"answer_label": None, "yes_score": None} for key in ids}}
    matches = list(re.finditer(r"(?m)^[ \t]*(object_\d+)[ \t]*:[ \t]*([YN])[ \t]*\r?$", text))
    remainder = text
    for match in reversed(matches):
        remainder = remainder[:match.start()] + remainder[match.end():]
    if (output["finish_reason"] != "stop" or remainder.strip()
            or [m[1] for m in matches] != ids):
        return {**result, "score_unavailable_reason": "unfinished/malformed/missing/duplicate/out-of-order answers"}
    result["format_valid"] = True
    result["candidates"] = {m[1]: {"answer_label": m[2], "yes_score": None} for m in matches}
    if "".join(t["token"] for t in tokens) != text:
        return {**result, "score_unavailable_reason": "token text unavailable or mismatches output"}
    spans, offset = [], 0
    for index, token in enumerate(tokens):
        spans.append((offset, offset + len(token["token"]), index))
        offset += len(token["token"])
    for slot, match in enumerate(matches):
        start, end = match.span(2)
        indexes = [index for a, b, index in spans if a < end and b > start]
        row = result["candidates"][match[1]]
        row.update(answer_slot=slot, answer_character_span=[start, end],
                   decision_tokens=[{"index": i, "token": tokens[i]["token"],
                                     "logprob": tokens[i].get("logprob")} for i in indexes])
        # Accept attached whitespace but not an ID/colon/another answer inside
        # the same token. Do not borrow alternatives from a different position.
        if len(indexes) != 1 or tokens[indexes[0]]["token"].strip() != match[2]:
            row["score_unavailable_reason"] = "single separable Y/N answer token unavailable"
            continue
        index = indexes[0]
        row.update(answer_token_position=index, **answer_token_score(tokens[index]))
    return result


def load_reference():
    from scripts.analyze_vlm_candidate_verification_pilot import validate_records

    payload = json.loads((REFERENCE_ROOT / "records.json").read_text())
    records = validate_records(payload, expected_count=12, fixed_calibration_states=False)
    for record in records:
        scene = record["scene"]
        if scene["partition"] != "exploratory_new_objects" or scene["saved_initial_state_used"]:
            raise ValueError("Only the frozen fresh-reset new-object pilot is allowed.")
        sample = json.loads((Path(scene["image_path"]).parent / "sample.json").read_text())
        checked, _ = prepare_scene(sample)
        if checked != scene:
            raise ValueError("Reference scene differs from frozen RGB/localization/metadata.")
    return records


def run_condition(reference, condition):
    from openai import OpenAI

    scene = reference["scene"]
    image_bytes = Path(scene["image_path"]).read_bytes()
    args = joint_arguments(scene, image_bytes, condition)
    settings = {k: v for k, v in args.items() if k != "messages"}
    settings.update(system_prompt=JOINT_SYSTEM, prompt=args["messages"][1]["content"][0]["text"],
                    image_detail="high", image_sha256=digest(image_bytes))
    settings_hash = digest(json.dumps(settings, sort_keys=True).encode())
    path = OUTPUT_ROOT / "responses" / f"{scene['sample_id']}_{condition}.json"
    if path.exists():
        row = json.loads(path.read_text())
        if row["settings_sha256"] != settings_hash or row["input_sha256"] != scene["input_sha256"]:
            raise ValueError(f"Resume input/settings mismatch: {path}")
    else:
        with OpenAI(timeout=60, max_retries=2) as client:
            raw = paced_completion(client, args).model_dump(mode="json", exclude_none=True)
        row = {"sample_id": scene["sample_id"], "condition": condition,
               "input_sha256": scene["input_sha256"], "settings": settings,
               "settings_sha256": settings_hash, "provider_response": raw,
               "captured_at": datetime.now(timezone.utc).isoformat()}
        save_json(path, row)  # Preserve actual response before any parsing.
    raw = row["provider_response"]
    if raw["model"] != MODEL:
        raise ValueError("Returned model snapshot differs from the frozen experiment.")
    parsed = parse_joint_response(raw, answer_order(scene, condition))
    prediction = scene_prediction({key: c["yes_score"] for key, c in parsed["candidates"].items()})
    prediction["correct"] = prediction["predicted_object_id"] == scene["ground_truth_object_id"]
    result = {"scene": scene, "condition": condition, **parsed, "verification": prediction,
              "response_path": str(path), "settings_sha256": settings_hash}
    save_json(OUTPUT_ROOT / "scenes" / path.name, result)
    print(f"{scene['sample_id']} {condition}: format={parsed['format_valid']} "
          f"prediction={prediction['predicted_object_id']} correct={prediction['correct']} "
          f"support={prediction['support']} gap={prediction['gap']} "
          f"scores={sum(c['yes_score'] is not None for c in parsed['candidates'].values())}/{len(parsed['candidates'])}", flush=True)
    return result


def run(stage):
    if stage not in ("check", "run", "analyze"):
        raise ValueError("Use check, run or analyze; no automatic full experiment.")
    if stage == "analyze":
        from scripts.analyze_vlm_joint_verification_pilot import analyze
        return analyze()
    reference = load_reference()
    for directory in (OUTPUT_ROOT, OUTPUT_ROOT / "responses", OUTPUT_ROOT / "scenes"):
        directory.mkdir(parents=True, exist_ok=True)
    settings = request_arguments(b"", "")
    protocol = {"model": MODEL, "temperature": settings["temperature"], "image_detail": "high",
                "top_logprobs": settings["top_logprobs"], "max_completion_tokens": MAX_OUTPUT_TOKENS,
                "system": JOINT_SYSTEM, "instruction": JOINT_INSTRUCTION,
                "conditions": list(CONDITIONS), "primary_condition": "forward",
                "sample_inputs": {r["scene"]["sample_id"]: r["scene"]["input_sha256"] for r in reference},
                "reference_sha256": digest((REFERENCE_ROOT / "records.json").read_bytes()),
                "selection": "all 12 usable new-object captures; no reset or saved initial-state access",
                "score": "per answer slot exp(lY-logsumexp(lY,lN)); conditional on preceding output",
                "unavailability": "missing/invalid alternative for any candidate or invalid format defers scene",
                "gate": "support>=tau_support AND gap>=tau_gap; exact top-score ties defer",
                "reference_gates": {"support": .9, "gap": .1},
                "analysis": "exploratory; no test set or post-hoc choice between answer orders",
                "confounds": "cached reference versus fresh joint calls; changed output protocol/budget; reversal changes answer-order instruction",
                "api_documentation": "https://developers.openai.com/api/reference/resources/chat"}
    path = OUTPUT_ROOT / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Protocol changed; refuse to mix runs.")
    if not path.exists():
        save_json(path, protocol)
    print(f"{len(reference)} frozen scenes; 24 joint calls (12 forward, 12 reverse); baseline/independent cached.", flush=True)
    if stage == "check":
        return
    jobs = []
    for r in reference:
        conditions = CONDITIONS if r["scene"]["sample_index"] % 2 == 0 else CONDITIONS[::-1]
        jobs.extend((r, condition) for condition in conditions)
    # Inspect the first real output before dispatching the rest; malformed
    # output is logged, not repaired through extra model calls.
    results = [run_condition(*jobs[0])]
    if not results[0]["format_valid"]:
        raise RuntimeError("Pilot's first response violates the output format; inspect saved raw response before continuing.")
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_condition, *job) for job in jobs[1:]]
        results.extend(future.result() for future in futures)
    results.sort(key=lambda r: (r["scene"]["sample_id"], r["condition"]))
    save_json(OUTPUT_ROOT / "records.json", {"protocol": protocol, "records": results})


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) == 2 else "check")
