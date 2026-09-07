"""Joint letter:name association, 10 frozen calibration scenes / 40 requests.

Only role + task instruction + existing marked image reach the VLM. Ground truth
is evaluation-only. No production inference, HITL, CAD, or robot execution.
Run check/probe/run/report from the repository root; successful calls resume.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import inspect
import json
import math
from pathlib import Path
import re
import sys

from scripts.run_vlm_output_format_ablation import (
    SCENE_ROOT, digest, paced_completion, request_arguments as baseline_arguments,
)
from scripts.run_vlm_candidate_verification_pilot import save_json


OUTPUT_ROOT = SCENE_ROOT.parent / "multi_object_joint_pilot"
SOURCE_INPUTS = SCENE_ROOT.parent / "explanation_expansion" / "inputs.json"
AUDIT_ROOT = SCENE_ROOT.parent / "candidate_identity_audit" / "identities"
MODELS = ("gpt-4.1-mini-2025-04-14", "gpt-5.2-2025-12-11")
TARGET_COUNTS = (2, 3)
STATE = 1
WORKERS = 2
# Fixed before inference; extra targets must have unique, matched audit IDs.
EXTRA_ORDER = ("milk", "tomato sauce", "alphabet soup", "orange juice",
               "salad dressing", "bbq sauce", "ketchup", "cream cheese",
               "butter", "chocolate pudding")
ROLE = """You identify objects mentioned in a task instruction by associating them
with already-localized candidates in the marked image.
Return one line per distinct object mentioned in the instruction:
<existing image letter>: <object description from the instruction>
Use the letters already shown in the image. Preserve the instruction's object
names and identifying modifiers. Do not classify unrelated objects.
Do not output explanations, confidence numbers, coordinates, task roles, or actions.
If a mentioned object is not present, return N: <object description>."""


def request_arguments(image_bytes, instruction, model):
    if model not in MODELS:
        raise ValueError("Unknown pilot model.")
    args = baseline_arguments(image_bytes, "")
    args.update(model=model, max_completion_tokens=128)
    args["messages"][0]["content"] = ROLE + "\n\nTask instruction:\n" + instruction
    # Reuse the exact image bytes/detail; do not send targets, maps or metadata.
    args["messages"][1]["content"] = [args["messages"][1]["content"][1]]
    if model == MODELS[1]:
        args.update(reasoning_effort="none", top_logprobs=5)
    return args


def normalized_name(text):
    return " ".join(text.casefold().split())


def token_likelihood(tokens):
    """Original provider probabilities only; no clipping or length normalization."""
    if not tokens or any(not isinstance(t.get("logprob"), (int, float))
                         or not math.isfinite(t["logprob"])
                         or not -9999 < t["logprob"] <= 0 for t in tokens):
        return None, None
    logprob = math.fsum(t["logprob"] for t in tokens)
    return logprob, math.exp(logprob)


def parse_response(raw, mapping, instruction):
    """Parse without target ground truth; both names and marks encode decisions.

    Full score includes all output tokens. Per-entry score includes tokens
    overlapping the mark or name, excluding separable colons/whitespace. Mixed
    tokens are indivisible. No score is extracted from a retokenized string.
    """
    choice = raw["choices"][0]
    text = choice["message"].get("content") or ""
    tokens = (choice.get("logprobs") or {}).get("content") or []
    aligned = bool(tokens) and "".join(t["token"] for t in tokens) == text
    stopped = choice["finish_reason"] == "stop"
    full_logprob, full_score = token_likelihood(tokens) if aligned and stopped else (None, None)
    entries, errors, seen = [], [], set()
    spans, offset = [], 0
    for index, token in enumerate(tokens):
        end = offset + len(token["token"])
        spans.append((offset, end, {"index": index, "token": token["token"], "logprob": token.get("logprob")}))
        offset = end
    for line in re.finditer(r"[^\r\n]+", text):
        if not line[0].strip():
            continue
        match = re.fullmatch(r"\s*(?P<label>[A-Z]):[ \t]*(?P<name>\S(?:.*?\S)?)[ \t]*", line[0])
        if match is None or match["label"] not in mapping:
            errors.append("Invalid letter:name line: " + line[0])
            continue
        name = normalized_name(match["name"])
        if name in seen:
            errors.append("Duplicate object description: " + name)
        seen.add(name)
        # Only phrase containment, not hidden expected-target extraction.
        if not re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", normalized_name(instruction)):
            errors.append("Object description not preserved from instruction: " + name)
        fields = [(line.start() + a, line.start() + b) for a, b in
                  (match.span("label"), match.span("name"))]
        selected = [t for start, end, t in spans if any(start < b and end > a for a, b in fields)] if aligned else []
        lp, score = token_likelihood(selected) if stopped else (None, None)
        entries.append({"choice_label": match["label"], "target_description": name,
                        "predicted_object_id": mapping[match["label"]],
                        "decision_tokens": selected, "decision_token_count": len(selected),
                        "raw_log_probability": lp, "raw_association_likelihood": score})
    # A token crossing two entries cannot yield separate entry likelihoods.
    indices = [t["index"] for e in entries for t in e["decision_tokens"]]
    for entry in entries:
        if any(indices.count(t["index"]) > 1 for t in entry["decision_tokens"]):
            entry.update(raw_log_probability=None, raw_association_likelihood=None,
                         score_unavailable_reason="one provider token spans multiple associations")
    if not entries:
        errors.append("No associations returned.")
    if not stopped:
        errors.append("Unfinished provider response.")
    format_valid = not errors
    return {"generated_output_text": text, "entries": entries, "format_valid": format_valid,
            "format_errors": errors, "full_raw_log_probability": full_logprob,
            "full_response_likelihood": full_score, "output_token_count": len(tokens),
            "joint_gate_score": full_score if format_valid else None,
            "score_unavailable_reason": (None if full_score is not None else
                                         "Missing, invalid, sentinel or unaligned output logprobs; or unfinished response.")}


def evaluate_response(parsed, ground_truth):
    """Ground truth enters only after parsing; missing targets count as errors."""
    expected = {normalized_name(k): v for k, v in ground_truth.items()}
    rows = []
    for name, object_id in expected.items():
        matches = [e for e in parsed["entries"] if e["target_description"] == name]
        entry = matches[0] if len(matches) == 1 else {}
        rows.append({"target_description": name, "ground_truth_object_id": object_id,
                     "predicted_object_id": entry.get("predicted_object_id"),
                     "returned_count": len(matches), "correct": len(matches) == 1 and entry["predicted_object_id"] == object_id,
                     "raw_association_likelihood": entry.get("raw_association_likelihood"),
                     "raw_log_probability": entry.get("raw_log_probability"),
                     "decision_token_count": entry.get("decision_token_count", 0)})
    extras = [e["target_description"] for e in parsed["entries"] if e["target_description"] not in expected]
    return {"associations": rows, "missing_targets": [r["target_description"] for r in rows if not r["returned_count"]],
            "extra_targets": extras, "correct": parsed["format_valid"] and not extras and all(r["correct"] for r in rows)}


def autonomous(parsed, threshold):
    """Exploratory joint gate, never uses evaluation correctness/completeness."""
    score = parsed["joint_gate_score"]
    return score is not None and score >= threshold


def checked_file(path, expected):
    if digest(Path(path).read_bytes()) != expected:
        raise ValueError(f"Frozen file changed: {path}")


def select_scenes(samples):
    selected = sorted((s for s in samples if s.get("cohort") == "additional_states"
                       and s["initial_state_index"] == STATE), key=lambda s: s["task_index"])
    if [s["task_index"] for s in selected] != list(range(10)) or any(s["partition"] != "calibration" for s in selected):
        raise ValueError("Expected ten state-1 calibration scenes, never validation/test.")
    return selected


def make_tasks(sample, audit):
    if not (audit["evaluation_only"] and audit["excluded_from_vlm_inputs"]
            and audit["replay_check"]["rgb_exact_match"] and audit["target_label_agrees"]):
        raise ValueError("Unverified frozen candidate identity audit.")
    if audit["sample_id"] != sample["sample_id"] or audit["source_sha256"]["localization"] != sample["localization_sha256"]:
        raise ValueError("Audit does not describe the frozen localization.")
    objects = audit["candidates"]
    if any(c["status"] != "matched" for c in objects) or {c["object_id"] for c in objects} != set(sample["candidate_map"].values()) - {None}:
        raise ValueError("Incomplete candidate identity map.")
    identities = {c["semantic_identity"]: c["object_id"] for c in objects}
    if len(identities) != len(objects) or identities.get(sample["target_description"]) != sample["ground_truth_object_id"]:
        raise ValueError("Nonunique identity or anchor mismatch.")
    names = [sample["target_description"]] + [n for n in EXTRA_ORDER if n in identities and n != sample["target_description"]][:2]
    if len(names) != 3:
        raise ValueError("Need three uniquely audited products.")
    result = []
    for count in TARGET_COUNTS:
        instruction = f"Put the {names[0]} on top of the {names[1]}."
        if count == 3:
            instruction = instruction[:-1] + f", then place the {names[2]} next to the {names[1]}."
        result.append({"trial_id": f"{sample['sample_id']}_targets_{count}", "target_count": count,
                       "instruction": instruction, "evaluation_ground_truth": {n: identities[n] for n in names[:count]}})
    return result


def prepare():
    import openai
    from openai.resources.chat.completions import Completions

    params = inspect.signature(Completions.create).parameters
    if not all(k in params for k in ("logprobs", "top_logprobs", "reasoning_effort")):
        raise RuntimeError("Installed SDK lacks required token-logprob parameters.")
    source = json.loads(SOURCE_INPUTS.read_text())
    rows = []
    for sample in select_scenes(source["samples"]):
        audit_path = AUDIT_ROOT / (sample["sample_id"] + ".json")
        audit = json.loads(audit_path.read_text())
        for path, expected in ((sample["new_image_path"], sample["new_image_sha256"]),
                               (sample["localization_path"], sample["localization_sha256"]),
                               (SCENE_ROOT / sample["evaluation_ground_truth_path"], sample["ground_truth_sha256"])):
            checked_file(path, expected)
        if audit["source_sha256"]["target_ground_truth"] != sample["ground_truth_sha256"]:
            raise ValueError("Target audit hash mismatch.")
        if audit["replay_check"]["initial_state_sha256"] != sample["initial_state_sha256"]:
            raise ValueError("Initial-state replay hash mismatch.")
        for task in make_tasks(sample, audit):
            row = {**task, **{k: sample[k] for k in ("sample_id", "partition", "task_index", "initial_state_index",
                                                   "candidate_map", "new_image_path", "new_image_sha256", "localization_path", "localization_sha256")},
                   "identity_audit_path": str(audit_path), "identity_audit_sha256": digest(audit_path.read_bytes())}
            row["input_sha256"] = digest(json.dumps(row, sort_keys=True).encode())
            rows.append(row)
    protocol = {"models": list(MODELS), "openai_sdk_version": openai.__version__, "role": ROLE,
                "settings": {m: {k: v for k, v in request_arguments(b"", "", m).items() if k != "messages"} for m in MODELS},
                "image_detail": "high", "source_inputs_sha256": digest(SOURCE_INPUTS.read_bytes()),
                "source_sha256": {p: digest(Path(p).read_bytes()) for p in
                                  ("scripts/run_vlm_multi_object_pilot.py", "scripts/run_vlm_output_format_ablation.py",
                                   "scripts/run_vlm_candidate_verification_pilot.py")},
                "selection": "one state (1) per original task; calibration only; 20 instructions / 40 calls / 100 required associations",
                "extra_target_priority": list(EXTRA_ORDER), "none_policy": "N allowed; all requested products actually present; no absent-target performance claim",
                "score": "Full output-token product for joint response; letter+name token products per association. No normalization or fabricated logprobs.",
                "boundary": "No pre-extracted target list, ordered targets, semantic identity map, geometry metadata or explanations in request. No execution.",
                "analysis": "Exploratory thresholds only, separately per model and target count; no deployment threshold selection or held-out claim."}
    payload = {"protocol": protocol, "samples": rows}
    (OUTPUT_ROOT / "responses").mkdir(parents=True, exist_ok=True)
    path = OUTPUT_ROOT / "inputs.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError("Frozen pilot settings changed; refusing to mix runs.")
    if not path.exists():
        save_json(path, payload)
    print("Verified 10 frozen scenes / 20 instructions / 40 joint calls; all candidate identities matched.", flush=True)
    return payload


def run_trial(sample, model, client, output_root=OUTPUT_ROOT):
    for field, hash_field in (("new_image_path", "new_image_sha256"), ("localization_path", "localization_sha256"),
                              ("identity_audit_path", "identity_audit_sha256")):
        checked_file(sample[field], sample[hash_field])
    args = request_arguments(Path(sample["new_image_path"]).read_bytes(), sample["instruction"], model)
    request_hash = digest(json.dumps(args, sort_keys=True).encode())
    path = output_root / "responses" / f"{sample['trial_id']}_{model}.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["request_sha256"] != request_hash or saved["input_sha256"] != sample["input_sha256"]:
            raise ValueError("Resume input/request mismatch.")
    else:
        raw = paced_completion(client, args).model_dump(mode="json", exclude_none=True)
        saved = {"request_sha256": request_hash, "input_sha256": sample["input_sha256"],
                 "captured_at": datetime.now(timezone.utc).isoformat(), "provider_response": raw}
        save_json(path, saved)
    raw = saved["provider_response"]
    if raw["model"] != model:
        raise ValueError("Provider returned a different model snapshot.")
    parsed = parse_response(raw, sample["candidate_map"], sample["instruction"])
    row = {**{k: sample[k] for k in ("trial_id", "sample_id", "partition", "target_count", "instruction", "input_sha256",
                                    "new_image_sha256", "localization_sha256", "identity_audit_sha256")},
           **parsed, **evaluate_response(parsed, sample["evaluation_ground_truth"]),
           "model": model, "provider": "openai", "candidate_map": sample["candidate_map"],
           "usage": raw.get("usage", {}), "system_fingerprint": raw.get("system_fingerprint"),
           "response_path": str(path), "response_sha256": digest(path.read_bytes()), "request_sha256": request_hash}
    print(f"{sample['trial_id']} {model}: {sum(a['correct'] for a in row['associations'])}/{sample['target_count']} "
          f"joint={row['correct']} full_likelihood={row['full_response_likelihood']}", flush=True)
    return row


def run(stage):
    if stage == "report":
        from scripts.analyze_vlm_multi_object_pilot import report
        return report()
    if stage not in ("check", "probe", "run"):
        raise ValueError("Use check/probe/run/report; no automatic full experiment.")
    payload = prepare()
    if stage == "check":
        return
    from openai import OpenAI
    with OpenAI(timeout=120, max_retries=3) as client:
        if stage == "probe":
            return [run_trial(payload["samples"][0], model, client) for model in MODELS]
        jobs = [(s, m) for i, s in enumerate(payload["samples"]) for m in (MODELS if i % 2 == 0 else MODELS[::-1])]
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = [pool.submit(run_trial, s, m, client) for s, m in jobs]
            records = [f.result() for f in as_completed(futures)]
    records.sort(key=lambda r: (r["trial_id"], r["model"]))
    save_json(OUTPUT_ROOT / "records.json", {"protocol": payload["protocol"], "records": records})
    from scripts.analyze_vlm_multi_object_pilot import report
    report()


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) == 2 else "check")
