"""Paired output-representation ablation on immutable LIBERO scene files.

Run as a module with `pilot`, `full`, or `check`. Production perception and
HITL are not invoked. Existing completed responses are reused on resume.
"""

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from threading import Lock
import time

from prompt import _vlm_prompt
from vlm_module import candidate_choice_map, candidate_prompt_records, load_localized_objects


SCENE_ROOT = Path("outputs/simulation/experiments/put_all_objects_into_basket/openai_vlm_confidence/candidate_normalized_500")
FROZEN_ROOT = Path("experiments/put_all_objects_into_basket/proposed_framework/vlm_confidence_500")
OUTPUT_ROOT = SCENE_ROOT.parent / "output_format_ablation"
MODEL = "gpt-4.1-mini-2025-04-14"
CONDITIONS = ("compact", "object_id")
PILOT_STATES = (0, 1)
WORKERS = 4
REQUEST_INTERVAL_S = 1.3
_request_lock = Lock()
_next_request_time = 0.0
SYSTEM_PROMPT = (
    "Associate one semantic target with an already-localized "
    "candidate. Return only the requested choice label."
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def build_prompt(target, detections, condition):
    choices = candidate_choice_map(detections)
    candidates = candidate_prompt_records(detections, choices)
    if condition == "compact":
        return _vlm_prompt(target, candidates), {**choices, "N": None}
    if condition != "object_id":
        raise ValueError(f"Unknown condition: {condition}")
    for candidate in candidates:
        candidate["choice"] = candidate["object_id"]
    # Keep every sentence and metadata field identical except the NONE label
    # and output/candidate representation. Images retain their existing IDs.
    prompt = _vlm_prompt(target, candidates)
    prompt = prompt.replace("\nN means none", "\nnone means none")
    if not prompt.endswith(", N"):
        raise ValueError("The shared prompt's choice-list layout has changed.")
    prompt = prompt[:-1] + "none"
    return prompt, {**{value: value for value in choices.values()}, "none": None}


def request_arguments(image_bytes, prompt):
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii"),
                    "detail": "high",
                }},
            ]},
        ],
        "temperature": 0,
        "max_completion_tokens": 8,
        "logprobs": True,
        "top_logprobs": 20,
    }


def extract_decision(text, tokens, choices):
    """Find the valid leading decision span; sum only overlapping token logprobs.

    Standalone surrounding formatting tokens are excluded. A token that mixes
    the decision with formatting is indivisible and included in full. Internal
    underscores in object IDs ARE decision characters, even as separate tokens.
    No per-token or length normalization is performed.
    """
    labels = "|".join(re.escape(key) for key in sorted(choices, key=len, reverse=True))
    formatting = r"[\s`\"'\[\](){}.,:;*!]"
    match = re.match(rf"^{formatting}*(?P<label>{labels})(?=$|{formatting})", text, re.I)
    result = {"choice": None, "predicted_object_id": None, "decision_tokens": [],
              "decision_token_count": 0, "raw_log_probability": None,
              "raw_sequence_likelihood": None}
    if match is None:
        return {**result, "score_unavailable_reason": "invalid output choice"}
    choice = next(key for key in choices if key.casefold() == match["label"].casefold())
    result.update(choice=choice, predicted_object_id=choices[choice])
    if "".join(str(item["token"]) for item in tokens) != text:
        return {**result, "score_unavailable_reason": "token text unavailable or mismatches output"}
    start, end = match.span("label")
    offset = 0
    selected = []
    for index, item in enumerate(tokens):
        token_end = offset + len(item["token"])
        if offset < end and token_end > start:
            selected.append({"index": index, "token": item["token"],
                             "logprob": item.get("logprob")})
        offset = token_end
    result.update(decision_tokens=selected, decision_token_count=len(selected))
    if not selected or any(
        item["logprob"] is None or not math.isfinite(item["logprob"])
        or item["logprob"] > 0 or item["logprob"] <= -9999 for item in selected
    ):
        return {**result, "score_unavailable_reason": "decision-token logprobs unavailable"}
    logprob = math.fsum(item["logprob"] for item in selected)
    result.update(raw_log_probability=logprob, raw_sequence_likelihood=math.exp(logprob))
    return result


def load_samples():
    frozen = json.loads((FROZEN_ROOT / "dataset_manifest.json").read_text())
    local = json.loads((SCENE_ROOT / "manifest.json").read_text())
    if frozen["samples"] != local["samples"]:
        raise ValueError("Local scene manifest differs from the frozen experiment.")
    samples = sorted(frozen["samples"], key=lambda row: (row["task_index"], row["initial_state_index"]))
    if len(samples) != 500 or len({row["sample_id"] for row in samples}) != 500:
        raise ValueError("Expected 500 distinct frozen scenes.")
    for sample in samples:
        for field in ("image_path", "localization_path"):
            if not (SCENE_ROOT / sample[field]).is_file():
                raise FileNotFoundError(SCENE_ROOT / sample[field])
    return samples


def experiment_spec():
    return {
        "schema_version": 1, "model": MODEL, "temperature": 0,
        "max_completion_tokens": 8, "top_logprobs": 20, "image_detail": "high",
        "conditions": list(CONDITIONS), "pilot_initial_states": list(PILOT_STATES),
        "manifest_sha256": digest((FROZEN_ROOT / "dataset_manifest.json").read_bytes()),
        "prompt_source_sha256": digest(Path("prompt.py").read_bytes()),
        "selection": "actual generated decision; candidate-normalized scores unused",
        "likelihood": "exp(sum(decision-bearing output-token logprobs)), no length normalization",
        "pair_order": "compact first when (task_index + initial_state_index) is even; otherwise object_id first",
        "analysis_protocol": "fresh >=90% and >=95% thresholds selected independently for each format using calibration/validation only; frozen thresholds evaluated on test",
        "pilot_role": "implementation check only; proceed to full if valid and too few paired errors to support a representation recommendation",
    }


def paced_completion(client, arguments):
    """Respect the account TPM budget; retry 429s without altering the request."""
    from openai import RateLimitError

    global _next_request_time
    for attempt in range(6):
        with _request_lock:
            time.sleep(max(0.0, _next_request_time - time.monotonic()))
            _next_request_time = time.monotonic() + REQUEST_INTERVAL_S
        try:
            return client.chat.completions.create(**arguments)
        except RateLimitError:
            if attempt == 5:
                raise
            time.sleep(min(5 * 2**attempt, 30))


def run_pair(sample):
    from openai import OpenAI

    image_path = SCENE_ROOT / sample["image_path"]
    localization_path = SCENE_ROOT / sample["localization_path"]
    image_bytes = image_path.read_bytes()
    detections = load_localized_objects(localization_path)
    hashes = {"image_sha256": digest(image_bytes),
              "localization_sha256": digest(localization_path.read_bytes())}
    order = CONDITIONS if (sample["task_index"] + sample["initial_state_index"]) % 2 == 0 else CONDITIONS[::-1]
    rows = []
    with OpenAI(timeout=120, max_retries=3) as client:
        for order_index, condition in enumerate(order):
            path = OUTPUT_ROOT / "responses" / f"{sample['sample_id']}_{condition}.json"
            prompt, choices = build_prompt(sample["target_description"], detections, condition)
            prompt_hash = digest(prompt.encode())
            if path.is_file():
                row = json.loads(path.read_text())
                if any(row[key] != value for key, value in hashes.items()) or row["prompt_sha256"] != prompt_hash:
                    raise ValueError(f"Frozen inputs changed for {path}")
                rows.append(row)
                continue
            response = paced_completion(client, request_arguments(image_bytes, prompt))
            raw = response.model_dump(mode="json", exclude_none=True)
            output = raw["choices"][0]
            text = output["message"].get("content") or ""
            token_records = output.get("logprobs", {}).get("content") or []
            decision = extract_decision(text, token_records, choices)
            row = {key: sample[key] for key in (
                "sample_id", "partition", "task_index", "initial_state_index",
                "target_description", "ground_truth_object_id", "target_localized",
            )}
            row.update({
                "condition": condition, "requested_model": MODEL, "model": raw["model"],
                "pair_order_index": order_index, "image_path": str(image_path),
                "localization_path": str(localization_path), **hashes,
                "prompt": prompt, "prompt_sha256": prompt_hash, "candidate_map": choices,
                "generated_output_text": text, **decision,
                "correct": decision["choice"] is not None and decision["predicted_object_id"] == sample["ground_truth_object_id"],
                "provider_response": raw, "captured_at": datetime.now(timezone.utc).isoformat(),
            })
            if raw["model"] != MODEL:
                raise RuntimeError(f"Unexpected returned model {raw['model']!r}")
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(row, indent=2), encoding="utf-8")
            temporary.replace(path)
            rows.append(row)
    return rows


def run(stage):
    samples = load_samples()
    if stage == "check":
        print(f"Verified {len(samples)} frozen scene/image/localization pairs.")
        return
    if stage not in ("pilot", "full"):
        raise ValueError("Use pilot, full, or check.")
    if stage == "pilot":
        samples = [row for row in samples if row["initial_state_index"] in PILOT_STATES]
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "responses").mkdir(exist_ok=True)
    spec_path = OUTPUT_ROOT / "protocol.json"
    spec = experiment_spec()
    if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
        raise ValueError("Experiment protocol changed; do not mix runs.")
    if not spec_path.exists():
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    completed = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(run_pair, sample): sample for sample in samples}
        for future in as_completed(futures):
            rows = future.result()
            completed += 1
            print(f"{stage}: {completed}/{len(samples)} pairs | " + " | ".join(
                f"{row['condition']} {row['sample_id']} {row['generated_output_text']!r} "
                f"correct={row['correct']} tokens={row['decision_token_count']} p={row['raw_sequence_likelihood']}"
                for row in rows), flush=True)
    records = [json.loads(path.read_text()) for path in sorted((OUTPUT_ROOT / "responses").glob("*.json"))]
    ids = {sample["sample_id"] for sample in samples}
    records = [row for row in records if row["sample_id"] in ids]
    (OUTPUT_ROOT / f"{stage}_records.json").write_text(json.dumps({"protocol": spec, "records": records}, indent=2), encoding="utf-8")
    for condition in CONDITIONS:
        rows = [row for row in records if row["condition"] == condition]
        print(condition, "n=", len(rows), "correct=", sum(row["correct"] for row in rows),
              "scored=", sum(row["raw_sequence_likelihood"] is not None for row in rows),
              "mean_decision_tokens=", sum(row["decision_token_count"] for row in rows)/len(rows), flush=True)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) == 2 else "pilot")
