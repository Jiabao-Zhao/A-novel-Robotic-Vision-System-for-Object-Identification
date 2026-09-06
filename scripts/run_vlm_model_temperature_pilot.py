"""Small paired model/temperature pilot; never invokes HITL or robot execution.

Run from the repo root with `check` or `run`. Reuses immutable compact prompts
and images from the output-format experiment, not its old model predictions.
Completed responses are cached; a resume never replaces a successful call.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

from scripts.run_vlm_output_format_ablation import (
    OUTPUT_ROOT as REFERENCE_ROOT, SCENE_ROOT, SYSTEM_PROMPT,
    digest, extract_decision, load_samples, paced_completion, request_arguments,
)


OUTPUT_ROOT = SCENE_ROOT.parent / "model_temperature_pilot"
MODELS = ("gpt-4.1-mini-2025-04-14", "gpt-4o-2024-08-06")
TEMPERATURES = (0.0, 0.5, 1.0)
REPEATS = 3
INITIAL_STATES = (0, 1)
CONDITIONS = tuple((model, temperature) for model in MODELS for temperature in TEMPERATURES)


def call_key(sample_id, model, temperature, repeat):
    return f"{sample_id}_{model}_t{temperature:g}_r{repeat}"


def condition_order(sample, repeat):
    offset = (sample["task_index"] * 2 + sample["initial_state_index"] + repeat) % len(CONDITIONS)
    order = CONDITIONS[offset:] + CONDITIONS[:offset]
    return order if repeat % 2 == 0 else order[::-1]


def prepare_scene(sample):
    reference = json.loads((REFERENCE_ROOT / "responses" / f"{sample['sample_id']}_compact.json").read_text())
    for field in ("sample_id", "partition", "target_description", "ground_truth_object_id"):
        if sample[field] != reference[field]:
            raise ValueError(f"Frozen {field} differs: {sample['sample_id']}")
    image_path = SCENE_ROOT / sample["image_path"]
    localization_path = SCENE_ROOT / sample["localization_path"]
    image_bytes = image_path.read_bytes()
    hashes = {"image_sha256": digest(image_bytes),
              "localization_sha256": digest(localization_path.read_bytes()),
              "prompt_sha256": digest(reference["prompt"].encode())}
    if any(reference[field] != value for field, value in hashes.items()):
        raise ValueError(f"Input bytes differ from the frozen ablation: {sample['sample_id']}")
    return {**sample, **hashes, "image_path": str(image_path),
            "localization_path": str(localization_path),
            "prompt": reference["prompt"], "candidate_map": reference["candidate_map"]}, image_bytes


def pilot_samples():
    samples = [s for s in load_samples() if s["initial_state_index"] in INITIAL_STATES]
    if len(samples) != 20 or any(s["partition"] != "calibration" for s in samples):
        raise ValueError("Expected two calibration scenes from each of ten tasks.")
    return samples


def protocol(scenes):
    return {
        "schema_version": 1, "models": MODELS, "temperatures": TEMPERATURES,
        "repeats": REPEATS, "system_prompt": SYSTEM_PROMPT,
        "max_completion_tokens": 8, "top_logprobs": 20, "image_detail": "high",
        "top_p": "omitted, provider default; unchanged from frozen request",
        "selection": "initial states 0 and 1 of every class, calibration partition only",
        "source": str(REFERENCE_ROOT), "samples": scenes,
        "score": "exp(sum(generated decision-bearing token logprobs)); no normalization",
        "prediction": "actual generated compact label; N means no candidate",
        "same_token_comparison": "exact token text at output position zero, common returned alternatives only",
        "ordering": "cyclic condition rotation per scene/repeat; odd repeats reversed; four scene workers",
        "analysis": "descriptive pilot only; no operating threshold selection or held-out test evaluation",
        "limitations": "20 scenes, not 360 independent scenes; GPT-4o snapshot in paper unspecified; no absent-target scenes",
    }


def first_position_logprobs(raw, choices):
    """Exact token variants at position zero, never a reconstructed distribution.

    Do not merge ' A' into 'A' or compare a later decision after whitespace to
    position zero. Shared exact tokens have the same generated-prefix context.
    """
    records = (raw["choices"][0].get("logprobs") or {}).get("content") or []
    if not records:
        return {}
    position = records[0]
    result = {}
    for item in [position, *(position.get("top_logprobs") or [])]:
        token, value = item["token"], item.get("logprob")
        if (token.strip() in choices and value is not None
                and math.isfinite(value) and -9999 < value <= 0):
            if token in result and result[token] != value:
                raise ValueError("Chosen and alternative logprobs disagree for the same token.")
            result[token] = value
    return result


def response_record(scene, model, temperature, repeat, order_index, raw):
    output = raw["choices"][0]
    text = output["message"].get("content") or ""
    tokens = (output.get("logprobs") or {}).get("content") or []
    decision = extract_decision(text, tokens, scene["candidate_map"])
    if output["finish_reason"] != "stop":
        decision.update(raw_log_probability=None, raw_sequence_likelihood=None,
                        score_unavailable_reason=f"finish_reason={output['finish_reason']}")
    full_logprob = None
    if tokens and all(t.get("logprob") is not None and math.isfinite(t["logprob"])
                      and -9999 < t["logprob"] <= 0 for t in tokens):
        full_logprob = math.fsum(t["logprob"] for t in tokens)
    return {
        **scene, "requested_model": model, "model": raw["model"],
        "temperature": temperature, "repeat": repeat, "order_index": order_index,
        "call_key": call_key(scene["sample_id"], model, temperature, repeat),
        "generated_output_text": text, **decision,
        "correct": decision["choice"] is not None and decision["predicted_object_id"] == scene["ground_truth_object_id"],
        "full_response_log_probability": full_logprob,
        "first_position_choice_logprobs": first_position_logprobs(raw, scene["candidate_map"]),
        "provider_response": raw, "captured_at": datetime.now(timezone.utc).isoformat(),
    }


def arguments_for(image_bytes, prompt, model, temperature):
    return {**request_arguments(image_bytes, prompt), "model": model, "temperature": temperature}


def run_scene(scene, image_bytes):
    from openai import OpenAI

    rows = []
    with OpenAI(timeout=60, max_retries=2) as client:
        for repeat in range(REPEATS):
            for index, (model, temperature) in enumerate(condition_order(scene, repeat)):
                path = OUTPUT_ROOT / "responses" / f"{call_key(scene['sample_id'], model, temperature, repeat)}.json"
                if path.exists():
                    row = json.loads(path.read_text())
                    if any(row.get(k) != scene[k] for k in scene):
                        raise ValueError(f"Cached input mismatch: {path}")
                    if (row["requested_model"], row["temperature"], row["repeat"]) != (model, temperature, repeat):
                        raise ValueError(f"Cached condition mismatch: {path}")
                else:
                    response = paced_completion(client, arguments_for(image_bytes, scene["prompt"], model, temperature))
                    raw = response.model_dump(mode="json", exclude_none=True)
                    row = response_record(scene, model, temperature, repeat, index, raw)
                    temporary = path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(row, indent=2, allow_nan=False), encoding="utf-8")
                    temporary.replace(path)
                if row["model"] != model:
                    raise RuntimeError(f"Returned snapshot differs: {row['model']}")
                rows.append(row)
                print(f"{row['call_key']}: {row['generated_output_text']!r} "
                      f"correct={row['correct']} likelihood={row['raw_sequence_likelihood']}", flush=True)
    return rows


def run(stage):
    if stage not in ("check", "run"):
        raise ValueError("Use check or run.")
    prepared = [prepare_scene(sample) for sample in pilot_samples()]
    spec = protocol([scene for scene, image in prepared])
    # JSON round-trip makes tuples comparable to persisted arrays on resume.
    spec = json.loads(json.dumps(spec))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "responses").mkdir(exist_ok=True)
    path = OUTPUT_ROOT / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != spec:
        raise ValueError("Pilot protocol changed; do not mix conditions or frozen inputs.")
    if not path.exists():
        path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(f"Verified {len(prepared)} frozen scenes; {len(prepared)*len(CONDITIONS)*REPEATS} planned responses.", flush=True)
    if stage == "check":
        return
    # First scene verifies all six model/temperature conditions before bulk work.
    rows = run_scene(*prepared[0])
    if any(row["raw_sequence_likelihood"] is None for row in rows):
        raise RuntimeError("First-scene logprob/format check failed; saved raw responses for diagnosis.")
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_scene, *scene) for scene in prepared[1:]]
        for future in as_completed(futures):
            rows.extend(future.result())
    rows.sort(key=lambda row: row["call_key"])
    (OUTPUT_ROOT / "records.json").write_text(json.dumps({"protocol": spec, "records": rows}, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Completed {len(rows)} responses. Production pipeline and threshold unchanged.", flush=True)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) == 2 else "check")
