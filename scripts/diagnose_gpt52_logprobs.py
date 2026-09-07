"""One raw-HTTP reproduction; never alters pilot responses or confidence scores.

Run from the repository root with python -m scripts.diagnose_gpt52_logprobs.
Completed captures replay offline. The saved body is the HTTP client's decoded
entity bytes, before SDK JSON/model parsing, not a TLS/network packet capture.
"""

from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path

from scripts.run_vlm_multi_object_pilot import OUTPUT_ROOT as PILOT_ROOT, request_arguments
from scripts.run_vlm_output_format_ablation import digest
from scripts.run_vlm_candidate_verification_pilot import save_json


MODEL = "gpt-5.2-2025-12-11"
TRIAL_ID = "task_00_alphabet_soup_state_01_targets_2"
OUTPUT_ROOT = PILOT_ROOT.parent / "gpt52_raw_logprob_diagnostic"
SAFE_HEADERS = ("x-request-id", "date", "content-type", "openai-processing-ms")


def logprob_fields(payload):
    """Include chosen tokens and all alternatives, without changing their values."""
    fields = {}
    for choice_index, choice in enumerate(payload["choices"]):
        for channel in ("content", "refusal"):
            tokens = (choice.get("logprobs") or {}).get(channel) or []
            for index, token in enumerate(tokens):
                path = f"choices[{choice_index}].logprobs.{channel}[{index}]"
                fields[path] = (token["token"], token.get("logprob"))
                for rank, alternative in enumerate(token.get("top_logprobs") or []):
                    fields[f"{path}.top_logprobs[{rank}]"] = (
                        alternative["token"], alternative.get("logprob"))
    return fields


def compare_response(body, parsed):
    # Decimal preserves the JSON number's sign/value independently of SDK floats.
    raw = json.loads(body, parse_float=Decimal)
    raw_fields, sdk_fields = logprob_fields(raw), logprob_fields(parsed)
    differences, positives = [], []
    for path in sorted(raw_fields.keys() | sdk_fields.keys()):
        raw_token, raw_lp = raw_fields.get(path, (None, None))
        sdk_token, sdk_lp = sdk_fields.get(path, (None, None))
        converted = float(raw_lp) if raw_lp is not None else None
        if path not in raw_fields or path not in sdk_fields or (raw_token, converted) != (sdk_token, sdk_lp):
            differences.append(path)
        if raw_lp is not None and raw_lp > 0:
            positives.append({"path": path, "token": raw_token,
                              "raw_json_decimal": str(raw_lp), "sdk_logprob": sdk_lp,
                              "is_alternative": ".top_logprobs[" in path})
    return {
        "response_body_sha256": digest(body), "model": raw["model"],
        "generated_output": raw["choices"][0]["message"].get("content"),
        "chosen_logprob_fields": sum(".top_logprobs[" not in p for p in raw_fields),
        "alternative_logprob_fields": sum(".top_logprobs[" in p for p in raw_fields),
        "all_logprob_fields_match_sdk": bool(raw_fields) and not differences,
        "mismatched_fields": differences,
        "positive_chosen_logprobs": sum(not p["is_alternative"] for p in positives),
        "positive_alternative_logprobs": sum(p["is_alternative"] for p in positives),
        "positive_values": positives,
    }


def run():
    import openai

    inputs_path = PILOT_ROOT / "inputs.json"
    inputs = json.loads(inputs_path.read_text())
    sample = next(s for s in inputs["samples"] if s["trial_id"] == TRIAL_ID)
    image_bytes = Path(sample["new_image_path"]).read_bytes()
    if digest(image_bytes) != sample["new_image_sha256"]:
        raise ValueError("Frozen image hash mismatch; no request made.")
    args = request_arguments(image_bytes, sample["instruction"], MODEL)
    request_hash = digest(json.dumps(args, sort_keys=True).encode())
    checkpoint = PILOT_ROOT / "responses" / f"{TRIAL_ID}_{MODEL}.json"
    previous = json.loads(checkpoint.read_text())
    if request_hash != previous["request_sha256"]:
        raise ValueError("Request differs from frozen pilot; no request made.")
    protected = [inputs_path, PILOT_ROOT / "records.json", *sorted((PILOT_ROOT / "responses").glob("*.json"))]
    protected += [Path(p) for p in inputs["protocol"]["source_sha256"]]
    hashes = {str(p): digest(p.read_bytes()) for p in protected}
    protocol = {
        "trial_id": TRIAL_ID, "input_sha256": sample["input_sha256"],
        "request_sha256": request_hash, "openai_sdk_version": openai.__version__,
        "settings": {k: v for k, v in args.items() if k != "messages"},
        "image_path": sample["new_image_path"], "image_sha256": digest(image_bytes),
        "instruction": sample["instruction"], "source_sha256": digest(Path(__file__).read_bytes()),
        "max_new_requests": 1, "automatic_retries": 0, "protected_sha256": hashes,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    protocol_path = OUTPUT_ROOT / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Diagnostic protocol changed; refusing to overwrite capture.")
    else:
        save_json(protocol_path, protocol)
    body_path = OUTPUT_ROOT / "response_body.json"
    parsed_path = OUTPUT_ROOT / "sdk_parsed_response.json"
    metadata_path = OUTPUT_ROOT / "http_metadata.json"
    capture_paths = (body_path, parsed_path, metadata_path)
    if not any(p.exists() for p in capture_paths):
        with openai.OpenAI(timeout=60, max_retries=0) as client:
            if str(client.base_url) != "https://api.openai.com/v1/":
                raise ValueError("Diagnostic requires the official HTTPS API endpoint.")
            print("Making one fresh raw-response request (retries disabled).", flush=True)
            response = client.chat.completions.with_raw_response.create(**args)
            body = response.http_response.content
            body_path.write_bytes(body)  # Preserve before calling SDK parse().
            save_json(metadata_path, {
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "endpoint": str(response.url), "status_code": response.status_code,
                "headers": {k: response.headers[k] for k in SAFE_HEADERS if k in response.headers},
                "response_body_sha256": digest(body), "retries_taken": response.retries_taken,
                "capture_stage": "HTTP entity bytes saved before LegacyAPIResponse.parse()",
            })
            parsed = response.parse().model_dump(mode="json", exclude_none=True)
            save_json(parsed_path, parsed)
    if not all(p.exists() for p in capture_paths):
        raise RuntimeError("Partial capture preserved; refusing to make another API call.")
    body = body_path.read_bytes()
    metadata = json.loads(metadata_path.read_text())
    if digest(body) != metadata["response_body_sha256"]:
        raise ValueError("Captured body hash mismatch.")
    result = compare_response(body, json.loads(parsed_path.read_text()))
    if result["model"] != MODEL:
        raise ValueError("Returned model snapshot differs from the requested model.")
    if any(digest(Path(p).read_bytes()) != h for p, h in hashes.items()):
        raise ValueError("A protected pilot file changed during the diagnostic.")
    result.update(frozen_pilot_files_unchanged=True, new_requests_in_saved_capture=1)
    save_json(OUTPUT_ROOT / "comparison.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
