"""Joint instruction-to-object association for the controlled LIBERO runner."""

import base64
import json
import math
import os
import re
from pathlib import Path

from openai import OpenAI

from vlm_module import candidate_choice_map, load_localized_objects, resolve_association


ROLE = """You identify objects mentioned in a task instruction by associating them
with already-localized candidates in the marked image.
Return one line per distinct object mentioned in the instruction:
<existing image letter>: <object description from the instruction>
Use the letters already shown in the image. Preserve the instruction's object
names and identifying modifiers. Do not classify unrelated objects.
Do not output explanations, confidence numbers, coordinates, task roles, or actions.
If a mentioned object is not present, return N: <object description>."""


def request_arguments(image_bytes, instruction, model):
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": ROLE + "\n\nTask instruction:\n" + instruction},
            {"role": "user", "content": [{
                "type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii"),
                    "detail": "high",
                },
            }]},
        ],
        "temperature": 0, "max_completion_tokens": 128,
        "logprobs": True, "top_logprobs": 20,
    }


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
    """Parse letter:name entries, retaining original token likelihood diagnostics."""
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
    # Provider tokens spanning two entries cannot have independent entry scores.
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


def object_phrase(text):
    """Ignore a copied leading article, preserving every identifying modifier."""
    return normalized_name(text).removeprefix("the ")


def label_likelihood(raw, description):
    """Score only the original tokens overlapping this entry's decision letter."""
    choice = raw["choices"][0]
    text = choice["message"].get("content") or ""
    tokens = (choice.get("logprobs") or {}).get("content") or []
    if "".join(token["token"] for token in tokens) != text:
        return None, None, []
    lines = [line for line in re.finditer(
        r"^[ \t]*(?P<label>[A-Z]):[ \t]*(?P<name>[^\r\n]+)", text, re.MULTILINE
    ) if object_phrase(line["name"]) == description]
    if len(lines) != 1:
        return None, None, []
    start, end = lines[0].span("label")
    selected, offset = [], 0
    for token in tokens:
        stop = offset + len(token["token"])
        if offset < end and stop > start:
            selected.append(token)
        offset = stop
    logprob, score = token_likelihood(selected)
    return logprob, score, selected


def joint_inferences(raw, instruction, mapping, target_descriptions):
    """Required names are checked locally; never supplied as a target list to the VLM."""
    parsed = parse_response(raw, {**mapping, "N": None}, instruction)
    required = [object_phrase(name) for name in target_descriptions]
    returned = [object_phrase(entry["target_description"]) for entry in parsed["entries"]]
    complete = len(returned) == len(required) and sorted(returned) == sorted(required)
    rows = []
    for description in required:
        entries = [entry for entry in parsed["entries"]
                   if object_phrase(entry["target_description"]) == description]
        entry = entries[0] if len(entries) == 1 else {}
        label = entry.get("choice_label")
        logprob, score, tokens = label_likelihood(raw, description)
        if not parsed["format_valid"] or not complete:
            logprob, score = None, None
        rows.append({
            "target_description": description,
            "vlm_object_id": mapping.get(label),
            "association_score": score,
            "diagnostics": {
                "provider": "openai", "model": raw.get("model"),
                "candidate_map": mapping, "choice_label": label,
                "score_type": "raw_label_likelihood",
                "raw_log_probability": logprob, "raw_association_likelihood": score,
                "decision_tokens": tokens,
                "joint_format_valid": parsed["format_valid"],
                "required_names_complete": complete,
                "full_response_likelihood_diagnostic": parsed["full_response_likelihood"],
                "entry_letter_and_name_likelihood_diagnostic": entry.get("raw_association_likelihood"),
                "score_unavailable_reason": (None if score is not None else
                    "Invalid/incomplete joint output or unavailable original decision-letter logprobs."),
            },
        })
    return rows, parsed


def associate_instruction_from_localization(
    instruction, target_descriptions, *, image_path, localization_path, output_path,
    threshold, human_resolver=None, assumed_human=False, clarification_bboxes=None,
):
    """Resolve the joint instruction using the decision-letter threshold contract."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapping = candidate_choice_map(load_localized_objects(localization_path))
    model = os.environ.get("OPENAI_VLM_MODEL", "gpt-4.1-mini-2025-04-14")
    args = request_arguments(Path(image_path).read_bytes(), instruction, model)
    request_record = {
        "model": model, "role": ROLE, "instruction": instruction,
        "image_path": str(image_path), "candidate_map_local_only": mapping,
        "settings": {key: value for key, value in args.items() if key != "messages"},
        "image_detail": "high",
        "vlm_inputs": "Role plus original task instruction and letter-marked image only.",
        "gate": "Each entry uses only raw generated decision-letter likelihood; full-output and letter+name likelihoods are diagnostics.",
        "threshold": threshold,
        "threshold_validation": "Provisional inherited value; not calibrated for the joint prompt.",
        "assumed_correct_human_on_deferral": assumed_human,
    }
    (output_path.parent / "vlm_request.json").write_text(
        json.dumps(request_record, indent=2), encoding="utf-8",
    )
    raw = OpenAI().chat.completions.create(**args).model_dump(mode="json")
    (output_path.parent / "vlm_provider_response.json").write_text(
        json.dumps(raw, indent=2), encoding="utf-8",
    )
    inferences, parsed = joint_inferences(raw, instruction, mapping, target_descriptions)
    associations = []
    for inference in inferences:
        result = resolve_association(
            inference, threshold, human_resolver,
            visual_prompt_path=image_path, candidate_bboxes=clarification_bboxes,
        )
        diagnostics = result["diagnostics"]
        if assumed_human and "human_response_time_s" in diagnostics:
            diagnostics["assumed_human_computation_time_s"] = diagnostics.pop("human_response_time_s")
            diagnostics["resolution_source"] = "simulator_identity_as_assumed_correct_human"
            if result["resolution"].startswith("human_"):
                result["resolution"] = "assumed_" + result["resolution"]
            if diagnostics.get("target_absence_source") == "human":
                diagnostics["target_absence_source"] = "assumed_human"
        associations.append(result)
    payload = {
        "schema_version": 1, "method": "joint instruction to letter:name association",
        "instruction": instruction, "visual_prompt_path": str(image_path),
        "parsed_joint_response": parsed, "associations": associations,
        "assumed_correct_human_on_deferral": assumed_human,
        "simulator_identity_used_for_assumed_human": any(
            a["diagnostics"].get("resolution_source") == "simulator_identity_as_assumed_correct_human"
            for a in associations
        ),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path
