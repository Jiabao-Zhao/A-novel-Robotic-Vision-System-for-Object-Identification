"""Joint instruction-to-object association for the controlled LIBERO runner."""

import json
import os
import re
from pathlib import Path

from openai import OpenAI

from scripts.run_vlm_multi_object_pilot import (
    ROLE, parse_response, request_arguments, token_likelihood,
)
from vlm_module import candidate_choice_map, load_localized_objects, resolve_association


def object_phrase(text):
    """Ignore a copied leading article, preserving every identifying modifier."""
    return " ".join(text.casefold().split()).removeprefix("the ")


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
    """Reuse the joint pilot prompt; preserve the decision-letter threshold contract."""
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
