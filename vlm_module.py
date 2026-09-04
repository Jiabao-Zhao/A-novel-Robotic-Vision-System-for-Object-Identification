import base64
import json
import math
import os
import string
from pathlib import Path

import cv2
import numpy as np

from prompt import _vlm_prompt


NONE_CHOICE = "N"
# Development-only gate. Calibrate later on held-out data; callers may supply a
# provider-specific threshold without changing the association implementation.
PROVISIONAL_ASSOCIATION_THRESHOLD = 0.75
TOP_LOGPROBS_LIMIT = 20
SCORE_SEMANTICS = (
    "The association score is always exp(sum(log probabilities)) over the "
    "decision-bearing generated label tokens. It is the provider model's raw "
    "label likelihood over its full output vocabulary, not a calibrated "
    "probability of correct object identity. Candidate-normalized scores and "
    "their margin are diagnostics only and never control the threshold gate."
)
THRESHOLD_NOTE = (
    "The threshold is provisional. Select it on held-out calibration or "
    "validation data, then evaluate the fixed threshold on independent test "
    "data using "
    "autonomous coverage, autonomous accuracy, human intervention rate, and "
    "false autonomous acceptance rate."
)


class OpenAIVLM:
    def __init__(self):
        from openai import OpenAI

        self.client = OpenAI()
        self.model_name = os.environ.get("OPENAI_VLM_MODEL", "gpt-4.1-mini")
        self._logprobs_supported = None
        self._top_logprobs_supported = None
        self._logprob_error = None
        self._top_logprob_error = None

    def associate(self, image_path, prompt):
        image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        request = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Associate one semantic target with an already-localized "
                        "candidate. Return only the requested choice label."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
            "temperature": 0,
            "max_completion_tokens": 8,
        }

        if self._logprobs_supported is not False:
            request_top_logprobs = self._top_logprobs_supported is not False
            logprob_request = {"logprobs": True}
            if request_top_logprobs:
                logprob_request["top_logprobs"] = TOP_LOGPROBS_LIMIT
            try:
                response = self.client.chat.completions.create(
                    **logprob_request,
                    **request,
                )
            except Exception as error:
                if not _logprobs_unavailable(error):
                    raise
                if request_top_logprobs:
                    self._top_logprobs_supported = False
                    self._top_logprob_error = str(error)
                    try:
                        response = self.client.chat.completions.create(
                            logprobs=True,
                            **request,
                        )
                    except Exception as chosen_error:
                        if not _logprobs_unavailable(chosen_error):
                            raise
                        self._logprobs_supported = False
                        self._logprob_error = str(chosen_error)
                    else:
                        self._logprobs_supported = True
                        return _openai_provider_result(
                            response,
                            self.model_name,
                            top_logprob_error=self._top_logprob_error,
                        )
                else:
                    self._logprobs_supported = False
                    self._logprob_error = str(error)
            else:
                self._logprobs_supported = True
                if request_top_logprobs:
                    self._top_logprobs_supported = True
                return _openai_provider_result(
                    response,
                    self.model_name,
                    top_logprob_error=self._top_logprob_error,
                )

        response = self.client.chat.completions.create(**request)
        return _openai_provider_result(
            response,
            self.model_name,
            logprob_error=self._logprob_error
            or "The OpenAI response did not include output token log probabilities.",
        )


def associate_targets_from_localization(
    target_descriptions,
    image_path=Path("outputs/physical/annotation/RGB_point_cloud_roi_annotation.png"),
    localization_path=Path(
        "outputs/physical/point_cloud_localization/point_cloud_localization.json"
    ),
    output_path=Path("outputs/physical/vlm/semantic_associations.json"),
    threshold=PROVISIONAL_ASSOCIATION_THRESHOLD,
    human_resolver=None,
    provider=None,
):
    detections = load_localized_objects(localization_path)
    results = associate_targets(
        target_descriptions=target_descriptions,
        image_path=image_path,
        detections=detections,
        threshold=threshold,
        human_resolver=human_resolver,
        provider=provider,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "visual_prompt_path": str(image_path),
                "score_semantics": SCORE_SEMANTICS,
                "threshold_note": THRESHOLD_NOTE,
                "associations": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path


def associate_targets(
    target_descriptions,
    image_path,
    detections,
    threshold=PROVISIONAL_ASSOCIATION_THRESHOLD,
    human_resolver=None,
    provider=None,
):
    descriptions = [str(value).strip() for value in target_descriptions]
    if not descriptions or any(not value for value in descriptions):
        raise ValueError("target_descriptions must contain nonempty semantic descriptions.")

    active_provider = OpenAIVLM() if provider is None else provider
    return [
        associate_target(
            target_description=description,
            image_path=image_path,
            detections=detections,
            threshold=threshold,
            human_resolver=human_resolver,
            provider=active_provider,
        )
        for description in descriptions
    ]


def associate_target(
    target_description,
    image_path,
    detections,
    threshold=PROVISIONAL_ASSOCIATION_THRESHOLD,
    human_resolver=None,
    provider=None,
):
    inference = infer_target_association(
        target_description=target_description,
        image_path=image_path,
        detections=detections,
        provider=provider,
    )
    return resolve_association(
        inference,
        threshold,
        human_resolver,
        visual_prompt_path=str(image_path),
    )


def infer_target_association(target_description, image_path, detections, provider=None):
    choice_map = candidate_choice_map(detections)
    prompt_candidates = candidate_prompt_records(detections, choice_map)
    prompt = _vlm_prompt(target_description, prompt_candidates)
    active_provider = OpenAIVLM() if provider is None else provider
    provider_result = active_provider.associate(image_path, prompt)
    generated_text = str(provider_result.get("generated_text") or "")
    choice_label = parse_model_choice(generated_text, choice_map)
    decision_tokens, raw_log_probability = decision_sequence_log_probability(
        choice_label,
        provider_result.get("token_logprobs") or [],
    )
    raw_association_likelihood = (
        math.exp(raw_log_probability)
        if raw_log_probability is not None
        else None
    )
    choice_log_probabilities, candidate_scores, association_margin = (
        candidate_relative_scores(choice_map, decision_tokens)
    )
    association_score = raw_association_likelihood

    diagnostics = {
        "candidate_map": choice_map,
        "candidate_distribution_complete": candidate_scores is not None,
        "raw_log_probability": raw_log_probability,
        "raw_association_likelihood": raw_association_likelihood,
        "score_type": "raw_label_likelihood",
    }
    for key in ("provider", "model"):
        value = provider_result.get(key)
        if value:
            diagnostics[key] = value
    if choice_label is not None:
        diagnostics["choice_label"] = choice_label
    elif generated_text:
        diagnostics["unparsed_output"] = generated_text
    if choice_log_probabilities:
        diagnostics["available_choice_log_probabilities"] = choice_log_probabilities
    if candidate_scores is not None:
        diagnostics["candidate_scores"] = candidate_scores
        diagnostics["association_margin"] = association_margin
    for key in ("logprob_error", "top_logprob_error"):
        value = provider_result.get(key)
        if value:
            diagnostics[key] = value

    return {
        "target_description": str(target_description).strip(),
        "vlm_object_id": object_id_for_choice(choice_map, choice_label),
        "association_score": association_score,
        "diagnostics": diagnostics,
    }


def resolve_association(
    vlm_result,
    threshold,
    human_resolver=None,
    visual_prompt_path=None,
):
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("The provisional association threshold must be between 0 and 1.")

    diagnostics = dict(vlm_result.get("diagnostics") or {})
    result = {
        "target_description": vlm_result.get("target_description"),
        "vlm_object_id": vlm_result.get("vlm_object_id"),
        "association_score": vlm_result.get("association_score"),
        "threshold": threshold,
        "final_object_id": None,
        "diagnostics": diagnostics,
    }
    score = result.get("association_score")
    raw_likelihood = diagnostics.get("raw_association_likelihood")
    scores_match = score is None and raw_likelihood is None
    if score is not None and raw_likelihood is not None:
        scores_match = math.isclose(
            float(score),
            float(raw_likelihood),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    if not scores_match:
        raise ValueError(
            "association_score must equal diagnostics.raw_association_likelihood."
        )
    choice_label = diagnostics.get("choice_label")
    vlm_object_id = result.get("vlm_object_id")

    if choice_label is not None and score is not None and score >= threshold:
        result["final_object_id"] = vlm_object_id
        result["resolution"] = (
            "target_not_present" if choice_label == NONE_CHOICE else "vlm_accepted"
        )
        if choice_label == NONE_CHOICE:
            diagnostics["target_absence_source"] = "vlm"
        return result

    if human_resolver is None:
        if score is None:
            result["resolution"] = "confidence_unavailable"
            return result
        result["resolution"] = "deferred"
        return result

    candidate_map = diagnostics.get("candidate_map") or {}
    valid_object_ids = set(candidate_map.values())
    clarification = dict(result)
    clarification["candidate_object_ids"] = list(candidate_map.values())
    if visual_prompt_path is not None:
        clarification["visual_prompt_path"] = str(visual_prompt_path)
    human_selection = human_resolver(clarification)
    if human_selection is None or str(human_selection).strip().lower() == "none":
        human_object_id = None
    else:
        human_object_id = str(human_selection).strip()
        if human_object_id not in valid_object_ids:
            raise ValueError(
                "Human resolver selected an invalid localized object ID: "
                f"{human_object_id}. Valid choices: {sorted(valid_object_ids)} or none."
            )

    result["final_object_id"] = human_object_id
    if human_object_id is None:
        result["resolution"] = "target_not_present"
        diagnostics["target_absence_source"] = "human"
    elif human_object_id == vlm_object_id:
        result["resolution"] = "human_confirmed"
    else:
        result["resolution"] = "human_corrected"
    return result


def console_human_resolver(association):
    candidate_ids = association.get("candidate_object_ids") or []
    print(f"Visual prompt: {association.get('visual_prompt_path')}")
    print(f"Target: {association.get('target_description')}")
    print("VLM prediction:")
    print(f"  object_id = {association.get('vlm_object_id')}")
    print(f"  association_score = {association.get('association_score')}")
    print(f"  provisional threshold = {association.get('threshold')}")
    print("Select the correct localized object:")
    for object_id in candidate_ids:
        print(f"  {object_id}")
    print("  none")
    return input("Selection: ").strip()


def candidate_choice_map(detections):
    object_ids = sorted(
        str(item["object_id"])
        for item in detections
        if item.get("object_id") is not None
    )
    if not object_ids:
        raise ValueError("No localized candidates are available for VLM association.")
    if len(set(object_ids)) != len(object_ids):
        raise ValueError("Localized candidate object IDs must be unique.")

    labels = [label for label in string.ascii_uppercase if label != NONE_CHOICE]
    if len(object_ids) > len(labels):
        raise ValueError(f"At most {len(labels)} localized candidates are supported.")
    return dict(zip(labels, object_ids))


def object_id_for_choice(choice_map, choice):
    if choice is None or choice == NONE_CHOICE:
        return None
    if choice not in choice_map:
        raise ValueError(f"Unknown candidate choice label: {choice}")
    return choice_map[choice]


def choice_for_object_id(choice_map, object_id):
    for choice, candidate_object_id in choice_map.items():
        if candidate_object_id == object_id:
            return choice
    raise ValueError(f"Unknown localized object ID: {object_id}")


def candidate_prompt_records(detections, choice_map):
    detections_by_id = {str(item["object_id"]): item for item in detections}
    return [
        {
            "choice": choice,
            "object_id": object_id,
            "bbox_2d_xyxy": detections_by_id[object_id].get("bbox_2d_xyxy"),
            "centroid_3d_m": detections_by_id[object_id].get("centroid_3d_m"),
            "size_3d_m": detections_by_id[object_id].get("size_3d_m"),
        }
        for choice, object_id in choice_map.items()
    ]


def parse_model_choice(generated_text, choice_map):
    output = str(generated_text or "").strip().upper()
    valid_choices = set(choice_map) | {NONE_CHOICE}
    if output in valid_choices:
        return output
    for choice in sorted(valid_choices, key=len, reverse=True):
        if output.startswith(choice) and len(output) > len(choice):
            next_character = output[len(choice)]
            if next_character.isspace() or next_character in string.punctuation:
                return choice
    return None


def decision_sequence_log_probability(choice_label, token_logprobs):
    if choice_label is None:
        return [], None

    consumed = []
    generated = ""
    raw_log_probability = 0.0
    for record in token_logprobs:
        token = str(record.get("token") or "")
        if not consumed and not token.strip():
            continue
        log_probability = record.get("log_probability")
        if log_probability is None or not math.isfinite(float(log_probability)):
            return [], None
        decision_record = {
            "token": token,
            "log_probability": float(log_probability),
        }
        top_logprobs = record.get("top_logprobs") or []
        if top_logprobs:
            decision_record["top_logprobs"] = top_logprobs
        consumed.append(decision_record)
        generated += token
        raw_log_probability += float(log_probability)
        stripped = generated.strip().upper()
        if stripped == choice_label:
            return consumed, raw_log_probability
        if stripped and not choice_label.startswith(stripped):
            return [], None
    return [], None


def candidate_relative_scores(choice_map, decision_tokens):
    """Normalize over valid labels only when every label is in one top-k step."""
    if len(decision_tokens) != 1:
        return {}, None, None

    valid_labels = list(choice_map) + [NONE_CHOICE]
    decision = decision_tokens[0]
    alternatives = [decision, *(decision.get("top_logprobs") or [])]
    unique_tokens = {}
    for alternative in alternatives:
        token = str(alternative.get("token") or "")
        log_probability = alternative.get("log_probability")
        if log_probability is None or not math.isfinite(float(log_probability)):
            continue
        normalized_label = token.strip().upper()
        if normalized_label not in valid_labels:
            continue
        token_key = (normalized_label, token)
        unique_tokens[token_key] = max(
            float(log_probability),
            unique_tokens.get(token_key, -math.inf),
        )

    label_values = {label: [] for label in valid_labels}
    for (label, _), log_probability in unique_tokens.items():
        label_values[label].append(log_probability)
    available_log_probabilities = {
        label: _logsumexp(values)
        for label, values in label_values.items()
        if values
    }
    if set(available_log_probabilities) != set(valid_labels):
        return available_log_probabilities, None, None

    normalizer = _logsumexp(list(available_log_probabilities.values()))
    label_scores = {
        label: math.exp(log_probability - normalizer)
        for label, log_probability in available_log_probabilities.items()
    }
    candidate_scores = {
        object_id: label_scores[label]
        for label, object_id in choice_map.items()
    }
    candidate_scores["none"] = label_scores[NONE_CHOICE]
    ordered_scores = sorted(candidate_scores.values(), reverse=True)
    association_margin = ordered_scores[0] - ordered_scores[1]
    return available_log_probabilities, candidate_scores, association_margin


def _logsumexp(values):
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def create_roi_contact_sheet(
    rgb_path,
    localization_path,
    output_path,
    tile_size_px=224,
    columns=4,
):
    """Create a full-scene plus enlarged-candidate visual prompt for the VLM."""
    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read RGB image for VLM prompt: {rgb_path}")
    payload = json.loads(Path(localization_path).read_text(encoding="utf-8"))
    objects = list(payload.get("objects", []))
    if not objects:
        raise RuntimeError("Cannot create a VLM contact sheet without localized objects.")

    tile_count = 1 + len(objects)
    row_count = math.ceil(tile_count / columns)
    canvas = np.full(
        (row_count * tile_size_px, columns * tile_size_px, 3),
        245,
        dtype=np.uint8,
    )
    _place_contact_sheet_tile(canvas, image, "full scene", 0, tile_size_px, columns)
    image_height, image_width = image.shape[:2]
    for tile_index, item in enumerate(objects, start=1):
        roi = item.get("roi", {})
        padding = 8
        x1 = max(0, int(roi.get("x1", 0)) - padding)
        y1 = max(0, int(roi.get("y1", 0)) - padding)
        x2 = min(image_width, int(roi.get("x2", 0)) + padding)
        y2 = min(image_height, int(roi.get("y2", 0)) + padding)
        if x2 <= x1 or y2 <= y1:
            raise RuntimeError(f"Invalid ROI for {item.get('object_id')}: {roi}")
        _place_contact_sheet_tile(
            canvas,
            image[y1:y2, x1:x2],
            str(item.get("object_id", "unknown")),
            tile_index,
            tile_size_px,
            columns,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Could not save VLM contact sheet: {output_path}")
    return output_path


def _place_contact_sheet_tile(canvas, image, label, index, tile_size, columns):
    header_height = 34
    margin = 8
    row = index // columns
    column = index % columns
    tile_y = row * tile_size
    tile_x = column * tile_size
    available_width = tile_size - 2 * margin
    available_height = tile_size - header_height - margin
    scale = min(
        available_width / image.shape[1],
        available_height / image.shape[0],
    )
    resized_width = max(1, int(round(image.shape[1] * scale)))
    resized_height = max(1, int(round(image.shape[0] * scale)))
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_CUBIC,
    )
    image_x = tile_x + (tile_size - resized_width) // 2
    image_y = tile_y + header_height + (available_height - resized_height) // 2
    canvas[image_y:image_y + resized_height, image_x:image_x + resized_width] = resized
    cv2.putText(
        canvas,
        label,
        (tile_x + margin, tile_y + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


def load_localized_objects(localization_path):
    payload = json.loads(Path(localization_path).read_text(encoding="utf-8"))
    coordinate_frame = str(payload.get("frame") or "camera")
    detections = []
    for item in payload.get("objects", []):
        roi = item.get("roi", {})
        detections.append(
            {
                "object_id": item.get("object_id"),
                "bbox_2d_xyxy": [
                    int(roi.get("x1", 0)),
                    int(roi.get("y1", 0)),
                    int(roi.get("x2", 0)),
                    int(roi.get("y2", 0)),
                ],
                "centroid_3d_m": item.get("centroid_3d_m"),
                "size_3d_m": item.get("size_3d_m"),
                "point_count": item.get("point_count"),
                "geometry_frame": coordinate_frame,
            }
        )
    return detections


def _openai_provider_result(
    response,
    requested_model,
    logprob_error=None,
    top_logprob_error=None,
):
    choice = response.choices[0]
    content_logprobs = choice.logprobs.content if choice.logprobs is not None else []
    token_logprobs = []
    for item in content_logprobs or []:
        token_logprobs.append(
            {
                "token": item.token,
                "log_probability": item.logprob,
                "top_logprobs": [
                    {
                        "token": alternative.token,
                        "log_probability": alternative.logprob,
                    }
                    for alternative in (item.top_logprobs or [])
                ],
            }
        )
    if not token_logprobs and logprob_error is None:
        logprob_error = "The OpenAI response did not include output token log probabilities."
    if (
        token_logprobs
        and not any(record.get("top_logprobs") for record in token_logprobs)
        and top_logprob_error is None
    ):
        top_logprob_error = "The OpenAI response did not include top-token alternatives."
    return {
        "provider": "openai",
        "model": response.model or requested_model,
        "generated_text": choice.message.content or "",
        "token_logprobs": token_logprobs,
        "logprob_error": logprob_error,
        "top_logprob_error": top_logprob_error,
    }


def _logprobs_unavailable(error):
    message = str(error).lower()
    return "logprob" in message and any(
        phrase in message
        for phrase in ("not enabled", "not supported", "unsupported", "invalid argument")
    )
