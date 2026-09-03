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
PROVISIONAL_ASSOCIATION_THRESHOLD = 0.75
SCORE_SEMANTICS = (
    "The association score is the likelihood of the generated decision-label "
    "sequence under the provider model and prompt. It is not a calibrated "
    "probability that the object identity is correct."
)


class HumanClarificationRequired(RuntimeError):
    """Raised when a low-score association has no human resolver."""


class GeminiVLM:
    def __init__(self):
        from google import genai

        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model_name = os.environ.get("GEMINI_VLM_MODEL", "gemini-2.5-flash")
        self._logprobs_supported = None
        self._logprob_error = None

    def associate(self, image_path, prompt):
        from google.genai import types

        image_bytes = Path(image_path).read_bytes()
        contents = [
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            prompt,
        ]
        common_config = {"temperature": 0, "max_output_tokens": 64}

        if self._logprobs_supported is not False:
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_logprobs=True,
                        **common_config,
                    ),
                )
                self._logprobs_supported = True
                return _gemini_provider_result(response, self.model_name)
            except Exception as error:
                if not _logprobs_unavailable(error):
                    raise
                self._logprobs_supported = False
                self._logprob_error = str(error)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(**common_config),
        )
        return _gemini_provider_result(
            response,
            self.model_name,
            logprob_error=self._logprob_error
            or "The Gemini response did not include output token log probabilities.",
        )


class OpenAIVLM:
    def __init__(self):
        from openai import OpenAI

        self.client = OpenAI()
        self.model_name = os.environ.get("OPENAI_VLM_MODEL", "gpt-4.1-mini")
        self._logprobs_supported = None
        self._logprob_error = None

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
            try:
                response = self.client.chat.completions.create(
                    logprobs=True,
                    **request,
                )
                self._logprobs_supported = True
                return _openai_provider_result(response, self.model_name)
            except Exception as error:
                if not _logprobs_unavailable(error):
                    raise
                self._logprobs_supported = False
                self._logprob_error = str(error)

        response = self.client.chat.completions.create(**request)
        return _openai_provider_result(
            response,
            self.model_name,
            logprob_error=self._logprob_error
            or "The OpenAI response did not include output token log probabilities.",
        )


class GeminiThenOpenAI:
    """Use Gemini first and instantiate OpenAI only after a Gemini call failure."""

    def __init__(self):
        self.gemini = None
        self.gemini_initialization_error = None
        try:
            self.gemini = GeminiVLM()
        except Exception as error:
            self.gemini_initialization_error = error
        self.openai = None

    def associate(self, image_path, prompt):
        gemini_error = self.gemini_initialization_error
        if self.gemini is not None:
            try:
                return self.gemini.associate(image_path, prompt)
            except Exception as error:
                gemini_error = error
        try:
            if self.openai is None:
                self.openai = OpenAIVLM()
            return self.openai.associate(image_path, prompt)
        except Exception as openai_error:
            raise RuntimeError(
                "Gemini VLM failed, then OpenAI VLM failed. "
                f"Gemini: {gemini_error} | OpenAI: {openai_error}"
            ) from openai_error


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
                "schema_version": 2,
                "visual_prompt_path": str(image_path),
                "score_semantics": SCORE_SEMANTICS,
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

    active_provider = GeminiThenOpenAI() if provider is None else provider
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
    inference["visual_prompt_path"] = str(image_path)
    return resolve_association(inference, threshold, human_resolver)


def infer_target_association(target_description, image_path, detections, provider=None):
    choice_map = candidate_choice_map(detections)
    prompt_candidates = candidate_prompt_records(detections, choice_map)
    prompt = _vlm_prompt(target_description, prompt_candidates)
    active_provider = GeminiThenOpenAI() if provider is None else provider
    provider_result = active_provider.associate(image_path, prompt)
    generated_text = str(provider_result.get("generated_text") or "")
    model_choice = parse_model_choice(generated_text, choice_map)
    decision_tokens, raw_log_probability = decision_sequence_log_probability(
        model_choice,
        provider_result.get("token_logprobs") or [],
    )
    association_score = (
        math.exp(raw_log_probability)
        if raw_log_probability is not None
        else None
    )

    return {
        "target_description": str(target_description).strip(),
        "provider": provider_result.get("provider"),
        "model": provider_result.get("model"),
        "model_output": generated_text,
        "model_choice": model_choice,
        "vlm_object_id": object_id_for_choice(choice_map, model_choice),
        "raw_log_probability": raw_log_probability,
        "association_score": association_score,
        "decision_token_logprobs": decision_tokens,
        "logprob_error": provider_result.get("logprob_error"),
        "candidate_map": choice_map,
    }


def resolve_association(vlm_result, threshold, human_resolver=None):
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("The provisional association threshold must be between 0 and 1.")

    result = dict(vlm_result)
    result.update(
        {
            "threshold": threshold,
            "final_object_id": None,
            "resolution": None,
            "requires_human_clarification": False,
            "human_intervention": False,
            "human_selected_object_id": None,
        }
    )
    score = result.get("association_score")
    model_choice = result.get("model_choice")
    vlm_object_id = result.get("vlm_object_id")

    if model_choice is not None and score is not None and score >= threshold:
        result["final_object_id"] = vlm_object_id
        result["resolution"] = (
            "target_not_present" if model_choice == NONE_CHOICE else "vlm_accepted"
        )
        return result

    result["requires_human_clarification"] = True
    if human_resolver is None:
        if score is None:
            result["resolution"] = "confidence_unavailable"
            return result
        raise HumanClarificationRequired(
            f"Association score {score:.6f} is below the provisional threshold "
            f"{threshold:.6f} for target {result.get('target_description')!r}."
        )

    human_selection = human_resolver(dict(result))
    valid_object_ids = set(result.get("candidate_map", {}).values())
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
    result["human_selected_object_id"] = human_object_id
    result["human_intervention"] = True
    result["requires_human_clarification"] = False
    if human_object_id is None:
        result["resolution"] = "target_not_present"
    elif human_object_id == vlm_object_id:
        result["resolution"] = "human_confirmed"
    else:
        result["resolution"] = "human_corrected"
    return result


def console_human_resolver(association):
    candidate_ids = list(association.get("candidate_map", {}).values())
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
    choice = str(generated_text or "").strip().upper()
    valid_choices = set(choice_map) | {NONE_CHOICE}
    return choice if choice in valid_choices else None


def decision_sequence_log_probability(model_choice, token_logprobs):
    if model_choice is None:
        return [], None

    consumed = []
    generated = ""
    raw_log_probability = 0.0
    for record in token_logprobs:
        token = str(record.get("token") or "")
        log_probability = record.get("log_probability")
        if log_probability is None or not math.isfinite(float(log_probability)):
            return [], None
        consumed.append(
            {
                "token": token,
                "log_probability": float(log_probability),
            }
        )
        generated += token
        raw_log_probability += float(log_probability)
        stripped = generated.strip().upper()
        if stripped == model_choice:
            return consumed, raw_log_probability
        if stripped and not model_choice.startswith(stripped):
            return [], None
    return [], None


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


def _gemini_provider_result(response, model_name, logprob_error=None):
    candidate = response.candidates[0] if response.candidates else None
    logprobs_result = candidate.logprobs_result if candidate is not None else None
    chosen = logprobs_result.chosen_candidates if logprobs_result is not None else []
    token_logprobs = [
        {"token": item.token, "log_probability": item.log_probability}
        for item in (chosen or [])
    ]
    if not token_logprobs and logprob_error is None:
        logprob_error = "The Gemini response did not include chosen-token log probabilities."
    return {
        "provider": "gemini",
        "model": model_name,
        "generated_text": response.text or "",
        "token_logprobs": token_logprobs,
        "logprob_error": logprob_error,
    }


def _openai_provider_result(response, requested_model, logprob_error=None):
    choice = response.choices[0]
    content_logprobs = choice.logprobs.content if choice.logprobs is not None else []
    token_logprobs = [
        {"token": item.token, "log_probability": item.logprob}
        for item in (content_logprobs or [])
    ]
    if not token_logprobs and logprob_error is None:
        logprob_error = "The OpenAI response did not include output token log probabilities."
    return {
        "provider": "openai",
        "model": response.model or requested_model,
        "generated_text": choice.message.content or "",
        "token_logprobs": token_logprobs,
        "logprob_error": logprob_error,
    }


def _logprobs_unavailable(error):
    message = str(error).lower()
    return "logprob" in message and any(
        phrase in message
        for phrase in ("not enabled", "not supported", "unsupported", "invalid argument")
    )
