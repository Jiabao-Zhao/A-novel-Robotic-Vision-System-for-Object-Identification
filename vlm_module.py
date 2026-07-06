import base64
import json
import os
import re
from pathlib import Path

from prompt import _vlm_prompt

TARGET_MATCH_VALUES = ("match", "plausible_match", "not_match")
CANDIDATE_MATCH_VALUES = ("match", "plausible_match")
DECISION_RULE = "continue only when exactly one object is match or plausible_match"


class GeminiVLM:
    def __init__(self):
        from google import genai

        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model_name = os.environ.get("GEMINI_VLM_MODEL", "gemini-2.5-flash")

    def analyze_image(self, image_path, user_text, detections):
        from google.genai import types

        image_bytes = Path(image_path).read_bytes()
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                _vlm_prompt(user_text, detections),
            ],
        )
        return response.text or "{}"


class OpenAIVLM:
    def __init__(self):
        from openai import OpenAI

        self.client = OpenAI()
        self.model_name = os.environ.get("OPENAI_VLM_MODEL", "gpt-4.1-mini")

    def analyze_image(self, image_path, user_text, detections):
        image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You classify user-described objects from an annotated "
                        "robot workspace image. Return only JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _vlm_prompt(user_text, detections)},
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
            temperature=0,
        )
        return response.choices[0].message.content or "{}"


def classify_from_localization(
    user_text,
    image_path=Path("output/annotation/RGB_point_cloud_roi_annotation.png"),
    localization_path=Path("output/point_cloud_localization/point_cloud_localization.json"),
    output_path=Path("output/vlm/vlm_result.json"),
):
    detections = load_localized_objects(localization_path)
    raw_response, provider = classify_with_gemini_then_openai(
        image_path=image_path,
        user_text=user_text,
        detections=detections,
    )
    result = parse_json_response(raw_response)
    normalized_result = normalize_vlm_result(result, detections, user_text=user_text)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "provider": provider,
                "user_instruction": str(user_text),
                "raw_result": result,
                "normalized_result": normalized_result,
                "decision_rule": DECISION_RULE,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path


def classify_with_gemini_then_openai(image_path, user_text, detections):
    try:
        return GeminiVLM().analyze_image(image_path, user_text, detections), "gemini"
    except Exception as gemini_error:
        try:
            return OpenAIVLM().analyze_image(image_path, user_text, detections), "openai"
        except Exception as openai_error:
            raise RuntimeError(
                "Gemini VLM failed, then OpenAI VLM failed. "
                f"Gemini: {gemini_error} | OpenAI: {openai_error}"
            ) from openai_error


def load_localized_objects(localization_path):
    payload = json.loads(Path(localization_path).read_text(encoding="utf-8"))
    image_width, image_height = image_size_from_localization(payload)
    detections = []
    for item in payload.get("objects", []):
        roi = item.get("roi", {})
        bbox = [
            int(roi.get("x1", 0)),
            int(roi.get("y1", 0)),
            int(roi.get("x2", 0)),
            int(roi.get("y2", 0)),
        ]
        region = absolute_image_region(bbox, image_width, image_height)
        detections.append(
            {
                "object_id": item.get("object_id"),
                "visual_label": str(item.get("object_id", "")).removeprefix("object_"),
                "bbox_2d_xyxy": bbox,
                "spatial_description": region["spatial_description"],
                "image_region": {
                    "horizontal": region["horizontal"],
                    "vertical": region["vertical"],
                },
            }
        )
    return detections


def image_size_from_localization(payload):
    intrinsics = payload.get("camera_intrinsics", {})
    width = int(intrinsics.get("width", 0) or payload.get("image_width", 0) or 0)
    height = int(intrinsics.get("height", 0) or payload.get("image_height", 0) or 0)

    if width > 0 and height > 0:
        return width, height

    max_x = 0
    max_y = 0
    for item in payload.get("objects", []):
        roi = item.get("roi", {})
        max_x = max(max_x, int(roi.get("x2", 0) or 0))
        max_y = max(max_y, int(roi.get("y2", 0) or 0))
    return max_x, max_y


def absolute_image_region(bbox, image_width, image_height):
    x1, y1, x2, y2 = bbox
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)
    horizontal = region_name(center_x, image_width, ("left", "center", "right"))
    vertical = region_name(center_y, image_height, ("top", "middle", "bottom"))

    if horizontal == "center" and vertical == "middle":
        description = "center region"
    else:
        description = f"{horizontal}-{vertical} region"

    return {
        "horizontal": horizontal,
        "vertical": vertical,
        "spatial_description": description,
    }


def region_name(value, extent, names):
    if extent <= 0:
        return names[1]
    if value < extent / 3.0:
        return names[0]
    if value < 2.0 * extent / 3.0:
        return names[1]
    return names[2]


def normalize_vlm_result(result, detections, user_text=None):
    result = result if isinstance(result, dict) else {}
    detection_by_id = {
        str(detection["object_id"]): detection
        for detection in detections
        if detection.get("object_id") is not None
    }

    if "object_evaluations" not in result and "object_id" in result:
        result = old_format_to_evaluations(result, detections)

    returned_evaluations = {
        str(evaluation.get("object_id")): evaluation
        for evaluation in result.get("object_evaluations", [])
        if isinstance(evaluation, dict)
        and str(evaluation.get("object_id")) in detection_by_id
    }

    object_evaluations = []
    for detection in detections:
        object_id = str(detection.get("object_id"))
        source = returned_evaluations.get(object_id, {})
        object_evaluations.append(normalize_object_evaluation(source, detection))

    candidates = [
        evaluation
        for evaluation in object_evaluations
        if evaluation["target_match"] in CANDIDATE_MATCH_VALUES
        and evaluation["object_id"] in detection_by_id
    ]

    if len(candidates) == 1:
        selected = candidates[0]
        selected_object_id = selected["object_id"]
        selected_object_type = (
            selected["predicted_type"]
            or result.get("selected_object_type")
            or str(user_text)
        )
        needs_human_clarification = False
        clarification_reason = None
        clarification_question = None
    elif len(candidates) > 1:
        selected_object_id = None
        selected_object_type = None
        needs_human_clarification = True
        clarification_reason = "multiple localized objects are possible matches for the target"
        clarification_question = build_clarification_question(candidates, object_evaluations)
    else:
        selected_object_id = None
        selected_object_type = None
        needs_human_clarification = True
        clarification_reason = "no localized object was labeled as a match or plausible match"
        clarification_question = build_no_candidate_question(object_evaluations)

    return {
        "object_evaluations": object_evaluations,
        "selected_object_id": selected_object_id,
        "selected_object_type": selected_object_type,
        "needs_human_clarification": needs_human_clarification,
        "clarification_reason": clarification_reason,
        "clarification_question": clarification_question,
        "candidate_count": len(candidates),
        "candidate_object_ids": [candidate["object_id"] for candidate in candidates],
        "decision_rule": DECISION_RULE,
    }


def old_format_to_evaluations(result, detections):
    object_id = result.get("object_id")
    object_type = result.get("object_type")
    evaluations = []
    for detection in detections:
        is_old_selection = detection.get("object_id") == object_id
        evaluations.append(
            {
                "object_id": detection.get("object_id"),
                "visual_label": detection.get("visual_label"),
                "target_match": "plausible_match" if is_old_selection else "not_match",
                "predicted_type": object_type if is_old_selection else None,
                "visual_evidence": (
                    "Old VLM response selected this object without per-object evidence."
                    if is_old_selection
                    else "Old VLM response did not evaluate this object independently."
                ),
                "missing_or_uncertain_cues": (
                    "Old response format did not provide the simplified object_evaluations schema."
                ),
                "spatial_description": detection.get("spatial_description"),
            }
        )
    return {
        "object_evaluations": evaluations,
    }


def normalize_object_evaluation(source, detection):
    target_match = str(source.get("target_match", "not_match")).strip().lower()
    if target_match == "uncertain":
        target_match = "plausible_match"
    if target_match not in TARGET_MATCH_VALUES:
        target_match = "not_match"

    return {
        "object_id": detection.get("object_id"),
        "visual_label": str(source.get("visual_label") or detection.get("visual_label") or ""),
        "target_match": target_match,
        "predicted_type": none_if_empty(source.get("predicted_type")),
        "bbox_2d_xyxy": detection.get("bbox_2d_xyxy"),
        "visual_evidence": none_if_empty(source.get("visual_evidence"))
        or "No independent visual evidence was provided for this object.",
        "missing_or_uncertain_cues": none_if_empty(source.get("missing_or_uncertain_cues")),
        "spatial_description": none_if_empty(source.get("spatial_description"))
        or detection.get("spatial_description")
        or "center region",
    }


def none_if_empty(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "null":
        return None
    return text


def build_clarification_question(candidates, object_evaluations):
    choices = " or ".join(
        describe_candidate(candidate, object_evaluations)
        for candidate in candidates[:4]
    )
    return (
        "Multiple localized objects could match the requested target. "
        f"Which one should I use: {choices}?"
    )


def build_no_candidate_question(object_evaluations):
    if not object_evaluations:
        return "No localized objects were available for classification."
    choices = ", ".join(
        describe_candidate(evaluation, object_evaluations)
        for evaluation in object_evaluations[:4]
    )
    return (
        "None of the localized objects clearly matches the requested target. "
        f"Which labeled object should I use: {choices}?"
    )


def describe_candidate(candidate, object_evaluations):
    description = f"{candidate['object_id']} in the {candidate['spatial_description']}"
    reference = nearest_known_reference(candidate, object_evaluations)
    if reference:
        description += f", near {reference['object_id']} ({reference['predicted_type']})"
    return description


def nearest_known_reference(candidate, object_evaluations):
    candidate_center = bbox_center(candidate.get("bbox_2d_xyxy"))
    if candidate_center is None:
        return None

    best = None
    best_distance_sq = None
    for evaluation in object_evaluations:
        if evaluation["object_id"] == candidate["object_id"]:
            continue
        reference_center = bbox_center(evaluation.get("bbox_2d_xyxy"))
        if (
            evaluation["target_match"] == "not_match"
            and evaluation.get("predicted_type")
            and reference_center is not None
        ):
            distance_sq = (
                (candidate_center[0] - reference_center[0]) ** 2
                + (candidate_center[1] - reference_center[1]) ** 2
            )
            if best_distance_sq is None or distance_sq < best_distance_sq:
                best = evaluation
                best_distance_sq = distance_sq
    return best


def bbox_center(bbox):
    if not bbox or len(bbox) != 4:
        return None
    return ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)


def parse_json_response(raw_response):
    text = str(raw_response or "").strip()
    if text.startswith("```"):
        text = "\n".join(
            line
            for line in text.splitlines()
            if not line.strip().startswith("```")
        ).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise
