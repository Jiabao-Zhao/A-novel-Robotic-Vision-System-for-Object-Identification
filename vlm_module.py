import base64
import json
import os
import re
from pathlib import Path

from prompt import _vlm_prompt


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

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "provider": provider,
                "user_instruction": str(user_text),
                "result": result,
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
    detections = []
    for item in payload.get("objects", []):
        roi = item.get("roi", {})
        detections.append(
            {
                "object_id": item.get("object_id"),
                "visual_label": str(item.get("object_id", "")).removeprefix("object_"),
                "bbox_2d_xyxy": [
                    int(roi.get("x1", 0)),
                    int(roi.get("y1", 0)),
                    int(roi.get("x2", 0)),
                    int(roi.get("y2", 0)),
                ],
            }
        )
    return detections


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
