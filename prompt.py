import json

def _vlm_classification_prompt(user_text, detections):
    return f"""
Classify all requested targets in the image.

Targets: {user_text}
Boxes: {json.dumps(detections, separators=(",", ":"))}

Return only JSON: [{{"id":"6","c":"green block"}}]
If none found: []
""".strip()


def _vlm_classification_localization_prompt(user_text):
    return f"""
Classify and localize targets in the image.

Target: {user_text}

Return only JSON: [{{"c":"green block","box":[120,80,180,140]}}]
If none found: []
""".strip()
