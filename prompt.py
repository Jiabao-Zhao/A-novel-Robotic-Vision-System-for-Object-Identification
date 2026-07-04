import json

def _vlm_prompt(user_text, detections):
    detections_json = json.dumps(detections, separators=(",", ":"))

    return f"""
You are classifying already-localized objects in a robot workspace image.

The point-cloud localization module has already localized the objects. You must NOT perform localization.
Your job is to classify the target objects from the human instruction and associate it with the detected object_ids.

You are provided with following information:
- a RGB image that annotated the localized objects with bounding boxes and object_ids.
- a list of detected objects with object_ids, visual labels, and projected 2D bounding boxes [x1, y1, x2, y2].
- a human instruction that contains target object.

Rules:
- The visual label on the image may be the numeric suffix of object_id, for example visual_label "001" means object_id "object_001".
- Choose object_id only from the localized objects list.
- Do not return coordinates, bounding boxes, masks, depth values, or robot poses.
- Return only compact JSON.
- If the target object is visible and classifiable, return:
{{"object_id":"object_1","object_type":"green block"}}
- If the target object is not visible or cannot be matched to any localized object, return:
{{"object_id":null,"object_type":null}}

User instruction:
{user_text}

Localized objects:
{detections_json}
""".strip()
