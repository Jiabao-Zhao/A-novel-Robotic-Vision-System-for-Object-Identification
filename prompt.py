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


def _llm_planner_prompt(user_text, perception_context, action_schema):
    perception_json = json.dumps(perception_context, separators=(",", ":"))
    actions_json = json.dumps(action_schema, separators=(",", ":"))

    return f"""
You are a high-level robotic task planner. Your job is to choose a compact
sequence actions from the provided action schema to complete the user request.

The perception system has already selected the target object. 
You must not write robot code or invent low-level motion commands


Rules:
- Return only valid JSON.
- Use only action names from the approved action schema.
- Do not invent new functions.
- Do not invent object IDs.
- If the user only asks to find, inspect, locate, or identify an object, return
  a non-executing report_pose plan.
- If the user asks to pick an object, use pick_object with object_id and
  object_type only. Do not include point_camera_mm or object_rpy_base_deg.
- If the request is unsafe or missing required perception data, return a plan
  with status "blocked" and explain the missing information in "reason".

Return JSON in this shape:
{{
  "status": "ready" | "blocked",
  "reason": "short explanation",
  "target_object_id": "object id or null",
  "target_object_type": "object type or null",
  "actions": [
    {{
      "action": "approved action name",
      "arguments": {{}}
    }}
  ]
}}

User instruction:
{user_text}

Perception context:
{perception_json}

Approved action schema:
{actions_json}
""".strip()
