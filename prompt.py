import json

def _vlm_prompt(user_text, detections):
    detections_json = json.dumps(detections, separators=(",", ":"))

    return f"""
You are classifying already-localized objects in a robot workspace image.

The point-cloud localization module has already localized the objects. You must NOT perform localization.
Your job is to evaluate each localized object independently against the human target instruction.

You are provided with:
- an RGB visual prompt containing the full scene and/or enlarged localized-object crops labeled with object_ids.
- localized object metadata with object_ids, visual labels, 2D bounding boxes, and absolute image regions.
- a human instruction that contains the target objects.

Independent classification rule:
- Evaluate every localized object by itself.
- Ask: does this object, by itself, visually match the user's target description?
- If the instruction mentions multiple physical objects, mark each mentioned object that is visible.
- Do not rank objects against each other.
- Do not compare one object to another when deciding whether it matches.
- Do not assign confidence scores.
- Spatial region text helps the human understand object location, but it is not identity evidence.

Classification rules:
- The visual label on the image may be the numeric suffix of object_id, for example visual_label "001" means object_id "object_001".
- Choose object IDs only from the localized objects list.
- For each object, target_match must be exactly one of: "match", "plausible_match", "not_match".
- Use instruction_role "moved_object" for the object being moved, picked, placed, or inspected.
- Use instruction_role "reference_object" for a support, destination, fixture, or relation object.
- Use instruction_role "other_target" for a mentioned target object that is not clearly moved or reference.
- Use null when the object is not mentioned by the instruction.
- Use "match" when the object has clear visual evidence for the target.
- Use "plausible_match" when the object could be the target but the class is uncommon, specialized, partly occluded, or visually ambiguous.
- Use "not_match" when the object clearly does not match the target.
- Manufacturing components may look visually similar. For each object independently, inspect body shape, rectangular vs cylindrical geometry, cable attachment, connector face, visible pins, color, distinctive housing features, and partial occlusion.
- The Python pipeline will decide whether to continue or ask the human. Do not return selected/not_found/needs_clarification status.

Return only compact valid JSON in this exact schema:
{{
  "object_evaluations": [
    {{
      "object_id": "object_001",
      "visual_label": "001",
      "target_match": "match | plausible_match | not_match",
      "predicted_type": "short type or null",
      "instruction_role": "moved_object | reference_object | other_target | null",
      "visual_evidence": "brief evidence based only on this object",
      "missing_or_uncertain_cues": "brief explanation or null",
      "spatial_description": "absolute image/workspace location"
    }}
  ]
}}

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
