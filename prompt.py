import json


def _vlm_prompt(target_description, candidates):
    """Build one compact semantic-to-localized-object association query."""
    target_description = str(target_description).strip()
    if not target_description:
        raise ValueError("target_description must be a nonempty semantic description.")
    candidates_json = json.dumps(candidates, separators=(",", ":"))
    labels = [str(candidate["choice"]) for candidate in candidates] + ["N"]

    return f"""
The image contains objects already localized by an RGB-D point-cloud pipeline.
Associate exactly one semantic target with exactly one labeled candidate.
Do not estimate coordinates, add objects, or interpret the surrounding robot task.

Target: {target_description}

Candidate map and depth-derived metadata:
{candidates_json}

N means none of the localized candidates corresponds to the target.
Return only one label from: {", ".join(labels)}
""".strip()


def _llm_planner_prompt(user_text, perception_context, action_schema):
    perception_json = json.dumps(perception_context, separators=(",", ":"))
    actions_json = json.dumps(action_schema, separators=(",", ":"))

    return f"""
You are a high-level robotic task planner. Your job is to choose a compact
sequence actions from the provided action schema to complete the user request.

The perception system has independently associated semantic target descriptions
with localized object IDs. Infer manipulation, destination, and reference
relationships only from the original user instruction. Do not change the
provided object identities.
You must not write robot code or invent low-level motion commands.


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
