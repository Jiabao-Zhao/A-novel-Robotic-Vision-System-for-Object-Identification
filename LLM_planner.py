import json
import os
import re
from pathlib import Path

from prompt import _llm_planner_prompt


APPROVED_ACTION_SCHEMA = [
    {
        "name": "report_pose",
        "controller_method": "RTDECommander.report_pose",
        "description": "Report the selected object's estimated robot-base pose without moving the robot.",
        "arguments": {
            "object_id": "string",
            "object_type": "string",
        },
    },
    {
        "name": "pick_object",
        "controller_method": "RTDECommander.pick_object",
        "description": (
            "Pick the selected object. The executor must bind point_camera_mm "
            "and object_rpy_base_deg from perception outside the LLM."
        ),
        "arguments": {
            "object_id": "string",
            "object_type": "string",
            "grasp_width_mm": "float or null",
            "object_height_mm": "float or null",
            "clearance_z_mm": "float, normally 50.0",
            "go_home_after": "bool",
        },
    },
    {
        "name": "place_object",
        "controller_method": "RTDECommander.place_object",
        "description": "Place the currently grasped object at a target pose chosen by the executor.",
        "arguments": {
            "target_location": "string",
            "open_width_mm": "float or null",
            "clearance_z_mm": "float, normally 50.0",
            "go_home_after": "bool",
        },
    },
    {
        "name": "move_in_z",
        "controller_method": "RTDECommander.move_in_z",
        "description": "Move the current TCP along robot-base Z by dz millimeters.",
        "arguments": {
            "dz": "float mm",
        },
    },
    {
        "name": "go_home",
        "controller_method": "RTDECommander.go_home",
        "description": "Move the robot to its configured safe home joint pose.",
        "arguments": {
            "open_gripper": "bool",
            "pre_lift_z_mm": "float",
        },
    },
    {
        "name": "set_gripper_open",
        "controller_method": "RTDECommander.set_gripper_open",
        "description": "Open the RG2 gripper to the default width.",
        "arguments": {
            "blocking": "bool",
        },
    },
    {
        "name": "set_gripper_close",
        "controller_method": "RTDECommander.set_gripper_close",
        "description": "Close the RG2 gripper to the default width.",
        "arguments": {
            "blocking": "bool",
        },
    },
]


class OpenAIPlanner:
    def __init__(self):
        from openai import OpenAI

        self.client = OpenAI()
        self.model_name = os.environ.get("OPENAI_LLM_MODEL", "gpt-4.1-mini")

    def plan(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a robot task planner. Return only valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content or "{}"


def plan_from_outputs(
    user_text,
    vlm_path=Path("output/vlm/vlm_result.json"),
    registration_path=Path("output/registered_point_cloud/cad_registration_result.json"),
    robot_pose_path=Path("output/robot_pose/object_pose_base.json"),
    output_path=Path("output/planner/llm_plan.json"),
):
    context = load_perception_context(vlm_path, registration_path, robot_pose_path)
    prompt = _llm_planner_prompt(user_text, context, APPROVED_ACTION_SCHEMA)
    raw_response, provider = plan_with_openai(prompt)
    plan = parse_json_response(raw_response)
    validate_plan(plan)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "provider": provider,
        "user_instruction": str(user_text),
        "perception_context": context,
        "approved_action_schema": APPROVED_ACTION_SCHEMA,
        "raw_response": raw_response,
        "plan": plan,
        "execution_note": "This file is a plan only. It does not execute robot motion.",
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def load_perception_context(vlm_path, registration_path, robot_pose_path):
    vlm = json.loads(Path(vlm_path).read_text(encoding="utf-8"))
    selected = vlm.get("result", {})
    context = {
        "selected_object_id": selected.get("object_id"),
        "selected_object_type": selected.get("object_type"),
    }
    context["pose_available"] = Path(registration_path).exists() and Path(robot_pose_path).exists()
    return context


def plan_with_openai(prompt):
    return OpenAIPlanner().plan(prompt), "openai"


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


def validate_plan(plan):
    allowed_actions = {item["name"] for item in APPROVED_ACTION_SCHEMA}
    if plan.get("status") not in {"ready", "blocked"}:
        raise ValueError("Planner status must be 'ready' or 'blocked'.")
    actions = plan.get("actions")
    if not isinstance(actions, list):
        raise ValueError("Planner output must contain an actions list.")
    for action in actions:
        action_name = action.get("action")
        if action_name not in allowed_actions:
            raise ValueError(f"Planner returned unsupported action: {action_name}")


if __name__ == "__main__":
    print(plan_from_outputs("find the red block"))
