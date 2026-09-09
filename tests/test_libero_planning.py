import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from openai import OpenAIError
from openai.types.chat import ChatCompletion

from LLM_planner import OpenAIPlanner
from simulation.libero_planning import (
    PLAN_RESPONSE_FORMAT, build_simulation_context, evaluate_simulation_plan,
    execute_simulation_plan, generate_simulation_plan, reference_pick_place_plan,
    validate_simulation_plan,
)


def scene_context():
    objects = []
    for index, name in enumerate(("milk", "butter", "basket", "other basket")):
        pose = np.eye(4)
        pose[:3, 3] = [index * .05, -.1, .06]
        objects.append({"object_id": f"object_{index}", "description": name,
                        "world_T_grasp": pose.tolist() if index < 2 else None,
                        "centroid_world_m": [index * .05, .2, .08],
                        "placement_relations": [] if index < 2 else ["in"]})
    return {"objects": objects, "frame": "MuJoCo world", "length_unit": "m",
            "initially_holding_object_id": None}


def completion(text, refusal=None, finish_reason="stop"):
    return ChatCompletion.model_validate({
        "id": "test-request", "created": 0, "model": "test-model", "object": "chat.completion",
        "choices": [{"index": 0, "finish_reason": finish_reason,
                     "message": {"role": "assistant", "content": text, "refusal": refusal}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125},
    })


class SimulationPlanningTests(unittest.TestCase):
    def setUp(self):
        self.context = scene_context()
        self.plan = reference_pick_place_plan("object_0", "object_2")

    def test_pickup_only_requires_explicit_held_completion_condition(self):
        pickup = {"status": "ready", "reason": "Lift and hold.",
                  "actions": [{"action": "pick", "object_id": "object_0"}]}
        with self.assertRaises(ValueError):
            validate_simulation_plan(pickup, self.context)
        self.context["completion_condition"] = "held"
        validate_simulation_plan(pickup, self.context)
        with self.assertRaises(ValueError):
            validate_simulation_plan(self.plan, self.context)

    def test_structured_request_preserves_physical_planner_interface(self):
        planner = OpenAIPlanner.__new__(OpenAIPlanner)
        planner.model_name = "test-model"
        planner.client = MagicMock()
        planner.client.chat.completions.create.return_value = completion(json.dumps(self.plan))
        self.assertEqual(json.loads(planner.plan("physical prompt")), self.plan)
        self.assertNotIn("response_format", planner.client.chat.completions.create.call_args.kwargs)
        planner.complete("simulation prompt", PLAN_RESPONSE_FORMAT)
        self.assertEqual(planner.client.chat.completions.create.call_args.kwargs["response_format"],
                         PLAN_RESPONSE_FORMAT)

    def test_rejects_invalid_sequences_and_invented_motion_arguments_before_any_motion(self):
        pick, place = self.plan["actions"]
        invalid_actions = [
            [], [place, pick], [pick], [pick, pick, place],
            [{**pick, "object_id": "missing"}, place],
            [pick, {**place, "object_id": "object_1"}],
            [pick, {**place, "destination_id": "missing"}],
            [pick, {**place, "destination_id": "object_0"}],
            [pick, {**place, "relation": "on"}],
            [{**pick, "action": "turn_on_stove"}, place],
            [{**pick, "xyz": [0, 0, 0]}, place],
            [{**pick, "object_id": "object_2"}, place],
            [pick, place, pick, place],
        ]
        for actions in invalid_actions:
            with self.subTest(actions=actions), patch("simulation.libero_planning.pick_object") as pick_skill:
                with self.assertRaises(ValueError):
                    execute_simulation_plan(None, {}, {**self.plan, "actions": actions}, self.context)
                pick_skill.assert_not_called()

    def test_executor_uses_model_selected_objects_destinations_and_order(self):
        plan = reference_pick_place_plan("object_1", "object_3")
        plan["actions"] += self.plan["actions"]
        calls = []
        observations = []
        def pick(env, obs, grasp, callback):
            calls.append(("pick", grasp))
            observations.append(obs)
            callback("close", 0, {}, np.zeros(7), 0., False, {})
            return {"picked": len(calls)}
        def place(env, obs, grasp, destination, callback):
            calls.append(("place", destination))
            observations.append(obs)
            callback("release", 0, {}, np.zeros(7), 0., False, {})
            return {"placed": len(calls)}
        callback = MagicMock()
        with patch("simulation.libero_planning.pick_object", side_effect=pick), patch(
            "simulation.libero_planning.place_object", side_effect=place
        ):
            result = execute_simulation_plan(None, {"initial": True}, plan, self.context, callback)
        self.assertEqual([item[0] for item in calls], ["pick", "place", "pick", "place"])
        self.assertEqual(calls[0][1], self.context["objects"][1]["world_T_grasp"])
        self.assertEqual(calls[1][1], self.context["objects"][3]["centroid_world_m"])
        self.assertEqual(observations[1], {"picked": 1})
        self.assertEqual(result, {"placed": 4})
        self.assertEqual([call.args[0] for call in callback.call_args_list], [0, 1, 2, 3])

    def test_semantic_evaluation_does_not_repair_or_constrain_executable_plan(self):
        wrong = reference_pick_place_plan("object_1", "object_3")
        original = copy.deepcopy(wrong)
        validate_simulation_plan(wrong, self.context)
        result = evaluate_simulation_plan({"status": "ready", "plan": wrong}, "object_0", "object_2")
        self.assertEqual(result["status"], "task_mismatch")
        self.assertIsNone(result["causal_failure_stage"])
        self.assertEqual(wrong, original)

    def test_provider_results_preserve_raw_evidence_and_failure_categories(self):
        cases = [
            (completion(json.dumps(self.plan)), "ready"),
            (completion("not JSON"), "invalid_plan"),
            (completion(json.dumps({**self.plan, "actions": self.plan["actions"][::-1]})), "invalid_plan"),
            (completion(json.dumps({"status": "blocked", "reason": "Cannot plan", "actions": []})), "blocked"),
            (completion(None, refusal="Refused"), "refused"),
            (completion('{"status":', finish_reason="length"), "incomplete_response"),
            (OpenAIError("test connection failure"), "service_error"),
        ]
        for response, status in cases:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as root:
                output = Path(root) / "llm_plan.json"
                with patch("simulation.libero_planning.OpenAIPlanner") as factory:
                    factory.return_value.model_name = "test-model"
                    def request(*args, **kwargs):
                        self.assertEqual(json.loads(output.read_text())["status"], "request_started")
                        if isinstance(response, Exception):
                            raise response
                        return response
                    factory.return_value.complete.side_effect = request
                    record = generate_simulation_plan("put milk in basket", self.context, output)
                saved = json.loads(output.read_text())
                self.assertEqual(record, saved)
                self.assertEqual(record["status"], status)
                self.assertEqual(record["context"], self.context)
                if status != "service_error":
                    self.assertEqual(record["provider_response"]["id"], "test-request")
                    self.assertEqual(record["provider_response"]["usage"]["total_tokens"], 125)
                if status == "invalid_plan" and response.choices[0].message.content == "not JSON":
                    self.assertEqual(record["raw_response"], "not JSON")

    def test_context_uses_estimated_geometry_without_role_or_ground_truth_leakage(self):
        localization = {"objects": [
            {"object_id": "a", "centroid_3d_m": [1, 2, 3], "size_3d_m": [.1, .2, .3]},
            {"object_id": "b", "centroid_3d_m": [0, .2, .1], "size_3d_m": [.3, .3, .1]},
        ], "ground_truth": "must not appear"}
        associations = [{"final_object_id": key, "target_description": name,
                         "resolution": "vlm_accepted"} for key, name in (("b", "basket"), ("a", "milk"))]
        registration = {"object_id": "a", "registered_center_world_m": [.1, -.2, .07],
                        "world_T_cad": np.eye(4).tolist(), "cad_extent_m": [.1, .2, .3]}
        camera = np.eye(4)
        camera[0, 3] = .2
        robot = {"robot0_eef_pos": [0, 0, .2], "robot0_eef_quat": [0, 0, 0, 1],
                 "robot0_gripper_qpos": [.04, .04], "object_ground_truth": "private"}
        context = build_simulation_context(localization, associations, registration, camera, np.eye(4), robot)
        self.assertEqual(context["objects"][0]["centroid_world_m"], [.1, -.2, .07])
        self.assertEqual(context["objects"][1]["centroid_world_m"], [.2, .2, .1])
        self.assertNotIn("ground_truth", json.dumps(context))
        self.assertNotIn("target_object_id", json.dumps(context))
        self.assertNotIn("destination_id", json.dumps(context))


if __name__ == "__main__":
    unittest.main()
