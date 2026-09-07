import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from openai.types.chat import ChatCompletion

from scripts.libero_task_execution import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LIBERO_RAW_ASSOCIATION_THRESHOLD,
    RUN_VARIANT,
    VLM_CONTACT_SHEET_TILE_SIZE_PX,
    _write_pipeline_failure,
    main,
    replay_with_reference_plan,
    task_associations_from_vlm_result,
)
from simulation.libero_planning import reference_pick_place_plan


def completion(text):
    return ChatCompletion.model_validate({
        "id": "test", "created": 0, "model": "test-model", "object": "chat.completion",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": text}}],
    })


def association(target_description, object_id, **updates):
    result = {
        "target_description": target_description,
        "final_object_id": object_id,
        "resolution": "vlm_accepted",
    }
    result.update(updates)
    return result


class LiberoTaskExecutionTests(unittest.TestCase):
    def test_proposed_resolution_ablation_uses_semantic_association(self):
        self.assertEqual((IMAGE_HEIGHT, IMAGE_WIDTH), (768, 768))
        self.assertEqual(VLM_CONTACT_SHEET_TILE_SIZE_PX, 448)
        self.assertEqual(
            RUN_VARIANT,
            "768x768_joint_instruction_assumed_human_v2",
        )
        self.assertAlmostEqual(
            LIBERO_RAW_ASSOCIATION_THRESHOLD,
            0.9999832372181827,
        )

    def test_task_uses_two_independent_semantic_associations(self):
        payload = {
            "associations": [
                association("basket", "object_001"),
                association("milk", "object_003"),
            ]
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_path = Path(temporary_directory) / "vlm_result.json"
            result_path.write_text(json.dumps(payload), encoding="utf-8")
            result = task_associations_from_vlm_result(result_path, "milk")

        self.assertEqual(result["target_object_id"], "object_003")
        self.assertEqual(result["basket_object_id"], "object_001")
        self.assertNotIn("instruction_role", json.dumps(result))

    def test_task_rejects_missing_semantic_target(self):
        payload = {"associations": [association("basket", "object_001")]}
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_path = Path(temporary_directory) / "vlm_result.json"
            result_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "independent associations"):
                task_associations_from_vlm_result(result_path, "milk")

    def test_task_rejects_unresolved_association(self):
        payload = {
            "associations": [
                association(
                    "milk",
                    None,
                    resolution="confidence_unavailable",
                ),
                association("basket", "object_001"),
            ]
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_path = Path(temporary_directory) / "vlm_result.json"
            result_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "human clarification"):
                task_associations_from_vlm_result(result_path, "milk")

    def test_task_rejects_target_not_present(self):
        payload = {
            "associations": [
                association("milk", None, resolution="target_not_present"),
                association("basket", "object_001"),
            ]
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_path = Path(temporary_directory) / "vlm_result.json"
            result_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Target not present"):
                task_associations_from_vlm_result(result_path, "milk")

    def test_task_rejects_same_object_for_both_associations(self):
        payload = {
            "associations": [
                association("butter", "object_002"),
                association("basket", "object_002"),
            ]
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_path = Path(temporary_directory) / "vlm_result.json"
            result_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "same localized object"):
                task_associations_from_vlm_result(result_path, "butter")

    def test_pre_control_failure_is_preserved_as_unsuccessful_episode(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            with patch(
                "scripts.libero_task_execution._official_initial_state_sha256",
                return_value="a" * 64,
            ):
                _write_pipeline_failure(
                    output_root=output_root,
                    task_index=6,
                    target_slug="butter",
                    instruction="pick up the butter and place it in the basket",
                    error=RuntimeError("association failed"),
                )
            payload = json.loads(
                (output_root / "episode.json").read_text(encoding="utf-8")
            )

        self.assertEqual(payload["run_status"], "pipeline_error")
        self.assertFalse(payload["success"])
        self.assertEqual(payload["termination_reason"], "pipeline_error")
        self.assertEqual(payload["initial_state_sha256"], "a" * 64)
        self.assertIn("association failed", payload["execution_error"])

    def test_existing_task_result_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "init_state_00"
            output_root.mkdir()
            episode_path = output_root / "episode.json"
            episode_path.write_text('{"success": true}', encoding="utf-8")
            with patch(
                "scripts.libero_task_execution.episode_result_dir",
                return_value=output_root,
            ), patch("scripts.libero_task_execution._run_task") as run_task:
                with self.assertRaisesRegex(SystemExit, "Refusing to overwrite"):
                    main(7)

            run_task.assert_not_called()
            self.assertEqual(
                episode_path.read_text(encoding="utf-8"),
                '{"success": true}',
            )

    def test_main_persists_pre_control_failure_for_comparison(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "init_state_00"
            with patch(
                "scripts.libero_task_execution.episode_result_dir",
                return_value=output_root,
            ), patch(
                "scripts.libero_task_execution._run_task",
                side_effect=RuntimeError("CAD registration failed"),
            ), patch(
                "scripts.libero_task_execution._official_initial_state_sha256",
                return_value="b" * 64,
            ):
                with self.assertRaisesRegex(RuntimeError, "CAD registration failed"):
                    main(6)

            payload = json.loads(
                (output_root / "episode.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["task_index"], 6)
            self.assertEqual(payload["run_status"], "pipeline_error")
            self.assertFalse(payload["success"])
            self.assertEqual(payload["initial_state_sha256"], "b" * 64)


class RunnerPlanningIntegrationTests(unittest.TestCase):
    """Exercise the complete runner/report/replay flow with deterministic inputs."""

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.raw = {"robot0_eef_pos": np.array([0., 0., .2]),
                    "robot0_eef_quat": np.array([0., 0., 0., 1.]),
                    "robot0_gripper_qpos": np.array([.04, .04]),
                    "agentview_image": np.zeros((8, 8, 3), dtype=np.uint8)}
        self.env = MagicMock()
        self.env.__enter__.return_value = self.env
        self.env.last_init_state_sha256 = "a" * 64
        self.env.initial_state_count = 50
        self.env.sim.get_state.return_value.flatten.return_value = np.array([0., 1., 2.])
        self.env.control_mode = "relative"
        self.env.last_observation = self.raw
        def reset(**kwargs):
            self.env.check_success.return_value = False
            return self.raw
        self.env.reset.side_effect = reset
        self.patch("LiberoTaskEnvironment", return_value=self.env)
        self.patch("hold_gripper", return_value=self.raw)
        robot = {key: value for key, value in self.raw.items() if key.startswith("robot0_")}
        observation = SimpleNamespace(
            instruction="pick up the milk and place it in the basket",
            world_T_camera=np.eye(4), robot_state=robot,
        )
        self.patch("LiberoRGBDSensor").return_value.capture.return_value = observation
        self.patch("save_libero_observation", return_value={"rgb": self.root / "rgb.png"})
        self.patch("save_video")
        self.patch("create_roi_contact_sheet", return_value=self.root / "contact.png")
        self.patch("contact_sheet_candidate_bbox_map", return_value={})
        localization = {"object_count": 2, "objects": [
            {"object_id": "object_0", "centroid_3d_m": [.05, -.1, .04], "size_3d_m": [.04, .04, .08]},
            {"object_id": "object_2", "centroid_3d_m": [.05, .2, .08], "size_3d_m": [.2, .2, .1]},
        ]}
        self.localizer = self.patch("run_libero_localization", return_value=(localization,
            {"localization": self.root / "loc.json", "annotated_rgb": self.root / "ann.png"},
            np.ones((8, 8), dtype=bool)))
        def associations(*args, **kwargs):
            output = kwargs["output_path"]
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({"associations": [association("milk", "object_0"),
                                                          association("basket", "object_2")]}))
            return output
        self.vlm = self.patch("associate_instruction_from_localization", side_effect=associations)
        pose = np.eye(4)
        pose[:3, :3] = np.diag([1., -1., -1.])
        pose[:3, 3] = [.05, -.1, .04]
        registration = {"object_id": "object_0", "registered_center_world_m": [.05, -.1, .04],
                        "world_T_cad": np.eye(4).tolist(), "cad_center_cad_m": [0., 0., .04],
                        "cad_extent_m": [.04, .04, .08], "cad_up_axis": "Z",
                        "exact_rendering_cad_prior_used": True, "constrained_rmse_m": .001,
                        "result_path": "cad.json", "aligned_cad_cloud_path": "aligned.ply",
                        "augmented_cloud_path": "augmented.ply"}
        self.cad = self.patch("register_libero_cad_to_observation", return_value=registration)
        self.patch("top_down_grasp_pose", return_value=pose)
        self.patch("_official_initial_state_sha256", return_value="a" * 64)
        self.planner = self.stack.enter_context(patch("simulation.libero_planning.OpenAIPlanner"))
        self.planner.return_value.model_name = "test-model"
        self.plan = reference_pick_place_plan("object_0", "object_2")
        self.planner.return_value.complete.return_value = completion(json.dumps(self.plan))
        def pick(env, raw, grasp, callback):
            callback("lift_target", 0, raw, np.zeros(7), 0., False, {})
            return raw
        def place(env, raw, grasp, destination, callback):
            self.env.check_success.return_value = True
            callback("release_target", 0, raw, np.zeros(7), 0., False, {})
            return raw
        self.pick = self.stack.enter_context(patch("simulation.libero_planning.pick_object", side_effect=pick))
        self.place = self.stack.enter_context(patch("simulation.libero_planning.place_object", side_effect=place))

    def patch(self, name, **kwargs):
        return self.stack.enter_context(patch("scripts.libero_task_execution." + name, **kwargs))

    def run_episode(self, name="original"):
        return main(7, initial_state_index=4, output_root=self.root / name)

    def test_generated_plan_drives_actions_and_report(self):
        result = json.loads(self.run_episode().read_text())
        self.assertTrue(result["success"])
        self.assertEqual(result["initial_state_index"], 4)
        self.assertEqual(result["planning_status"], "ready")
        self.assertEqual(result["plan_evaluation"]["status"], "consistent")
        self.assertEqual([action["plan_action_index"] for action in result["actions"]], [0, 1])
        self.assertEqual(result["actions"][1]["planned_action"]["destination_id"], "object_2")
        self.assertEqual(result["workspace_rgbd_pixels"], 64)
        self.vlm.assert_called_once()
        self.assertEqual(self.vlm.call_args.kwargs["threshold"], LIBERO_RAW_ASSOCIATION_THRESHOLD)

    def test_invalid_and_blocked_plans_are_unsuccessful_without_task_motion(self):
        for index, plan in enumerate((
            {**self.plan, "actions": self.plan["actions"][::-1]},
            {"status": "blocked", "reason": "Cannot plan", "actions": []},
        )):
            self.planner.return_value.complete.return_value = completion(json.dumps(plan))
            result = json.loads(self.run_episode(str(index)).read_text())
            self.assertFalse(result["success"])
            self.assertEqual(result["action_steps"], 0)
            self.assertEqual(result["run_status"], "planning_stopped")
            self.assertIsNone(result["execution_error"])
        self.pick.assert_not_called()
        self.place.assert_not_called()

    def test_controller_error_does_not_become_planning_error(self):
        self.pick.side_effect = RuntimeError("test controller failure")
        result = json.loads(self.run_episode().read_text())
        self.assertFalse(result["success"])
        self.assertEqual(result["planning_status"], "ready")
        self.assertEqual(result["termination_reason"], "execution_error")
        self.assertIsNone(result["plan_evaluation"]["causal_failure_stage"])

    def test_reference_replay_changes_only_plan_and_preserves_original(self):
        self.planner.return_value.complete.return_value = completion("not JSON")
        original = self.run_episode()
        before = original.read_bytes()
        replay = replay_with_reference_plan(original.parent, self.root / "reference")
        self.assertEqual(original.read_bytes(), before)
        result = json.loads(replay.read_text())
        self.assertTrue(result["success"])
        self.assertEqual(result["method"], "reference_plan_diagnostic")
        comparison = json.loads((replay.parent / "reference_comparison.json").read_text())
        self.assertTrue(comparison["planning_contribution_supported"])
        self.assertFalse(comparison["same_executable_plan"])
        self.assertFalse(comparison["included_in_main_evaluation"])
        self.planner.assert_called_once()
        self.localizer.assert_called_once()
        self.vlm.assert_called_once()
        self.cad.assert_called_once()

    def test_identical_plan_replay_cannot_attribute_execution_variability_to_planning(self):
        normal_pick = self.pick.side_effect
        self.pick.side_effect = RuntimeError("transient controller failure")
        original = self.run_episode()
        self.pick.side_effect = normal_pick
        replay = replay_with_reference_plan(original.parent, self.root / "reference")
        comparison = json.loads((replay.parent / "reference_comparison.json").read_text())
        self.assertTrue(comparison["reference_success"])
        self.assertTrue(comparison["same_executable_plan"])
        self.assertFalse(comparison["planning_contribution_supported"])
        self.assertIn("same plan produced different outcomes", comparison["interpretation"])

    def test_reference_replay_rejects_changed_initial_state(self):
        self.planner.return_value.complete.return_value = completion("not JSON")
        original = self.run_episode()
        self.env.last_init_state_sha256 = "b" * 64
        with self.assertRaisesRegex(ValueError, "initial-state hash"):
            replay_with_reference_plan(original.parent, self.root / "reference")
        self.pick.assert_not_called()

    def test_reference_replay_rejects_changed_frozen_perception(self):
        self.planner.return_value.complete.return_value = completion("not JSON")
        original = self.run_episode()
        path = original.parent / "execution_inputs.json"
        inputs = json.loads(path.read_text())
        inputs["perception"]["context"]["objects"][0]["centroid_world_m"][0] += .1
        path.write_text(json.dumps(inputs))
        with self.assertRaisesRegex(ValueError, "Frozen perception"):
            replay_with_reference_plan(original.parent, self.root / "reference")
        self.pick.assert_not_called()

    def test_reference_replay_rejects_changed_post_settle_simulator_state(self):
        self.planner.return_value.complete.return_value = completion("not JSON")
        original = self.run_episode()
        self.env.sim.get_state.return_value.flatten.return_value = np.array([0., 1., 3.])
        with self.assertRaisesRegex(ValueError, "simulator state differs after settling"):
            replay_with_reference_plan(original.parent, self.root / "reference")
        self.pick.assert_not_called()


if __name__ == "__main__":
    unittest.main()
