import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from main import register_resolved_associations
from prompt import _vlm_prompt
from vlm_module import (
    HumanClarificationRequired,
    associate_targets,
    associate_targets_from_localization,
    candidate_choice_map,
    choice_for_object_id,
    console_human_resolver,
    decision_sequence_log_probability,
    infer_target_association,
    load_localized_objects,
    object_id_for_choice,
    resolve_association,
)


DETECTIONS = [
    {
        "object_id": "object_002",
        "bbox_2d_xyxy": [30, 40, 50, 60],
        "centroid_3d_m": [0.2, 0.0, 1.0],
        "size_3d_m": [0.04, 0.04, 0.02],
    },
    {
        "object_id": "object_001",
        "bbox_2d_xyxy": [10, 20, 30, 40],
        "centroid_3d_m": [0.1, 0.0, 1.0],
        "size_3d_m": [0.05, 0.05, 0.03],
    },
]


def provider_result(choice="A", log_probability=math.log(0.9)):
    return {
        "provider": "test",
        "model": "test-vlm",
        "generated_text": choice,
        "token_logprobs": [
            {"token": choice, "log_probability": log_probability}
        ],
        "logprob_error": None,
    }


class FakeProvider:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts = []

    def associate(self, image_path, prompt):
        self.prompts.append(prompt)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def inference(score=0.9, vlm_object_id="object_002", model_choice="B"):
    return {
        "target_description": "white gear",
        "provider": "test",
        "model": "test-vlm",
        "model_choice": model_choice,
        "vlm_object_id": vlm_object_id,
        "raw_log_probability": math.log(score) if score is not None else None,
        "association_score": score,
        "candidate_map": {"A": "object_001", "B": "object_002"},
        "decision_token_logprobs": [],
        "logprob_error": None if score is not None else "unavailable",
    }


class VLMModuleTests(unittest.TestCase):
    def test_candidate_choice_maps_to_object_id_deterministically(self):
        choice_map = candidate_choice_map(DETECTIONS)

        self.assertEqual(choice_map, {"A": "object_001", "B": "object_002"})
        self.assertEqual(object_id_for_choice(choice_map, "B"), "object_002")
        self.assertIsNone(object_id_for_choice(choice_map, "N"))

    def test_object_id_maps_to_candidate_choice(self):
        choice_map = candidate_choice_map(DETECTIONS)

        self.assertEqual(choice_for_object_id(choice_map, "object_001"), "A")
        with self.assertRaisesRegex(ValueError, "Unknown localized object ID"):
            choice_for_object_id(choice_map, "object_999")

    def test_prompt_is_one_target_compact_choice_without_task_roles(self):
        prompt = _vlm_prompt(
            "white gear",
            [{"choice": "A", "object_id": "object_001"}],
        )

        self.assertIn("Target: white gear", prompt)
        self.assertIn("N means none", prompt)
        self.assertIn("Return only one label", prompt)
        self.assertNotIn("moved_object", prompt)
        self.assertNotIn("reference_object", prompt)
        self.assertNotIn("put the white gear", prompt)

    def test_high_confidence_association_is_automatically_accepted(self):
        result = resolve_association(inference(score=0.91), threshold=0.75)

        self.assertEqual(result["final_object_id"], "object_002")
        self.assertEqual(result["resolution"], "vlm_accepted")
        self.assertFalse(result["human_intervention"])

    def test_low_confidence_association_invokes_human_resolver(self):
        calls = []

        def resolver(request):
            calls.append(request)
            return "object_002"

        result = resolve_association(
            inference(score=0.61),
            threshold=0.75,
            human_resolver=resolver,
        )

        self.assertEqual(len(calls), 1)
        self.assertTrue(result["human_intervention"])

    def test_human_can_confirm_vlm_selection(self):
        result = resolve_association(
            inference(score=0.61),
            threshold=0.75,
            human_resolver=lambda _: "object_002",
        )

        self.assertEqual(result["final_object_id"], "object_002")
        self.assertEqual(result["resolution"], "human_confirmed")
        self.assertAlmostEqual(result["association_score"], 0.61)

    def test_human_can_correct_vlm_selection(self):
        result = resolve_association(
            inference(score=0.61),
            threshold=0.75,
            human_resolver=lambda _: "object_001",
        )

        self.assertEqual(result["final_object_id"], "object_001")
        self.assertEqual(result["resolution"], "human_corrected")
        self.assertAlmostEqual(result["association_score"], 0.61)

    def test_human_can_choose_none(self):
        result = resolve_association(
            inference(score=0.61),
            threshold=0.75,
            human_resolver=lambda _: "none",
        )

        self.assertIsNone(result["final_object_id"])
        self.assertEqual(result["resolution"], "target_not_present")
        self.assertTrue(result["human_intervention"])

    def test_confidence_unavailable_defers_to_human(self):
        result = resolve_association(
            inference(score=None),
            threshold=0.75,
            human_resolver=lambda _: "object_001",
        )

        self.assertEqual(result["final_object_id"], "object_001")
        self.assertEqual(result["resolution"], "human_corrected")
        self.assertIsNone(result["association_score"])

    def test_confidence_unavailable_is_explicit_without_human_resolver(self):
        result = resolve_association(inference(score=None), threshold=0.75)

        self.assertIsNone(result["final_object_id"])
        self.assertEqual(result["resolution"], "confidence_unavailable")
        self.assertTrue(result["requires_human_clarification"])

    def test_low_confidence_without_human_resolver_fails_clearly(self):
        with self.assertRaises(HumanClarificationRequired):
            resolve_association(inference(score=0.2), threshold=0.75)

    def test_invalid_human_object_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid localized object ID"):
            resolve_association(
                inference(score=0.4),
                threshold=0.75,
                human_resolver=lambda _: "object_999",
            )

    def test_console_clarification_shows_same_visual_prompt_and_candidates(self):
        request = inference(score=0.4)
        request.update(
            {
                "visual_prompt_path": "annotated.png",
                "threshold": 0.75,
            }
        )
        output = io.StringIO()
        with patch("builtins.input", return_value="object_001"), redirect_stdout(output):
            selection = console_human_resolver(request)

        displayed = output.getvalue()
        self.assertEqual(selection, "object_001")
        self.assertIn("annotated.png", displayed)
        self.assertIn("Target: white gear", displayed)
        self.assertIn("object_002", displayed)
        self.assertIn("object_001", displayed)
        self.assertIn("none", displayed)

    def test_high_confidence_none_choice_reports_target_not_present(self):
        result = resolve_association(
            inference(score=0.9, vlm_object_id=None, model_choice="N"),
            threshold=0.75,
        )

        self.assertIsNone(result["final_object_id"])
        self.assertEqual(result["resolution"], "target_not_present")
        self.assertFalse(result["requires_human_clarification"])

    def test_cad_retrieval_does_not_run_for_null_final_object_id(self):
        calls = []
        with self.assertRaisesRegex(LookupError, "CAD retrieval stopped"):
            register_resolved_associations(
                [
                    {
                        "target_description": "white gear",
                        "final_object_id": None,
                    }
                ],
                localization_payload={"objects": []},
                plane_model=[0, 0, 1, 0],
                cad_retrieval=lambda query: calls.append(query),
            )

        self.assertEqual(calls, [])

    def test_cad_retrieval_uses_target_description_not_task_instruction(self):
        queries = []

        def retrieve(query):
            queries.append(query)
            return SimpleNamespace(file_path="white_gear.stl"), {
                "selected_cad_name": "white gear",
                "selected_file_path": "white_gear.stl",
            }

        registrar = SimpleNamespace(
            run=lambda **_: {
                "aligned_cad_cloud_path": "aligned.ply",
                "augmented_cloud_path": "augmented.ply",
                "result_path": "registration.json",
            }
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            register_resolved_associations(
                [
                    {
                        "target_description": "white gear",
                        "final_object_id": "object_002",
                    }
                ],
                localization_payload={
                    "objects": [
                        {
                            "object_id": "object_002",
                            "pointcloud_path": "observed.ply",
                        }
                    ]
                },
                plane_model=[0, 0, 1, 0],
                cad_retrieval=retrieve,
                registrar=registrar,
                registration_root=Path(temporary_directory),
            )

        self.assertEqual(queries, ["white gear"])
        self.assertNotIn("put the white gear", queries)

    def test_vlm_result_contains_no_task_role_reasoning(self):
        provider = FakeProvider(provider_result("B"))
        result = associate_targets(
            ["white gear"],
            image_path="annotated.png",
            detections=DETECTIONS,
            threshold=0.75,
            provider=provider,
        )[0]
        serialized = json.dumps(result)

        self.assertNotIn("instruction_role", serialized)
        self.assertNotIn("moved_object", serialized)
        self.assertNotIn("reference_object", serialized)
        self.assertNotIn("target_match", serialized)

    def test_multiple_semantic_targets_are_associated_independently(self):
        provider = FakeProvider(provider_result("B"), provider_result("A"))
        results = associate_targets(
            ["white gear", "red block"],
            image_path="annotated.png",
            detections=DETECTIONS,
            threshold=0.75,
            provider=provider,
        )

        self.assertEqual(
            [(item["target_description"], item["final_object_id"]) for item in results],
            [("white gear", "object_002"), ("red block", "object_001")],
        )
        self.assertEqual(len(provider.prompts), 2)

    def test_multi_token_decision_likelihood_uses_logprobability_sum(self):
        tokens, raw_log_probability = decision_sequence_log_probability(
            "A",
            [
                {"token": "\n", "log_probability": -0.1},
                {"token": "A", "log_probability": -0.2},
                {"token": ".", "log_probability": -3.0},
            ],
        )

        self.assertEqual([item["token"] for item in tokens], ["\n", "A"])
        self.assertAlmostEqual(raw_log_probability, -0.3)
        self.assertAlmostEqual(math.exp(raw_log_probability), math.exp(-0.3))

    def test_provider_without_logprobs_never_gets_fake_confidence(self):
        provider = FakeProvider(
            {
                "provider": "test",
                "model": "no-logprobs",
                "generated_text": "A",
                "token_logprobs": [],
                "logprob_error": "unsupported",
            }
        )
        result = infer_target_association(
            "white gear",
            "annotated.png",
            DETECTIONS,
            provider=provider,
        )

        self.assertEqual(result["vlm_object_id"], "object_001")
        self.assertIsNone(result["raw_log_probability"])
        self.assertIsNone(result["association_score"])

    def test_provider_failure_never_creates_association_result(self):
        provider = FakeProvider(RuntimeError("provider unavailable"))
        with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
            infer_target_association(
                "white gear",
                "annotated.png",
                DETECTIONS,
                provider=provider,
            )

    def test_saved_contract_preserves_required_experiment_fields(self):
        localization = {
            "frame": "camera",
            "objects": [
                {
                    "object_id": "object_001",
                    "roi": {"x1": 10, "y1": 20, "x2": 30, "y2": 40},
                    "centroid_3d_m": [0.1, 0.0, 1.0],
                    "size_3d_m": [0.05, 0.05, 0.03],
                    "point_count": 100,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            localization_path = Path(temporary_directory) / "localization.json"
            output_path = Path(temporary_directory) / "associations.json"
            localization_path.write_text(json.dumps(localization), encoding="utf-8")
            associate_targets_from_localization(
                ["white gear"],
                image_path="annotated.png",
                localization_path=localization_path,
                output_path=output_path,
                threshold=0.75,
                provider=FakeProvider(provider_result("A")),
            )
            result = json.loads(output_path.read_text(encoding="utf-8"))["associations"][0]

        expected = {
            "target_description",
            "provider",
            "model",
            "vlm_object_id",
            "raw_log_probability",
            "association_score",
            "threshold",
            "final_object_id",
            "resolution",
            "human_intervention",
        }
        self.assertTrue(expected.issubset(result))

    def test_localized_prompt_keeps_depth_derived_geometry(self):
        localization = {
            "frame": "camera",
            "objects": [
                {
                    "object_id": "object_003",
                    "roi": {"x1": 10, "y1": 20, "x2": 30, "y2": 50},
                    "centroid_3d_m": [0.1, -0.2, 1.0],
                    "size_3d_m": [0.05, 0.08, 0.12],
                    "point_count": 420,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "localization.json"
            path.write_text(json.dumps(localization), encoding="utf-8")
            detections = load_localized_objects(path)

        self.assertEqual(detections[0]["centroid_3d_m"], [0.1, -0.2, 1.0])
        self.assertEqual(detections[0]["size_3d_m"], [0.05, 0.08, 0.12])
        self.assertEqual(detections[0]["point_count"], 420)
        self.assertEqual(detections[0]["geometry_frame"], "camera")


if __name__ == "__main__":
    unittest.main()
