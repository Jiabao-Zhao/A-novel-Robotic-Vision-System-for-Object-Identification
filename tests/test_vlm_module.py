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
    OpenAIVLM,
    TOP_LOGPROBS_LIMIT,
    associate_targets,
    associate_targets_from_localization,
    candidate_relative_scores,
    candidate_choice_map,
    choice_for_object_id,
    console_human_resolver,
    decision_sequence_log_probability,
    infer_target_association,
    load_localized_objects,
    object_id_for_choice,
    resolve_association,
    _gemini_provider_result,
    _openai_provider_result,
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


def provider_result(
    choice="A",
    log_probability=math.log(0.9),
    top_logprobs=None,
    token_logprobs=None,
):
    return {
        "provider": "test",
        "model": "test-vlm",
        "generated_text": choice,
        "token_logprobs": token_logprobs
        if token_logprobs is not None
        else [
            {
                "token": choice,
                "log_probability": log_probability,
                "top_logprobs": top_logprobs or [],
            }
        ],
        "logprob_error": None,
        "top_logprob_error": None,
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


def inference(score=0.9, vlm_object_id="object_002", choice_label="B"):
    diagnostics = {
        "provider": "test",
        "model": "test-vlm",
        "candidate_map": {"A": "object_001", "B": "object_002"},
        "candidate_distribution_complete": False,
    }
    if choice_label is not None:
        diagnostics["choice_label"] = choice_label
    if score is not None:
        diagnostics.update(
            {
                "raw_log_probability": math.log(score),
                "raw_association_likelihood": score,
                "score_type": "raw_label_likelihood",
            }
        )
    else:
        diagnostics["logprob_error"] = "unavailable"
    return {
        "target_description": "white gear",
        "vlm_object_id": vlm_object_id,
        "association_score": score,
        "diagnostics": diagnostics,
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
        self.assertEqual(
            set(result),
            {
                "target_description",
                "vlm_object_id",
                "association_score",
                "threshold",
                "final_object_id",
                "resolution",
                "diagnostics",
            },
        )

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
        self.assertEqual(result["resolution"], "human_confirmed")

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
        self.assertEqual(result["diagnostics"]["target_absence_source"], "human")

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

    def test_low_confidence_without_human_resolver_is_deferred(self):
        result = resolve_association(inference(score=0.2), threshold=0.75)

        self.assertIsNone(result["final_object_id"])
        self.assertEqual(result["resolution"], "deferred")

    def test_score_equal_to_threshold_is_accepted(self):
        result = resolve_association(inference(score=0.75), threshold=0.75)

        self.assertEqual(result["final_object_id"], "object_002")
        self.assertEqual(result["resolution"], "vlm_accepted")

    def test_score_above_threshold_is_accepted(self):
        result = resolve_association(inference(score=0.750001), threshold=0.75)

        self.assertEqual(result["final_object_id"], "object_002")
        self.assertEqual(result["resolution"], "vlm_accepted")

    def test_gate_rejects_score_that_differs_from_raw_likelihood(self):
        result = inference(score=0.9)
        result["association_score"] = 0.8

        with self.assertRaisesRegex(ValueError, "raw_association_likelihood"):
            resolve_association(result, threshold=0.75)

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
                "candidate_object_ids": ["object_001", "object_002"],
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
            inference(score=0.9, vlm_object_id=None, choice_label="N"),
            threshold=0.75,
        )

        self.assertIsNone(result["final_object_id"])
        self.assertEqual(result["resolution"], "target_not_present")
        self.assertEqual(result["diagnostics"]["target_absence_source"], "vlm")

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
                        "diagnostics": {
                            "provider": "ignored-by-cad",
                            "choice_label": "B",
                        },
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

    def test_pure_leading_whitespace_does_not_affect_decision_likelihood(self):
        tokens, raw_log_probability = decision_sequence_log_probability(
            "A",
            [
                {"token": "\n", "log_probability": -0.1},
                {"token": "A", "log_probability": -0.2},
                {"token": ".", "log_probability": -3.0},
            ],
        )

        self.assertEqual([item["token"] for item in tokens], ["A"])
        self.assertAlmostEqual(raw_log_probability, -0.2)

    def test_selected_label_logprob_is_extracted_without_trailing_text(self):
        tokens, raw_log_probability = decision_sequence_log_probability(
            "B",
            [
                {"token": " B", "log_probability": -0.25},
                {"token": ".", "log_probability": -3.0},
            ],
        )

        self.assertEqual([item["token"] for item in tokens], [" B"])
        self.assertAlmostEqual(raw_log_probability, -0.25)

    def test_inference_ignores_punctuation_after_identified_label(self):
        provider = FakeProvider(
            provider_result(
                "A.",
                token_logprobs=[
                    {"token": "A", "log_probability": -0.2},
                    {"token": ".", "log_probability": -3.0},
                ],
            )
        )

        result = infer_target_association(
            "white gear",
            "annotated.png",
            DETECTIONS,
            provider=provider,
        )

        self.assertEqual(result["vlm_object_id"], "object_001")
        self.assertAlmostEqual(result["association_score"], math.exp(-0.2))
        self.assertAlmostEqual(
            result["diagnostics"]["raw_log_probability"],
            -0.2,
        )

    def test_multi_token_decision_likelihood_uses_only_decision_tokens(self):
        tokens, raw_log_probability = decision_sequence_log_probability(
            "AB",
            [
                {"token": "\n", "log_probability": -0.1},
                {"token": "A", "log_probability": -0.2},
                {"token": "B", "log_probability": -0.3},
                {"token": "!", "log_probability": -4.0},
            ],
        )

        self.assertEqual([item["token"] for item in tokens], ["A", "B"])
        self.assertAlmostEqual(raw_log_probability, -0.5)

    def test_candidate_relative_distribution_requires_every_valid_label(self):
        choice_map = {"A": "object_001", "B": "object_002"}
        decision_tokens = [
            {
                "token": "A",
                "log_probability": math.log(0.7),
                "top_logprobs": [
                    {"token": "A", "log_probability": math.log(0.7)},
                    {"token": " B", "log_probability": math.log(0.2)},
                    {"token": "N", "log_probability": math.log(0.1)},
                ],
            }
        ]

        available, scores, margin = candidate_relative_scores(
            choice_map,
            decision_tokens,
        )

        self.assertEqual(set(available), {"A", "B", "N"})
        self.assertAlmostEqual(scores["object_001"], 0.7)
        self.assertAlmostEqual(scores["object_002"], 0.2)
        self.assertAlmostEqual(scores["none"], 0.1)
        self.assertAlmostEqual(margin, 0.5)

    def test_incomplete_top_logprobs_do_not_fabricate_distribution(self):
        choice_map = {"A": "object_001", "B": "object_002"}
        available, scores, margin = candidate_relative_scores(
            choice_map,
            [
                {
                    "token": "A",
                    "log_probability": math.log(0.7),
                    "top_logprobs": [
                        {"token": "A", "log_probability": math.log(0.7)},
                        {"token": "B", "log_probability": math.log(0.2)},
                    ],
                }
            ],
        )

        self.assertEqual(set(available), {"A", "B"})
        self.assertIsNone(scores)
        self.assertIsNone(margin)

    def test_complete_distribution_is_diagnostic_and_never_replaces_raw_score(self):
        provider = FakeProvider(
            provider_result(
                "A",
                math.log(0.4),
                top_logprobs=[
                    {"token": "A", "log_probability": math.log(0.4)},
                    {"token": "B", "log_probability": math.log(0.1)},
                    {"token": "N", "log_probability": math.log(0.1)},
                ],
            )
        )

        result = infer_target_association(
            "white gear",
            "annotated.png",
            DETECTIONS,
            provider=provider,
        )

        self.assertAlmostEqual(result["association_score"], 0.4)
        diagnostics = result["diagnostics"]
        self.assertEqual(diagnostics["score_type"], "raw_label_likelihood")
        self.assertAlmostEqual(diagnostics["raw_association_likelihood"], 0.4)
        self.assertAlmostEqual(
            diagnostics["candidate_scores"]["object_001"],
            2 / 3,
        )
        self.assertAlmostEqual(diagnostics["association_margin"], 0.5)
        gated = resolve_association(result, threshold=0.5)
        self.assertEqual(gated["resolution"], "deferred")
        self.assertIsNone(gated["final_object_id"])

    def test_incomplete_distribution_falls_back_to_raw_label_likelihood(self):
        provider = FakeProvider(
            provider_result(
                "A",
                math.log(0.7),
                top_logprobs=[
                    {"token": "A", "log_probability": math.log(0.7)},
                    {"token": "B", "log_probability": math.log(0.2)},
                ],
            )
        )

        result = infer_target_association(
            "white gear",
            "annotated.png",
            DETECTIONS,
            provider=provider,
        )

        self.assertAlmostEqual(result["association_score"], 0.7)
        diagnostics = result["diagnostics"]
        self.assertFalse(diagnostics["candidate_distribution_complete"])
        self.assertNotIn("candidate_scores", diagnostics)
        self.assertEqual(diagnostics["score_type"], "raw_label_likelihood")

    def test_top_logprob_availability_never_changes_score_definition(self):
        complete = FakeProvider(
            provider_result(
                "A",
                math.log(0.4),
                top_logprobs=[
                    {"token": "A", "log_probability": math.log(0.4)},
                    {"token": "B", "log_probability": math.log(0.1)},
                    {"token": "N", "log_probability": math.log(0.1)},
                ],
            )
        )
        chosen_only = FakeProvider(provider_result("A", math.log(0.4)))

        complete_result = infer_target_association(
            "white gear", "annotated.png", DETECTIONS, provider=complete
        )
        chosen_only_result = infer_target_association(
            "white gear", "annotated.png", DETECTIONS, provider=chosen_only
        )

        self.assertAlmostEqual(complete_result["association_score"], 0.4)
        self.assertAlmostEqual(chosen_only_result["association_score"], 0.4)
        self.assertEqual(
            complete_result["diagnostics"]["score_type"],
            "raw_label_likelihood",
        )
        self.assertEqual(
            chosen_only_result["diagnostics"]["score_type"],
            "raw_label_likelihood",
        )

    def test_openai_parser_reads_actual_top_logprob_shape(self):
        response = SimpleNamespace(
            model="gpt-4.1-mini-2025-04-14",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="A"),
                    logprobs=SimpleNamespace(
                        content=[
                            SimpleNamespace(
                                token="A",
                                logprob=-0.1,
                                top_logprobs=[
                                    SimpleNamespace(token="A", logprob=-0.1),
                                    SimpleNamespace(token="B", logprob=-1.2),
                                ],
                            )
                        ]
                    ),
                )
            ],
        )

        parsed = _openai_provider_result(response, "gpt-4.1-mini")

        self.assertEqual(parsed["token_logprobs"][0]["token"], "A")
        self.assertEqual(
            parsed["token_logprobs"][0]["top_logprobs"][1],
            {"token": "B", "log_probability": -1.2},
        )

    def test_openai_provider_requests_maximum_top_logprobs(self):
        calls = []
        response = SimpleNamespace(
            model="gpt-4.1-mini-2025-04-14",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="A"),
                    logprobs=SimpleNamespace(
                        content=[
                            SimpleNamespace(
                                token="A",
                                logprob=-0.1,
                                top_logprobs=[],
                            )
                        ]
                    ),
                )
            ],
        )

        def create(**kwargs):
            calls.append(kwargs)
            return response

        provider = object.__new__(OpenAIVLM)
        provider.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        provider.model_name = "gpt-4.1-mini"
        provider._logprobs_supported = None
        provider._top_logprobs_supported = None
        provider._logprob_error = None
        provider._top_logprob_error = None
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "image.png"
            image_path.write_bytes(b"test-image")
            provider.associate(image_path, "Choose A or N")

        self.assertTrue(calls[0]["logprobs"])
        self.assertEqual(calls[0]["top_logprobs"], TOP_LOGPROBS_LIMIT)

    def test_gemini_parser_reads_actual_top_candidate_shape(self):
        response = SimpleNamespace(
            text="A",
            candidates=[
                SimpleNamespace(
                    logprobs_result=SimpleNamespace(
                        chosen_candidates=[
                            SimpleNamespace(token="A", log_probability=-0.1)
                        ],
                        top_candidates=[
                            SimpleNamespace(
                                candidates=[
                                    SimpleNamespace(
                                        token="A",
                                        log_probability=-0.1,
                                    ),
                                    SimpleNamespace(
                                        token="N",
                                        log_probability=-2.0,
                                    ),
                                ]
                            )
                        ],
                    )
                )
            ],
        )

        parsed = _gemini_provider_result(response, "gemini-test")

        self.assertEqual(parsed["token_logprobs"][0]["token"], "A")
        self.assertEqual(
            parsed["token_logprobs"][0]["top_logprobs"][1],
            {"token": "N", "log_probability": -2.0},
        )

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
        self.assertIsNone(result["association_score"])
        diagnostics = result["diagnostics"]
        self.assertIsNone(diagnostics["raw_log_probability"])
        self.assertIsNone(diagnostics["raw_association_likelihood"])
        self.assertEqual(diagnostics["score_type"], "raw_label_likelihood")
        self.assertEqual(diagnostics["logprob_error"], "unsupported")

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
            saved = json.loads(output_path.read_text(encoding="utf-8"))
            result = saved["associations"][0]

        self.assertEqual(saved["schema_version"], 4)
        self.assertEqual(
            set(result),
            {
                "target_description",
                "vlm_object_id",
                "association_score",
                "threshold",
                "final_object_id",
                "resolution",
                "diagnostics",
            },
        )
        diagnostics = result["diagnostics"]
        self.assertEqual(diagnostics["provider"], "test")
        self.assertEqual(diagnostics["model"], "test-vlm")
        self.assertAlmostEqual(diagnostics["raw_log_probability"], math.log(0.9))
        self.assertEqual(diagnostics["score_type"], "raw_label_likelihood")
        self.assertNotIn("model_output", diagnostics)
        self.assertNotIn("model_choice", diagnostics)
        self.assertNotIn("decision_token_logprobs", diagnostics)

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
