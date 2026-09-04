import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.run_vlm_output_format_ablation import build_prompt, extract_decision, request_arguments, MODEL
from vlm_module import OpenAIVLM
from scripts.analyze_vlm_output_format_ablation import paired_accuracy, validate_pairs


class OutputFormatAblationTests(unittest.TestCase):
    def test_only_representation_changes_between_prompts(self):
        objects = [{"object_id": "object_002", "bbox_2d_xyxy": [1,2,3,4]}, {"object_id": "object_001"}]
        a, mapping_a = build_prompt("milk", objects, "compact")
        b, mapping_b = build_prompt("milk", objects, "object_id")
        expected = a.replace('"choice":"A"', '"choice":"object_001"').replace('"choice":"B"', '"choice":"object_002"')
        expected = expected.replace("\nN means none", "\nnone means none").replace("from: A, B, N", "from: object_001, object_002, none")
        self.assertEqual(b, expected)
        self.assertEqual(list(mapping_a.values()), list(mapping_b.values()))

    def test_request_matches_production_openai_settings(self):
        calls = []
        provider = OpenAIVLM.__new__(OpenAIVLM)
        provider.model_name = MODEL
        provider._logprobs_supported = None
        provider._top_logprobs_supported = None
        provider._top_logprob_error = None
        response = SimpleNamespace(model=MODEL, choices=[SimpleNamespace(message=SimpleNamespace(content="A"), logprobs=None)])
        provider.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: (calls.append(kw) or response))))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"image.png"
            path.write_bytes(b"same-image")
            provider.associate(path, "same prompt")
        self.assertEqual(calls[0], request_arguments(b"same-image", "same prompt"))

    def test_exact_multitoken_likelihood_excludes_separable_formatting(self):
        parts = [("\n", -.9), ('"', -.6), ("object", -.01), ("_", -.02), ("003", -.3), ('"', -.7), (".\n", -.8)]
        result = extract_decision(''.join(t for t,p in parts), [{"token":t,"logprob":p} for t,p in parts], {"object_003":"object_003", "none":None})
        self.assertEqual(result["predicted_object_id"], "object_003")
        self.assertEqual(result["decision_token_count"], 3)
        self.assertAlmostEqual(result["raw_log_probability"], -.33)
        self.assertAlmostEqual(result["raw_sequence_likelihood"], math.exp(-.33))

    def test_inseparable_formatting_token_keeps_full_logprob(self):
        result = extract_decision(" A.", [{"token":" A.","logprob":-.2}], {"A":"object_001"})
        self.assertAlmostEqual(result["raw_sequence_likelihood"], math.exp(-.2))

    def test_none_and_missing_probability_are_distinct(self):
        result = extract_decision("none", [{"token":"none","logprob":-.1}], {"none":None})
        self.assertEqual(result["choice"], "none")
        self.assertIsNone(result["predicted_object_id"])
        missing = extract_decision("A", [{"token":"A"}], {"A":"object_001"})
        self.assertIsNone(missing["raw_sequence_likelihood"])

    def test_invalid_prefix_does_not_become_valid_object(self):
        result = extract_decision("object_0010", [{"token":"object_0010","logprob":-.1}], {"object_001":"object_001"})
        self.assertIsNone(result["choice"])

    def test_paired_accuracy_counts_both_directions(self):
        pairs = [(dict(correct=a,predicted_object_id="object_001"),
                  dict(correct=b,predicted_object_id="object_002"))
                 for a,b in ((False,True),(False,True),(True,False),(True,True))]
        metrics=paired_accuracy(pairs)
        self.assertEqual(metrics["compact_wrong_id_correct"],2)
        self.assertEqual(metrics["compact_correct_id_wrong"],1)
        self.assertAlmostEqual(metrics["accuracy_difference_id_minus_compact"],.25)

    def test_pair_audit_rejects_changed_image(self):
        row={"sample_id":"s1","condition":"compact","decision_tokens":[{"logprob":0}],
             "raw_sequence_likelihood":1.,"decision_token_count":1,"correct":True,
             "choice":"A","predicted_object_id":"object_001","ground_truth_object_id":"object_001",
             "candidate_map":{"A":"object_001"},"image_sha256":"image1",
             "localization_sha256":"localization1","target_description":"milk",
             "partition":"calibration","model":MODEL}
        other={**row,"condition":"object_id","choice":"object_001",
               "candidate_map":{"object_001":"object_001"},"image_sha256":"image2"}
        with self.assertRaisesRegex(ValueError,"image_sha256 mismatch"):
            validate_pairs([row,other])


if __name__ == "__main__":
    unittest.main()
