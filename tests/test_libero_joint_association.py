import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from openai.types.chat import ChatCompletion

from simulation.libero_assumed_human import assumed_human_selection
from simulation.libero_joint_association import (
    ROLE, associate_instruction_from_localization, joint_inferences,
)


INSTRUCTION = "pick up the milk and place it in the basket"
MAPPING = {"A": "object_001", "B": "object_002", "C": "object_003"}


def response(tokens):
    return {
        "id": "test", "created": 0, "model": "gpt-4.1-mini-2025-04-14",
        "object": "chat.completion",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "".join(t for t, _ in tokens)},
                     "logprobs": {"content": [{"token": t, "logprob": p, "top_logprobs": []}
                                              for t, p in tokens]}}],
    }


class JointAssociationTests(unittest.TestCase):
    def test_scores_only_original_decision_letters_and_accepts_copied_article(self):
        raw = response([("B", -.1), (": the milk", -4.), ("\n", -.3),
                        ("A", -.2), (": the basket", -6.)])
        rows, parsed = joint_inferences(raw, INSTRUCTION, MAPPING, ["milk", "basket"])
        self.assertAlmostEqual(rows[0]["association_score"], math.exp(-.1))
        self.assertAlmostEqual(rows[1]["association_score"], math.exp(-.2))
        self.assertEqual([r["vlm_object_id"] for r in rows], ["object_002", "object_001"])
        self.assertLess(parsed["full_response_likelihood"], rows[0]["association_score"])

    def test_missing_duplicate_and_invalid_outputs_defer_without_fabricated_scores(self):
        for tokens in ([('B: milk', 0.)],
                       [('B: milk\nC: the milk\nA: basket', 0.)],
                       [('Z: milk\nA: basket', 0.)]):
            rows, _ = joint_inferences(response(tokens), INSTRUCTION, MAPPING, ["milk", "basket"])
            self.assertTrue(all(r["association_score"] is None for r in rows))
        rows, _ = joint_inferences(response([('B', .00001), (': milk\nA', 0.), (': basket', 0.)]),
                                   INSTRUCTION, MAPPING, ["milk", "basket"])
        self.assertIsNone(rows[0]["association_score"])

    def run_association(self, tokens, resolver):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        image_path = root / "image.png"
        image_path.write_bytes(b"mock image bytes")
        localization = root / "localization.json"
        localization.write_text(json.dumps({"objects": [
            {"object_id": value, "roi": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}}
            for value in MAPPING.values()
        ]}))
        with patch("simulation.libero_joint_association.OpenAI") as client:
            client.return_value.chat.completions.create.return_value = ChatCompletion.model_validate(response(tokens))
            path = associate_instruction_from_localization(
                INSTRUCTION, ["milk", "basket"], image_path=image_path,
                localization_path=localization, output_path=root / "vlm_result.json",
                threshold=.99, human_resolver=resolver, assumed_human=True,
            )
            args = client.return_value.chat.completions.create.call_args.kwargs
        return json.loads(path.read_text()), args, root

    def test_prompt_matches_joint_protocol_and_assumed_correction_is_explicit(self):
        resolver = MagicMock(return_value="object_003")
        payload, args, root = self.run_association(
            [("B", -.5), (": the milk\n", -3.), ("A", 0.), (": the basket", -3.)], resolver,
        )
        self.assertEqual(args["messages"][0]["content"], ROLE + "\n\nTask instruction:\n" + INSTRUCTION)
        self.assertEqual(len(args["messages"][1]["content"]), 1)
        self.assertEqual(args["messages"][1]["content"][0]["type"], "image_url")
        self.assertNotIn("centroid", json.dumps(args["messages"]))
        self.assertNotIn("object_003", json.dumps(args["messages"]))
        milk, basket = payload["associations"]
        self.assertEqual(milk["resolution"], "assumed_human_corrected")
        self.assertEqual(milk["final_object_id"], "object_003")
        self.assertEqual(milk["vlm_object_id"], "object_002")
        self.assertAlmostEqual(milk["association_score"], math.exp(-.5))
        self.assertNotIn("human_response_time_s", milk["diagnostics"])
        self.assertEqual(basket["resolution"], "vlm_accepted")
        self.assertTrue(payload["simulator_identity_used_for_assumed_human"])
        resolver.assert_called_once()
        self.assertTrue((root / "vlm_provider_response.json").is_file())

    def test_accepted_predictions_are_not_corrected_by_the_assumed_human(self):
        resolver = MagicMock(side_effect=AssertionError("Accepted decisions cannot use oracle identity"))
        payload, _, _ = self.run_association(
            [("B", 0.), (": milk\n", -5.), ("A", 0.), (": basket", -5.)], resolver,
        )
        resolver.assert_not_called()
        self.assertFalse(payload["simulator_identity_used_for_assumed_human"])
        self.assertEqual(payload["associations"][0]["final_object_id"], "object_002")

    def test_assumed_human_returns_a_candidate_id_and_rejects_ambiguous_localization(self):
        env = MagicMock()
        env.env.env.objects = [SimpleNamespace(name="milk_1", category_name="milk", root_body="milk")]
        env.env.env.fixtures = []
        env.sim.model.body_name2id.return_value = 0
        env.sim.data.body_xpos = np.array([[0., 0., .1]])
        request = {"target_description": "milk", "vlm_object_id": "object_001",
                   "association_score": .5, "candidate_object_ids": ["object_001", "object_002"]}
        with tempfile.TemporaryDirectory() as directory, patch(
            "simulation.libero_assumed_human.match_candidates"
        ) as match:
            match.return_value = [{"status": "matched", "semantic_identity": "milk", "object_id": "object_002"}]
            self.assertEqual(assumed_human_selection(request, env, {}, np.eye(4), directory), "object_002")
            evidence = json.loads((Path(directory) / "milk.json").read_text())
            self.assertFalse(evidence["actual_human_response"])
            match.return_value = [{"status": "ambiguous_near_tie", "semantic_identity": None, "object_id": "object_002"}]
            with self.assertRaisesRegex(RuntimeError, "missing or ambiguous"):
                assumed_human_selection(request, env, {}, np.eye(4), directory)


if __name__ == "__main__":
    unittest.main()
