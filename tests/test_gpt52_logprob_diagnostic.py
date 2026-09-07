"""Raw-body/SDK comparison tests; no API calls or hardware."""

import copy
import json
import unittest

from scripts.diagnose_gpt52_logprobs import compare_response


class RawLogprobTests(unittest.TestCase):
    def response(self, logprob):
        return {"model": "gpt-5.2-2025-12-11", "choices": [{
            "message": {"content": "B"},
            "logprobs": {"content": [{"token": "B", "logprob": logprob,
                                     "top_logprobs": [{"token": "N", "logprob": -8.0}]}]},
        }]}

    def test_positive_value_exists_before_sdk_and_is_not_clamped(self):
        raw = self.response(0.00000762939453125)
        before = copy.deepcopy(raw)
        result = compare_response(json.dumps(raw).encode(), raw)
        self.assertTrue(result["all_logprob_fields_match_sdk"])
        self.assertEqual(result["positive_chosen_logprobs"], 1)
        self.assertEqual(result["positive_values"][0]["raw_json_decimal"], "0.00000762939453125")
        self.assertEqual(raw, before)

    def test_detects_changed_sdk_value(self):
        raw = self.response(0.00000762939453125)
        parsed = self.response(0.0)
        result = compare_response(json.dumps(raw).encode(), parsed)
        self.assertFalse(result["all_logprob_fields_match_sdk"])
        self.assertEqual(len(result["mismatched_fields"]), 1)

    def test_checks_alternatives_too(self):
        raw = self.response(-0.1)
        raw["choices"][0]["logprobs"]["content"][0]["top_logprobs"][0]["logprob"] = 0.00001
        result = compare_response(json.dumps(raw).encode(), raw)
        self.assertEqual(result["positive_chosen_logprobs"], 0)
        self.assertEqual(result["positive_alternative_logprobs"], 1)

    def test_negative_zero_and_missing_values_are_not_positive(self):
        for value in (-2.3, -9999.0, -0.0, 0.0, None):
            raw = self.response(value)
            result = compare_response(json.dumps(raw).encode(), raw)
            self.assertTrue(result["all_logprob_fields_match_sdk"])
            self.assertEqual(result["positive_chosen_logprobs"], 0)

    def test_missing_sdk_tokens_are_detected(self):
        raw = self.response(-0.1)
        parsed = self.response(-0.1)
        parsed["choices"][0]["logprobs"] = None
        self.assertFalse(compare_response(json.dumps(raw).encode(), parsed)["all_logprob_fields_match_sdk"])

    def test_unavailable_logprobs_do_not_claim_a_verified_match(self):
        raw = self.response(-0.1)
        raw["choices"][0]["logprobs"] = None
        result = compare_response(json.dumps(raw).encode(), raw)
        self.assertFalse(result["all_logprob_fields_match_sdk"])
        self.assertEqual(result["chosen_logprob_fields"], 0)
