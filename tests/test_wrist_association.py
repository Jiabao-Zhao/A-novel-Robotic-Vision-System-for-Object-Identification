"""Evaluation plumbing tests; real DINOv2 / MuJoCo results come from the runner."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import cv2
import numpy as np

import main
from scripts.run_wrist_association import (TARGETS, PLACEMENTS, configure_scene,
                                           evaluate_rankings, save_identity_image)


class WristAssociationTests(unittest.TestCase):
    def test_scene_has_exactly_requested_parts_and_retains_robot(self):
        self.assertEqual(len(TARGETS), 6)
        self.assertEqual(set(TARGETS), set(PLACEMENTS))
        worldbody = ET.fromstring('<worldbody><body name="nist_board"/><body name="nist_fixture_0"/>'
                                 '<body name="robot0_base"/><body name="nist_part_KET16_Square_16mm"/></worldbody>')
        configure_scene(SimpleNamespace(worldbody=worldbody))
        self.assertEqual([body.get("name") for body in worldbody],
                         ["robot0_base", "nist_part_KET16_Square_16mm"])
        quat = np.fromstring(worldbody[-1].get("quat"), sep=" ")
        self.assertAlmostEqual(np.linalg.norm(quat), 1.)

    def test_evaluation_uses_audited_identity_not_candidate_order(self):
        results = [
            {"cad_id": "red_block", "target_description": "red block",
             "predictions": {"visual": "object_009", "geometry": "object_001", "fused": "object_009"}},
            {"cad_id": "blue_block", "target_description": "blue block",
             "predictions": {"visual": "object_009", "geometry": "object_001", "fused": "object_009"}},
        ]
        audit = [{"object_id": "object_001", "simulator_instance": "red_block", "status": "matched"},
                 {"object_id": "object_009", "simulator_instance": "blue_block", "status": "matched"}]
        evaluation = evaluate_rankings(results, audit)
        self.assertEqual(evaluation["per_target"][0]["expected_object_id"], "object_001")
        self.assertEqual(evaluation["accuracy"]["fused"], {"correct": 1, "total": 2, "fraction": .5})
        self.assertFalse(evaluation["per_target"][0]["correct"]["visual"])
        # Missed localization / missing prediction must never become a correct match.
        results[0]["predictions"] = dict.fromkeys(("visual", "geometry", "fused"))
        audit[0]["status"] = "ambiguous_possible_split"
        evaluation = evaluate_rankings(results, audit)
        self.assertEqual(evaluation["localized_target_count"], 1)
        self.assertFalse(any(evaluation["per_target"][0]["correct"].values()))

    def test_custom_library_preserves_known_simulation_cad(self):
        def associate(model, payload, rgb, plane, **kwargs):
            return {"target_description": kwargs["target_description"], "cad_id": model.cad_id,
                    "cad_path": model.file_path, "selected_object_id": "object_002",
                    "final_object_id": "object_002", "resolution": "ranking_only"}

        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "cad_library.json"
            library.write_text(json.dumps([{"cad_id": "pulley", "cad_name": "pulley", "description": "pulley",
                                            "file_path": "simulation/pulley.stl", "category": "simulation",
                                            "base_color_rgb": [.28, .30, .32]}]))
            with patch.object(main, "associate_cad_to_candidates", side_effect=associate), \
                    patch.object(main.CADRetrieval, "retrieve") as text_retrieval:
                results, fixed, _ = main.associate_targets_with_cad(
                    ["pulley"], {}, "rgb.png", output_dir=directory,
                    known_cad_ids={"pulley": "pulley"}, library_path=library)
            text_retrieval.assert_not_called()
            self.assertEqual(results[0]["cad_id"], "pulley")
            self.assertEqual(fixed["pulley"][0].file_path, "simulation/pulley.stl")
            self.assertEqual(fixed["pulley"][0].base_color_rgb, (.28, .30, .32))

    def test_identity_overlay_accepts_existing_roi_dictionary(self):
        with tempfile.TemporaryDirectory() as directory:
            rgb = Path(directory) / "rgb.png"
            result = Path(directory) / "identities.png"
            cv2.imwrite(str(rgb), np.zeros((120, 200, 3), dtype=np.uint8))
            audit = [{"object_id": "object_001", "simulator_instance": "pulley",
                      "roi": {"x1": 40, "y1": 40, "x2": 90, "y2": 90}}]
            save_identity_image(rgb, audit, result)
            self.assertGreater(cv2.imread(str(result)).sum(), 0)


if __name__ == "__main__":
    unittest.main()
