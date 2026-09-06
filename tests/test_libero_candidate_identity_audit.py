import unittest

import numpy as np

from scripts.audit_libero_candidate_identities import match_candidates, semantic_prediction, projected_centroid_in_roi


def candidate(object_id, xyz):
    return {"object_id": object_id, "centroid_3d_m": xyz,
            "roi": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}}


def simulator_object(instance, semantic_identity, xyz):
    return {"instance": instance, "semantic_identity": semantic_identity,
            "position_world_m": xyz}


class CandidateIdentityAuditTests(unittest.TestCase):
    def test_matches_all_candidates_including_basket_without_using_id_order(self):
        objects = [simulator_object("milk_1", "milk", [.2, 0, 0]),
                   simulator_object("basket_1", "basket", [0, 0, 0])]
        localization = {"objects": [candidate("object_001", [0, 0, .1]),
                                    candidate("object_002", [.2, 0, .1])]}
        result = match_candidates(localization, np.eye(4), objects)
        self.assertEqual([row["semantic_identity"] for row in result], ["basket", "milk"])

    def test_camera_to_world_transform_is_applied(self):
        transform = np.eye(4)
        transform[0, 3] = .3
        result = match_candidates({"objects": [candidate("object_001", [0, 0, .1])]},
                                  transform, [simulator_object("milk_1", "milk", [.3, 0, 0])])
        self.assertEqual(result[0]["semantic_identity"], "milk")
        self.assertAlmostEqual(result[0]["nearest_xy_distance_m"], 0)

    def test_distant_candidate_is_unresolved_not_forced(self):
        result = match_candidates({"objects": [candidate("object_001", [.5, 0, .1])]},
                                  np.eye(4), [simulator_object("milk_1", "milk", [0, 0, 0])])
        self.assertIsNone(result[0]["semantic_identity"])
        self.assertEqual(result[0]["status"], "unmatched_distance")

    def test_two_clusters_for_one_object_are_flagged_as_possible_split(self):
        result = match_candidates({"objects": [candidate("object_001", [0, 0, .1]),
                                               candidate("object_002", [.02, 0, .1])]},
                                  np.eye(4), [simulator_object("milk_1", "milk", [0, 0, 0])])
        self.assertTrue(all(row["semantic_identity"] is None for row in result))
        self.assertTrue(all(row["status"] == "ambiguous_possible_split" for row in result))

    def test_near_tied_object_bodies_are_unresolved(self):
        result = match_candidates({"objects": [candidate("object_001", [0, 0, .1])]},
                                  np.eye(4), [simulator_object("a", "milk", [-.002, 0, 0]),
                                             simulator_object("b", "butter", [.002, 0, 0])])
        self.assertEqual(result[0]["status"], "ambiguous_near_tie")
        self.assertIsNone(result[0]["semantic_identity"])

    def test_missing_catalog_and_duplicate_ids_fail(self):
        obj = candidate("object_001", [0, 0, .1])
        with self.assertRaises(ValueError):
            match_candidates({"objects": [obj]}, np.eye(4), [])
        with self.assertRaises(ValueError):
            match_candidates({"objects": [obj, obj]}, np.eye(4), [simulator_object("a", "milk", [0, 0, 0])])

    def test_saved_prediction_is_joined_by_id_not_target_text(self):
        audit = {"candidates": [{"object_id": "object_003", "semantic_identity": "tomato sauce"},
                                {"object_id": "object_006", "semantic_identity": "chocolate pudding"}]}
        prediction = {"predicted_object_id": "object_006", "target_description": "tomato sauce"}
        self.assertEqual(semantic_prediction(prediction, audit), "chocolate pudding")
        self.assertEqual(prediction["predicted_object_id"], "object_006")

    def test_none_is_not_an_unresolved_mapping(self):
        self.assertEqual(semantic_prediction({"predicted_object_id": None}, {"candidates": []}), "none")

    def test_unknown_candidate_is_not_guessed(self):
        self.assertEqual(semantic_prediction({"predicted_object_id": "object_999"}, {"candidates": []}), "invalid_candidate_id")

    def test_unresolved_identity_stays_unresolved(self):
        audit = {"candidates": [{"object_id": "object_001", "semantic_identity": None}]}
        self.assertEqual(semantic_prediction({"predicted_object_id": "object_001"}, audit), "unresolved_identity")

    def test_projected_centroid_matches_its_own_roi(self):
        calibration = {"fx": 100, "fy": 100, "cx": 5, "cy": 5}
        self.assertTrue(projected_centroid_in_roi(candidate("object_001", [0, 0, 1]), calibration))
        self.assertFalse(projected_centroid_in_roi(candidate("object_001", [.2, 0, 1]), calibration))
        self.assertFalse(projected_centroid_in_roi(candidate("object_001", [0, 0, -1]), calibration))


if __name__ == "__main__":
    unittest.main()
