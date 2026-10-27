"""CPU-only checks of official native methods and the scoped loading adapters."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

LOCAL = Path(__file__).resolve().parent
sys.path.insert(0, str(LOCAL.parent))
import common
inputs = common.setup()
from compat import import_native, deformable_attention_fallback

METHOD = sys.argv.pop(1)
NATIVE = import_native(METHOD, common)

import numpy as np
import torch


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward_features(self, images):
        tokens = torch.arange(1, 256*8+1, dtype=torch.float32).reshape(1, 256, 8)
        tokens = tokens.repeat(len(images), 1, 1)
        return {"x_norm_patchtokens": tokens, "x_norm_clstoken": tokens.mean(1)}

    def forward(self, images, is_training=False):
        result = self.forward_features(images)
        return result if is_training else result["x_norm_clstoken"]


class NativeContract(unittest.TestCase):
    def test_manifest_has_no_scene_proposals_or_poses(self):
        self.assertEqual(len(inputs["scenes"]), 8)
        self.assertEqual(len(inputs["objects"]), 12)
        for scene in inputs["scenes"]:
            self.assertEqual(set(scene), {"scene_id", "rgb", "depth", "intrinsics"})

    def test_global_installer_is_blocked(self):
        if METHOD == "nids_basic":
            self.assertEqual(NATIVE.blocked_import_commands, ["python setup.py build develop --user"])
        else:
            self.assertEqual(NATIVE.blocked_import_commands, [])

    def test_shipped_deformable_operator_adapter(self):
        import groundingdino.models.GroundingDINO.ms_deform_attn as module
        value = torch.arange(24, dtype=torch.float32).reshape(1, 4, 2, 3)
        shapes = torch.tensor([[2, 2]], dtype=torch.long)
        locations = torch.tensor([.15, .3, .75, .6], dtype=torch.float32).reshape(1, 1, 1, 1, 2, 2).repeat(1, 2, 2, 1, 1, 1)
        weights = torch.full((1, 2, 2, 1, 2), .5)
        expected = module.multi_scale_deformable_attn_pytorch(value, shapes, locations, weights)
        mode = deformable_attention_fallback()
        if mode == "official_pytorch_operator_on_selected_device":
            actual = module.MultiScaleDeformableAttnFunction.apply(value, shapes, torch.tensor([0]), locations, weights, 64)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_native_query_features_keep_proposal_order(self):
        if METHOD == "nids_basic":
            descriptor = NATIVE.descriptor.CustomDINOv2("dinov2_vitl14", TinyBackbone(), "x_norm_clstoken", 224, 2, 640)
        else:
            descriptor = NATIVE.descriptor.CustomDINOv2("dinov2_vitl14", TinyBackbone(), 224, 2)
        image = np.full((256, 256, 3), 128, np.uint8)
        masks = torch.zeros((2, 256, 256))
        masks[0, 25:125, 35:180] = 1
        masks[1, 100:210, 90:220] = 1
        proposals = SimpleNamespace(boxes=torch.tensor([[35, 25, 180, 125], [90, 100, 220, 210]]), masks=masks)
        values = descriptor(image, proposals)
        self.assertEqual(len(values), 3 if METHOD == "nids_basic" else 2)
        self.assertTrue(all(len(value) == 2 and torch.isfinite(value).all() for value in values))
        if METHOD == "nids_basic":
            image = torch.zeros((2, 3, 224, 224))
            masks = torch.ones((2, 1, 224, 224))
            patch = descriptor.compute_masked_patch_feature(image, masks)
            self.assertEqual(tuple(patch.shape), (2, 256, 8))

    def test_native_filters_and_nms_preserve_proposal_ids(self):
        from omegaconf import OmegaConf
        masks = torch.zeros((3, 200, 200))
        masks[:, 30:70, 30:70] = 1
        data = dict(boxes=torch.tensor([[25, 25, 75, 75], [26, 26, 76, 76], [25, 25, 75, 75]]),
                    masks=masks, proposal_ids=torch.tensor([11, 22, 33]))
        det = NATIVE.utils.Detections(data)
        det.remove_very_small_detections(OmegaConf.create(dict(min_box_size=.05, min_mask_size=.0003)))
        det.add_attribute("object_ids", torch.tensor([0, 0, 1]))
        det.add_attribute("scores", torch.tensor([.9, .8, .7]))
        if METHOD == "nids_basic":
            det.apply_nms_per_object_id(nms_thresh=.25)
        else:
            det.apply_nms_per_object_id(nms_threshold=.25)
        self.assertEqual(det.proposal_ids.tolist(), [11, 33])

    def test_native_template_index_source_is_recorded(self):
        metadata = common.read(common.HERE / "templates42/metadata.json")
        if METHOD == "nids_basic":
            from src.poses.utils import load_index_level_in_level2
            indices = load_index_level_in_level2(0, "all")
            np.testing.assert_array_equal(indices, metadata["index_templates"])
        else:
            from src.poses.utils import load_indices_level_in_level_mapping
            indices = load_indices_level_in_level_mapping(str(NATIVE.source / "datasets/predefined_poses/ico"))
            self.assertEqual(len(indices), 42)
            self.assertIn(301, indices)
            self.assertLess(len(set(indices.tolist())), 42)
            for obj in inputs["objects"]:
                for index in indices:
                    self.assertTrue((common.HERE / "templates42" / f'obj_{obj["object_id"]:06d}' / f'{index:06d}.png').is_file())

    def test_native_matcher_construction_and_identity_mapping(self):
        from run_native import build_matcher
        if METHOD == "nids_basic":
            descriptor = NATIVE.descriptor.CustomDINOv2("dinov2_vitl14", TinyBackbone(), "x_norm_clstoken", 224, 2, 640)
        else:
            descriptor = NATIVE.descriptor.CustomDINOv2("dinov2_vitl14", TinyBackbone(), 224, 2)
        model = build_matcher(METHOD, NATIVE, descriptor, SimpleNamespace(obj_ids=[1, 2]), LOCAL / "test_logs")
        refs = torch.tensor([[[1., 0.], [1., 0.]], [[0., 1.], [0., 1.]]])
        query = torch.tensor([[1., 0.], [0., 1.]])
        if METHOD == "nids_basic":
            model.ref_data = dict(descriptors=refs)
            selected, identities, scores, views = model.find_matched_proposals(query)
            self.assertEqual(selected.tolist(), [0, 1])
            self.assertEqual(identities.tolist(), [0, 1])
            self.assertEqual(views.tolist(), [0, 0])
            self.assertEqual(scores.tolist(), [1, 1])
        else:
            model.ref_data = dict(cls_descriptors=refs)
            semantic = model.compute_semantic_score(query)
            selected, identities, scores, distribution = model.choose_best_and_filter(semantic, .2, 100)
            self.assertEqual(identities.tolist(), [0, 1])
            self.assertEqual(tuple(distribution.shape), (2, 2))

    def test_native_appearance_scoring_with_small_chunks(self):
        patches = torch.zeros((1, 256, 4))
        patches[0, :120, 0] = 1
        references = patches[:, None].repeat(2, 3, 1, 1)
        if METHOD == "nids_basic":
            from src.model.detector import MaskedPatch_MatrixSimilarity
            score = MaskedPatch_MatrixSimilarity().compute_straight(patches, references[:1, 0])
            self.assertEqual(tuple(score.shape), (1,))
        else:
            from src.model.scoring import compute_appearance_similarity
            score = compute_appearance_similarity(patches, references, min_threshold=0,
                                                   cycle_threshold=5, chunk_size=1, obj_chunk_size=1)
            self.assertEqual(tuple(score.shape), (1, 2, 3))
        self.assertTrue(torch.isfinite(score).all())


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(NativeContract)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    common.save(LOCAL / f"{METHOD}_CPU_tests.json", dict(tests=result.testsRun, failures=len(result.failures),
        errors=len(result.errors), passed=result.wasSuccessful(), GPU_inference=False,
        source_sha256=common.digest(LOCAL / "test_native.py")))
    sys.exit(0 if result.wasSuccessful() else 1)
