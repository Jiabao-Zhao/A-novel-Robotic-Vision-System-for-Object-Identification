import unittest
from unittest.mock import patch

import numpy as np
import open3d as o3d
import torch

import cad_object_association as association
from scripts.run_wrist_input_ablation import masked_crop


class InputAblationTests(unittest.TestCase):
    def test_saved_canvas_and_original_crop_produce_identical_encoder_tensors(self):
        class CaptureEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
                self.batches = []

            def forward(self, batch):
                self.batches.append(batch.clone())
                return batch.mean(dim=(2, 3))

        encoder = CaptureEncoder().eval()
        rng = np.random.default_rng(0)
        crops = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
                 for h, w in ((57, 54), (42, 29), (20, 80), (80, 20), (224, 224))]
        canvases = [association.letterbox_rgb(crop) for crop in crops]
        with patch.object(association, "load_dino_encoder", return_value=encoder):
            association.extract_dino_features(crops)
            association.extract_dino_features(canvases)
        torch.testing.assert_close(encoder.batches[0], encoder.batches[1], rtol=0, atol=0)
        self.assertEqual(canvases[2].shape, (224, 224, 3))
        self.assertTrue(np.all(canvases[2][:84] == 255))
        self.assertTrue(np.all(canvases[2][140:] == 255))

    def test_mask_keeps_projected_rgb_only_preserves_holes_and_tightly_crops(self):
        rgb = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector([
            [2, 3, 1], [4, 5, 1], [4, 6, 2], [0, 0, -1],
        ]))
        crop, mask, details = masked_crop(rgb, cloud, {
            "fx": 1, "fy": 1, "cx": 0, "cy": 0, "width": 8, "height": 8,
        })
        self.assertEqual(details["crop_bbox_2d_xyxy"], [2, 3, 4, 5])
        self.assertEqual(details["projected_point_count"], 3)
        self.assertEqual(details["mask_pixel_count"], 2)
        self.assertEqual(crop.shape, (3, 3, 3))
        np.testing.assert_array_equal(crop[0, 0], rgb[3, 2])
        np.testing.assert_array_equal(crop[2, 2], rgb[5, 4])
        self.assertTrue(np.all(crop[1] == 255))
        self.assertEqual(int(mask.sum()), 2)


if __name__ == "__main__":
    unittest.main()
