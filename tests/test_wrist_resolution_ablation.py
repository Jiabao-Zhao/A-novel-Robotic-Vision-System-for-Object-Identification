import unittest

import numpy as np

from scripts.run_wrist_resolution_ablation import observation_input, scaled_roi


class ResolutionCropTests(unittest.TestCase):
    def test_inclusive_roi_keeps_last_pixel_and_scales_size(self):
        self.assertEqual(scaled_roi([370, 409, 398, 450], 5), [1850, 2045, 1994, 2254])
        rgb = np.zeros((768, 768, 3), dtype=np.uint8)
        rgb[409:451, 370:399] = [10, 80, 240]
        high = np.repeat(np.repeat(rgb, 5, axis=0), 5, axis=1)
        _, crop, box = observation_input(high, [370, 409, 398, 450])
        self.assertEqual(crop.shape, (210, 145, 3))
        self.assertEqual(box, [1850, 2045, 1994, 2254])
        np.testing.assert_array_equal(crop, np.broadcast_to([10, 80, 240], crop.shape))

    def test_scaled_mask_preserves_holes_and_original_support(self):
        rgb = np.full((768, 768, 3), [10, 80, 240], dtype=np.uint8)
        mask = np.zeros((768, 768), dtype=bool)
        mask[20:25, 30:35] = True
        mask[22, 32] = False
        _, low, low_box = observation_input(rgb, [0, 0, 767, 767], mask)
        high_rgb = np.repeat(np.repeat(rgb, 5, axis=0), 5, axis=1)
        _, high, high_box = observation_input(high_rgb, [0, 0, 767, 767], mask)
        self.assertEqual(high_box, scaled_roi(low_box, 5))
        np.testing.assert_array_equal(high, np.repeat(np.repeat(low, 5, axis=0), 5, axis=1))
        self.assertEqual(int(np.all(high == 255, axis=-1).sum()), 25)


if __name__ == "__main__":
    unittest.main()
