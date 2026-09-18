import unittest

import numpy as np

from scripts.run_wrist_blenderproc_semantic import prepare_rgba


class BlenderTemplateInputTests(unittest.TestCase):
    def test_transparency_is_white_and_aspect_ratio_survives_tight_crop(self):
        rgba = np.zeros((12, 12, 4), np.uint8)
        rgba[4:8, 2:10, :3] = [255, 0, 0]
        rgba[4:8, 2:10, 3] = 255
        # Transparent hole must reveal white, not Blender's black RGB storage.
        rgba[5:7, 5:7, 3] = 0
        canvas, mask, bbox = prepare_rgba(rgba)
        self.assertEqual(tuple(bbox), (2, 4, 9, 7))
        self.assertEqual(mask.sum(), 28)
        np.testing.assert_array_equal(canvas[:56], 255)
        np.testing.assert_array_equal(canvas[168:], 255)
        np.testing.assert_array_equal(canvas[112, 112], [255, 255, 255])
        np.testing.assert_array_equal(canvas[112, 20], [255, 0, 0])

    def test_empty_or_clipped_templates_fail(self):
        for rgba in (np.zeros((12, 12, 4), np.uint8), np.full((12, 12, 4), 255, np.uint8)):
            with self.assertRaises(ValueError):
                prepare_rgba(rgba)


if __name__ == "__main__":
    unittest.main()
