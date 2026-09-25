from unittest.mock import patch

import numpy as np
import pytest
import torch

from scripts.cad_descriptors import encode, letterbox_rgb, letterbox_mask, patch_occupancy, prepare_masked_input


def test_masked_crop_and_rgb_mask_alignment():
    rgb = np.full((70, 110, 3), [23, 45, 67], dtype=np.uint8)
    mask = np.zeros((70, 110), dtype=bool)
    mask[10:40, 20:80] = True
    mask[15:20, 25:30] = False
    prepared, crop, bounds = prepare_masked_input(rgb, mask)
    assert bounds == [20, 10, 79, 39]
    assert crop.shape == (30, 60, 3)
    assert prepared.shape == (224, 224, 3)
    np.testing.assert_array_equal(crop[5:10, 5:10], np.full((5, 5, 3), 255))
    occupancy = patch_occupancy(letterbox_mask(mask[10:40, 20:80]))
    assert occupancy.shape == (16, 16)
    assert not occupancy[:4].any() and not occupancy[12:].any()


def test_empty_mask_is_unavailable():
    with pytest.raises(ValueError, match='empty mask'):
        prepare_masked_input(np.zeros((20, 20, 3), dtype=np.uint8), np.zeros((20, 20), dtype=bool))


def test_prepared_image_is_unchanged():
    image = np.random.default_rng(9).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    np.testing.assert_array_equal(letterbox_rgb(image), image)


def test_patch_occupancy_is_fraction_not_binary():
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[:7, :14] = 1
    occupancy = patch_occupancy(mask)
    assert occupancy[0, 0] == .5 and occupancy.sum() == .5


def test_encode_contract_without_download(tmp_path):
    class Model(torch.nn.Module):
        patch_size = 14
        embed_dim = 4

        def forward_features(self, batch):
            values = torch.tensor([1., 2., 3., 4.], device=batch.device)
            return {'x_norm_clstoken': values.expand(len(batch), 4),
                    'x_norm_patchtokens': values.expand(len(batch), 256, 4)}

    with patch('torch.hub.load', return_value=Model()), patch('torch.hub.set_dir'):
        cls, tokens, runtime = encode('test_model', 4, [np.zeros((224, 224, 3), dtype=np.uint8)], {0}, tmp_path)
    np.testing.assert_allclose(np.linalg.norm(cls, axis=-1), 1)
    assert tokens[0].shape == (16, 16, 4)
    assert runtime['device'] == 'cpu'
    saved = np.load(tmp_path/'features.npz')
    np.testing.assert_array_equal(saved['cls_features'], cls)
