import json
from pathlib import Path
import types

import numpy as np
import pytest
import torch
from torch import nn

from visual_ablation import CADVisualAdapter, aggregate_cached_visuals, _sha, _digest


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.calls = []
        self.eval()

    def get_intermediate_layers(self, image, **kwargs):
        self.calls.append((tuple(image.shape), kwargs, image.detach().clone()))
        side = image.shape[-1] // 14
        return [image.mean().expand(1, 1536, side, side).contiguous()]


class Dino:
    def __init__(self):
        self.torch, self.device = torch, torch.device("cpu")
        self.model, self.layer_index, self.input_size = Model(), 18, 224
        self.calls = 0
        self.provenance = {"model": "dinov2_vitg14", "layer_index": 18, "input_size": 224,
                           "normalization": "ImageNet_mean_std", "precision": "float32",
                           "checkpoint_sha256": "1" * 64, "tf32_matmul": False, "tf32_cudnn": False}

    def extract(self, image):
        self.calls += 1
        return np.full((16, 16, 1536), float(image[0, 0, 0]), dtype=np.float32)


def sample_fixture(fmap, uv, image_size):
    # Deliberately sensitive to the coordinate frame supplied by the aggregator.
    return np.repeat((uv[:, :1] / image_size[1]).astype(np.float32), 1536, axis=1)


def make_geometry(tmp_path, uv_x=100.0):
    folder = tmp_path / "geometry"
    (folder / "views").mkdir(parents=True)
    identity = {"schema": "CPU-synthetic-geometry"}
    geom_id = _digest(identity)
    np.savez_compressed(folder / "geometry.npz", identity=np.array(geom_id),
        points_m=np.array([[uv_x, 20., 1.], [40., 50., 1.]]),
        visibility=np.ones((2, 2), bool), R=np.repeat(np.eye(3)[None], 2, axis=0),
        t_m=np.zeros((2, 3)), K=np.eye(3))
    views = {}
    for view in range(2):
        path = folder / "views" / f"{view:03d}.npz"
        np.savez_compressed(path, identity=np.array(geom_id),
                            rgb=np.zeros((480, 480, 3), np.uint8))
        views[str(view)] = {"sha256": _sha(path)}
    complete = {"identity": identity, "files": {"geometry.npz": _sha(folder / "geometry.npz")},
                "view_file_hashes": views}
    (folder / "complete.json").write_text(json.dumps(complete))
    return folder, complete


def aggregate(folder, out, adapter, **kwargs):
    return aggregate_cached_visuals(folder, out, adapter, min_views=2,
                                   expected_views=2, minimum_free_gb=0,
                                   sampler=sample_fixture, **kwargs)


def test_baseline_delegates_without_changing_dino():
    dino = Dino()
    adapter = CADVisualAdapter(dino, "baseline224")
    image = np.zeros((480, 480, 3), np.uint8)
    image[0, 0, 0] = 31
    result = adapter.extract(image)
    assert result.shape == (16, 16, 1536) and result[0, 0, 0] == 31
    assert dino.calls == 1 and not dino.model.calls
    assert dino.input_size == 224 and dino.layer_index == 18


def test_476_crops_bottom_right_remainders_and_reuses_original_layer_norm():
    dino = Dino()
    adapter = CADVisualAdapter(dino, "cad476")
    image = np.zeros((480, 480, 3), np.uint8)
    image[476:, :, :] = 255
    image[:, 476:, :] = 255
    result = adapter.extract(image)
    shape, args, normalized = dino.model.calls[0]
    assert shape == (1, 3, 476, 476) and result.shape == (34, 34, 1536)
    assert args == {"n": [18], "reshape": True, "return_class_token": False, "norm": True}
    expected = -torch.tensor([.485, .456, .406]) / torch.tensor([.229, .224, .225])
    torch.testing.assert_close(normalized[0, :, 0, 0], expected)
    torch.testing.assert_close(normalized[0, :, -1, -1], expected)
    assert dino.calls == 0 and dino.input_size == 224
    assert adapter.sampling_image_hw == (476, 476)
    assert not adapter.provenance["public_repository_replication"]


@pytest.mark.parametrize("bad", [np.zeros((224, 224, 3), np.uint8), np.zeros((480, 480, 3), np.float32)])
def test_rejects_changed_render_inputs(bad):
    with pytest.raises(ValueError, match="cached480x480"):
        CADVisualAdapter(Dino(), "cad476").extract(bad)


def test_rejects_layer_change_or_trainable_network():
    dino = Dino(); dino.layer_index = 30
    with pytest.raises(ValueError, match="block18"):
        CADVisualAdapter(dino)
    dino = Dino(); dino.model.train()
    with pytest.raises(ValueError, match="evaluation"):
        CADVisualAdapter(dino)


def test_aggregation_and_resume_retain_all_original_indices(tmp_path):
    folder, _ = make_geometry(tmp_path)
    dino = Dino(); adapter = CADVisualAdapter(dino)
    out = tmp_path / "visual"
    result = aggregate(folder, out, adapter)
    np.testing.assert_array_equal(result["original_indices"], [0, 1])
    np.testing.assert_array_equal(result["view_counts"], [2, 2])
    np.testing.assert_allclose(result["visual_mean"][:, 0], np.array([100, 40]) / 480)
    assert dino.calls == 2
    again = aggregate(folder, out, adapter)
    np.testing.assert_array_equal(again["visual_mean"], result["visual_mean"])
    assert dino.calls == 2


def test_changed_mode_or_source_cannot_reuse_cache(tmp_path):
    folder, _ = make_geometry(tmp_path)
    out = tmp_path / "visual"
    aggregate(folder, out, CADVisualAdapter(Dino()))
    with pytest.raises(ValueError, match="Incompatible"):
        aggregate(folder, out, CADVisualAdapter(Dino(), "cad476"))
    (folder / "views" / "000.npz").write_bytes(b"corrupt original render")
    with pytest.raises(ValueError, match="Original rendered view checksum"):
        aggregate(folder, out, CADVisualAdapter(Dino()))


def test_476_uses_cropped_coordinate_extent(tmp_path):
    folder, _ = make_geometry(tmp_path)
    result = aggregate(folder, tmp_path / "new476", CADVisualAdapter(Dino(), "cad476"))
    np.testing.assert_allclose(result["visual_mean"][:, 0], np.array([100, 40]) / 476)


def test_lost_crop_visibility_is_explicit_not_silent_row_removal(tmp_path):
    folder, _ = make_geometry(tmp_path, uv_x=477.0)
    with pytest.raises(ValueError, match="required visibility"):
        aggregate(folder, tmp_path / "new476", CADVisualAdapter(Dino(), "cad476"))
    assert not (tmp_path / "new476" / "complete.json").exists()


def test_reuses_hash_bound_baseline_maps_without_forwarding(tmp_path):
    folder, geometry = make_geometry(tmp_path)
    old = tmp_path / "old_features"; (old / "visual_maps").mkdir(parents=True)
    dino = Dino()
    identity = {"geometry_identity": _digest(geometry["identity"]),
                "geometry_sha256": geometry["files"]["geometry.npz"],
                "backends": {"dino": dino.provenance}}
    (old / "identity.json").write_text(json.dumps(identity))
    progress = {"maps": {}}
    for view in range(2):
        path = old / "visual_maps" / f"{view:03d}.npz"
        np.savez_compressed(path, identity=np.array(_digest(identity)),
                            map=np.zeros((16, 16, 1536), np.float32))
        progress["maps"][str(view)] = {"sha256": _sha(path)}
    (old / "progress.json").write_text(json.dumps(progress))
    result = aggregate(folder, tmp_path / "new224", CADVisualAdapter(dino), baseline_map_folder=old)
    assert dino.calls == 0 and result["manifest"]["map_storage"] == "original224 references"
    assert not (tmp_path / "new224" / "visual_maps").exists()
