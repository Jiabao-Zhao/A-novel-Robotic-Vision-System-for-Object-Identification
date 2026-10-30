"""Offline feature-contract tests; no model downloads or fake benchmark output."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from freezev2_repro.features import (
    FeatureDependencyError, FrozenDINOv2, FrozenGeDi, PCATransform,
    aggregate_view_features, freeze_model, fuse_features, microbatch_frozen_model, sample_feature_map,
)


def test_bilinear_feature_sampling_pixel_centers_and_border():
    fmap = np.array([[[0.0], [2.0]], [[4.0], [6.0]]], dtype=np.float32)
    uv = np.array([[0, 0], [3, 3], [1.5, 1.5], [0.5, 0.5]])
    np.testing.assert_allclose(sample_feature_map(fmap, uv, (4, 4))[:, 0], [0, 6, 3, 0])
    with pytest.raises(ValueError, match="outside"):
        sample_feature_map(fmap, [[-1, 0]], (4, 4))


def test_bilinear_mapping_matches_torch_upsample_at_integer_pixels():
    torch = pytest.importorskip("torch")
    fmap = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    tensor = torch.from_numpy(fmap).permute(2, 0, 1)[None]
    up = torch.nn.functional.interpolate(tensor, size=(7, 9), mode="bilinear", align_corners=False)
    uv = np.array([[0, 0], [2, 1], [4, 3], [8, 6]], dtype=np.float32)
    actual = sample_feature_map(fmap, uv, (7, 9))
    expected = up[0, :, uv[:, 1].astype(int), uv[:, 0].astype(int)].T.numpy()
    np.testing.assert_allclose(actual, expected, atol=2e-6)


def test_visible_only_aggregation_does_not_leak_invisible_values():
    features = np.array([[[1, 3], [np.nan, np.nan]], [[3, 5], [7, 9]]], dtype=float)
    visible = np.array([[True, False], [True, True]])
    mean, counts = aggregate_view_features(features, visible)
    np.testing.assert_array_equal(counts, [2, 1])
    np.testing.assert_allclose(mean, [[2, 4], [7, 9]])
    weighted, _ = aggregate_view_features(features, visible, weights=np.array([1, 3]))
    np.testing.assert_allclose(weighted, [[2.5, 4.5], [7, 9]])
    unseen, count = aggregate_view_features(np.full((1, 1, 2), np.nan), [[False]])
    assert np.isnan(unseen).all() and count[0] == 0


def test_pca_cad_fit_is_unchanged_by_observation_and_cache(tmp_path):
    rng = np.random.default_rng(13)
    cad, observation = rng.normal(size=(20, 6)), rng.normal(size=(4, 6)) + 15
    pca = PCATransform(3).fit(cad)
    mean, components = pca.mean_.copy(), pca.components_.copy()
    projected = pca.transform(observation)
    np.testing.assert_allclose(projected, (observation - mean) @ components.T, atol=1e-5)
    np.testing.assert_array_equal(pca.mean_, mean)
    np.testing.assert_array_equal(pca.components_, components)
    pca.save(tmp_path / "pca.npz")
    restored = PCATransform.load(tmp_path / "pca.npz")
    np.testing.assert_array_equal(restored.transform(observation), projected)
    with pytest.raises(ValueError, match="finite"):
        pca.transform([[np.nan] * 6])
    with pytest.raises(ValueError, match="requires"):
        PCATransform(64).fit(cad)


def test_equal_modality_l2_normalization_and_nonfinite_rejection():
    rng = np.random.default_rng(8)
    fused = fuse_features(rng.normal(size=(5, 64)), rng.normal(size=(5, 64)))
    assert fused.shape == (5, 128) and np.isfinite(fused).all()
    np.testing.assert_allclose(np.linalg.norm(fused[:, :64], axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(fused[:, 64:], axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(fused, axis=1), np.sqrt(2), atol=1e-6)
    with pytest.raises(ValueError, match="zero"):
        fuse_features(np.zeros((1, 64)), np.ones((1, 64)))


def test_model_freeze_sets_eval_and_all_gradients_false():
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(0.8))
    freeze_model(model)
    assert not model.training
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_real_adapter_offline_contract_without_loading_a_real_checkpoint(tmp_path, monkeypatch):
    """A tiny test double checks the adapter's mechanics, never pose accuracy."""
    torch = pytest.importorskip("torch")

    class AdapterProbe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
            self.seen = None

        def get_intermediate_layers(self, value, **kwargs):
            self.seen = (torch.is_inference_mode_enabled(), value.clone(), kwargs)
            return [torch.ones((1, 1536, 16, 16)) * self.scale]

    repo = tmp_path / "dinov2"
    repo.mkdir()
    (repo / "hubconf.py").write_text("# test probe only\n")
    source_hash = hashlib.sha256((repo / "hubconf.py").read_bytes()).hexdigest()
    checkpoint = tmp_path / "probe.pth"
    model = AdapterProbe()
    torch.save(model.state_dict(), checkpoint)
    calls = []

    def load(repo_or_dir, model_name, **kwargs):
        calls.append((repo_or_dir, model_name, kwargs))
        return model

    monkeypatch.setattr(torch.hub, "load", load)
    extractor = FrozenDINOv2(
        repo, checkpoint, layer_index=18, device="cpu",
        expected_source_hashes={"hubconf.py": source_hash},
        source_revision_hint="test-archive-not-a-verified-git-revision",
    )
    result = extractor.extract(np.zeros((20, 35, 3), dtype=np.uint8))
    assert result.shape == (16, 16, 1536) and np.isfinite(result).all()
    assert calls[0][2] == {"source": "local", "pretrained": False}
    assert model.seen[0] and not model.training and not model.scale.requires_grad
    assert model.seen[2]["n"] == [18] and model.seen[2]["norm"] is True
    assert extractor.provenance["source_fingerprints"]["hubconf.py"] == source_hash
    assert extractor.provenance["verified_source_hashes"] == {"hubconf.py": source_hash}
    assert extractor.provenance["verified_git_revision"] is None
    assert extractor.provenance["source_revision"] is None
    assert extractor.provenance["source_revision_hint"] == "test-archive-not-a-verified-git-revision"
    np.testing.assert_allclose(model.seen[1][0, :, 0, 0].numpy(),
                               -np.array([.485, .456, .406]) / np.array([.229, .224, .225]), atol=1e-6)
    with pytest.raises(ValueError, match="uint8"):
        extractor.extract(np.zeros((20, 35, 3), dtype=float))


def test_missing_models_fail_closed(tmp_path):
    with pytest.raises(FeatureDependencyError, match="no download"):
        FrozenDINOv2(tmp_path, tmp_path / "missing.pth", layer_index=18, device="cpu")
    with pytest.raises(ValueError, match="ViT-giant"):
        FrozenDINOv2(tmp_path, tmp_path / "missing.pth", layer_index=18, model_name="dinov2_vitl14")
    with pytest.raises(FeatureDependencyError, match="source missing"):
        FrozenGeDi(tmp_path, tmp_path / "missing.tar", expected_source_sha256="abc")


def test_gedi_requires_pinned_external_source_before_import(tmp_path):
    source = tmp_path / "gedi.py"
    source.write_text("raise RuntimeError('must not execute unpinned source')\n")
    with pytest.raises(FeatureDependencyError, match="SHA256"):
        FrozenGeDi(tmp_path, tmp_path / "missing.tar", expected_source_sha256="wrong")
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(FeatureDependencyError, match="checkpoint missing"):
        FrozenGeDi(tmp_path, tmp_path / "missing.tar", expected_source_sha256=sha)


def test_gedi_adapter_two_native_diameter_scales_and_rng_restoration():
    """Exercise API plumbing with a labeled probe, never a trained descriptor."""
    torch = pytest.importorskip("torch")
    calls = []

    class GeDiAPIProbe:
        def __init__(self):
            self.lrf = SimpleNamespace(r_lrf=None)

        def compute(self, pts, pcd):
            assert torch.is_inference_mode_enabled()
            assert pts.dtype == torch.float32 and pcd.dtype == torch.float32
            assert self.r_lrf == self.lrf.r_lrf
            calls.append(self.r_lrf)
            np.random.rand(5)  # Mimic the external original sampler.
            return np.full((len(pts), 32), self.r_lrf)

    adapter = FrozenGeDi.__new__(FrozenGeDi)
    adapter.backend, adapter.torch, adapter.seed = GeDiAPIProbe(), torch, 7
    before = np.random.get_state()
    result = adapter.compute(np.zeros((4, 3)), np.ones((6, 3)), diameter=0.2)
    after = np.random.get_state()
    assert calls == [0.06, 0.08000000000000002]
    assert result.shape == (4, 64)
    np.testing.assert_allclose(result[:, :32], 0.06)
    np.testing.assert_allclose(result[:, 32:], 0.08)
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    with pytest.raises(ValueError, match="diameter"):
        adapter.compute(np.zeros((4, 3)), np.ones((6, 3)), diameter=0)


def test_dino_source_pin_mismatch_fails_before_loading_model(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    (tmp_path / "hubconf.py").write_text("# local archive\n")
    checkpoint = tmp_path / "unread-checkpoint.pth"
    checkpoint.write_bytes(b"Not a real checkpoint: must never be loaded")
    called = []
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: called.append(True))
    with pytest.raises(FeatureDependencyError, match="source SHA256 mismatch"):
        FrozenDINOv2(tmp_path, checkpoint, layer_index=18, device="cpu",
                     expected_source_hashes={"hubconf.py": "0" * 64})
    assert not called


def test_dino_missing_or_unsafe_pinned_source_is_rejected(tmp_path):
    (tmp_path / "hubconf.py").write_text("# local archive\n")
    checkpoint = tmp_path / "unread-checkpoint.pth"
    checkpoint.write_bytes(b"No model load is permitted")
    for path in ("../outside.py", "/absolute.py", "C:/outside.py"):
        with pytest.raises(ValueError, match="relative Python"):
            FrozenDINOv2(tmp_path, checkpoint, layer_index=18, device="cpu",
                         expected_source_hashes={path: "0" * 64})
    with pytest.raises(FeatureDependencyError, match="source file missing"):
        FrozenDINOv2(tmp_path, checkpoint, layer_index=18, device="cpu",
                     expected_source_hashes={"dinov2/models/vision_transformer.py": "0" * 64})


def test_forward_microbatch_keeps_prepared_inputs_and_eval_state():
    """This mechanics test is separate from real trained GeDi equivalence."""
    torch = pytest.importorskip("torch")
    torch.manual_seed(11)
    model = torch.nn.Sequential(torch.nn.Linear(5, 8), torch.nn.BatchNorm1d(8), torch.nn.Dropout(.7))
    freeze_model(model)
    prepared = torch.randn(17, 5)
    state_before = torch.random.get_rng_state().clone()
    with torch.inference_mode():
        whole = model(prepared)
        wrapper = microbatch_frozen_model(model, 4)
        chunked = wrapper(prepared)
    torch.testing.assert_close(chunked, whole, rtol=1e-6, atol=1e-6)
    assert not wrapper.training and not wrapper.unwrapped_model.training
    assert not any(parameter.requires_grad for parameter in wrapper.parameters())
    assert torch.equal(state_before, torch.random.get_rng_state())
    with pytest.raises(ValueError, match="positive"):
        microbatch_frozen_model(model, 0)


def test_gedi_forward_resource_setter_keeps_sampling_settings():
    torch = pytest.importorskip("torch")
    adapter = FrozenGeDi.__new__(FrozenGeDi)
    adapter._unwrapped_network = freeze_model(torch.nn.Linear(5, 3))
    adapter.backend = SimpleNamespace(samples_per_batch=500)
    adapter.provenance = {'samples_per_batch': 500, 'resources': {}}
    adapter.set_forward_microbatch_size(32)
    assert adapter.forward_microbatch_size == 32
    assert adapter.provenance['resources']['network_forward_microbatch_size'] == 32
    adapter.set_forward_microbatch_size(16)
    assert adapter.backend.samples_per_batch == adapter.provenance['samples_per_batch'] == 500
    assert adapter.backend.gedi_net.unwrapped_model is adapter._unwrapped_network
    adapter.set_forward_microbatch_size(None)
    assert adapter.backend.gedi_net is adapter._unwrapped_network
