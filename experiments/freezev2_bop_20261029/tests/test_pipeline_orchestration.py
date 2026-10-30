"""API orchestration regressions with explicitly synthetic, mocked backends.

These tests do not run trained DINO/GeDi, produce benchmark predictions, or
assert dataset pose accuracy.  They exercise the real pipeline's geometry,
feature mapping, score wiring, ranking, diagnostics and BOP writer interfaces.
"""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from freezev2_repro import pipeline
from freezev2_repro.evaluation import read_bop_csv
from freezev2_repro.features import PCATransform
from freezev2_repro.masks import encode_rle
from freezev2_repro.registration import Correspondences, ICPResult, PoseResult, PoseScores


class ToyDINO:
    """Finite spatially varying tokens for mapping tests, not learned features."""

    provenance = {"test_only": True, "backend": "mock_spatial_tokens"}

    def extract(self, rgb):
        assert np.asarray(rgb).ndim == 3
        y, x = np.mgrid[:16, :16]
        return np.stack([x + 1, y + 1, (x + 1) * (y + 1), x * x + y + 1], -1).astype(np.float32)


class ToyGeDi:
    provenance = {"test_only": True, "backend": "mock_finite_geometry"}

    def compute(self, points, dense_cloud, diameter):
        assert points.shape[1] == dense_cloud.shape[1] == 3
        assert diameter > 0
        return np.column_stack([np.ones(len(points)), np.asarray(points)[:, 0] + 2]).astype(np.float32)


def _base_config(tmp_path):
    config = deepcopy(pipeline.load_config(Path(__file__).parents[1] / "config.json"))
    config["paths"]["output_root"] = str(tmp_path / "outputs")
    config["paths"]["cnos_poses"] = str(tmp_path / "camera_poses.npy")
    config["features"]["pca_components"] = 2  # Toy test dimension, not a base-run setting.
    config["cad"].update(initial_poisson_points=4, points=3, min_visible_views=1)
    config["render"]["resolution"] = 16
    np.save(config["paths"]["cnos_poses"], np.tile(np.eye(4), (162, 1, 1)))
    return config


def _toy_model():
    return {"vertices": np.array([[-0.02, -0.02, 0], [0.02, -0.02, 0], [0.02, 0.02, 0], [-0.02, 0.02, 0]]),
            "triangles": np.array([[0, 1, 2], [0, 2, 3]]), "colors": np.ones((4, 3)),
            "diameter_m": 0.2, "sha256": "synthetic_test_fixture_not_a_dataset_model"}


def _fake_dataset(monkeypatch):
    class ToyDataset:
        def __init__(self, root, dataset):
            self.root, self.dataset = root, dataset

        def model(self, object_id):
            assert object_id == 1
            return _toy_model()

        def frame(self, scene_id, image_id):
            # No annotation files or GT poses exist in this fixture.
            assert (scene_id, image_id) == (5, 7)
            return {"rgb": np.full((24, 24, 3), 127, np.uint8),
                    "depth_m": np.ones((24, 24)),
                    "K": np.array([[100.0, 0, 11.5], [0, 100.0, 11.5], [0, 0, 1]]),
                    "source": {"test_only": "synthetic RGB-D, no ground truth"}}

    monkeypatch.setattr(pipeline, "BOPDataset", ToyDataset)


def test_geometry_onboard_unpacks_real_visibility_interface(tmp_path, monkeypatch):
    config = _base_config(tmp_path)
    _fake_dataset(monkeypatch)
    points = _toy_model()["vertices"].copy()
    monkeypatch.setattr(pipeline, "_poisson_surface", lambda model, count, seed: points)
    K = np.array([[20.0, 0, 7.5], [0, 20.0, 7.5], [0, 0, 1]])
    # One controlled view avoids generating the actual 162-view CAD cache in a
    # unit test. visible_surface/project remain their real implementations.
    monkeypatch.setattr(pipeline, "camera_poses", lambda *args, **kwargs: (K, [(np.eye(3), np.array([0, 0, 1.0]))]))
    class ToyRenderer:
        def __init__(self, *args, **kwargs):
            pass

        def render(self, *args, **kwargs):
            return {"rgb": np.full((16, 16, 3), 100, np.uint8), "depth_m": np.ones((16, 16))}

    monkeypatch.setattr(pipeline, "RaycastRenderer", ToyRenderer)
    result = pipeline.geometry_onboard(config)
    assert result["status"] == "geometry_and_visibility_complete"
    assert result["retained_points"] == 3
    assert result["views"][0]["visible_points"] == 4
    assert result["views"][0]["depth_residual_max_m"] == 0
    with np.load(pipeline._object_dir(config) / "geometry.npz", allow_pickle=False) as data:
        assert data["points_m"].shape == (3, 3)
        assert data["visibility"].shape == (1, 3)
        assert data["visibility"].all()


def test_feature_onboard_uses_project_array_and_cad_only_pca(tmp_path, monkeypatch):
    config = _base_config(tmp_path)
    _fake_dataset(monkeypatch)
    output = pipeline._object_dir(config)
    (output / "views").mkdir(parents=True)
    points = _toy_model()["vertices"]
    K = np.array([[150.0, 0, 7.5], [0, 150.0, 7.5], [0, 0, 1]])
    np.savez_compressed(output / "geometry.npz", points_m=points, support_points_m=points,
                        visibility=np.ones((2, 4), bool), K=K,
                        R=np.tile(np.eye(3), (2, 1, 1)), t_m=np.tile([0, 0, 1.0], (2, 1)))
    for view in range(2):
        Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(output / "views" / f"{view:03d}.png")
    monkeypatch.setattr(pipeline, "geometry_onboard", lambda config: {"signature": {"test_only": True}})
    monkeypatch.setattr(pipeline, "load_feature_backends", lambda config: (ToyDINO(), ToyGeDi()))
    result = pipeline.feature_onboard(config)
    assert result["PCA_fit_scope"] == "this CAD only, no observed/test data"
    assert result["feature_shape"] == [4, 4]  # Toy PCA2 + geometry2.
    with np.load(output / "features.npz", allow_pickle=False) as data:
        assert np.isfinite(data["fused"]).all()
        assert data["view_counts"].tolist() == [2, 2, 2, 2]
    pca = PCATransform.load(output / "pca.npz")
    assert pca.components_.shape == (2, 4)


@pytest.mark.parametrize("single_mask", [True, False])
def test_infer_single_and_multiple_candidates_wire_scores_and_diagnostics(tmp_path, monkeypatch, single_mask):
    config = _base_config(tmp_path)
    config["cohort"] = [{"scene_id": 5, "im_id": 7}]
    config["paths"]["targets"] = str(tmp_path / "test_targets.json")
    Path(config["paths"]["targets"]).write_text(json.dumps([
        {"scene_id": 5, "im_id": 7, "obj_id": 1, "inst_count": 1}]))
    mask = np.zeros((24, 24), bool)
    mask[4:20, 4:20] = True
    detections = [{"scene_id": 5, "image_id": 7, "category_id": 1, "score": confidence,
                   "segmentation": encode_rle(mask)} for confidence in (0.99, 0.1, 0.05)]
    mask_path = tmp_path / "synthetic_predicted_format_masks.json"
    mask_path.write_text(json.dumps(detections))
    config["masks"] = [{"path": str(mask_path), "segmentor": "test_fixture", "source_id": "synthetic_api_test_only", "source_kind": "predicted"}]
    _fake_dataset(monkeypatch)
    monkeypatch.setattr(pipeline, "feature_onboard", lambda config: {"test_only": True})
    monkeypatch.setattr(pipeline, "load_feature_backends", lambda config: (ToyDINO(), ToyGeDi()))
    output = pipeline._object_dir(config)
    output.mkdir(parents=True)
    cad = _toy_model()["vertices"]
    np.savez_compressed(output / "geometry.npz", points_m=cad)
    np.savez_compressed(output / "features.npz", fused=np.ones((4, 4), np.float32))
    PCATransform(2).fit(np.array([[1., 1, 1, 1], [2, 3, 4, 2], [4, 2, 1, 3], [3, 5, 2, 4]])).save(output / "pca.npz")
    observed_calls, score_calls = [], []

    def coarse_fixture(cad_xyz, observed_xyz, cad_features, observed_features, diameter, registration):
        index = len(observed_calls)
        assert observed_features.shape == (len(observed_xyz), 4)
        assert np.isfinite(observed_features).all()
        pairs = Correspondences(np.zeros((len(observed_xyz), 1), np.int64), np.ones((len(observed_xyz), 1)))
        coarse = PoseResult(np.eye(3), np.array([index * 0.001, 0, 1.0]), [0.1, 1.5][index], "ok", pairs, {"test_only_candidate_index": index})
        observed_calls.append(coarse)
        return coarse

    def icp_fixture(cad_xyz, dense_xyz, rotation, translation, diameter, registration):
        return ICPResult(rotation, translation + [0.0001, 0, 0], 0.5, 2, 0.001, 1, "converged_transform_delta", {"test_only": True})

    def score_fixture(cad_xyz, observed_xyz, rotation, translation, diameter, correspondences, *, dense_xyz, config):
        index = len(score_calls)
        assert correspondences is observed_calls[index].correspondences
        assert np.allclose(translation, observed_calls[index].t + [0.0001, 0, 0])
        assert len(dense_xyz) > 0
        score_calls.append(index)
        return PoseScores([0.4, 1.2][index], 0.5, 2, len(observed_xyz), 2, len(cad_xyz), 0.006)

    monkeypatch.setattr(pipeline, "feature_ransac", coarse_fixture)
    monkeypatch.setattr(pipeline, "refine_icp", icp_fixture)
    monkeypatch.setattr(pipeline, "score_pose", score_fixture)
    # Real ranking, overlay projection, diagnostic writes and BOP writer execute.
    result = pipeline.infer(config, single_mask=single_mask)
    expected_calls = 1 if single_mask else 2  # N+1 base pool, third candidate excluded.
    assert len(observed_calls) == len(score_calls) == expected_calls
    predictions = read_bop_csv(result["bop_csv"])
    assert len(predictions) == 1
    best_index = 0 if single_mask else 1
    expected_score = 0.1 * 0.4 * 0.5 if single_mask else 1.5 * 1.2 * 0.5
    assert predictions[0].score == pytest.approx(expected_score)
    assert predictions[0].t_m[0] == pytest.approx(best_index * 0.001 + 0.0001)
    assert result["GT_pose_access"] is False
    traces = result["images"][0]["candidates"]
    assert all(trace["status"] == "ok" for trace in traces)
    assert result["images"][0]["selected"] == [f"test_fixture:000005:000007:{best_index}"]
    scope = "single_mask_diagnostic" if single_mask else "predicted_mask_localization"
    for index in range(expected_calls):
        folder = Path(config["paths"]["output_root"]) / scope / "000005_000007" / f"candidate_{index:06d}_test_fixture"
        assert all((folder / filename).is_file() for filename in ["mask.png", "crop.png", "observed.npz", "pose_overlay.png", "diagnostic.json"])
        saved = json.loads((folder / "diagnostic.json").read_text())
        assert saved["candidate"]["source_id"] == "synthetic_api_test_only"
        assert saved["joint_score"] == pytest.approx(saved["coarse_feature_score"] * saved["refined_feature_score"] * saved["geometry_score"])
