"""Focused geometry checks for the isolated runtime grasp experiment.

Run with the existing local WSL environment. Asset-dependent Panda checks skip
only when the retained local source scene is unavailable; no downloads occur.
"""
from hashlib import sha1, sha256
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import open3d as o3d
import pytest
from scipy.spatial.transform import Rotation
import torch

from experiments.runtime_antipodal_assembly_20261023 import grasp_planning as gp
from experiments.runtime_antipodal_assembly_20261023.vendor import (
    AntipodalSampler, AntipodalSamplerCfg,
)


HERE = Path(__file__).resolve().parent
PANDA_SCENES = (
    Path("/mnt/c/Users/Jiabao Zhao/Documents/Codex/2026-06-02/can-you-add-this-skill-to/nist-taskboard-workspace/objects-centered/scene.xml"),
    Path("C:/Users/Jiabao Zhao/Documents/Codex/2026-06-02/can-you-add-this-skill-to/nist-taskboard-workspace/objects-centered/scene.xml"),
)


@pytest.fixture(scope="module")
def panda():
    scene = next((path for path in PANDA_SCENES if path.is_file()), None)
    if scene is None:
        pytest.skip("Retained local Panda scene is unavailable; no asset download.")
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "objects_centered")
    assert key >= 0
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    return model, data


def _candidate(pose, opening=.04):
    return gp.Candidate(pose, np.zeros((2, 3)), .02, opening, 0., 0., 0.)


def test_vendor_original_bytes_and_only_documented_import_adaptation():
    vendor = HERE / "vendor"
    record = json.loads((vendor / "provenance.json").read_text())
    for entry in record["files"]:
        raw = (vendor / entry["path"]).read_bytes()
        assert sha256(raw).hexdigest() == entry["sha256"], entry["path"]
    source = (vendor / "upstream/antipodal_sampler.py").read_bytes()
    license_bytes = (vendor / "LICENSE.txt").read_bytes()
    for raw, expected in (
        (source, record["source_git_blob_sha1"]),
        (license_bytes, record["license_git_blob_sha1"]),
    ):
        assert sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == expected
    old = b"from embodichain.utils import configclass\nfrom embodichain.utils import logger"
    assert source.count(old) == 1
    expected = source.replace(old, b"from ._compat import configclass, logger")
    assert (vendor / "antipodal_sampler.py").read_bytes() == expected


def test_point_transform_uses_rotation_then_translation():
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    pose[:3, 3] = [.3, -.2, .1]
    result = gp.transform_points(pose, np.array([[1., 0., 0.], [0., 2., 0.]]))
    np.testing.assert_allclose(result, [[.3, .8, .1], [-1.7, -.2, .1]], atol=1e-12)


@pytest.mark.parametrize("pair", [
    [[0., 0., .05], [.02, 0., .05]],
    [[0., 0., .05], [0., 0., .075]],
    [[.01, .02, .05], [.03, .04, .065]],
])
def test_contact_frames_are_right_handed_and_align_jaw_axis(pair):
    pair = np.asarray(pair)
    frames = gp.candidate_frames(pair[None], np.eye(3), pair.mean(axis=0), limit_pairs=1)
    assert frames
    assert len(frames) <= 6 * 2 * 4
    axis = pair[1] - pair[0]
    axis /= np.linalg.norm(axis)
    for candidate in frames:
        rotation = candidate.world_T_grip[:3, :3]
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        assert np.linalg.det(rotation) == pytest.approx(1.)
        assert abs(rotation[:, 0] @ axis) == pytest.approx(1.)
        assert rotation[2, 2] <= .1 + 1e-12
        expected_midpoint = gp.transform_points(
            candidate.world_T_grip, np.array([[0., 0., candidate.pad_depth_m]]))[0]
        np.testing.assert_allclose(expected_midpoint, pair.mean(axis=0), atol=1e-12)
        assert candidate.opening_m >= candidate.width_m
        assert candidate.opening_m <= .08


def test_sampler_box_returns_finite_surface_contacts_in_width_range():
    mesh = o3d.geometry.TriangleMesh.create_box(.03, .02, .01)
    vertices = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32)
    faces = torch.tensor(np.asarray(mesh.triangles).copy(), dtype=torch.int64)
    cfg = AntipodalSamplerCfg(n_sample=128, max_angle=np.pi / 12,
                             min_length=.001, max_length=.08)
    pairs = AntipodalSampler(cfg).sample(vertices, faces).numpy()
    assert 0 < len(pairs) <= 2 * cfg.n_sample
    assert pairs.shape[1:] == (2, 3)
    assert np.all(np.isfinite(pairs))
    widths = np.linalg.norm(pairs[:, 1] - pairs[:, 0], axis=1)
    assert np.all((widths > cfg.min_length) & (widths < cfg.max_length))
    assert np.all(pairs >= -1e-5)
    assert np.all(pairs <= np.array([.03, .02, .01]) + 1e-5)
    boundary_distance = np.minimum(np.abs(pairs), np.abs(pairs - [.03, .02, .01]))
    assert np.all(boundary_distance.min(axis=-1) < 1e-5)


def test_runtime_contacts_follow_supplied_cad_pose(tmp_path):
    mesh = o3d.geometry.TriangleMesh.create_box(.03, .02, .01)
    path = tmp_path / "box.ply"
    assert o3d.io.write_triangle_mesh(str(path), mesh)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("xyz", [15, 20, 35], degrees=True).as_matrix()
    pose[:3, 3] = [.3, -.2, .1]
    _, _, _, pairs, stats = gp.sample_contacts(path, pose, n_sample=128)
    assert len(pairs) > 0
    assert stats["raw_upstream_pairs"] >= stats["qualified_unique_pairs"] == len(pairs)
    local = gp.transform_points(np.linalg.inv(pose), pairs.reshape(-1, 3))
    assert np.all(local >= -1e-5)
    assert np.all(local <= np.array([.03, .02, .01]) + 1e-5)


def test_parallel_endpoint_normals_are_rejected(tmp_path, monkeypatch):
    # Two same-facing sheets expose the absolute-normal ambiguity of the
    # upstream sampler; they cannot be accepted as opposing finger contacts.
    vertices = np.array([[x, y, z] for x in (0., .02)
                         for y, z in ((0., 0.), (.02, 0.), (.02, .02), (0., .02))])
    triangles = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                   o3d.utility.Vector3iVector(triangles))
    path = tmp_path / "parallel_same_facing.ply"
    assert o3d.io.write_triangle_mesh(str(path), mesh)
    pair = torch.tensor([[[0., .01, .01], [.02, .01, .01]]], dtype=torch.float32)
    monkeypatch.setattr(gp.AntipodalSampler, "sample", lambda self, v, f: pair.clone())
    _, _, _, qualified, stats = gp.sample_contacts(path, np.eye(4), n_sample=1)
    assert stats["raw_upstream_pairs"] == 1
    assert len(qualified) == stats["qualified_unique_pairs"] == 0


def test_actual_panda_fingers_close_on_grip_site_x_axis(panda):
    model, reference = panda
    site = model.site("gripper0_grip_site").id
    joint_ids = [model.joint(f"gripper0_finger_joint{i}").id for i in (1, 2)]
    qadr = model.jnt_qposadr[joint_ids]
    body_ids = model.jnt_bodyid[joint_ids]
    positions = []
    for width in (.02, .04):
        data = mujoco.MjData(model)
        data.qpos[:] = reference.qpos
        data.qpos[qadr] = [width / 2, -width / 2]
        mujoco.mj_forward(model, data)
        rotation = data.site_xmat[site].reshape(3, 3)
        positions.append((data.xpos[body_ids] - data.site_xpos[site]) @ rotation)
    delta = positions[1] - positions[0]
    np.testing.assert_allclose(np.abs(delta[:, 0]), [.01, .01], atol=1e-8)
    np.testing.assert_allclose(delta[:, 1:], 0., atol=1e-8)
    assert delta[0, 0] * delta[1, 0] < 0.


def test_table_filter_rejects_penetration_and_checks_approach_sweep(panda):
    model, data = panda
    surface = gp.gripper_surface_local(model, data, .04)
    assert len(surface) > 8  # full collidable meshes, not only a point proxy
    minimum = surface[:, 2].min()
    clear_endpoint = np.eye(4)
    clear_endpoint[2, 3] = .05 - minimum
    blocked = _candidate(clear_endpoint)
    assert gp.table_filter([blocked], model, data, table_z=0.) == []
    assert blocked.table_clearance_m == pytest.approx(-.05, abs=1e-8)
    high = np.eye(4)
    high[2, 3] = .5 - minimum
    candidate = _candidate(high)
    assert gp.table_filter([candidate], model, data, table_z=0.) == [candidate]
    assert candidate.table_clearance_m >= .0001


def test_target_palm_obstruction_is_rejected_independently_of_loose_parts():
    xml = """<mujoco><worldbody>
      <body name="robot0_palm" pos="0 0 .1">
        <joint name="arm" type="slide" axis="1 0 0"/>
        <geom name="palm" type="box" size=".02 .02 .02"/>
        <body name="gripper0_leftfinger" pos="0 .035 0">
          <joint name="left" type="slide" axis="0 1 0"/>
          <geom type="box" size=".005 .005 .02"/>
        </body>
        <body name="gripper0_rightfinger" pos="0 -.035 0">
          <joint name="right" type="slide" axis="0 1 0"/>
          <geom type="box" size=".005 .005 .02"/>
        </body>
      </body>
      <body name="target" pos="0 0 .1"><freejoint/>
        <geom type="sphere" size=".01"/>
      </body>
    </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    trial = SimpleNamespace(m=model, d=data,
                            qa=np.array([model.jnt_qposadr[model.joint("arm").id]]),
                            fqa=np.array([model.jnt_qposadr[model.joint(name).id]
                                          for name in ("left", "right")]),
                            target_bid=model.body("target").id)
    blocked = gp.target_obstruction_report(trial, np.array([0.]), .02)
    assert not blocked["valid"]
    assert any(row["body"] == "robot0_palm" for row in blocked["contacts"])
    clear = gp.target_obstruction_report(trial, np.array([.1]), .02)
    assert clear["valid"]
