"""Compute evaluation-only alignment targets from the saved in-hand peg pose."""

import json
from pathlib import Path

import numpy as np


TRIAL = Path(__file__).resolve().parents[1] / "outputs/simulation/nist_task_board_1/peg_pilot/20260905T142446744069Z"
PEG_LENGTH_M = .050


def pose(rotation, position):
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform


def main():
    rows = json.loads((TRIAL / "offset_diagnosis/baseline.json").read_text())
    row = next(item for item in rows if item["step"] == 192)
    hole = np.array(json.loads((TRIAL / "framework_human/evaluation.json").read_text())["world_T_hole"])
    eef = pose(row["eef_rotation_world"], row["eef_world_m"])
    peg = pose(row["peg_rotation_world"], row["peg_world_m"])
    eef_T_peg = np.linalg.inv(eef) @ peg
    hole_T_peg = np.linalg.inv(hole) @ peg
    tip = hole_T_peg[:3, 3] - hole_T_peg[:3, 2] * PEG_LENGTH_M / 2

    # Remove axis tilt with the shortest rotation; circular-peg yaw is irrelevant.
    axis, target_axis = peg[:3, 2], hole[:3, 2]
    cross = np.cross(axis, target_axis)
    cosine = np.dot(axis, target_axis)
    if cosine <= -1 + 1e-8:
        raise ValueError("Antiparallel peg axis requires an explicit end selection")
    x, y, z = cross
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    correction = np.eye(3) + skew + skew @ skew / (1 + cosine)
    aligned_rotation = correction @ peg[:3, :3]
    targets = {}
    for name, tip_height in (("preinsert_tip_10mm_above", .010), ("insert_tip_5mm_below", -.005)):
        desired_peg = pose(aligned_rotation, hole[:3, 3] + target_axis * (PEG_LENGTH_M / 2 + tip_height))
        desired_eef = desired_peg @ np.linalg.inv(eef_T_peg)
        reconstructed = np.linalg.inv(hole) @ desired_eef @ eef_T_peg
        reconstructed_tip = reconstructed[:3, 3] - reconstructed[:3, 2] * PEG_LENGTH_M / 2
        np.testing.assert_allclose(reconstructed_tip, [0, 0, tip_height], atol=1e-9)
        np.testing.assert_allclose(reconstructed[:3, 2], [0, 0, 1], atol=1e-9)
        targets[name] = {"world_T_eef": desired_eef.tolist(),
                         "eef_world_mm": (desired_eef[:3, 3] * 1000).tolist(),
                         "eef_translation_from_snapshot_mm": ((desired_eef[:3, 3] - eef[:3, 3]) * 1000).tolist()}
    result = {"source": "saved simulator ground truth; evaluation-only geometry, not a perception estimate",
              "snapshot_step": 192, "snapshot_phase": "approach_hole",
              "assumption": "peg-to-gripper transform remains fixed during correction; no motion executed",
              "hole_world_mm": (hole[:3, 3] * 1000).tolist(),
              "peg_center_in_hole_mm": (hole_T_peg[:3, 3] * 1000).tolist(),
              "peg_tip_in_hole_mm": (tip * 1000).tolist(),
              "peg_axis_tilt_deg": float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))),
              "eef_T_peg": eef_T_peg.tolist(), "targets": targets}
    output = TRIAL / "offset_diagnosis/hole_alignment.json"
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
