"""Simulation-only correct-human assumption, invoked solely after gate deferral."""

from collections import Counter
import json
from pathlib import Path

import numpy as np


MAX_DISTANCE_M = 0.075
MIN_SEPARATION_M = 0.01


def match_candidates(localization, world_T_camera, simulator_objects):
    """Accept separated, mutual-nearest XY pairs; ambiguous identities stay unresolved."""
    candidates = localization["objects"]
    ids = [item["object_id"] for item in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate persistent candidate ID.")
    names = [item["instance"] for item in simulator_objects]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate simulator instance.")
    if not candidates or not simulator_objects:
        raise ValueError("Audit needs localized candidates and simulator objects.")
    camera_points = np.array([item["centroid_3d_m"] for item in candidates])
    transform = np.asarray(world_T_camera)
    world_points = camera_points @ transform[:3, :3].T + transform[:3, 3]
    positions = np.array([item["position_world_m"] for item in simulator_objects])
    distances = np.linalg.norm(world_points[:, None, :2] - positions[None, :, :2], axis=2)
    nearest = distances.argmin(axis=1)
    counts = Counter(nearest.tolist())
    result = []
    for i, item in enumerate(candidates):
        j = int(nearest[i])
        ranked = np.sort(distances[i])
        gap = float(ranked[1] - ranked[0]) if len(ranked) > 1 else None
        reverse = np.sort(distances[:, j])
        reverse_gap = float(reverse[1] - reverse[0]) if len(reverse) > 1 else None
        status = "matched"
        if ranked[0] > MAX_DISTANCE_M:
            status = "unmatched_distance"
        elif counts[j] > 1 or distances[:, j].argmin() != i:
            status = "ambiguous_possible_split"
        elif (gap is not None and gap < MIN_SEPARATION_M) or (reverse_gap is not None and reverse_gap < MIN_SEPARATION_M):
            status = "ambiguous_near_tie"
        obj = simulator_objects[j]
        result.append({
            "object_id": item["object_id"], "roi": item["roi"],
            "semantic_identity": obj["semantic_identity"] if status == "matched" else None,
            "simulator_instance": obj["instance"] if status == "matched" else None,
            "status": status, "centroid_world_m": world_points[i].tolist(),
            "nearest_simulator_instance": obj["instance"],
            "nearest_xy_distance_m": float(ranked[0]),
            "next_object_distance_gap_m": gap,
            "next_cluster_distance_gap_m": reverse_gap,
        })
    return result


def assumed_human_selection(association, environment, localization, world_T_camera, output_dir):
    """Return an audited candidate ID, never a simulator pose for task execution."""
    inner = environment.env.env
    objects = []
    for obj in [*inner.objects, *inner.fixtures]:
        body_id = environment.sim.model.body_name2id(obj.root_body)
        objects.append({
            "instance": obj.name,
            "semantic_identity": obj.category_name.replace("_", " "),
            "position_world_m": environment.sim.data.body_xpos[body_id].tolist(),
        })
    matches = match_candidates(localization, world_T_camera, objects)
    target = association["target_description"]
    present = [obj for obj in objects if obj["semantic_identity"] == target]
    matching = [item for item in matches
                if item["status"] == "matched" and item["semantic_identity"] == target]
    if not present:
        selection = None
    elif len(present) == len(matching) == 1:
        selection = matching[0]["object_id"]
    else:
        raise RuntimeError(
            f"Assumed human cannot resolve {target!r}: target is present but its localized candidate is missing or ambiguous."
        )
    if selection is not None and selection not in association["candidate_object_ids"]:
        raise ValueError("Assumed human selected an object outside the localized candidates.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / (target.replace(" ", "_") + ".json")).write_text(json.dumps({
        "experimental_condition": "assumed_correct_human_on_deferral",
        "actual_human_response": False, "target_description": target,
        "original_vlm_object_id": association["vlm_object_id"],
        "original_association_score": association["association_score"],
        "selected_object_id": selection, "candidate_identity_audit": matches,
        "simulator_identity_used": True, "execution_poses_remain_perception_estimates": True,
    }, indent=2), encoding="utf-8")
    return selection
