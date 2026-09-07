"""Simulation-only correct-human assumption, invoked solely after gate deferral."""

import json
from pathlib import Path

from scripts.audit_libero_candidate_identities import match_candidates


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
